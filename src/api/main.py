"""
src/api/main.py
Jalankan   : uvicorn src.api.main:app --reload
Ganti model: MODEL_PATH=models/char_crf.pt uvicorn src.api.main:app --reload
Dokumentasi interaktif otomatis di http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import schema as S
from src.model import load_checkpoint
from src.preprocess import make_batch
from src.tokenizer import tokenize_with_offsets

MODEL_PATH = os.getenv("MODEL_PATH", "models/word_crf.pt")
LOW_CONF = float(os.getenv("LOW_CONF", "0.60"))   # di bawah ini = perlu dicek admin
MAX_CHARS, MAX_TOKENS = 2000, 300

STATE: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model, wv, cv = load_checkpoint(MODEL_PATH, "cpu")
    STATE.update(model=model, wv=wv, cv=cv, name=os.path.basename(MODEL_PATH))
    yield
    STATE.clear()


app = FastAPI(title="Firaza Order Extraction API", version="0.1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # React (Vite)
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Skema request / response (selaras dengan schema.empty_result)
# ---------------------------------------------------------------------------
class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)


class Span(BaseModel):
    label: str
    text: str
    start: int            # posisi karakter di teks asli (untuk sorotan di UI)
    end: int
    confidence: float
    low_confidence: bool


class ExtractResponse(BaseModel):
    model: str
    text: str
    jenis_pesanan: Optional[str] = None   # diisi klasifier di Step 6
    entities: Dict[str, List[str]]
    spans: List[Span]
    missing_fields: List[str]
    low_confidence: List[str]
    sentence_confidence: float


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------
def spans_from_tags(tags: List[str]) -> List[Tuple[str, int, int]]:
    """BIO -> [(label, start_token, end_token_eksklusif)]. I- tanpa B- dianggap awal span."""
    spans: List[Tuple[str, int, int]] = []
    cur = None
    for i, tag in enumerate(tags):
        if tag == "O":
            if cur:
                spans.append(tuple(cur))
                cur = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or cur is None or cur[0] != label:
            if cur:
                spans.append(tuple(cur))
            cur = [label, i, i + 1]
        else:
            cur[2] = i + 1
    if cur:
        spans.append(tuple(cur))
    return spans


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE.get("name"), "low_conf_threshold": LOW_CONF}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    text = req.text
    toks = tokenize_with_offsets(text)
    if not toks:
        raise HTTPException(status_code=400, detail="Teks tidak berisi kata.")
    if len(toks) > MAX_TOKENS:
        raise HTTPException(status_code=400, detail=f"Maksimal {MAX_TOKENS} token.")

    tokens = [t for t, _, _ in toks]
    batch = make_batch([tokens], STATE["wv"], STATE["cv"])
    paths, tok_conf, seq_conf = STATE["model"].decode_with_conf(batch)
    tags = [S.ID2TAG[i] for i in paths[0]]
    conf = tok_conf[0]

    entities: Dict[str, List[str]] = {label: [] for label in S.ENTITY_LABELS}
    spans: List[Span] = []
    low_labels: List[str] = []
    for label, a, b in spans_from_tags(tags):
        s_char, e_char = toks[a][1], toks[b - 1][2]
        c = min(conf[a:b])                      # konservatif: token paling ragu
        is_low = c < LOW_CONF
        piece = text[s_char:e_char]
        entities[label].append(piece)
        spans.append(Span(label=label, text=piece, start=s_char, end=e_char,
                          confidence=round(c, 4), low_confidence=is_low))
        if is_low and label not in low_labels:
            low_labels.append(label)

    return ExtractResponse(
        model=STATE["name"],
        text=text,
        jenis_pesanan=None,
        entities=entities,
        spans=spans,
        missing_fields=S.missing_fields(entities),   # field wajib yang kosong, tidak ditebak
        low_confidence=low_labels,
        sentence_confidence=round(seq_conf[0], 4),
    )