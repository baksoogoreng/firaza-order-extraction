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
from src.classifier import CLF_PATH, load_classifier, predict_one
from src.model import load_checkpoint
from src.preprocess import make_batch
from src.tokenizer import tokenize_with_offsets
from src.summarizer import summarize_order
from src.normalizer import normalize_entities

MODEL_PATH = os.getenv("MODEL_PATH", "models/word_crf.pt")
LOW_CONF = float(os.getenv("LOW_CONF", "0.60"))   # di bawah ini = perlu dicek admin
MAX_CHARS, MAX_TOKENS = 2000, 300

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, wv, cv = load_checkpoint(MODEL_PATH, "cpu")
    STATE.update(model=model, wv=wv, cv=cv, name=os.path.basename(MODEL_PATH))
    STATE["clf"] = load_classifier(CLF_PATH) if os.path.exists(CLF_PATH) else None
    yield
    STATE.clear()


app = FastAPI(title="Firaza Order Extraction API", version="0.2", lifespan=lifespan)
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
    jenis_pesanan: Optional[str] = None
    jenis_confidence: Optional[float] = None
    entities: Dict[str, List[str]]
    spans: List[Span]
    missing_fields: List[str]
    low_confidence: List[str]
    sentence_confidence: float
    normalisasi: Dict[str, List[Dict[str, Optional[str]]]] = {}

class SummarizeRequest(BaseModel):
    jenis_pesanan: Optional[str] = None
    entities: Dict[str, List[str]]


class SummarizeResponse(BaseModel):
    summary: str
    source: str
    note: Optional[str] = None
    missing_fields: List[str]


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    ents = {l: [v.strip() for v in req.entities.get(l, []) if v.strip()] for l in S.ENTITY_LABELS}
    missing = S.missing_fields(ents, req.jenis_pesanan)
    out = summarize_order(req.jenis_pesanan, ents, missing)
    return SummarizeResponse(**out, missing_fields=missing)

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
    return {"status": "ok", "model": STATE.get("name"),
            "classifier_loaded": STATE.get("clf") is not None,
            "low_conf_threshold": LOW_CONF}

@app.get("/catalog")
def catalog():
    """Katalog contoh (asumsi desain). Dibaca dari schema, sumber yang sama dengan training."""
    produk = {}
    for term in S.PRODUCT_TERMS:
        produk.setdefault(S.category_of(term), []).append(term)
    return {
        "catatan": "Katalog contoh (asumsi desain, bukan data asli Firaza)",
        "produk": produk,
        "warna": S.COLORS,
        "ukuran": {"huruf": S.SIZE_LETTERS, "dimensi": S.SIZE_DIMS,
                   "berlaku_untuk": sorted(S.SIZE_CATEGORIES)},
        "satuan_jumlah": S.QTY_UNITS,
        "jenis_pesanan": S.ORDER_TYPES,
        "field_wajib": {"semua_produk": S.ALWAYS_REQUIRED,
                        **{cat: rule["required"] for cat, rule in S.FIELD_RULES.items()
                           if cat != "unknown"}},
    }

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

    jenis, jenis_conf = None, None
    if STATE.get("clf") is not None:
        jenis, jenis_conf = predict_one(STATE["clf"], text)
        jenis_conf = round(jenis_conf, 4)

    return ExtractResponse(
        model=STATE["name"],
        text=text,
        jenis_pesanan=jenis,
        jenis_confidence=jenis_conf,
        entities=entities,
        spans=spans,
        missing_fields=S.missing_fields(entities, jenis),
        low_confidence=low_labels,
        sentence_confidence=round(seq_conf[0], 4),
        normalisasi=normalize_entities(entities),
    )