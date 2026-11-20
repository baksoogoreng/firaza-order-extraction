"""
src/summarizer.py
Ringkasan pesanan: Gemini (input HANYA JSON terstruktur) dengan cadangan template.
Kunci dibaca dari environment variable GEMINI_API_KEY (jangan ditulis di kode).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Dict, List, Optional

from src import schema as S

GEMINI_MODELS = [m.strip() for m in
                 os.getenv("GEMINI_MODELS", "gemini-3.5-flash,gemini-2.5-flash").split(",")]
MAX_CALLS = int(os.getenv("GEMINI_MAX_CALLS", "50"))   # batas panggilan per proses server
_CACHE: Dict[str, dict] = {}
_CALLS = {"n": 0}
_CLIENT = {"obj": None}

DISPLAY = {"NAMA_PERSONALISASI": "Nama personalisasi", "PRODUK": "Produk", "JUMLAH": "Jumlah",
           "WARNA": "Warna", "UKURAN": "Ukuran", "TENGGAT": "Tenggat",
           "TEKS_KARTU": "Teks kartu", "PENERIMA_ACARA": "Penerima/acara"}

SYSTEM_PROMPT = (
    "Kamu menyusun ringkasan pesanan untuk admin toko hampers dan hadiah personalisasi. "
    "Gunakan HANYA data pada JSON yang diberikan. Jangan menambah, menebak, atau mengubah "
    "informasi apa pun (nama, angka, tanggal, warna). Jika sebuah field wajib masih kosong, "
    "sebutkan bahwa informasinya belum ada. Tulis 2-3 kalimat bahasa Indonesia yang ringkas."
)


def _client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    if _CLIENT["obj"] is None:
        from google import genai
        _CLIENT["obj"] = genai.Client(api_key=key)
    return _CLIENT["obj"]


def template_summary(data: dict) -> str:
    """Ringkasan tanpa LLM: cadangan kalau API gagal atau ditolak pemeriksaan."""
    lines = [f"Jenis pesanan: {data['jenis_pesanan'] or 'belum ditentukan'}."]
    for label in S.ENTITY_LABELS:
        vals = data["entitas"].get(label)
        if vals:
            lines.append(f"{DISPLAY[label]}: {', '.join(vals)}.")
    if data["field_wajib_kosong"]:
        lines.append("Belum ada: " + ", ".join(DISPLAY[f] for f in data["field_wajib_kosong"]) + ".")
    return " ".join(lines)


def summary_is_faithful(summary: str, data: dict) -> bool:
    """Pemeriksaan sederhana: semua angka di ringkasan harus ada di data terstruktur."""
    allowed = set(re.findall(r"\d+", json.dumps(data, ensure_ascii=False)))
    return all(n in allowed for n in re.findall(r"\d+", summary))


def summarize_order(jenis: Optional[str], entities: Dict[str, List[str]],
                    missing: List[str]) -> dict:
    data = {"jenis_pesanan": jenis,
            "entitas": {k: v for k, v in entities.items() if v},
            "field_wajib_kosong": missing}
    key = hashlib.md5(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if key in _CACHE:
        return _CACHE[key]

    out = {"summary": template_summary(data), "source": "template", "note": None}
    client = _client()
    if client is None:
        out["note"] = "GEMINI_API_KEY belum diatur: memakai ringkasan template"
    elif _CALLS["n"] >= MAX_CALLS:
        out["note"] = "batas panggilan Gemini tercapai: memakai ringkasan template"
    else:
        from google.genai import types
        last_error = None
        for model_name in GEMINI_MODELS:
            try:
                _CALLS["n"] += 1
                resp = client.models.generate_content(
                    model=model_name,
                    contents=json.dumps(data, ensure_ascii=False),   # hanya JSON, bukan chat mentah
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
                                                       temperature=0.0))
                text = (resp.text or "").strip()
                if text and summary_is_faithful(text, data):
                    out = {"summary": text, "source": f"gemini:{model_name}", "note": None}
                    break
                last_error = "ringkasan kosong atau gagal pemeriksaan angka"
            except Exception as e:
                last_error = str(e)[:150]
        if out["source"] == "template" and last_error:
            out["note"] = f"memakai template ({last_error})"
    _CACHE[key] = out
    return out