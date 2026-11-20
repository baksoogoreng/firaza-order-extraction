"""
src/normalizer.py
Normalisasi nilai PRODUK dan WARNA ke istilah katalog (typo dan singkatan chat).
Model NER tetap yang MENEMUKAN span; modul ini hanya memetakan teks ke istilah katalog.
Nilai asli tidak dihapus; yang kurang mirip dibiarkan apa adanya (tidak menebak).
"""
from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional

from src import schema as S

_VOWELS = re.compile(r"[aiueo]")
_PRODUCT_WORDS = sorted({w for t in S.PRODUCT_TERMS for w in t.split()})
_COLORS = [c.lower() for c in S.COLORS]


def _skeleton(w: str) -> str:
    return _VOWELS.sub("", w.lower())


def _closest_word(word: str, vocab: List[str], cutoff: float = 0.8) -> Optional[str]:
    w = word.lower()
    if w in vocab:
        return w
    # singkatan tanpa vokal: sjdh -> sajadah (hanya jika cocok dengan tepat satu kata)
    if len(w) >= 3 and not _VOWELS.search(w):
        hits = [v for v in vocab if _skeleton(v) == w]
        if len(hits) == 1:
            return hits[0]
    m = difflib.get_close_matches(w, vocab, n=1, cutoff=cutoff)
    return m[0] if m else None


def normalize_product(value: str) -> Optional[str]:
    words = [w[:-3] if (len(w) >= 7 and w.endswith("nya")) else w for w in value.lower().split()]
    mapped = [_closest_word(w, _PRODUCT_WORDS) for w in words]
    if not mapped or any(m is None for m in mapped):
        return None
    phrase = " ".join(mapped)
    return phrase if phrase in S.PRODUCT_TERMS else None


def normalize_color(value: str) -> Optional[str]:
    m = difflib.get_close_matches(value.lower(), _COLORS, n=1, cutoff=0.8)
    return m[0] if m else None


def normalize_entities(entities: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Optional[str]]]]:
    """{label: [{"asli": teks asli, "normal": istilah katalog atau None}]} untuk PRODUK dan WARNA."""
    out: Dict[str, List[Dict[str, Optional[str]]]] = {}
    for label, fn in (("PRODUK", normalize_product), ("WARNA", normalize_color)):
        items = []
        for v in entities.get(label, []):
            n = fn(v)
            items.append({"asli": v, "normal": n if (n and n != v.lower()) else None})
        out[label] = items
    return out