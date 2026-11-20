"""
src/make_handwritten.py
Ubah pesan beranotasi ringan -> data/handwritten_test.jsonl (format BIO).

Input : data/handwritten_raw.txt   (satu pesan per baris)
Format: <jenis> | pesan dengan [LABEL: teks] ...
  jenis : sajadah | handuk | hampers | ready | other   (boleh dikosongkan)
  LABEL : NAMA, PRODUK, JUMLAH, WARNA, UKURAN, TENGGAT, KARTU, PENERIMA (atau nama penuh)
  Baris kosong dan baris berawalan # diabaikan.

Jalankan: python -m src.make_handwritten
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from src import schema as S
from src.tokenizer import tokenize_with_offsets

RAW = Path("data/handwritten_raw.txt")
OUT = Path("data/handwritten_test.jsonl")

ALIAS = {label: label for label in S.ENTITY_LABELS}
ALIAS.update({"NAMA": "NAMA_PERSONALISASI", "KARTU": "TEKS_KARTU",
              "PENERIMA": "PENERIMA_ACARA", "ACARA": "PENERIMA_ACARA"})

ORDER_ALIAS = {"sajadah": "Custom Sajadah", "handuk": "Custom Handuk",
               "hampers": "Hampers", "ready": "Ready Stock", "other": "Other"}

SPAN_RE = re.compile(r"\[([A-Za-z_]+):\s*([^\]]+?)\s*\]")


def parse_body(body: str) -> Tuple[str, List[Tuple[str, int, int]]]:
    """Buang penanda, kembalikan (teks bersih, [(label, start, end)])."""
    parts, spans, pos, last = [], [], 0, 0
    for m in SPAN_RE.finditer(body):
        before = body[last:m.start()]
        parts.append(before)
        pos += len(before)
        label = ALIAS.get(m.group(1).upper())
        if label is None:
            raise ValueError(f"label tidak dikenal: '{m.group(1)}'")
        txt = m.group(2)
        spans.append((label, pos, pos + len(txt)))
        parts.append(txt)
        pos += len(txt)
        last = m.end()
    parts.append(body[last:])
    clean = "".join(parts)
    if "[" in clean or "]" in clean:
        raise ValueError("ada tanda [ atau ] yang tidak berpasangan / format salah")
    return clean, spans


def to_bio(clean: str, spans) -> Tuple[List[str], List[str]]:
    toks = tokenize_with_offsets(clean)
    tags = ["O"] * len(toks)
    for label, s, e in spans:
        idx = [i for i, (_, a, b) in enumerate(toks) if a >= s and b <= e]
        if not idx:
            raise ValueError(f"span '{clean[s:e]}' tidak cocok dengan token apa pun")
        if toks[idx[0]][1] != s or toks[idx[-1]][2] != e:
            raise ValueError(f"span '{clean[s:e]}' memotong sebagian token "
                             f"(tandai kata utuh, mis. '10lusin' bukan '10')")
        if any(tags[i] != "O" for i in idx):
            raise ValueError(f"span '{clean[s:e]}' tumpang tindih dengan span lain")
        for k, i in enumerate(idx):
            tags[i] = f"{'B' if k == 0 else 'I'}-{label}"
    return [t for t, _, _ in toks], tags


def main() -> None:
    if not RAW.exists():
        sys.exit(f"File {RAW} belum ada. Buat dulu (lihat contoh format).")

    records, errors, seen = [], [], set()
    for ln, line in enumerate(RAW.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if "|" not in line:
                raise ValueError("tidak ada tanda '|' (format: jenis | pesan)")
            jenis, body = (x.strip() for x in line.split("|", 1))
            order = None
            if jenis:
                if jenis.lower() not in ORDER_ALIAS:
                    raise ValueError(f"jenis '{jenis}' tidak dikenal "
                                     f"({', '.join(ORDER_ALIAS)})")
                order = ORDER_ALIAS[jenis.lower()]
            clean, spans = parse_body(body)
            tokens, tags = to_bio(clean, spans)
            if clean.lower() in seen:
                raise ValueError("pesan duplikat")
            seen.add(clean.lower())
            assert all(t in S.TAG2ID for t in tags)
            records.append({
                "id": f"hw-{len(records):04d}", "split": "handwritten", "text": clean,
                "tokens": tokens, "tags": tags, "order_type": order,
                "template_id": "handwritten", "augmented": False,
            })
        except ValueError as e:
            errors.append(f"  baris {ln}: {e}")

    if errors:
        print("Ada kesalahan, file belum ditulis:")
        print("\n".join(errors))
        sys.exit(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ent = Counter(t[2:] for r in records for t in r["tags"] if t.startswith("B-"))
    print(f"OK: {len(records)} pesan -> {OUT}")
    print("Span per entitas:", {k: ent.get(k, 0) for k in S.ENTITY_LABELS})
    print("Jenis pesanan   :", dict(Counter(r["order_type"] for r in records)))
    if len(records) < 50:
        print(f"Catatan: baru {len(records)} pesan (target 50-100).")


if __name__ == "__main__":
    main()