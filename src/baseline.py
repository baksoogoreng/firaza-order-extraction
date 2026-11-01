"""
src/baseline.py
Baseline berbasis aturan: gazetteer + regex pada token. Tanpa pembelajaran.
Output berformat BIO yang sama dengan model, jadi bisa dievaluasi dengan seqeval.

Jalankan:  python -m src.baseline
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from seqeval.metrics import classification_report, f1_score

from src import schema as S
from src.generator import extract_spans
from src.preprocess import load_jsonl
from src.tokenizer import tokenize

# singkatan chat -> bentuk baku (hanya untuk MATCHING; token asli tidak diubah)
NORM = {"yg": "yang", "nm": "nama", "tgl": "tanggal", "utk": "untuk", "dgn": "dengan",
        "pc": "pcs", "bh": "buah", "wrn": "warna", "uk": "ukuran", "jml": "jumlah",
        "sbnyk": "sbnyk", "bs": "bisa"}
NORM["sbnyk"] = "sebanyak"


def norm(tok: str) -> str:
    t = tok.lower()
    return NORM.get(t, t)


QTY_ATTACHED = re.compile(r"^\d+(pcs|pc|buah|bh|set|pasang|lusin|biji)$")
DIM_RE = re.compile(r"^\d+x\d+$")
DAY_RE = re.compile(r"^\d{1,2}$")

NAME_CUES = sorted([
    ("nama", "yang", "dibordir"), ("nama", "yang", "dicetak"), ("tulisan", "nama"),
    ("isi", "nama"), ("cetak", "nama"), ("dengan", "nama"), ("nama",), ("namanya",),
    ("bordir",), ("personalisasi",), ("tertulis",),
], key=len, reverse=True)

NAME_STOP = {
    "yang", "dan", "untuk", "dengan", "nama", "namanya", "warna", "warnanya", "ukuran",
    "size", "jumlah", "jumlahnya", "ya", "kak", "min", "mau", "butuh", "tolong", "bisa",
    "pesan", "order", "ready", "dibordir", "bordir", "isi", "tulisan", "tertulis",
    "cetak", "personalisasi", "qty", "totalnya", "selesai", "kirim", "deadline",
    "sebelum", "perlu", "dipakai", "tenggat", "acara", "custom", "request",
}


def _phrases(items) -> List[tuple]:
    return sorted((tuple(tokenize(x.lower())) for x in items), key=len, reverse=True)


class RuleBaseline:
    def __init__(self):
        self.units = set(S.QTY_UNITS)
        self.products = _phrases(S.PRODUCT_TERMS.keys())
        self.colors = _phrases(S.COLORS)
        self.qty_words = _phrases(w for w in S.QTY_WORDS if w != "lusin")
        self.relative = _phrases(S.DEADLINE_RELATIVE)
        self.events = _phrases(S.EVENTS)
        self.recipients = _phrases(S.RECIPIENTS)
        self.size_letters = set(S.SIZE_LETTERS)

    def predict(self, tokens: List[str]) -> List[str]:
        n = len(tokens)
        nt = [norm(t) for t in tokens]
        tags = ["O"] * n
        taken = [False] * n

        def claim(i: int, j: int, label: str) -> bool:
            if i >= j or j > n or any(taken[i:j]):
                return False
            for k in range(i, j):
                taken[k] = True
                tags[k] = f"{'B' if k == i else 'I'}-{label}"
            return True

        def match(i: int, ph: tuple) -> bool:
            return i + len(ph) <= n and tuple(nt[i:i + len(ph)]) == ph

        def gazetteer(phrases, label: str) -> None:
            for i in range(n):
                if taken[i]:
                    continue
                for ph in phrases:
                    if match(i, ph):
                        claim(i, i + len(ph), label)
                        break

        # 1) TEKS_KARTU: isi di dalam tanda kutip
        quotes = [i for i, t in enumerate(tokens) if t == '"']
        for a, b in zip(quotes[0::2], quotes[1::2]):
            taken[a] = taken[b] = True
            if b > a + 1:
                claim(a + 1, b, "TEKS_KARTU")

        # 2) PRODUK
        gazetteer(self.products, "PRODUK")

        # 3) JUMLAH: angka + satuan, atau kata bilangan + satuan
        for i in range(n):
            if taken[i]:
                continue
            if QTY_ATTACHED.match(tokens[i].lower()):
                claim(i, i + 1, "JUMLAH")
            elif nt[i].isdigit() and i + 1 < n and nt[i + 1] in self.units:
                claim(i, i + 2, "JUMLAH")
            else:
                for ph in self.qty_words:
                    if match(i, ph) and i + len(ph) < n and nt[i + len(ph)] in self.units:
                        claim(i, i + len(ph) + 1, "JUMLAH")
                        break

        # 4) UKURAN: dimensi, atau huruf ukuran setelah kata 'ukuran'/'size'
        def size_at(j: int) -> bool:
            if j >= n or taken[j]:
                return False
            if DIM_RE.match(nt[j]):
                end = j + 1
                if end < n and nt[end] == "cm":
                    end += 1
                return claim(j, end, "UKURAN")
            return False

        for i in range(n):
            if nt[i] in ("ukuran", "size") and i + 1 < n and not taken[i + 1]:
                if not size_at(i + 1) and tokens[i + 1].upper() in self.size_letters:
                    claim(i + 1, i + 2, "UKURAN")
        for i in range(n):
            if not taken[i]:
                size_at(i)

        # 5) TENGGAT: tanggal+bulan, dd/mm, frasa relatif
        for i in range(n):
            if taken[i]:
                continue
            j = i + 1 if nt[i] == "tanggal" else i
            if j + 1 < n and DAY_RE.match(nt[j]) and nt[j + 1] in S.MONTHS:
                if claim(i, j + 2, "TENGGAT"):
                    continue
            if DAY_RE.match(nt[i]) and i + 2 < n and nt[i + 1] == "/" and nt[i + 2].isdigit():
                claim(i, i + 3, "TENGGAT")
        gazetteer(self.relative, "TENGGAT")

        # 6) WARNA
        gazetteer(self.colors, "WARNA")

        # 7) PENERIMA_ACARA: 'untuk <penerima>', 'acara <acara>', atau nama acara saja
        for i in range(n):
            if taken[i]:
                continue
            if nt[i] == "untuk":
                for ph in self.recipients:
                    if match(i + 1, ph):
                        claim(i, i + 1 + len(ph), "PENERIMA_ACARA")
                        break
            elif nt[i] == "acara":
                for ph in self.events:
                    if match(i + 1, ph):
                        claim(i, i + 1 + len(ph), "PENERIMA_ACARA")
                        break
        gazetteer(self.events, "PENERIMA_ACARA")

        # 8) NAMA_PERSONALISASI: kata setelah kata kunci (nama, bordir, ...)
        def name_ok(j: int) -> bool:
            return (j < n and not taken[j] and tokens[j].isalpha()
                    and nt[j] not in NAME_STOP)

        for i in range(n):
            for cue in NAME_CUES:
                if match(i, cue) and not any(taken[i:i + len(cue)]):
                    j = i + len(cue)
                    if name_ok(j):
                        k = j + 1
                        if k + 1 < n and nt[k] in ("&", "dan") and name_ok(k + 1):
                            k += 2
                        claim(j, k, "NAMA_PERSONALISASI")
                        break
        return tags


def evaluate(name: str, records: List[dict], model: RuleBaseline) -> dict:
    gold = [r["tags"] for r in records]
    pred = [model.predict(r["tokens"]) for r in records]
    print(f"\n===== BASELINE: {name} ({len(records)} pesan) =====")
    print(classification_report(gold, pred, digits=3))
    rep = classification_report(gold, pred, digits=3, output_dict=True)
    out = {"micro_f1": round(f1_score(gold, pred), 4),
           "per_entity": {k: {m: round(v[m], 4) for m in ("precision", "recall", "f1-score")}
                          for k, v in rep.items() if k in S.ENTITY_LABELS}}
    return out, pred


def show_errors(records: List[dict], pred: List[List[str]], k: int = 8) -> None:
    print(f"\n--- {k} contoh kesalahan pertama ---")
    shown = 0
    for r, p in zip(records, pred):
        if p != r["tags"]:
            print("\nTEKS :", r["text"])
            print("GOLD :", extract_spans(r["tokens"], r["tags"]))
            print("PRED :", extract_spans(r["tokens"], p))
            shown += 1
            if shown >= k:
                break


if __name__ == "__main__":
    model = RuleBaseline()
    d = Path("data/processed")
    sets = {"val": d / "val.jsonl", "test": d / "test.jsonl",
            "handwritten": Path("data/handwritten_test.jsonl")}
    scores = {}
    for name, path in sets.items():
        if not path.exists():
            print(f"(lewati {name}: {path} belum ada)")
            continue
        recs = load_jsonl(path)
        scores[name], pred = evaluate(name, recs, model)
        if name == "test":
            show_errors(recs, pred)

    Path("docs").mkdir(exist_ok=True)
    with open("docs/baseline_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print("\nSkor disimpan di docs/baseline_scores.json")