"""
src/evaluate.py
Evaluasi seqeval pada val, test sintetis, dan test tulisan tangan.
Jalankan:  python -m src.evaluate --ckpt models/char_crf.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from src import schema as S
from src.generator import extract_spans
from src.model import load_checkpoint
from src.preprocess import NERDataset, Vocab, collate_fn, load_jsonl


def predict_records(model, records: List[dict], wv: Vocab, cv: Vocab,
                    device: str, batch_size: int = 64) -> List[List[str]]:
    model.eval()
    dl = DataLoader(NERDataset(records, wv, cv, train=False), batch_size=batch_size,
                    shuffle=False, collate_fn=collate_fn)
    preds: List[List[str]] = []
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        preds += [[S.ID2TAG[i] for i in p] for p in model.decode(b)]
    return preds


def score(gold: List[List[str]], pred: List[List[str]]) -> dict:
    rep = classification_report(gold, pred, digits=4, output_dict=True)
    return {
        "micro_f1": round(f1_score(gold, pred), 4),
        "macro_f1": round(rep["macro avg"]["f1-score"], 4),
        "per_entity": {k: {m: round(v[m], 4) for m in ("precision", "recall", "f1-score")}
                       for k, v in rep.items() if k in S.ENTITY_LABELS},
    }


def show_errors(records, pred, k: int = 10) -> None:
    print(f"\n--- {k} kesalahan pertama ---")
    shown = 0
    for r, p in zip(records, pred):
        if p != r["tags"]:
            print("\nTEKS :", r["text"])
            print("GOLD :", extract_spans(r["tokens"], r["tags"]))
            print("PRED :", extract_spans(r["tokens"], p))
            shown += 1
            if shown >= k:
                break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--errors", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, wv, cv = load_checkpoint(args.ckpt, device)
    name = Path(args.ckpt).stem

    base = {}
    bp = Path("docs/baseline_scores.json")
    if bp.exists():
        base = json.loads(bp.read_text(encoding="utf-8"))

    sets = {"val": "data/processed/val.jsonl", "test": "data/processed/test.jsonl",
            "handwritten": "data/handwritten_test.jsonl"}
    results: Dict[str, dict] = {}
    for sname, path in sets.items():
        if not Path(path).exists():
            print(f"(lewati {sname}: {path} belum ada)")
            continue
        recs = load_jsonl(path)
        gold = [r["tags"] for r in recs]
        pred = predict_records(model, recs, wv, cv, device)
        results[sname] = score(gold, pred)
        print(f"\n===== {name} | {sname} ({len(recs)} pesan) =====")
        print(classification_report(gold, pred, digits=3))
        if sname in base:
            print(f"Baseline micro F1: {base[sname]['micro_f1']:.4f} "
                  f"| Model micro F1: {results[sname]['micro_f1']:.4f}")
        if sname in ("test", "handwritten"):
            show_errors(recs, pred, args.errors)

    Path("docs").mkdir(exist_ok=True)
    with open(f"docs/eval_{name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nTersimpan: docs/eval_{name}.json")


if __name__ == "__main__":
    main()