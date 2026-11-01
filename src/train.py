"""
src/train.py
Contoh:
  python -m src.train                       # char + CRF (model utama)
  python -m src.train --no-crf              # char + softmax
  python -m src.train --no-char --no-crf    # BiLSTM word-only (model paling dasar)
  python -m src.train --no-char             # word-only + CRF
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader

from src import schema as S
from src.evaluate import predict_records
from src.model import NERModel, save_checkpoint
from src.preprocess import NERDataset, Vocab, collate_fn, load_jsonl


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--crf", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--word-dim", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", type=str, default=None)
    args = ap.parse_args()

    name = args.name or f"{'char' if args.char else 'word'}_{'crf' if args.crf else 'softmax'}"
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {name} | device: {device}")

    d = Path("data/processed")
    train, val = load_jsonl(d / "train.jsonl"), load_jsonl(d / "val.jsonl")
    wv, cv = Vocab.build_words(train, min_freq=2), Vocab.build_chars(train)  # hanya dari train
    print(f"train={len(train)} val={len(val)} | vocab kata={len(wv)} karakter={len(cv)}")

    config = dict(word_dim=args.word_dim, hidden=args.hidden, dropout=args.dropout,
                  use_char=args.char, use_crf=args.crf)
    model = NERModel(len(wv), len(cv), len(S.TAGS), **config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Jumlah parameter: {n_params:,}")

    dl = DataLoader(NERDataset(train, wv, cv, train=True), batch_size=args.batch,
                    shuffle=True, collate_fn=collate_fn)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    gold_val = [r["tags"] for r in val]

    Path("models").mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)
    ckpt = f"models/{name}.pt"
    best, bad, history = -1.0, 0, []

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad()
            loss = model.loss(b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        f1 = f1_score(gold_val, predict_records(model, val, wv, cv, device))
        history.append({"epoch": ep, "train_loss": round(total / len(dl), 4),
                        "val_f1": round(f1, 4)})
        flag = ""
        if f1 > best:
            best, bad, flag = f1, 0, "  <- terbaik, disimpan"
            save_checkpoint(ckpt, model, config, wv, cv,
                {"epoch": ep, "val_f1": float(round(f1, 4))})
        else:
            bad += 1
        print(f"epoch {ep:2d} | loss {total / len(dl):8.4f} | val F1 {f1:.4f} "
              f"| {time.time() - t0:4.0f}s{flag}")
        if bad >= args.patience:
            print(f"Early stopping (tidak membaik {args.patience} epoch).")
            break

    with open(f"docs/train_{name}.json", "w", encoding="utf-8") as f:
        json.dump({"name": name, "config": config, "best_val_f1": round(best, 4),
                   "n_params": n_params, "history": history}, f, indent=2)
    print(f"\nSelesai. Val F1 terbaik: {best:.4f}")
    print(f"Evaluasi lengkap:  python -m src.evaluate --ckpt {ckpt}")


if __name__ == "__main__":
    main()