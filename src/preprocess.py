"""
src/preprocess.py
Vocab kata & karakter, encoding, Dataset, dan collate untuk PyTorch.
Vocab HANYA dibangun dari train (val/test tidak boleh mengintip).

Tes cepat:  python -m src.preprocess
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from src import schema as S

PAD, UNK, NUM = "<PAD>", "<UNK>", "<NUM>"
MAX_WORD_LEN = 20  # karakter per token (dipotong kalau lebih)

assert S.TAG2ID["O"] == 0  # padding tag = 0 = "O" (aman karena ada mask)


def load_jsonl(path: str | Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_word(tok: str) -> str:
    """Lowercase; angka murni -> <NUM> (angka sering unik, tidak perlu dihafal)."""
    t = tok.lower()
    return NUM if t.isdigit() else t


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------
class Vocab:
    def __init__(self, itos: List[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.unk_id = self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, item: str) -> int:
        return self.stoi.get(item, self.unk_id)

    @classmethod
    def build_words(cls, records: List[dict], min_freq: int = 2) -> "Vocab":
        cnt = Counter(normalize_word(t) for r in records for t in r["tokens"])
        words = sorted(w for w, c in cnt.items() if c >= min_freq and w != NUM)
        return cls([PAD, UNK, NUM] + words)

    @classmethod
    def build_chars(cls, records: List[dict]) -> "Vocab":
        # huruf besar/kecil dipertahankan: kapital adalah petunjuk nama
        chars = sorted({c for r in records for t in r["tokens"] for c in t})
        return cls([PAD, UNK] + chars)


def save_vocabs(path: str | Path, word_vocab: Vocab, char_vocab: Vocab) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"words": word_vocab.itos, "chars": char_vocab.itos},
                  f, ensure_ascii=False)


def load_vocabs(path: str | Path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return Vocab(d["words"]), Vocab(d["chars"])


# ---------------------------------------------------------------------------
# Encoding satu kalimat
# ---------------------------------------------------------------------------
def encode_example(tokens: List[str], tags: List[str], wv: Vocab, cv: Vocab,
                   p_unk_entity: float = 0.0, p_unk_other: float = 0.0,
                   p_char_mask: float = 0.0) -> dict:
    word_ids, char_ids = [], []
    for tok, tag in zip(tokens, tags):
        wid = wv.encode(normalize_word(tok))
        chars = [cv.encode(c) for c in tok[:MAX_WORD_LEN]]
        p = p_unk_entity if tag != "O" else p_unk_other
        if p > 0 and random.random() < p:
            wid = wv.unk_id
            # kata entitas yang dijatuhkan: sembunyikan juga hurufnya (kadang),
            # supaya model belajar dari KONTEKS, bukan menghafal ejaan
            if tag != "O" and random.random() < p_char_mask:
                chars = [cv.unk_id] * len(chars)
        word_ids.append(wid)
        char_ids.append(chars)
    tag_ids = [S.TAG2ID[t] for t in tags]
    return {"word_ids": word_ids, "char_ids": char_ids, "tag_ids": tag_ids}


class NERDataset(Dataset):
    def __init__(self, records: List[dict], word_vocab: Vocab, char_vocab: Vocab,
                 train: bool = False, p_unk_entity: float = 0.2,
                 p_unk_other: float = 0.12, p_char_mask: float = 0.3):
        self.records = records
        self.wv, self.cv = word_vocab, char_vocab
        self.pe, self.po, self.pc = ((p_unk_entity, p_unk_other, p_char_mask)
                                     if train else (0.0, 0.0, 0.0))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return encode_example(r["tokens"], r["tags"], self.wv, self.cv,
                              self.pe, self.po, self.pc)


def collate_fn(batch: List[dict]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    T = max(len(b["word_ids"]) for b in batch)
    C = max(len(c) for b in batch for c in b["char_ids"])
    words = torch.zeros(B, T, dtype=torch.long)
    chars = torch.zeros(B, T, C, dtype=torch.long)
    tags = torch.zeros(B, T, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["word_ids"])
        words[i, :n] = torch.tensor(b["word_ids"])
        tags[i, :n] = torch.tensor(b["tag_ids"])
        mask[i, :n] = True
        lengths[i] = n
        for j, cid in enumerate(b["char_ids"]):
            chars[i, j, :len(cid)] = torch.tensor(cid)
    return {"words": words, "chars": chars, "tags": tags, "mask": mask, "lengths": lengths}


def make_batch(tokens_list: List[List[str]], wv: Vocab, cv: Vocab) -> Dict[str, torch.Tensor]:
    """Untuk inference (API): token mentah -> batch tensor."""
    items = [encode_example(t, ["O"] * len(t), wv, cv) for t in tokens_list]
    return collate_fn(items)


# ---------------------------------------------------------------------------
# Tes cepat
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    d = Path("data/processed")
    train, val, test = (load_jsonl(d / f"{s}.jsonl") for s in ("train", "val", "test"))

    wv = Vocab.build_words(train, min_freq=2)
    cv = Vocab.build_chars(train)
    save_vocabs(d / "vocab.json", wv, cv)
    print(f"Vocab kata: {len(wv)} | vocab karakter: {len(cv)}")

    def oov_rate(recs):
        toks = [(t, tg) for r in recs for t, tg in zip(r["tokens"], r["tags"])]
        all_oov = sum(normalize_word(t) not in wv.stoi for t, _ in toks) / len(toks)
        ent = [t for t, tg in toks if tg != "O"]
        ent_oov = sum(normalize_word(t) not in wv.stoi for t in ent) / max(1, len(ent))
        return all_oov, ent_oov

    for name, recs in (("train", train), ("val", val), ("test", test)):
        a, e = oov_rate(recs)
        print(f"OOV {name:5s}: semua token {a:.1%} | token entitas {e:.1%}")

    dl = DataLoader(NERDataset(train, wv, cv, train=True), batch_size=32,
                    shuffle=True, collate_fn=collate_fn)
    b = next(iter(dl))
    print({k: tuple(v.shape) for k, v in b.items()})
    assert b["mask"][:, 0].all()  # syarat pytorch-crf: timestep pertama selalu valid
    print("OK")