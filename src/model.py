"""
src/model.py
Word emb (+ CharCNN) -> BiLSTM -> Linear -> (CRF | softmax).
Semua bobot dilatih dari nol (tidak ada embedding pretrained).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF  # pip install pytorch-crf (layer CRF, bukan model pretrained)

from src import schema as S
from src.preprocess import Vocab


class CharCNN(nn.Module):
    """Embedding karakter -> Conv1d -> max-pool: representasi bentuk huruf per kata."""
    def __init__(self, n_chars: int, char_dim: int = 30, n_filters: int = 50,
                 kernel: int = 3, dropout: float = 0.25):
        super().__init__()
        self.emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.conv = nn.Conv1d(char_dim, n_filters, kernel_size=kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:   # (B, T, C)
        B, T, C = chars.shape
        x = self.drop(self.emb(chars.view(B * T, C)))          # (BT, C, E)
        x = F.relu(self.conv(x.transpose(1, 2)))               # (BT, F, C)
        return x.max(dim=2).values.view(B, T, -1)              # (B, T, F)


class NERModel(nn.Module):
    def __init__(self, n_words: int, n_chars: int, n_tags: int,
                 word_dim: int = 100, char_dim: int = 30, char_filters: int = 50,
                 hidden: int = 128, dropout: float = 0.5,
                 use_char: bool = True, use_crf: bool = True):
        super().__init__()
        self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=0)
        self.char_cnn = CharCNN(n_chars, char_dim, char_filters) if use_char else None
        in_dim = word_dim + (char_filters if use_char else 0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden, n_tags)
        self.crf = CRF(n_tags, batch_first=True) if use_crf else None

    def emissions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.word_emb(batch["words"])
        if self.char_cnn is not None:
            x = torch.cat([x, self.char_cnn(batch["chars"])], dim=-1)
        x = self.drop(x)
        packed = pack_padded_sequence(x, batch["lengths"].cpu(),
                                      batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=batch["words"].size(1))
        return self.fc(self.drop(out))                          # (B, T, n_tags)

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        em = self.emissions(batch)
        if self.crf is not None:
            return -self.crf(em, batch["tags"], mask=batch["mask"], reduction="mean")
        return F.cross_entropy(em[batch["mask"]], batch["tags"][batch["mask"]])

    @torch.no_grad()
    def decode(self, batch: Dict[str, torch.Tensor]) -> List[List[int]]:
        em = self.emissions(batch)
        if self.crf is not None:
            return self.crf.decode(em, mask=batch["mask"])
        pred = em.argmax(-1)
        return [pred[i, :n].tolist() for i, n in enumerate(batch["lengths"].tolist())]

    @torch.no_grad()
    def decode_with_conf(self, batch: Dict[str, torch.Tensor]
                         ) -> Tuple[List[List[int]], List[List[float]], List[float]]:
        """
        Untuk API. Mengembalikan (path tag, confidence per token, confidence kalimat).
        - per token : softmax emission pada tag terpilih (penanda kasar)
        - kalimat   : CRF -> P(jalur Viterbi) = exp(log-likelihood jalur); non-CRF -> hasil kali
        """
        em = self.emissions(batch)
        mask = batch["mask"]
        probs = em.softmax(-1)
        paths = self.crf.decode(em, mask=mask) if self.crf is not None else \
            [em.argmax(-1)[i, :n].tolist() for i, n in enumerate(batch["lengths"].tolist())]
        tok_conf = [[probs[i, j, t].item() for j, t in enumerate(p)]
                    for i, p in enumerate(paths)]
        if self.crf is not None:
            path_t = torch.zeros_like(batch["tags"])
            for i, p in enumerate(paths):
                path_t[i, :len(p)] = torch.tensor(p, device=path_t.device)
            seq_conf = self.crf(em, path_t, mask=mask, reduction="none").exp().tolist()
        else:
            seq_conf = [float(torch.tensor(c).prod()) for c in tok_conf]
        return paths, tok_conf, seq_conf


# ---------------------------------------------------------------------------
# Simpan / muat checkpoint (vocab ikut disimpan, jadi API cukup 1 file .pt)
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, model: NERModel, config: dict,
                    wv: Vocab, cv: Vocab, extra: dict | None = None) -> None:
    torch.save({"state": model.state_dict(), "config": config,
                "words": wv.itos, "chars": cv.itos, "extra": extra or {}}, path)


def load_checkpoint(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    wv, cv = Vocab(ck["words"]), Vocab(ck["chars"])
    model = NERModel(len(wv), len(cv), len(S.TAGS), **ck["config"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), wv, cv