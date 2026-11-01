"""
src/tokenizer.py
Satu-satunya tokenizer proyek. Generator, preprocessing, baseline, dan API
semuanya WAJIB memakai ini supaya token saat training = token saat inference.
"""
import re
from typing import List, Tuple

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

_ATTACH_PREV = {",", ".", "!", "?", ";", ":", ")", "-", "/", "%"}
_ATTACH_NEXT = {"(", "-", "/"}


def tokenize(text: str) -> List[str]:
    """Kata/angka utuh = 1 token, tiap tanda baca = 1 token."""
    return _TOKEN_RE.findall(text)

def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Token yang sama dengan tokenize(), plus posisi karakter (start, end)."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]

def detokenize(tokens: List[str]) -> str:
    """Kebalikan tokenize (spasi dirapikan). tokenize(detokenize(t)) == t."""
    out: List[str] = []
    quote_open = False
    no_space = True
    for tok in tokens:
        if tok == '"':
            if not quote_open:
                if out:
                    out.append(" ")
                out.append(tok)
                quote_open, no_space = True, True
            else:
                out.append(tok)
                quote_open, no_space = False, False
            continue
        if not out or tok in _ATTACH_PREV or no_space:
            out.append(tok)
        else:
            out.append(" " + tok)
        no_space = tok in _ATTACH_NEXT
    return "".join(out)


if __name__ == "__main__":
    s = 'halo kak, pesan sajadah abu-abu "Selamat wisuda, sukses" tgl 20/11 H-3 acara'
    t = tokenize(s)
    print(t)
    print(detokenize(t))
    assert tokenize(detokenize(t)) == t
    print("OK")