"""
src/generator.py
Generator data sintetis + anotasi BIO otomatis + split anti-bocor.

Jalankan:  python -m src.generator
Output  :  data/processed/{train,val,test}.jsonl + stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from src import schema as S
from src.tokenizer import detokenize, tokenize

# ---------------------------------------------------------------------------
# 1. TEMPLATE per split (keluarga template TIDAK dibagi antar split)
#    Template = list klausa. "[...]" = klausa opsional. Klausa bisa diacak urutannya.
#    Klausa tanpa {PRODUK} juga bisa hilang sesekali (simulasi pesan tidak lengkap).
# ---------------------------------------------------------------------------
TEMPLATES: Dict[str, Dict[str, List[List[str]]]] = {
    "train": {
        "custom": [
            ["mau pesan {PRODUK}", "{JUMLAH}", "nama yang dibordir {NAMA_PERSONALISASI}",
             "[warna {WARNA}]", "[ukuran {UKURAN}]", "[butuh {TENGGAT}]"],
            ["order {PRODUK} {JUMLAH}", "namanya {NAMA_PERSONALISASI}",
             "[warnanya {WARNA}]", "[deadline {TENGGAT}]"],
            ["bisa custom {PRODUK} {WARNA}", "[mau {JUMLAH}]",
             "tulisan nama {NAMA_PERSONALISASI}", "[selesai {TENGGAT}]"],
            ["pesan {JUMLAH} {PRODUK} {PENERIMA_ACARA}", "nama {NAMA_PERSONALISASI}",
             "[warna {WARNA}]", "[sebelum {TENGGAT}]"],
            ["{PRODUK} {JUMLAH}", "nama {NAMA_PERSONALISASI}", "[ukuran {UKURAN}]",
             "[{PENERIMA_ACARA}]"],
            ["tolong buatkan {PRODUK} dengan nama {NAMA_PERSONALISASI}", "{JUMLAH}",
             "[ukuran {UKURAN}]", "[warna {WARNA}]", "[butuh {TENGGAT}]"],
        ],
        "hampers": [
            ["mau pesan {PRODUK}", "{JUMLAH}", "[{PENERIMA_ACARA}]",
             '[kartu ucapan tulisannya "{TEKS_KARTU}"]', "[butuh {TENGGAT}]"],
            ["order {PRODUK} {JUMLAH}", "[tema warna {WARNA}]",
             '[kartunya: "{TEKS_KARTU}"]', "[dikirim {TENGGAT}]"],
            ["{PRODUK} {PENERIMA_ACARA}", "jumlah {JUMLAH}", "[tema {WARNA}]",
             '[kartu ucapan "{TEKS_KARTU}"]', "[deadline {TENGGAT}]"],
        ],
        "plain": [
            ["ready {PRODUK}?", "[warna {WARNA}]", "[ukuran {UKURAN}]", "{JUMLAH}"],
            ["beli {JUMLAH} {PRODUK}", "[warna {WARNA}]", "[butuh {TENGGAT}]"],
            ["stok {PRODUK} {WARNA} ada?", "[mau {JUMLAH}]", "[ukuran {UKURAN}]"],
        ],
    },
    "val": {
        "custom": [
            ["ada {PRODUK} bisa request nama?", "namanya {NAMA_PERSONALISASI}", "{JUMLAH}",
             "[warna {WARNA}]", "[perlu {TENGGAT}]"],
            ["saya butuh {JUMLAH} {PRODUK}", "bordir {NAMA_PERSONALISASI}",
             "[ukuran {UKURAN}]", "[kirim {TENGGAT}]"],
            ["pengen {PRODUK} warna {WARNA}", "isi nama {NAMA_PERSONALISASI}",
             "[jumlahnya {JUMLAH}]", "[harus jadi {TENGGAT}]"],
            ["{PRODUK} {PENERIMA_ACARA}", "nama {NAMA_PERSONALISASI}", "{JUMLAH}",
             "[mohon selesai {TENGGAT}]"],
        ],
        "hampers": [
            ["cari {PRODUK}", "sebanyak {JUMLAH}", "[{PENERIMA_ACARA}]",
             '[tolong tulis di kartu "{TEKS_KARTU}"]', "[kirim {TENGGAT}]"],
            ["pesan {JUMLAH} {PRODUK}", "[warna {WARNA}]",
             '[ucapan kartu: "{TEKS_KARTU}"]', "[perlu {TENGGAT}]"],
        ],
        "plain": [
            ["{PRODUK} masih ada?", "[warna {WARNA}]", "[ambil {JUMLAH}]", "[ukuran {UKURAN}]"],
            ["mau beli {PRODUK} {JUMLAH}", "[sampai {TENGGAT}]", "[warna {WARNA}]"],
        ],
    },
    "test": {
        "custom": [
            ["boleh pesan {PRODUK}?", "personalisasi {NAMA_PERSONALISASI}", "totalnya {JUMLAH}",
             "[maunya {WARNA}]", "[size {UKURAN}]", "[dipakai {TENGGAT}]"],
            ["order custom {PRODUK} {PENERIMA_ACARA}", "cetak nama {NAMA_PERSONALISASI}",
             "qty {JUMLAH}", "[tolong siap {TENGGAT}]"],
            ["{JUMLAH} {PRODUK} ya", "tertulis {NAMA_PERSONALISASI}",
             "[warna dasar {WARNA}]", "[tenggat {TENGGAT}]"],
            ["mau tanya, {PRODUK} dengan nama {NAMA_PERSONALISASI} bisa?", "butuh {JUMLAH}",
             "[warna {WARNA}]", "[sebelum {TENGGAT}]"],
        ],
        "hampers": [
            ["mau {PRODUK} {JUMLAH}", "[{PENERIMA_ACARA}]",
             '[isi kartu "{TEKS_KARTU}"]', "[sebelum {TENGGAT}]"],
            ["bisa {JUMLAH} {PRODUK}?", "[bertema {WARNA}]",
             '[kartunya bertuliskan "{TEKS_KARTU}"]', "[butuh {TENGGAT}]"],
        ],
        "plain": [
            ["cek stok {PRODUK}", "[maunya {WARNA}]", "[sebanyak {JUMLAH}]", "[ukuran {UKURAN}]"],
            ["{JUMLAH} {PRODUK} {WARNA} tersedia?", "[perlu {TENGGAT}]"],
        ],
    },
}

TEMPLATES["train"]["custom"] += [
    ["halo, aku mau {PRODUK}", "namanya {NAMA_PERSONALISASI} ya", "{JUMLAH}",
     "[warna {WARNA}]", "[tenggatnya {TENGGAT}]"],
    ["pesan {PRODUK} {WARNA} {JUMLAH}", "nama bordir: {NAMA_PERSONALISASI}",
     "[ukuran {UKURAN}]", "[maksimal {TENGGAT}]"],
    ["{JUMLAH} {PRODUK} ya kak", "isi nama {NAMA_PERSONALISASI}",
     "[warna yang {WARNA}]", "[paling lambat {TENGGAT}]"],
    ["request {PRODUK} nama {NAMA_PERSONALISASI}", "sebanyak {JUMLAH}",
     "[ukuran {UKURAN}]", "[{PENERIMA_ACARA}]", "[paling lambat {TENGGAT}]"],
    ["kak, {PRODUK} custom dong", "nama {NAMA_PERSONALISASI}", "{JUMLAH}",
     "[{PENERIMA_ACARA}]", "[warna {WARNA}]"],
]
TEMPLATES["train"]["hampers"] += [
    ["kak mau {PRODUK}", "sebanyak {JUMLAH}", "[{PENERIMA_ACARA}]",
     '[kartu: "{TEKS_KARTU}"]', "[paling lambat {TENGGAT}]"],
    ["{JUMLAH} {PRODUK}", "[warna {WARNA}]", '[di kartu tulis "{TEKS_KARTU}"]',
     "[butuh {TENGGAT}]"],
]
TEMPLATES["train"]["plain"] += [
    ["{PRODUK} {WARNA} ada?", "mau {JUMLAH}", "[ukuran {UKURAN}]"],
    ["mau beli {PRODUK}", "[warna {WARNA}]", "[{JUMLAH}]", "[butuh {TENGGAT}]"],
]

KINDS = ["custom", "hampers", "plain"]
KIND_WEIGHTS = [0.50, 0.25, 0.25]
KIND_ORDER_TYPES = {
    "custom": ("Custom Sajadah", "Custom Handuk"),
    "hampers": ("Hampers",),
    "plain": ("Ready Stock", "Other"),
}

# ---------------------------------------------------------------------------
# 2. Augmentasi (gaya chat asli). Semua one-to-one di level token -> label aman.
# ---------------------------------------------------------------------------
ABBR = {
    "yang": "yg", "tanggal": "tgl", "untuk": "utk", "tolong": "tlg",
    "dengan": "dgn", "tidak": "gak", "bisa": "bs", "sebelum": "sblm",
    "pcs": "pc", "buah": "bh", "warna": "wrn", "ukuran": "uk",
    "jumlah": "jml", "kirim": "krm", "nama": "nm", "sebanyak": "sbnyk",
    "deadline": "dl", "tersedia": "tsedia", "butuh": "btuh",
}
GREETINGS = ["halo kak", "hai kak", "permisi kak", "kak", "min", "halo min",
             "assalamualaikum", "selamat pagi kak", "selamat siang kak"]
CLOSINGS = ["ya", "makasih", "terima kasih kak", "thanks ya", "mohon info",
            "ditunggu kak", "makasih kak"]


def typo(word: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(word) - 1)  # huruf pertama dipertahankan
    op = rng.choice(["swap", "drop", "dup", "vowel"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "dup":
        return word[:i] + word[i] + word[i:]
    if word[i].lower() in "aiueo":
        return word[:i] + rng.choice("aiueo") + word[i + 1:]
    return word


FILLERS = ["yang", "ya", "kira-kira", "pokoknya", "nih", "aja", "kalau", "jenis",
           "model", "utama", "tuh", "gitu"]


def augment(tokens: List[str], tags: List[str], rng: random.Random,
            fillers: bool = False) -> Tuple[List[str], List[str]]:
    tokens, tags = list(tokens), list(tags)
    if rng.random() < 0.6:  # singkatan
        tokens = [ABBR.get(t.lower(), t) if rng.random() < 0.5 else t for t in tokens]
    if rng.random() < 0.6:  # typo
        tokens = [typo(t, rng) if (t.isalpha() and len(t) >= 4 and rng.random() < 0.07) else t
                  for t in tokens]
    if fillers and rng.random() < 0.35:   # kata pengisi sebelum entitas (hanya train)
        nt, ng = [], []
        for t, g in zip(tokens, tags):
            if g.startswith("B-") and rng.random() < 0.25:
                f = tokenize(rng.choice(FILLERS))
                nt += f
                ng += ["O"] * len(f)
            nt.append(t)
            ng.append(g)
        tokens, tags = nt, ng
    if rng.random() < 0.5:  # sapaan di depan
        pre = tokenize(rng.choice(GREETINGS))
        if rng.random() < 0.5:
            pre.append(",")
        tokens, tags = pre + tokens, ["O"] * len(pre) + tags
    if rng.random() < 0.35:  # penutup
        suf = tokenize(rng.choice(CLOSINGS))
        tokens, tags = tokens + suf, tags + ["O"] * len(suf)
    if rng.random() < 0.4:  # huruf kecil semua
        tokens = [t.lower() for t in tokens]
    return tokens, tags

# ---------------------------------------------------------------------------
# 3. Sampler nilai entitas
# ---------------------------------------------------------------------------
def pool_for_split(pool: List[str], split: str) -> List[str]:
    """Bagi pool terbuka (mis. teks kartu) jadi train/val/test tanpa tumpang tindih."""
    n = len(pool)
    n_test = max(1, round(0.2 * n))
    n_val = max(1, round(0.15 * n))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Pool terlalu kecil untuk dibagi tiga split")
    parts = {"train": pool[:n_train],
             "val": pool[n_train:n_train + n_val],
             "test": pool[n_train + n_val:]}
    return parts[split]


class ValueSampler:
    def __init__(self, split: str, rng: random.Random):
        self.rng = rng
        self.names = S.NAMES_BY_SPLIT[split]
        self.cards = pool_for_split(S.CARD_TEXTS, split)

    def sample(self, label: str, cache: Dict[str, str]) -> str:
        if label not in cache:
            cache[label] = getattr(self, f"_{label}")()
        return cache[label]

    def _NAMA_PERSONALISASI(self) -> str:
        if self.rng.random() < 0.7:
            return self.rng.choice(self.names)
        a, b = self.rng.sample(self.names, 2)
        return f"{a} {self.rng.choice(['&', 'dan'])} {b}"

    def _JUMLAH(self) -> str:
        r = self.rng
        if r.random() < 0.7:
            n = r.choice([1, 2, 3, 4, 5, 6, 10, 12, 20, 24, 25, 30, 50, 100])
            sep = "" if r.random() < 0.15 else " "
            return f"{n}{sep}{r.choice(S.QTY_UNITS)}"
        word = r.choice([w for w in S.QTY_WORDS if w != "lusin"])
        units = [u for u in S.QTY_UNITS if not (u == "lusin" and word in
                 ("sepuluh", "dua belas", "dua puluh"))]
        return f"{word} {r.choice(units)}"

    def _WARNA(self) -> str:
        return self.rng.choice(S.COLORS)

    def _UKURAN(self) -> str:
        r = self.rng
        if r.random() < 0.4:
            return r.choice(S.SIZE_LETTERS)
        return f"{r.choice(S.SIZE_DIMS)} {r.choice(S.SIZE_UNITS)}".strip()

    def _TENGGAT(self) -> str:
        r = self.rng
        d, m = r.randint(1, 30), r.choice(S.MONTHS)
        x = r.random()
        if x < 0.3:
            return f"tanggal {d} {m}"
        if x < 0.5:
            return f"{d} {m}"
        if x < 0.6:
            return f"tgl {d} {m}"
        if x < 0.7:
            return f"{d}/{r.randint(1, 12)}"
        return r.choice(S.DEADLINE_RELATIVE)

    def _TEKS_KARTU(self) -> str:
        return self.rng.choice(self.cards)

    def _PENERIMA_ACARA(self) -> str:
        r = self.rng
        x = r.random()
        if x < 0.5:
            return f"untuk {r.choice(S.RECIPIENTS)}"
        if x < 0.8:
            return r.choice(S.EVENTS)
        return f"acara {r.choice(S.EVENTS)}"

# ---------------------------------------------------------------------------
# 4. Render template -> segmen -> token + BIO
# ---------------------------------------------------------------------------
PH = re.compile(r"(\{[A-Z_]+\})")


def resolve_clauses(template: List[str], rng: random.Random,
                    keep_opt: float = 0.8, keep_req: float = 0.92) -> List[str]:
    out = []
    for clause in template:
        optional = clause.startswith("[") and clause.endswith("]")
        text = clause[1:-1] if optional else clause
        if "{PRODUK}" in text:
            keep = True
        else:
            keep = rng.random() < (keep_opt if optional else keep_req)
        if keep:
            out.append(text)
    return out


def build_segments(clauses: List[str], value_fn, rng: random.Random,
                   shuffle_p: float = 0.35) -> List[Tuple[str, str | None]]:
    clauses = list(clauses)
    if rng.random() < shuffle_p:
        rng.shuffle(clauses)
    segs: List[Tuple[str, str | None]] = []
    for i, clause in enumerate(clauses):
        if i:
            segs.append((rng.choice([", ", " ", ". ", " "]), None))
        for part in PH.split(clause):
            if not part:
                continue
            if part.startswith("{"):
                label = part[1:-1]
                assert label in S.ENTITY_LABELS, f"label template tidak dikenal: {label}"
                segs.append((value_fn(label), label))
            else:
                segs.append((part, None))
    return segs


def to_bio(segs: List[Tuple[str, str | None]]) -> Tuple[List[str], List[str]]:
    tokens, tags = [], []
    for text, label in segs:
        for j, tok in enumerate(tokenize(text)):
            tokens.append(tok)
            if label is None:
                tags.append("O")
            else:
                tags.append(f"{'B' if j == 0 else 'I'}-{label}")
    return tokens, tags


def extract_spans(tokens: List[str], tags: List[str]) -> List[Tuple[str, str]]:
    """BIO -> [(label, teks)]. Dipakai statistik, evaluasi, dan API nanti."""
    spans, cur = [], None
    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if cur:
                spans.append(cur)
            cur = [tag[2:], [tok]]
        elif tag.startswith("I-") and cur and cur[0] == tag[2:]:
            cur[1].append(tok)
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    return [(label, detokenize(toks)) for label, toks in spans]

# ---------------------------------------------------------------------------
# 5. Generate satu split
# ---------------------------------------------------------------------------
def generate_split(split: str, n: int, rng: random.Random) -> List[dict]:
    sampler = ValueSampler(split, rng)
    products = {k: [t for t, o in S.PRODUCT_TERMS.items() if o in ots]
                for k, ots in KIND_ORDER_TYPES.items()}
    records, seen, attempts = [], set(), 0

    while len(records) < n and attempts < n * 30:
        attempts += 1
        kind = rng.choices(KINDS, weights=KIND_WEIGHTS)[0]
        t_idx = rng.randrange(len(TEMPLATES[split][kind]))
        template = TEMPLATES[split][kind][t_idx]
        product = rng.choice(products[kind])
        cache = {"PRODUK": product}

        clauses = resolve_clauses(template, rng)
        segs = build_segments(clauses, lambda lb: sampler.sample(lb, cache), rng)
        tokens, tags = to_bio(segs)

        augmented = rng.random() >= 0.15  # 15% pesan dibiarkan bersih
        if augmented:
            tokens, tags = augment(tokens, tags, rng, fillers=(split == "train"))

        text = detokenize(tokens)
        if tokenize(text) != tokens:       # jaga konsistensi tokenizer
            continue
        if text.lower() in seen:
            continue
        seen.add(text.lower())

        assert len(tokens) == len(tags) and all(t in S.TAG2ID for t in tags)
        records.append({
            "id": f"{split}-{len(records):05d}",
            "split": split,
            "text": text,
            "tokens": tokens,
            "tags": tags,
            "order_type": S.PRODUCT_TERMS[product],
            "template_id": f"{split}-{kind}-{t_idx}",
            "augmented": augmented,
        })
    return records

# ---------------------------------------------------------------------------
# 6. Cek kebocoran + statistik
# ---------------------------------------------------------------------------
def name_tokens(records: List[dict]) -> set:
    out = set()
    for r in records:
        for label, text in extract_spans(r["tokens"], r["tags"]):
            if label == "NAMA_PERSONALISASI":
                out |= {t.lower() for t in tokenize(text)} - {"&", "dan"}
    return out


def drop_overlap(records: List[dict], reference: List[dict]) -> List[dict]:
    ref = {r["text"].lower() for r in reference}
    return [r for r in records if r["text"].lower() not in ref]


def compute_stats(data: Dict[str, List[dict]]) -> dict:
    stats = {}
    for split, recs in data.items():
        ent = Counter()
        for r in recs:
            for label, _ in extract_spans(r["tokens"], r["tags"]):
                ent[label] += 1
        stats[split] = {
            "n_messages": len(recs),
            "avg_tokens": round(sum(len(r["tokens"]) for r in recs) / max(1, len(recs)), 1),
            "order_types": dict(Counter(r["order_type"] for r in recs)),
            "entity_spans": {k: ent.get(k, 0) for k in S.ENTITY_LABELS},
        }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/processed")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = {
        "train": generate_split("train", args.n_train, rng),
        "val": generate_split("val", args.n_val, rng),
        "test": generate_split("test", args.n_test, rng),
    }

    # buang teks yang kebetulan sama antar split
    data["val"] = drop_overlap(data["val"], data["train"])
    data["test"] = drop_overlap(data["test"], data["train"] + data["val"])

    # cek nama bocor (peringatan, bukan error: typo bisa menciptakan kebetulan)
    tr = name_tokens(data["train"])
    for sp in ("val", "test"):
        leak = tr & name_tokens(data[sp])
        if leak:
            print(f"[PERINGATAN] nama bocor train vs {sp}: {sorted(leak)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, recs in data.items():
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = compute_stats(data)
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n--- contoh train ---")
    for r in data["train"][:8]:
        print(r["text"])
        print("   ", r["order_type"], extract_spans(r["tokens"], r["tags"]))


if __name__ == "__main__":
    main()