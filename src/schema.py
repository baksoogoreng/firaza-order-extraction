"""
src/schema.py
Single source of truth: label entitas, tag BIO, skema JSON, aturan field per produk,
dan kamus awal. File lain (generator, preprocess, model, api) HARUS mengimpor dari
sini supaya label tidak pernah tidak konsisten.
"""
from __future__ import annotations

import random as _random
from typing import Dict, List

# ---------------------------------------------------------------------------
# 1. Entitas & panduan anotasi (dipakai juga di dokumentasi)
# ---------------------------------------------------------------------------
ENTITY_LABELS: List[str] = [
    "NAMA_PERSONALISASI",  # nama yang dicetak/dibordir di produk
    "PRODUK",
    "JUMLAH",
    "WARNA",
    "UKURAN",
    "TENGGAT",
    "TEKS_KARTU",          # isi ucapan di kartu
    "PENERIMA_ACARA",      # penerima (buat ibu) atau acara (nikahan, wisuda)
]

LABEL_GUIDE: Dict[str, str] = {
    "NAMA_PERSONALISASI": "Nama yang akan dicetak/dibordir. Bukan nama pemesan. "
                          "Contoh: 'Rizky & Salsa', 'Aisyah'.",
    "PRODUK": "Frasa produk lengkap beserta modifier jenis. "
              "Contoh: 'sajadah travel', 'handuk bordir', 'hampers lebaran'.",
    "JUMLAH": "Angka/kata bilangan BESERTA satuannya. "
              "Contoh: '2 pcs', 'dua lusin', '50 set'.",
    "WARNA": "Hanya kata warna. Contoh: 'navy', 'dusty pink'.",
    "UKURAN": "Ukuran huruf atau dimensi beserta satuan. "
              "Contoh: 'XL', '70x140 cm'.",
    "TENGGAT": "Seluruh frasa waktu. Contoh: 'tanggal 20 november', "
               "'H-3 acara', 'minggu depan'.",
    "TEKS_KARTU": "Isi ucapan kartu saja, tanpa kata pengantar 'tulisannya'.",
    "PENERIMA_ACARA": "Relasi penerima atau nama acara. "
                      "Contoh: 'untuk ibu', 'nikahan', 'wisuda'.",
}

# ---------------------------------------------------------------------------
# 2. Tag BIO
# ---------------------------------------------------------------------------
PAD_TAG = "<PAD>"


def build_tag_list() -> List[str]:
    tags = ["O"]
    for label in ENTITY_LABELS:
        tags.append(f"B-{label}")
        tags.append(f"I-{label}")
    return tags


TAGS: List[str] = build_tag_list()
TAG2ID: Dict[str, int] = {t: i for i, t in enumerate(TAGS)}
ID2TAG: Dict[int, str] = {i: t for t, i in TAG2ID.items()}

# ---------------------------------------------------------------------------
# 3. Jenis pesanan, skema JSON output, dan aturan field per produk
# ---------------------------------------------------------------------------
ORDER_TYPES: List[str] = [
    "Ready Stock", "Custom Sajadah", "Custom Handuk", "Hampers", "Other",
]


def empty_result() -> dict:
    """Skema JSON output API. Dikunci sejak sekarang."""
    return {
        "jenis_pesanan": None,
        "entities": {label: [] for label in ENTITY_LABELS},
        "missing_fields": [],
        "low_confidence": [],
    }


# Aturan field per kategori produk (ASUMSI DESAIN, belum divalidasi ke Firaza)
ALWAYS_REQUIRED: List[str] = ["PRODUK", "JUMLAH"]

FIELD_RULES: Dict[str, Dict[str, List[str]]] = {
    "sajadah": {"required": ["NAMA_PERSONALISASI", "UKURAN", "TENGGAT"],
                "optional": ["WARNA"]},
    "handuk":  {"required": ["NAMA_PERSONALISASI", "UKURAN", "TENGGAT"],
                "optional": ["WARNA"]},
    "mug":     {"required": ["NAMA_PERSONALISASI", "TENGGAT"],
                "optional": ["WARNA"]},
    "hampers": {"required": ["TENGGAT"],
                "optional": ["WARNA", "TEKS_KARTU", "PENERIMA_ACARA"]},
    "mukena":  {"required": ["UKURAN"],
                "optional": ["WARNA", "TENGGAT"]},
    "unknown": {"required": ["TENGGAT"], "optional": []},
}

# kata kunci -> kategori (dicek berurutan, memakai substring)
CATEGORY_KEYWORDS = [
    ("sajadah", "sajadah"), ("handuk", "handuk"),
    ("mug", "mug"), ("tumbler", "mug"),
    ("hampers", "hampers"), ("kado", "hampers"),
    ("mukena", "mukena"),
]

# kategori yang punya pilihan ukuran (dipakai generator)
SIZE_CATEGORIES = {"sajadah", "handuk", "mukena"}


def category_of(product_text: str | None) -> str:
    p = (product_text or "").lower()
    for kw, cat in CATEGORY_KEYWORDS:
        if kw in p:
            return cat
    return "unknown"


def missing_fields(entities: Dict[str, List[str]]) -> List[str]:
    """Label wajib yang masih kosong. Tidak menebak nilai apa pun."""
    products = entities.get("PRODUK") or []
    cat = category_of(products[0] if products else None)
    needed = ALWAYS_REQUIRED + FIELD_RULES[cat]["required"]
    return [f for f in needed if not entities.get(f)]


# ---------------------------------------------------------------------------
# 4. Kamus
# ---------------------------------------------------------------------------
# Nama dipisah per split supaya TIDAK ada nama yang muncul di dua split.
_REAL_NAMES: Dict[str, List[str]] = {
    "train": [
        "Aisyah", "Budi", "Citra", "Dimas", "Eka", "Farhan", "Gita", "Hana",
        "Indra", "Jihan", "Kiki", "Laras", "Maya", "Naufal", "Okta", "Putri",
        "Rizky", "Salsa", "Tegar", "Umar", "Vina", "Wulan", "Yusuf", "Zahra",
        "Alya", "Bagas", "Dewi", "Fikri", "Hafiz", "Intan",
    ],
    "val": ["Nadia", "Reza", "Siti", "Taufik", "Ulfa", "Wahyu", "Yoga", "Zaki",
            "Bella", "Fajar"],
    "test": ["Amira", "Cahyo", "Davina", "Erlangga", "Fadila", "Galih", "Hilda",
             "Ilham", "Jasmine", "Kevin"],
}

_BLOCKED = {"nama", "namanya", "mama", "papa", "kaka", "bisa", "mau", "sama", "tamu",
            "satu", "dua", "tiga", "lima", "ibu", "adik", "nenek", "guru", "mug",
            "hari", "bulan", "tahun", "kado", "warna", "paket", "stok", "kartu"}


def _synthetic_names(n: int, seed: int = 7) -> List[str]:
    """Nama bergaya Indonesia dari suku kata; jumlah besar supaya tidak bisa dihafal."""
    rng = _random.Random(seed)
    cons = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "r", "s", "t",
            "w", "y", "z", "sy", "kh"]
    vows = ["a", "i", "u", "e", "o", "a", "i"]
    ends = ["", "", "n", "h", "ah", "ni", "ra", "ya", "na", "a", "l", "m", "ka"]
    known = {x.lower() for v in _REAL_NAMES.values() for x in v}
    out: set = set()
    while len(out) < n:
        k = rng.choice([2, 2, 3])
        w = "".join(rng.choice(cons) + rng.choice(vows) for _ in range(k)) + rng.choice(ends)
        if len(w) >= 4 and w not in known and w not in _BLOCKED:
            out.add(w)
    return sorted(w.capitalize() for w in out)


_extra = _synthetic_names(400)
_rng = _random.Random(11)
_rng.shuffle(_extra)
_c1, _c2 = int(len(_extra) * 0.70), int(len(_extra) * 0.85)

NAMES_BY_SPLIT: Dict[str, List[str]] = {
    "train": _REAL_NAMES["train"] + _extra[:_c1],
    "val": _REAL_NAMES["val"] + _extra[_c1:_c2],
    "test": _REAL_NAMES["test"] + _extra[_c2:],
}

# istilah produk -> jenis pesanan (dipakai generator untuk label klasifier)
PRODUCT_TERMS: Dict[str, str] = {
    "sajadah": "Custom Sajadah",
    "sajadah travel": "Custom Sajadah",
    "sajadah anak": "Custom Sajadah",
    "sajadah bordir": "Custom Sajadah",
    "handuk": "Custom Handuk",
    "handuk mandi": "Custom Handuk",
    "handuk bordir": "Custom Handuk",
    "handuk kimono": "Custom Handuk",
    "handuk bayi": "Custom Handuk",
    "hampers": "Hampers",
    "hampers lebaran": "Hampers",
    "hampers ramadan": "Hampers",
    "hampers wisuda": "Hampers",
    "kado pernikahan": "Hampers",
    "paket kado": "Hampers",
    "mukena": "Ready Stock",
    "tumbler": "Other",
    "mug": "Other",
}

COLORS: List[str] = [
    "navy", "maroon", "hitam", "putih", "abu-abu", "hijau sage", "dusty pink",
    "cream", "coklat", "biru muda", "emerald", "mocca", "tosca", "ungu",
]

SIZE_LETTERS: List[str] = ["S", "M", "L", "XL", "XXL"]
SIZE_DIMS: List[str] = ["30x50", "50x100", "60x120", "70x140", "100x200"]
SIZE_UNITS: List[str] = ["cm", ""]  # kadang customer lupa menulis satuan

QTY_WORDS: Dict[str, int] = {
    "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
    "sepuluh": 10, "dua belas": 12, "dua puluh": 20, "lusin": 12,
}
QTY_UNITS: List[str] = ["pcs", "buah", "set", "pasang", "lusin", "biji"]

EVENTS: List[str] = [
    "nikahan", "pernikahan", "ulang tahun", "wisuda", "lebaran", "aqiqah",
    "tunangan", "umroh", "reuni", "syukuran",
]
RECIPIENTS: List[str] = [
    "ibu", "bapak", "mama", "papa", "kakak", "adik", "calon suami",
    "calon istri", "guru", "sahabat", "mertua", "nenek",
]

MONTHS: List[str] = [
    "januari", "februari", "maret", "april", "mei", "juni", "juli",
    "agustus", "september", "oktober", "november", "desember",
]
DEADLINE_RELATIVE: List[str] = [
    "besok", "lusa", "minggu depan", "bulan depan", "akhir bulan",
    "secepatnya", "hari jumat", "H-3 acara", "H-7 acara",
]

CARD_TEXTS: List[str] = [
    "Semoga selalu dalam lindungan Allah",
    "Terima kasih sudah menjadi sahabat terbaik",
    "Selamat Idul Fitri, mohon maaf lahir dan batin",
    "Happy birthday, semoga panjang umur",
    "Congratulations on your wedding",
    "Selamat atas kelulusanmu",
    "Semoga berkah dan bahagia selalu",
    "Barakallah atas pernikahan kalian",
    "Doa terbaik untuk kalian berdua",
    "Sukses terus ya, bangga sama kamu",
    "Selamat atas kelahiran si kecil",
    "Semoga umrahnya mabrur",
]

# ---------------------------------------------------------------------------
# 5. Self-test:  python -m src.schema
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # a) tidak ada nama yang bocor antar split
    s = {k: set(v) for k, v in NAMES_BY_SPLIT.items()}
    assert not (s["train"] & s["val"]), "nama bocor train/val"
    assert not (s["train"] & s["test"]), "nama bocor train/test"
    assert not (s["val"] & s["test"]), "nama bocor val/test"
    # b) jumlah tag benar: 1 (O) + 2 per entitas
    assert len(TAGS) == 1 + 2 * len(ENTITY_LABELS)
    # c) aturan field per produk
    assert category_of("sajadah travel") == "sajadah"
    assert "UKURAN" not in missing_fields({"PRODUK": ["mug"], "JUMLAH": ["2 pcs"],
                                           "NAMA_PERSONALISASI": ["Aisyah"],
                                           "TENGGAT": ["besok"]})
    assert "UKURAN" in missing_fields({"PRODUK": ["sajadah"], "JUMLAH": ["2 pcs"]})
    print("OK. Jumlah tag:", len(TAGS))
    print("Nama per split:", {k: len(v) for k, v in NAMES_BY_SPLIT.items()})
    print("Contoh output kosong:", empty_result())