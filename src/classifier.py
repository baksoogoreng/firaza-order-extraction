"""
src/classifier.py
Klasifier jenis pesanan: TF-IDF karakter + Logistic Regression (scikit-learn, dilatih dari nol).
Latih & evaluasi:  python -m src.classifier
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from src import schema as S
from src.preprocess import load_jsonl

CLF_PATH = "models/order_clf.joblib"


def build_pipeline() -> Pipeline:
    # n-gram KARAKTER supaya tahan typo ("sajdah", "hameprs")
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, analyzer="char_wb",
                                  ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
        ("lr", LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")),
    ])


def load_classifier(path: str = CLF_PATH):
    return joblib.load(path)


def predict_one(clf, text: str) -> Tuple[str, float]:
    proba = clf.predict_proba([text])[0]
    i = int(proba.argmax())
    return str(clf.classes_[i]), float(proba[i])


# Pembanding: aturan kata kunci sederhana (tanpa belajar)
_TERMS = sorted(S.PRODUCT_TERMS, key=len, reverse=True)


def rule_predict(text: str) -> str:
    t = text.lower()
    for term in _TERMS:
        if term in t:
            return S.PRODUCT_TERMS[term]
    return "Other"


def evaluate(name: str, recs: List[dict], clf) -> dict:
    X = [r["text"] for r in recs]
    y = [r["order_type"] for r in recs]
    pred = list(clf.predict(X))
    rule = [rule_predict(x) for x in X]
    print(f"\n===== Klasifier | {name} ({len(recs)} pesan) =====")
    print(classification_report(y, pred, digits=3, zero_division=0))
    print("Confusion matrix (baris=gold, kolom=pred), urutan:", S.ORDER_TYPES)
    print(confusion_matrix(y, pred, labels=S.ORDER_TYPES))
    out = {
        "model_macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "model_accuracy": round(sum(p == g for p, g in zip(pred, y)) / len(y), 4),
        "rule_macro_f1": round(f1_score(y, rule, average="macro"), 4),
        "rule_accuracy": round(sum(p == g for p, g in zip(rule, y)) / len(y), 4),
    }
    print("Ringkasan:", out)
    shown = 0
    for x, g, p in zip(X, y, pred):
        if g != p and shown < 8:
            print(f"  SALAH | gold={g} | pred={p} | {x}")
            shown += 1
    return out


def main() -> None:
    d = Path("data/processed")
    train, val, test = (load_jsonl(d / f"{s}.jsonl") for s in ("train", "val", "test"))

    clf = build_pipeline()
    clf.fit([r["text"] for r in train], [r["order_type"] for r in train])
    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf, CLF_PATH)

    scores = {"val": evaluate("val", val, clf), "test": evaluate("test", test, clf)}
    hw = Path("data/handwritten_test.jsonl")
    if hw.exists():
        recs = [r for r in load_jsonl(hw) if r.get("order_type")]
        if recs:
            scores["handwritten"] = evaluate("handwritten", recs, clf)

    Path("docs").mkdir(exist_ok=True)
    with open("docs/clf_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"\nModel: {CLF_PATH} | skor: docs/clf_scores.json")


if __name__ == "__main__":
    main()