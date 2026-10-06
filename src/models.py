"""Load the trained models and predict a (label id, confidence) pair for each clause."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
METRICS_PATH = ROOT / "reports" / "metrics.json"

DISPLAY_NAMES = {
    "distilbert": "DistilBERT (deep learning)",
    "tfidf_svm": "TF-IDF + Linear SVM",
    "tfidf_logreg": "TF-IDF + Logistic Regression",
}


def _exists(key: str) -> bool:
    if key == "distilbert":
        return (MODELS_DIR / "distilbert" / "config.json").exists()
    return (MODELS_DIR / f"{key}.joblib").exists()


def available_models() -> list[str]:
    """Model keys that are trained and on disk, best (by validation macro-F1) first."""
    keys = [k for k in DISPLAY_NAMES if _exists(k)]
    if METRICS_PATH.exists():
        scores = {k: v.get("validation_macro_f1", 0) for k, v in json.loads(METRICS_PATH.read_text())["models"].items()}
        keys.sort(key=lambda k: scores.get(k, 0), reverse=True)
    return keys


class _SklearnModel:
    def __init__(self, path: Path):
        import joblib

        self.pipeline = joblib.load(path)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        proba = self.pipeline.predict_proba(texts)
        # classes_ may not be 0..8 in order; map columns onto label ids.
        full = np.zeros((len(texts), 9))
        full[:, self.pipeline.classes_] = proba
        return full


class _TransformerModel:
    BATCH = 32

    def __init__(self, path: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.BATCH):
                enc = self.tokenizer(texts[i:i + self.BATCH], truncation=True, max_length=128,
                                     padding=True, return_tensors="pt")
                out.append(self.torch.softmax(self.model(**enc).logits, dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, 9))


@lru_cache(maxsize=3)
def load(key: str):
    if not _exists(key):
        raise FileNotFoundError(f"Model '{key}' is not trained yet. Run the training scripts first.")
    if key == "distilbert":
        return _TransformerModel(MODELS_DIR / "distilbert")
    return _SklearnModel(MODELS_DIR / f"{key}.joblib")


def predict(model, clauses: list[str]) -> list[tuple[int, float]]:
    """For each clause return (predicted label id, probability of that label)."""
    if not clauses:
        return []
    proba = model.predict_proba(clauses)
    ids = proba.argmax(axis=1)
    return [(int(i), float(proba[row, i])) for row, i in enumerate(ids)]
