"""Load the trained models and predict a (label id, confidence) pair for each clause."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.labels import LABELS

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
METRICS_PATH = ROOT / "reports" / "metrics.json"

# One place for model keys, their order (charts, tables, report) and their names (must match training/).
MODEL_ORDER = ["tfidf_logreg", "tfidf_svm", "distilbert"]
DISPLAY_NAMES = {
    "tfidf_logreg": "TF-IDF + Logistic Regression",
    "tfidf_svm": "TF-IDF + Linear SVM",
    "distilbert": "DistilBERT (fine-tuned)",
}
N_CLASSES = len(LABELS)


@lru_cache(maxsize=1)
def _metrics_for(mtime: float) -> dict:
    return json.loads(METRICS_PATH.read_text())


def load_metrics() -> dict | None:
    """reports/metrics.json, parsed once and re-read only when the file changes."""
    if not METRICS_PATH.exists():
        return None
    return _metrics_for(METRICS_PATH.stat().st_mtime)


def _exists(key: str) -> bool:
    if key == "distilbert":
        return (MODELS_DIR / "distilbert" / "config.json").exists()
    return (MODELS_DIR / f"{key}.joblib").exists()


def available_models() -> list[str]:
    """Model keys that are trained and on disk, best (by validation macro-F1) first."""
    keys = [k for k in MODEL_ORDER if _exists(k)]
    metrics = load_metrics()
    if metrics:
        scores = {k: v.get("validation_macro_f1", 0) for k, v in metrics["models"].items()}
        keys.sort(key=lambda k: scores.get(k, 0), reverse=True)
    return keys


class _SklearnModel:
    def __init__(self, path: Path):
        import joblib

        self.pipeline = joblib.load(path)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        proba = self.pipeline.predict_proba(texts)
        # classes_ may not be 0..8 in order; map columns onto label ids.
        full = np.zeros((len(texts), N_CLASSES))
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
        return np.concatenate(out) if out else np.zeros((0, N_CLASSES))


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
