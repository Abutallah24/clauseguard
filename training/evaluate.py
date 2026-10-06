"""Shared evaluation: computes every metric for one model and stores it in reports/metrics.json."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.labels import LABELS

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "metrics.json"
UNFAIR_IDS = list(range(1, len(LABELS)))


def evaluate(y_true, y_pred, name: str, extra: dict | None = None) -> dict:
    """Metrics on the 9-class task, the 8 unfair classes only (LexGLUE style), and Fair-vs-Unfair."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(LABELS)))

    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="micro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    bin_true, bin_pred = (y_true > 0).astype(int), (y_pred > 0).astype(int)
    bp, br, bf, _ = precision_recall_fscore_support(bin_true, bin_pred, average="binary", zero_division=0)

    result = {
        "name": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro": {"precision": p_mac, "recall": r_mac, "f1": f_mac},
        "micro": {"precision": p_mic, "recall": r_mic, "f1": f_mic},
        "weighted": {"precision": p_w, "recall": r_w, "f1": f_w},
        "unfair_only": {
            "micro_f1": f1_score(y_true, y_pred, labels=UNFAIR_IDS, average="micro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, labels=UNFAIR_IDS, average="macro", zero_division=0),
        },
        "binary": {"precision": bp, "recall": br, "f1": bf, "accuracy": accuracy_score(bin_true, bin_pred)},
        "per_class": classification_report(
            y_true, y_pred, labels=labels, target_names=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_test": int(len(y_true)),
    }
    if extra:
        result.update(extra)
    return _to_builtin(result)


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_metrics(key: str, result: dict) -> None:
    """Merge one model's results into reports/metrics.json (other models are kept)."""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {"models": {}}
    data["models"][key] = result
    best = max(data["models"].items(), key=lambda kv: kv[1]["validation_macro_f1"])
    data["best_model"] = best[0]
    METRICS_PATH.write_text(json.dumps(data, indent=2))


def print_summary(result: dict) -> None:
    print(
        f"{result['name']}: acc={result['accuracy']:.3f} macroF1={result['macro']['f1']:.3f} "
        f"microF1={result['micro']['f1']:.3f} unfair-macroF1={result['unfair_only']['macro_f1']:.3f} "
        f"binaryF1={result['binary']['f1']:.3f}"
    )
