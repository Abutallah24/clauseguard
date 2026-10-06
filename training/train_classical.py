"""Train the two classical models: TF-IDF + Logistic Regression and TF-IDF + Linear SVM.

Run:  .venv\\Scripts\\python -m training.train_classical
"""

import time
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, make_union
from sklearn.svm import LinearSVC

from training.data import load_split
from training.evaluate import evaluate, print_summary, save_metrics

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def expand_multilabel(df):
    """A sentence with two unfair labels becomes two training rows (one per label)."""
    rows = df.explode("labels")
    rows["label"] = rows["labels"].apply(lambda v: 0 if v != v else int(v) + 1)  # NaN (no labels) -> Fair
    return rows["text"].tolist(), rows["label"].tolist()


def features():
    # Word 1-2 grams capture phrases like "sole discretion"; char 3-5 grams add robustness to OCR typos.
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60000),
    )


MODELS = {
    "tfidf_logreg": (
        "TF-IDF + Logistic Regression",
        lambda: LogisticRegression(C=10, class_weight="balanced", max_iter=3000),
    ),
    "tfidf_svm": (
        "TF-IDF + Linear SVM",
        # Calibration wraps the SVM so it outputs probabilities (needed for confidence + risk score).
        lambda: CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3),
    ),
}


def main() -> None:
    train, val, test = load_split("train"), load_split("validation"), load_split("test")
    x_train, y_train = expand_multilabel(train)
    MODELS_DIR.mkdir(exist_ok=True)

    for key, (name, make_clf) in MODELS.items():
        start = time.time()
        model = Pipeline([("features", features()), ("clf", make_clf())])
        model.fit(x_train, y_train)
        train_seconds = time.time() - start

        val_f1 = f1_score(val["label"], model.predict(val["text"]), average="macro", zero_division=0)
        result = evaluate(
            test["label"],
            model.predict(test["text"]),
            name,
            extra={"validation_macro_f1": val_f1, "train_seconds": round(train_seconds, 1), "family": "classical"},
        )
        save_metrics(key, result)
        joblib.dump(model, MODELS_DIR / f"{key}.joblib", compress=3)
        print_summary(result)
        print(f"  validation macro-F1={val_f1:.3f}, trained in {train_seconds:.1f}s")


if __name__ == "__main__":
    main()
