"""Exploratory data analysis of UNFAIR-ToS. Run:  .venv\\Scripts\\python -m training.eda"""

import json
from pathlib import Path

from src.labels import LABELS
from training.charts import class_distribution
from training.data import load_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    splits = {s: load_split(s) for s in ("train", "validation", "test")}
    train = splits["train"]

    counts = {name: 0 for name in LABELS}
    for labels in train["labels"]:
        if not labels:
            counts["Fair"] += 1
        for v in labels:
            counts[LABELS[v + 1]] += 1

    lengths = train["text"].str.split().str.len()
    eda = {
        "split_sizes": {s: len(df) for s, df in splits.items()},
        "unfair_share": {s: round(float((df["label"] > 0).mean()), 4) for s, df in splits.items()},
        "multi_label_sentences": int((train["labels"].str.len() > 1).sum()),
        "train_class_counts": counts,
        "words_per_sentence": {"mean": round(float(lengths.mean()), 1), "median": int(lengths.median()),
                               "max": int(lengths.max())},
        "examples": {
            LABELS[i]: train.loc[train["label"] == i, "text"].iloc[0] for i in range(len(LABELS))
            if (train["label"] == i).any()
        },
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "eda.json").write_text(json.dumps(eda, indent=2))
    class_distribution(counts)
    print(json.dumps({k: v for k, v in eda.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
