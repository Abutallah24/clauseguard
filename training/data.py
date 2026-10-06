"""Download UNFAIR-ToS (LexGLUE) and save the official splits as CSV files.

Run:  .venv\\Scripts\\python -m training.data
Output: data/unfair_tos_{train,validation,test}.csv with columns
        text, labels (JSON list of original 0-7 label ids), label (9-class id, 0 = Fair)
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS = ("train", "validation", "test")


def primary_label(labels: list[int]) -> int:
    """9-class target: 0 = Fair, otherwise the first unfair label shifted by one."""
    return labels[0] + 1 if labels else 0


def _load_splits() -> dict[str, pd.DataFrame]:
    try:
        from datasets import load_dataset

        ds = load_dataset("coastalcph/lex_glue", "unfair_tos")
        return {s: ds[s].to_pandas() for s in SPLITS}
    except Exception as exc:  # newer `datasets` versions refuse script-based repos
        print(f"load_dataset failed ({exc!r}); falling back to the hub's parquet export")
        from huggingface_hub import HfApi, hf_hub_download

        files = HfApi().list_repo_files("coastalcph/lex_glue", repo_type="dataset", revision="refs/convert/parquet")
        frames = {}
        for split in SPLITS:
            parts = [f for f in files if f.startswith("unfair_tos/") and f"/{split}/" in f or f.startswith(f"unfair_tos/{split}")]
            paths = [hf_hub_download("coastalcph/lex_glue", f, repo_type="dataset", revision="refs/convert/parquet") for f in parts]
            frames[split] = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        return frames


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"unfair_tos_{split}.csv")
    df["labels"] = df["labels"].apply(json.loads)
    return df


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for split, df in _load_splits().items():
        df = df[["text", "labels"]].copy()
        df["labels"] = df["labels"].apply(lambda x: [int(v) for v in x])
        df["label"] = df["labels"].apply(primary_label)
        out = df.assign(labels=df["labels"].apply(json.dumps))
        out.to_csv(DATA_DIR / f"unfair_tos_{split}.csv", index=False)
        print(f"{split:<10} {len(df):>5} sentences, {int((df['label'] > 0).sum()):>4} unfair")


if __name__ == "__main__":
    main()
