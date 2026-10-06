"""Write the real results table (from reports/metrics.json) into README.md and VIVA_NOTES.md.

Run after training:  .venv\\Scripts\\python scripts/fill_results.py
The table lives between <!-- RESULTS:START --> and <!-- RESULTS:END --> so the script can be re-run.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import MODEL_ORDER, load_metrics  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def table() -> str:
    metrics = load_metrics()
    best = metrics["best_model"]
    lines = [
        "| Model | Accuracy | Macro F1 | Unfair-only macro F1 | Fair vs unfair F1 | Training time |",
        "|---|---|---|---|---|---|",
    ]
    for key in MODEL_ORDER:
        if key not in metrics["models"]:
            continue
        m = metrics["models"][key]
        secs = m.get("train_seconds", 0)
        duration = f"{secs / 60:.0f} min" if secs >= 60 else f"{secs:.0f} s"
        name = f"**{m['name']}** (used by default)" if key == best else m["name"]
        lines.append(f"| {name} | {m['accuracy']:.3f} | {m['macro']['f1']:.3f} | {m['unfair_only']['macro_f1']:.3f} "
                     f"| {m['binary']['f1']:.3f} | {duration} |")
    lines.append("")
    lines.append(f"Default model = best validation macro F1 ({metrics['models'][best]['validation_macro_f1']:.3f}).")
    return "\n".join(lines)


def fill(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    wrapped = f"{START}\n{block}\n{END}"
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: wrapped, text, flags=re.S)
    else:
        text = text.replace("RESULTS_TABLE_PLACEHOLDER", wrapped)
    path.write_text(text, encoding="utf-8")
    print("updated", path.name)


if __name__ == "__main__":
    block = table()
    for name in ("README.md", "VIVA_NOTES.md"):
        fill(ROOT / name, block)
    print(block)
