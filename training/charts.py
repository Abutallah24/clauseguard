"""Static PNG charts for the report and the Model Performance page.

Colours follow the dataviz reference palette: categorical slots 1-3 (blue, orange, aqua)
for the three models, a single blue sequential ramp for confusion matrices, and
recessive grey axes/gridlines. Run:  .venv\\Scripts\\python -m training.charts
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.labels import LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "reports" / "metrics.json"

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3, validated all-pairs
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
MODEL_ORDER = ["tfidf_logreg", "tfidf_svm", "distilbert"]


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    return fig, ax


def _title(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2)


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, facecolor=SURFACE)
    plt.close(fig)


def class_distribution(counts: dict[str, int]) -> None:
    """Single-series horizontal bar chart of training sentences per class (log scale: Fair dominates)."""
    names = list(counts)[::-1]
    values = [counts[n] for n in names]
    fig, ax = _figure(7, 3.8)
    bars = ax.barh(names, values, color=SERIES[0], height=0.62)
    ax.set_xscale("log")
    _style(ax, "x")
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() * 1.08, bar.get_y() + bar.get_height() / 2, f"{v:,}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Training sentences (log scale)", color=MUTED, fontsize=9)
    _title(ax, "Class distribution — UNFAIR-ToS training split", "Fair sentences outnumber every unfair type by far")
    _save(fig, "class_distribution.png")


def model_comparison(metrics: dict) -> None:
    keys = [k for k in MODEL_ORDER if k in metrics["models"]]
    groups = [("Accuracy", lambda m: m["accuracy"]), ("Macro F1", lambda m: m["macro"]["f1"]),
              ("Micro F1", lambda m: m["micro"]["f1"]), ("Unfair-only\nmacro F1", lambda m: m["unfair_only"]["macro_f1"]),
              ("Fair vs Unfair\nF1", lambda m: m["binary"]["f1"])]
    x = np.arange(len(groups))
    width = 0.8 / len(keys)
    fig, ax = _figure(8, 4)
    for i, key in enumerate(keys):
        m = metrics["models"][key]
        vals = [fn(m) for _, fn in groups]
        bars = ax.bar(x + (i - (len(keys) - 1) / 2) * width, vals, width * 0.92, color=SERIES[i], label=m["name"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(x, [g for g, _ in groups])
    ax.set_ylim(0, 1.08)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(keys), labelcolor=INK_2)
    _title(ax, "Model comparison on the test split", "Higher is better · all scores 0–1")
    _save(fig, "model_comparison.png")


def per_class_f1(metrics: dict) -> None:
    keys = [k for k in MODEL_ORDER if k in metrics["models"]]
    names = LABELS[::-1]
    y = np.arange(len(names))
    height = 0.8 / len(keys)
    fig, ax = _figure(7.5, 5)
    for i, key in enumerate(keys):
        m = metrics["models"][key]
        vals = [m["per_class"][n]["f1-score"] for n in names]
        ax.barh(y - (i - (len(keys) - 1) / 2) * height, vals, height * 0.9, color=SERIES[i], label=m["name"])
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1)
    _style(ax, "x")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(keys), labelcolor=INK_2)
    _title(ax, "F1 score per clause type (test split)")
    _save(fig, "per_class_f1.png")


def confusion(metrics: dict, key: str) -> None:
    m = metrics["models"][key]
    cm = np.array(m["confusion_matrix"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)  # row-normalised: recall per true class
    fig, ax = _figure(7, 6)
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    short = ["Fair", "Liability", "Termination", "Change", "Removal", "By using", "Law", "Jurisdiction", "Arbitration"]
    ax.set_xticks(range(len(short)), short, rotation=40, ha="right")
    ax.set_yticks(range(len(short)), short)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for i in range(len(short)):
        for j in range(len(short)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xlabel("Predicted", color=MUTED, fontsize=9)
    ax.set_ylabel("Actual", color=MUTED, fontsize=9)
    _title(ax, f"Confusion matrix — {m['name']}", "Counts; shading = share of the actual class")
    _save(fig, f"confusion_{key}.png")


def architecture() -> None:
    """System diagram: inputs -> text extraction -> clause splitting -> model + rules -> risk score -> report."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 3.1), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 31)
    ax.axis("off")

    def box(x, y, w, h, title, body, fill="#ffffff", edge=AXIS, title_color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    linewidth=1.2, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=6.8, color=INK_2, linespacing=1.45)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=MUTED, linewidth=1.2))

    box(0.5, 4, 14, 23, "Inputs", "PDF / scanned PDF\nWord (.docx)\nPhotos (multi-page)\nCamera\nWeb link\nPasted text")
    box(19, 4, 14, 23, "Text extraction", "PyMuPDF text layer\nRapidOCR for scans\n& photos\npython-docx\ntrafilatura (links)")
    box(36.5, 4, 15, 23, "Clause splitting", "Sentences &\nnumbered items;\nheadings dropped")
    box(55, 17, 17, 12.5, "Trained classifier", "TF-IDF + LogReg / SVM\nor DistilBERT\n9 classes", fill="#eef0ff", edge="#3a3fd1", title_color="#3a3fd1")
    box(55, 1.5, 17, 12.5, "Rule checks", "auto-renewal, fees,\ndata sharing, bond,\ndeposit forfeiture")
    box(76.5, 4, 10, 23, "Risk score", "severity ×\nconfidence\n\n0–100\nLow / Med /\nHigh")
    box(90.5, 4, 9, 23, "Report", "highlighted\nclauses +\nplain-English\nreasons")

    arrow(15, 15.5, 18.6, 15.5)
    arrow(33.5, 15.5, 36.1, 15.5)
    arrow(52, 17, 54.6, 22)
    arrow(52, 14, 54.6, 8)
    arrow(72.5, 22, 76.1, 17.5)
    arrow(72.5, 8, 76.1, 13.5)
    arrow(87, 15.5, 90.1, 15.5)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "architecture.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    architecture()
    metrics = json.loads(METRICS_PATH.read_text())
    model_comparison(metrics)
    per_class_f1(metrics)
    for key in metrics["models"]:
        confusion(metrics, key)
    print(f"charts written to {FIG_DIR}")


if __name__ == "__main__":
    main()
