"""Model Performance page: real test-set metrics and charts produced by the training scripts."""

import json
from html import escape
from pathlib import Path

import streamlit as st

from app.ui.components import html

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "reports" / "metrics.json"
EDA = ROOT / "reports" / "eda.json"
FIGURES = ROOT / "reports" / "figures"

html('<h1 class="cg-title">Model performance</h1>')
html(
    '<p class="cg-lede">Every number here was measured on the official UNFAIR-ToS test split '
    "(sentences the models never saw during training). They are read from <code>reports/metrics.json</code>, "
    "which the training scripts write.</p>"
)

if not METRICS.exists():
    st.info("No metrics yet. Run the training scripts first (see README).")
    st.stop()

metrics = json.loads(METRICS.read_text())
best = metrics["best_model"]
order = [k for k in ("tfidf_logreg", "tfidf_svm", "distilbert") if k in metrics["models"]]

rows = []
for key in order:
    m = metrics["models"][key]
    cells = [m["accuracy"], m["macro"]["precision"], m["macro"]["recall"], m["macro"]["f1"],
             m["unfair_only"]["macro_f1"], m["binary"]["f1"]]
    minutes = m.get("train_seconds", 0) / 60
    train_time = f"{minutes:.1f} min" if minutes >= 1 else f"{m.get('train_seconds', 0):.0f} s"
    rows.append(
        f'<tr class="{"best" if key == best else ""}"><td>{escape(m["name"])}{' <span class="cg-tag ai">Best</span>' if key == best else ""}</td>'
        + "".join(f"<td>{v:.3f}</td>" for v in cells) + f"<td>{train_time}</td></tr>"
    )
html(
    '<div class="cg-table-wrap"><table class="cg-table"><thead><tr><th>Model</th><th>Accuracy</th>'
    "<th>Macro precision</th><th>Macro recall</th><th>Macro F1</th><th>Unfair-only macro F1</th>"
    f'<th>Fair vs unfair F1</th><th>Training time</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
)
html(
    '<p class="cg-note" style="margin-top:.6rem">Best = model with the best validation macro F1, used by default in the app. '
    "Macro F1 averages all 9 classes equally, so rare clause types count as much as common ones — that is why it is "
    "lower than accuracy, which is inflated by the ~90% of sentences that are fair.</p>"
)

html('<h2 class="cg-h2">Comparison</h2>')
def chart(name: str) -> None:
    path = FIGURES / name
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.info("This chart has not been generated yet. Run: .venv\Scripts\python -m training.charts")


chart("model_comparison.png")

c1, c2 = st.columns(2, gap="large")
with c1:
    html('<h2 class="cg-h2">F1 per clause type</h2>')
    chart("per_class_f1.png")
with c2:
    html('<h2 class="cg-h2">Confusion matrix</h2>')
    choice = st.selectbox("Model", order, index=order.index(best),
                          format_func=lambda k: metrics["models"][k]["name"], label_visibility="collapsed")
    chart(f"confusion_{choice}.png")

html('<h2 class="cg-h2">The dataset</h2>')
if EDA.exists():
    eda = json.loads(EDA.read_text())
    sizes = eda["split_sizes"]
    html(
        f'<p class="cg-lede" style="margin-bottom:.8rem">UNFAIR-ToS (LexGLUE benchmark): {sum(sizes.values()):,} sentences '
        f'from 50 real online terms of service — {sizes["train"]:,} for training, {sizes["validation"]:,} for validation and '
        f'{sizes["test"]:,} for testing. Only about {eda["unfair_share"]["train"]:.0%} of sentences are unfair, '
        "so the models use class weights to avoid always predicting “fair”.</p>"
    )
chart("class_distribution.png")

with st.expander("Per-class precision, recall and F1"):
    for key in order:
        m = metrics["models"][key]
        st.markdown(f"**{m['name']}**")
        table = {name: v for name, v in m["per_class"].items() if isinstance(v, dict) and "support" in v
                 and name not in ("macro avg", "weighted avg", "micro avg")}
        st.dataframe(
            {"Class": list(table), "Precision": [round(v["precision"], 3) for v in table.values()],
             "Recall": [round(v["recall"], 3) for v in table.values()],
             "F1": [round(v["f1-score"], 3) for v in table.values()],
             "Test sentences": [int(v["support"]) for v in table.values()]},
            hide_index=True, use_container_width=True,
        )
