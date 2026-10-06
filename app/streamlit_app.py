"""ClauseGuard — Streamlit entry point.

Run from the project folder:  .venv\\Scripts\\streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # make `src` and `app` importable when Streamlit runs this file
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.ui.components import inject_css  # noqa: E402
from src import models  # noqa: E402

UI_DIR = ROOT / "app" / "ui"
st.set_page_config(page_title="ClauseGuard — find risky clauses", page_icon=str(UI_DIR / "icon.svg"), layout="wide")
st.logo(str(UI_DIR / "logo.svg"), size="large", icon_image=str(UI_DIR / "icon.svg"))
inject_css()

pages = st.navigation(
    [
        st.Page("views/analyze.py", title="Analyze a document", icon=":material/description:", default=True),
        st.Page("views/performance.py", title="Model performance", icon=":material/insights:"),
        st.Page("views/about.py", title="How it works", icon=":material/help:"),
    ]
)

with st.sidebar:
    st.caption("Reads terms and contracts so you don't sign something you didn't mean to.")
    available = models.available_models()
    if available:
        st.selectbox(
            "Model",
            available,
            format_func=lambda k: models.DISPLAY_NAMES[k] + (" (best)" if k == available[0] else ""),
            key="model_key",
            help="All three models were trained by us on the UNFAIR-ToS dataset. The best one by validation score is selected by default.",
        )
    else:
        st.error("No trained model found. Run the training scripts (see README).")

pages.run()
