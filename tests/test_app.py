"""Streamlit app behaviour, driven headlessly with Streamlit's AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import models

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
pytestmark = pytest.mark.skipif(len(models.available_models()) < 2, reason="needs two trained models")


def _html(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def _run_sample(at):
    at.run(timeout=120)
    next(b for b in at.button if b.label == "Rental agreement").click().run(timeout=120)
    return at


def test_switching_model_reanalyses_the_document():
    at = _run_sample(AppTest.from_file(APP, default_timeout=120))
    first = at.session_state["model_key"]
    other = next(k for k in models.available_models() if k != first)
    assert models.DISPLAY_NAMES[first] in _html(at)

    at.selectbox(key="model_key").set_value(other).run(timeout=120)
    assert at.session_state["result"].model_key == other
    assert models.DISPLAY_NAMES[other] in _html(at)


def test_failed_analysis_clears_previous_report():
    at = _run_sample(AppTest.from_file(APP, default_timeout=120))
    assert "Report for" in _html(at)
    next(b for b in at.button if b.label == "Analyze text").click().run(timeout=120)  # empty paste
    assert at.error, "an error message should be shown"
    assert "Report for" not in _html(at)


def test_truncated_text_shows_warning():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run(timeout=120)
    at.text_area[0].input("We may terminate your account at any time without notice. " * 4000)
    next(b for b in at.button if b.label == "Analyze text").click().run(timeout=180)
    assert any("only the first" in w.value for w in at.warning)
