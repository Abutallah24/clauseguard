"""HTML built for the app and the downloadable report must escape user text."""

import numpy as np

from app.ui.components import document_html, findings_html, verdict_html
from app.ui.report_html import build_report
from src.labels import CATEGORY_INFO
from src.models import _SklearnModel
from src.pipeline import AnalysisResult
from src.risk import Finding

EVIL = 'We may terminate <script>alert("x")</script> your account " onmouseover="alert(1) at any time.'


def _result():
    info = CATEGORY_INFO["Unilateral termination"]
    finding = Finding(EVIL, "Unilateral termination", 0.9, info["severity"], "model", info["why"])
    return AnalysisResult(EVIL, [EVIL], [finding], 50, "Medium", "tfidf_logreg", {"Unilateral termination": 1})


def test_user_text_is_escaped_everywhere():
    result = _result()
    verdict_html_output = verdict_html(result)
    for markup in (document_html(result), findings_html(result), verdict_html_output,
                   build_report(result, '<img src=x onerror=alert(1)>.pdf')):
        assert "<script>" not in markup
        assert 'onmouseover="alert' not in markup
        assert "<img src=x" not in markup
        if markup is not verdict_html_output:  # the verdict panel contains no user text
            assert "&lt;script&gt;" in markup


def test_sklearn_columns_are_mapped_to_label_ids():
    class FakePipeline:
        classes_ = np.array([0, 3, 8])  # model trained without some classes

        def predict_proba(self, texts):
            return np.array([[0.1, 0.2, 0.7]] * len(texts))

    model = _SklearnModel.__new__(_SklearnModel)
    model.pipeline = FakePipeline()
    proba = model.predict_proba(["a", "b"])
    assert proba.shape == (2, 9)
    assert proba[0, 8] == 0.7 and proba[0, 3] == 0.2 and proba[0, 1] == 0
