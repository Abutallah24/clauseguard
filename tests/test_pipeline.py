import pytest

from src import models
from src.pipeline import analyze

pytestmark = pytest.mark.skipif(not models.available_models(), reason="no trained model on disk")

UNFAIR = (
    "We may terminate your account at any time without notice and for any reason. "
    "Any dispute shall be resolved exclusively by binding arbitration and you waive any class action. "
    "We reserve the right to modify these terms at any time at our sole discretion."
)
FAIR = "The office is open from 9 am to 6 pm on weekdays. You can contact us by email for any help you need."


@pytest.mark.parametrize("model_key", ["tfidf_logreg", "tfidf_svm"])
def test_unfair_text_is_flagged(model_key):
    result = analyze(UNFAIR, model_key)
    model_categories = {f.category for f in result.findings if f.source == "model"}
    assert len(result.clauses) == 3
    assert {"Arbitration", "Unilateral termination"} <= model_categories
    assert result.score > 30


def test_fair_text_scores_low():
    result = analyze(FAIR, "tfidf_logreg")
    assert result.findings == []
    assert (result.score, result.label) == (0, "Low")


def test_findings_sorted_by_risk():
    result = analyze(UNFAIR, "tfidf_logreg")
    risks = [f.risk for f in result.findings]
    assert risks == sorted(risks, reverse=True)
