import math

from src.labels import CATEGORY_INFO, LABELS
from src.risk import Finding, document_score, risk_label


def make(category, confidence=1.0, source="model"):
    return Finding(
        clause="x" * 30,
        category=category,
        confidence=confidence,
        severity=CATEGORY_INFO[category]["severity"],
        source=source,
        why=CATEGORY_INFO[category]["why"],
    )


def test_labels_order_matches_lexglue():
    assert LABELS[0] == "Fair"
    assert LABELS[1:] == [
        "Limitation of liability",
        "Unilateral termination",
        "Unilateral change",
        "Content removal",
        "Contract by using",
        "Choice of law",
        "Jurisdiction",
        "Arbitration",
    ]


def test_every_category_has_severity_and_explanation():
    for name in LABELS[1:]:
        info = CATEGORY_INFO[name]
        assert 0 < info["severity"] <= 1
        assert len(info["why"]) > 20


def test_score_zero_when_no_findings():
    assert document_score([]) == (0, "Low")


def test_score_matches_formula():
    findings = [make("Arbitration", 0.8)]
    expected = round(100 * (1 - math.exp(-(0.9 * 0.8) / 2.5)))
    assert document_score(findings)[0] == expected


def test_score_grows_with_more_findings_and_stays_below_100():
    one = document_score([make("Arbitration")])[0]
    many = document_score([make("Arbitration")] * 20)[0]
    assert one < many <= 100


def test_risk_label_thresholds():
    assert risk_label(29) == "Low"
    assert risk_label(30) == "Medium"
    assert risk_label(60) == "Medium"
    assert risk_label(61) == "High"
