"""End-to-end analysis: text -> clauses -> model + rules -> findings -> risk score."""

from collections import Counter
from dataclasses import dataclass, field

from src import models
from src.clauses import split_clauses
from src.labels import CATEGORY_INFO, FAIR, LABELS
from src.risk import Finding, document_score
from src.rules import detect_rules

# A clause is reported as unfair only if the model is at least this confident.
MIN_CONFIDENCE = 0.5


@dataclass
class AnalysisResult:
    text: str
    clauses: list[str]
    findings: list[Finding]
    score: int
    label: str
    model_key: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def risky_clause_count(self) -> int:
        return len({f.clause for f in self.findings})


def analyze(text: str, model_key: str | None = None) -> AnalysisResult:
    model_key = model_key or models.available_models()[0]
    clauses = split_clauses(text)
    predictions = models.predict(models.load(model_key), clauses)

    findings: list[Finding] = []
    for clause, (label_id, confidence) in zip(clauses, predictions):
        if label_id != FAIR and confidence >= MIN_CONFIDENCE:
            category = LABELS[label_id]
            info = CATEGORY_INFO[category]
            findings.append(Finding(clause, category, confidence, info["severity"], "model", info["why"]))
        for hit in detect_rules(clause):
            findings.append(Finding(clause, hit.category, 1.0, hit.severity, "rules", hit.why))

    # Most severe first so the UI shows the worst problems at the top.
    findings.sort(key=lambda f: f.risk, reverse=True)
    score, label = document_score(findings)
    counts = dict(Counter(f.category for f in findings).most_common())
    return AnalysisResult(text, clauses, findings, score, label, model_key, counts)
