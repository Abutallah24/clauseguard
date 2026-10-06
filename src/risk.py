"""Turn individual risky-clause findings into a 0-100 document risk score."""

import math
from dataclasses import dataclass

# Larger value = score rises more slowly. With 2.5, one high-severity clause gives ~30
# (Medium) and three to four give > 60 (High).
SCORE_SCALE = 2.5
LOW_MAX = 29  # 0-29 Low
MEDIUM_MAX = 60  # 30-60 Medium, 61-100 High


@dataclass
class Finding:
    clause: str
    category: str
    confidence: float  # model probability (1.0 for rule hits)
    severity: float  # category weight in [0, 1]
    source: str  # "model" or "rules"
    why: str

    @property
    def risk(self) -> float:
        return self.severity * self.confidence


def risk_label(score: int) -> str:
    if score <= LOW_MAX:
        return "Low"
    if score <= MEDIUM_MAX:
        return "Medium"
    return "High"


def document_score(findings: list[Finding]) -> tuple[int, str]:
    """score = 100 * (1 - exp(-sum(severity * confidence) / SCORE_SCALE)).

    Each extra risky clause adds less than the previous one, so the score saturates
    towards 100 but never exceeds it.
    """
    total = sum(f.risk for f in findings)
    score = round(100 * (1 - math.exp(-total / SCORE_SCALE)))
    return score, risk_label(score)
