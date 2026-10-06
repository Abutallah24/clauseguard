"""Rule-based supplementary detector.

The trained model only knows the 8 UNFAIR-ToS categories. Contracts that students
actually sign (rent agreements, offer letters, subscriptions) often contain other
risky patterns, so a small set of transparent keyword rules flags those. Findings
from here are always labelled "Rule-based" in the UI so they are never confused
with the machine-learning predictions.
"""

import re
from dataclasses import dataclass

from src.labels import RULE_CATEGORY_INFO


@dataclass(frozen=True)
class RuleHit:
    category: str
    severity: float
    why: str


_PATTERNS: dict[str, list[str]] = {
    "Auto-renewal": [
        r"\bauto(?:matically)?[- ]?renew",
        r"\brenew(?:s|ed)? automatically\b",
        r"\brecurring (?:charge|payment|billing)",
        r"\bunless (?:you )?cancel",
    ],
    "Hidden fees / penalties": [
        r"\blate (?:fee|charge|payment (?:fee|charge))",
        r"\b(?:processing|convenience|service|admin(?:istrative)?|cancellation) (?:fee|charge)s?\b",
        r"\bpenalt(?:y|ies)\b",
        r"\binterest (?:at|of) \d",
        r"\bnon[- ]?refundable\b",
    ],
    "Data sharing": [
        r"\bshare (?:your |the )?(?:personal )?(?:data|information)\b.*\bthird[- ]part",
        r"\b(?:sell|disclose|transfer) (?:your |the )?(?:personal )?(?:data|information)",
        r"\bthird[- ]part(?:y|ies)\b.*\b(?:advertis|marketing|partners)",
        r"\baccess (?:to )?your (?:contacts|location|photos|sms|messages)",
    ],
    "Bond / lock-in penalty": [
        r"\bbond\b",
        r"\block[- ]?in (?:period|of)\b",
        r"\bminimum (?:service|commitment|tenure) (?:period|of)\b",
        r"\b(?:leave|resign|quit|vacate|terminate)s? (?:the [a-z]+ )?(?:before|prior to)\b.*\b(?:pay|liable|forfeit)",
    ],
    "Deposit forfeiture": [
        r"\b(?:deposit|advance)\b.*\bforfeit",
        r"\bforfeit\w*\b.*\b(?:deposit|advance)\b",
        r"\b(?:deposit|advance)\b.*\bnot (?:be )?refund",
    ],
}

_COMPILED = {cat: [re.compile(p, re.IGNORECASE) for p in pats] for cat, pats in _PATTERNS.items()}


def detect_rules(clause: str) -> list[RuleHit]:
    """Return one RuleHit per rule category that matches the clause."""
    hits = []
    for category, patterns in _COMPILED.items():
        if any(p.search(clause) for p in patterns):
            info = RULE_CATEGORY_INFO[category]
            hits.append(RuleHit(category, info["severity"], info["why"]))
    return hits
