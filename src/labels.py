"""Clause categories used by ClauseGuard.

LABELS follows the UNFAIR-ToS (LexGLUE) label order, shifted by one so that
index 0 is the extra "Fair" class we add for 9-way classification.
"""

LABELS: list[str] = [
    "Fair",
    "Limitation of liability",
    "Unilateral termination",
    "Unilateral change",
    "Content removal",
    "Contract by using",
    "Choice of law",
    "Jurisdiction",
    "Arbitration",
]

FAIR = 0

# severity: how much this clause type can hurt the user (0-1), used by the risk score.
# short: one-line label for UI chips. why: plain-English explanation shown to the user.
CATEGORY_INFO: dict[str, dict] = {
    "Limitation of liability": {
        "severity": 0.85,
        "short": "Company avoids responsibility",
        "why": "The company limits or excludes its responsibility for losses, data loss or damage "
        "it causes. If something goes wrong, you may not be able to claim compensation.",
    },
    "Unilateral termination": {
        "severity": 0.75,
        "short": "Can cut you off anytime",
        "why": "The company can suspend or close your account or the contract at any time, "
        "sometimes without a reason or notice, and you may lose access to what you paid for.",
    },
    "Unilateral change": {
        "severity": 0.8,
        "short": "Can change terms anytime",
        "why": "The company can change the terms, prices or service whenever it wants. Continuing "
        "to use the service usually counts as accepting the new terms.",
    },
    "Content removal": {
        "severity": 0.6,
        "short": "Can delete your content",
        "why": "The company can remove or delete your content (posts, files, data) at its own "
        "discretion, often without warning or a way to appeal.",
    },
    "Contract by using": {
        "severity": 0.55,
        "short": "Bound just by using it",
        "why": "You are legally bound by the terms just by visiting or using the service, even if "
        "you never read or explicitly accepted them.",
    },
    "Choice of law": {
        "severity": 0.4,
        "short": "Foreign law applies",
        "why": "Disputes are governed by the law of a place chosen by the company, which may give "
        "you fewer protections than your own country's consumer laws.",
    },
    "Jurisdiction": {
        "severity": 0.5,
        "short": "Court far from you",
        "why": "Any case must be filed in courts chosen by the company, possibly in another city or "
        "country, making it expensive and impractical for you to take legal action.",
    },
    "Arbitration": {
        "severity": 0.9,
        "short": "No court, no class action",
        "why": "Disputes must go to private arbitration instead of a court, often with a waiver of "
        "class actions. This usually favours the company and limits your legal options.",
    },
}

# Categories found by the rule-based supplementary detector (not part of the trained model).
RULE_CATEGORY_INFO: dict[str, dict] = {
    "Auto-renewal": {
        "severity": 0.6,
        "short": "Renews automatically",
        "why": "The subscription or contract renews and charges you automatically unless you "
        "remember to cancel in time.",
    },
    "Hidden fees / penalties": {
        "severity": 0.6,
        "short": "Extra charges",
        "why": "Extra fees, late charges or penalties may be added on top of the price you expect.",
    },
    "Data sharing": {
        "severity": 0.65,
        "short": "Shares your data",
        "why": "Your personal data may be shared with or sold to third parties such as advertisers "
        "or partners.",
    },
    "Bond / lock-in penalty": {
        "severity": 0.8,
        "short": "Pay to leave early",
        "why": "You are locked in for a minimum period and must pay a penalty or bond amount if you "
        "leave early.",
    },
    "Deposit forfeiture": {
        "severity": 0.7,
        "short": "Can keep your deposit",
        "why": "Your security deposit or advance can be kept (forfeited) by the other party under "
        "broad conditions.",
    },
}

ALL_CATEGORY_INFO: dict[str, dict] = {**CATEGORY_INFO, **RULE_CATEGORY_INFO}
