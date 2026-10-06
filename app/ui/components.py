"""HTML building blocks for the Streamlit UI. All user text is escaped before it is rendered."""

from html import escape
from pathlib import Path

import streamlit as st

from src.labels import ALL_CATEGORY_INFO
from src.pipeline import AnalysisResult

CSS_PATH = Path(__file__).with_name("style.css")
LEVEL_TEXT = {"high": "High risk", "medium": "Medium risk", "low": "Low risk"}
BAR_COLOR = {"high": "var(--high)", "medium": "var(--medium)", "low": "var(--low)"}

LOGO_SVG = """<svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
<path d="M16 2.5 4.5 7v8.2c0 7 4.9 12.6 11.5 14.3 6.6-1.7 11.5-7.3 11.5-14.3V7L16 2.5Z" fill="#3a3fd1"/>
<path d="M10 11.5h12M10 15.5h12M10 19.5h7" stroke="#fff" stroke-width="2" stroke-linecap="round"/>
<path d="M19.5 20.5 22 23l4-4.5" stroke="#ffd88a" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def level_for(risk: float) -> str:
    """Map a clause's risk (severity x confidence) onto a three-step severity level."""
    if risk >= 0.65:
        return "high"
    if risk >= 0.45:
        return "medium"
    return "low"


def group_findings(result: AnalysisResult) -> list[dict]:
    """One entry per risky clause (a clause can have several issues), most severe first."""
    groups: dict[str, dict] = {}
    for f in result.findings:
        g = groups.setdefault(f.clause, {"clause": f.clause, "issues": [], "risk": 0.0})
        g["issues"].append(f)
        g["risk"] = max(g["risk"], f.risk)
    ordered = sorted(groups.values(), key=lambda g: g["risk"], reverse=True)
    for n, g in enumerate(ordered, start=1):
        g["n"], g["level"] = n, level_for(g["risk"])
    return ordered


def gauge_svg(score: int, level: str) -> str:
    """Semicircle gauge: grey track with Low/Medium/High zone ticks and a coloured value arc."""
    import math

    cx, cy, r = 110, 105, 88

    def point(value: float) -> tuple[float, float]:
        angle = math.pi * (1 - value / 100)  # 0 -> left end, 100 -> right end
        return cx + r * math.cos(angle), cy - r * math.sin(angle)

    def arc(start: float, end: float) -> str:
        (x1, y1), (x2, y2) = point(start), point(end)
        return f"M{x1:.1f},{y1:.1f} A{r},{r} 0 0 1 {x2:.1f},{y2:.1f}"

    ticks = "".join(
        f'<line x1="{point(v)[0]:.1f}" y1="{point(v)[1]:.1f}" '
        f'x2="{cx + (r + 12) * math.cos(math.pi * (1 - v / 100)):.1f}" '
        f'y2="{cy - (r + 12) * math.sin(math.pi * (1 - v / 100)):.1f}" stroke="#c3cbd6" stroke-width="2"/>'
        for v in (30, 60)
    )
    value = arc(0, max(score, 0.5)) if score > 0 else ""
    return f"""<svg viewBox="0 0 220 124" class="cg-gauge" role="img" aria-label="Risk score {score} out of 100, {LEVEL_TEXT[level]}">
<path d="{arc(0, 100)}" stroke="#edf1f6" stroke-width="16" fill="none" stroke-linecap="round"/>
<path d="{value}" stroke="{BAR_COLOR[level]}" stroke-width="16" fill="none" stroke-linecap="round"/>
{ticks}
<text x="{cx}" y="{cy - 8}" text-anchor="middle" class="cg-gauge-num">{score}</text>
<text x="{cx}" y="{cy + 14}" text-anchor="middle" class="cg-gauge-sub">out of 100</text>
<text x="18" y="122" class="cg-gauge-sub">0</text><text x="190" y="122" class="cg-gauge-sub">100</text>
</svg>"""


def verdict_html(result: AnalysisResult) -> str:
    level = result.label.lower()
    groups = group_findings(result)
    high = sum(1 for g in groups if g["level"] == "high")
    return f"""
<div class="cg-panel cg-verdict" role="region" aria-label="Risk score">
  <div class="cg-note">Document risk score</div>
  {gauge_svg(result.score, level)}
  <span class="cg-pill {level}">{escape(result.label)} risk</span>
  <div class="cg-facts">
    <div class="cg-fact"><b>{len(result.clauses)}</b><span>clauses read</span></div>
    <div class="cg-fact"><b>{len(groups)}</b><span>flagged</span></div>
    <div class="cg-fact"><b>{high}</b><span>high risk</span></div>
  </div>
</div>"""


def category_summary_html(result: AnalysisResult) -> str:
    if not result.counts:
        return ""
    top = max(result.counts.values())
    rows = []
    for category, count in result.counts.items():
        severity = ALL_CATEGORY_INFO[category]["severity"]
        level = level_for(severity)
        rows.append(
            f'<div class="cg-cat"><span>{escape(category)}</span><span class="count">{count}</span>'
            f'<span class="bar"><i style="width:{100 * count / top:.0f}%;background:{BAR_COLOR[level]}"></i></span></div>'
        )
    return '<div class="cg-panel"><div class="cg-note" style="margin-bottom:.6rem">What was found</div>' \
           f'<div class="cg-cats">{"".join(rows)}</div></div>'


def document_html(result: AnalysisResult) -> str:
    """The whole document, clause by clause, with risky clauses highlighted and numbered in the margin."""
    by_clause = {g["clause"]: g for g in group_findings(result)}
    paragraphs = []
    for clause in result.clauses:
        g = by_clause.get(clause)
        if g:
            title = escape(", ".join(sorted({f.category for f in g["issues"]})))
            paragraphs.append(
                f'<p><span class="ref {g["level"]}" aria-hidden="true">{g["n"]}</span>'
                f'<mark class="{g["level"]}" title="{title}">{escape(clause)}</mark></p>'
            )
        else:
            paragraphs.append(f"<p>{escape(clause)}</p>")
    legend = ('<div class="cg-legend" aria-hidden="true"><span class="high">High risk</span>'
              '<span class="medium">Medium risk</span><span class="low">Low risk</span></div>')
    return f'{legend}<div class="cg-doc" tabindex="0" aria-label="Document with risky clauses highlighted">{"".join(paragraphs)}</div>'


def findings_html(result: AnalysisResult) -> str:
    items = []
    for g in group_findings(result):
        issues = []
        for f in sorted(g["issues"], key=lambda f: f.risk, reverse=True):
            source = ('<span class="cg-tag ai">AI model</span>' if f.source == "model"
                      else '<span class="cg-tag">Rule check</span>')
            conf = f'<span class="cg-conf">{f.confidence:.0%} confidence</span>' if f.source == "model" else ""
            issues.append(
                f'<div class="cg-issue"><div class="cg-issue-head">{escape(f.category)} {source} {conf}</div>'
                f"<p>{escape(f.why)}</p></div>"
            )
        items.append(
            f'<article class="cg-finding"><span class="num {g["level"]}" role="img" '
            f'aria-label="{LEVEL_TEXT[g["level"]]}, clause {g["n"]}">{g["n"]}</span>'
            f'<div><blockquote>{escape(g["clause"])}</blockquote>{"".join(issues)}</div></article>'
        )
    return f'<div class="cg-panel" style="padding-top:.2rem;padding-bottom:.2rem">{"".join(items)}</div>'


def empty_findings_html() -> str:
    return ('<div class="cg-empty"><b>No risky clauses found.</b>'
            "The model and the rule checks did not flag anything in this document. "
            "Still read anything about payments, cancellation and your data before you accept.</div>")
