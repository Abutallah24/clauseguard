"""Self-contained HTML report the user can download (opens in any browser, prints to PDF)."""

from datetime import datetime
from html import escape

from app.ui.components import CSS_PATH, document_html, findings_html, verdict_html
from src.models import DISPLAY_NAMES
from src.pipeline import AnalysisResult


def build_report(result: AnalysisResult, source_name: str) -> str:
    css = CSS_PATH.read_text(encoding="utf-8")
    generated = datetime.now().strftime("%d %b %Y, %H:%M")
    findings = findings_html(result) if result.findings else "<p>No risky clauses were found.</p>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ClauseGuard report — {escape(source_name)}</title>
<style>{css}
body {{ background: var(--plane); margin: 0; }}
main {{ max-width: 900px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; display: grid; gap: 1.5rem; }}
.cg-doc {{ max-height: none; }}
@media print {{ .cg-panel, .cg-doc {{ box-shadow: none; }} }}
</style></head>
<body><main>
<header>
  <h1 class="cg-title">ClauseGuard <strong>risk report</strong></h1>
  <p class="cg-lede">Source: {escape(source_name)}<br>Model: {escape(DISPLAY_NAMES.get(result.model_key, result.model_key))}<br>Generated: {generated}</p>
</header>
{verdict_html(result)}
<section><h2 class="cg-h2">Risky clauses</h2>{findings}</section>
<section><h2 class="cg-h2">Full document</h2>{document_html(result)}</section>
<p class="cg-note">ClauseGuard is a student project that uses a machine-learning model trained on the UNFAIR-ToS dataset
plus simple rule checks. It is not legal advice. Always read the full document and ask a lawyer about important contracts.</p>
</main></body></html>"""
