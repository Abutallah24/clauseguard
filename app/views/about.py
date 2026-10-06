"""How it works: the problem, the pipeline, the risk score and the limitations."""

from app.ui.components import html
from src.labels import CATEGORY_INFO, RULE_CATEGORY_INFO
from src.risk import SCORE_SCALE

html('<h1 class="cg-title">How ClauseGuard works</h1>')
html(
    '<p class="cg-lede">Most people accept terms and contracts without reading them, because they are long and written '
    "in legal language. ClauseGuard reads them for you and points at the clauses that can hurt you.</p>"
)

html(
    '<ol class="cg-steps">'
    "<li><b>Extract the text</b>PDFs are read directly; scanned pages, photos and camera shots go through OCR; "
    "Word files and web pages are parsed.</li>"
    "<li><b>Split into clauses</b>The text is cut into sentences and numbered clauses.</li>"
    "<li><b>Classify each clause</b>Our trained model labels each clause as fair or one of 8 unfair types. "
    "Simple rules add 5 extra checks for contracts students often sign.</li>"
    "</ol>"
)

html('<h2 class="cg-h2">The risk score</h2>')
html(
    '<div class="cg-panel"><p style="margin:0 0 .6rem">Each flagged clause gets a risk value = '
    "<b>severity of its type</b> × <b>model confidence</b>. The document score is</p>"
    f'<p style="font-size:1.15rem;margin:0 0 .6rem"><b>score = 100 × (1 − e<sup>−(sum of risks) / {SCORE_SCALE}</sup>)</b></p>'
    '<p class="cg-note" style="margin:0">Every extra risky clause raises the score, but by less each time, so it never '
    "goes past 100. 0–29 is Low, 30–60 Medium and 61–100 High.</p></div>"
)

html('<h2 class="cg-h2">What it looks for</h2>')
rows = "".join(
    f'<tr><td>{name}</td><td>{info["severity"]:.2f}</td><td style="text-align:left">{info["short"]}</td><td>AI model</td></tr>'
    for name, info in CATEGORY_INFO.items()
) + "".join(
    f'<tr><td>{name}</td><td>{info["severity"]:.2f}</td><td style="text-align:left">{info["short"]}</td><td>Rule check</td></tr>'
    for name, info in RULE_CATEGORY_INFO.items()
)
html(
    '<div class="cg-table-wrap"><table class="cg-table"><thead><tr><th>Clause type</th><th>Severity</th>'
    f'<th style="text-align:left">In short</th><th>Found by</th></tr></thead><tbody>{rows}</tbody></table></div>'
)

html('<h2 class="cg-h2">Limitations</h2>')
html(
    '<div class="cg-panel"><ul style="margin:0;padding-left:1.1rem;line-height:1.7">'
    "<li>The model learned from English terms of service of online platforms. It is less reliable on Indian rental "
    "agreements, offer letters or other legal styles, which is why the rule checks exist.</li>"
    "<li>OCR can misread blurry or tilted photos; check the extracted text shown under each report.</li>"
    "<li>It flags clauses that are <i>potentially</i> unfair. It is not legal advice.</li>"
    "<li>English only for now.</li></ul></div>"
)
