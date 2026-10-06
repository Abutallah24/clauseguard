"""Analyze page: choose an input, extract its text, run the model and show the risk report."""

from html import escape

import streamlit as st

from app.ui.components import (
    category_summary_html,
    document_html,
    empty_findings_html,
    findings_html,
    html,
    verdict_html,
)
from app.ui.report_html import build_report
from samples.sample_texts import SAMPLES
from src import extract, models
from src.extract import ExtractionError
from src.pipeline import analyze

html('<h1 class="cg-title">Know what you\'re agreeing to.</h1>')
html(
    '<p class="cg-lede">Upload terms and conditions, a rental agreement or an offer letter. '
    "ClauseGuard reads every clause, highlights the risky ones and explains them in plain English.</p>"
)


def run(source_name: str, get_text) -> None:
    """Extract text with get_text(), analyse it and keep the result in the session."""
    model_key = st.session_state.get("model_key")
    if not model_key:
        st.error("No trained model is available yet. Train the models first (see README).")
        return
    st.session_state.pop("result", None)  # never show an old report under a new error
    try:
        with st.status("Reading the document…", expanded=False) as status:
            text = get_text()
            status.update(label="Checking each clause…")
            result = analyze(text, model_key)
            status.update(label="Done", state="complete")
    except ExtractionError as exc:
        st.error(str(exc))
        return
    except Exception:  # anything unexpected: friendly message instead of a traceback
        st.error("Something went wrong while reading this document. Try another file, or paste the text instead.")
        return
    if not result.clauses:
        st.warning("The text was read, but it has no full sentences to check. Try a longer document.")
        return
    st.session_state["result"] = result
    st.session_state["source_name"] = source_name
    st.session_state["truncated"] = len(text) >= extract.MAX_TEXT_CHARS


# ---------- samples ----------
html('<div class="cg-note" style="margin-bottom:.4rem">No document handy? Try a sample:</div>')
sample_cols = st.columns(len(SAMPLES))
for col, (key, sample) in zip(sample_cols, SAMPLES.items()):
    if col.button(sample["kind"], key=f"sample_{key}", use_container_width=True):
        run(sample["title"], lambda s=sample: extract.from_text(s["text"]))

# ---------- inputs ----------
tab_paste, tab_pdf, tab_word, tab_images, tab_camera, tab_link = st.tabs(
    ["Paste text", "PDF", "Word", "Images", "Camera", "Link"]
)

with tab_paste:
    pasted = st.text_area("Paste the terms or contract text", height=200,
                          placeholder="Paste the full text here…")
    if st.button("Analyze text", type="primary", key="go_paste"):
        run("Pasted text", lambda: extract.from_text(pasted))

with tab_pdf:
    pdf = st.file_uploader("Upload a PDF (scanned PDFs work too)", type=["pdf"], key="pdf")
    if st.button("Analyze PDF", type="primary", key="go_pdf", disabled=pdf is None):
        run(pdf.name, lambda: extract.from_pdf(pdf.getvalue()))

with tab_word:
    docx_file = st.file_uploader("Upload a Word document (.docx)", type=["docx"], key="docx")
    if st.button("Analyze Word file", type="primary", key="go_docx", disabled=docx_file is None):
        run(docx_file.name, lambda: extract.from_docx(docx_file.getvalue()))

with tab_images:
    images = st.file_uploader(
        "Upload photos or screenshots — one per page, in page order",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="images",
    )
    if images:
        st.caption(f"{len(images)} page(s): " + ", ".join(i.name for i in images))
    if st.button("Analyze images", type="primary", key="go_images", disabled=not images):
        run(f"{len(images)} image(s)", lambda: extract.from_images([i.getvalue() for i in images]))

with tab_camera:
    st.caption("Hold the page flat, fill the frame and avoid shadows. On a phone, open the app's Network URL.")
    use_camera = st.toggle("Turn on camera", key="camera_on")
    if use_camera:
        shot = st.camera_input("Take a photo of the page")
        if st.button("Analyze photo", type="primary", key="go_camera", disabled=shot is None):
            run("Camera photo", lambda: extract.from_images([shot.getvalue()]))

with tab_link:
    url = st.text_input("Link to a terms page", placeholder="https://example.com/terms")
    if st.button("Analyze link", type="primary", key="go_link", disabled=not url.strip()):
        run(url.strip(), lambda: extract.from_url(url))

# ---------- results ----------
result = st.session_state.get("result")
if result is None:
    html(
        '<ol class="cg-steps">'
        "<li><b>Add a document</b>Paste text, upload a PDF, Word file or photos, or enter a link.</li>"
        "<li><b>We check every clause</b>A model we trained on 5,500 labelled sentences from real terms of service looks for 8 kinds of unfair clause.</li>"
        "<li><b>Read the report</b>Get a 0–100 risk score, the risky clauses highlighted, and why each one matters.</li>"
        "</ol>"
    )
    st.stop()

model_key = st.session_state.get("model_key")
if model_key and result.model_key != model_key:  # the user switched model: re-check the same text
    with st.spinner(f"Re-checking with {models.DISPLAY_NAMES[model_key]}…"):
        result = analyze(result.text, model_key)
    st.session_state["result"] = result

st.divider()
source = st.session_state.get("source_name", "Document")
html(f'<h2 class="cg-h2" style="margin-top:.5rem">Report for {escape(source)}</h2>')
html(f'<p class="cg-note" style="margin:-.4rem 0 1rem">Checked with {escape(models.DISPLAY_NAMES[result.model_key])}</p>')
if st.session_state.get("truncated"):
    st.warning(f"This document is very long, so only the first {extract.MAX_TEXT_CHARS:,} characters were checked.")

left, right = st.columns([1, 1.9], gap="large")
with left:
    html(verdict_html(result))
    st.write("")
    html(category_summary_html(result))
    st.write("")
    st.download_button(
        "Download report (HTML)",
        data=build_report(result, source),
        file_name="clauseguard_report.html",
        mime="text/html",
        type="primary",
        use_container_width=True,
    )
with right:
    html('<h3 class="cg-h3">The document, with risky clauses marked</h3>')
    html(document_html(result))

html('<h2 class="cg-h2">Why these clauses are risky</h2>')
html(findings_html(result) if result.findings else empty_findings_html())

with st.expander("See the text we extracted"):
    st.text(result.text[:20000] + ("\n…" if len(result.text) > 20000 else ""))

html(
    '<p class="cg-note" style="margin-top:1.5rem">ClauseGuard is a student project, not legal advice. '
    "Findings tagged “Rule check” come from simple keyword rules; “AI model” findings come from the trained model.</p>"
)
