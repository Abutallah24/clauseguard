# ClauseGuard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train unfair-clause classifiers on UNFAIR-ToS and ship a Streamlit app that scores T&C/contract risk from PDF, DOCX, images, camera, URL or text.

**Architecture:** `src/` holds pure, tested modules (labels, extraction, clause splitting, rules, risk, model loading/prediction). `training/` scripts download data, run EDA, train 3 models and write `reports/metrics.json` + PNG charts. `app/` is a Streamlit multipage UI that only calls `src.pipeline.analyze()`.

**Tech Stack:** Python 3.12 (`.venv`), scikit-learn, PyTorch CPU + transformers (DistilBERT), datasets, PyMuPDF, python-docx, RapidOCR, requests + trafilatura, Streamlit + Plotly, matplotlib, pytest, Playwright MCP.

**Spec:** `docs/superpowers/specs/2026-10-06-clauseguard-design.md`

## Global Constraints

- Only real metrics, read from `reports/metrics.json`; never hard-coded in app or report.
- No LLM/API calls; the app works offline except for URL input.
- Must run on Windows with `.venv\Scripts\python`.
- Upload limit 15 MB per file, max 20 images, text capped at 200,000 chars.
- Risk score: `100 × (1 − exp(−Σr / 2.5))`; Low < 30 ≤ Medium ≤ 60 < High.
- No PowerPoint.

## Review Focus

- Scanned PDF with no text layer → OCR fallback must produce text (test: `test_scanned_pdf_uses_ocr`).
- Empty / whitespace input or a blank image → friendly `ExtractionError`, not a crash (`test_empty_text_raises`).
- URL to localhost / private IP or non-http scheme → rejected (`test_url_blocks_private_hosts`).
- Document with zero risky clauses → score 0, label Low, UI shows "no risky clauses" state (`test_score_zero_when_no_findings`).
- Very long document (hundreds of clauses) → truncated at cap, still scored (`test_text_cap`).

---

### Task 1: Labels, clause splitting, rules, risk scoring
**Files:** `src/labels.py`, `src/clauses.py`, `src/rules.py`, `src/risk.py`, `tests/test_clauses.py`, `tests/test_risk.py`, `tests/test_rules.py`
**Produces:**
- `labels.LABELS: list[str]` (index 0 = "Fair", 1..8 = LexGLUE order: Limitation of liability, Unilateral termination, Unilateral change, Content removal, Contract by using, Choice of law, Jurisdiction, Arbitration); `labels.CATEGORY_INFO: dict[str, dict]` with `severity`, `short`, `why`.
- `clauses.split_clauses(text: str) -> list[str]`
- `rules.detect_rules(clause: str) -> list[RuleHit]` (`RuleHit(category, severity, why)`)
- `risk.Finding` dataclass (`clause, category, confidence, severity, source: "model"|"rules", why`); `risk.document_score(findings) -> tuple[int, str]`
- [ ] Tests: split keeps numbered items separate, drops < 25 chars; `document_score([]) == (0, "Low")`; score monotonic and < 100; auto-renewal / bond / data-sharing phrases detected; benign sentence → no hits.
- [ ] Implement, run `pytest -q`, commit.

### Task 2: Text extraction
**Files:** `src/extract.py`, `tests/test_extract.py`
**Produces:** `ExtractionError`; `from_pdf(bytes) -> str`, `from_docx(bytes) -> str`, `from_images(list[bytes]) -> str`, `from_url(str) -> str`, `from_text(str) -> str`.
- [ ] Tests: text PDF, scanned PDF (OCR), DOCX with table, image OCR, private URL blocked, empty text raises, text cap.
- [ ] Implement (PyMuPDF; page OCR when < 40 chars; RapidOCR singleton; URL: scheme check, DNS → reject private/loopback/link-local, timeout 10 s, 5 MB cap, trafilatura).
- [ ] Run tests, commit.

### Task 3: Data + EDA
**Files:** `training/data.py`, `training/eda.py`
**Produces:** `data/unfair_tos_{train,validation,test}.csv` (columns `text`, `labels` JSON list, `label` primary int); `reports/figures/class_distribution.png`, `reports/eda.json`.
- [ ] Download via `datasets` (fallback: parquet from HF hub), save CSVs, run EDA, commit.

### Task 4: Classical models
**Files:** `training/train_classical.py`, `training/evaluate.py`
**Produces:** `models/tfidf_logreg.joblib`, `models/tfidf_svm.joblib`; `evaluate.evaluate(y_true, y_pred, name) -> dict` (accuracy, macro/micro/weighted P/R/F1 over 9 classes, unfair-only micro/macro F1, binary fair-vs-unfair P/R/F1, per-class report, confusion matrix) merged into `reports/metrics.json`.
- [ ] Train, evaluate on test, save confusion-matrix PNGs, commit.

### Task 5: DistilBERT
**Files:** `training/train_transformer.py`
**Produces:** `models/distilbert/` (HF format) and its metrics in `reports/metrics.json`.
- [ ] Train 3 epochs, max_len 128, batch 16, lr 5e-5, class-weighted CE, best epoch by val macro-F1; evaluate on test; commit.

### Task 6: Prediction + pipeline + charts
**Files:** `src/models.py`, `src/pipeline.py`, `training/charts.py`, `tests/test_pipeline.py`
**Produces:** `models.available_models() -> list[str]`, `models.load(name)`, `predict(model, clauses) -> list[tuple[int, float]]`; `pipeline.analyze(text: str, model_name: str) -> AnalysisResult` (`score, label, findings, clauses, counts, text`); comparison/F1 charts.
- [ ] Test analyze() on a known-unfair text yields ≥ 1 model finding; commit.

### Task 7: Samples
**Files:** `samples/make_samples.py` → `samples/*.txt|pdf|docx|png`, `samples/scanned_rental_agreement.pdf`.

### Task 8: Streamlit app
**Files:** `app/Home.py` (entry), `app/pages/1_Model_Performance.py`, `app/pages/2_About.py`, `app/ui/style.css`, `app/ui/components.py`, `app/ui/report_html.py`, `.streamlit/config.toml`
- [ ] Input tabs, samples, results (gauge, summary, highlighted doc, cards, download), performance page; mobile CSS.

### Task 9: E2E tests, reviews, screenshots
- [ ] Playwright: each input type; fix bugs; screenshots to `reports/screenshots/`.
- [ ] security-review, code-review, simplify, web-design-guidelines, impeccable pass.

### Task 10: Report, README, viva notes, CLAUDE.md
**Files:** `reports/ClauseGuard_Report.docx`, `README.md`, `VIVA_NOTES.md`, `CLAUDE.md`, `requirements.txt`.
