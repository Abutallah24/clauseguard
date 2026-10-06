# ClauseGuard — project notes for Claude

AI Lab CA mini-project (K. J. Somaiya, course 216U41C501 Artificial Intelligence). This file overrides the
internship-app CLAUDE.md in the parent Downloads folder, which belongs to a different project.

## What it is
NLP model that flags unfair clauses in Terms & Conditions / contracts, plus a Streamlit app that accepts
PDF, DOCX, images, camera photos, URLs or pasted text and shows a 0–100 risk score with highlighted clauses.

## Rules
- The core must be OUR trained model (UNFAIR-ToS / LexGLUE). No LLM or external AI API calls.
- Never invent metrics: every number comes from `reports/metrics.json`, written by `training/`.
- No PowerPoint deliverable.
- Must run on Windows with the project venv: always use `.venv\Scripts\python` (the system `python` is a
  32-bit alpha build that cannot install PyTorch).

## Layout
- `src/` — pure, tested logic: `labels`, `clauses` (splitting), `rules` (keyword checks), `risk` (score),
  `extract` (PDF/DOCX/OCR/URL), `models` (load + predict), `pipeline.analyze()`.
- `training/` — `data` → `eda` → `train_classical` → `train_transformer` → `charts`.
- `app/` — `streamlit_app.py` entry, `views/` pages, `ui/` CSS + HTML components.
- `samples/` — demo documents for every input type (`make_samples.py` regenerates them).
- `reports/` — metrics, figures, screenshots, Word report. `tests/` — pytest suite.

## Commands
- Tests: `.venv\Scripts\python -m pytest`
- App: `.venv\Scripts\streamlit run app/streamlit_app.py`
- Retrain: `python -m training.data`, `training.eda`, `training.train_classical`, `training.train_transformer`, `training.charts`

## Conventions
- All user-facing text escaped with `html.escape` before `unsafe_allow_html` rendering.
- Extraction errors raise `ExtractionError` with a user-facing message.
- Chart colours follow the dataviz reference palette (see `training/charts.py`).
