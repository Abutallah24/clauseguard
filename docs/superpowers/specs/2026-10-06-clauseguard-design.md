# ClauseGuard — Design Spec

Date: 2026-10-06 · Course: 216U41C501 Artificial Intelligence (K. J. Somaiya) · Lab CA mini-project

## Goal

Train our own NLP model that detects potentially unfair clauses in Terms & Conditions / contracts,
and wrap it in a Streamlit web app that accepts PDF, DOCX, images, camera photos, URLs or pasted text,
then shows a 0–100 risk score and the risky clauses with plain-English explanations.

Success = (1) models trained by us with real test-set metrics, (2) every input type works end-to-end,
(3) a 2–3 page Word report with real results, (4) README + viva notes.

## Data

- **UNFAIR-ToS** (LexGLUE, `coastalcph/lex_glue`, config `unfair_tos`). Sentences from 50 online ToS,
  labelled with 0+ of 8 unfair-clause types; an empty label list means the sentence is fair.
- Official train / validation / test splits are kept as-is. Test split is used only for final evaluation.

## Task formulation

- **9-class single-label classification**: `Fair` + 8 unfair types.
  Sentences with several labels are rare; for training they are expanded to one row per label,
  for evaluation the gold label is the first label (or `Fair`).
- Additionally reported: the LexGLUE-style **micro/macro F1 over the 8 unfair classes**, and a
  **binary Fair-vs-Unfair** view (the most practical question for a user).

## Models (all trained locally)

| # | Model | Notes |
|---|---|---|
| 1 | TF-IDF (word 1–2 grams) + Logistic Regression | `class_weight="balanced"` |
| 2 | TF-IDF + Linear SVM | `class_weight="balanced"`, probability via calibration |
| 3 | DistilBERT fine-tuned | CPU training, max length 128, class-weighted cross-entropy |

The model with the best validation macro-F1 is the app's default; the user can switch models in the app.

## Risk scoring

- Each clause type has a severity weight in [0, 1] (e.g. arbitration 0.9 … choice of law 0.4).
- Clause risk `r = severity × model confidence` (only if predicted unfair and confidence ≥ threshold).
- Document score `= 100 × (1 − exp(−Σr / 2.5))` — rises fast for the first few risky clauses,
  saturates near 100, never exceeds it. Labels: `< 30` Low, `30–60` Medium, `> 60` High.
- **Supplementary rule detector** (clearly labelled as rule-based, not the ML model) for common
  clauses the dataset does not cover: auto-renewal, hidden fees/penalties, data sharing,
  bond/lock-in, deposit forfeiture. Their weight is added the same way. This also gives a
  rule-vs-ML comparison point for the report.

## Input pipeline (`src/extract.py`)

| Input | Method |
|---|---|
| PDF | PyMuPDF text layer; pages with < 40 chars of text are rendered and OCR'd (scanned PDF) |
| DOCX | python-docx paragraphs + table cells |
| Images (multiple) / camera | RapidOCR (pip-only, ONNX), pages joined in upload order |
| URL | `requests` (http/https only, timeout, 5 MB cap, private/loopback IPs blocked) → trafilatura main-text extraction |
| Paste | as-is |

Limits: 15 MB per upload, 20 images, 200k characters of text. Every failure raises a typed
`ExtractionError` whose message is shown to the user.

Clause splitting (`src/clauses.py`): regex sentence/numbered-item splitter; drops fragments < 25 chars.

## App (`app/`)

Streamlit multipage: **Analyze** (input tabs → results), **Model Performance**, **About**.
Results: risk gauge, category summary, highlighted document (colour by severity), clause cards
(category, confidence, why risky), extracted-text expander, HTML report download.
Built-in samples: ToS, rental agreement, internship offer letter (also as files in `samples/`).
Custom CSS design system; mobile-friendly single column under 768 px.

## Testing

pytest for extraction, clause splitting, risk scoring and rules. Playwright end-to-end for every input type.

## Out of scope

Real accounts, databases, cloud deployment, LLM calls, Hindi/regional languages (future scope).
