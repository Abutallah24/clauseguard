# ClauseGuard

**Find the risky clauses in terms & conditions and contracts — before you agree to them.**

AI Lab CA mini-project · K. J. Somaiya School of Engineering · Course 216U41C501 Artificial Intelligence

ClauseGuard is an NLP system we trained ourselves on the **UNFAIR-ToS** dataset (LexGLUE benchmark). It reads a
document — PDF, Word file, photos, a camera shot, a web link or pasted text — finds clauses that are potentially
unfair to you, explains each one in plain English, and gives the whole document a **0–100 risk score**.

| | |
|---|---|
| Problem | 91% of people accept terms without reading them (Deloitte, 2017) |
| Dataset | UNFAIR-ToS: 9,414 sentences from 50 real terms of service, 8 unfair-clause types |
| Models (trained by us) | TF-IDF + Logistic Regression · TF-IDF + Linear SVM · fine-tuned DistilBERT |
| App | Streamlit web app, works on laptop and phone |

---

## 1. Setup (one time)

You need Windows with **64-bit Python 3.12**. This project already has a virtual environment in `.venv`.
Open PowerShell in the project folder and run:

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

> Your default `python` command points to a 32-bit preview build that cannot install PyTorch.
> Always use `.venv\Scripts\python` (or activate the venv with `.venv\Scripts\activate`).

## 2. Run the app

```bash
.venv\Scripts\streamlit run app/streamlit_app.py
```

The app opens at **http://localhost:8501**. Try one of the three sample buttons, or upload a file from `samples/`.

### Open it on your phone (for the camera demo)

Phone browsers only allow the camera on secure `https://` pages, so start the app in HTTPS mode for the phone demo.

1. Create a local certificate (one time):
   ```bash
   .venv\Scripts\python scripts/make_cert.py
   ```
2. Start the app with HTTPS:
   ```bash
   .venv\Scripts\streamlit run app/streamlit_app.py --server.sslCertFile certs/cert.pem --server.sslKeyFile certs/key.pem
   ```
3. Connect the phone to the **same Wi-Fi** as the laptop and open the **Network URL** the terminal prints,
   for example `https://192.168.1.9:8501`.
4. The phone warns that the connection is not private (the certificate is self-made). Tap **Advanced → Proceed**.
5. Open the **Camera** tab, turn the camera on, allow camera access and photograph a printed page.

If the page does not load at all, allow Python through Windows Firewall when Windows asks (Private networks).
Plan B: take the photo with the phone's camera app and upload it in the **Images** tab — it runs the same OCR.

## 3. What's in the project

```
app/            Streamlit app (streamlit_app.py, views/, ui/ styles + components)
src/            Core logic: extract.py (PDF/DOCX/OCR/URL) · clauses.py · rules.py · risk.py · models.py · pipeline.py
training/       data.py → eda.py → train_classical.py → train_transformer.py → charts.py
models/         Trained models (tfidf_logreg.joblib, tfidf_svm.joblib, distilbert/)
data/           UNFAIR-ToS splits as CSV
samples/        Demo files for every input type (PDF, scanned PDF, DOCX, page images, camera photo)
reports/        metrics.json, figures/, screenshots/, ClauseGuard_Report.docx
tests/          pytest suite
VIVA_NOTES.md   Plain-English explanations and likely viva questions
```

## 4. Re-train the models (optional)

The trained models are already in `models/`. To reproduce everything from scratch:

```bash
.venv\Scripts\python -m training.data
```
```bash
.venv\Scripts\python -m training.eda
```
```bash
.venv\Scripts\python -m training.train_classical
```
```bash
.venv\Scripts\python -m training.train_transformer --epochs 3
```
```bash
.venv\Scripts\python -m training.charts
```

The classical models train in seconds. DistilBERT takes roughly an hour on a laptop CPU.

## 5. Run the tests

```bash
.venv\Scripts\python -m pytest
```

## 6. Results

RESULTS_TABLE_PLACEHOLDER

All numbers are measured on the official test split and stored in `reports/metrics.json`.

## 7. How the risk score works

Each flagged clause gets `risk = severity of its type × model confidence`. The document score is
`100 × (1 − e^(−Σ risk / 2.5))`: every extra risky clause adds less than the previous one, so it never exceeds 100.
**0–29 Low · 30–60 Medium · 61–100 High.**

## 8. Limitations

- Trained on English online terms of service; less reliable on Indian rental agreements and offer letters
  (5 rule-based checks cover common patterns there, always labelled "Rule check").
- OCR can misread blurry or tilted photos — the extracted text is shown under every report so you can check it.
- Some websites load their terms with JavaScript or block automated reading; use "Paste text" for those.
- It flags *potentially* unfair clauses. It is not legal advice.

## References

1. Lippi, M. et al. (2019). *CLAUDETTE: an automated detector of potentially unfair clauses in online terms of service.* Artificial Intelligence and Law, 27, 117–139.
2. Chalkidis, I. et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.* ACL 2022.
3. Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv:1910.01108.
4. Deloitte (2017). Global Mobile Consumer Survey — 91% of consumers accept terms without reading.
5. Obar, J. & Oeldorf-Hirsch, A. (2020). *The biggest lie on the Internet: ignoring the privacy policies and terms of service policies of social networking services.* Information, Communication & Society.
6. McDonald, A. & Cranor, L. F. (2008). *The Cost of Reading Privacy Policies.* I/S: A Journal of Law and Policy.
