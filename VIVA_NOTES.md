# ClauseGuard — Viva notes (plain English)

Read this once before the viva. Every answer is short enough to say out loud.

---

## The 30-second pitch

> People accept terms and conditions without reading them. We trained our own NLP model on 9,414 labelled
> sentences from 50 real terms of service to detect 8 kinds of unfair clause. The app takes a PDF, Word file,
> photo, link or pasted text, finds the risky clauses, explains each one in plain English and gives a 0–100 risk score.

---

## Who, what, where, why, when

**Who is it for?** Anyone about to accept terms: app users, students signing rental agreements or internship
offer letters, people signing up for subscriptions or loan apps.

**What does it do?** Reads a document, splits it into clauses, classifies every clause as *fair* or one of 8 unfair
types, adds 5 rule-based checks, and turns everything into a risk score with highlighted clauses.

**Where does the AI live?** In `models/` — three models **we trained ourselves**: TF-IDF + Logistic Regression,
TF-IDF + Linear SVM, and a fine-tuned DistilBERT. No ChatGPT/Gemini/API is used anywhere.

**Why does it matter?** Deloitte (2017): 91% of people accept terms without reading them (97% of 18–34 year olds).
Obar & Oeldorf-Hirsch: 98% of students missed a clause giving away their first-born child.
McDonald & Cranor (2008): reading every privacy policy you meet in a year would take 76 work days.

**When does each step happen?** Training happens once, offline (`training/` scripts). Prediction happens live, in
about a second for a normal document, every time someone uploads something.

---

## How it works, step by step

1. **Get text** — PDF text is read directly (PyMuPDF). Scanned PDFs, photos and camera shots go through **OCR**
   (optical character recognition: turning an image of text into real text) using RapidOCR. Word files are
   parsed with python-docx; links are downloaded and the main text is pulled out with trafilatura.
2. **Split into clauses** — a regex splitter cuts the text at sentence ends and numbered items, and skips headings.
3. **Classify** — the model gives each clause a probability for each of 9 classes (Fair + 8 unfair types).
   If the top class is unfair and the probability is ≥ 0.5, it is flagged.
4. **Rule checks** — 5 keyword rules catch things the dataset doesn't cover (auto-renewal, hidden fees, data
   sharing, bond / lock-in, deposit forfeiture). They are always labelled "Rule check", never "AI model".
5. **Score** — each finding's risk = severity of its type × confidence.
   `score = 100 × (1 − e^(−sum of risks / 2.5))`. 0–29 Low, 30–60 Medium, 61–100 High.

---

## The 8 unfair clause types (from the dataset)

| Type | Meaning in one line |
|---|---|
| Limitation of liability | Company won't pay if it causes you loss |
| Unilateral termination | Company can close your account anytime |
| Unilateral change | Company can change the terms anytime |
| Content removal | Company can delete your content anytime |
| Contract by using | Just using the site = you agreed |
| Choice of law | A foreign / company-chosen law applies |
| Jurisdiction | You must sue in a court the company chooses |
| Arbitration | No court; private arbitration, no class action |

---

## Likely viva questions

**Q: Which algorithm did you use?**
Three, and we compared them: TF-IDF + Logistic Regression (classic ML), TF-IDF + Linear SVM (classic ML) and
DistilBERT (deep learning, a transformer). The app uses the one with the best validation macro-F1 by default.

**Q: What is TF-IDF?**
A way to turn text into numbers. Each word (and 2-word phrase) gets a weight: high if it appears often in this
sentence (term frequency) but rarely across all sentences (inverse document frequency). So "sole discretion" gets a
high weight; "the" gets almost none. We also add character 3–5-grams so OCR typos still match.

**Q: How does Logistic Regression classify?**
It learns one weight per TF-IDF feature for each class, adds them up, and turns the totals into probabilities with
softmax. Words like "arbitration" get a large positive weight for the Arbitration class.

**Q: What does the SVM do differently?**
It finds the boundary that separates classes with the widest margin. SVMs don't output probabilities, so we wrap
it in `CalibratedClassifierCV`, which learns how to convert its scores into probabilities (needed for confidence and
the risk score).

**Q: What is DistilBERT?**
A smaller, faster version of BERT (a transformer language model pre-trained on huge amounts of English text). It
reads the whole sentence with attention, so it understands context ("we may terminate" vs "you may terminate").
We **fine-tuned** it: kept its language knowledge and trained a new 9-class output layer on our dataset for 3
epochs on the CPU.

**Q: What is class imbalance and how did you handle it?**
About 89% of sentences are fair. A lazy model could say "fair" every time and get ~89% accuracy while being useless.
We used `class_weight="balanced"` in the classical models and weighted cross-entropy loss in DistilBERT, so
mistakes on rare unfair classes cost more.

**Q: Why F1 and not accuracy?**
Because of the imbalance. F1 is the balance (harmonic mean) of precision (of what we flagged, how much was really
unfair) and recall (of the unfair clauses, how many we caught). **Macro F1** averages F1 over all classes equally, so
rare classes like Arbitration count as much as Fair.

**Q: What is a confusion matrix?**
A table of actual class (rows) vs predicted class (columns). The diagonal is correct predictions; off-diagonal cells
show which classes get confused (e.g. Jurisdiction vs Choice of law, which often appear in the same sentence).

**Q: How did you split the data?**
We used the official LexGLUE splits: 5,532 train, 2,275 validation, 1,607 test. Validation picked the best model and
epoch; the test set was used only once, for the final numbers.

**Q: Is the risk score learned?**
No — it is a transparent formula on top of the model's output. Severity weights are our judgement of how harmful
each clause type is; the model's confidence scales them. This keeps the score explainable.

**Q: Limitations?**
Trained only on English website terms, so it is weaker on Indian rental agreements and offer letters (that's why
rules exist). OCR can misread blurry photos. It flags *potentially* unfair clauses; it is not legal advice.

**Q: Future scope?**
Train on Indian contracts (rent, employment, loan apps), support Hindi/Marathi, fine-tune a legal-domain model
(LegalBERT), clause rewriting suggestions, a browser extension that scans terms before you click "I agree".

**Q: Is this just ChatGPT?**
No. There are no external AI calls. The models are files in `models/` that we trained with the scripts in
`training/`; the app works offline (except for the link input, which needs the internet to download the page).

---

## Our results

See `reports/metrics.json` and the "Model performance" page in the app. Summary table (test split):

RESULTS_TABLE_PLACEHOLDER
