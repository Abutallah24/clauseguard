"""Fine-tune DistilBERT for 9-class unfair-clause classification (runs on CPU).

Run:  .venv\\Scripts\\python -m training.train_transformer [--epochs 3]
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.labels import LABELS
from training.data import load_split
from training.evaluate import evaluate, print_summary, save_metrics
from training.train_classical import expand_multilabel

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models" / "distilbert"
BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 128
BATCH = 16
LR = 5e-5
SEED = 42


class ClauseDataset(Dataset):
    """Tokenised sentences (no padding yet); padding happens per batch in `collate`."""

    def __init__(self, tokenizer, texts, labels=None):
        self.ids = tokenizer(list(texts), truncation=True, max_length=MAX_LEN)["input_ids"]
        self.labels = list(labels) if labels is not None else [0] * len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def make_collate(pad_id: int):
    def collate(batch):
        # Dynamic padding: pad only to the longest sentence in this batch (avg ~45 tokens, not 128).
        width = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, (ids, _) in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            mask[row, : len(ids)] = 1
        return input_ids, mask, torch.tensor([label for _, label in batch], dtype=torch.long)

    return collate


@torch.no_grad()
def predict(model, dataset, collate) -> np.ndarray:
    model.eval()
    preds = []
    for input_ids, mask, _ in DataLoader(dataset, batch_size=64, collate_fn=collate):
        preds.append(model(input_ids=input_ids, attention_mask=mask).logits.argmax(-1).numpy())
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    train, val, test = load_split("train"), load_split("validation"), load_split("test")
    x_train, y_train = expand_multilabel(train)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)},
    )

    collate = make_collate(tokenizer.pad_token_id)
    train_ds = ClauseDataset(tokenizer, x_train, y_train)
    val_ds, test_ds = ClauseDataset(tokenizer, val["text"]), ClauseDataset(tokenizer, test["text"])
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(SEED))

    # Class weights counter the imbalance (about 90% of sentences are Fair).
    weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=np.array(y_train))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(np.sqrt(weights), dtype=torch.float))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_f1, start = -1.0, time.time()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (input_ids, mask, labels) in enumerate(loader, start=1):
            logits = model(input_ids=input_ids, attention_mask=mask).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {running / step:.4f} "
                      f"elapsed {(time.time() - start) / 60:.1f} min", flush=True)

        val_f1 = f1_score(val["label"], predict(model, val_ds, collate), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": running / len(loader), "validation_macro_f1": val_f1})
        print(f"epoch {epoch} done: train loss {running / len(loader):.4f}, validation macro-F1 {val_f1:.3f}", flush=True)
        if val_f1 > best_f1:  # keep the epoch with the best validation score
            best_f1 = val_f1
            if OUT_DIR.exists():
                shutil.rmtree(OUT_DIR)
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)

    train_seconds = time.time() - start
    best = AutoModelForSequenceClassification.from_pretrained(OUT_DIR)
    result = evaluate(
        test["label"],
        predict(best, test_ds, collate),
        "DistilBERT (fine-tuned)",
        extra={
            "validation_macro_f1": best_f1,
            "train_seconds": round(train_seconds, 1),
            "family": "deep learning",
            "history": history,
            "hyperparameters": {"base_model": BASE_MODEL, "max_len": MAX_LEN, "batch_size": BATCH,
                                "learning_rate": LR, "epochs": args.epochs},
        },
    )
    save_metrics("distilbert", result)
    print_summary(result)
    print(f"total training time {train_seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
