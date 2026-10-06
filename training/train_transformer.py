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
from torch.utils.data import DataLoader, TensorDataset
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


def encode(tokenizer, texts, labels=None):
    enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    tensors = [enc["input_ids"], enc["attention_mask"]]
    if labels is not None:
        tensors.append(torch.tensor(list(labels), dtype=torch.long))
    return TensorDataset(*tensors)


@torch.no_grad()
def predict(model, dataset) -> np.ndarray:
    model.eval()
    preds = []
    for input_ids, mask, *_ in DataLoader(dataset, batch_size=64):
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

    train_ds = encode(tokenizer, x_train, y_train)
    val_ds, test_ds = encode(tokenizer, val["text"]), encode(tokenizer, test["text"])
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(SEED))

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
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {running / step:.4f} "
                      f"elapsed {(time.time() - start) / 60:.1f} min", flush=True)

        val_f1 = f1_score(val["label"], predict(model, val_ds), average="macro", zero_division=0)
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
        predict(best, test_ds),
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
