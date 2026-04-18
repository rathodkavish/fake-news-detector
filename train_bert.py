"""
train_bert.py  - Fine-tunes distilbert-base-uncased for fake news detection
Saves model to models/bert_model/
"""

import os, numpy as np, pandas as pd
from data_loader import load_and_prepare
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
from torch.utils.data import Dataset

BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "models", "bert_model")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
MODEL_NAME = "distilbert-base-uncased"

# ── Dataset wrapper ────────────────────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ── metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = accuracy_score(labels, preds)
    return {"accuracy": acc}


# ── main ───────────────────────────────────────────────────────────────────────
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("BERT TRAINING — distilbert-base-uncased")
    print("=" * 60)

    # 1. load data
    print("\nSTEP 1 — Loading data")
    df = load_and_prepare()
    print(f"  Total: {len(df):,}  Fake: {(df.label==0).sum():,}  Real: {(df.label==1).sum():,}")

    # use raw title (not stemmed) — BERT handles its own tokenisation
    texts  = df["title"].fillna("").tolist()
    labels = df["label"].tolist()

    # 2. split
    print("\nSTEP 2 — Splitting (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

    # 3. tokenise
    print("\nSTEP 3 — Tokenising (this may take a few minutes) ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    test_enc  = tokenizer(X_test,  truncation=True, padding=True, max_length=128)

    train_ds = NewsDataset(train_enc, y_train)
    test_ds  = NewsDataset(test_enc,  y_test)

    # 4. model
    print("\nSTEP 4 — Loading DistilBERT model ...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    # 5. training args
    training_args = TrainingArguments(
        output_dir              = MODEL_DIR,
        num_train_epochs        = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 32,
        warmup_steps            = 200,
        weight_decay            = 0.01,
        logging_dir             = os.path.join(MODEL_DIR, "logs"),
        logging_steps           = 100,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "accuracy",
        report_to               = "none",
        fp16                    = torch.cuda.is_available(),
    )

    # 6. train
    print("\nSTEP 5 — Training (3 epochs) ...")
    print("  This takes 10-30 mins on CPU, 3-5 mins on GPU")
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = test_ds,
        compute_metrics = compute_metrics,
    )
    trainer.train()

    # 7. evaluate
    print("\nSTEP 6 — Evaluating on test set ...")
    preds_output = trainer.predict(test_ds)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Fake","Real"])
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n  Accuracy: {acc:.4f}")
    print(f"\n{report}")
    print(f"  Confusion matrix:\n{cm}")

    # 8. save
    print(f"\nSTEP 7 — Saving model to {MODEL_DIR}")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # 9. save plots
    _save_plots(y_test, y_pred, preds_output.predictions, cm)

    print("\n" + "=" * 60)
    print("BERT Training complete!")
    print(f"Model saved to: {MODEL_DIR}")
    print("Now run: python app_bert.py")
    print("=" * 60)


def _save_plots(y_test, y_pred, logits, cm):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, roc_auc_score
    from scipy.special import softmax

    proba   = softmax(logits, axis=1)[:, 1]
    auc_val = roc_auc_score(y_test, proba)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake","Real"], yticklabels=["Fake","Real"], ax=axes[0])
    axes[0].set_title(f"Confusion Matrix\nAccuracy={accuracy_score(y_test,y_pred):.3f}",
                      fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[1].plot(fpr, tpr, lw=2, color="#4C72B0", label=f"AUC={auc_val:.3f}")
    axes[1].plot([0,1],[0,1],"k--",lw=1)
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="#4C72B0")
    axes[1].set_title("ROC Curve", fontweight="bold")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].legend()

    # confidence distribution
    axes[2].hist(proba[np.array(y_test)==0], bins=40, alpha=0.65,
                 color="#C44E52", label="Fake")
    axes[2].hist(proba[np.array(y_test)==1], bins=40, alpha=0.65,
                 color="#55A868", label="Real")
    axes[2].axvline(0.5, color="black", linestyle="--", lw=1.2)
    axes[2].set_title("Confidence Distribution", fontweight="bold")
    axes[2].set_xlabel("P(Real)"); axes[2].legend()

    fig.suptitle("BERT Fake News Detector — DistilBERT",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "evaluation_bert.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Plots saved → {path}")


if __name__ == "__main__":
    train()
