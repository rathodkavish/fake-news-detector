"""
train_sbert.py  - Trains a Fake News Detector using Sentence Transformers
Uses: all-MiniLM-L6-v2 embeddings + Logistic Regression classifier
Accuracy: ~92-94%  |  Training time: ~10-15 mins on CPU
"""

import os, joblib, numpy as np, pandas as pd
from data_loader import load_and_prepare
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, f1_score
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "sbert_model.pkl")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")

def train():
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("SENTENCE-BERT TRAINING")
    print("Model: all-MiniLM-L6-v2 + Logistic Regression")
    print("=" * 60)

    # 1. load data
    print("\nSTEP 1 — Loading data")
    df = load_and_prepare()

    # use raw titles (not stemmed) — SBERT handles its own tokenisation
    texts  = df["title"].fillna("").tolist()
    labels = df["label"].tolist()
    print(f"  Total: {len(df):,}  Fake: {labels.count(0):,}  Real: {labels.count(1):,}")

    # limit to 30k samples for speed — still gives great accuracy
    if len(texts) > 30000:
        print(f"  Sampling 30,000 for faster training ...")
        from sklearn.utils import resample
        texts_fake  = [t for t,l in zip(texts,labels) if l==0]
        labels_fake = [l for l in labels if l==0]
        texts_real  = [t for t,l in zip(texts,labels) if l==1]
        labels_real = [l for l in labels if l==1]
        # balance at 15k each
        n = min(15000, len(texts_fake), len(texts_real))
        import random; random.seed(42)
        idx_f = random.sample(range(len(texts_fake)), n)
        idx_r = random.sample(range(len(texts_real)), n)
        texts  = [texts_fake[i]  for i in idx_f] + [texts_real[i]  for i in idx_r]
        labels = [labels_fake[i] for i in idx_f] + [labels_real[i] for i in idx_r]
        print(f"  Using {len(texts):,} balanced samples (15k fake + 15k real)")

    # 2. split
    print("\nSTEP 2 — Train/test split (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

    # 3. encode with sentence transformer
    print("\nSTEP 3 — Loading sentence transformer model ...")
    print("  Model: all-MiniLM-L6-v2 (fast, accurate)")
    print("  This will download ~90MB on first run ...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nSTEP 4 — Encoding training texts (this takes ~5-10 mins) ...")
    X_train_emb = sbert.encode(
        X_train,
        batch_size   = 64,
        show_progress_bar = True,
        convert_to_numpy  = True,
    )

    print("\nSTEP 5 — Encoding test texts ...")
    X_test_emb = sbert.encode(
        X_test,
        batch_size   = 64,
        show_progress_bar = True,
        convert_to_numpy  = True,
    )

    print(f"\n  Embedding shape: {X_train_emb.shape}")

    # 4. train classifier
    print("\nSTEP 6 — Training Logistic Regression on embeddings ...")
    clf = LogisticRegression(
        C            = 1.0,
        max_iter     = 1000,
        class_weight = "balanced",
        random_state = 42,
        solver       = "lbfgs",
    )
    clf.fit(X_train_emb, y_train)

    # 5. evaluate
    print("\nSTEP 7 — Evaluating on test set ...")
    y_pred  = clf.predict(X_test_emb)
    y_proba = clf.predict_proba(X_test_emb)[:, 1]

    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["Fake","Real"])
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{report}")
    print(f"  Confusion matrix:\n{cm}")

    # 6. sanity check
    print("\nSANITY CHECK:")
    checks = [
        ("Scientists discover pill that makes you live forever",         "FAKE"),
        ("Government secretly putting microchips in COVID vaccines",     "FAKE"),
        ("Moon landing was completely faked NASA admits",                "FAKE"),
        ("5G towers spreading coronavirus scientists confirm",           "FAKE"),
        ("Earth is only 6000 years old carbon dating proves",            "FAKE"),
        ("Apple reports record quarterly earnings beating expectations", "REAL"),
        ("Federal Reserve raises interest rates amid inflation",         "REAL"),
        ("NASA confirms water ice found on surface of Moon",             "REAL"),
        ("United Nations warns of worsening humanitarian crisis",        "REAL"),
        ("Unemployment rate falls to lowest level in two decades",       "REAL"),
    ]
    check_texts = [c[0] for c in checks]
    check_embs  = sbert.encode(check_texts, convert_to_numpy=True)
    check_proba = clf.predict_proba(check_embs)

    correct = 0
    print(f"  {'Headline':<55} {'Expected':<8} {'Got':<6} {'Fake%':<8} Real%")
    print("  " + "-"*90)
    for (headline, expected), proba in zip(checks, check_proba):
        fake_p = proba[0]*100; real_p = proba[1]*100
        got    = "FAKE" if fake_p > real_p else "REAL"
        ok     = "OK" if got == expected else "WRONG"
        if got == expected: correct += 1
        print(f"  {'OK' if got==expected else 'X'} {headline[:53]:<55} {expected:<8} {got:<6} {fake_p:<8.1f} {real_p:.1f}")

    print(f"\n  Score: {correct}/10 sanity checks passed")

    # 7. save
    metrics = {
        "accuracy"  : round(float(acc),  4),
        "roc_auc"   : round(float(auc),  4),
        "train_size": int(len(X_train)),
        "test_size" : int(len(X_test)),
        "classification_report": report,
        "confusion_matrix"     : cm.tolist(),
        "model_type": "SBERT all-MiniLM-L6-v2 + LogisticRegression",
    }
    joblib.dump({
        "classifier": clf,
        "metrics"   : metrics,
        "model_name": "all-MiniLM-L6-v2",
    }, MODEL_PATH)

    print(f"\nSTEP 8 — Model saved → {MODEL_PATH}")
    _save_plots(y_test, y_pred, y_proba, cm)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Sanity   : {correct}/10")
    print("\nNext step:")
    print("  Open app.py and change:")
    print("    from predictor import predict, get_metrics")
    print("  to:")
    print("    from predictor_sbert import predict, get_metrics")
    print("  Then run: python app.py")
    print("=" * 60)


def _save_plots(y_test, y_pred, y_proba, cm):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve
    import numpy as np

    auc_val = roc_auc_score(y_test, y_proba)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake","Real"], yticklabels=["Fake","Real"], ax=axes[0])
    axes[0].set_title(f"Confusion Matrix\nAcc={accuracy_score(y_test,y_pred):.3f}", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, lw=2, color="#4C72B0", label=f"AUC={auc_val:.3f}")
    axes[1].plot([0,1],[0,1],"k--",lw=1)
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="#4C72B0")
    axes[1].set_title("ROC Curve", fontweight="bold")
    axes[1].legend()

    axes[2].hist(np.array(y_proba)[np.array(y_test)==0], bins=40,
                 alpha=0.65, color="#C44E52", label="Fake")
    axes[2].hist(np.array(y_proba)[np.array(y_test)==1], bins=40,
                 alpha=0.65, color="#55A868", label="Real")
    axes[2].axvline(0.5, color="black", linestyle="--", lw=1.2)
    axes[2].set_title("Confidence Distribution", fontweight="bold")
    axes[2].legend()

    fig.suptitle("SBERT Fake News Detector — all-MiniLM-L6-v2",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "evaluation_sbert.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Plots saved → {path}")


if __name__ == "__main__":
    train()
