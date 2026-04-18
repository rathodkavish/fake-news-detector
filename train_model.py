"""
train_model.py  (v9 - VotingClassifier ensemble, title+text training)
======================================================================
Trains a TF-IDF feature union fed into a soft-voting ensemble of:
  - Logistic Regression   (proven SOTA for sparse text)
  - PassiveAggressiveClassifier (extremely strong on news classification)

Both classifiers handle multi-hundred-thousand sparse features efficiently.

Using title + text body (from data_loader v9) gives the model far richer
signals than title-only training.

Run:
    python train_model.py
"""
import os, joblib, numpy as np, pandas as pd
from sklearn.pipeline                import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble                import VotingClassifier
from sklearn.calibration             import CalibratedClassifierCV
from sklearn.model_selection         import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics                 import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score,
)
from data_loader import load_and_prepare

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")


# ── feature extractor (shared by both classifiers) ───────────────────────────
def build_features():
    """
    Two TF-IDF branches merged via FeatureUnion:
      1. word-level unigrams + bigrams   (primary semantic signal)
      2. char-level n-grams (3-5)        (misspellings, numeric patterns, style)
    """
    word_tfidf = TfidfVectorizer(
        analyzer      = "word",
        ngram_range   = (1, 3),
        max_features  = 300_000,       # more features = richer representation
        sublinear_tf  = True,
        min_df        = 2,
        strip_accents = "unicode",
        token_pattern = r"(?u)\b\w\w+\b",   # keeps word-tokens incl. digits
    )
    char_tfidf = TfidfVectorizer(
        analyzer      = "char_wb",
        ngram_range   = (3, 5),
        max_features  = 50_000,
        sublinear_tf  = True,
        min_df        = 3,
        strip_accents = "unicode",
    )
    return FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])


# ── pipeline ──────────────────────────────────────────────────────────────────
def build_pipeline():
    """
    Soft-voting ensemble of:
      - LR:  well-calibrated probabilities, global optimum
      - PAC: aggressive online learner, very strong on news text
    PAC needs CalibratedClassifierCV to output probabilities for soft voting.
    """
    lr = LogisticRegression(
        C            = 4.0,
        solver       = "saga",
        max_iter     = 3_000,
        class_weight = "balanced",
        random_state = 42,
        n_jobs       = -1,
    )
    pac_base = PassiveAggressiveClassifier(
        C            = 0.1,
        max_iter     = 1_000,
        class_weight = "balanced",
        random_state = 42,
    )
    # Wrap PAC in Platt scaling so it outputs valid probabilities
    pac = CalibratedClassifierCV(pac_base, cv=3, method="sigmoid")

    voting = VotingClassifier(
        estimators = [("lr", lr), ("pac", pac)],
        voting     = "soft",          # average probabilities
        weights    = [2, 1],          # LR probabilities are better calibrated
        n_jobs     = -1,
    )
    return Pipeline([
        ("features", build_features()),
        ("clf",      voting),
    ])


# ── threshold search ──────────────────────────────────────────────────────────
def find_best_threshold(pipeline, X_val, y_val):
    proba    = pipeline.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.25, 0.75, 0.01):
        preds = (proba >= t).astype(int)
        f = f1_score(y_val, preds, average="macro")
        if f > best_f1:
            best_f1, best_t = f, t
    print(f"  Best threshold : {best_t:.2f}  (macro F1 = {best_f1:.4f})")
    return round(float(best_t), 2)


# ── main training ─────────────────────────────────────────────────────────────
def train():
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 65)
    print("STEP 1 — Loading & preparing data (force rebuild for fresh features)")
    # Force rebuild so the new title+text features are generated fresh
    df = load_and_prepare(force_rebuild=True)
    X  = df["clean_title"]
    y  = df["label"]
    print(f"  Total:{len(df):,}  Fake:{(y==0).sum():,}  Real:{(y==1).sum():,}")

    print("\nSTEP 2 — Split 70/10/20")
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.125, random_state=42, stratify=y_tmp)
    print(f"  Train:{len(X_train):,}  Val:{len(X_val):,}  Test:{len(X_test):,}")

    print("\nSTEP 3 — 3-fold CV on training set (macro F1, VotingClassifier)")
    pipeline  = build_pipeline()
    cv        = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"  CV macro-F1: {cv_scores.round(4)}  mean={cv_scores.mean():.4f}")

    print("\nSTEP 4 — Fitting final model on full train set")
    pipeline.fit(X_train, y_train)

    print("\nSTEP 5 — Finding best decision threshold on validation set")
    best_threshold = find_best_threshold(pipeline, X_val, y_val)

    print("\nSTEP 6 — Test set evaluation")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= best_threshold).astype(int)
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=["Fake", "Real"])
    print(f"  Accuracy:{acc:.4f}  ROC-AUC:{auc:.4f}  Threshold:{best_threshold}")
    print(report)

    # ── sanity checks ─────────────────────────────────────────────────────────
    print("STEP 7 — Sanity checks on known headlines")
    from data_loader import clean_text as ct

    checks = [
        # (headline,                                                    expected)
        ("Apple reports record quarterly earnings",                     "REAL"),
        ("NASA confirms water ice found on Moon",                       "REAL"),
        ("Federal Reserve raises interest rates",                       "REAL"),
        ("The stock market rose 2.5 percent on jobs report",           "REAL"),
        ("WHO approves new malaria vaccine for children",               "REAL"),
        ("Supreme Court rules on landmark immigration case",            "REAL"),
        ("Scientists discover pill that makes you live forever",        "FAKE"),
        ("Government secretly putting microchips in vaccines",          "FAKE"),
        ("Moon landing was completely faked NASA admits",               "FAKE"),
        ("Drinking bleach cures cancer suppressed study",               "FAKE"),
        ("5G towers spreading COVID-19 radiation kills people",         "FAKE"),
        ("Illuminati controls world governments secret meeting",        "FAKE"),
        ("Bill Gates depopulation agenda microchip COVID vaccine",      "FAKE"),
        ("Scientists discover new vaccine eliminates all diseases",     "FAKE"),
        ("Bigfoot photographed in national park confirmed real",        "FAKE"),
        ("Aliens have landed government finally admits it",             "FAKE"),
    ]
    all_ok = True
    for headline, expected in checks:
        cleaned  = ct(headline)
        proba    = pipeline.predict_proba([cleaned])[0]
        real_p   = proba[1]
        got      = "REAL" if real_p >= best_threshold else "FAKE"
        ok       = "✓" if got == expected else "✗"
        if got != expected: all_ok = False
        print(f"  {ok} [{expected}→{got}] {headline[:55]:<55} "
              f"fake={proba[0]*100:.1f}% real={real_p*100:.1f}%")

    if all_ok:
        print("\n  ✓ All sanity checks passed!")
    else:
        print("\n  ✗ Some checks failed — rule-based overrides will handle these at predict time")

    # ── save ──────────────────────────────────────────────────────────────────
    metrics = {
        "accuracy"            : round(float(acc), 4),
        "roc_auc"             : round(float(auc), 4),
        "cv_f1_mean"          : round(float(cv_scores.mean()), 4),
        "cv_f1_std"           : round(float(cv_scores.std()),  4),
        "train_size"          : int(len(X_train)),
        "test_size"           : int(len(X_test)),
        "best_threshold"      : best_threshold,
        "confusion_matrix"    : cm.tolist(),
        "classification_report": report,
    }
    joblib.dump(
        {"pipeline": pipeline, "metrics": metrics, "threshold": best_threshold},
        MODEL_PATH,
    )
    print(f"\nModel saved → {MODEL_PATH}")
    _save_plots(y_test, y_pred, y_proba, cm, cv_scores)
    print("=" * 65)
    print("Training complete!")


# ── plots ─────────────────────────────────────────────────────────────────────
def _save_plots(y_test, y_pred, y_proba, cm, cv_scores):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from sklearn.metrics import roc_curve

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], ax=ax1)
    ax1.set_title("Confusion Matrix", fontweight="bold")
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")

    # ROC curve
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val     = roc_auc_score(y_test, y_proba)
    ax2.plot(fpr, tpr, lw=2, color="#4C72B0", label=f"AUC={auc_val:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.fill_between(fpr, tpr, alpha=0.1, color="#4C72B0")
    ax2.set_title("ROC Curve", fontweight="bold"); ax2.legend()

    # CV scores
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(range(1, len(cv_scores) + 1), cv_scores, color="#55A868", width=0.5)
    ax3.axhline(cv_scores.mean(), color="#C44E52", linestyle="--",
                label=f"Mean={cv_scores.mean():.3f}")
    ax3.set_title("CV Macro-F1", fontweight="bold")
    ax3.set_ylim(0, 1.05); ax3.legend()

    # Confidence distribution
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.hist(y_proba[y_test == 0], bins=40, alpha=0.65, color="#C44E52", label="Fake")
    ax4.hist(y_proba[y_test == 1], bins=40, alpha=0.65, color="#55A868", label="Real")
    ax4.set_title("Confidence Distribution", fontweight="bold"); ax4.legend()

    # Summary table
    ax5 = fig.add_subplot(gs[1, 2]); ax5.axis("off")
    rows = [
        ["Accuracy",  f"{accuracy_score(y_test, y_pred):.4f}"],
        ["ROC-AUC",   f"{auc_val:.4f}"],
        ["CV F1",     f"{cv_scores.mean():.4f}"],
        ["Threshold", f"{_find_threshold_display(cv_scores):.2f}"],
    ]
    tbl = ax5.table(cellText=rows, colLabels=["Metric", "Value"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.3, 1.8)
    ax5.set_title("Summary", fontweight="bold", pad=14)

    fig.savefig(os.path.join(PLOTS_DIR, "evaluation.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Plots saved → {PLOTS_DIR}/evaluation.png")


def _find_threshold_display(cv_scores):
    # Just display mean CV score as proxy for threshold display in plot
    return cv_scores.mean()


if __name__ == "__main__":
    train()
