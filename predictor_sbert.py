"""
predictor_sbert.py  - Inference using SBERT embeddings + Logistic Regression
Drop-in replacement for predictor.py — same predict() and get_metrics() API
"""

import os, joblib
from sentence_transformers import SentenceTransformer

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "sbert_model.pkl")

_bundle = None
_sbert  = None

def _load():
    global _bundle, _sbert
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"SBERT model not found at {MODEL_PATH}\n"
                "Please run: python train_sbert.py"
            )
        print("[predictor_sbert] Loading model ...")
        _bundle = joblib.load(MODEL_PATH)
        _sbert  = SentenceTransformer(_bundle["model_name"])
        print("[predictor_sbert] Ready!")
    return _bundle, _sbert


def predict(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"label":"UNKNOWN","confidence":0.0,
                "fake_prob":0.0,"real_prob":0.0,"clean_text":""}

    bundle, sbert = _load()
    clf = bundle["classifier"]

    # encode single text
    emb   = sbert.encode([text], convert_to_numpy=True)
    proba = clf.predict_proba(emb)[0]

    fake_prob  = float(proba[0])
    real_prob  = float(proba[1])
    label      = "REAL" if real_prob > fake_prob else "FAKE"
    confidence = max(fake_prob, real_prob) * 100

    return {
        "label"      : label,
        "confidence" : round(confidence, 2),
        "fake_prob"  : round(fake_prob, 4),
        "real_prob"  : round(real_prob, 4),
        "clean_text" : text,
        "threshold"  : 0.5,
    }


def get_metrics() -> dict:
    bundle, _ = _load()
    return bundle.get("metrics", {})
