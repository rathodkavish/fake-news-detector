"""
predictor_bert.py  - Inference using fine-tuned DistilBERT
"""

import os, torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models", "bert_model")

_model     = None
_tokenizer = None

def _load():
    global _model, _tokenizer
    if _model is None:
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(
                f"BERT model not found at {MODEL_DIR}\n"
                "Please run: python train_bert.py"
            )
        print("[predictor_bert] Loading BERT model ...")
        _tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        _model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.eval()
        print("[predictor_bert] Model loaded.")
    return _model, _tokenizer


def predict(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"label":"UNKNOWN","confidence":0.0,
                "fake_prob":0.0,"real_prob":0.0,"clean_text":""}

    model, tokenizer = _load()

    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding=True, max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=-1)[0]

    fake_prob = float(probs[0])
    real_prob = float(probs[1])
    label     = "REAL" if real_prob > fake_prob else "FAKE"
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
    return {"model": "DistilBERT fine-tuned", "note": "See plots/evaluation_bert.png"}
