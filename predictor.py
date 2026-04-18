"""
predictor.py  (v4 - rule-based fast-path + aligned text cleaning)
------------------------------------------------------------------
Loads the saved pipeline + threshold and exposes predict().
"""

import os, re, joblib, nltk
from nltk.corpus import stopwords
from nltk.stem   import PorterStemmer

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

_model_bundle = None
_stemmer      = None
_stopwords    = None


def _ensure_nltk():
    for res in ["stopwords", "punkt"]:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass


def _get_tools():
    global _stemmer, _stopwords
    if _stemmer is None:
        _ensure_nltk()
        _stemmer   = PorterStemmer()
        _stopwords = set(stopwords.words("english"))
    return _stemmer, _stopwords


def _load_model():
    global _model_bundle
    if _model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}.\n"
                "Please run  python train_model.py  first."
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def _clean(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    stemmer, stop = _get_tools()
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    for w in text.split():
        if len(w) > 1:
            if w.isdigit():
                tokens.append(w)
            elif w not in stop:
                tokens.append(stemmer.stem(w))
    return " ".join(tokens)


FAKE_RULES = [
    r"\blive\s+forever\b",
    r"\bimmort(al|ality)\b",
    r"\bcure[sd]?\s+(all|every|cancer|aids|hiv|diabetes|covid)\b",
    r"\belimin(ate|ates|ating)\s+all\s+(disease|cancer|virus)\b",
    r"\bpill\s+that\s+(makes?|will)\s+you\s+live\b",
    r"\bcure\s+cancer\b",
    r"\bcure[sd]?\s+every\b",
    r"\bdiscover(ed|s)?\s+secret\s+cure\b",
    r"\bdoctors\s+don.?t\s+want\s+you\s+to\s+know\b",
    r"\bsuppressed\s+(by|cure|study|treatment)\b",
    r"\bdrink(ing)?\s+bleach\b",
    r"\bmicrochip[s]?\s+in\s+(vaccine|covid|shot)\b",
    r"\bchip[s]?\s+in\s+(vaccine|blood|body)\b",
    r"\bmind.?control\b",
    r"\bdeep\s+state\b",
    r"\bnew\s+world\s+order\b",
    r"\billuminati\b",
    r"\bsatani[cs]\b",
    r"\bchemtrail[s]?\b",
    r"\bflat\s+earth\b",
    r"\brepti(lian|le)\b",
    r"\bqanon\b",
    r"\badrenochr[oa]me\b",
    r"\bpizzagate\b",
    r"\bhollo?w\s+earth\b",
    r"\bfema\s+camp[s]?\b",
    r"\b5g\s+(kill[s]?|cause[s]?|spread[s]?|radiation)\b",
    r"\bwifi\s+kill[s]?\b",
    r"\bvaccine\s+kill[s]?\b",
    r"\bbill\s+gates\s+(control|chip|kill|depop)\b",
    r"\bgeorge\s+soros\b",
    r"\bmoon\s+landing\s+.*(fake|faked|hoax|never\s+happen|admit)\b",
    r"\bnasa\s+(admit[s]?|confess)\s+.*(fake|faked|hoax|lie)\b",
    r"\bnasa\s+(finally|secret|confess)\b",
    r"\bbigfoot\b",
    r"\bufo\s+(land[s]?|seen|crash|proof|confirm)\b",
    r"\balien\s+(invasion|contact|confirm|proof|admit)\b",
    r"\bghost\s+(photo|proof|confirm|real)\b",
    r"\b(government|they)\s+(don.?t\s+want|hid|hiding|won.?t\s+tell)\b",
    r"\bwhat\s+they\s+(don.?t|aren.?t)\s+tell\b",
    r"\bwake\s+up\s+sheeple\b",
    r"\bcrisis\s+actor[s]?\b",
    r"\bfalse\s+flag\b",
    r"\bdepopulat(ion|e)\b",
]

REAL_RULES = [
    r"\b(federal reserve|central bank|imf|world bank)\s+(raise[sd]?|cut[s]?|hike[sd]?|lower[sd]?)\s+(rate[s]?|interest)\b",
    r"\b(gdp|inflation|unemployment)\s+(rose?|fell?|grew?|shrank?|jump[sed]*|drop[ped]*)\b",
    r"\b(supreme court|congress|senate|parliament)\s+(rule[sd]?|pass[ed]?|approv[ed]*|vote[sd]?)\b",
    r"\bnasa\s+confirm[s]?\s+.*(water|ice|oxygen|mission|launch)\b",
    r"\b(who|cdc|nih|fda)\s+(approv[ed]*|authoriz[ed]*|recommend[s]?|warn[s]?)\b",
]

_FAKE_RE = [re.compile(p, re.IGNORECASE) for p in FAKE_RULES]
_REAL_RE = [re.compile(p, re.IGNORECASE) for p in REAL_RULES]

_MATH_RE = re.compile(
    r"^[\d\s\+\-\*/\^=\(\)\.%,]+$"
    r"|^\d+\s*[\+\-\*/]\s*\d+\s*=\s*\d+$",
    re.IGNORECASE,
)
_SHORT_THRESHOLD = 4


def _rule_check(raw_text: str):
    stripped = raw_text.strip()
    if _MATH_RE.match(stripped):
        return "NOT NEWS", "This looks like a mathematical expression, not a news headline.", "math"

    for pat in _FAKE_RE:
        if pat.search(stripped):
            return (
                "FAKE",
                "⚠️ This headline contains language strongly associated with "
                "<b>misinformation, impossible claims, or conspiracy theories</b>.",
                "conspiracy_fake",
            )

    for pat in _REAL_RE:
        if pat.search(stripped):
            return (
                "REAL",
                "This headline matches patterns of <b>credible institutional reporting</b>.",
                "credible_real",
            )

    return None, None, None


def predict(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {
            "label": "UNKNOWN", "confidence": 0.0,
            "fake_prob": 0.0, "real_prob": 0.0,
            "clean_text": "", "threshold": 0.5,
            "input_type": "empty", "message": "Please enter a headline to analyse.",
        }

    rule_label, rule_msg, input_type = _rule_check(text)

    if rule_label == "NOT NEWS":
        return {
            "label": "NOT NEWS", "confidence": 100.0,
            "fake_prob": 0.0, "real_prob": 0.0,
            "clean_text": text, "threshold": 0.5,
            "input_type": input_type, "message": rule_msg,
        }

    bundle    = _load_model()
    pipeline  = bundle["pipeline"]
    threshold = 0.5

    cleaned = _clean(text)
    if not cleaned or len(cleaned.split()) < _SHORT_THRESHOLD:
        return {
            "label": "UNKNOWN", "confidence": 0.0,
            "fake_prob": 0.0, "real_prob": 0.0,
            "clean_text": cleaned, "threshold": threshold,
            "input_type": "too_short",
            "message": "Headline is too short to classify reliably. Please add more context.",
        }

    proba     = pipeline.predict_proba([cleaned])[0]
    fake_prob = float(proba[0])
    real_prob = float(proba[1])
    ml_label  = "REAL" if real_prob >= threshold else "FAKE"

    # ── rule override ──────────────────────────────────────────────────────────
    if rule_label == "FAKE":
        # Rule always wins for FAKE — force high fake probability
        label     = "FAKE"
        fake_prob = 0.87
        real_prob = 0.13
        msg       = rule_msg
        itype     = input_type or "conspiracy_fake"
    elif rule_label == "REAL" and ml_label == "FAKE":
        label     = "REAL"
        fake_prob = 0.13
        real_prob = 0.87
        msg       = rule_msg
        itype     = input_type or "credible_real"
    else:
        label  = ml_label
        msg    = ""
        itype  = "normal"

    confidence = (real_prob if label == "REAL" else fake_prob) * 100

    return {
        "label"      : label,
        "confidence" : round(confidence, 2),
        "fake_prob"  : round(fake_prob, 4),
        "real_prob"  : round(real_prob, 4),
        "clean_text" : cleaned,
        "threshold"  : threshold,
        "input_type" : itype,
        "message"    : msg,
    }


def get_metrics() -> dict:
    bundle = _load_model()
    return bundle.get("metrics", {})