import joblib
import numpy as np
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Load model
b = joblib.load('models/model.pkl')
p = b['pipeline']
tfidf = p.named_steps['tfidf']
clf   = p.named_steps['clf']

# Check top words
feat_names = tfidf.get_feature_names_out()
coefs      = clf.coef_[0]
top_fake   = [feat_names[i] for i in np.argsort(coefs)[:20]]
top_real   = [feat_names[i] for i in np.argsort(coefs)[-20:]]

print("Top 20 words that push toward FAKE:")
print(top_fake)
print()
print("Top 20 words that push toward REAL:")
print(top_real)
print()

# Test specific headlines
stemmer = PorterStemmer()
stop    = set(stopwords.words("english"))

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(stemmer.stem(w) for w in text.split()
                    if w not in stop and len(w) > 2)

tests = [
    ("Apple reports record quarterly earnings beating expectations", "REAL"),
    ("Federal Reserve raises interest rates amid inflation",         "REAL"),
    ("NASA confirms water ice found on surface of Moon",             "REAL"),
    ("Scientists discover pill that makes you live forever",         "FAKE"),
    ("Government secretly putting microchips in COVID vaccines",     "FAKE"),
    ("Moon landing was completely faked NASA admits",                "FAKE"),
]

print(f"{'Headline':<55} {'Expected':<8} {'Got':<6} {'Fake%':<8} {'Real%'}")
print("-" * 95)
for headline, expected in tests:
    c     = clean(headline)
    proba = p.predict_proba([c])[0]
    fake_p = round(proba[0]*100, 1)
    real_p = round(proba[1]*100, 1)
    got   = "FAKE" if fake_p > real_p else "REAL"
    ok    = "OK" if got == expected else "WRONG"
    print(f"{headline:<55} {expected:<8} {got:<6} {fake_p:<8} {real_p}  {ok}")
