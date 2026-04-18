"""verify.py — quick sanity check after retraining"""
from predictor import predict, get_metrics

tests = [
    ("Scientists discover pill that makes you live forever",          "FAKE"),
    ("Government secretly putting microchips in vaccines",            "FAKE"),
    ("Moon landing was completely faked NASA admits",                  "FAKE"),
    ("Drinking bleach cures cancer suppressed study",                  "FAKE"),
    ("5G towers spreading COVID radiation kills people",               "FAKE"),
    ("Aliens have landed government finally admits it",                "FAKE"),
    ("Scientists discover new vaccine eliminates all diseases overnight", "FAKE"),
    ("Bill Gates depopulation agenda microchip COVID vaccine",         "FAKE"),
    ("Illuminati controls all world governments secret meeting",       "FAKE"),
    ("Bigfoot photographed in national park confirmed real",           "FAKE"),
    ("Federal Reserve raises interest rates",                          "REAL"),
    ("NASA confirms water ice found on Moon",                          "REAL"),
    ("Apple reports record quarterly earnings",                        "REAL"),
    ("The stock market rose 2.5 percent on jobs report",              "REAL"),
    ("WHO approves new malaria vaccine for children",                  "REAL"),
    ("Supreme Court rules on immigration case",                        "REAL"),
    ("2 + 3 = 5",                                                     "NOT NEWS"),
]

print("VERIFICATION RESULTS")
print("=" * 72)
all_ok = True
for text, expected in tests:
    r = predict(text)
    ok = "PASS" if r["label"] == expected else "FAIL"
    if ok == "FAIL":
        all_ok = False
    print(f"{ok} [{expected:8s} -> {r['label']:8s}] conf={r['confidence']:5.1f}%  {text[:50]}")

print()
m = get_metrics()
print("MODEL METRICS:")
print(f"  Accuracy : {m.get('accuracy', 'N/A')}")
print(f"  ROC-AUC  : {m.get('roc_auc',  'N/A')}")
print(f"  CV F1    : {m.get('cv_f1_mean','N/A')} +/- {m.get('cv_f1_std','N/A')}")
print(f"  Threshold: {m.get('best_threshold','N/A')}")
print(f"  TrainSize: {m.get('train_size','N/A')}")
print()
print("ALL PASS ✓" if all_ok else "SOME FAILED ✗ — review above")
