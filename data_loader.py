"""
data_loader.py  (v9 - title + text body, Pro Version)
------------------------------------------------------
Loads ALL available datasets in priority order:

  1. ISOT        - True.csv (real Reuters news)  ← uses title + full text body
  2. ISOT        - Fake.csv (all fake, no filter) ← uses title + full text body
  3. FakeNewsNet - PolitiFact + GossipCop (GitHub auto-download)
  4. WELFake     - WELFake_Dataset.csv (if present in data/)  ← title + text
  5. LIAR        - liar.csv (if present in data/)
  6. COVID Fake  - covid_fake.csv (if present in data/)

Key improvement vs v8:
  Using `title + " " + text` (when text column is available) gives the model
  far richer features than title-only training, pushing accuracy noticeably higher.

Label convention:  0 = FAKE,  1 = REAL
"""

import os, re, requests, pandas as pd, nltk
from nltk.corpus import stopwords
from nltk.stem   import PorterStemmer
from tqdm        import tqdm
from io          import StringIO

BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "news_data.csv")

# ── file paths ─────────────────────────────────────────────────────────────────
ISOT_FAKE    = os.path.join(DATA_DIR, "Fake.csv")
ISOT_TRUE    = os.path.join(DATA_DIR, "True.csv")
WELFAKE_PATH = os.path.join(DATA_DIR, "WELFake_Dataset.csv")
LIAR_PATH    = os.path.join(DATA_DIR, "liar.csv")
COVID_PATH   = os.path.join(DATA_DIR, "covid_fake.csv")

# ── FakeNewsNet GitHub URLs ────────────────────────────────────────────────────
FNN_BASE  = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/"
FNN_FILES = {
    "politifact_fake": (FNN_BASE + "politifact_fake.csv", 0),
    "politifact_real": (FNN_BASE + "politifact_real.csv", 1),
    "gossipcop_fake":  (FNN_BASE + "gossipcop_fake.csv",  0),
    "gossipcop_real":  (FNN_BASE + "gossipcop_real.csv",  1),
}

# ── NLTK ───────────────────────────────────────────────────────────────────────
_stemmer_obj = None
_stop_obj    = None

def _ensure_nltk():
    for r in ["stopwords", "punkt"]:
        try: nltk.download(r, quiet=True)
        except: pass

def _tools():
    global _stemmer_obj, _stop_obj
    if _stemmer_obj is None:
        _ensure_nltk()
        _stemmer_obj = PorterStemmer()
        _stop_obj    = set(stopwords.words("english"))
    return _stemmer_obj, _stop_obj


# ── text cleaning (canonical – must match predictor._clean exactly) ───────────
def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip(): return ""
    stemmer, stop = _tools()
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"<[^>]+>", " ", text)             # strip HTML
    text = re.sub(r"[^a-z0-9\s]", " ", text)         # keep letters + digits
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    for w in text.split():
        if len(w) > 1:
            if w.isdigit():
                tokens.append(w)                # keep numbers as-is
            elif w not in stop:
                tokens.append(stemmer.stem(w))  # stem non-stop words
    return " ".join(tokens)


def _combine_title_text(row, title_col, text_col=None):
    """Combine title and body text for richer features."""
    title = str(row.get(title_col, "") or "")
    if text_col and text_col in row.index:
        body = str(row.get(text_col, "") or "")
        # Truncate body to ~500 chars to avoid memory explosion
        body = body[:500]
        combined = (title + " " + body).strip()
    else:
        combined = title
    return combined if combined else ""


# ── helpers ────────────────────────────────────────────────────────────────────
def _read_csv_safe(path):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()


# ── Dataset 1 : ISOT ───────────────────────────────────────────────────────────
def _load_isot():
    frames = []

    # True.csv — all rows are high-quality Reuters real news
    if os.path.exists(ISOT_TRUE):
        df    = _read_csv_safe(ISOT_TRUE)
        tcol  = "title" if "title" in df.columns else df.columns[0]
        txcol = "text"  if "text"  in df.columns else None
        if txcol:
            df["combined"] = df.apply(lambda r: _combine_title_text(r, tcol, txcol), axis=1)
        else:
            df["combined"] = df[tcol].astype(str)
        out = df[["combined"]].copy()
        out.columns = ["title"]
        out["label"]  = 1
        out["source"] = "isot_real"
        frames.append(out)
        print(f"    ISOT True.csv       : {len(out):,} rows  (real, using {'title+text' if txcol else 'title only'})")
    else:
        print("    [INFO] True.csv not found — skipping")

    # Fake.csv — use ALL rows
    if os.path.exists(ISOT_FAKE):
        df    = _read_csv_safe(ISOT_FAKE)
        tcol  = "title" if "title" in df.columns else df.columns[0]
        txcol = "text"  if "text"  in df.columns else None
        if txcol:
            df["combined"] = df.apply(lambda r: _combine_title_text(r, tcol, txcol), axis=1)
        else:
            df["combined"] = df[tcol].astype(str)
        out = df[["combined"]].copy()
        out.columns = ["title"]
        out["label"]  = 0
        out["source"] = "isot_fake"
        frames.append(out)
        print(f"    ISOT Fake.csv       : {len(out):,} rows  (fake, using {'title+text' if txcol else 'title only'})")
    else:
        print("    [INFO] Fake.csv not found — skipping")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Dataset 2 : FakeNewsNet (auto-download) ───────────────────────────────────
def _load_fakenewsnet():
    print("  [2/5] FakeNewsNet (GitHub auto-download) ...")
    frames = []
    for name, (url, label) in FNN_FILES.items():
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            if "title" not in df.columns: continue
            df = df[["title"]].copy()
            df["label"] = label; df["source"] = name
            frames.append(df)
            print(f"    {name:<25}: {len(df):,} rows")
        except Exception as e:
            print(f"    [WARNING] {name}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Dataset 3 : WELFake ───────────────────────────────────────────────────────
def _load_welfake():
    print("  [3/5] WELFake dataset ...")
    if not os.path.exists(WELFAKE_PATH):
        print("    [INFO] WELFake_Dataset.csv not found — skipping")
        return pd.DataFrame()
    df = _read_csv_safe(WELFAKE_PATH)
    # WELFake columns: Unnamed:0, title, text, label  (0=fake, 1=real)
    tcol  = "title" if "title" in df.columns else df.columns[1]
    txcol = "text"  if "text"  in df.columns else None
    lcol  = "label" if "label" in df.columns else df.columns[-1]
    if txcol:
        df["combined"] = df.apply(lambda r: _combine_title_text(r, tcol, txcol), axis=1)
    else:
        df["combined"] = df[tcol].astype(str)
    out = df[["combined", lcol]].copy()
    out.columns = ["title", "label"]
    out["label"]  = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)
    out["source"] = "welfake"
    out = out.dropna(subset=["title"])
    fake_n = (out.label==0).sum(); real_n = (out.label==1).sum()
    print(f"    WELFake_Dataset.csv : {len(out):,} rows  (fake={fake_n:,} real={real_n:,}, {'title+text' if txcol else 'title only'})")
    return out[["title", "label", "source"]]


# ── Dataset 4 : LIAR ──────────────────────────────────────────────────────────
def _load_liar():
    print("  [4/5] LIAR dataset ...")
    if not os.path.exists(LIAR_PATH):
        print("    [INFO] liar.csv not found — skipping")
        return pd.DataFrame()
    df = _read_csv_safe(LIAR_PATH)
    title_col = next((c for c in ["statement","title","headline","text"] if c in df.columns), None)
    label_col = next((c for c in ["label","Label","truth","verdict"] if c in df.columns), None)
    if not title_col or not label_col:
        print(f"    [WARNING] Could not find title/label columns. Columns: {df.columns.tolist()}")
        return pd.DataFrame()
    out = df[[title_col, label_col]].copy()
    out.columns = ["title", "raw_label"]
    fake_labels = {"false","pants-fire","barely-true","pants on fire","lie","fake"}
    out["label"]  = out["raw_label"].apply(
        lambda x: 0 if str(x).strip().lower() in fake_labels else 1
    )
    out["source"] = "liar"
    out = out.dropna(subset=["title"])[["title","label","source"]]
    fake_n = (out.label==0).sum(); real_n = (out.label==1).sum()
    print(f"    liar.csv            : {len(out):,} rows  (fake={fake_n:,} real={real_n:,})")
    return out


# ── Dataset 5 : COVID Fake News ───────────────────────────────────────────────
def _load_covid():
    print("  [5/5] COVID-19 Fake News dataset ...")
    if not os.path.exists(COVID_PATH):
        print("    [INFO] covid_fake.csv not found — skipping")
        return pd.DataFrame()
    df = _read_csv_safe(COVID_PATH)
    title_col = next((c for c in ["title","headline","text","statement"] if c in df.columns), None)
    label_col = next((c for c in ["label","Label","verdict","real"] if c in df.columns), None)
    if not title_col or not label_col:
        print(f"    [WARNING] Columns not recognised: {df.columns.tolist()}")
        return pd.DataFrame()
    out = df[[title_col, label_col]].copy()
    out.columns = ["title","raw_label"]
    out["label"]  = out["raw_label"].apply(
        lambda x: 0 if str(x).strip().lower() in {"fake","false","0"} else 1
    )
    out["source"] = "covid_fake"
    out = out.dropna(subset=["title"])[["title","label","source"]]
    fake_n = (out.label==0).sum(); real_n = (out.label==1).sum()
    print(f"    covid_fake.csv      : {len(out):,} rows  (fake={fake_n:,} real={real_n:,})")
    return out


# ── main ───────────────────────────────────────────────────────────────────────
def load_and_prepare(force_rebuild: bool = False) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_CSV) and not force_rebuild:
        print(f"[data_loader] Cached data found -> {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV)
        print(f"  Rows: {len(df):,}  Fake: {(df.label==0).sum():,}  Real: {(df.label==1).sum():,}")
        return df

    print("[data_loader] Building dataset from all sources ...\n")

    print("  [1/5] ISOT dataset ...")
    isot   = _load_isot()
    fnn    = _load_fakenewsnet()
    wel    = _load_welfake()
    liar   = _load_liar()
    covid  = _load_covid()

    all_frames = [f for f in [isot, fnn, wel, liar, covid] if not f.empty]
    if not all_frames:
        raise RuntimeError("No data loaded! Check your data/ folder.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined[["title","label","source"]].copy()
    combined.dropna(subset=["title"], inplace=True)
    combined["title"] = combined["title"].astype(str).str.strip()
    combined = combined[combined["title"].str.len() > 5]
    combined.drop_duplicates(subset=["title"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Per-source summary
    print("\n  Per-source breakdown:")
    for src, grp in combined.groupby("source"):
        f = (grp.label==0).sum(); r = (grp.label==1).sum()
        print(f"    {src:<28}: {len(grp):>6,}  fake={f:>5,}  real={r:>5,}")

    print("\n[data_loader] Cleaning text ...")
    tqdm.pandas(desc="  Preprocessing")
    combined["clean_title"] = combined["title"].progress_apply(clean_text)
    combined = combined[combined["clean_title"].str.len() > 0].reset_index(drop=True)

    combined.to_csv(OUTPUT_CSV, index=False)
    fake_n = (combined.label==0).sum()
    real_n = (combined.label==1).sum()
    print(f"\n[data_loader] Final dataset saved:")
    print(f"  Total : {len(combined):,}")
    print(f"  Fake  : {fake_n:,}  ({100*fake_n/len(combined):.1f}%)")
    print(f"  Real  : {real_n:,}  ({100*real_n/len(combined):.1f}%)")
    print(f"  Path  : {OUTPUT_CSV}")
    return combined


if __name__ == "__main__":
    load_and_prepare(force_rebuild=True)
