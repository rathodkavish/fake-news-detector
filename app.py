"""
app.py
------
Gradio web GUI for the Fake News Detector.

Run:
    python app.py

Then open http://127.0.0.1:7860 in your browser.
"""

import os
import sys
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── ensure the project root is on sys.path ─────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from predictor import predict, get_metrics

PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# ── example news headlines ─────────────────────────────────────────────────────
EXAMPLES = [
    ["Scientists discover new vaccine that eliminates all known diseases overnight"],
    ["NASA confirms water found on the Moon's surface"],
    ["Government secretly putting mind-control chips in COVID vaccines, whistleblower claims"],
    ["Federal Reserve raises interest rates by 0.25 percent amid inflation concerns"],
    ["Drinking bleach cures cancer according to suppressed study"],
    ["Apple announces new iPhone model with improved battery life"],
    ["Moon landing was completely faked, NASA finally admits"],
    ["2 + 3 = 5"],
    ["The stock market rose 2.5% on Friday after strong jobs report"],
    ["Scientists discover pill that makes you live forever"],
]

# ── confidence gauge plot ──────────────────────────────────────────────────────
def _make_gauge(confidence: float, label: str) -> plt.Figure:
    """Semi-circle gauge showing prediction confidence."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    is_fake  = label == "FAKE"
    color    = "#ef4444" if is_fake else "#22c55e"
    bg_color = "#1f1f1f"

    # background arc (grey)
    theta  = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), lw=18, color=bg_color,
            solid_capstyle="round")

    # filled arc proportional to confidence
    fill_end = np.pi - (confidence / 100) * np.pi
    theta_fill = np.linspace(np.pi, fill_end, 200)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), lw=18, color=color,
            solid_capstyle="round")

    # centre text
    emoji = "🔴" if is_fake else "🟢"
    ax.text(0, 0.15, f"{confidence:.1f}%",
            ha="center", va="center", fontsize=26,
            fontweight="bold", color=color)
    ax.text(0, -0.22, label,
            ha="center", va="center", fontsize=18,
            fontweight="bold", color=color)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.55, 1.15)
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig


# ── probability bar chart ──────────────────────────────────────────────────────
def _make_bar(fake_prob: float, real_prob: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 2.2))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    bars = ax.barh(
        ["Real", "Fake"],
        [real_prob * 100, fake_prob * 100],
        color=["#22c55e", "#ef4444"],
        height=0.45, edgecolor="none",
    )
    for bar, val in zip(bars, [real_prob * 100, fake_prob * 100]):
        ax.text(
            bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=12,
            color="white", fontweight="bold",
        )
    ax.set_xlim(0, 115)
    ax.set_xlabel("Probability (%)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")
    plt.tight_layout(pad=0.5)
    return fig


# ── main prediction function called by Gradio ──────────────────────────────────
def run_prediction(text: str):
    """Returns (verdict_html, gauge_fig, bar_fig, detail_text)."""
    text = text.strip() if text else ""

    if not text:
        return (
            "<p style='color:#888;font-size:15px;'>Enter a headline above and click <b>Analyse</b>.</p>",
            None, None, ""
        )

    if not os.path.exists(MODEL_PATH):
        return (
            "<p style='color:#f97316;font-size:14px;'>⚠️ Model not found. "
            "Please run <code>python train_model.py</code> first, then restart the app.</p>",
            None, None, ""
        )

    result = predict(text)
    label  = result["label"]
    conf   = result["confidence"]
    fp     = result["fake_prob"]
    rp     = result["real_prob"]

    # ── special input type message ─────────────────────────────────────────────
    special_msg = result.get("message", "")
    input_type  = result.get("input_type", "normal")

    # ── verdict HTML ──────────────────────────────────────────────────────────
    if label == "FAKE":
        badge_style = (
            "background:#7f1d1d;color:#fca5a5;padding:4px 16px;"
            "border-radius:8px;font-size:24px;font-weight:700;"
        )
        icon = "🚨"
        msg  = special_msg or "This headline shows characteristics of <b>fake news</b>."
    elif label == "REAL":
        badge_style = (
            "background:#14532d;color:#86efac;padding:4px 16px;"
            "border-radius:8px;font-size:24px;font-weight:700;"
        )
        icon = "✅"
        msg  = special_msg or "This headline appears to be <b>credible / real news</b>."
    elif label == "NOT NEWS":
        badge_style = (
            "background:#1e3a5f;color:#93c5fd;padding:4px 16px;"
            "border-radius:8px;font-size:24px;font-weight:700;"
        )
        icon = "🔢"
        msg  = special_msg or "This input is not a news headline."
    else:  # UNKNOWN / too short
        badge_style = (
            "background:#292524;color:#a8a29e;padding:4px 16px;"
            "border-radius:8px;font-size:24px;font-weight:700;"
        )
        icon = "❓"
        msg  = special_msg or "Could not classify — please enter a more complete headline."

    # Show probability stats only for normal classifications
    if label in ("NOT NEWS", "UNKNOWN"):
        prob_html = f"<p style='font-size:13px;color:#888;'>Input type: <b style='color:#60a5fa;'>{input_type.replace('_',' ').title()}</b></p>"
    else:
        prob_html = f"""
        <p style="font-size:13px;color:#888;">
          Confidence: <b style="color:white;">{conf:.1f}%</b> &nbsp;|&nbsp;
          Fake prob: <b style="color:#ef4444;">{fp*100:.1f}%</b> &nbsp;|&nbsp;
          Real prob: <b style="color:#22c55e;">{rp*100:.1f}%</b>
        </p>"""

    verdict_html = f"""
    <div style="font-family:sans-serif;padding:12px 0;">
      <span style="{badge_style}">{icon} &nbsp; {label}</span>
      <p style="margin-top:12px;font-size:15px;color:#ccc;">{msg}</p>
      {prob_html}
    </div>
    """

    # ── detail accordion text ─────────────────────────────────────────────────
    detail = (
        f"Input text   : {text}\n"
        f"Input type   : {result.get('input_type', 'normal')}\n"
        f"Cleaned text : {result['clean_text']}\n\n"
        f"Raw fake probability : {fp:.6f}\n"
        f"Raw real probability : {rp:.6f}\n"
        f"Decision threshold   : {result.get('threshold', 0.5):.2f}\n"
        f"Predicted label      : {label}\n"
        f"Confidence           : {conf:.2f}%\n"
        f"Note                 : {result.get('message', '')}"
    )

    # Don't show gauge/bar for NOT NEWS / UNKNOWN
    if label in ("NOT NEWS", "UNKNOWN"):
        return verdict_html, None, None, detail
    return verdict_html, _make_gauge(conf, label), _make_bar(fp, rp), detail


# ── model stats tab helper ─────────────────────────────────────────────────────
def load_stats() -> str:
    if not os.path.exists(MODEL_PATH):
        return "Model not trained yet. Run **python train_model.py** first."
    try:
        m = get_metrics()
        lines = [
            "### 📊 Model Evaluation Metrics\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Accuracy  | **{m.get('accuracy', 'N/A')}** |",
            f"| ROC-AUC   | **{m.get('roc_auc', 'N/A')}** |",
            f"| CV F1 Mean| **{m.get('cv_f1_mean', 'N/A')}** |",
            f"| CV F1 Std | **± {m.get('cv_f1_std', 'N/A')}** |",
            f"| Train size| **{m.get('train_size', 'N/A'):,}** |",
            f"| Test size | **{m.get('test_size', 'N/A'):,}** |",
            f"| Threshold | **{m.get('best_threshold', 'N/A')}** |",
            "",
            "---",
            "**Model:** TF-IDF (unigrams + bigrams, max 150k features) + Logistic Regression",
            "**Data:**  ISOT (Reuters Real + Fake filtered) + WELFake (72k articles)",
        ]
        return "\n".join(str(l) for l in lines)
    except Exception as exc:
        return f"Error loading metrics: {exc}"


def load_eval_plot():
    path = os.path.join(PLOTS_DIR, "evaluation.png")
    return path if os.path.exists(path) else None


# ── Gradio layout ──────────────────────────────────────────────────────────────
CSS = """
body, .gradio-container { background: #0a0a0a !important; }
.gr-button-primary { background: #6d28d9 !important; border-color: #6d28d9 !important; }
.gr-box { border-color: #2a2a2a !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title  = "Fake News Detector",
    theme  = gr.themes.Base(
        primary_hue   = "purple",
        neutral_hue   = "zinc",
        font          = gr.themes.GoogleFont("Inter"),
    ),
    css    = CSS,
) as demo:

    # ── header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px;">
      <h1 style="font-size:2.2rem;font-weight:800;margin:0;">
        🕵️ Fake News Detector
      </h1>
      <p style="color:#aaa;font-size:1rem;margin-top:8px;">
        TF-IDF + Logistic Regression &nbsp;·&nbsp;
        Trained on ISOT + WELFake (100k+ articles) &nbsp;·&nbsp;
        Numbers &amp; Math Aware
      </p>
    </div>
    """)

    with gr.Tabs():

        # ── TAB 1: Detector ───────────────────────────────────────────────────
        with gr.Tab("🔍 Detector"):
            with gr.Row():
                with gr.Column(scale=3):
                    inp = gr.Textbox(
                        label       = "News Headline or Article Snippet",
                        placeholder = "Paste a headline here, e.g.  'Scientists discover cure for all diseases'",
                        lines       = 4,
                    )
                    with gr.Row():
                        btn_analyse = gr.Button("🔍 Analyse", variant="primary", scale=2)
                        btn_clear   = gr.Button("🗑️ Clear",   variant="secondary", scale=1)

                    verdict = gr.HTML(
                        value="<p style='color:#555;font-size:14px;'>Result will appear here.</p>"
                    )

                with gr.Column(scale=2):
                    gauge_plot = gr.Plot(label="Confidence Gauge")
                    bar_plot   = gr.Plot(label="Probability Breakdown")

            with gr.Accordion("🔬 Debug / Preprocessing Details", open=False):
                detail_box = gr.Textbox(label="", lines=8, interactive=False)

            gr.Examples(
                examples   = EXAMPLES,
                inputs     = [inp],
                label      = "📰 Try these examples",
            )

            # wire up buttons
            btn_analyse.click(
                fn      = run_prediction,
                inputs  = [inp],
                outputs = [verdict, gauge_plot, bar_plot, detail_box],
            )
            inp.submit(
                fn      = run_prediction,
                inputs  = [inp],
                outputs = [verdict, gauge_plot, bar_plot, detail_box],
            )
            btn_clear.click(
                fn      = lambda: ("", None, None, ""),
                outputs = [inp, gauge_plot, bar_plot, detail_box],
            )

        # ── TAB 2: Model Stats ────────────────────────────────────────────────
        with gr.Tab("📊 Model Stats"):
            gr.Markdown(load_stats())
            eval_img = load_eval_plot()
            if eval_img:
                gr.Image(value=eval_img, label="Evaluation Plots", show_label=True)
            else:
                gr.Markdown(
                    "_Evaluation plots not found. "
                    "Run `python train_model.py` to generate them._"
                )

        # ── TAB 3: How It Works ───────────────────────────────────────────────
        with gr.Tab("ℹ️ How It Works"):
            gr.Markdown("""
## How this detector works

### Pipeline overview
```
Raw headline
     │
     ▼
Text cleaning  (lowercase · remove URLs · strip punctuation · remove stopwords · stem)
     │
     ▼
TF-IDF Vectorizer  (unigrams + bigrams · max 50,000 features · sublinear TF scaling)
     │
     ▼
Logistic Regression  (C=5 · L2 · balanced class weights)
     │
     ▼
Prediction  →  FAKE / REAL  +  Confidence %
```

### Dataset — FakeNewsNet
| Split | Source | Fake | Real |
|-------|--------|------|------|
| PolitiFact | Political news fact-checked by PolitiFact.com | ✓ | ✓ |
| GossipCop  | Entertainment news fact-checked by GossipCop.com | ✓ | ✓ |

### Limitations
- The model is trained **only on headlines/titles** (not full article text).
- It may struggle with **satirical** headlines or **very recent** events.
- Always cross-check with trusted fact-checking sites like [Snopes](https://snopes.com) or [PolitiFact](https://politifact.com).

### Tech stack
`Python` · `scikit-learn` · `NLTK` · `Gradio` · `Matplotlib`
            """)

    gr.HTML("""
    <div style="text-align:center;padding:16px 0 4px;color:#555;font-size:12px;">
      Built with ❤️ using FakeNewsNet dataset &nbsp;·&nbsp; Model: TF-IDF + Logistic Regression
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
        inbrowser   = True,
    )
