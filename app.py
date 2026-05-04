import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SentimentIQ", page_icon="◈", layout="wide")

# ── Theme ─────────────────────────────────────────────────────────────────────
dark_mode = st.toggle("◐  Dark mode", value=True)

if dark_mode:
    PAGE_BG    = "#07060a"   # deep black-purple
    PANEL_BG   = "#12101a"
    BORDER     = "#2a1b2f"


    TEXT       = "#f5e9ff"   # soft white-pink (main text)
    TEXT2      = "#b9a6c9"   # secondary text


    ACCENT_A   = "#ff4fd8"
    ACCENT_B   = "#ff2d75"
    ACCENT_C   = "#c77dff"


    POSITIVE   = "#2ef2c0"
    NEGATIVE   = "#ff4d6d"
    NEUTRAL = "#ffbe0b"


    CHART_BG   = "#0e0b14"
    CHART_GRID = "#2a1b2f"
    CHART_TEXT = "#e6d6ff"   # brighter for visibility

    INPUT_BG   = "#14101d"


    CMAP       = "twilight_shifted"

else:
    PAGE_BG    = "#fff7fb"
    PANEL_BG   = "#ffffff"
    BORDER     = "#f3d6e7"

    TEXT       = "#1a0f1f"
    TEXT2      = "#6b4b5f"

    ACCENT_A   = "#ff4fd8"
    ACCENT_B   = "#ff2d75"
    ACCENT_C   = "#8b5cf6"

    POSITIVE   = "#00b894"
    NEGATIVE   = "#e84393"
    NEUTRAL = "#f59f00"

    CHART_BG   = "#ffffff"
    CHART_GRID = "#f3d6e7"
    CHART_TEXT = "#6b4b5f"

    INPUT_BG   = "#fff0f7"

    CMAP       = "PuRd"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, .stApp {{ background-color: {PAGE_BG} !important; }}

/* Remove max-width cap and tighten padding so it fills the browser window */
.block-container {{
    padding: 1.2rem 2rem 1.5rem !important;
    max-width: 100% !important;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

/* Collapse default top padding from Streamlit's toolbar area */
[data-testid="stAppViewContainer"] > section {{ padding-top: 0 !important; }}

.stToggle > label {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: {TEXT2} !important;
    letter-spacing: 0.05em;
}}

[data-testid="stMetricLabel"] p {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {TEXT2} !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Syne', sans-serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    color: {TEXT} !important;
}}

/* Progress bar */
.stProgress > div > div {{
    background-color: {BORDER} !important;
    height: 4px !important;
    border-radius: 99px !important;
}}
.stProgress > div > div > div {{
    background: linear-gradient(90deg, {ACCENT_A}, {ACCENT_B}) !important;
    border-radius: 99px !important;
}}

/* Selectbox */
[data-baseweb="select"] > div {{
    background-color: {INPUT_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}}

/* Text input */
.stTextInput input {{
    background-color: {INPUT_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 0.5rem 1rem !important;
}}
.stTextInput input:focus {{
    border-color: {ACCENT_A} !important;
    box-shadow: 0 0 0 3px {ACCENT_A}22 !important;
}}

/* Button */
.stButton > button {{
    background: linear-gradient(135deg, {ACCENT_A}, {ACCENT_B}) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 0.45rem 1.4rem !important;
}}
.stButton > button:hover {{ opacity: 0.85 !important; }}

hr {{ border-color: {BORDER} !important; margin: 0.6rem 0 !important; }}

/* Remove extra spacing Streamlit adds between elements */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {{
    gap: 0 !important;
}}
div[data-testid="column"] > div {{ gap: 0 !important; }}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:2px">
  <svg width="420" height="44"  viewBox="0 0 420 44" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%"   stop-color="{ACCENT_A}"/>
        <stop offset="100%" stop-color="{ACCENT_B}"/>
      </linearGradient>
    </defs>
    <text x="0" y="34"
          font-family="Syne, sans-serif"
          font-size="20"
          font-weight="900"
           letter-spacing="0.5"
          fill="url(#logoGrad)"> Sentiment Analysis </text>
  </svg>
 
</div>
<p style="font-family:'DM Sans',sans-serif;font-size:12px;color:{TEXT2};margin:0 0 0.5rem">
  Real-time NLP sentiment classification · Two-model ensemble
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Load metrics ──────────────────────────────────────────────────────────────
try:
    res = requests.get("http://127.0.0.1:8000/metrics", timeout=5)
    res.raise_for_status()
    metrics = res.json()
except Exception as e:
    st.error(f"⚠ Could not reach API: {e}")
    st.stop()

lr = metrics["logistic_regression"]
nb = metrics["naive_bayes"]

# ── Row 1: Model cards ────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="medium")

def model_card(col, tag, tag_color, title, data):
    with col:
        st.markdown(f"""
        <div style="background:{PANEL_BG};border:1px solid {BORDER};border-radius:16px;padding:16px 20px 8px">
          <span style="display:inline-block;padding:2px 9px;border-radius:6px;
                       font-family:'JetBrains Mono',monospace;font-size:10px;
                       letter-spacing:.1em;text-transform:uppercase;
                       background:{tag_color}18;border:1px solid {tag_color}44;color:{tag_color}">
            {tag}
          </span>
          <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
                      color:{TEXT};margin:8px 0 12px">{title}</div>
        </div>
        """, unsafe_allow_html=True)
        for label, key in [("Accuracy","accuracy"),("F1 Score","f1"),("Precision","precision"),("Recall","recall")]:
            val = data[key]
            st.markdown(f"""
            <div style="background:{PANEL_BG};border-left:1px solid {BORDER};
                        border-right:1px solid {BORDER};padding:0 20px 2px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">
                <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                             text-transform:uppercase;letter-spacing:.1em;color:{TEXT2}">{label}</span>
                <span style="font-family:'Syne',sans-serif;font-size:14px;
                             font-weight:700;color:{TEXT}">{val:.3f}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(val)
        st.markdown(f'<div style="background:{PANEL_BG};border:1px solid {BORDER};border-top:none;'
                    f'border-radius:0 0 16px 16px;height:12px"></div>', unsafe_allow_html=True)

model_card(col1, "LR", ACCENT_A, "Logistic Regression", lr)
model_card(col2, "NB", ACCENT_C, "Naive Bayes", nb)

st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

# ── Row 2: Comparison chart + Confusion matrix ────────────────────────────────
chart_col, cm_col = st.columns(2, gap="medium", vertical_alignment="top")

# — Comparison bar chart —
with chart_col:
    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;color:{TEXT};margin-bottom:1px">
      Model comparison
    </div>
    <div style="font-family:'DM Sans',sans-serif;font-size:11px;color:{TEXT2};margin-bottom:10x">
      Accuracy vs F1 score
    </div>
    """, unsafe_allow_html=True)

    models   = ["Logistic Regression", "Naive Bayes"]
    acc_vals = [lr["accuracy"], nb["accuracy"]]
    f1_vals  = [lr["f1"],       nb["f1"]]
    x = np.arange(len(models))
    W = 0.28

    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    bars1 = ax.bar(x - W/2, acc_vals, W, color=ACCENT_A, alpha=0.9, zorder=3)
    bars2 = ax.bar(x + W/2, f1_vals,  W, color=ACCENT_B, alpha=0.9, zorder=3)
    for bar in [*bars1, *bars2]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.004, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8, color=CHART_TEXT, fontfamily="monospace")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, color=CHART_TEXT)
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors=CHART_TEXT, labelsize=8)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.yaxis.grid(True, color=CHART_GRID, linewidth=0.5, zorder=0)
    ax.tick_params(axis="y", colors=CHART_TEXT, labelsize=7)
    ax.legend(
        handles=[mpatches.Patch(color=ACCENT_A, label="Accuracy"),
                 mpatches.Patch(color=ACCENT_B, label="F1 Score")],
        frameon=False, labelcolor=CHART_TEXT, fontsize=8, loc="lower right"
    )
    plt.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=True)

# — Confusion matrix —
with cm_col:
    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;color:{TEXT};margin-bottom:1px">
      Confusion matrix
    </div>
    <div style="font-family:'DM Sans',sans-serif;font-size:11px;color:{TEXT2};margin-bottom:8px">
      Per-class prediction breakdown
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox("Model", models, label_visibility="collapsed")

    cm = np.array(
        lr["confusion_matrix"]
        if model_choice == "Logistic Regression"
        else nb["confusion_matrix"]
    )

    total = cm.sum()
    n = cm.shape[0]

    labels = ["Negative", "Neutral", "Positive"]  # 3-class labels

    fig2, ax2 = plt.subplots(figsize=(4.5, 3.6))
    fig2.patch.set_facecolor(CHART_BG)
    ax2.set_facecolor(CHART_BG)

    im = ax2.imshow(cm, cmap=CMAP, aspect="auto")

    #  dynamic loop
    for i in range(n):
        for j in range(n):
            val = cm[i][j]
            ax2.text(
                j, i - 0.1,
                str(val),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=TEXT,
                fontfamily="monospace"
            )
            ax2.text(
                j, i + 0.25,
                f"{val / total * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color=CHART_TEXT,
                fontfamily="monospace"
            )

    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))

    ax2.set_xticklabels(labels[:n], fontsize=9, color=CHART_TEXT)
    ax2.set_yticklabels(labels[:n], fontsize=9, color=CHART_TEXT)

    ax2.set_xlabel("Predicted", fontsize=9, color=CHART_TEXT, labelpad=6)
    ax2.set_ylabel("Actual", fontsize=9, color=CHART_TEXT, labelpad=6)

    ax2.tick_params(colors=CHART_TEXT)
    ax2.spines[["top", "right", "left", "bottom"]].set_color(CHART_GRID)

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7, colors=CHART_TEXT)
    cbar.outline.set_edgecolor(CHART_GRID)

    plt.tight_layout(pad=1.0)
    st.pyplot(fig2, use_container_width=True)
st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

# ── Row 3: Live prediction ────────────────────────────────────────────────────
st.markdown(f"""
<div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;color:{TEXT};margin-bottom:1px">
  Live prediction
</div>
<div style="font-family:'DM Sans',sans-serif;font-size:11px;color:{TEXT2};margin-bottom:8px">
  Run both models on any input text
</div>
""", unsafe_allow_html=True)

inp_col, btn_col = st.columns([5, 1],  vertical_alignment="bottom")
with inp_col:
    text_input = st.text_input("Text", placeholder="Type something to analyze…", label_visibility="collapsed")
with btn_col:
    st.markdown("<div style='height:-8px'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("◈  Analyze", use_container_width=True)

if analyze_btn:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            pred_res = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": text_input}, timeout=5
            )
            pred_res.raise_for_status()
            data = pred_res.json()

            rc1, rc2 = st.columns(2, gap="medium")
            for col, tag, name, result in [
                (rc1, "LR", "Logistic Regression", data["logistic_regression"]),
                (rc2, "NB", "Naive Bayes",         data["naive_bayes"]),
            ]:
                sent = result["sentiment"]
                conf = result.get("confidence")
                sent_lower = sent.lower()
                if sent_lower == "positive":
                    color = POSITIVE
                    icon = "▲"
                elif sent_lower == "neutral":
                    color = NEUTRAL
                    icon = "●"
                else:
                    color = NEGATIVE
                    icon = "▼"
                conf_line = (f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;'
                             f'color:{TEXT2};margin-top:3px">Confidence: {conf:.2%}</div>') if conf else ""
                with col:
                    st.markdown(f"""
                    <div style="background:{color}0f;border:1px solid {color}44;
                                border-left:4px solid {color};border-radius:0 12px 12px 0;
                                padding:12px 18px;margin-top:4px">
                      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                                  text-transform:uppercase;letter-spacing:.1em;color:{TEXT2};margin-bottom:5px">
                        {tag} · {name}
                      </div>
                      <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:{color}">
                        {icon} {sent.capitalize()}
                      </div>
                      {conf_line}
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"API error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
    border-top:2px solid {ACCENT_B};
    margin-top:1.2rem;
    padding-top:10px;
    display:flex;
    justify-content:space-between;
    font-family:'JetBrains Mono',monospace;
    font-size:10px;
    letter-spacing:.08em;
    text-transform:uppercase;
    color:{ACCENT_B};
    font-weight:800;
">
  <span style="color:{ACCENT_B}; font-weight:900;">
    ◈ SentimentIQ · ML Dashboard
  </span>
  <span style="color:{ACCENT_B}; font-weight:900;">
    Logistic Regression + Naive Bayes
  </span>
</div>
""", unsafe_allow_html=True)