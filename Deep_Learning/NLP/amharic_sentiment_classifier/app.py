"""
Amharic Sentiment Analyzer — Streamlit app
Serves two fine-tuned XLM-RoBERTa models from the Hugging Face Hub:
  - v3 (binary: positive/negative) — cleaner, higher accuracy, trained on professionally-labeled data
  - v2 (3-class: positive/neutral/negative) — trained on real tweets, includes political content

Run locally with:  streamlit run app.py
"""

import io
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Amharic Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS = {
    "v3-binary": {
        "repo": "tys22/amharic-sentiment-xlmr-v3-binary",
        "label": "Binary (positive / negative)",
        "description": "Trained on professionally-labeled EBC & GCAO comments. Cleaner, higher accuracy (~82% test).",
        "caveat": None,
    },
    "v2-3class": {
        "repo": "tys22/amharic-sentiment-xlmr-v2",
        "label": "3-class (positive / neutral / negative)",
        "description": "Trained on real Amharic tweets (AfriSenti), including political content.",
        "caveat": (
            "Known limitation: tends to under-detect negative sentiment in politically-charged text "
            "written in a flat, factual tone — such text is often predicted as neutral instead."
        ),
    },
}

LABEL_COLORS = {
    "positive": "#C9A876",
    "neutral": "#8A93A6",
    "negative": "#B85C4A",
    "0": "#B85C4A",
    "1": "#C9A876",
}

EXAMPLES = [
    "አንቺ ቆንጆ ነሽ ፣ በጣም ደስ ብሎኛል",
    "ይህ በጣም መጥፎ ነገር ነው፣ አልወደድኩትም",
    "ዛሬ ወደ ገበያ ሄጄ ነበር",
    "የፈረንጅ እርቅ! እርቅ! ህወሓትን ለማስታጠቅ!",
]

# ---------------------------------------------------------------------------
# Styling — custom theme, not default Streamlit chrome
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #1C1B22;
        --bg-panel: #24232C;
        --bg-panel-alt: #2A2933;
        --gold: #C9A876;
        --clay: #B85C4A;
        --ink: #ECE8E0;
        --ink-dim: #9B98A6;
        --border: #38363F;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--ink);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, .hero-title {
        font-family: 'Fraunces', serif;
    }

    .hero {
        padding: 2.5rem 0 1.5rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.3rem;
        letter-spacing: -0.01em;
    }
    .hero-sub {
        color: var(--ink-dim);
        font-size: 1.05rem;
        max-width: 640px;
        line-height: 1.5;
    }

    .result-card {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.5rem 1.75rem;
        margin-top: 1rem;
    }

    .bar-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0.55rem 0;
    }
    .bar-label {
        width: 90px;
        font-size: 0.92rem;
        color: var(--ink-dim);
        text-transform: capitalize;
    }
    .bar-track {
        flex: 1;
        background: var(--bg-panel-alt);
        border-radius: 6px;
        height: 14px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 6px;
    }
    .bar-pct {
        width: 52px;
        text-align: right;
        font-size: 0.9rem;
        color: var(--ink-dim);
        font-variant-numeric: tabular-nums;
    }

    .verdict {
        font-family: 'Fraunces', serif;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.9rem;
    }

    .caveat-box {
        background: rgba(184, 92, 74, 0.12);
        border-left: 3px solid var(--clay);
        padding: 0.7rem 1rem;
        border-radius: 4px;
        font-size: 0.88rem;
        color: var(--ink-dim);
        margin-top: 0.75rem;
    }

    .example-chip {
        display: inline-block;
        background: var(--bg-panel-alt);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 0.35rem 0.9rem;
        font-size: 0.85rem;
        color: var(--ink-dim);
        margin: 0.2rem 0.3rem 0.2rem 0;
        cursor: pointer;
    }

    .stat-pill {
        background: var(--bg-panel-alt);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        text-align: center;
    }
    .stat-pill-num {
        font-family: 'Fraunces', serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--gold);
    }
    .stat-pill-label {
        font-size: 0.78rem;
        color: var(--ink-dim);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-panel);
        border-right: 1px solid var(--border);
    }

    .stButton > button {
        background-color: var(--gold);
        color: #1C1B22;
        border: none;
        font-weight: 600;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background-color: #d9bd8f;
        color: #1C1B22;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model loading — cached so it only downloads/loads once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(repo_id: str):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    return {model.config.id2label[i]: float(probs[i]) for i in range(len(probs))}


def render_bars(results: dict):
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    top_label, top_score = sorted_results[0]
    st.markdown(f'<div class="verdict">{top_label.capitalize()} · {top_score:.0%}</div>', unsafe_allow_html=True)
    for label, score in sorted_results:
        color = LABEL_COLORS.get(label.lower(), "#C9A876")
        st.markdown(
            f"""
            <div class="bar-row">
                <div class="bar-label">{label}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{score*100:.1f}%; background:{color};"></div>
                </div>
                <div class="bar-pct">{score:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model")
    model_key = st.radio(
        "Choose a model",
        options=list(MODELS.keys()),
        format_func=lambda k: MODELS[k]["label"],
        label_visibility="collapsed",
    )
    st.caption(MODELS[model_key]["description"])
    if MODELS[model_key]["caveat"]:
        st.markdown(f'<div class="caveat-box">{MODELS[model_key]["caveat"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Session stats")
    total = len(st.session_state.history)
    st.markdown(
        f"""
        <div class="stat-pill">
            <div class="stat-pill-num">{total}</div>
            <div class="stat-pill-label">Texts analyzed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if total > 0 and st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.caption("Fine-tuned XLM-RoBERTa · Hosted on Hugging Face Hub")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Amharic Sentiment Analyzer</div>
        <div class="hero-sub">
            Fine-tuned transformer models for reading sentiment in Amharic text —
            built for a language most sentiment tools ignore.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner(f"Loading {MODELS[model_key]['label']} model..."):
    tokenizer, model = load_model(MODELS[model_key]["repo"])

# ---------------------------------------------------------------------------
# Tabs: single text / batch
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Analyze text", "Batch (CSV)"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        text = st.text_area(
            "Amharic text",
            value=st.session_state.text_input,
            height=140,
            placeholder="Type or paste Amharic text here...",
            label_visibility="collapsed",
        )

        st.markdown("**Try an example:**")
        chip_cols = st.columns(len(EXAMPLES))
        for i, ex in enumerate(EXAMPLES):
            with chip_cols[i]:
                if st.button(ex[:18] + ("…" if len(ex) > 18 else ""), key=f"ex_{i}"):
                    st.session_state.text_input = ex
                    st.rerun()

        analyze = st.button("Analyze", type="primary", use_container_width=False)

    with col2:
        if text:
            word_count = len(text.split())
            char_count = len(text)
            st.markdown(
                f"""
                <div class="stat-pill" style="margin-bottom:0.6rem;">
                    <div class="stat-pill-num">{word_count}</div>
                    <div class="stat-pill-label">Words</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-pill-num">{char_count}</div>
                    <div class="stat-pill-label">Characters</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if analyze and text.strip():
        with st.spinner("Analyzing..."):
            start = time.time()
            results = predict(text, tokenizer, model)
            elapsed = time.time() - start

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        render_bars(results)
        st.caption(f"Inference time: {elapsed*1000:.0f}ms")
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.history.insert(
            0,
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "text": text,
                "model": MODELS[model_key]["label"],
                "prediction": max(results, key=results.get),
                "confidence": max(results.values()),
            },
        )
    elif analyze:
        st.warning("Enter some text first.")

    if st.session_state.history:
        st.markdown("### Recent analyses")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("Upload a CSV with a column of Amharic text to analyze in bulk.")
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)

        text_col = st.selectbox("Which column contains the text?", df.columns)

        if st.button("Run batch analysis", type="primary"):
            progress = st.progress(0, text="Analyzing...")
            predictions = []
            confidences = []
            n = len(df)
            for i, row_text in enumerate(df[text_col].astype(str)):
                results = predict(row_text, tokenizer, model)
                top_label = max(results, key=results.get)
                predictions.append(top_label)
                confidences.append(results[top_label])
                progress.progress((i + 1) / n, text=f"Analyzing... {i+1}/{n}")

            df["predicted_sentiment"] = predictions
            df["confidence"] = confidences
            progress.empty()

            st.success(f"Done — {n} rows analyzed.")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Distribution chart
            st.markdown("**Prediction distribution:**")
            dist = df["predicted_sentiment"].value_counts()
            st.bar_chart(dist)

            # Download button
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download results as CSV",
                data=csv_buffer.getvalue(),
                file_name="amharic_sentiment_results.csv",
                mime="text/csv",
            )
