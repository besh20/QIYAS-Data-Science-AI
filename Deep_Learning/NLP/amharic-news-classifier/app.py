"""Amharic News Topic Classifier — Streamlit app."""

import json
from datetime import datetime

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "tys22/amharic-news-classifier"
MAX_LENGTH = 256

EN_LABELS = {
    0: "Local News",
    1: "Sport",
    2: "Politics",
    3: "International News",
    4: "Business",
    5: "Entertainment",
}

EMOJI = {0: "🇪🇹", 1: "⚽", 2: "🏛️", 3: "🌍", 4: "💼", 5: "🎬"}

EXAMPLES = {
    "Sport": "የኢትዮጵያ ብሄራዊ ቡድን በዛሬው ጨዋታ 2-1 በማሸነፍ ወደ ቀጣይ ዙር ተሳካላት።",
    "Politics": "መንግስት በኢኮኖሚ ዘርፍ ላይ አዲስ ፖሊሲ እንደሚያውጣ የሚኒስቴር ገለጹ።",
    "Business": "የብሔራዊ ባንክ የውጭ ምንዛሬ ማዕቀፍ ለማሻሻል አዲስ መመሪያ አውጧል።",
    "Local news": "በአዲስ አበባ የመንገድ ግንባታ ፕሮጀክት በአንድ ዓመት ውስጥ ለመጠናቀቅ ተቀድሟል።",
    "International news": "በሶማሊያ መዲና ሞቃዲሾ በተፈፀመ የቦምብ ጥቃት ሰዎች መሞታቸው ተነግሯል።",
}

METRICS = {
    "accuracy": 0.8767,
    "f1_weighted": 0.8772,
    "f1_macro": 0.8486,
    "test_samples": 10_297,
}


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


def predict(text: str, top_k: int = 3):
    tokenizer, model = load_model()
    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

    scores, indices = probs.topk(min(top_k, len(probs)))
    results = []
    id2label = model.config.id2label
    for score, idx in zip(scores.tolist(), indices.tolist()):
        am_label = id2label.get(idx, id2label.get(str(idx)))
        results.append(
            {
                "id": idx,
                "amharic": am_label,
                "english": EN_LABELS[idx],
                "emoji": EMOJI[idx],
                "score": score,
            }
        )
    return results


# ── UI helpers ────────────────────────────────────────────────────────────────
def show_predictions(results: list[dict]):
    top = results[0]
    st.success(
        f"**{top['emoji']} {top['english']}** · {top['amharic']} · "
        f"**{top['score']:.1%}** confidence"
    )
    st.caption("All predictions")
    for r in results:
        st.progress(r["score"], text=f"{r['emoji']} {r['english']} ({r['score']:.1%})")


def add_to_history(text: str, results: list[dict]):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "preview": text[:80] + ("…" if len(text) > 80 else ""),
            "label": results[0]["english"],
            "score": results[0]["score"],
        },
    )
    st.session_state.history = st.session_state.history[:10]


# ── Pages ─────────────────────────────────────────────────────────────────────
def page_classify():
    st.subheader("Classify news text")

    col1, col2 = st.columns([1, 1])
    with col1:
        example = st.selectbox("Try an example", ["—"] + list(EXAMPLES.keys()))
    with col2:
        input_mode = st.radio("Input mode", ["Single text", "Headline + article"], horizontal=True)

    if example != "—":
        st.session_state["text_input"] = EXAMPLES[example]

    if input_mode == "Single text":
        text = st.text_area(
            "Paste Amharic news text",
            height=160,
            placeholder="Paste Amharic news text here…",
            key="text_input",
        )
    else:
        headline = st.text_input("Headline")
        article = st.text_area("Article body", height=120)
        text = f"{headline.strip()}. {article.strip()}".strip(". ")

    c1, c2, c3 = st.columns(3)
    with c1:
        run = st.button("Classify", type="primary", use_container_width=True)
    with c2:
        top_k = st.selectbox("Show top", [3, 6], index=0)
    with c3:
        st.metric("Characters", len(text) if text else 0)

    if run:
        if not text or len(text.strip()) < 10:
            st.warning("Please enter at least 10 characters of Amharic text.")
            return
        with st.spinner("Classifying…"):
            results = predict(text, top_k=top_k)
        show_predictions(results)
        add_to_history(text, results)
        st.download_button(
            "Download result (JSON)",
            data=json.dumps({"text": text, "predictions": results}, ensure_ascii=False, indent=2),
            file_name="prediction.json",
            mime="application/json",
        )


def page_batch():
    st.subheader("Batch classify (CSV)")
    st.markdown(
        "Upload a CSV with a **`text`** column (or we use the first column). "
        "Max **200 rows** per run."
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if not uploaded:
        st.info("Example CSV:\n\n```\ntext\nየኢትዮጵያ ብሄራዊ ቡድን …\nመንግስት አዲስ ፖሊሲ …\n```")
        return

    df = pd.read_csv(uploaded)
    text_col = "text" if "text" in df.columns else df.columns[0]
    df = df.head(200).copy()

    if st.button("Run batch classification", type="primary"):
        rows = []
        bar = st.progress(0)
        for i, raw in enumerate(df[text_col].astype(str)):
            if len(raw.strip()) >= 10:
                top = predict(raw.strip(), top_k=1)[0]
                rows.append(
                    {
                        "text": raw,
                        "category_en": top["english"],
                        "category_am": top["amharic"],
                        "confidence": round(top["score"], 4),
                    }
                )
            bar.progress((i + 1) / len(df))

        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True)
        st.download_button(
            "Download results CSV",
            data=out.to_csv(index=False).encode("utf-8-sig"),
            file_name="batch_predictions.csv",
            mime="text/csv",
        )


def page_about():
    st.subheader("About this project")
    st.markdown(
        """
        **Amharic News Topic Classifier** categorizes news into 6 topics using a
        fine-tuned [Afro-XLMR](https://huggingface.co/Davlan/afro-xlmr-base) model.

        | Topic | Amharic label |
        |-------|---------------|
        | Local News | ሀገር አቀፍ ዜና |
        | Sport | ስፖርት |
        | Politics | ፖለቲካ |
        | International News | ዓለም አቀፍ ዜና |
        | Business | ቢዝነስ |
        | Entertainment | መዝናኛ |
        """
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{METRICS['accuracy']:.1%}")
    m2.metric("F1 (weighted)", f"{METRICS['f1_weighted']:.1%}")
    m3.metric("F1 (macro)", f"{METRICS['f1_macro']:.1%}")
    m4.metric("Test set", f"{METRICS['test_samples']:,}")

    st.markdown(
        """
        **Dataset:** [Amharic News Text Classification](https://huggingface.co/datasets/israel/Amharic-News-Text-classification-Dataset)

        **Model:** [tys22/amharic-news-classifier](https://huggingface.co/tys22/amharic-news-classifier)

        **Note:** Business has fewer training samples, so it may score lower than other classes.
        """
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Amharic News Classifier", page_icon="", layout="wide")

    with st.sidebar:
        st.title("Amharic News")
        st.caption("Topic classifier for Ethiopian news")
        page = st.radio("Navigate", ["Classify", "Batch CSV", "About"], label_visibility="collapsed")
        st.divider()
        st.link_button("View model →", "https://huggingface.co/tys22/amharic-news-classifier")

        if st.session_state.get("history"):
            st.divider()
            st.caption("Recent predictions")
            for item in st.session_state.history[:5]:
                st.markdown(
                    f"**{item['label']}** ({item['score']:.0%})  \n"
                    f"<small>{item['preview']}</small>",
                    unsafe_allow_html=True,
                )

    if page == "Classify":
        page_classify()
    elif page == "Batch CSV":
        page_batch()
    else:
        page_about()


if __name__ == "__main__":
    main()
