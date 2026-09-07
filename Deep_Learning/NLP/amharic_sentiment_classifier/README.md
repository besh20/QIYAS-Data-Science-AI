# Amharic Sentiment Analyzer

A fine-tuned transformer pipeline for sentiment classification in Amharic — a language largely ignored by mainstream sentiment-analysis tools, despite being spoken by over 30 million people and dominant on Ethiopian social media.

**Live demo:** _[add your deployed Streamlit/HF Space link here]_
**Models on Hugging Face Hub:** [`tys22/amharic-sentiment-xlmr-v3-binary`](https://huggingface.co/tys22/amharic-sentiment-xlmr-v3-binary) · [`tys22/amharic-sentiment-xlmr-v2`](https://huggingface.co/tys22/amharic-sentiment-xlmr-v2)

---

## Why this project

Amharic is a genuinely underserved language in NLP: platforms like Facebook and Telegram moderate English content reasonably well, but Amharic content largely goes unmoderated, and there's no equivalent of "Google Alerts" for Amharic brand/sentiment monitoring. This project builds and evaluates working sentiment classifiers as a step toward closing that gap.

## What's here

Two fine-tuned XLM-RoBERTa models, each trained for a different tradeoff:

| Model | Task | Data | Test performance |
|---|---|---|---|
| **v3-binary** | Positive / Negative | Professionally-labeled Facebook & news comments (EBC, GCAO — via Zenodo, Alemneh 2021) | **82% accuracy**, F1 0.81 |
| **v2-3class** | Positive / Neutral / Negative | Human-annotated tweets, incl. political content ([AfriSenti](https://github.com/afrisenti-semeval/afrisenti-semeval), SemEval 2023) | 43% accuracy (vs. 33% random baseline); strong on positive/neutral, weak recall on negative |

Both are fine-tuned versions of `xlm-roberta-base`, served via a Streamlit app with:
- Live single-text analysis with confidence visualization
- Batch CSV upload and analysis with downloadable results
- Model switching (binary vs. 3-class) with in-app caveats
- Session history

## A real finding, not just a working demo

An earlier version of this project used a much larger (1.2M-row) Amharic sentiment dataset that turned out to be **auto-labeled by another AI model** rather than by humans — training stalled at random-guess accuracy no matter how hyperparameters were tuned, which turned out to be a data quality problem, not a modeling one. Switching to human-annotated benchmarks (AfriSenti, then the Zenodo professionally-labeled corpus) fixed this immediately, which is itself the more interesting engineering lesson from this project: **verify label provenance before debugging the model.**

The 3-class model also has a documented, specific weakness: it under-detects negative sentiment in politically-charged text stated in a flat, factual tone (e.g., news-style reports of violence or political conflict), likely because it lacks the real-world context human annotators use to judge such text as negative. This is flagged directly in the app when that model is selected.

## Tech stack

- **Model:** `xlm-roberta-base` (Hugging Face `transformers`), fine-tuned with the `Trainer` API
- **Data:** AfriSenti (SemEval 2023), Zenodo professionally-labeled Amharic corpus (Alemneh, 2021, CC-BY 4.0)
- **Training:** Google Colab (free T4 GPU) / local NVIDIA RTX 3050
- **Serving:** Streamlit, models hosted on Hugging Face Hub
- **Evaluation:** scikit-learn (precision/recall/F1, confusion matrices), custom hand-labeled test set

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Models download automatically from the Hugging Face Hub on first run and are cached afterward.

## Limitations

- The 3-class model's negative-detection weakness (see above) means it should not be used as-is for content moderation without further work.
- Both models are trained on relatively small datasets (5,975–9,479 rows) — performance on informal, code-switched, or highly dialectal Amharic text is untested.
- Not evaluated for bias across dialects, regions, or demographic groups.

## Attribution

- AfriSenti dataset: Muhammad et al., "AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages," SemEval 2023.
- Zenodo dataset: Girma Neshir Alemneh, "Negation handling for Amharic sentiment classification," Zenodo, DOI: [10.5281/zenodo.5005968](https://zenodo.org/records/5005968), CC-BY 4.0.
