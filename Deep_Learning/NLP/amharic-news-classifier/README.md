# Amharic News Topic Classifier

Fine-tuned transformer that classifies Amharic news into 6 topics.  
**Live model:** [tys22/amharic-news-classifier](https://huggingface.co/tys22/amharic-news-classifier)

## Run locally

```bash
cd amharic-news-classifier
pip install -r requirements.txt
streamlit run app.py
```

First run downloads ~1.1GB model from Hugging Face (cached after that).

## Deploy free (Streamlit Cloud)

1. Push this folder to GitHub (include `app.py`, `requirements.txt`, `.streamlit/config.toml`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Select your repo, main file: `app.py`.
4. Deploy. No secrets needed — model is public on HF.

## App features

| Feature | Description |
|---------|-------------|
| **Classify** | Paste text or use headline + article mode |
| **Examples** | One-click sample texts per category |
| **Top-k** | Show top 3 or all 6 predictions with confidence bars |
| **History** | Last 5 predictions in sidebar |
| **Batch CSV** | Upload CSV, classify up to 200 rows, download results |
| **Export** | Download single prediction as JSON |
| **About** | Model metrics, dataset links, limitations |

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 87.7% |
| F1 (weighted) | 87.7% |
| F1 (macro) | 84.9% |

## Project structure

```
amharic-news-classifier/
├── app.py                          # Streamlit web app
├── requirements.txt
├── .streamlit/config.toml
├── notebooks/
│   └── amharic_news_classifier.ipynb
└── README.md
```

## Dataset

[israel/Amharic-News-Text-classification-Dataset](https://huggingface.co/datasets/israel/Amharic-News-Text-classification-Dataset) — ~51k articles from Ethiopian news sites.
