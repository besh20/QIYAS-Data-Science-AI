# Customer Segmentation with RFM + K-Means Clustering

Segmenting e-commerce customers into behavioral groups using **RFM analysis** (Recency, Frequency, Monetary value) and **K-Means clustering** — including a full analysis notebook, a production-style scoring pipeline with drift monitoring, and an interactive dashboard app for non-technical users.

## Key Finding

Just **16.2% of customers generate 64.4% of total revenue.** RFM clustering surfaces exactly who those customers are — and separates them from the 37% of customers who haven't purchased in ~6 months and are at risk of churning.

## Dataset

[UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — ~541K transactions from a UK-based online retailer, Dec 2010–Dec 2011.

## Project Workflow

1. **Data cleaning** — removed missing customer IDs, duplicates, cancelled orders, and non-product fees (postage, manual adjustments), taking the dataset from 541,909 → 391,295 rows
2. **EDA** — revenue trends, top markets, customer spend distribution
3. **Feature engineering** — built an RFM table (one row per customer) from raw transactions
4. **Preprocessing** — log-transformed and standardized RFM features to handle skew and make them usable for a distance-based algorithm
5. **Clusterability check** — verified the data has genuine group structure (Hopkins statistic = 0.949) before clustering
6. **Model selection** — used the elbow method + silhouette score to choose K=4
7. **Model training** — fit K-Means (scikit-learn) with K=4
8. **Evaluation & interpretation** — evaluated cluster quality and profiled/named each segment
9. **Production pipeline** — packaged RFM computation + scaling + clustering + segment naming into a single reusable `CustomerSegmentationPipeline` class
10. **Temporal validation** — trained on the first 75% of the timeline, tested on the last 25% to check generalization on unseen future data
11. **Drift monitoring** — added Population Stability Index (PSI) checks to flag when the model needs retraining
12. **Dashboard app** — built an interactive Streamlit app so non-technical users can upload data and get actionable segments without touching code

## Results — The 4 Segments

| Segment | Recency | Frequency | Avg. Spend | % of Customers | % of Revenue |
|---|---|---|---|---|---|
| 🏆 VIP / Champions | 12 days | 13.7 orders | £8,004 | 16.2% | 64.4% |
| 🔁 Loyal / Regulars | 69 days | 4.1 orders | £1,792 | 27.1% | 24.1% |
| 🌱 New / Promising | 19 days | 2.1 orders | £525 | 19.6% | 5.1% |
| ⚠️ At Risk / Lapsed | 185 days | 1.3 orders | £347 | 37.0% | 6.4% |

## Model Validation

- **Silhouette Score:** 0.337 (chosen K=4 via elbow method + silhouette comparison across K=2–10)
- **Temporal validation:** silhouette on a held-out future time period (last 25% of the timeline, unseen during training) dropped to 0.139 — an honest signal that segments partially depend on the time window trained on, documented in the notebook rather than hidden
- **Drift monitoring:** implemented via Population Stability Index (PSI), with a documented methodology caveat about comparing RFM windows of different lengths

## Dashboard App

An interactive Streamlit dashboard for non-technical users:
- Upload a transaction export (.csv/.xlsx) and get instant segment assignments
- KPI overview, segment distribution charts, revenue share breakdown
- Filterable/searchable customer explorer with **CSV export** — ready to hand to a marketing campaign
- Model Health tab with automated drift detection (PSI) and cluster separation quality
- Built-in "About This App" explainer for non-technical stakeholders

**To run the dashboard:**
```bash
cd segmentation_app
pip install -r requirements.txt
python train_pipeline.py "Online Retail.xlsx"   # trains & saves the model (one-time)
streamlit run app.py                            # launches the dashboard
```

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (`KMeans`, `StandardScaler`, `PCA`, silhouette/Davies-Bouldin/Calinski-Harabasz metrics)
- Matplotlib (notebook), Plotly (dashboard)
- Streamlit (dashboard app)
- joblib (model persistence)

## Repo Structure

```
├── Customer_Segmentation_RFM_KMeans.ipynb   # full analysis: cleaning -> modeling -> validation
├── Clustering_Theory_Notes.md               # background theory on clustering
├── README.md
└── segmentation_app/
    ├── app.py                               # Streamlit dashboard
    ├── pipeline.py                          # CustomerSegmentationPipeline class (shared logic)
    ├── train_pipeline.py                    # one-time script to train & save the model
    └── requirements.txt
```

*(Note: `Online Retail.xlsx` is not included in this repo — see below to obtain it.)*

## How to Run the Notebook

1. Download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/352/online+retail) and place `Online Retail.xlsx` in the repo root
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn openpyxl joblib
   ```
3. Run the notebook top to bottom

## Limitations

- Single retailer, single ~1-year window — segment definitions (e.g. what "recent" means) are specific to this business's purchase cadence and wouldn't transfer as-is to, say, a subscription business
- K-Means assumes roughly spherical, similarly-sized clusters; real customer behavior is a continuum, not neatly separated groups
- No demographic or product-category data — segments are purely behavioral
- Drift monitoring compares RFM windows of different lengths in the temporal validation exercise (documented in the notebook); a production deployment should compare fixed-length lookback windows instead
- The dashboard is a batch-scoring tool, not a real-time system — segments update when new data is uploaded, not on every transaction

## Possible Extensions

- Compare against DBSCAN or hierarchical clustering
- Add features beyond RFM (average order value, product category diversity)
- Recompute segments on a rolling, fixed-length window and track customers moving between segments over time
- Automate the retrain trigger (e.g. auto-refit when PSI > 0.25)
- Add authentication and scheduled data pulls for a true production deployment
