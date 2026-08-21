"""
Customer Segmentation Dashboard
--------------------------------
A Streamlit app for scoring customers into RFM-based behavioral segments
(VIP / Loyal / New / At Risk) using a pre-trained K-Means pipeline, built
for non-technical users (marketing/ops) to upload transaction data and get
an actionable, exportable customer list back -- no code required.

Run with:  streamlit run app.py
"""

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from pipeline import CustomerSegmentationPipeline, clean_transactions

MODEL_PATH = "segmentation_pipeline.pkl"

SEGMENT_COLORS = {
    "VIP / Champions": "#7c3aed",
    "Loyal / Regulars": "#2563eb",
    "New / Promising": "#0ea5e9",
    "At Risk / Lapsed": "#f97316",
}

SEGMENT_DESCRIPTIONS = {
    "VIP / Champions": "Buy often, spend the most, purchased recently. Your most valuable customers.",
    "Loyal / Regulars": "Consistent repeat buyers with solid spend. The backbone of steady revenue.",
    "New / Promising": "Recent buyers who haven't built up purchase history yet. Early days -- worth nurturing.",
    "At Risk / Lapsed": "Haven't purchased in a long time. Prime candidates for a win-back campaign.",
}

SEGMENT_ACTIONS = {
    "VIP / Champions": "Loyalty rewards, early access to new products, personal outreach.",
    "Loyal / Regulars": "Cross-sell and upsell campaigns, bundle offers.",
    "New / Promising": "Onboarding sequences, encourage a second purchase with a small incentive.",
    "At Risk / Lapsed": "Win-back discount codes, re-engagement email campaigns, or a customer satisfaction check-in.",
}

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="\U0001F6CD\uFE0F",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Minimal styling polish
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background-color: rgba(124, 58, 237, 0.06);
        border: 1px solid rgba(124, 58, 237, 0.15);
        border-radius: 12px;
        padding: 14px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if os.path.exists(path):
        return CustomerSegmentationPipeline.load(path)
    return None


@st.cache_data(show_spinner=False)
def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def segment_order(names):
    """Sort segment names in a consistent, intuitive VIP-to-lapsed order."""
    priority = ["VIP / Champions", "Loyal / Regulars", "New / Promising", "At Risk / Lapsed"]
    return sorted(names, key=lambda n: priority.index(n) if n in priority else 99)


def kpi_row(rfm: pd.DataFrame):
    total_customers = len(rfm)
    total_revenue = rfm["Monetary"].sum()
    avg_order_value = (rfm["Monetary"] / rfm["Frequency"]).mean()
    top_segment = rfm.groupby("Segment")["Monetary"].sum().idxmax()
    top_segment_share = rfm.groupby("Segment")["Monetary"].sum().max() / total_revenue * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Total Revenue", f"£{total_revenue:,.0f}")
    c3.metric("Avg. Order Value", f"£{avg_order_value:,.2f}")
    c4.metric(f"Revenue from {top_segment}", f"{top_segment_share:.1f}%")


def render_dashboard(rfm: pd.DataFrame):
    st.subheader("Overview")
    kpi_row(rfm)
    st.write("")

    seg_counts = rfm["Segment"].value_counts().reindex(segment_order(rfm["Segment"].unique()))
    seg_revenue = rfm.groupby("Segment")["Monetary"].sum().reindex(seg_counts.index)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            x=seg_counts.index,
            y=seg_counts.values,
            color=seg_counts.index,
            color_discrete_map=SEGMENT_COLORS,
            labels={"x": "Segment", "y": "Number of Customers"},
            title="Customers per Segment",
        )
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            names=seg_revenue.index,
            values=seg_revenue.values,
            color=seg_revenue.index,
            color_discrete_map=SEGMENT_COLORS,
            title="Revenue Share by Segment",
            hole=0.45,
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Profiles")
    profile = rfm.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        Avg_Recency_Days=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum"),
    ).reindex(segment_order(rfm["Segment"].unique())).round(1)
    profile["% of Customers"] = (profile["Customers"] / profile["Customers"].sum() * 100).round(1)
    profile["% of Revenue"] = (profile["Total_Revenue"] / profile["Total_Revenue"].sum() * 100).round(1)
    st.dataframe(profile, use_container_width=True)

    for seg in segment_order(rfm["Segment"].unique()):
        with st.expander(f"What does '{seg}' mean, and what should we do about it?"):
            st.write(f"**Who they are:** {SEGMENT_DESCRIPTIONS.get(seg, '')}")
            st.write(f"**Suggested action:** {SEGMENT_ACTIONS.get(seg, '')}")

    st.subheader("Customer Landscape (2D projection)")
    st.caption(
        "Each dot is a customer, positioned by their overall RFM profile and colored by segment. "
        "Customers close together behave similarly."
    )
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features_log = np.log1p(features)
    pca = PCA(n_components=2)
    coords = pca.fit_transform((features_log - features_log.mean()) / features_log.std())
    plot_df = rfm[["CustomerID", "Segment"]].copy()
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]
    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="Segment",
        color_discrete_map=SEGMENT_COLORS, opacity=0.6,
        hover_data=["CustomerID"], height=480,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_customer_explorer(rfm: pd.DataFrame):
    st.subheader("Customer Explorer")
    st.caption("Filter, search, and export customer lists -- ready to hand to a marketing campaign.")

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_segments = st.multiselect(
            "Filter by segment", options=segment_order(rfm["Segment"].unique()),
            default=list(rfm["Segment"].unique()),
        )
    with col2:
        search_id = st.text_input("Search by Customer ID", "")

    filtered = rfm[rfm["Segment"].isin(selected_segments)]
    if search_id.strip():
        filtered = filtered[filtered["CustomerID"].astype(str).str.contains(search_id.strip())]

    sort_col = st.selectbox("Sort by", ["Monetary", "Frequency", "Recency", "CustomerID"], index=0)
    ascending = sort_col == "Recency"
    filtered = filtered.sort_values(sort_col, ascending=ascending)

    st.write(f"Showing **{len(filtered):,}** of {len(rfm):,} customers")
    st.dataframe(
        filtered[["CustomerID", "Segment", "Recency", "Frequency", "Monetary"]],
        use_container_width=True,
        height=420,
    )

    csv_bytes = filtered[["CustomerID", "Segment", "Recency", "Frequency", "Monetary"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "\U0001F4E5 Download this list as CSV",
        data=csv_bytes,
        file_name="customer_segments.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_model_health(pipeline: CustomerSegmentationPipeline, rfm: pd.DataFrame):
    st.subheader("Model Health")
    st.caption(
        "Checks whether the customer base still looks like the data this model was trained on. "
        "Large shifts are a signal the model may need retraining."
    )

    if pipeline.baseline_rfm_ is None:
        st.info("No training baseline stored with this model -- drift check unavailable.")
        return

    drift_df = pipeline.drift_report(rfm)

    def status_badge(status):
        return {"Stable": "\U0001F7E2 Stable", "Monitor": "\U0001F7E1 Monitor", "Retrain recommended": "\U0001F534 Retrain recommended"}.get(status, status)

    drift_df_display = drift_df.copy()
    drift_df_display["Status"] = drift_df_display["Status"].map(status_badge)
    st.dataframe(drift_df_display, use_container_width=True, hide_index=True)

    worst = drift_df["Status"].map({"Stable": 0, "Monitor": 1, "Retrain recommended": 2}).max()
    if worst == 2:
        st.error("One or more features show significant drift. Consider retraining the model on recent data.")
    elif worst == 1:
        st.warning("Some drift detected. Keep an eye on this -- not urgent yet.")
    else:
        st.success("No significant drift detected. Model is healthy.")

    st.divider()
    st.subheader("Cluster Separation Quality")
    try:
        X = pipeline.scaler.transform(
            np.log1p(rfm[["Recency", "Frequency", "Monetary"]]).rename(
                columns={"Recency": "Recency_log", "Frequency": "Frequency_log", "Monetary": "Monetary_log"}
            )
        )
        sil = silhouette_score(X, rfm["Cluster"])
        st.metric("Silhouette Score (this dataset)", f"{sil:.3f}", help="Higher is better, range -1 to 1. Reflects how well-separated the current segments are.")
    except Exception:
        st.info("Silhouette score unavailable for this dataset.")


def render_about():
    st.subheader("About This App")
    st.markdown(
        """
This dashboard segments customers into behavioral groups based on their **purchase history**,
using an approach called **RFM analysis** combined with **K-Means clustering** -- a standard,
widely-used technique in retail and e-commerce analytics.

### What it does
1. You upload a transaction export (one row per order line item)
2. The app calculates three things for every customer:
   - **Recency** -- how many days since their last purchase
   - **Frequency** -- how many separate orders they've placed
   - **Monetary** -- how much they've spent in total
3. A pre-trained machine learning model groups customers into 4 segments based on these patterns
4. You get an interactive dashboard plus a downloadable, ready-to-use customer list per segment

### The segments
| Segment | What it means |
|---|---|
| **VIP / Champions** | Buy often, spend the most, purchased recently |
| **Loyal / Regulars** | Consistent repeat buyers with solid spend |
| **New / Promising** | Recent buyers who haven't built up history yet |
| **At Risk / Lapsed** | Haven't purchased in a long time -- may be churning |

### What this app is (and isn't)
- **It is:** a tool for understanding who your customers are today, and exporting targeted
  lists for marketing campaigns (e.g. "give everyone in At Risk a win-back discount").
- **It isn't:** a real-time system. Segments update whenever you upload fresh data, not
  instantly with every purchase. It also can't say anything about a customer with zero
  purchase history yet.

### Required columns in your upload
Your file needs these columns (standard for most order-export formats):
`InvoiceNo`, `StockCode`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`

### Model Health tab
Flags when the uploaded customer base looks meaningfully different from the data the model
was trained on (using a technique called Population Stability Index), so you know when it's
time to retrain rather than trusting stale segments indefinitely.

---
Built with Python, scikit-learn, and Streamlit. Model: K-Means (K=4) on standardized,
log-transformed RFM features.
"""
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.title("\U0001F6CD\uFE0F Customer Segmentation Dashboard")
st.caption("Upload transaction data, get actionable customer segments -- no code required.")

with st.sidebar:
    st.header("1. Load Data")
    uploaded_file = st.file_uploader("Upload transaction file (.csv or .xlsx)", type=["csv", "xlsx"])

    st.header("2. Model")
    model = load_model(MODEL_PATH)
    mode = st.radio(
        "Mode",
        ["Score with existing model", "Train a new model on this data"],
        index=0 if model is not None else 1,
        help="Use the existing trained model to segment new customers, or train a fresh model on the uploaded data.",
    )

    run_button = st.button("\U0001F680 Run Segmentation", type="primary", use_container_width=True)

    st.divider()
    if model is not None:
        st.success("Trained model found and loaded.")
    else:
        st.warning("No trained model found yet -- train one first.")

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

if "rfm_result" not in st.session_state:
    st.session_state.rfm_result = None
    st.session_state.pipeline_used = None

if run_button:
    if uploaded_file is None:
        st.error("Please upload a transaction file first.")
    else:
        with st.spinner("Cleaning data and computing segments..."):
            try:
                raw = read_uploaded_file(uploaded_file)
                clean_df = clean_transactions(raw)

                if mode == "Train a new model on this data":
                    pipeline = CustomerSegmentationPipeline(n_clusters=4, random_state=42)
                    rfm_result = pipeline.fit(clean_df)
                    pipeline.save(MODEL_PATH)
                    load_model.clear()
                    model = pipeline
                    st.success(f"Trained a new model on {len(rfm_result):,} customers and saved it.")
                else:
                    if model is None:
                        st.error("No existing model found. Switch to 'Train a new model on this data' first.")
                        st.stop()
                    pipeline = model
                    rfm_result = pipeline.predict(clean_df)
                    st.success(f"Scored {len(rfm_result):,} customers using the existing model.")

                st.session_state.rfm_result = rfm_result
                st.session_state.pipeline_used = pipeline
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Something went wrong processing this file: {e}")

tab_dashboard, tab_explorer, tab_health, tab_about = st.tabs(
    ["\U0001F4CA Dashboard", "\U0001F465 Customer Explorer", "\U0001FA7A Model Health", "\u2139\uFE0F About This App"]
)

with tab_dashboard:
    if st.session_state.rfm_result is not None:
        render_dashboard(st.session_state.rfm_result)
    else:
        st.info("\U0001F448 Upload a transaction file and click **Run Segmentation** in the sidebar to get started.")

with tab_explorer:
    if st.session_state.rfm_result is not None:
        render_customer_explorer(st.session_state.rfm_result)
    else:
        st.info("Run segmentation first to explore individual customers.")

with tab_health:
    if st.session_state.rfm_result is not None and st.session_state.pipeline_used is not None:
        render_model_health(st.session_state.pipeline_used, st.session_state.rfm_result)
    else:
        st.info("Run segmentation first to see model health.")

with tab_about:
    render_about()
