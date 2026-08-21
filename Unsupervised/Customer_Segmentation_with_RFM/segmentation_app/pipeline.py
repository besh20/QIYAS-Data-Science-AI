"""
CustomerSegmentationPipeline
-----------------------------
The core RFM + K-Means logic behind the segmentation dashboard.

This is the same pipeline built and validated in the analysis notebook,
moved into its own module so both the notebook and the Streamlit app can
import and use the exact same logic (no copy-pasted drift between them).
"""

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Force pandas to use its classic (non-PyArrow-backed) string handling.
# Recent pandas versions default to PyArrow-backed strings, which has a
# known Windows bug causing a native crash ("access violation") inside
# pd.to_datetime() on some pandas/pyarrow version combinations. This line
# avoids that buggy code path entirely.
pd.set_option("future.infer_string", False)

REQUIRED_COLUMNS = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]

NON_PRODUCT_CODES = ["POST", "DOT", "M", "D", "S", "AMAZONFEE", "CRUK", "PADS", "m", "B"]


_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",   # e.g. 2010-12-01 08:26:00 (CSV round-trip format)
    "%m/%d/%Y %H:%M",      # e.g. 12/1/2010 8:26 (original UCI xlsx format)
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d",
)


def _parse_date_safe(value):
    """
    Parse a single date value using pure-Python datetime.strptime, trying
    each known format in turn. Returns pd.NaT if nothing matches or the
    value is missing.

    This deliberately avoids pandas' pd.to_datetime() string-inference path,
    which on some pandas/pyarrow/Windows combinations internally calls into
    PyArrow's compiled C++ library and crashes with a native "access
    violation" -- a binary compatibility issue (e.g. missing CPU
    instructions) that no amount of dtype-juggling in pandas can avoid,
    since it happens regardless of the input's declared dtype. Pure Python
    string parsing never touches PyArrow at all, so it's immune to this.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime)):
        return value
    s = str(value).strip()
    if not s:
        return pd.NaT
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return pd.NaT


def _de_arrow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any PyArrow-backed columns to plain numpy/object dtypes.

    Recent pandas versions can default to PyArrow-backed string columns.
    On some pandas/pyarrow/Windows combinations, calling .map() or
    .astype("object") on those columns triggers a native crash ("access
    violation") -- and pd.to_datetime() calls .map() internally on
    Arrow-backed input. to_numpy() sidesteps that crashing code path
    entirely (it goes through pyarrow's own C-level conversion instead),
    so we use it here to strip Arrow-backing from every column up front,
    regardless of what read the data in or what pandas defaults are active.
    """
    df = df.copy()
    for col in df.columns:
        dtype_str = str(df[col].dtype).lower()
        if "pyarrow" in dtype_str or "arrow" in dtype_str:
            df[col] = df[col].to_numpy(dtype=object)
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same cleaning rules used in the analysis notebook to raw
    transaction data, so the app can accept a raw export directly.
    """
    df = _de_arrow(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing required column(s): {', '.join(missing)}")

    parsed_dates = [_parse_date_safe(v) for v in df["InvoiceDate"].tolist()]
    df["InvoiceDate"] = pd.Series(parsed_dates, index=df.index, dtype="datetime64[ns]")
    df = df.dropna(subset=["InvoiceDate"])  # drop any rows where the date couldn't be parsed
    df = df.dropna(subset=["CustomerID"])
    df = df.drop_duplicates()
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["CustomerID"] = df["CustomerID"].astype(int)

    return df


def population_stability_index(expected, actual, buckets=10):
    """
    Measures how much a distribution ('actual') has shifted compared to a
    baseline ('expected'), by binning both into the same percentile buckets
    and comparing bucket proportions.

    PSI < 0.10  -> no significant shift
    PSI 0.10-0.25 -> moderate shift, worth monitoring
    PSI > 0.25  -> significant shift, consider retraining
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    breakpoints = np.linspace(0, 100, buckets + 1)
    bucket_edges = np.percentile(expected, breakpoints)
    bucket_edges[0] -= 1e-6
    bucket_edges[-1] += 1e-6

    expected_pct = np.histogram(expected, bins=bucket_edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bucket_edges)[0] / len(actual)

    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


class CustomerSegmentationPipeline:
    """
    End-to-end customer segmentation pipeline: raw transactions -> RFM ->
    scaled features -> K-Means cluster -> stable, human-readable segment name.
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.feature_cols = ["Recency_log", "Frequency_log", "Monetary_log"]
        self.segment_names = {}
        self.snapshot_date_ = None
        self.baseline_rfm_ = None  # stored at fit time, used later for drift checks

    def _compute_rfm(self, df: pd.DataFrame, snapshot_date) -> pd.DataFrame:
        rfm = df.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        ).reset_index()
        rfm["Recency_log"] = np.log1p(rfm["Recency"])
        rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
        rfm["Monetary_log"] = np.log1p(rfm["Monetary"])
        return rfm

    def _assign_segment_names(self, rfm_with_clusters: pd.DataFrame) -> dict:
        """
        Rank clusters by a composite RFM score so segment names stay stable
        and meaningful even if the model is retrained and cluster numbers
        shuffle around.
        """
        profile = rfm_with_clusters.groupby("Cluster").agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
        )
        recency_rank = profile["Recency"].rank(ascending=True)
        freq_rank = profile["Frequency"].rank(ascending=False)
        monetary_rank = profile["Monetary"].rank(ascending=False)
        composite_score = recency_rank + freq_rank + monetary_rank
        ordered_clusters = composite_score.sort_values().index.tolist()

        n = len(ordered_clusters)
        if n == 4:
            names = ["VIP / Champions", "Loyal / Regulars", "New / Promising", "At Risk / Lapsed"]
        else:
            names = [f"Segment {i + 1}" for i in range(n)]
        return dict(zip(ordered_clusters, names))

    def fit(self, df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
        if snapshot_date is None:
            snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        self.snapshot_date_ = snapshot_date

        rfm = self._compute_rfm(df, snapshot_date)
        X = self.scaler.fit_transform(rfm[self.feature_cols])
        rfm["Cluster"] = self.kmeans.fit_predict(X)
        self.segment_names = self._assign_segment_names(rfm)
        rfm["Segment"] = rfm["Cluster"].map(self.segment_names)

        self.baseline_rfm_ = rfm[["Recency", "Frequency", "Monetary"]].copy()
        return rfm

    def predict(self, df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
        if snapshot_date is None:
            snapshot_date = self.snapshot_date_

        rfm = self._compute_rfm(df, snapshot_date)
        X = self.scaler.transform(rfm[self.feature_cols])
        rfm["Cluster"] = self.kmeans.predict(X)
        rfm["Segment"] = rfm["Cluster"].map(self.segment_names)
        return rfm

    def drift_report(self, new_rfm: pd.DataFrame) -> pd.DataFrame:
        """Compare a new RFM table's distributions against the training baseline."""
        rows = []
        for col in ["Recency", "Frequency", "Monetary"]:
            psi = population_stability_index(self.baseline_rfm_[col], new_rfm[col])
            if psi > 0.25:
                status = "Retrain recommended"
            elif psi > 0.10:
                status = "Monitor"
            else:
                status = "Stable"
            rows.append({"Feature": col, "PSI": round(psi, 3), "Status": status})
        return pd.DataFrame(rows)

    def save(self, path="segmentation_pipeline.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="segmentation_pipeline.pkl"):
        return joblib.load(path)