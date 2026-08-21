import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

from models import (
    naive_flat_forecast,
    naive_drift_forecast,
    arima_drift_forecast,
    sarimax_exog_forecast,
    run_backtest,
)

st.set_page_config(page_title="USD/ETB Forecast", page_icon="🇪🇹", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .subtitle { color: #9aa0a6; font-size: 1rem; margin-top: 0; }
    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🇪🇹 USD/ETB Exchange Rate Forecast</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Live rate tracking with trend-aware forecasting, validated across multiple horizons</p>',
    unsafe_allow_html=True
)
st.write("")


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=3600)
def load_data():
    try:
        data = yf.download("USDETB=X", start="2024-08-01", progress=False, multi_level_index=False)
        df = data[["Close"]].rename(columns={"Close": "Rate"})
        df = df.asfreq("D")
        df["Rate"] = df["Rate"].ffill()
        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("Live fetch returned no data")
        return df, "live"
    except Exception:
        df = pd.read_pickle("history.pkl")
        return df, "cached"


@st.cache_data(ttl=3600)
def load_dxy():
    """US Dollar Index — used as an optional external factor."""
    try:
        data = yf.download("DX-Y.NYB", start="2024-08-01", progress=False, multi_level_index=False)
        s = data["Close"].asfreq("D").ffill().dropna()
        s.name = "DXY"
        return s
    except Exception:
        return None


@st.cache_data(ttl=21600)  # backtest is a bit expensive — refresh every 6 hours, not every reload
def cached_backtest(series):
    return run_backtest(series)


df, data_source = load_data()

if data_source == "cached":
    st.warning(
        "⚠️ Couldn't reach the live data source right now — showing the last saved "
        "snapshot instead. Forecasts below are based on this cached data."
    )

series = df["Rate"]
last_value = series.iloc[-1]
last_date = series.index[-1]

backtest_df = cached_backtest(series)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Forecast Settings")
horizon = st.sidebar.select_slider(
    "Forecast horizon (days)", options=[7, 14, 30], value=14,
    help="Matches the horizons validated in the backtest below, so the 'recommended model' "
         "callout is always based on real evidence at your selected horizon."
)
show_history_days = st.sidebar.slider("History to display (days)", min_value=30, max_value=180, value=90)

st.sidebar.markdown("---")
show_exog = st.sidebar.checkbox(
    "Include experimental model (USD Index as external factor)",
    value=False,
    help="Adds a SARIMAX model using the US Dollar Index (DXY) as a predictor. "
         "Marked experimental because future DXY is itself estimated, not known."
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current rate**\n\n### {last_value:.2f} ETB\nper 1 USD")

status_col1, status_col2 = st.columns([3, 1])
with status_col1:
    st.caption(f"📅 Data last updated: **{last_date.date()}**  ·  Source: **{data_source}**")
with status_col2:
    st.page_link("pages/1_About_This_Project.py", label="ℹ️ How this works", icon="ℹ️")

with st.expander("🔍 What am I looking at?", expanded=False):
    st.markdown(
        "This app forecasts the USD → ETB rate using three models, all validated on real "
        "historical data before being trusted:\n\n"
        "- **Naive (flat)** — assumes the rate never changes. Simple, but blind to trend.\n"
        "- **Naive (drift)** — extends the recent average daily change forward. Simple *and* trend-aware.\n"
        "- **ARIMA (drift)** — a statistical model that also accounts for trend, plus its own "
        "short-term pattern detection.\n\n"
        "👉 See the **Model Accuracy by Horizon** section below — the drift-aware models "
        "meaningfully outperform the flat naive forecast at every horizon beyond 1 day, "
        "which makes sense given the Birr's steady depreciation trend."
    )


# ============================================================
# RECOMMENDED MODEL (based on real backtest evidence, not a guess)
# ============================================================
recommended_model = backtest_df.loc[horizon].idxmin()
recommended_error = backtest_df.loc[horizon].min()

st.success(
    f"✅ **Recommended model for {horizon}-day forecasts: {recommended_model}** "
    f"— lowest backtested error at this horizon ({recommended_error:.2f}% MAPE)."
)


# ============================================================
# GENERATE FORECASTS FOR DISPLAY
# ============================================================
nf = naive_flat_forecast(series, horizon)
nd = naive_drift_forecast(series, horizon)
ad, ad_conf = arima_drift_forecast(series, horizon)

forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

exog_forecast = None
exog_available = False
if show_exog:
    dxy = load_dxy()
    if dxy is not None:
        aligned = pd.concat([series, dxy], axis=1).dropna()
        if len(aligned) > 60:
            try:
                exog_forecast, exog_conf = sarimax_exog_forecast(
                    aligned["Rate"], aligned["DXY"], horizon
                )
                exog_available = True
            except Exception:
                exog_available = False
    if not exog_available:
        st.sidebar.warning("Experimental model unavailable right now (data or fit issue) — showing core models only.")


# ============================================================
# CHART
# ============================================================
st.subheader(f"{horizon}-Day Forecast")

history = series.tail(show_history_days)

# Give the recommended model a thicker line so it visually stands out
line_widths = {"Naive (flat)": 2, "Naive (drift)": 2, "ARIMA (drift)": 2}
line_widths[recommended_model] = 4

fig = go.Figure()

fig.add_trace(go.Scatter(x=history.index, y=history, mode="lines", name="Recent History",
                          line=dict(color="#F0F2F6", width=2)))

fig.add_trace(go.Scatter(x=forecast_dates, y=nf, mode="lines", name="Naive (flat)",
                          line=dict(color="#9aa0a6", width=line_widths["Naive (flat)"], dash="dot")))

fig.add_trace(go.Scatter(x=forecast_dates, y=nd, mode="lines", name="Naive (drift)",
                          line=dict(color="#4C9AFF", width=line_widths["Naive (drift)"], dash="dash")))

fig.add_trace(go.Scatter(x=forecast_dates, y=ad_conf[:, 1], mode="lines",
                          line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=forecast_dates, y=ad_conf[:, 0], mode="lines",
                          line=dict(width=0), fill="tonexty", fillcolor="rgba(255,75,75,0.15)",
                          name="ARIMA 95% CI", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=forecast_dates, y=ad, mode="lines", name="ARIMA (drift)",
                          line=dict(color="#FF4B4B", width=line_widths["ARIMA (drift)"], dash="dash")))

if exog_available:
    fig.add_trace(go.Scatter(x=forecast_dates, y=exog_forecast, mode="lines",
                              name="SARIMAX + USD Index (experimental)",
                              line=dict(color="#F5C542", width=2, dash="dashdot")))

fig.update_layout(
    template="plotly_dark", height=480, margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title="ETB per 1 USD",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# EXPLICIT FORECAST TABLE + CSV EXPORT
# ============================================================
st.subheader("📋 Forecasted Values")

table_data = {
    "Date": forecast_dates.date,
    "Naive (flat)": np.round(nf, 2),
    "Naive (drift)": np.round(nd, 2),
    "ARIMA (drift)": np.round(ad, 2),
}
if exog_available:
    table_data["SARIMAX + USD Index"] = np.round(exog_forecast, 2)

forecast_table = pd.DataFrame(table_data)

checkpoints = [d for d in (7, 14, 30) if d <= horizon]
if checkpoints:
    cp_cols = st.columns(len(checkpoints))
    for col, day in zip(cp_cols, checkpoints):
        with col:
            st.metric(f"Day {day} forecast (ARIMA)", f"{ad[day-1]:.2f} ETB")

table_col, download_col = st.columns([4, 1])
with table_col:
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)
with download_col:
    st.write("")
    st.write("")
    st.download_button(
        "⬇️ Download CSV",
        data=forecast_table.to_csv(index=False),
        file_name=f"usd_etb_forecast_{horizon}d.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# MODEL ACCURACY BY HORIZON (the real validation)
# ============================================================
st.subheader("📊 Model Accuracy by Horizon")
st.caption("Rolling-origin backtest — each model re-tested from multiple starting points in the past, "
           "then measured at 1, 7, 14, and 30 days out. This is a fair, like-for-like comparison.")

bt_display = backtest_df.copy()
bt_display.index.name = "Horizon (days)"
bt_display = bt_display.round(2).astype(str) + "%"
st.dataframe(bt_display, use_container_width=True)

st.info(
    f"**Finding:** flat naive forecasting looks competitive at very short horizons (1 day), but "
    f"falls behind at longer horizons because it ignores the Birr's ongoing depreciation trend. "
    f"**{recommended_model}** performs best at the {horizon}-day horizon in this backtest. Full "
    f"methodology, including how an earlier version of this evaluation was flawed and corrected, "
    f"is on the **About This Project** page."
)

with st.expander("📄 View raw historical data"):
    st.dataframe(df.tail(30), use_container_width=True)

st.caption("Built as a time series practice project · Data via Yahoo Finance · Models: statsmodels ARIMA/SARIMAX")
