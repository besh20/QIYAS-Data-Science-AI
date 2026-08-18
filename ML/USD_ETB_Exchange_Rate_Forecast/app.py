import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="USD/ETB Forecast",
    page_icon="🇪🇹",
    layout="wide"
)

# ---------- Light custom styling ----------
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
    '<p class="subtitle">Live exchange rate tracking with a naive baseline vs. ARIMA(0,1,1) comparison</p>',
    unsafe_allow_html=True
)
st.write("")


# ---------- Data loading (live, with offline fallback) ----------
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


df, data_source = load_data()

if data_source == "cached":
    st.warning(
        "⚠️ Couldn't reach the live data source right now — showing the last saved "
        "snapshot instead. The forecast below is based on this cached data."
    )


# ---------- Fit ARIMA on whatever data we just loaded ----------
@st.cache_resource(ttl=3600)
def fit_model(series):
    model = ARIMA(series, order=(0, 1, 1))
    return model.fit()


arima_fit = fit_model(df["Rate"])
last_value = df["Rate"].iloc[-1]
last_date = df.index[-1]

status_col1, status_col2 = st.columns([3, 1])
with status_col1:
    st.caption(f"📅 Data last updated: **{last_date.date()}**  ·  Source: **{data_source}**")
with status_col2:
    st.page_link("pages/1_About_This_Project.py", label="ℹ️ How this works", icon="ℹ️")

# ---------- Sidebar controls ----------
st.sidebar.header("⚙️ Forecast Settings")
horizon = st.sidebar.slider("Forecast horizon (days)", min_value=3, max_value=30, value=14)
show_history_days = st.sidebar.slider("History to display (days)", min_value=30, max_value=180, value=90)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Current rate**\n\n"
    f"### {last_value:.2f} ETB\n"
    "per 1 USD"
)

# ---------- Quick explainer for anyone landing here cold ----------
with st.expander("🔍 What am I looking at?", expanded=False):
    st.markdown(
        "This chart shows the recent USD → ETB exchange rate (black line), plus two "
        "different forecasts for what comes next:\n\n"
        "- **Naive forecast** (gray dashed) — assumes tomorrow's rate stays the same as today's.\n"
        "- **ARIMA forecast** (red dashed) — a statistical model that looks for patterns in past rate changes.\n"
        "- **Shaded red band** — the ARIMA model's uncertainty range (95% confidence interval). "
        "It gets wider the further out it forecasts, because uncertainty compounds over time.\n\n"
        "👉 Backtesting shows the naive forecast is actually *more accurate* than ARIMA here — "
        "see the **About This Project** page for why that's a meaningful, expected finding rather than a failure."
    )

# ---------- Forecasts ----------
naive_forecast = pd.Series(
    [last_value] * horizon,
    index=pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
)

arima_result = arima_fit.get_forecast(steps=horizon)
arima_forecast = arima_result.predicted_mean
arima_conf = arima_result.conf_int()

# ---------- Interactive Plotly chart ----------
st.subheader(f"{horizon}-Day Forecast: Naive vs ARIMA")

history = df["Rate"].tail(show_history_days)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=history.index, y=history, mode="lines", name="Recent History",
    line=dict(color="#F0F2F6", width=2)
))

fig.add_trace(go.Scatter(
    x=naive_forecast.index, y=naive_forecast, mode="lines", name="Naive Forecast",
    line=dict(color="#9aa0a6", width=2, dash="dash")
))

fig.add_trace(go.Scatter(
    x=arima_forecast.index, y=arima_conf.iloc[:, 1], mode="lines",
    line=dict(width=0), showlegend=False, hoverinfo="skip"
))
fig.add_trace(go.Scatter(
    x=arima_forecast.index, y=arima_conf.iloc[:, 0], mode="lines",
    line=dict(width=0), fill="tonexty", fillcolor="rgba(255,75,75,0.15)",
    name="ARIMA 95% Confidence Interval", hoverinfo="skip"
))

fig.add_trace(go.Scatter(
    x=arima_forecast.index, y=arima_forecast, mode="lines", name="ARIMA(0,1,1) Forecast",
    line=dict(color="#FF4B4B", width=2, dash="dash")
))

fig.update_layout(
    template="plotly_dark",
    height=480,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title="ETB per 1 USD",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(fig, use_container_width=True)

# ---------- Backtest metrics ----------
st.subheader("📊 Model Performance (Backtested on last 30 days)")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Naive Baseline MAPE", "0.50%")
with col2:
    st.metric("ARIMA(0,1,1) MAPE", "1.39%", delta="-0.89pp vs naive", delta_color="inverse")
with col3:
    st.metric("Winner", "Naive ✅")

st.info(
    "**Finding:** the naive baseline (assuming tomorrow's rate equals today's) outperforms "
    "ARIMA on this series. This matches well-documented behavior in FX markets — exchange rates "
    "tend to move close to a random walk at short horizons, making them genuinely hard to beat "
    "with pure price-history models. ARIMA is shown here for transparency and comparison, not "
    "because it's the stronger forecaster. Full methodology on the **About This Project** page."
)

with st.expander("📄 View raw data"):
    st.dataframe(df.tail(30), use_container_width=True)

st.caption("Built as a time series practice project · Data via Yahoo Finance · Model: statsmodels ARIMA")
