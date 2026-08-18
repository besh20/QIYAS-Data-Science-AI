import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="USD/ETB Forecast", layout="wide")
st.title("🇪🇹 USD/ETB Exchange Rate Forecast")
st.caption("Live daily rate + naive baseline vs ARIMA(0,1,1) short-horizon forecast")


# --- Step 1: Try to fetch LIVE data. Falls back to the saved snapshot if the ---
# --- live source is unavailable (keeps the app working even if Yahoo is down) ---
@st.cache_data(ttl=3600)  # refresh once per hour — avoids hammering the API on every click
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
        # Fallback: last known good snapshot bundled with the app
        df = pd.read_pickle("history.pkl")
        return df, "cached"


df, data_source = load_data()

if data_source == "cached":
    st.warning(
        "⚠️ Could not reach the live data source right now — showing the last saved "
        "snapshot instead. Forecasts below are based on this cached data."
    )


# --- Step 2: Fit ARIMA on whatever data we just loaded (live or fallback) ---
# Cached too, so it only refits when the underlying data actually changes (once per hour)
@st.cache_resource(ttl=3600)
def fit_model(series):
    model = ARIMA(series, order=(0, 1, 1))
    fit = model.fit()
    return fit


arima_fit = fit_model(df["Rate"])
last_value = df["Rate"].iloc[-1]
last_date = df.index[-1]

st.caption(f"Data last updated: {last_date.date()} ({data_source} source)")

# --- Sidebar controls ---
st.sidebar.header("Forecast Settings")
horizon = st.sidebar.slider("Forecast horizon (days)", min_value=3, max_value=30, value=14)
show_history_days = st.sidebar.slider("History to display (days)", min_value=30, max_value=180, value=90)

# --- Naive forecast: flat line at the last known rate ---
naive_forecast = pd.Series(
    [last_value] * horizon,
    index=pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
)

# --- ARIMA forecast with confidence interval ---
arima_result = arima_fit.get_forecast(steps=horizon)
arima_forecast = arima_result.predicted_mean
arima_conf = arima_result.conf_int()

# --- Main chart ---
st.subheader(f"{horizon}-Day Forecast: Naive vs ARIMA")

fig, ax = plt.subplots(figsize=(12, 5))

history = df["Rate"].tail(show_history_days)
ax.plot(history.index, history, label="Recent History", color="black")

ax.plot(naive_forecast.index, naive_forecast, label="Naive Forecast (today=tomorrow)",
        linestyle="--", color="gray")

ax.plot(arima_forecast.index, arima_forecast, label="ARIMA(0,1,1) Forecast",
        linestyle="--", color="red")
ax.fill_between(arima_forecast.index, arima_conf.iloc[:, 0], arima_conf.iloc[:, 1],
                 color="red", alpha=0.15, label="ARIMA 95% Confidence Interval")

ax.legend()
ax.set_ylabel("ETB per 1 USD")
st.pyplot(fig)

# --- Backtest metrics (fixed values from notebook evaluation — not recomputed live) ---
st.subheader("Model Performance (Backtested on last 30 days)")

col1, col2 = st.columns(2)
with col1:
    st.metric("Naive Baseline MAPE", "0.50%")
with col2:
    st.metric("ARIMA(0,1,1) MAPE", "1.39%", delta="-0.89pp vs naive", delta_color="inverse")

st.info(
    "📊 **Finding:** The naive baseline (assuming tomorrow's rate equals today's) "
    "outperforms ARIMA on this series. This is consistent with exchange rates "
    "behaving close to a random walk at short horizons — a well-documented pattern "
    "in FX markets. ARIMA is shown here for comparison and transparency, not because "
    "it's the better forecaster."
)

with st.expander("View raw data"):
    st.dataframe(df.tail(30))