import streamlit as st

st.set_page_config(page_title="About · USD/ETB Forecast", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About This Project")
st.caption("What this app does, how it was built, and why the result is what it is")

st.markdown("""
### What this is

A short-horizon forecasting tool for the **USD/ETB exchange rate**, built to demonstrate a
complete, honest time series workflow — from raw data to a deployed, live-updating app.

### Why USD/ETB

The Ethiopian Birr has been depreciating steadily against the US Dollar, making short-term
rate forecasts practically useful for budgeting and planning. It's also a genuinely hard,
real-world dataset — noisy and non-stationary — rather than a clean textbook example.

---

### Methodology

1. **Data collection** — daily rates pulled live from Yahoo Finance, with a cached fallback
   snapshot in case the live source is ever unavailable.
2. **Stationarity testing** — the raw series was tested with the Augmented Dickey-Fuller (ADF)
   test and found non-stationary (as expected, given the strong long-term trend).
3. **Transformation** — log-transforming and differencing the series once made it stationary,
   confirmed by a follow-up ADF test.
4. **Seasonality check** — decomposition showed no meaningful repeating seasonal pattern
   (FX rates are driven by economics and policy, not the calendar), so a seasonal model
   (SARIMA) wasn't necessary — plain ARIMA was the right-sized tool.
5. **Model selection** — a grid search was run across several `(p, d, q)` combinations.
   `ARIMA(0, 1, 1)` came out as the best performer.
6. **Baseline comparison** — critically, a **naive baseline** ("tomorrow's rate = today's rate")
   was tested alongside ARIMA, to check whether the model was actually adding value —
   not just assumed to be useful because it's a "proper" statistical model.

### The finding

The naive baseline outperformed every ARIMA configuration tested (**0.50% MAPE vs. 1.39%
MAPE** for ARIMA). This is a well-documented property of financial time series:
exchange rates behave close to a **random walk** at short horizons, meaning there's very
little exploitable pattern in past prices alone for a model like ARIMA to find.

This app deliberately shows both forecasts side by side, instead of hiding the weaker one,
because reporting that finding honestly is more valuable — and more true to good data
science practice — than presenting a misleading "the model works" result.

### Limitations

- Forecasts rely only on past price history — no external factors like inflation, interest
  rate differentials, remittance flows, or policy announcements are included.
- Accuracy drops quickly beyond a few days; longer-horizon forecasts shown here should be
  treated as illustrative, not reliable predictions.
- Free data sources for USD/ETB can occasionally be unreliable — hence the offline fallback.

### Possible next steps

- Add external economic indicators to see if a richer model can meaningfully beat the naive
  baseline.
- Compare against additional methods (exponential smoothing, gradient boosting on lag features).
- Track live forecast accuracy over time instead of a single backtest snapshot.

---

### Tech stack

Python · pandas · statsmodels (ARIMA, ADF test, decomposition) · scikit-learn (evaluation) ·
yfinance (data) · Plotly (charts) · Streamlit (app + deployment)

### Source

Full analysis notebook (data exploration, ADF tests, ACF/PACF plots, grid search) is included
in the project repository alongside this app's source code.
""")
