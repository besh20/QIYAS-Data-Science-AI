import streamlit as st

st.set_page_config(page_title="About · USD/ETB Forecast", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About This Project")
st.caption("What this app does, how it was validated, and a real mistake I caught and fixed along the way")

st.markdown("""
### What this is

A short-horizon forecasting tool for the **USD/ETB exchange rate** — built to demonstrate a
complete, honest time series workflow: data collection → modeling → rigorous validation →
live deployment.

### Why USD/ETB

The Ethiopian Birr has been depreciating steadily against the US Dollar, making short-term
rate forecasts practically useful for budgeting and planning. It's also a genuinely hard,
real-world dataset — noisy and non-stationary — rather than a clean textbook example.

---

### A mistake I caught, and why it matters

An earlier version of this project's evaluation compared a "naive" forecast against ARIMA and
found naive winning by a wide margin. On closer inspection, that comparison was flawed: the
naive baseline was using each day's *real* value to predict the next day (a 1-day-ahead
recursive check), while ARIMA was making a genuine blind 30-day-ahead forecast from a single
starting point. **Those are two different tasks — not a fair comparison.**

Re-running both models on the *same* task (a real static forecast from one point in time)
showed they perform almost identically when neither accounts for trend. That fix mattered:
it changed the entire conclusion of the project.

### The real finding: trend awareness is what matters

The Birr has a clear, sustained depreciation trend. Any model that ignores it — flat naive,
or ARIMA without a trend term — will underperform on any forecast longer than a day or two.
Adding a trend/drift component to both the naive baseline and ARIMA cut forecast error by
more than half on a real 30-day test.

To validate this properly (not just on one lucky test window), the app runs a
**rolling-origin backtest** — testing each model from several different starting points in
the past, at four different horizons (1, 7, 14, and 30 days) — rather than relying on a
single train/test split. That backtest is what's shown live in the **Model Accuracy by
Horizon** table on the main page.

**Result:** flat naive is competitive only at a 1-day horizon. Beyond that, the drift-aware
models — Naive (drift) and ARIMA (drift) — consistently win, and the gap widens as the
horizon grows. This is the finding the app is actually built around.

### The experimental model

There's an optional toggle to add a SARIMAX model using the **US Dollar Index (DXY)** as an
external factor — the idea being that broad USD strength plausibly relates to Birr pressure.
It's marked experimental deliberately: since future DXY values are unknown, they're
themselves forecasted with a simple drift model before being fed into SARIMAX. That's a real
assumption baked into the model, not a fact, so it shouldn't be trusted as much as the core
models above it.

---

### Limitations

- All core models rely only on price history and trend — no inflation, interest rate
  differentials, remittance flows, or policy announcements are directly modeled (aside from
  the experimental USD Index add-on).
- Forecast uncertainty grows with horizon — the shaded confidence band on the chart reflects
  this, and longer-horizon forecasts should be treated as directional, not exact.
- Free data sources for USD/ETB can occasionally be unreliable — hence the offline fallback
  snapshot.

### Possible next steps

- Replace the drift-forecasted exogenous input with a real forward-looking macro indicator
  (e.g. interest rate differentials) where available.
- Track live forecast accuracy over time as new data comes in, instead of only backtesting
  on historical windows.
- Add more comparison models (e.g. exponential smoothing with trend) to the accuracy table.

---

### Tech stack

Python · pandas · statsmodels (ARIMA, SARIMAX) · scikit-learn (evaluation) · yfinance (data) ·
Plotly (charts) · Streamlit (app + deployment)

### Source

The full analysis notebook — including the original flawed evaluation, the diagnosis, and the
corrected methodology — is included in the project repository alongside this app's source code.
""")