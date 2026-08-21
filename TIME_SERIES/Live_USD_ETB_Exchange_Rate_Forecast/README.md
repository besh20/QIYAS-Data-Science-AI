# 🇪🇹 USD/ETB Exchange Rate Forecast

A **live**, trend-aware forecasting app for the USD/ETB exchange rate — built to demonstrate a
complete, honest time series workflow: data collection → modeling → rigorous validation →
tested, deployed software. Includes a real methodological mistake that was caught, diagnosed,
and fixed along the way — documented rather than hidden.

🔗 **Live app:** [add your Streamlit Cloud URL here]

---

## What it does

- Pulls **live** USD/ETB exchange rate data on every session (cached hourly)
- Forecasts 7, 14, or 30 days ahead using three validated models
- Backtests all models live, across multiple time horizons, and tells you which one to trust
- Shows real forecasted numbers in a downloadable table — not just a chart
- Includes an optional experimental model using the US Dollar Index as an external factor
- Falls back gracefully to a cached snapshot if the live data source is ever unreachable

## Why USD/ETB

The Ethiopian Birr has been depreciating steadily against the US Dollar, making short-term
rate forecasts practically useful for budgeting and planning. It's also a genuinely hard,
real-world dataset — noisy and non-stationary — rather than a clean textbook example.

---

## The story: a mistake worth telling

An early version of this project compared a naive forecast against ARIMA and found naive
winning by a wide margin. Digging in, the comparison was flawed: the naive baseline was using
each day's *real* value to predict the next (an easy 1-day-ahead check), while ARIMA was
making a genuine blind 30-day-ahead forecast. Those aren't the same task.

Fixing the evaluation to compare both models fairly showed they perform almost identically —
**when neither accounts for trend.** That led to the real insight this project is built around:

> The Birr has a clear depreciation trend. Any model that ignores it — including "naive" —
> underperforms on any forecast longer than a day or two.

Adding trend/drift to both the naive baseline and ARIMA more than halved forecast error on a
real 30-day test, confirmed across multiple time windows with a rolling-origin backtest —
not just one lucky test split.

Full derivation — ADF tests, ACF/PACF plots, the original flawed comparison, the diagnosis,
and the fix — is in the analysis notebook. The app shows the conclusions live; the notebook
shows the reasoning behind them.

---

## Live model comparison (what the app actually validates)

| Horizon | Naive (flat) | Naive (drift) | ARIMA (drift) |
|---|---|---|---|
| 1 day   | ~0.6% | ~0.6% | ~0.7% |
| 7 days  | ~1.1% | ~1.0% | **~0.96%** |
| 14 days | ~1.0% | ~0.87% | **~0.87%** |
| 30 days | ~1.2% | **~0.94%** | ~0.96% |

*(MAPE, lower is better — these are recomputed live in-app, so exact numbers shift slightly
as new data comes in. The app always highlights the current best model per horizon.)*

---

## Project structure

```
USD_ETB_Exchange_Rate_Forecast/
├── app.py                          # Streamlit UI — imports model logic, handles display
├── models.py                       # Pure forecasting logic — no Streamlit dependency
├── conftest.py                     # Lets pytest find models.py from anywhere in the project
├── tests/
│   └── test_models.py              # Automated tests for the forecasting logic
├── pages/
│   └── 1_About_This_Project.py     # In-app methodology & findings writeup
├── history.pkl                     # Offline fallback data snapshot
├── requirements.txt
├── USD_ETB Forecast.ipynb          # Full analysis notebook (evidence & derivation, not runtime)
└── README.md
```

## Tech stack

Python · pandas, numpy · statsmodels (ARIMA, SARIMAX, ADF test, decomposition) ·
scikit-learn (evaluation) · yfinance (live data) · Plotly (interactive charts) ·
Streamlit (app + deployment) · pytest (testing)

---

## Running it locally

```bash
pip install -r requirements.txt

# Run the test suite first — validates the model logic independent of the UI
pytest tests/

# Then run the app
streamlit run app.py
```

## Deploying

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app," select the repo, and set the main file path to `app.py`.
4. Streamlit Cloud builds and deploys automatically, including the `pages/` folder for
   in-app navigation.

---

## Limitations

- Core models rely on price history and trend only — no inflation, interest rate
  differentials, or policy announcements (aside from the experimental USD Index add-on).
- Forecast uncertainty grows with horizon; longer-range forecasts are directional, not exact.
- Free FX data sources can occasionally be unreliable — hence the offline fallback.
- The experimental SARIMAX model forecasts its own external factor's future values, which is
  an assumption, not a fact — treat it as a secondary signal, not a primary one.

## Roadmap

- Replace the drift-forecasted exogenous input with a real forward-looking macro indicator.
- Add CI (e.g. GitHub Actions) to run the test suite automatically on every push.
- Track live forecast accuracy over time as new data arrives, instead of only backtesting on
  historical windows.
- Add more comparison models (e.g. exponential smoothing with trend) to the accuracy table.
