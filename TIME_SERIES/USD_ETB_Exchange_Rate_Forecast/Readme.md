# USD/ETB Exchange Rate Forecast

A live-updating short-horizon forecasting app for the USD/ETB exchange rate, built to practice
the full time series workflow: data collection → stationarity testing → modeling → honest
evaluation → deployment.

🔗 **Live app:** [add your Streamlit Cloud URL here]

## What this project does

Pulls daily USD/ETB exchange rate data live, fits an ARIMA model, and compares it against a
naive baseline. Both forecasts are shown side by side in a deployed Streamlit app, along with
an in-app page explaining the methodology and findings.

## Why USD/ETB

The Ethiopian Birr has been depreciating steadily, so a short-term forecast is practically
useful (planning forex needs, budgeting in USD terms). It's also a good real-world dataset
because it's noisy and non-stationary — a step up in difficulty from clean textbook data.

## What I did, and why

**1. Data collection**
Pulled daily rates via `yfinance` (ticker `USDETB=X`), starting from 2024-08-01 to keep the
data recent. The deployed app fetches this live on each load (cached hourly), with a saved
snapshot (`history.pkl`) as an offline fallback if the live source is ever unreachable.

**2. Cleaning**
Resampled to daily frequency and forward-filled gaps (weekends/holidays have no trading data),
since a time series model needs a continuous, evenly-spaced index.

**3. Exploratory plotting**
Plotted the raw series to check for trend and seasonality visually before doing anything
statistical.

**4. Decomposition**
Ran seasonal decomposition to check for a repeating yearly pattern. Found none — FX rates are
driven by economics/policy, not the calendar. This confirmed SARIMA (seasonal ARIMA) wasn't
needed; a plain ARIMA was the right-sized tool.

**5. Stationarity testing (ADF test)**
The raw series was non-stationary (high p-value, as expected given the strong trend). After
log-transforming and differencing once, the ADF test confirmed stationarity (p-value ≈ 0.0000).

**6. ACF/PACF plots**
Checked autocorrelation structure to get a starting guess for ARIMA's `p` and `q` parameters.
Signal beyond lag 1 was weak — an early hint the series behaves close to a random walk.

**7. Train/test split + naive baseline**
Held out the last 30 days to test forecast accuracy, and built a **naive baseline**
("tomorrow's rate = today's rate") to check whether ARIMA actually adds value, instead of
just trusting the model output on its own.

**8. Grid search over ARIMA orders**
Tested multiple `(p,d,q)` combinations. All landed in a tight, similar error range
(~1.37%–1.40% MAPE), and none beat the naive baseline (0.50% MAPE). `ARIMA(0,1,1)` performed
best among them and was selected as the final model.

**9. Result: naive baseline wins**
The naive model outperformed every ARIMA configuration. This isn't a failure of the analysis —
it's a well-documented property of FX markets: exchange rates behave close to a random walk at
short horizons, so "today's rate" is genuinely hard to beat as a 1-day forecast.

**10. Deployment**
Deployed as a live Streamlit app that fetches fresh data and refits the model on each session
(cached hourly), rather than relying on a frozen snapshot — so the chart and forecasts stay
current. Both models are shown side by side, with the naive-vs-ARIMA finding presented
transparently rather than hidden, because that's a more credible result than a cherry-picked one.

## Project structure

```
USD_ETB_Exchange_Rate_Forecast/
├── app.py                          # Main Streamlit app (forecast + chart)
├── pages/
│   └── 1_About_This_Project.py     # In-app explainer: methodology & findings
├── history.pkl                     # Offline fallback data snapshot
├── requirements.txt
├── USD_ETB_Forecast.ipynb          # Full analysis notebook (ADF, ACF/PACF, grid search)
└── README.md
```

## Tech stack

- Python, pandas, numpy
- statsmodels (ARIMA, ADF test, decomposition)
- Plotly (interactive charts)
- scikit-learn (evaluation metrics)
- yfinance (live data source)
- Streamlit (app + deployment)

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app," select the repo, and set the main file path to `app.py`.
4. Streamlit Cloud builds and deploys automatically — it also picks up the `pages/` folder
   for in-app navigation.

## Limitations

- ARIMA uses only past price history — no external factors like inflation, interest rate
  differentials, or policy announcements, which likely drive Birr depreciation more than the
  price history itself.
- Forecast reliability drops quickly beyond a few days; long-horizon predictions should not be
  trusted for real financial decisions.
- Data availability for USD/ETB from free sources can be inconsistent; a production version
  would need a more reliable paid data feed.

## Possible next steps

- Add external features (inflation data, remittance flows) to see if a richer model can
  actually beat the naive baseline.
- Try exponential smoothing or a simple machine learning model (e.g. gradient boosting on lag
  features) as additional comparisons.
- Track live forecast accuracy over time instead of a single fixed backtest.