"""
Forecasting model logic for the USD/ETB app.

Deliberately kept free of any Streamlit imports — this makes the logic testable
and reusable on its own, independent of the UI layer.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error


def naive_flat_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Tomorrow = today, held flat for the whole horizon."""
    return np.array([train.iloc[-1]] * horizon)


def naive_drift_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Extends the last value using the average historical daily change."""
    avg_change = train.diff().mean()
    last = train.iloc[-1]
    return np.array([last + avg_change * (i + 1) for i in range(horizon)])


def arima_drift_forecast(train: pd.Series, horizon: int, order=(0, 1, 1)):
    """ARIMA with a trend term, so it doesn't flatten to zero change."""
    model = ARIMA(train, order=order, trend="t")
    fit = model.fit()
    result = fit.get_forecast(steps=horizon)
    return result.predicted_mean.values, result.conf_int().values


def sarimax_exog_forecast(train_y: pd.Series, train_x: pd.Series, horizon: int, order=(0, 1, 1)):
    """
    Experimental: ARIMA + an external factor (e.g. US Dollar Index) as an exogenous input.
    Future exog values are unknown, so they're forecasted with a simple drift model first,
    then fed into the main model. That's a real assumption, not a fact.
    """
    model = SARIMAX(train_y, exog=train_x, order=order, trend="t",
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)

    avg_change = train_x.diff().mean()
    last_x = train_x.iloc[-1]
    future_x = np.array([last_x + avg_change * (i + 1) for i in range(horizon)]).reshape(-1, 1)

    result = fit.get_forecast(steps=horizon, exog=future_x)
    return result.predicted_mean.values, result.conf_int().values


def run_backtest(series: pd.Series, horizons=(1, 7, 14, 30), max_horizon=30, step=15, n_origins=6) -> pd.DataFrame:
    """
    Rolling-origin backtest: re-tests each model from several past starting points,
    then measures accuracy at each horizon in `horizons`. More robust than a single
    train/test split, since it isn't dependent on one lucky (or unlucky) test window.

    Returns a DataFrame: rows = horizon, columns = model name, values = mean MAPE (%).
    """
    results = {h: {"Naive (flat)": [], "Naive (drift)": [], "ARIMA (drift)": []} for h in horizons}
    total_len = len(series)
    origin_ends = [total_len - max_horizon - i * step for i in range(n_origins)]

    for end in origin_ends:
        if end < 100:
            continue
        train = series.iloc[:end]
        actual = series.iloc[end:end + max_horizon]
        if len(actual) < max_horizon:
            continue

        nf = naive_flat_forecast(train, max_horizon)
        nd = naive_drift_forecast(train, max_horizon)
        try:
            ad, _ = arima_drift_forecast(train, max_horizon)
        except Exception:
            continue

        for h in horizons:
            a = actual.values[:h]
            results[h]["Naive (flat)"].append(mean_absolute_percentage_error(a, nf[:h]) * 100)
            results[h]["Naive (drift)"].append(mean_absolute_percentage_error(a, nd[:h]) * 100)
            results[h]["ARIMA (drift)"].append(mean_absolute_percentage_error(a, ad[:h]) * 100)

    summary = {h: {model: (np.mean(vals) if vals else np.nan) for model, vals in results[h].items()}
               for h in horizons}
    return pd.DataFrame(summary).T
