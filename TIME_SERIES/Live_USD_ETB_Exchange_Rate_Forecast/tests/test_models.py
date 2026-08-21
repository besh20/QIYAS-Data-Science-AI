"""
Tests for models.py — run with: pytest tests/test_models.py

These use synthetic data (a clean upward-trending series with mild noise) so
tests are fast, deterministic, and don't depend on internet access or the
real yfinance data.
"""

import numpy as np
import pandas as pd
import pytest

from models import (
    naive_flat_forecast,
    naive_drift_forecast,
    arima_drift_forecast,
    run_backtest,
)


@pytest.fixture
def trending_series():
    """A synthetic series with a clear upward trend + small noise, like a depreciating currency."""
    np.random.seed(42)
    n = 300
    trend = np.linspace(100, 160, n)
    noise = np.random.normal(0, 0.5, n)
    values = trend + noise
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(values, index=dates)


def test_naive_flat_is_constant(trending_series):
    forecast = naive_flat_forecast(trending_series, horizon=14)
    assert len(forecast) == 14
    assert np.all(forecast == forecast[0]), "Naive flat forecast should never change"


def test_naive_drift_trends_upward(trending_series):
    forecast = naive_drift_forecast(trending_series, horizon=14)
    assert len(forecast) == 14
    assert forecast[-1] > forecast[0], "Naive drift should extend the upward trend, not stay flat"
    # Forecast should start close to the last known value
    assert abs(forecast[0] - trending_series.iloc[-1]) < 5


def test_naive_drift_beats_flat_on_trending_data(trending_series):
    """On genuinely trending data, drift-aware naive should get closer to a held-out truth
    than flat naive would, since flat naive ignores the trend entirely."""
    train = trending_series.iloc[:-14]
    actual_last = trending_series.iloc[-1]

    flat_forecast = naive_flat_forecast(train, horizon=14)
    drift_forecast = naive_drift_forecast(train, horizon=14)

    flat_error = abs(flat_forecast[-1] - actual_last)
    drift_error = abs(drift_forecast[-1] - actual_last)

    assert drift_error < flat_error, "Drift-aware naive should outperform flat naive on trending data"


def test_arima_drift_forecast_shape(trending_series):
    forecast, conf_int = arima_drift_forecast(trending_series, horizon=10)
    assert len(forecast) == 10
    assert conf_int.shape == (10, 2)


def test_arima_confidence_interval_widens_over_horizon(trending_series):
    """Uncertainty should grow the further out the forecast goes — a basic sanity check
    that the model isn't producing an unrealistically flat/overconfident interval."""
    _, conf_int = arima_drift_forecast(trending_series, horizon=14)
    early_width = conf_int[0, 1] - conf_int[0, 0]
    late_width = conf_int[-1, 1] - conf_int[-1, 0]
    assert late_width > early_width, "Confidence interval should widen further into the forecast"


def test_run_backtest_returns_expected_shape(trending_series):
    result = run_backtest(trending_series, horizons=(1, 7, 14), max_horizon=14, step=20, n_origins=3)
    assert list(result.columns) == ["Naive (flat)", "Naive (drift)", "ARIMA (drift)"]
    assert set(result.index) == {1, 7, 14}
    # All values should be valid, non-negative percentages
    assert (result.values >= 0).all()


def test_run_backtest_drift_beats_flat_at_longer_horizon(trending_series):
    """On clearly trending synthetic data, drift-aware models should show lower
    backtested error than flat naive at the longest horizon tested."""
    result = run_backtest(trending_series, horizons=(14,), max_horizon=14, step=20, n_origins=3)
    assert result.loc[14, "Naive (drift)"] < result.loc[14, "Naive (flat)"]
