"""Lightweight demand forecaster: linear trend + weekly seasonal profile + residual band.

Deterministic and CPU-only, so runs stay reproducible and fast. Accurate enough to feed the
allocator; swap in ETS/ARIMA/Prophet for longer horizons or strong autocorrelation. Returns a
point forecast, an 80% band, a backtested MAPE, and a confidence in [0, 1].
"""
from __future__ import annotations

import numpy as np


def _fit_seasonal_trend(y: np.ndarray, season: int) -> tuple[float, float, np.ndarray]:
    """Return (intercept, slope, seasonal_profile[season])."""
    t = np.arange(len(y))
    # Linear trend via least squares.
    slope, intercept = np.polyfit(t, y, 1)
    detrended = y - (intercept + slope * t)
    # Average residual per season index = seasonal profile.
    profile = np.zeros(season)
    for s in range(season):
        vals = detrended[s::season]
        profile[s] = vals.mean() if len(vals) else 0.0
    return float(intercept), float(slope), profile


def forecast_demand(history: list[float], horizon: int, season: int = 7) -> dict:
    """Forecast ``horizon`` future periods from a demand ``history``.

    Args:
        history: chronological demand values (most recent last).
        horizon: number of future periods to predict.
        season: seasonal cycle length (7 = weekly).
    """
    y = np.asarray(history, dtype=float)
    if len(y) < 2 * season:
        # Fallback: flat mean forecast for very short series.
        mean = float(y.mean()) if len(y) else 0.0
        std = float(y.std()) if len(y) else 1.0
        pt = [mean] * horizon
        return {
            "point_forecast": pt,
            "lower_80": [max(0.0, mean - 1.28 * std)] * horizon,
            "upper_80": [mean + 1.28 * std] * horizon,
            "method": "flat_mean",
            "backtest_mape": float("nan"),
            "confidence": 0.4,
        }

    intercept, slope, profile = _fit_seasonal_trend(y, season)
    t = np.arange(len(y))
    fitted = intercept + slope * t + profile[t % season]
    resid_std = float(np.std(y - fitted))

    future_t = np.arange(len(y), len(y) + horizon)
    point = intercept + slope * future_t + profile[future_t % season]
    point = np.maximum(0.0, point)

    # Rolling one-step backtest on the last `season*2` points to estimate MAPE.
    errs = []
    for cut in range(len(y) - season, len(y)):
        yi, yt = y[:cut], np.arange(cut)
        s, b = np.polyfit(yt, yi, 1)
        prof = _fit_seasonal_trend(yi, season)[2]
        pred = max(0.0, b + s * cut + prof[cut % season])
        actual = y[cut]
        if actual > 0:
            errs.append(abs(pred - actual) / actual)
    mape = float(np.mean(errs)) if errs else float("nan")
    # Map MAPE -> confidence: 0% err -> 1.0, 40%+ err -> ~0.2.
    confidence = float(np.clip(1.0 - 2.0 * (mape if mape == mape else 0.5), 0.15, 0.95))

    z = 1.28  # ~80% band
    return {
        "point_forecast": [round(float(v), 2) for v in point],
        "lower_80": [round(float(max(0.0, v - z * resid_std)), 2) for v in point],
        "upper_80": [round(float(v + z * resid_std), 2) for v in point],
        "method": "seasonal_trend",
        "backtest_mape": round(mape, 4) if mape == mape else mape,
        "confidence": round(confidence, 3),
    }
