# risk_indicators.py

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

# ======================
# DATA HELPERS
# ======================

def get_close_series(ticker, period="6mo", interval=None):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"][ticker].dropna()
    return df["Close"].dropna()

# ======================
# STAT HELPERS
# ======================

def zscore(series, window=60):
    if len(series) < window:
        return 0.0
    mean = series[-window:].mean()
    std = series[-window:].std()
    if std == 0 or np.isnan(std):
        return 0.0
    return (series.iloc[-1] - mean) / std

def normalize_z(z, cap=3.0):
    return min(max(abs(z) / cap, 0.0), 1.0)

# ======================
# INDICATORS
# ======================

def volatility_expansion_score():
    vix = get_close_series("^VIX", "6mo")
    if len(vix) < 10:
        return 0.0
    roc = vix.pct_change(3).dropna()
    z = zscore(roc, 60)
    return normalize_z(z)

def options_hedging_score():
    vix = get_close_series("^VIX", "1y")
    vix3m = get_close_series("^VIX3M", "1y")
    if vix.empty or vix3m.empty:
        return 0.0

    spread = vix - vix3m
    z = zscore(spread, 120)
    return normalize_z(z)

def options_percentile():
    vix = get_close_series("^VIX", "5y")
    vix3m = get_close_series("^VIX3M", "5y")
    if vix.empty or vix3m.empty:
        return None

    spread = (vix - vix3m).dropna()
    if len(spread) < 100:
        return None

    current = spread.iloc[-1]
    return int(percentileofscore(spread, current))

def credit_stress_score():
    hyg = get_close_series("HYG", "1y")
    ief = get_close_series("IEF", "1y")
    if hyg.empty or ief.empty:
        return 0.0

    rel = hyg.pct_change() - ief.pct_change()
    z = zscore(rel.dropna(), 120)
    return normalize_z(z)

def small_cap_score():
    iwm = get_close_series("IWM", "1y")
    spy = get_close_series("SPY", "1y")
    if iwm.empty or spy.empty:
        return 0.0

    rel = iwm.pct_change(20) - spy.pct_change(20)
    z = zscore(rel.dropna(), 120)
    return normalize_z(z)
