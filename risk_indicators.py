import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

# ======================
# DATA HELPERS
# ======================

def get_close_series(ticker, period="6mo", interval="1d"):
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

def rolling_zscore(series: pd.Series, window: int = 20):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def cross_asset_confirmation_score():
    gold = get_close_series("GLD", "6mo")
    btc = get_close_series("BTC-USD", "6mo")

    if len(gold) < 40 or len(btc) < 40:
        return 0.0

    gold_ret = gold.pct_change().dropna()
    btc_ret = btc.pct_change().dropna()

    gold_z = zscore(gold_ret, 20)
    btc_z = zscore(btc_ret, 20)

    score = 0.0

    # Risk-off confirmation
    if gold_z > 0:
        score += normalize_z(gold_z)
    if btc_z < 0:
        score += normalize_z(-btc_z)

    # Risk-on contradiction (reduces confidence)
    if gold_z < 0:
        score -= normalize_z(-gold_z) * 0.5
    if btc_z > 0:
        score -= normalize_z(btc_z) * 0.5

    return min(max(score / 2.0, 0.0), 1.0)
