import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from typing import List

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

def volatility_compression_score(window: int = 60):
    """
    Detects when VIX is unusually low (fragile market).
    High score = fragile, pre-crash conditions.
    """
    vix = get_close_series("^VIX", "1y")
    if len(vix) < window:
        return 0.0

    z = zscore(vix, window)
    return normalize_z(-z)  # inverted: low VIX → high fragility

def credit_complacency_score(window: int = 120):
    """
    Detects when credit spreads are unusually tight.
    High score = fragile, late-cycle market.
    """
    hyg = get_close_series("HYG", "1y")
    ief = get_close_series("IEF", "1y")
    if hyg.empty or ief.empty:
        return 0.0

    rel = hyg.pct_change() - ief.pct_change()
    rolling_std = rel.rolling(window).std().dropna()
    if rolling_std.empty:
        return 0.0

    latest_std = rolling_std.iloc[-1]
    mean_std = rolling_std.mean()
    std_std = rolling_std.std()
    if std_std == 0 or np.isnan(std_std):
        return 0.0

    z = (mean_std - latest_std) / std_std  # inverted: low std = high fragility
    return normalize_z(z)

def breadth_divergence_score(window: int = 60):
    """
    Detects small-cap underperformance vs large-cap (fragile rally).
    High score = small caps lag → caution.
    """
    small = get_close_series("IWM", "6mo")
    large = get_close_series("SPY", "6mo")
    if len(small) < window or len(large) < window:
        return 0.0

    rel = small.pct_change() - large.pct_change()
    z = zscore(rel.dropna(), window)
    return normalize_z(-z)  # negative z = small caps lag → fragile

def risk_acceleration_score(composite_scores: List[float], window: int = 3):
    """
    Measures acceleration of risk score.
    Input: previous composite scores (0–1)
    Output: normalized 0–1 acceleration score
    """
    if len(composite_scores) < window + 1:
        return 0.0

    recent = np.array(composite_scores[-window-1:])
    accel = recent[2:] - 2 * recent[1:-1] + recent[:-2]
    if len(accel) == 0:
        return 0.0

    latest_accel = accel[-1]
    mean_accel = accel.mean()
    std_accel = accel.std()
    if std_accel == 0 or np.isnan(std_accel):
        return 0.0

    z = (latest_accel - mean_accel) / std_accel
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

def gold_crypto_confirmation(gold_prices: pd.Series, btc_prices: pd.Series):
    """
    Returns:
    - confirmation_score (float, -1..1)
    - gold_z (float)
    - btc_z (float)
    
    Logic:
    - Gold rising (Z>1) = risk-off
    - BTC falling (Z<-1) = risk-off
    - Opposite moves reduce score (risk-on)
    """
    gold_ret = gold_prices.pct_change().dropna()
    btc_ret = btc_prices.pct_change().dropna()

    if gold_ret.empty or btc_ret.empty:
        return 0.0, 0.0, 0.0

    gold_z_series = rolling_zscore(gold_ret, 20)
    btc_z_series  = rolling_zscore(btc_ret, 20)
    
    gold_z = float(gold_z_series.dropna().iloc[-1].item()) if len(gold_z_series.dropna()) > 0 else 0.0
    btc_z  = float(btc_z_series.dropna().iloc[-1].item())  if len(btc_z_series.dropna()) > 0 else 0.0

    score = 0.0

    # Risk-off confirmation
    if gold_z > 1.0:
        score += 0.5
    if btc_z < -1.0:
        score += 0.5

    # Risk-on contradiction
    if gold_z < -1.0:
        score -= 0.5
    if btc_z > 1.0:
        score -= 0.5

    # Clamp score to -1 .. 1
    score = max(min(score, 1.0), -1.0)

    return score, gold_z, btc_z
  
def btc_equity_correlation(sp500_prices: pd.Series, btc_prices: pd.Series, window=20):
    """
    Returns 0-1 score:
    - 1 = BTC trending opposite SPX (early risk-off)
    - 0 = positive or neutral correlation
    """
    sp_ret = sp500_prices.pct_change().dropna()
    btc_ret = btc_prices.pct_change().dropna()

    if len(sp_ret) < window or len(btc_ret) < window:
        return 0.0

    # Rolling correlation
    rolling_corr = sp_ret.rolling(window).corr(btc_ret)

    # Take the last valid numeric value
    corr = rolling_corr.dropna().iloc[-1] if not rolling_corr.dropna().empty else 0.0

    # Negative correlation → early risk-off → score 1, positive → 0
    score = min(max(-corr, 0.0), 1.0)
    return float(score)
