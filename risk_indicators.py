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
    return normalize_z(-z)

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

    z = (mean_std - latest_std) / std_std
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
    return normalize_z(-z)

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

    if gold_z > 0:
        score += normalize_z(gold_z)
    if btc_z < 0:
        score += normalize_z(-btc_z)

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

    if gold_z > 1.0:
        score += 0.5
    if btc_z < -1.0:
        score += 0.5

    if gold_z < -1.0:
        score -= 0.5
    if btc_z > 1.0:
        score -= 0.5

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

    rolling_corr = sp_ret.rolling(window).corr(btc_ret)

    corr = rolling_corr.dropna().iloc[-1] if not rolling_corr.dropna().empty else 0.0

    score = min(max(-corr, 0.0), 1.0)
    return float(score)

# ======================
# NEW: DRAWDOWN & RECOVERY
# ======================

def check_drawdown(ticker="SPY", short_days=3, short_thresh=-0.05, long_days=20, long_thresh=-0.10):
    """
    Returns True if drawdown exceeds thresholds.
    - short_days: rapid decline (default 3 days, -5%)
    - long_days: sustained decline (default 20 days, -10% from peak)
    """
    prices = get_close_series(ticker, "2mo")
    if len(prices) < long_days:
        return False
    
    short_drop = (prices.iloc[-1] / prices.iloc[-short_days-1]) - 1
    long_peak = prices[-long_days:].max()
    long_drop = (prices.iloc[-1] / long_peak) - 1
    
    return short_drop < short_thresh or long_drop < long_thresh

def check_recovery(vix_thresh=0.85, credit_thresh=-0.02, vix_days=5, credit_days=5):
    """
    Returns True if recovery signals present.
    - VIX declining below recent average
    - Credit (HYG) stabilizing
    """
    vix = get_close_series("^VIX", "1mo")
    hyg = get_close_series("HYG", "1mo")
    
    if len(vix) < vix_days or len(hyg) < credit_days:
        return False
    
    vix_falling = vix.iloc[-1] < vix[-vix_days:].mean() * vix_thresh
    credit_stable = hyg.pct_change(credit_days).iloc[-1] > credit_thresh
    
    return vix_falling and credit_stable

def get_persistent_risk(recent_scores: List[float], threshold=0.7, days=2):
    """
    Returns True if composite risk score exceeded threshold for N consecutive days.
    """
    if len(recent_scores) < days:
        return False
    return sum(s > threshold for s in recent_scores[-days:]) >= days

def vix_spike_score():
    """
    Detects rapid VIX spikes (2018-style flash crashes).
    Returns 0-1 score based on 1-3 day VIX acceleration.
    """
    vix = get_close_series("^VIX", "2mo")
    if len(vix) < 5:
        return 0.0
    
    # Check 1-day, 2-day, 3-day spikes
    spike_1d = (vix.iloc[-1] / vix.iloc[-2]) - 1
    spike_2d = (vix.iloc[-1] / vix.iloc[-3]) - 1
    spike_3d = (vix.iloc[-1] / vix.iloc[-4]) - 1
    
    max_spike = max(spike_1d, spike_2d, spike_3d)
    
    # 30%+ spike in 1-3 days = max score
    return min(max(max_spike / 0.30, 0.0), 1.0)

# ======================
# NEW INDICATORS
# ======================

def put_call_ratio_score(window: int = 60):
    """
    Detects elevated put buying (panic protection).
    High score = investors hedging aggressively = fear.
    Uses VIX/VIX3M as proxy for put/call ratio.
    """
    vix = get_close_series("^VIX", "1y")
    vix3m = get_close_series("^VIX3M", "1y")
    if vix.empty or vix3m.empty:
        return 0.0
    
    # Ratio > 1 means near-term fear > long-term
    ratio = vix / vix3m
    z = zscore(ratio.dropna(), window)
    return normalize_z(z)

def credit_spread_score(window: int = 120):
    """
    High yield spread (HYG-LQD) expansion.
    High score = credit markets pricing in stress.
    """
    hyg = get_close_series("HYG", "1y")  # High yield
    lqd = get_close_series("LQD", "1y")  # Investment grade
    if hyg.empty or lqd.empty:
        return 0.0
    
    # Underperformance of HY vs IG = spread widening
    spread = lqd.pct_change() - hyg.pct_change()
    z = zscore(spread.dropna(), window)
    return normalize_z(z)

def breadth_score(window: int = 60):
    """
    Advance/Decline line divergence.
    High score = narrow rally, few stocks supporting market.
    """
    spy = get_close_series("SPY", "6mo")
    iwm = get_close_series("IWM", "6mo")  # Small caps as breadth proxy
    if len(spy) < window or len(iwm) < window:
        return 0.0
    
    # Small cap underperformance = weak breadth
    rel = iwm.pct_change() - spy.pct_change()
    z = zscore(rel.dropna(), window)
    return normalize_z(-z)  # Negative rel = high score

def dollar_strength_score(window: int = 60):
    """
    Dollar strength (DXY) as safe haven indicator.
    High score = strong dollar = global stress.
    """
    dxy = get_close_series("DX-Y.NYB", "6mo")  # Dollar index
    if dxy.empty:
        # Fallback: use inverse of EUR/USD
        eurusd = get_close_series("EURUSD=X", "6mo")
        if eurusd.empty:
            return 0.0
        dxy = 1 / eurusd
    
    roc = dxy.pct_change(5).dropna()
    z = zscore(roc, window)
    return normalize_z(z)

def yield_curve_score(window: int = 120):
    """
    Treasury curve inversion (2y-10y spread).
    High score = inverted curve = recession signal.
    """
    ief = get_close_series("IEF", "1y")   # 7-10yr Treasury ETF
    sho = get_close_series("SHY", "1y")   # 1-3yr Treasury ETF
    if ief.empty or sho.empty:
        return 0.0
    
    # When short > long, spread is negative (inversion)
    # SHY outperforming IEF = curve flattening/inverting
    spread = ief.pct_change() - sho.pct_change()
    z = zscore(spread.dropna(), window)
    return normalize_z(-z)  # Negative spread = high score