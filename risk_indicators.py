import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from typing import List
from datetime import datetime, timedelta

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
    last_val = series.iloc[-1]
    if isinstance(last_val, pd.Series):
        last_val = last_val.item()
    return (last_val - mean) / std

def normalize_z(z, cap=3.0):
    return min(max(abs(z) / cap, 0.0), 1.0)

# ======================
# CORE INDICATORS
# ======================
def volatility_expansion_score():
    vix = get_close_series("^VIX", "6mo")
    if len(vix) < 10:
        return 0.0
    roc = vix.pct_change(3).dropna()
    z = zscore(roc, 60)
    return normalize_z(z)

def volatility_compression_score(window: int = 60):
    vix = get_close_series("^VIX", "1y")
    if len(vix) < window:
        return 0.0
    z = zscore(vix, window)
    return normalize_z(-z)

def credit_complacency_score(window: int = 120):
    hyg = get_close_series("HYG", "1y")
    ief = get_close_series("IEF", "1y")
    if hyg.empty or ief.empty:
        return 0.0
    rel = hyg.pct_change() - ief.pct_change()
    rolling_std = rel.rolling(window).std().dropna()
    if rolling_std.empty:
        return 0.0
    latest_std = rolling_std.iloc[-1]
    if isinstance(latest_std, pd.Series):
        latest_std = latest_std.item()
    mean_std = rolling_std.mean()
    std_std = rolling_std.std()
    if std_std == 0 or np.isnan(std_std):
        return 0.0
    z = (mean_std - latest_std) / std_std
    return normalize_z(z)

def breadth_divergence_score(window: int = 60):
    small = get_close_series("IWM", "6mo")
    large = get_close_series("SPY", "6mo")
    if len(small) < window or len(large) < window:
        return 0.0
    rel = small.pct_change() - large.pct_change()
    z = zscore(rel.dropna(), window)
    return normalize_z(-z)

def risk_acceleration_score(composite_scores: List[float], window: int = 3):
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
    if isinstance(current, pd.Series):
        current = current.item()
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
    sp_ret = sp500_prices.pct_change().dropna()
    btc_ret = btc_prices.pct_change().dropna()
    if len(sp_ret) < window or len(btc_ret) < window:
        return 0.0
    rolling_corr = sp_ret.rolling(window).corr(btc_ret)
    corr_val = rolling_corr.dropna().iloc[-1] if not rolling_corr.dropna().empty else 0.0
    if isinstance(corr_val, pd.Series):
        corr_val = corr_val.item()
    score = min(max(-corr_val, 0.0), 1.0)
    return float(score)

# ======================
# DRAWDOWN & RECOVERY (MORE SENSITIVE)
# ======================

def check_drawdown(ticker="SPY", short_days=2, short_thresh=-0.02, long_days=10, long_thresh=-0.05):
    """
    UPDATED: More sensitive thresholds
    - short: 2 days, -2% (was 3 days -5%)
    - long: 10 days, -5% from peak (was 20 days -10%)
    """
    prices = get_close_series(ticker, "2mo")
    if len(prices) < long_days:
        return False
    short_drop = (prices.iloc[-1] / prices.iloc[-short_days-1]) - 1
    long_peak = prices[-long_days:].max()
    long_drop = (prices.iloc[-1] / long_peak) - 1
    return short_drop < short_thresh or long_drop < long_thresh

def check_recovery(vix_thresh=0.85, credit_thresh=-0.02, vix_days=5, credit_days=5):
    vix = get_close_series("^VIX", "1mo")
    hyg = get_close_series("HYG", "1mo")
    if len(vix) < vix_days or len(hyg) < credit_days:
        return False
    vix_avg = vix[-vix_days:].mean()
    vix_last = vix.iloc[-1]
    if isinstance(vix_last, pd.Series):
        vix_last = vix_last.item()
    vix_falling = vix_last < vix_avg * vix_thresh
    hyg_change = hyg.pct_change(credit_days).iloc[-1]
    if isinstance(hyg_change, pd.Series):
        hyg_change = hyg_change.item()
    credit_stable = hyg_change > credit_thresh
    return vix_falling and credit_stable

def get_persistent_risk(recent_scores: List[float], threshold=0.7, days=2):
    if len(recent_scores) < days:
        return False
    return sum(s > threshold for s in recent_scores[-days:]) >= days

def vix_spike_score():
    vix = get_close_series("^VIX", "2mo")
    if len(vix) < 5:
        return 0.0
    spike_1d = (vix.iloc[-1] / vix.iloc[-2]) - 1
    spike_2d = (vix.iloc[-1] / vix.iloc[-3]) - 1
    spike_3d = (vix.iloc[-1] / vix.iloc[-4]) - 1
    max_spike = max(spike_1d, spike_2d, spike_3d)
    return min(max(max_spike / 0.30, 0.0), 1.0)

# ======================
# ADDITIONAL INDICATORS
# ======================

def put_call_ratio_score(window: int = 60):
    vix = get_close_series("^VIX", "1y")
    vix3m = get_close_series("^VIX3M", "1y")
    if vix.empty or vix3m.empty:
        return 0.0
    ratio = vix / vix3m
    z = zscore(ratio.dropna(), window)
    return normalize_z(z)

def credit_spread_score(window: int = 120):
    hyg = get_close_series("HYG", "1y")
    lqd = get_close_series("LQD", "1y")
    if hyg.empty or lqd.empty:
        return 0.0
    spread = lqd.pct_change() - hyg.pct_change()
    z = zscore(spread.dropna(), window)
    return normalize_z(z)

def breadth_score(window: int = 60):
    spy = get_close_series("SPY", "6mo")
    iwm = get_close_series("IWM", "6mo")
    if len(spy) < window or len(iwm) < window:
        return 0.0
    rel = iwm.pct_change() - spy.pct_change()
    z = zscore(rel.dropna(), window)
    return normalize_z(-z)

def dollar_strength_score(window: int = 60):
    dxy = get_close_series("DX-Y.NYB", "6mo")
    if dxy.empty:
        eurusd = get_close_series("EURUSD=X", "6mo")
        if eurusd.empty:
            return 0.0
        dxy = 1 / eurusd
    roc = dxy.pct_change(5).dropna()
    z = zscore(roc, window)
    return normalize_z(z)

def yield_curve_score(window: int = 120):
    ief = get_close_series("IEF", "1y")
    sho = get_close_series("SHY", "1y")
    if ief.empty or sho.empty:
        return 0.0
    spread = ief.pct_change() - sho.pct_change()
    z = zscore(spread.dropna(), window)
    return normalize_z(-z)

# ======================
# DEBT CEILING & TREASURY
# ======================

def debt_ceiling_stress_score():
    bil = get_close_series("BIL", "6mo")
    shy = get_close_series("SHY", "6mo")
    if bil.empty or shy.empty:
        return 0.0
    spread = shy.pct_change() - bil.pct_change()
    z = zscore(spread.dropna(), 60)
    return normalize_z(z)

def treasury_stress_score(window: int = 60):
    shy = get_close_series("SHY", "6mo")
    if shy.empty or len(shy) < window:
        return 0.0
    returns = shy.pct_change().dropna()
    rolling_vol = returns.rolling(20).std().dropna()
    if rolling_vol.empty:
        return 0.0
    current_vol = rolling_vol.iloc[-1]
    if isinstance(current_vol, pd.Series):
        current_vol = current_vol.item()
    z = zscore(rolling_vol, window)
    return normalize_z(z)

def days_to_debt_ceiling():
    today = datetime.now()
    x_date = datetime(2026, 8, 1)
    days = (x_date - today).days
    is_near = days <= 60
    return days, is_near

def budget_vote_risk_score():
    days, is_near = days_to_debt_ceiling()
    if not is_near:
        return 0.0
    time_score = min((60 - days) / 60, 1.0)
    treasury_stress = treasury_stress_score()
    debt_stress = debt_ceiling_stress_score()
    combined = (
        0.4 * time_score +
        0.3 * treasury_stress +
        0.3 * debt_stress
    )
    return min(combined, 1.0)

# ======================
# NEW: EARNINGS SEASON
# ======================

def is_earnings_season():
    """Q1: mid-Apr to early May, Q2: mid-Jul to early Aug, Q3: mid-Oct to early Nov, Q4: mid-Jan to early Feb"""
    today = datetime.now()
    month = today.month
    day = today.day
    earnings_windows = [
        (1, 15, 28, "Q4"), (2, 1, 10, "Q4"),
        (4, 15, 30, "Q1"), (5, 1, 10, "Q1"),
        (7, 15, 31, "Q2"), (8, 1, 10, "Q2"),
        (10, 15, 31, "Q3"), (11, 1, 10, "Q3"),
    ]
    for earn_month, start, end, quarter in earnings_windows:
        if month == earn_month and start <= day <= end:
            mid_point = (start + end) / 2
            distance_from_peak = abs(day - mid_point)
            max_distance = (end - start) / 2
            intensity = 1.0 - (distance_from_peak / max_distance) * 0.5
            return True, intensity, f"{quarter} earnings season"
    return False, 0.0, "No major earnings"

def earnings_volatility_score():
    """Elevated risk during earnings"""
    is_earnings, intensity, description = is_earnings_season()
    if not is_earnings:
        return 0.0
    qqq = get_close_series("QQQ", "1mo")
    if len(qqq) < 20:
        return intensity * 0.5
    returns = qqq.pct_change().dropna()
    recent_vol = returns[-5:].std()
    normal_vol = returns[-20:].std()
    if normal_vol == 0 or np.isnan(normal_vol):
        return intensity * 0.5
    vol_ratio = recent_vol / normal_vol
    vol_score = min(vol_ratio / 2.0, 1.0)
    combined = intensity * 0.5 + vol_score * 0.5
    return min(combined, 1.0)

# ======================
# NEW: CONGRESSIONAL BUDGET CALENDAR
# ======================

def congressional_budget_risk_score():
    """Tracks shutdown risk and fiscal votes"""
    today = datetime.now()
    month = today.month
    day = today.day
    high_risk_periods = [
        (9, 15, 30, 0.8, "End of fiscal year"),
        (10, 1, 15, 0.6, "Post-shutdown negotiations"),
        (12, 15, 31, 0.7, "Holiday CR deadline"),
        (1, 1, 15, 0.5, "Post-holiday budget battles"),
        (3, 15, 31, 0.6, "Spring CR deadline"),
        (4, 1, 15, 0.5, "Budget reconciliation"),
    ]
    for risk_month, start, end, risk_level, reason in high_risk_periods:
        if month == risk_month and start <= day <= end:
            treasury_stress = treasury_stress_score()
            combined = risk_level * 0.6 + treasury_stress * 0.4
            return min(combined, 1.0)
    treasury_stress = treasury_stress_score()
    if treasury_stress > 0.5:
        return treasury_stress * 0.3
    return 0.0

def get_budget_risk_details():
    """Human-readable description"""
    today = datetime.now()
    month = today.month
    congressional_score = congressional_budget_risk_score()
    debt_days, is_near_debt = days_to_debt_ceiling()
    details = []
    if congressional_score > 0.5:
        if month == 9:
            details.append("End of fiscal year approaching")
        elif month in [12, 1]:
            details.append("Holiday season CR deadline")
        elif month in [3, 4]:
            details.append("Spring budget battles")
        else:
            details.append("Congressional fiscal negotiations")
    if is_near_debt:
        details.append(f"Debt ceiling in {debt_days} days")
    if not details:
        details.append("No major fiscal deadlines")
    return " | ".join(details)
