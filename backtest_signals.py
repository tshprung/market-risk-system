"""
Backtest trade signals against historical crashes.
Run this to validate your thresholds before going live.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Simplified indicator calculations for backtest
def simple_vix_score(vix, date):
    # Ensure we have a Series
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    
    window = vix.loc[:date].tail(60)
    if len(window) < 60:
        return 0.0
    roc = window.pct_change(3).dropna()
    if len(roc) == 0:
        return 0.0
    
    mean_val = roc.mean()
    std_val = roc.std()
    
    # Convert to scalars
    if isinstance(mean_val, pd.Series):
        mean_val = mean_val.item()
    if isinstance(std_val, pd.Series):
        std_val = std_val.item()
    
    if std_val == 0 or np.isnan(std_val):
        return 0.0
    
    latest = roc.iloc[-1]
    if isinstance(latest, pd.Series):
        latest = latest.item()
    
    z = (latest - mean_val) / std_val
    return min(max(abs(z) / 3.0, 0.0), 1.0)

def simple_credit_score(hyg, ief, date):
    # Ensure we have Series
    if isinstance(hyg, pd.DataFrame):
        hyg = hyg.iloc[:, 0]
    if isinstance(ief, pd.DataFrame):
        ief = ief.iloc[:, 0]
    
    hyg_w = hyg.loc[:date].tail(120)
    ief_w = ief.loc[:date].tail(120)
    if len(hyg_w) < 120 or len(ief_w) < 120:
        return 0.0
    rel = hyg_w.pct_change() - ief_w.pct_change()
    rel = rel.dropna()
    if len(rel) == 0:
        return 0.0
    
    mean_val = rel.mean()
    std_val = rel.std()
    
    # Convert to scalars
    if isinstance(mean_val, pd.Series):
        mean_val = mean_val.item()
    if isinstance(std_val, pd.Series):
        std_val = std_val.item()
    
    if std_val == 0 or np.isnan(std_val):
        return 0.0
    
    latest = rel.iloc[-1]
    if isinstance(latest, pd.Series):
        latest = latest.item()
    
    z = (latest - mean_val) / std_val
    return min(max(abs(z) / 3.0, 0.0), 1.0)

def simple_vix_spike(vix, date):
    """Detect rapid VIX spikes"""
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    
    window = vix.loc[:date].tail(5)
    if len(window) < 4:
        return 0.0
    
    spike_1d = (window.iloc[-1] / window.iloc[-2]) - 1
    spike_2d = (window.iloc[-1] / window.iloc[-3]) - 1
    spike_3d = (window.iloc[-1] / window.iloc[-4]) - 1
    
    max_spike = max(spike_1d, spike_2d, spike_3d)
    
    # Convert to scalar if needed
    if isinstance(max_spike, pd.Series):
        max_spike = max_spike.item()
    
    return min(max(max_spike / 0.30, 0.0), 1.0)

def simple_drawdown(spy, date, short_days=3, long_days=20):
    # Ensure we have a Series
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    
    prices = spy.loc[:date].tail(long_days + 1)
    if len(prices) < long_days:
        return False
    
    p_latest = prices.iloc[-1]
    p_short = prices.iloc[-short_days-1]
    p_peak = prices.iloc[-long_days:].max()
    
    # Convert to scalars
    if isinstance(p_latest, pd.Series):
        p_latest = p_latest.item()
    if isinstance(p_short, pd.Series):
        p_short = p_short.item()
    if isinstance(p_peak, pd.Series):
        p_peak = p_peak.item()
    
    short_drop = (p_latest / p_short) - 1
    long_drop = (p_latest / p_peak) - 1
    
    return short_drop < -0.05 or long_drop < -0.10

# Load historical data
print("Loading data...")
spy_data = yf.download("SPY", start="2007-01-01", end="2025-01-25", progress=False)
vix_data = yf.download("^VIX", start="2007-01-01", end="2025-01-25", progress=False)
hyg_data = yf.download("HYG", start="2007-01-01", end="2025-01-25", progress=False)
ief_data = yf.download("IEF", start="2007-01-01", end="2025-01-25", progress=False)

# Extract Close prices as Series
spy = spy_data["Close"] if "Close" in spy_data.columns else spy_data.iloc[:, 0]
vix = vix_data["Close"] if "Close" in vix_data.columns else vix_data.iloc[:, 0]
hyg = hyg_data["Close"] if "Close" in hyg_data.columns else hyg_data.iloc[:, 0]
ief = ief_data["Close"] if "Close" in ief_data.columns else ief_data.iloc[:, 0]

# Define crash periods (rough dates)
crashes = [
    {"name": "2008 Financial Crisis", "start": "2008-09-15", "end": "2009-03-09", "peak_drop": -0.57},
    {"name": "Feb 2018 VIXplosion", "start": "2018-02-01", "end": "2018-02-14", "peak_drop": -0.12},
    {"name": "Dec 2018 selloff", "start": "2018-12-01", "end": "2018-12-26", "peak_drop": -0.19},
    {"name": "COVID crash", "start": "2020-02-19", "end": "2020-03-23", "peak_drop": -0.34},
    {"name": "Sep 2022 selloff", "start": "2022-09-01", "end": "2022-10-13", "peak_drop": -0.17}
]

results = []

for crash in crashes:
    print(f"\n{'='*60}")
    print(f"Testing: {crash['name']}")
    print(f"Crash period: {crash['start']} to {crash['end']}")
    
    crash_start = pd.to_datetime(crash["start"])
    crash_end = pd.to_datetime(crash["end"])
    
    # Look for signals 30 days before crash
    test_start = crash_start - timedelta(days=30)
    test_dates = spy[test_start:crash_start].index
    
    sell_signal_date = None
    sell_composite = None
    
    for date in test_dates:
        vol_score = simple_vix_score(vix, date)
        credit_score = simple_credit_score(hyg, ief, date)
        spike_score = simple_vix_spike(vix, date)
        composite = 0.4 * vol_score + 0.3 * credit_score + 0.3 * spike_score
        drawdown = simple_drawdown(spy, date)
        vix_spike = spike_score > 0.7
        
        if composite > 0.55 or drawdown or vix_spike:
            sell_signal_date = date
            sell_composite = composite
            break
    
    if sell_signal_date:
        days_early = (crash_start - sell_signal_date).days
        
        sell_price = spy.loc[sell_signal_date]
        crash_low = spy[crash_start:crash_end].min()
        
        # Convert to scalars
        if isinstance(sell_price, pd.Series):
            sell_price = sell_price.item()
        if isinstance(crash_low, pd.Series):
            crash_low = crash_low.item()
        
        avoided_loss = (crash_low / sell_price - 1) * 100
        
        print(f"✅ SELL signal: {sell_signal_date.date()}")
        print(f"   Days before crash: {days_early}")
        print(f"   Composite score: {sell_composite:.2f}")
        print(f"   Avoided loss: {avoided_loss:.1f}%")
        
        results.append({
            "crash": crash["name"],
            "signal_date": sell_signal_date,
            "days_early": days_early,
            "avoided_loss": avoided_loss
        })
    else:
        print(f"❌ NO SIGNAL detected in 30 days before crash")
        results.append({
            "crash": crash["name"],
            "signal_date": None,
            "days_early": None,
            "avoided_loss": None
        })

# Test for false positives (2021 - relatively stable year)
print(f"\n{'='*60}")
print("Testing false positives in 2021 (stable year)")
false_positives = 0
test_dates_2021 = spy["2021-01-01":"2021-12-31"].index

for date in test_dates_2021:
    vol_score = simple_vix_score(vix, date)
    credit_score = simple_credit_score(hyg, ief, date)
    spike_score = simple_vix_spike(vix, date)
    composite = 0.4 * vol_score + 0.3 * credit_score + 0.3 * spike_score
    drawdown = simple_drawdown(spy, date)
    vix_spike = spike_score > 0.7
    
    if composite > 0.55 or drawdown or vix_spike:
        false_positives += 1

print(f"False positive signals in 2021: {false_positives}")

# Summary
print(f"\n{'='*60}")
print("BACKTEST SUMMARY")
print(f"{'='*60}")
for r in results:
    if r["signal_date"]:
        print(f"{r['crash']}: {r['days_early']} days early, avoided {r['avoided_loss']:.1f}%")
    else:
        print(f"{r['crash']}: MISSED")
print(f"\nFalse positives in 2021: {false_positives}")
print(f"{'='*60}")