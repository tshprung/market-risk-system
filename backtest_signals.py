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
    window = vix.loc[:date].tail(60)
    if len(window) < 60:
        return 0.0
    roc = window.pct_change(3).dropna()
    mean, std = roc.mean(), roc.std()
    if std == 0:
        return 0.0
    z = (roc.iloc[-1] - mean) / std
    return min(max(abs(z) / 3.0, 0.0), 1.0)

def simple_credit_score(hyg, ief, date):
    hyg_w = hyg.loc[:date].tail(120)
    ief_w = ief.loc[:date].tail(120)
    if len(hyg_w) < 120 or len(ief_w) < 120:
        return 0.0
    rel = hyg_w.pct_change() - ief_w.pct_change()
    rel = rel.dropna()
    mean, std = rel.mean(), rel.std()
    if std == 0:
        return 0.0
    z = (rel.iloc[-1] - mean) / std
    return min(max(abs(z) / 3.0, 0.0), 1.0)

def simple_drawdown(spy, date, short_days=3, long_days=20):
    prices = spy.loc[:date].tail(long_days + 1)
    if len(prices) < long_days:
        return False
    short_drop = (prices.iloc[-1] / prices.iloc[-short_days-1]) - 1
    long_peak = prices.iloc[-long_days:].max()
    long_drop = (prices.iloc[-1] / long_peak) - 1
    return short_drop < -0.05 or long_drop < -0.10

# Load historical data
print("Loading data...")
spy = yf.download("SPY", start="2017-01-01", end="2025-01-25", progress=False)["Close"]
vix = yf.download("^VIX", start="2017-01-01", end="2025-01-25", progress=False)["Close"]
hyg = yf.download("HYG", start="2017-01-01", end="2025-01-25", progress=False)["Close"]
ief = yf.download("IEF", start="2017-01-01", end="2025-01-25", progress=False)["Close"]

# Define crash periods (rough dates)
crashes = [
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
        composite = 0.5 * vol_score + 0.5 * credit_score
        drawdown = simple_drawdown(spy, date)
        
        if composite > 0.7 or drawdown:
            sell_signal_date = date
            sell_composite = composite
            break
    
    if sell_signal_date:
        days_early = (crash_start - sell_signal_date).days
        sell_price = spy.loc[sell_signal_date]
        crash_low = spy[crash_start:crash_end].min()
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
    composite = 0.5 * vol_score + 0.5 * credit_score
    drawdown = simple_drawdown(spy, date)
    
    if composite > 0.7 or drawdown:
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
