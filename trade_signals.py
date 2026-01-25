from risk_indicators import (
    volatility_expansion_score,
    options_hedging_score,
    credit_stress_score,
    gold_crypto_confirmation,
    btc_equity_correlation,
    risk_acceleration_score
)
import yfinance as yf
import json
import os

STATE_FILE = "trade_signal_state.json"

# --- Load recent intraday scores for acceleration ---
try:
    with open(STATE_FILE) as f:
        prev = json.load(f)
except:
    prev = {"recent_scores": []}

recent_scores = prev.get("recent_scores", [])

# --- Fetch prices ---
sp500_prices = yf.download("^GSPC", period="3mo", progress=False)["Close"]
btc_prices   = yf.download("BTC-USD", period="3mo", progress=False)["Close"]
gold_prices  = yf.download("GLD", period="3mo", progress=False)["Close"]

# --- Compute indicators ---
vol_score   = volatility_expansion_score()
credit_score= credit_stress_score()
options_score= options_hedging_score()
cross_score, gold_z, btc_z = gold_crypto_confirmation(gold_prices, btc_prices)
btc_corr_score = btc_equity_correlation(sp500_prices, btc_prices)

recent_scores.append(vol_score*0.4 + credit_score*0.3 + options_score*0.3)
if len(recent_scores) > 20:
    recent_scores = recent_scores[-20:]  # keep last 20

accel_score = risk_acceleration_score(recent_scores)

# --- Composite score ---
composite = (
    0.4*vol_score +
    0.3*credit_score +
    0.3*options_score +
    0.15*cross_score +
    0.15*btc_corr_score +
    0.15*accel_score
)

composite = min(max(composite, 0.0), 1.0)  # clamp 0..1
composite_pct = int(composite*100)

# --- Sell / Rebuy Logic ---
signal = "HOLD"
if composite_pct >= 80:
    signal = "SELL — high risk"
elif composite_pct <= 50 and prev.get("composite_pct", 100) >= 80:
    signal = "REBUY — risk easing"

# --- Save state ---
with open(STATE_FILE, "w") as f:
    json.dump({
        "recent_scores": recent_scores,
        "composite_pct": composite_pct,
        "signal": signal
    }, f)

# --- Output for logs / email ---
print(f"Composite score: {composite_pct}/100")
print(f"Gold Z={gold_z:.2f}, BTC Z={btc_z:.2f}, BTC/SPX corr={btc_corr_score:.2f}, Accel={accel_score:.2f}")
print(f"Trade signal: {signal}")
