import os
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
from datetime import datetime, timedelta, timezone

STATE_FILE = "trade_signal_state.json"
SELL_COOLDOWN_DAYS = 5
BUY_COOLDOWN_DAYS = 3

# --- Load State ---
if not os.path.exists(STATE_FILE):
    state = {"signal": "HOLD", "last_action": None, "last_action_time": None, "recent_scores": []}
else:
    with open(STATE_FILE) as f:
        state = json.load(f)

recent_scores = state.get("recent_scores", [])

def in_cooldown(state, action, days):
    if state.get("last_action") != action or not state.get("last_action_time"):
        return False
    last = datetime.fromisoformat(state["last_action_time"])
    return datetime.now(timezone.utc) < last + timedelta(days=days)

# --- Fetch Prices ---
sp500_prices = yf.download("^GSPC", period="3mo", progress=False)["Close"]
btc_prices   = yf.download("BTC-USD", period="3mo", progress=False)["Close"]
gold_prices  = yf.download("GLD", period="3mo", progress=False)["Close"]

# --- Compute Indicators ---
vol_score      = volatility_expansion_score()
credit_score   = credit_stress_score()
options_score  = options_hedging_score()
cross_score, gold_z, btc_z = gold_crypto_confirmation(gold_prices, btc_prices)
btc_corr_score = btc_equity_correlation(sp500_prices, btc_prices)

# Acceleration
recent_scores.append(vol_score*0.4 + credit_score*0.3 + options_score*0.3)
recent_scores = recent_scores[-20:] 
accel_score = risk_acceleration_score(recent_scores)

# Composite (Clamped 0-1)
composite = min(max(
    (0.4*vol_score + 0.3*credit_score + 0.3*options_score + 
     0.15*cross_score + 0.15*btc_corr_score + 0.15*accel_score), 0.0), 1.0)
composite_pct = int(composite * 100)

# --- Decision Logic ---
signal = "HOLD"
sell_condition = composite > 0.7  # Example threshold
rebuy_condition = composite < 0.3 # Example threshold

if sell_condition:
    if not in_cooldown(state, "SELL", SELL_COOLDOWN_DAYS):
        signal = "SELL"
    else:
        signal = "HOLD (sell cooldown)"
elif rebuy_condition:
    if not in_cooldown(state, "REBUY", BUY_COOLDOWN_DAYS):
        signal = "REBUY"
    else:
        signal = "HOLD (rebuy cooldown)"

# --- Update and Save State ---
if signal in ["SELL", "REBUY"]:
    state["last_action"] = signal
    state["last_action_time"] = datetime.now(timezone.utc).isoformat()

state.update({
    "signal": signal,
    "recent_scores": recent_scores,
    "composite_pct": composite_pct
})

with open(STATE_FILE, "w") as f:
    json.dump(state, f, indent=4)

print(f"Composite: {composite_pct}/100 | Signal: {signal}")
print(f"Gold Z: {gold_z:.2f}, BTC Z: {btc_z:.2f}, BTC/SPX Corr: {btc_corr_score:.2f}")
