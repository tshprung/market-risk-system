import os
import numpy as np
from risk_indicators import (
    volatility_expansion_score,
    options_hedging_score,
    credit_stress_score,
    gold_crypto_confirmation,
    btc_equity_correlation,
    risk_acceleration_score,
    check_drawdown,
    check_recovery,
    get_persistent_risk,
    vix_spike_score,
    put_call_ratio_score,
    credit_spread_score,
    breadth_score,
    dollar_strength_score,
    yield_curve_score
)
import yfinance as yf
import json
from datetime import datetime, timedelta, timezone

STATE_FILE = "trade_signal_state.json"
SELL_COOLDOWN_DAYS = 5
BUY_COOLDOWN_DAYS = 3

# Thresholds
SELL_THRESHOLD = 0.55  # Lowered from 0.7 to catch more crashes
REBUY_THRESHOLD = 0.4
PERSISTENCE_DAYS = 2

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
spike_score    = vix_spike_score()
cross_score, gold_z, btc_z = gold_crypto_confirmation(gold_prices, btc_prices)
btc_corr_score = btc_equity_correlation(sp500_prices, btc_prices)

# New indicators
put_call_score = put_call_ratio_score()
spread_score   = credit_spread_score()
breadth_sc     = breadth_score()
dollar_score   = dollar_strength_score()
curve_score    = yield_curve_score()

# Ensure all scores are valid numbers
for score_var in [vol_score, credit_score, options_score, spike_score, cross_score, 
                  btc_corr_score, put_call_score, spread_score, breadth_sc, dollar_score, curve_score]:
    if score_var is None or np.isnan(score_var):
        score_var = 0.0

vol_score = 0.0 if vol_score is None or np.isnan(vol_score) else vol_score
credit_score = 0.0 if credit_score is None or np.isnan(credit_score) else credit_score
options_score = 0.0 if options_score is None or np.isnan(options_score) else options_score
spike_score = 0.0 if spike_score is None or np.isnan(spike_score) else spike_score
cross_score = 0.0 if cross_score is None or np.isnan(cross_score) else cross_score
btc_corr_score = 0.0 if btc_corr_score is None or np.isnan(btc_corr_score) else btc_corr_score
put_call_score = 0.0 if put_call_score is None or np.isnan(put_call_score) else put_call_score
spread_score = 0.0 if spread_score is None or np.isnan(spread_score) else spread_score
breadth_sc = 0.0 if breadth_sc is None or np.isnan(breadth_sc) else breadth_sc
dollar_score = 0.0 if dollar_score is None or np.isnan(dollar_score) else dollar_score
curve_score = 0.0 if curve_score is None or np.isnan(curve_score) else curve_score

# Composite (before acceleration)
base_composite = (
    0.20 * vol_score + 
    0.15 * credit_score + 
    0.15 * options_score +
    0.10 * spike_score +
    0.10 * put_call_score +   # NEW
    0.10 * spread_score +     # NEW
    0.08 * breadth_sc +       # NEW
    0.07 * dollar_score +     # NEW
    0.05 * curve_score        # NEW (longer lead time, lower weight)
)

# Track history
recent_scores.append(base_composite)
recent_scores = recent_scores[-20:]  # Keep 20 days

# Acceleration
accel_score = risk_acceleration_score(recent_scores)

# Final composite with acceleration boost
composite = min(max(base_composite + 0.15 * accel_score, 0.0), 1.0)

# Handle NaN
if np.isnan(composite):
    composite = 0.0

composite_pct = int(composite * 100)

# --- Decision Logic ---
signal = "HOLD"
reason = ""

# SELL CONDITIONS (any trigger)
drawdown_alert = check_drawdown()
vix_spike_alert = spike_score > 0.7  # Flash crash
persistent_high_risk = get_persistent_risk(recent_scores, SELL_THRESHOLD, PERSISTENCE_DAYS)

if drawdown_alert:
    if not in_cooldown(state, "SELL", SELL_COOLDOWN_DAYS):
        signal = "SELL"
        reason = "Drawdown circuit breaker triggered"
    else:
        signal = "HOLD (sell cooldown)"
        reason = "Drawdown detected but in cooldown"

elif vix_spike_alert:
    if not in_cooldown(state, "SELL", SELL_COOLDOWN_DAYS):
        signal = "SELL"
        reason = f"VIX spike detected ({spike_score:.2f})"
    else:
        signal = "HOLD (sell cooldown)"
        reason = "VIX spike but in cooldown"
        
elif composite > SELL_THRESHOLD and persistent_high_risk:
    if not in_cooldown(state, "SELL", SELL_COOLDOWN_DAYS):
        signal = "SELL"
        reason = f"Composite {composite_pct}% for {PERSISTENCE_DAYS}+ days"
    else:
        signal = "HOLD (sell cooldown)"
        reason = "High risk but in cooldown"

# REBUY CONDITIONS (recovery + low composite)
elif composite < REBUY_THRESHOLD and check_recovery():
    if not in_cooldown(state, "REBUY", BUY_COOLDOWN_DAYS):
        signal = "REBUY"
        reason = f"Recovery confirmed, composite {composite_pct}%"
    else:
        signal = "HOLD (rebuy cooldown)"
        reason = "Recovery detected but in cooldown"

else:
    reason = f"Composite {composite_pct}%, no clear signal"

# --- Update and Save State ---
if signal in ["SELL", "REBUY"]:
    state["last_action"] = signal
    state["last_action_time"] = datetime.now(timezone.utc).isoformat()

state.update({
    "signal": signal,
    "reason": reason,
    "recent_scores": recent_scores,
    "composite_pct": composite_pct,
    "vol_score": round(vol_score, 2),
    "credit_score": round(credit_score, 2),
    "options_score": round(options_score, 2),
    "spike_score": round(spike_score, 2),
    "put_call_score": round(put_call_score, 2),
    "spread_score": round(spread_score, 2),
    "breadth_score": round(breadth_sc, 2),
    "dollar_score": round(dollar_score, 2),
    "curve_score": round(curve_score, 2),
    "accel_score": round(accel_score, 2),
    "drawdown_alert": bool(drawdown_alert),
    "vix_spike_alert": bool(vix_spike_alert)
})

with open(STATE_FILE, "w") as f:
    json.dump(state, f, indent=4)

# --- Console Output ---
print(f"Composite: {composite_pct}/100 | Signal: {signal}")
print(f"Reason: {reason}")
print(f"Drawdown Alert: {drawdown_alert} | VIX Spike: {vix_spike_alert}")
print(f"Core: Vol={vol_score:.2f} Credit={credit_score:.2f} Options={options_score:.2f} Spike={spike_score:.2f}")
print(f"New: PutCall={put_call_score:.2f} Spread={spread_score:.2f} Breadth={breadth_sc:.2f} Dollar={dollar_score:.2f} Curve={curve_score:.2f}")
print(f"Cross-asset: Gold Z={gold_z:.2f} BTC Z={btc_z:.2f}")
print(f"Acceleration: {accel_score:.2f}")