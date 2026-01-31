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
    yield_curve_score,
    debt_ceiling_stress_score,
    treasury_stress_score,
    budget_vote_risk_score,
    days_to_debt_ceiling,
    earnings_volatility_score,
    is_earnings_season,
    congressional_budget_risk_score,
    get_budget_risk_details
)
import yfinance as yf
import json
from datetime import datetime, timedelta, timezone

STATE_FILE = "trade_signal_state.json"
SELL_COOLDOWN_DAYS = 5
BUY_COOLDOWN_DAYS = 3

# UPDATED: Lower thresholds for more sensitivity
SELL_THRESHOLD = 0.50  # Lowered from 0.55
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

def safe_float(value):
    """Convert to float and handle None/NaN"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(value)

# --- Fetch Prices ---
sp500_prices = yf.download("^GSPC", period="3mo", progress=False)["Close"]
btc_prices   = yf.download("BTC-USD", period="3mo", progress=False)["Close"]
gold_prices  = yf.download("GLD", period="3mo", progress=False)["Close"]

# --- Compute Core Indicators ---
vol_score      = safe_float(volatility_expansion_score())
credit_score   = safe_float(credit_stress_score())
options_score  = safe_float(options_hedging_score())
spike_score    = safe_float(vix_spike_score())
cross_score, gold_z, btc_z = gold_crypto_confirmation(gold_prices, btc_prices)
cross_score    = safe_float(cross_score)
gold_z         = safe_float(gold_z)
btc_z          = safe_float(btc_z)
btc_corr_score = safe_float(btc_equity_correlation(sp500_prices, btc_prices))

# --- Market Structure Indicators ---
put_call_score    = safe_float(put_call_ratio_score())
spread_score      = safe_float(credit_spread_score())
breadth_sc        = safe_float(breadth_score())
dollar_score      = safe_float(dollar_strength_score())
curve_score       = safe_float(yield_curve_score())

# --- Fiscal Risk Indicators ---
debt_stress       = safe_float(debt_ceiling_stress_score())
treasury_stress   = safe_float(treasury_stress_score())
budget_risk       = safe_float(budget_vote_risk_score())
congressional_risk = safe_float(congressional_budget_risk_score())

# --- NEW: Earnings Risk ---
earnings_risk = safe_float(earnings_volatility_score())
is_earnings, earnings_intensity, earnings_desc = is_earnings_season()

# --- Get context ---
days_to_x, is_near_deadline = days_to_debt_ceiling()
budget_details = get_budget_risk_details()

# ======================
# REWEIGHTED COMPOSITE
# ======================

# Base weights (total: 100%)
base_composite = (
    0.15 * vol_score +         
    0.15 * credit_score +      
    0.15 * options_score +     
    0.10 * spike_score +       
    0.12 * put_call_score +    
    0.10 * spread_score +      
    0.10 * breadth_sc +        
    0.05 * dollar_score +      
    0.08 * curve_score         
)

# ======================
# EVENT-BASED BOOSTS
# ======================

event_boost = 0.0

# 1. Debt Ceiling (up to 20% boost)
if is_near_deadline:
    debt_boost = 0.20 * budget_risk
    event_boost += debt_boost

# 2. Congressional Budget Issues (up to 15% boost)
if congressional_risk > 0.4:
    congress_boost = 0.15 * congressional_risk
    event_boost += congress_boost

# 3. Earnings Season (up to 10% boost)
if is_earnings and earnings_risk > 0.3:
    earnings_boost = 0.10 * earnings_risk
    event_boost += earnings_boost

# Apply event boosts (cap total composite at 1.0)
base_composite = min(base_composite + event_boost, 1.0)

# Track history
recent_scores.append(base_composite)
recent_scores = recent_scores[-20:]

# Acceleration
accel_score = safe_float(risk_acceleration_score(recent_scores))

# Final composite with acceleration boost
composite = min(max(base_composite + 0.15 * accel_score, 0.0), 1.0)

composite_pct = int(composite * 100)

# ======================
# DECISION LOGIC
# ======================

signal = "HOLD"
reason = ""

# --- SELL CONDITIONS ---
drawdown_alert = check_drawdown()  # NOW: -2% in 2 days OR -5% in 10 days
vix_spike_alert = spike_score > 0.7
persistent_high_risk = get_persistent_risk(recent_scores, SELL_THRESHOLD, PERSISTENCE_DAYS)

# Emergency conditions
debt_ceiling_emergency = is_near_deadline and days_to_x <= 14 and budget_risk > 0.6
congressional_emergency = congressional_risk > 0.7  # Shutdown imminent
earnings_crash = is_earnings and earnings_risk > 0.75 and composite > 0.60

if debt_ceiling_emergency:
    if not in_cooldown(state, "SELL", SELL_COOLDOWN_DAYS):
        signal = "SELL"
        reason = f"Debt ceiling deadline in {days_to_x} days, Treasury stress elevated"
    else:
        signal = "HOLD (sell cooldown)"
        reason = f"Debt ceiling risk but in cooldown ({days_to_x} days to X-date)"

elif congressional_emergency:
    if not in_cooldown(state, "SELL", SELL_COOLDOWN_DAYS):
        signal = "SELL"
        reason = f"Congressional budget crisis - {budget_details}"
    else:
        signal = "HOLD (sell cooldown)"
        reason = f"Budget crisis but in cooldown"

elif drawdown_alert:
    if not in_cooldown(state, "SELL", SELL_COOLDOWN_DAYS):
        signal = "SELL"
        reason = "Drawdown circuit breaker triggered (-2% in 2d or -5% in 10d)"
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

elif earnings_crash:
    if not in_cooldown(state, "SELL", SELL_COOLDOWN_DAYS):
        signal = "SELL"
        reason = f"{earnings_desc} volatility spike - composite {composite_pct}%"
    else:
        signal = "HOLD (sell cooldown)"
        reason = f"Earnings volatility but in cooldown"
        
elif composite > SELL_THRESHOLD and persistent_high_risk:
    if not in_cooldown(state, "SELL", SELL_COOLDOWN_DAYS):
        signal = "SELL"
        reason = f"Composite {composite_pct}% for {PERSISTENCE_DAYS}+ days"
    else:
        signal = "HOLD (sell cooldown)"
        reason = "High risk but in cooldown"

# --- REBUY CONDITIONS ---
elif composite < REBUY_THRESHOLD and check_recovery():
    if not in_cooldown(state, "REBUY", BUY_COOLDOWN_DAYS):
        signal = "REBUY"
        reason = f"Recovery confirmed, composite {composite_pct}%"
    else:
        signal = "HOLD (rebuy cooldown)"
        reason = "Recovery detected but in cooldown"

else:
    # Build context-aware reason
    context_parts = []
    if is_near_deadline:
        context_parts.append(f"debt ceiling {days_to_x}d")
    if congressional_risk > 0.4:
        context_parts.append("budget issues")
    if is_earnings:
        context_parts.append(earnings_desc)
    
    context_str = ", ".join(context_parts) if context_parts else "monitoring"
    reason = f"Composite {composite_pct}% - {context_str}"

# ======================
# UPDATE STATE
# ======================

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
    "debt_stress": round(debt_stress, 2),
    "treasury_stress": round(treasury_stress, 2),
    "budget_risk": round(budget_risk, 2),
    "congressional_risk": round(congressional_risk, 2),
    "earnings_risk": round(earnings_risk, 2),
    "accel_score": round(accel_score, 2),
    "drawdown_alert": bool(drawdown_alert),
    "vix_spike_alert": bool(vix_spike_alert),
    "debt_ceiling_alert": bool(debt_ceiling_emergency),
    "congressional_alert": bool(congressional_emergency),
    "earnings_alert": bool(is_earnings),
    "days_to_debt_ceiling": int(days_to_x),
    "budget_details": budget_details,
    "earnings_details": earnings_desc if is_earnings else "No earnings"
})

with open(STATE_FILE, "w") as f:
    json.dump(state, f, indent=4)

# ======================
# CONSOLE OUTPUT
# ======================

print(f"Composite: {composite_pct}/100 | Signal: {signal}")
print(f"Reason: {reason}")
print(f"Drawdown Alert: {drawdown_alert} | VIX Spike: {vix_spike_alert}")

if is_near_deadline:
    print(f"⚠️ DEBT CEILING: {days_to_x} days to X-date | Budget risk: {budget_risk:.2f}")

if congressional_risk > 0.4:
    print(f"⚠️ BUDGET: {budget_details} | Risk: {congressional_risk:.2f}")

if is_earnings:
    print(f"📊 EARNINGS: {earnings_desc} | Risk: {earnings_risk:.2f}")

print(f"Core: Vol={vol_score:.2f} Credit={credit_score:.2f} Options={options_score:.2f} Spike={spike_score:.2f}")
print(f"Market: PutCall={put_call_score:.2f} Spread={spread_score:.2f} Breadth={breadth_sc:.2f}")
print(f"Macro: Dollar={dollar_score:.2f} Curve={curve_score:.2f}")
print(f"Fiscal: DebtStress={debt_stress:.2f} TreasuryStress={treasury_stress:.2f} Congressional={congressional_risk:.2f}")
print(f"Events: Earnings={earnings_risk:.2f}")
print(f"Cross-asset: Gold Z={gold_z:.2f} BTC Z={btc_z:.2f}")
print(f"Acceleration: {accel_score:.2f}")
