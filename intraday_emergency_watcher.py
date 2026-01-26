import os
import json
import smtplib
import numpy as np
import yfinance as yf
from datetime import datetime
from email.message import EmailMessage
from trade_signals import STATE_FILE as TRADE_STATE_FILE

from risk_indicators import (
    volatility_expansion_score,
    credit_stress_score,
    options_hedging_score,
    gold_crypto_confirmation,
    risk_acceleration_score,
    btc_equity_correlation,
    check_drawdown,
    get_close_series
)

STATE_FILE = "intraday_state.json"

EMAIL_FROM = "tshprung@gmail.com"
EMAIL_TO = "tshprung@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Thresholds
EARLY_WARNING_THRESHOLD = 40
HIGH_RISK_THRESHOLD = 60
EMERGENCY_THRESHOLD = 80

# -----------------------------
# Compute base scores
# -----------------------------
score = (
    0.35 * volatility_expansion_score() +
    0.30 * credit_stress_score() +
    0.25 * options_hedging_score()
)

alerts = []

# -----------------------------
# Check drawdown circuit breaker
# -----------------------------
drawdown_alert = check_drawdown()
if drawdown_alert:
    score += 20  # Major boost to score
    alerts.append("⚠️ DRAWDOWN CIRCUIT BREAKER TRIGGERED")

# -----------------------------
# BTC/SPX correlation
# -----------------------------
sp500_prices = get_close_series("^GSPC", "3mo")
btc_prices = get_close_series("BTC-USD", "3mo")

btc_corr_score = btc_equity_correlation(sp500_prices, btc_prices)
score += 0.10 * btc_corr_score * 100

if btc_corr_score > 0.6:
    alerts.append(f"BTC/SPX negative correlation: {btc_corr_score:.2f}")

# -----------------------------
# Cross-asset confirmation
# -----------------------------
gold_prices = get_close_series("GLD", "3mo")
confirm_score, gold_z, btc_z = gold_crypto_confirmation(gold_prices, btc_prices)

# Only add if risk-off (positive confirmation)
if confirm_score > 0:
    score += confirm_score * 10
    alerts.append(f"Cross-asset risk-off: Gold Z={gold_z:.2f}, BTC Z={btc_z:.2f}")

# -----------------------------
# Load trade signal
# -----------------------------
try:
    with open(TRADE_STATE_FILE) as f:
        trade_state = json.load(f)
except:
    trade_state = {"signal": "HOLD"}

trade_signal = trade_state.get("signal", "HOLD")
trade_reason = trade_state.get("reason", "")

if trade_signal in ["SELL", "REBUY"]:
    alerts.append(f"Trade signal: {trade_signal} - {trade_reason}")

# -----------------------------
# Load previous state and compute acceleration
# -----------------------------
try:
    with open(STATE_FILE) as f:
        prev = json.load(f)
except:
    prev = {"score": 0, "recent_scores": [], "alert_count": 0}

recent_scores = prev.get("recent_scores", [])

# Add current score normalized 0-1
recent_scores.append(score / 100)

# Compute acceleration score (0-1)
accel = risk_acceleration_score(recent_scores)

# Apply acceleration multiplier (can increase score up to 50%)
score = min(int(np.nan_to_num(score * (1 + 0.5 * accel))), 100)

if accel > 0.6:
    alerts.append(f"Risk accelerating (accel score: {accel:.2f})")

# -----------------------------
# Intraday momentum check
# -----------------------------
spy_1d = get_close_series("SPY", "5d", "1m")  # 1-minute bars
if len(spy_1d) > 60:
    hour_return = (spy_1d.iloc[-1] / spy_1d.iloc[-60]) - 1
    if hour_return < -0.02:  # -2% in last hour
        alerts.append(f"Intraday momentum breakdown: {hour_return*100:.1f}% last hour")
        score = min(score + 15, 100)

# -----------------------------
# Exit if score not higher than previous or too low
# -----------------------------
prev_score = prev.get("score", 0)
alert_count = prev.get("alert_count", 0)

# Only send if:
# 1. Score >= threshold AND
# 2. (Score increased OR we haven't alerted in last 3 runs)
should_alert = score >= EARLY_WARNING_THRESHOLD and (score > prev_score or alert_count >= 3)

if not should_alert:
    # Save state but don't send
    recent_scores = recent_scores[-10:]
    with open(STATE_FILE, "w") as f:
        json.dump({
            "score": score, 
            "recent_scores": recent_scores,
            "alert_count": alert_count + 1
        }, f)
    print(f"Score {score}/100 - no alert sent (prev: {prev_score}, count: {alert_count})")
    exit(0)

# -----------------------------
# Save state (reset alert count)
# -----------------------------
recent_scores = recent_scores[-10:]
with open(STATE_FILE, "w") as f:
    json.dump({
        "score": score, 
        "recent_scores": recent_scores,
        "alert_count": 0  # Reset since we're sending alert
    }, f)

# -----------------------------
# Determine alert level
# -----------------------------
if score < HIGH_RISK_THRESHOLD:
    level = "EARLY WARNING"
    emoji = "🟡"
elif score < EMERGENCY_THRESHOLD:
    level = "HIGH RISK"
    emoji = "🟠"
else:
    level = "EMERGENCY"
    emoji = "🚨"

# -----------------------------
# Send email alert
# -----------------------------
msg = EmailMessage()
msg["From"] = EMAIL_FROM
msg["To"] = EMAIL_TO
msg["Subject"] = f"{emoji} {level} — Intraday Risk {score}/100"

alert_text = "\n• " + "\n• ".join(alerts) if alerts else "No specific signals."

body = f"""
Composite Intraday Risk Score: {score}/100
Previous score: {prev_score}/100
Change: {score - prev_score:+d}

LEVEL: {level}

ACTIVE SIGNALS:
{alert_text}

TRADE SIGNAL: {trade_signal}
{trade_reason}

---
Rapid deterioration in market microstructure detected.
UTC Time: {datetime.utcnow().isoformat()}

This is an automated alert. Review conditions before acting.
"""

msg.set_content(body)

try:
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls()
        s.login(EMAIL_FROM, EMAIL_PASSWORD)
        s.send_message(msg)
    print(f"Alert sent: {level} | Score {score}/100")
except Exception as e:
    print(f"Email failed: {e}")
