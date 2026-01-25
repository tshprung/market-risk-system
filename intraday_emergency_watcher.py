import os
import json
import smtplib
import yfinance as yf
from datetime import datetime
from email.message import EmailMessage

from risk_indicators import (
    volatility_expansion_score,
    credit_stress_score,
    options_hedging_score,
    gold_crypto_confirmation,
    risk_acceleration_score
)

STATE_FILE = "intraday_state.json"

EMAIL_FROM = "tshprung@gmail.com"
EMAIL_TO = "tshprung@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# -----------------------------
# Compute base scores
# -----------------------------
score = (
    0.4 * volatility_expansion_score() +
    0.3 * credit_stress_score() +
    0.3 * options_hedging_score()
)

sp500_prices = yf.download("^GSPC", period="3mo", progress=False)["Close"]
btc_prices = yf.download("BTC-USD", period="3mo", progress=False)["Close"]

btc_corr_score = btc_equity_correlation(sp500_prices, btc_prices)
score += 0.15 * btc_corr_score  # weight of correlation in total intraday score

if btc_corr_score > 0.6:
    alerts.append(f"BTC vs SPX negative correlation detected: {btc_corr_score:.2f}")

# -----------------------------
# Cross-asset confirmation
# -----------------------------
alerts = []

# Fetch gold and BTC prices
gold_prices = yf.download("GLD", period="3mo", progress=False)["Close"]
btc_prices = yf.download("BTC-USD", period="3mo", progress=False)["Close"]

confirm_score, gold_z, btc_z = gold_crypto_confirmation(
    gold_prices,
    btc_prices
)

score += int(confirm_score * 20)

if confirm_score > 0:
    alerts.append(
        f"Cross-asset confirmation: Gold Z={gold_z:.2f}, BTC Z={btc_z:.2f}"
    )

# Optional: add cross-asset confirmation multiplier (if you have another score)
# confirm = cross_asset_confirmation_score()  # only if defined
# score += 0.15 * confirm
# if confirm > 0.6:
#     alerts.append("Cross-asset risk-off confirmation (Gold/BTC)")

# -----------------------------
# Load previous state and compute acceleration
# -----------------------------
try:
    with open(STATE_FILE) as f:
        prev = json.load(f)
except:
    prev = {"score": 0, "recent_scores": []}

recent_scores = prev.get("recent_scores", [])

# Add current score normalized 0-1
recent_scores.append(score / 100)

# Compute acceleration score (0-1)
accel = risk_acceleration_score(recent_scores)

# Apply acceleration multiplier (max doubles the score)
score = min(int(score * (1 + accel)), 100)

# -----------------------------
# Exit if score not higher than previous or too low
# -----------------------------
if score < 40 or score <= prev.get("score", 0):
    # save recent_scores anyway
    recent_scores = recent_scores[-10:]
    with open(STATE_FILE, "w") as f:
        json.dump({"score": prev.get("score", 0), "recent_scores": recent_scores}, f)
    exit(0)

# -----------------------------
# Save state
# -----------------------------
recent_scores = recent_scores[-10:]  # keep last 10 entries
with open(STATE_FILE, "w") as f:
    json.dump({"score": score, "recent_scores": recent_scores}, f)

# -----------------------------
# Determine alert level
# -----------------------------
level = "EARLY WARNING" if score < 60 else "HIGH RISK" if score < 80 else "EMERGENCY"

# -----------------------------
# Send email alert
# -----------------------------
msg = EmailMessage()
msg["From"] = EMAIL_FROM
msg["To"] = EMAIL_TO
msg["Subject"] = f"{level} — Intraday Market Stress"

alert_text = "\n".join(alerts) if alerts else "No additional signals."

msg.set_content(
    f"Composite intraday risk score: {score}/100\n\n"
    f"{alert_text}\n\n"
    "Signals indicate rapid deterioration in market microstructure.\n"
    "Often precedes broader risk-off moves.\n\n"
    f"UTC Time: {datetime.utcnow().isoformat()}"
)

with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
    s.starttls()
    s.login(EMAIL_FROM, EMAIL_PASSWORD)
    s.send_message(msg)
