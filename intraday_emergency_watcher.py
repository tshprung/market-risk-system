import os
import json
import smtplib
from datetime import datetime
from email.message import EmailMessage

from risk_indicators import (
    volatility_expansion_score,
    credit_stress_score,
    options_hedging_score
)

STATE_FILE = "intraday_state.json"

EMAIL_FROM = "tshprung@gmail.com"
EMAIL_TO = "tshprung@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

score = (
    0.4 * volatility_expansion_score() +
    0.3 * credit_stress_score() +
    0.3 * options_hedging_score()
)

score = int(score * 100)

try:
    with open(STATE_FILE) as f:
        prev = json.load(f)
except:
    prev = {"score": 0}

if score < 40 or score <= prev["score"]:
    exit(0)

with open(STATE_FILE, "w") as f:
    json.dump({"score": score}, f)

level = "EARLY WARNING" if score < 60 else "HIGH RISK" if score < 80 else "EMERGENCY"

msg = EmailMessage()
msg["From"] = EMAIL_FROM
msg["To"] = EMAIL_TO
msg["Subject"] = f"{level} — Intraday Market Stress"

msg.set_content(
    f"Composite intraday risk score: {score}/100\n\n"
    "Signals indicate rapid deterioration in market microstructure.\n"
    "Often precedes broader risk-off moves.\n\n"
    f"UTC Time: {datetime.utcnow().isoformat()}"
)

with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
    s.starttls()
    s.login(EMAIL_FROM, EMAIL_PASSWORD)
    s.send_message(msg)
