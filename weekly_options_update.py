import os
import smtplib
from email.message import EmailMessage
from risk_indicators import options_hedging_score, options_percentile

EMAIL_FROM = "tshprung@gmail.com"
EMAIL_TO = "tshprung@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# -----------------------------
# Compute weekly options risk
# -----------------------------
stress = options_hedging_score()
if stress is None:
    stress = 0.0

pct = options_percentile()

# Determine regime
if stress < 0.3:
    regime = "LOW"
elif stress < 0.6:
    regime = "ELEVATED"
else:
    regime = "HIGH"

context = f"Current level is at the {pct}th percentile vs the past 5 years." if pct else ""

# -----------------------------
# Compose email
# -----------------------------
msg = EmailMessage()
msg["From"] = EMAIL_FROM
msg["To"] = EMAIL_TO
msg["Subject"] = f"Weekly Options Risk Update — {regime}"

msg.set_content(
    f"Weekly Options Risk Summary\n\n"
    f"Regime: {regime}\n"
    f"Stress level: {int(stress*100)}%\n"
    f"{context}\n\n"
    "This is a structural context update, not a trading signal."
)

# -----------------------------
# Send email
# -----------------------------
with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
    s.starttls()
    s.login(EMAIL_FROM, EMAIL_PASSWORD)
    s.send_message(msg)
