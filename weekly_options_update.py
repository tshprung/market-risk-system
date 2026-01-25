import smtplib
from email.message import EmailMessage
from risk_indicators import options_hedging_score, options_percentile

EMAIL_FROM = "tshprung@gmail.com"
EMAIL_TO = "tshprung@gmail.com"
EMAIL_PASSWORD = "efhk szhz humo zpmj"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

stress = options_hedging_score()
pct = options_percentile()

if stress < 0.3:
    regime = "LOW"
elif stress < 0.6:
    regime = "ELEVATED"
else:
    regime = "HIGH"

context = f"Current level is at the {pct}th percentile vs the past 5 years." if pct else ""

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

with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
    s.starttls()
    s.login(EMAIL_FROM, EMAIL_PASSWORD)
    s.send_message(msg)
