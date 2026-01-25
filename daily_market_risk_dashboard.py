import os
import json
import smtplib
import matplotlib.pyplot as plt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from risk_indicators import (
    volatility_expansion_score,
    options_hedging_score,
    credit_stress_score,
    small_cap_score
)

EMAIL_FROM = "tshprung@gmail.com"
EMAIL_TO = "tshprung@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

STATE_FILE = "yesterday_state.json"
OUTPUT_FILE = "risk_dashboard.png"

GREEN, RED = 0.33, 0.66

scores = {
    "Volatility expansion (early)": volatility_expansion_score(),
    "Options hedging stress": options_hedging_score(),
    "Credit stress": credit_stress_score(),
    "Small-cap underperformance": small_cap_score()
}

risk_levels = list(scores.values())
indicators = list(scores.keys())

red_count = sum(r >= RED for r in risk_levels)
yellow_count = sum(GREEN <= r < RED for r in risk_levels)

forced_selling = min(
    0.35 * scores["Credit stress"] +
    0.35 * scores["Options hedging stress"] +
    0.30 * scores["Volatility expansion (early)"],
    1.0
)

forced_pct = int(forced_selling * 100)

if red_count < 1:
    guidance, cash, emoji = "Normal conditions.", 10, "🟡"
elif red_count < 2:
    guidance, cash, emoji = "Elevated risk. Trim exposure.", 30, "🟠"
elif red_count < 3:
    guidance, cash, emoji = "Defensive posture.", 60, "🔴"
else:
    guidance, cash, emoji = "Crisis regime.", 85, "🚨"

try:
    with open(STATE_FILE) as f:
        prev = json.load(f)
except:
    prev = {}

change = "No material change"
if prev:
    if red_count > prev.get("red", 0):
        change = "↑ Risk increasing"
    elif red_count < prev.get("red", 0):
        change = "↓ Risk easing"

with open(STATE_FILE, "w") as f:
    json.dump({"red": red_count, "yellow": yellow_count}, f)

colors = ["green" if r < GREEN else "gold" if r < RED else "red" for r in risk_levels]

plt.figure(figsize=(9, 4))
plt.barh(indicators, risk_levels, color=colors)
plt.axvline(GREEN, linestyle="--", alpha=0.3)
plt.axvline(RED, linestyle="--", alpha=0.3)
plt.xlim(0, 1)
plt.title("Daily Market Risk Dashboard")
plt.tight_layout()
plt.savefig(OUTPUT_FILE)
plt.close()

msg = MIMEMultipart("related")
msg["From"], msg["To"] = EMAIL_FROM, EMAIL_TO
msg["Subject"] = f"Market Risk {emoji} | Cash {cash}% | {change}"

html = f"""
<p><b>Guidance:</b> {guidance}</p>
<p><b>Forced selling probability:</b> {forced_pct}%</p>
<p><b>{change}</b></p>
<img src="cid:dash">
"""

msg.attach(MIMEText(html, "html"))
with open(OUTPUT_FILE, "rb") as f:
    img = MIMEImage(f.read())
    img.add_header("Content-ID", "<dash>")
    msg.attach(img)

with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
    s.starttls()
    s.login(EMAIL_FROM, EMAIL_PASSWORD)
    s.send_message(msg)
