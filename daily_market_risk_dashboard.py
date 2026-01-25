import os
import json
import smtplib
import matplotlib.pyplot as plt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from trade_signals import STATE_FILE as TRADE_STATE_FILE
from risk_indicators import (
    get_close_series,
    volatility_expansion_score,
    options_hedging_score,
    credit_stress_score,
    small_cap_score,
    gold_crypto_confirmation,
    volatility_compression_score,
    credit_complacency_score,
    breadth_divergence_score
)

EMAIL_FROM = "tshprung@gmail.com"
EMAIL_TO = "tshprung@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

STATE_FILE = "yesterday_state.json"
OUTPUT_FILE = "risk_dashboard.png"

GREEN, RED = 0.33, 0.66

# -----------------------------
# Compute all scores
# -----------------------------
gold_prices = get_close_series("GLD")
btc_prices = get_close_series("BTC-USD")
cross_score, gold_z, btc_z = gold_crypto_confirmation(gold_prices, btc_prices)

scores = {
    "Volatility expansion": volatility_expansion_score(),
    "Options hedging stress": options_hedging_score(),
    "Credit stress": credit_stress_score(),
    "Small-cap underperformance": small_cap_score(),
    "Cross-asset confirmation": cross_score,
    "Volatility compression": volatility_compression_score(),
    "Credit complacency": credit_complacency_score(),
    "Breadth divergence": breadth_divergence_score()
}

# Ensure numeric values
for k in scores:
    if scores[k] is None or not isinstance(scores[k], (int, float)):
        scores[k] = 0.0

risk_levels = list(scores.values())
indicators = list(scores.keys())

# -----------------------------
# Risk counting
# -----------------------------
red_count = int(sum(float(r) >= RED for r in risk_levels))
yellow_count = int(sum(GREEN <= float(r) < RED for r in risk_levels))

# -----------------------------
# Forced selling probability
# -----------------------------
forced_selling = min(
    0.35 * scores["Credit stress"] +
    0.35 * scores["Options hedging stress"] +
    0.30 * scores["Volatility expansion"],  # fixed key
    1.0
)
forced_pct = int(forced_selling * 100)

# -----------------------------
# Guidance & emoji
# -----------------------------
if red_count < 1:
    guidance, cash, emoji = "Normal conditions.", 10, "🟡"
elif red_count < 2:
    guidance, cash, emoji = "Elevated risk. Trim exposure.", 30, "🟠"
elif red_count < 3:
    guidance, cash, emoji = "Defensive posture.", 60, "🔴"
else:
    guidance, cash, emoji = "Crisis regime.", 85, "🚨"

# -----------------------------
# Track changes vs yesterday
# -----------------------------
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

# Save today's state
with open(STATE_FILE, "w") as f:
    json.dump({"red": red_count, "yellow": yellow_count}, f)

# -----------------------------
# Colors & plot
# -----------------------------
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

# -----------------------------
# Email
# -----------------------------
# Load trade signal
try:
    with open(TRADE_STATE_FILE) as f:
        trade_state = json.load(f)
except:
    trade_state = {"signal": "HOLD"}

trade_signal = trade_state.get("signal", "HOLD")

msg = MIMEMultipart("related")
msg["From"], msg["To"] = EMAIL_FROM, EMAIL_TO
msg["Subject"] = f"Market Risk {emoji} | Cash {cash}% | {change} | Trade Signal: {trade_signal}"

html = f"""
<p><b>Guidance:</b> {guidance}</p>
<p><b>Forced selling probability:</b> {forced_pct}%</p>
<p><b>{change}</b></p>
<img src="cid:dash">
"""
html += f"<p><b>Trade signal:</b> {trade_signal}</p>"

msg.attach(MIMEText(html, "html"))
with open(OUTPUT_FILE, "rb") as f:
    img = MIMEImage(f.read())
    img.add_header("Content-ID", "<dash>")
    msg.attach(img)

with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
    s.starttls()
    s.login(EMAIL_FROM, EMAIL_PASSWORD)
    s.send_message(msg)
