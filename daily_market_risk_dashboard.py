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
    gold_crypto_confirmation,
    volatility_compression_score,
    credit_complacency_score,
    breadth_divergence_score,
    check_drawdown,
    check_recovery
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
    "Cross-asset confirmation": max(cross_score, 0.0),  # Only positive confirmation
    "Volatility compression": volatility_compression_score(),
    "Credit complacency": credit_complacency_score(),
    "Breadth divergence": breadth_divergence_score()
}

# Ensure numeric values
for k in scores:
    if scores[k] is None or not isinstance(scores[k], (int, float)):
        scores[k] = 0.0

# Check circuit breakers
drawdown_alert = check_drawdown()
recovery_signal = check_recovery()

sorted_items = sorted(scores.items(), key=lambda x: x[1])
indicators = [k for k, _ in sorted_items]
risk_levels = [v for _, v in sorted_items]

# -----------------------------
# Risk counting
# -----------------------------
red_count = int(sum(float(r) >= RED for r in risk_levels))
yellow_count = int(sum(GREEN <= float(r) < RED for r in risk_levels))

# Add drawdown to red count if triggered
if drawdown_alert:
    red_count += 1

# -----------------------------
# Forced selling probability
# -----------------------------
forced_selling = min(
    0.35 * scores["Credit stress"] +
    0.35 * scores["Options hedging stress"] +
    0.30 * scores["Volatility expansion"],
    1.0
)
forced_pct = int(forced_selling * 100)

# -----------------------------
# Guidance & emoji
# -----------------------------
if drawdown_alert:
    guidance, cash, emoji = "DRAWDOWN ALERT - Consider defensive action.", 80, "🚨"
elif red_count < 1:
    guidance, cash, emoji = "Normal conditions.", 10, "🟢"
elif red_count < 2:
    guidance, cash, emoji = "Elevated risk. Trim exposure.", 30, "🟡"
elif red_count < 3:
    guidance, cash, emoji = "Defensive posture recommended.", 60, "🟠"
else:
    guidance, cash, emoji = "Crisis regime. Preserve capital.", 85, "🔴"

# Recovery override
if recovery_signal and not drawdown_alert:
    guidance = "Recovery signals detected. " + guidance
    emoji = "🔄 " + emoji

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
    json.dump({"red": red_count, "yellow": yellow_count, "drawdown": drawdown_alert}, f)

# -----------------------------
# Colors & plot
# -----------------------------
colors = ["green" if r < GREEN else "gold" if r < RED else "red" for r in risk_levels]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(indicators, risk_levels, color=colors)
ax.axvline(GREEN, linestyle="--", alpha=0.3, color="gray")
ax.axvline(RED, linestyle="--", alpha=0.3, color="gray")
ax.set_xlim(0, 1)
ax.set_title("Daily Market Risk Dashboard", fontsize=14, fontweight="bold")
ax.set_xlabel("Risk Score (0 = safe, 1 = extreme)")

# Add warning banner if drawdown
if drawdown_alert:
    fig.text(0.5, 0.95, "⚠️ DRAWDOWN CIRCUIT BREAKER ACTIVE ⚠️", 
             ha='center', fontsize=12, color='red', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150)
plt.close()

# -----------------------------
# Email
# -----------------------------
# Load trade signal
try:
    with open(TRADE_STATE_FILE) as f:
        trade_state = json.load(f)
except:
    trade_state = {"signal": "HOLD", "reason": "No data"}

trade_signal = trade_state.get("signal", "HOLD")
trade_reason = trade_state.get("reason", "")
composite_pct = trade_state.get("composite_pct", 0)

# Determine subject urgency
if trade_signal == "SELL":
    subject_prefix = "🚨 SELL SIGNAL"
elif trade_signal == "REBUY":
    subject_prefix = "✅ REBUY SIGNAL"
elif drawdown_alert:
    subject_prefix = "⚠️ DRAWDOWN"
else:
    subject_prefix = f"Market Risk {emoji}"

msg = MIMEMultipart("related")
msg["From"], msg["To"] = EMAIL_FROM, EMAIL_TO
msg["Subject"] = f"{subject_prefix} | Cash {cash}% | {change}"

html = f"""
<h2>Market Risk Dashboard</h2>
<p><b>Overall guidance:</b> {guidance}</p>
<p><b>Suggested cash allocation:</b> {cash}%</p>
<p><b>Forced selling probability:</b> {forced_pct}%</p>
<p><b>Status change:</b> {change}</p>
<hr>
<h3>Trade Signal: {trade_signal}</h3>
<p><b>Composite risk:</b> {composite_pct}%</p>
<p><b>Reason:</b> {trade_reason}</p>
"""

if drawdown_alert:
    html += """
<p style="background-color: #ffcccc; padding: 10px; border-left: 4px solid red;">
<b>⚠️ DRAWDOWN ALERT:</b> Market has declined >5% in 3 days or >10% from 20-day high.
</p>
"""

if recovery_signal:
    html += """
<p style="background-color: #ccffcc; padding: 10px; border-left: 4px solid green;">
<b>🔄 RECOVERY SIGNAL:</b> VIX declining and credit stabilizing.
</p>
"""

html += f"""
<hr>
<img src="cid:dash" style="max-width: 100%; height: auto;">
<hr>
<p><b>Cross-asset details:</b><br>
Gold Z-score: {gold_z:.2f}<br>
BTC Z-score: {btc_z:.2f}<br>
Cross-asset confirmation: {cross_score:.2f}
</p>
<p><b>Component scores:</b><br>
"""

for indicator, score in scores.items():
    html += f"{indicator}: {score:.2f}<br>"

html += """
</p>
<p style="font-size: 0.9em; color: gray;">
<i>Automated alert from market risk system. Review before acting.</i>
</p>
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

print(f"Dashboard sent: {trade_signal} | Cash {cash}% | Composite {composite_pct}%")
