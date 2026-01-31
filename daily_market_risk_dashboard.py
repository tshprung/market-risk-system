import os
import json
import smtplib
import numpy as np
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
    check_recovery,
    put_call_ratio_score,
    credit_spread_score,
    breadth_score,
    dollar_strength_score,
    yield_curve_score,
    debt_ceiling_stress_score,
    treasury_stress_score,
    budget_vote_risk_score,
    days_to_debt_ceiling
)

EMAIL_FROM = "tshprung@gmail.com"
EMAIL_TO = "tshprung@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

STATE_FILE = "yesterday_state.json"
OUTPUT_FILE = "risk_dashboard.png"

GREEN, RED = 0.33, 0.66

def safe_float(value):
    """Convert to float and handle None/NaN"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(value)

# -----------------------------
# Compute all scores
# -----------------------------
gold_prices = get_close_series("GLD")
btc_prices = get_close_series("BTC-USD")
cross_score, gold_z, btc_z = gold_crypto_confirmation(gold_prices, btc_prices)

scores = {
    "Volatility expansion": safe_float(volatility_expansion_score()),
    "Options hedging stress": safe_float(options_hedging_score()),
    "Credit stress": safe_float(credit_stress_score()),
    "Cross-asset confirmation": max(safe_float(cross_score), 0.0),
    "Volatility compression": safe_float(volatility_compression_score()),
    "Credit complacency": safe_float(credit_complacency_score()),
    "Breadth divergence": safe_float(breadth_divergence_score()),
    "Put/Call ratio": safe_float(put_call_ratio_score()),
    "Credit spread (HY-IG)": safe_float(credit_spread_score()),
    "Market breadth": safe_float(breadth_score()),
    "Dollar strength": safe_float(dollar_strength_score()),
    "Yield curve inversion": safe_float(yield_curve_score()),
    "Debt ceiling stress": safe_float(debt_ceiling_stress_score()),
    "Treasury volatility": safe_float(treasury_stress_score()),
    "Budget vote risk": safe_float(budget_vote_risk_score())
}

# Get debt ceiling info
days_to_x, is_near_deadline = days_to_debt_ceiling()

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

# Add debt ceiling alert if critical
if is_near_deadline and days_to_x <= 14:
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
    guidance, cash, emoji, risk_level = "DRAWDOWN ALERT - Consider defensive action.", 80, "🚨", "CRISIS"
elif is_near_deadline and days_to_x <= 14:
    guidance, cash, emoji, risk_level = f"DEBT CEILING in {days_to_x} days - High uncertainty.", 70, "⚠️", "CRITICAL"
elif red_count < 1:
    guidance, cash, emoji, risk_level = "Normal conditions.", 10, "🟢", "LOW"
elif red_count < 2:
    guidance, cash, emoji, risk_level = "Elevated risk. Trim exposure.", 30, "🟡", "ELEVATED"
elif red_count < 3:
    guidance, cash, emoji, risk_level = "Defensive posture recommended.", 60, "🟠", "HIGH"
else:
    guidance, cash, emoji, risk_level = "Crisis regime. Preserve capital.", 85, "🔴", "CRITICAL"

# Recovery override
if recovery_signal and not drawdown_alert:
    guidance = "Recovery signals detected. " + guidance
    emoji = "🔄 " + emoji

# Debt ceiling warning
if is_near_deadline and days_to_x <= 60:
    guidance = f"[Debt ceiling in {days_to_x} days] " + guidance

# -----------------------------
# Track changes vs yesterday
# -----------------------------
try:
    with open(STATE_FILE) as f:
        prev = json.load(f)
except:
    prev = {}

change = "No material change"
score_changes = {}

if prev:
    if red_count > prev.get("red", 0):
        change = "↑ Risk increasing"
    elif red_count < prev.get("red", 0):
        change = "↓ Risk easing"
    
    # Track individual score changes
    prev_scores = prev.get("scores", {})
    for k, v in scores.items():
        prev_val = prev_scores.get(k, v)
        if v > prev_val + 0.1:
            score_changes[k] = "↑"
        elif v < prev_val - 0.1:
            score_changes[k] = "↓"
        else:
            score_changes[k] = "→"
else:
    # First run - no previous data
    score_changes = {k: "→" for k in scores.keys()}

# Save today's state
with open(STATE_FILE, "w") as f:
    json.dump({
        "red": int(red_count), 
        "yellow": int(yellow_count), 
        "drawdown": bool(drawdown_alert),
        "debt_ceiling_days": int(days_to_x),
        "scores": {k: float(v) for k, v in scores.items()}
    }, f)

# -----------------------------
# Colors & plot
# -----------------------------
colors = ["green" if r < GREEN else "gold" if r < RED else "red" for r in risk_levels]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(indicators, risk_levels, color=colors)
ax.axvline(GREEN, linestyle="--", alpha=0.3, color="gray")
ax.axvline(RED, linestyle="--", alpha=0.3, color="gray")
ax.set_xlim(0, 1)
ax.set_title("Daily Market Risk Dashboard", fontsize=14, fontweight="bold")
ax.set_xlabel("Risk Score (0 = safe, 1 = extreme)")

# Add warning banners
banner_y = 0.95

if drawdown_alert:
    fig.text(0.5, banner_y, "⚠️ DRAWDOWN CIRCUIT BREAKER ACTIVE ⚠️", 
             ha='center', fontsize=12, color='red', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    banner_y -= 0.05

if is_near_deadline and days_to_x <= 14:
    fig.text(0.5, banner_y, f"⚠️ DEBT CEILING DEADLINE IN {days_to_x} DAYS ⚠️", 
             ha='center', fontsize=11, color='darkred', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))

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
elif is_near_deadline and days_to_x <= 14:
    subject_prefix = f"⚠️ DEBT CEILING {days_to_x}d"
elif drawdown_alert:
    subject_prefix = "⚠️ DRAWDOWN"
else:
    subject_prefix = f"Market Risk {emoji}"

msg = MIMEMultipart("related")
msg["From"], msg["To"] = EMAIL_FROM, EMAIL_TO
msg["Subject"] = f"{subject_prefix} | RISK: {risk_level} | Cash {cash}% | {change}"

html = f"""
<h2>Market Risk Dashboard</h2>
<p><b>Overall guidance:</b> {guidance}</p>
<p><b>Suggested cash allocation:</b> {cash}%</p>
<p><b>Forced selling probability:</b> {forced_pct}%</p>
<p><b>Status change:</b> {change}</p>
"""

# Debt ceiling warning
if is_near_deadline:
    html += f"""
<div style="background-color: {'#ffcccc' if days_to_x <= 14 else '#fff3cd'}; 
     padding: 15px; border-left: 5px solid {'red' if days_to_x <= 14 else 'orange'}; 
     margin: 10px 0;">
<h3 style="margin: 0 0 10px 0;">⚠️ Debt Ceiling Alert</h3>
<p style="margin: 5px 0;"><b>Days to X-date:</b> {days_to_x}</p>
<p style="margin: 5px 0;"><b>Budget risk score:</b> {scores['Budget vote risk']:.2f}</p>
<p style="margin: 5px 0;"><b>Treasury stress:</b> {scores['Treasury volatility']:.2f}</p>
<p style="margin: 5px 0; font-size: 0.9em;">Historical pattern: Market typically ignores until 2 weeks before deadline, then sharp volatility spike (2011: -17% drop)</p>
</div>
"""

html += f"""
<hr>
<div style="background-color: {'#ffcccc' if trade_signal == 'SELL' else '#ccffcc' if trade_signal == 'REBUY' else '#f0f0f0'}; 
     padding: 15px; border-left: 5px solid {'red' if trade_signal == 'SELL' else 'green' if trade_signal == 'REBUY' else 'gray'}; 
     margin: 10px 0;">
<h3 style="margin: 0 0 10px 0;">Trade Signal: {trade_signal}</h3>
<p style="margin: 5px 0;"><b>Composite risk:</b> {composite_pct}%</p>
<p style="margin: 5px 0;"><b>Reason:</b> {trade_reason}</p>
</div>
"""

# Action items based on risk level
actions = {
    "LOW": "• Maintain normal allocation\n• Consider adding to positions on dips",
    "ELEVATED": "• Trim high-beta positions\n• Raise stop-losses\n• Monitor closely",
    "HIGH": "• Reduce exposure to 40-60% cash\n• Avoid new positions\n• Protect gains",
    "CRITICAL": "• Move to 85%+ cash\n• Hedge remaining positions\n• Wait for clarity\n• Watch debt ceiling closely",
    "CRISIS": "• Maximum defensive posture\n• Preserve capital above all\n• Do not fight the tape"
}

html += f"""
<div style="background-color: #ffffcc; padding: 10px; border-left: 4px solid orange; margin: 10px 0;">
<b>📋 Recommended Actions:</b><br>
{actions.get(risk_level, '').replace(chr(10), '<br>')}
</div>
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
<p><b>Component scores (with trends):</b><br>
"""

for indicator, score in scores.items():
    trend = score_changes.get(indicator, "→")
    color = "red" if score >= RED else "orange" if score >= GREEN else "green"
    html += f'<span style="color: {color};">{indicator}: {score:.2f} {trend}</span><br>'

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
if is_near_deadline:
    print(f"Debt ceiling: {days_to_x} days to X-date")
