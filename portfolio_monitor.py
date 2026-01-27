"""
Portfolio Monitor - Individual stock risk tracking
Analyzes each holding for buy/sell/hold signals
"""

import os
import json
import smtplib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

EMAIL_FROM = "tshprung@gmail.com"
EMAIL_TO = "tshprung@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

PORTFOLIO_FILE = "portfolio.csv"  # Your exported Yahoo Finance CSV
STATE_FILE = "portfolio_state.json"

# Risk thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
DRAWDOWN_WARNING = -0.10  # -10% from peak
DRAWDOWN_CRITICAL = -0.20  # -20% from peak
VOLUME_SPIKE = 2.0  # 2x average volume

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    loss_val = float(loss.iloc[-1])
    gain_val = float(gain.iloc[-1])
    
    if loss_val == 0:
        return 100.0
    
    rs = gain_val / loss_val
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)

def analyze_stock(symbol, shares, cost_basis):
    """Analyze individual stock for signals"""
    
    # Fetch data
    stock = yf.download(symbol, period="6mo", progress=False)
    if stock.empty:
        return None
    
    current_price = stock["Close"].iloc[-1]
    volume = stock["Volume"]
    
    # Technical indicators
    ma_50 = float(stock["Close"].rolling(50).mean().iloc[-1])
    ma_200 = float(stock["Close"].rolling(200).mean().iloc[-1]) if len(stock) >= 200 else None
    rsi = calculate_rsi(stock["Close"])
    
    # Peak drawdown
    peak_price = float(stock["Close"].rolling(60).max().iloc[-1])
    drawdown = (current_price / peak_price) - 1
    
    # Volume analysis
    avg_volume = float(volume[-20:].mean())
    current_volume = float(volume.iloc[-1])
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    
    # Support/Resistance
    week_high = float(stock["High"][-5:].max())
    week_low = float(stock["Low"][-5:].min())
    
    # Position info
    market_value = current_price * shares
    cost_value = cost_basis * shares
    unrealized_gain = market_value - cost_value
    unrealized_gain_pct = (unrealized_gain / cost_value) * 100 if cost_value > 0 else 0
    
    # Generate signals
    signals = []
    signal_type = "HOLD"
    risk_score = 0
    
    # SELL signals
    if drawdown < DRAWDOWN_CRITICAL:
        signals.append(f"📉 Down {drawdown*100:.1f}% from peak - CRITICAL")
        signal_type = "SELL"
        risk_score += 3
    elif drawdown < DRAWDOWN_WARNING:
        signals.append(f"⚠️ Down {drawdown*100:.1f}% from peak")
        signal_type = "TRIM"
        risk_score += 2
    
    if current_price < ma_50:
        signals.append(f"Below 50-day MA (${ma_50:.2f})")
        risk_score += 1
        if signal_type == "HOLD":
            signal_type = "WATCH"
    
    if ma_200 and current_price < ma_200:
        signals.append(f"Below 200-day MA (${ma_200:.2f}) - bearish")
        risk_score += 1
    
    if rsi < RSI_OVERSOLD:
        signals.append(f"RSI {rsi:.0f} - OVERSOLD (potential bounce)")
        if signal_type not in ["SELL", "TRIM"]:
            signal_type = "BUY_DIP"
    elif rsi > RSI_OVERBOUGHT:
        signals.append(f"RSI {rsi:.0f} - OVERBOUGHT")
        risk_score += 1
    
    if volume_ratio > VOLUME_SPIKE:
        signals.append(f"Volume spike {volume_ratio:.1f}x average")
        risk_score += 1
    
    # BUY signals (only if not already SELL/TRIM)
    if signal_type == "HOLD":
        if current_price > ma_50 and (ma_200 is None or current_price > ma_200):
            if 40 < rsi < 60:
                signals.append("✅ Uptrend intact, healthy RSI")
                signal_type = "STRONG_HOLD"
    
    # Determine emoji
    if signal_type == "SELL":
        emoji = "🔴"
    elif signal_type == "TRIM":
        emoji = "🟠"
    elif signal_type == "WATCH":
        emoji = "🟡"
    elif signal_type == "BUY_DIP":
        emoji = "🟢"
    elif signal_type == "STRONG_HOLD":
        emoji = "💎"
    else:
        emoji = "⚪"
    
    return {
        "symbol": symbol,
        "current_price": current_price,
        "shares": shares,
        "cost_basis": cost_basis,
        "market_value": market_value,
        "unrealized_gain": unrealized_gain,
        "unrealized_gain_pct": unrealized_gain_pct,
        "ma_50": ma_50,
        "ma_200": ma_200,
        "rsi": rsi,
        "drawdown": drawdown,
        "volume_ratio": volume_ratio,
        "signals": signals,
        "signal_type": signal_type,
        "risk_score": risk_score,
        "emoji": emoji
    }

def calculate_portfolio_beta(symbols):
    """Calculate portfolio beta vs SPY"""
    spy = yf.download("SPY", period="6mo", progress=False)["Close"]
    
    betas = []
    for symbol in symbols:
        stock = yf.download(symbol, period="6mo", progress=False)["Close"]
        if len(stock) > 60 and len(spy) > 60:
            # Align dates
            combined = pd.DataFrame({"spy": spy, "stock": stock}).dropna()
            if len(combined) > 60:
                spy_returns = combined["spy"].pct_change().dropna()
                stock_returns = combined["stock"].pct_change().dropna()
                
                # Beta = Cov(stock, spy) / Var(spy)
                covariance = stock_returns.cov(spy_returns)
                variance = spy_returns.var()
                beta = covariance / variance if variance != 0 else 1.0
                betas.append(beta)
    
    return np.mean(betas) if betas else 1.0

# Read portfolio from CSV
try:
    df = pd.read_csv(PORTFOLIO_FILE)
    portfolio = []
    
    for _, row in df.iterrows():
        portfolio.append({
            "symbol": row["Symbol"],
            "shares": float(row["Shares"]),
            "cost_basis": float(row["Avg Cost/Share"])
        })
except Exception as e:
    print(f"Error reading portfolio CSV: {e}")
    exit(1)

# Analyze each stock
results = []
for holding in portfolio:
    result = analyze_stock(holding["symbol"], holding["shares"], holding["cost_basis"])
    if result:
        results.append(result)

# Sort by risk score (highest first)
results.sort(key=lambda x: x["risk_score"], reverse=True)

# Portfolio aggregates
total_value = sum(r["market_value"] for r in results)
total_gain = sum(r["unrealized_gain"] for r in results)
total_gain_pct = (total_gain / (total_value - total_gain)) * 100 if total_value > 0 else 0

portfolio_beta = calculate_portfolio_beta([r["symbol"] for r in results])

# Count by signal type
sell_count = sum(1 for r in results if r["signal_type"] == "SELL")
trim_count = sum(1 for r in results if r["signal_type"] == "TRIM")
watch_count = sum(1 for r in results if r["signal_type"] == "WATCH")
buy_count = sum(1 for r in results if r["signal_type"] == "BUY_DIP")
hold_count = len(results) - sell_count - trim_count - watch_count - buy_count

# Load previous state
try:
    with open(STATE_FILE) as f:
        prev_state = json.load(f)
except:
    prev_state = {}

# Detect new signals
new_alerts = []
for r in results:
    prev_signal = prev_state.get(r["symbol"], {}).get("signal_type", "HOLD")
    if r["signal_type"] in ["SELL", "TRIM"] and prev_signal not in ["SELL", "TRIM"]:
        new_alerts.append(f"{r['emoji']} {r['symbol']} → {r['signal_type']}")

# Save current state
current_state = {r["symbol"]: {"signal_type": r["signal_type"], "price": r["current_price"]} for r in results}
with open(STATE_FILE, "w") as f:
    json.dump(current_state, f, indent=2)

# Generate email
msg = MIMEMultipart()
msg["From"] = EMAIL_FROM
msg["To"] = EMAIL_TO

if sell_count > 0:
    subject = f"🔴 PORTFOLIO ALERT: {sell_count} SELL signal(s)"
elif trim_count > 0:
    subject = f"🟠 Portfolio: {trim_count} TRIM signal(s)"
elif buy_count > 0:
    subject = f"🟢 Portfolio: {buy_count} buy opportunity"
else:
    subject = f"Portfolio Report: {hold_count} positions healthy"

msg["Subject"] = subject

html = f"""
<h2>Portfolio Health Report</h2>
<p><b>Total Value:</b> ${total_value:,.2f} | <b>Gain:</b> ${total_gain:,.2f} ({total_gain_pct:+.1f}%)</p>
<p><b>Portfolio Beta:</b> {portfolio_beta:.2f} ({abs(portfolio_beta-1)*100:.0f}% {'more' if portfolio_beta > 1 else 'less'} volatile than SPY)</p>
<p><b>Status:</b> {sell_count} SELL, {trim_count} TRIM, {watch_count} WATCH, {hold_count} HOLD, {buy_count} BUY</p>
"""

if new_alerts:
    html += f"""
<div style="background-color: #ffcccc; padding: 10px; border-left: 4px solid red; margin: 10px 0;">
<b>🚨 NEW ALERTS:</b><br>
{"<br>".join(new_alerts)}
</div>
"""

html += "<hr>"

for r in results:
    color = "#ffcccc" if r["signal_type"] == "SELL" else "#ffe5cc" if r["signal_type"] == "TRIM" else "#fff9cc" if r["signal_type"] == "WATCH" else "#ccffcc" if r["signal_type"] == "BUY_DIP" else "#f0f0f0"
    
    html += f"""
<div style="background-color: {color}; padding: 10px; margin: 10px 0; border-left: 4px solid gray;">
<h3 style="margin: 0;">{r['emoji']} {r['symbol']} - {r['signal_type']}</h3>
<p style="margin: 5px 0;">
<b>Price:</b> ${r['current_price']:.2f} | 
<b>Value:</b> ${r['market_value']:,.2f} | 
<b>Gain:</b> ${r['unrealized_gain']:,.2f} ({r['unrealized_gain_pct']:+.1f}%)<br>
<b>MA50:</b> ${r['ma_50']:.2f} | 
<b>RSI:</b> {r['rsi']:.0f} | 
<b>Drawdown:</b> {r['drawdown']*100:.1f}%
</p>
"""
    
    if r['signals']:
        html += "<p style='margin: 5px 0;'><b>Signals:</b><br>" + "<br>".join(f"• {s}" for s in r['signals']) + "</p>"
    
    html += "</div>"

html += f"""
<hr>
<p style="font-size: 0.9em; color: gray;">
<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</i>
</p>
"""

msg.attach(MIMEText(html, "html"))

try:
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls()
        s.login(EMAIL_FROM, EMAIL_PASSWORD)
        s.send_message(msg)
    print(f"Portfolio report sent: {sell_count} SELL, {trim_count} TRIM, {hold_count} HOLD")
except Exception as e:
    print(f"Email failed: {e}")
