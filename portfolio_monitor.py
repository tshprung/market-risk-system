"""
Portfolio Monitor - Individual stock risk tracking
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

# --- CONFIGURATION ---
EMAIL_FROM = "tshprung@gmail.com"
EMAIL_TO = "tshprung@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
PORTFOLIO_FILE = "portfolio.csv"
STATE_FILE = "portfolio_state.json"

# Risk thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
DRAWDOWN_WARNING = -0.10
DRAWDOWN_CRITICAL = -0.20
VOLUME_SPIKE = 2.0

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    avg_gain = gain.iloc[-1]
    avg_loss = loss.iloc[-1]
    
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))

def analyze_stock(symbol, shares, cost_basis):
    stock = yf.download(symbol, period="6mo", progress=False)
    if stock.empty: return None
    
    current_price = float(stock["Close"].iloc[-1])
    volume = stock["Volume"]
    
    ma_50 = float(stock["Close"].rolling(50).mean().iloc[-1])
    ma_200 = float(stock["Close"].rolling(200).mean().iloc[-1]) if len(stock) >= 200 else None
    rsi = calculate_rsi(stock["Close"])
    
    peak_price = float(stock["Close"].rolling(60).max().iloc[-1])
    drawdown = (current_price / peak_price) - 1
    
    avg_vol = float(volume[-20:].mean())
    curr_vol = float(volume.iloc[-1])
    vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
    
    market_value = current_price * shares
    unrealized_gain = market_value - (cost_basis * shares)
    gain_pct = (unrealized_gain / (cost_basis * shares)) * 100 if cost_basis > 0 else 0
    
    signals = []
    signal_type = "HOLD"
    risk_score = 0
    
    if drawdown < DRAWDOWN_CRITICAL:
        signals.append(f"📉 Down {drawdown*100:.1f}% from peak - CRITICAL")
        signal_type, risk_score = "SELL", risk_score + 3
    elif drawdown < DRAWDOWN_WARNING:
        signals.append(f"⚠️ Down {drawdown*100:.1f}% from peak")
        signal_type, risk_score = "TRIM", risk_score + 2
    
    if current_price < ma_50:
        signals.append(f"Below 50-day MA (${ma_50:.2f})")
        risk_score += 1
        if signal_type == "HOLD": signal_type = "WATCH"
    
    if ma_200 and current_price < ma_200:
        signals.append(f"Below 200-day MA (${ma_200:.2f})")
        risk_score += 1
    
    if rsi < RSI_OVERSOLD:
        signals.append(f"RSI {rsi:.0f} - OVERSOLD (potential bounce)")
        if signal_type not in ["SELL", "TRIM"]: signal_type = "BUY_DIP"
    elif rsi > RSI_OVERBOUGHT:
        signals.append(f"RSI {rsi:.0f} - OVERBOUGHT")
        risk_score += 1
    
    if vol_ratio > VOLUME_SPIKE:
        signals.append(f"Volume spike {vol_ratio:.1f}x average")
        risk_score += 1

    emoji = {"SELL": "🔴", "TRIM": "🟠", "WATCH": "🟡", "BUY_DIP": "🟢", "STRONG_HOLD": "💎"}.get(signal_type, "⚪")
    
    return {
        "symbol": symbol, "current_price": current_price, "shares": shares,
        "cost_basis": cost_basis, "market_value": market_value,
        "unrealized_gain": unrealized_gain, "unrealized_gain_pct": gain_pct,
        "ma_50": ma_50, "ma_200": ma_200, "rsi": rsi, "drawdown": drawdown,
        "volume_ratio": vol_ratio, "signals": signals,
        "signal_type": signal_type, "risk_score": risk_score, "emoji": emoji
    }

def calculate_portfolio_beta(symbols):
    spy = yf.download("SPY", period="6mo", progress=False)["Close"]
    betas = []
    for s in symbols:
        stock = yf.download(s, period="6mo", progress=False)["Close"]
        combined = pd.DataFrame({"spy": spy, "stock": stock}).dropna()
        if len(combined) > 30:
            returns = combined.pct_change().dropna()
            cov_val = returns['stock'].cov(returns['spy'])
            var_val = returns['spy'].var()
            if var_val != 0:
                betas.append(cov_val / var_val)
    return float(np.mean(betas)) if betas else 1.0

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        df = pd.read_csv(PORTFOLIO_FILE)
        results = []
        for _, row in df.iterrows():
            res = analyze_stock(row["Symbol"], float(row["Shares"]), float(row["Avg Cost/Share"]))
            if res: results.append(res)
        
        results.sort(key=lambda x: x["risk_score"], reverse=True)
        
        # Aggregate stats
        total_val = sum(r["market_value"] for r in results)
        total_gain = sum(r["unrealized_gain"] for r in results)
        total_gain_pct = (total_gain / (total_val - total_gain)) * 100 if total_val > 0 else 0
        p_beta = calculate_portfolio_beta([r["symbol"] for r in results])
        
        # Count signals
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
        
        # Detect new alerts
        new_alerts = []
        for r in results:
            prev_signal = prev_state.get(r["symbol"], {}).get("signal_type", "HOLD")
            if r["signal_type"] in ["SELL", "TRIM"] and prev_signal not in ["SELL", "TRIM"]:
                new_alerts.append(f"{r['emoji']} {r['symbol']} → {r['signal_type']}")
        
        # Save current state
        current_state = {r["symbol"]: {"signal_type": r["signal_type"], "price": r["current_price"]} for r in results}
        with open(STATE_FILE, "w") as f:
            json.dump(current_state, f, indent=2)
        
        # Email subject
        if sell_count > 0:
            subject = f"🔴 PORTFOLIO ALERT: {sell_count} SELL signal(s)"
        elif trim_count > 0:
            subject = f"🟠 Portfolio: {trim_count} TRIM signal(s)"
        elif buy_count > 0:
            subject = f"🟢 Portfolio: {buy_count} buy opportunity"
        else:
            subject = f"Portfolio Report: {hold_count} positions healthy"
        
        # Build email
        msg = MIMEMultipart()
        msg["From"], msg["To"] = EMAIL_FROM, EMAIL_TO
        msg["Subject"] = subject
        
        html = f"""
<h2>Portfolio Health Report</h2>
<p><b>Total Value:</b> ${total_val:,.2f} | <b>Gain:</b> ${total_gain:,.2f} ({total_gain_pct:+.1f}%)</p>
<p><b>Portfolio Beta:</b> {p_beta:.2f} ({abs(p_beta-1)*100:.0f}% {'more' if p_beta > 1 else 'less'} volatile than SPY)</p>
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
<b>Shares:</b> {r['shares']:.0f} | 
<b>Value:</b> ${r['market_value']:,.2f}<br>
<b>Cost Basis:</b> ${r['cost_basis']:.2f} | 
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
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
            s.starttls()
            s.login(EMAIL_FROM, EMAIL_PASSWORD)
            s.send_message(msg)
        
        print(f"Portfolio report sent: {sell_count} SELL, {trim_count} TRIM, {hold_count} HOLD, {buy_count} BUY")
        print(f"Total Value: ${total_val:,.2f} | Gain: ${total_gain:,.2f} ({total_gain_pct:+.1f}%)")
        
    except FileNotFoundError:
        print(f"Error: {PORTFOLIO_FILE} not found.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
