import os, json, smtplib
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
PORTFOLIO_FILE = "portfolio.csv"
STATE_FILE = "portfolio_state.json"

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    avg_gain, avg_loss = gain.iloc[-1], loss.iloc[-1]
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))

def analyze_stock(symbol, shares, cost_basis):
    # FIX: multi_level_index=False prevents KeyError: 'Close'
    stock = yf.download(symbol, period="7mo", progress=False, multi_level_index=False)
    if stock.empty: return None
    
    close = stock["Close"]
    current_price = float(close.iloc[-1])
    ma_50 = float(close.rolling(50).mean().iloc[-1])
    ma_200 = float(close.rolling(200).mean().iloc[-1]) if len(stock) >= 200 else None
    rsi = calculate_rsi(close)
    
    peak_price = float(close.rolling(60).max().iloc[-1])
    drawdown = (current_price / peak_price) - 1
    vol_ratio = float(stock["Volume"].iloc[-1] / stock["Volume"][-20:].mean())
    
    market_value = current_price * shares
    unrealized_gain = market_value - (cost_basis * shares)
    gain_pct = (unrealized_gain / (cost_basis * shares)) * 100 if cost_basis > 0 else 0
    
    signals, signal_type, risk_score = [], "HOLD", 0
    
    if drawdown < -0.20:
        signals.append(f"📉 Down {drawdown*100:.1f}% - CRITICAL")
        signal_type, risk_score = "SELL", 3
    elif drawdown < -0.10:
        signals.append(f"⚠️ Down {drawdown*100:.1f}% - Warning")
        signal_type, risk_score = "TRIM", 2
    
    if current_price < ma_50:
        signals.append(f"Below 50-day MA")
        risk_score += 1
        if signal_type == "HOLD": signal_type = "WATCH"
        
    if rsi < 30:
        signals.append(f"RSI {rsi:.0f} - OVERSOLD")
        if signal_type not in ["SELL", "TRIM"]: signal_type = "BUY_DIP"
    elif rsi > 70:
        signals.append(f"RSI {rsi:.0f} - OVERBOUGHT")
        risk_score += 1

    emoji = {"SELL": "🔴", "TRIM": "🟠", "WATCH": "🟡", "BUY_DIP": "🟢", "STRONG_HOLD": "💎"}.get(signal_type, "⚪")
    
    return {
        "symbol": symbol, "current_price": current_price, "shares": shares, "cost_basis": cost_basis,
        "market_value": market_value, "unrealized_gain": unrealized_gain, "unrealized_gain_pct": gain_pct,
        "ma_50": ma_50, "ma_200": ma_200, "rsi": rsi, "drawdown": drawdown, "signals": signals,
        "signal_type": signal_type, "risk_score": risk_score, "emoji": emoji, "df": stock
    }

def calculate_portfolio_beta(results):
    spy = yf.download("SPY", period="6mo", progress=False, multi_level_index=False)["Close"].pct_change().dropna()
    betas = []
    for r in results:
        stock_ret = r['df']["Close"].pct_change().dropna()
        combined = pd.DataFrame({"spy": spy, "stock": stock_ret}).dropna()
        if not combined.empty:
            betas.append(combined.cov().iloc[0,1] / combined['spy'].var())
    return float(np.mean(betas)) if betas else 1.0

if __name__ == "__main__":
    try:
        df = pd.read_csv(PORTFOLIO_FILE)
        results = [analyze_stock(row["Symbol"], float(row["Shares"]), float(row["Avg Cost/Share"])) for _, row in df.iterrows()]
        results = [r for r in results if r]
        results.sort(key=lambda x: x["risk_score"], reverse=True)

        # Totals & Beta
        total_val = sum(r["market_value"] for r in results)
        total_cost = sum(r["shares"] * r["cost_basis"] for r in results)
        total_gain_pct = ((total_val / total_cost) - 1) * 100 if total_cost > 0 else 0
        p_beta = calculate_portfolio_beta(results)

        # State Handling (New Alerts)
        try:
            with open(STATE_FILE) as f: prev_state = json.load(f)
        except: prev_state = {}
        
        new_alerts = [f"{r['emoji']} {r['symbol']} → {r['signal_type']}" for r in results 
                      if r["signal_type"] in ["SELL", "TRIM"] and prev_state.get(r["symbol"], {}).get("signal_type") not in ["SELL", "TRIM"]]

        with open(STATE_FILE, "w") as f:
            json.dump({r["symbol"]: {"signal_type": r["signal_type"]} for r in results}, f)

        # Email Generation
        sell_c = sum(1 for r in results if r["signal_type"] == "SELL")
        msg = MIMEMultipart()
        msg["Subject"] = f"{'🔴 SELL ALERT' if sell_c > 0 else '📊 Portfolio Report'}"
        msg["From"], msg["To"] = EMAIL_FROM, EMAIL_TO

        html = f"<h2>Portfolio Health</h2><p>Value: ${total_val:,.2f} ({total_gain_pct:+.1f}%) | Beta: {p_beta:.2f}</p>"
        if new_alerts: html += f"<div style='color:red;'><b>ALERTS:</b><br>{'<br>'.join(new_alerts)}</div>"
        
        for r in results:
            color = "#ffcccc" if r["signal_type"] == "SELL" else "#f0f0f0"
            html += f"<div style='background:{color}; padding:10px; margin:5px; border-left:4px solid gray;'>"
            html += f"<b>{r['emoji']} {r['symbol']}</b>: ${r['current_price']:.2f} | Gain: {r['unrealized_gain_pct']:+.1f}%<br>"
            html += f"Signals: {', '.join(r['signals']) if r['signals'] else 'Healthy'}</div>"

        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(EMAIL_FROM, EMAIL_PASSWORD)
            s.send_message(msg)
        print("Report sent successfully.")

    except Exception as e:
        print(f"Error: {e}")
