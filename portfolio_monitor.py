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
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") # MUST be a Google App Password

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
    # Fixed yfinance call for single-level columns
    stock = yf.download(symbol, period="6mo", progress=False, multi_level_index=False)
    if stock.empty: return None
    
    # Ensure we get single float values
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
    
    # Position math
    market_value = current_price * shares
    unrealized_gain = market_value - (cost_basis * shares)
    gain_pct = (unrealized_gain / (cost_basis * shares)) * 100 if cost_basis > 0 else 0
    
    # Signal Logic
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
    
    if rsi < RSI_OVERSOLD:
        signals.append(f"RSI {rsi:.0f} - OVERSOLD")
        if signal_type not in ["SELL", "TRIM"]: signal_type = "BUY_DIP"
    elif rsi > RSI_OVERBOUGHT:
        signals.append(f"RSI {rsi:.0f} - OVERBOUGHT")
        risk_score += 1

    emoji = {"SELL": "🔴", "TRIM": "🟠", "WATCH": "🟡", "BUY_DIP": "🟢", "STRONG_HOLD": "💎"}.get(signal_type, "⚪")
    
    return {
        "symbol": symbol, "current_price": current_price, "market_value": market_value,
        "unrealized_gain": unrealized_gain, "unrealized_gain_pct": gain_pct,
        "ma_50": ma_50, "rsi": rsi, "drawdown": drawdown, "signals": signals,
        "signal_type": signal_type, "risk_score": risk_score, "emoji": emoji
    }

def calculate_portfolio_beta(symbols):
    spy = yf.download("SPY", period="6mo", progress=False, multi_level_index=False)["Close"]
    betas = []
    for s in symbols:
        stock = yf.download(s, period="6mo", progress=False, multi_level_index=False)["Close"]
        combined = pd.DataFrame({"spy": spy, "stock": stock}).dropna()
        if len(combined) > 30:
            returns = combined.pct_change().dropna()
            beta = returns['stock'].cov(returns['spy']) / returns['spy'].var()
            betas.append(beta)
    return np.mean(betas) if betas else 1.0

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
        p_beta = calculate_portfolio_beta([r["symbol"] for r in results])
        
        # Email construction (simplified)
        msg = MIMEMultipart()
        msg["From"], msg["To"] = EMAIL_FROM, EMAIL_TO
        msg["Subject"] = f"Portfolio Report: {datetime.now().strftime('%Y-%m-%d')}"
        
        # (HTML generation remains similar to your original, just ensure it uses these fixed keys)
        # ... [Your HTML generation code] ...
        
        # Note: In a production environment, use 'with' for SMTP
        print(f"Analysis complete. Total Portfolio Value: ${total_val:,.2f}")
        
    except FileNotFoundError:
        print(f"Error: {PORTFOLIO_FILE} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
