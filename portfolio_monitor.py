"""
Portfolio Monitor - Enhanced with recovery potential and smarter signals
"""
import os
import json
import smtplib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
EXTENDED_GAIN_THRESHOLD = 0.20  # 20%+ gains = consider trimming

# Dividend-paying defensive stocks (add more as needed)
DEFENSIVE_TICKERS = ["KMB", "PG", "JNJ", "KO", "PEP", "WMT", "COST"]

def calculate_rsi(prices, period=14):
    if len(prices) < period: return 50.0
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    avg_gain = gain.iloc[-1]
    avg_loss = loss.iloc[-1]
    
    if pd.isna(avg_loss) or avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))

def calculate_days_below_ma(prices, ma_period=50):
    """Count consecutive days price has been below moving average"""
    if len(prices) < ma_period: return 0
    ma = prices.rolling(ma_period).mean()
    below_ma = prices < ma
    
    count = 0
    # Access values and handle potential Series/Array conversion
    vals = below_ma.values.flatten()
    for val in reversed(vals):
        if val:
            count += 1
        else:
            break
    return count

def calculate_volatility(prices, period=20):
    """Calculate recent volatility (annualized)"""
    if len(prices) < period: return 0.0
    returns = prices.pct_change().dropna()
    vol = returns.tail(period).std() * np.sqrt(252)
    return float(vol) if not pd.isna(vol) else 0.0

def analyze_stock(symbol, shares, cost_basis):
    # Fetch data - ensure we get a clean DataFrame
    stock_raw = yf.download(symbol, period="1y", progress=False)
    if stock_raw.empty: return None
    
    # Standardize column access (handles MultiIndex in newer yfinance versions)
    if isinstance(stock_raw.columns, pd.MultiIndex):
        stock = pd.DataFrame({
            "Close": stock_raw["Close"][symbol],
            "Volume": stock_raw["Volume"][symbol]
        })
    else:
        stock = stock_raw[["Close", "Volume"]]

    current_price = float(stock["Close"].iloc[-1])
    volume = stock["Volume"]
    
    # Metrics
    close_series = stock["Close"]
    ma_50 = float(close_series.rolling(50).mean().iloc[-1])
    ma_200 = float(close_series.rolling(200).mean().iloc[-1]) if len(close_series) >= 200 else None
    rsi = calculate_rsi(close_series)
    
    peak_price = float(close_series.rolling(60).max().iloc[-1])
    drawdown = (current_price / peak_price) - 1
    
    avg_vol = float(volume.tail(20).mean())
    curr_vol = float(volume.iloc[-1])
    vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
    
    days_below_ma50 = calculate_days_below_ma(close_series, 50)
    volatility = calculate_volatility(close_series)
    
    market_value = current_price * shares
    unrealized_gain = market_value - (cost_basis * shares)
    gain_pct = (unrealized_gain / (cost_basis * shares)) * 100 if cost_basis > 0 else 0
    
    is_defensive = symbol in DEFENSIVE_TICKERS
    
    signals = []
    signal_type = "HOLD"
    risk_score = 0
    stop_loss_price = None
    action_note = ""
    
    # --- SIGNAL LOGIC ---
    if drawdown < DRAWDOWN_CRITICAL:
        signals.append(f"📉 Down {drawdown*100:.1f}% from peak - CRITICAL")
        risk_score += 3
        if gain_pct > 0:
            signal_type = "TRIM_EXTENDED"
            action_note = f"Consider trimming - up {gain_pct:.1f}% but extended"
            stop_loss_price = current_price * 0.90
        else:
            signal_type = "SELL"
            action_note = "Cut losses before further deterioration"
            stop_loss_price = current_price * 0.88
            
    elif drawdown < DRAWDOWN_WARNING:
        signals.append(f"⚠️ Down {drawdown*100:.1f}% from peak")
        risk_score += 2
        if is_defensive and gain_pct > 0:
            signal_type = "HOLD"
            action_note = "Defensive position - drawdown is normal volatility"
        elif gain_pct > EXTENDED_GAIN_THRESHOLD * 100:
            signal_type = "TRIM"
            action_note = f"Take profits - up {gain_pct:.1f}% but showing weakness"
            stop_loss_price = cost_basis * 1.10
        else:
            signal_type = "WATCH"
            action_note = "Monitor closely for breakdown"
            stop_loss_price = current_price * 0.90
    
    if current_price < ma_50:
        signals.append(f"Below 50-day MA (${ma_50:.2f})")
        risk_score += 1
        if days_below_ma50 > 90:
            signals.append(f"Broken for {days_below_ma50} days - dead money")
            if signal_type == "HOLD": signal_type = "WATCH"
    
    if ma_200 and current_price < ma_200:
        signals.append(f"Below 200-day MA (${ma_200:.2f}) - bear market")
        risk_score += 1
    
    if rsi < RSI_OVERSOLD:
        signals.append(f"RSI {rsi:.0f} - OVERSOLD (potential bounce)")
        if signal_type not in ["SELL", "TRIM", "TRIM_EXTENDED"]:
            signal_type = "BUY_DIP"
            action_note = "Oversold - could bounce, but confirm trend first"
    elif rsi > RSI_OVERBOUGHT:
        signals.append(f"RSI {rsi:.0f} - OVERBOUGHT")
        risk_score += 1
        if gain_pct > 15 and signal_type == "HOLD":
            signal_type = "TRIM"
            action_note = f"Take profits - up {gain_pct:.1f}% and extended"
    
    if vol_ratio > VOLUME_SPIKE:
        signals.append(f"Volume spike {vol_ratio:.1f}x average")
        risk_score += 1
    
    recovery_potential = 0.0
    if signal_type in ["SELL", "WATCH", "BUY_DIP"]:
        if volatility > 0.4: recovery_potential += 0.3
        if days_below_ma50 < 30: recovery_potential += 0.4
        elif days_below_ma50 < 60: recovery_potential += 0.2
        if rsi < RSI_OVERSOLD: recovery_potential += 0.3
        recovery_potential = min(recovery_potential, 1.0)
    
    if signal_type == "HOLD" and gain_pct > 5:
        if current_price > ma_50 and (ma_200 is None or current_price > ma_200):
            if 40 < rsi < 60:
                signals.append("✅ Uptrend intact, healthy")
                action_note = "Strong position - hold"
    
    emoji_map = {"SELL": "🔴", "TRIM": "🟠", "TRIM_EXTENDED": "🟠", "WATCH": "🟡", "BUY_DIP": "🟢", "STRONG_HOLD": "💎"}
    emoji = emoji_map.get(signal_type, "⚪")
    
    return {
        "symbol": symbol, "current_price": current_price, "shares": shares, "cost_basis": cost_basis,
        "market_value": market_value, "unrealized_gain": unrealized_gain, "unrealized_gain_pct": gain_pct,
        "ma_50": ma_50, "ma_200": ma_200, "rsi": rsi, "drawdown": drawdown, "volume_ratio": vol_ratio,
        "days_below_ma50": days_below_ma50, "volatility": volatility, "recovery_potential": recovery_potential,
        "signals": signals, "signal_type": signal_type, "risk_score": risk_score, "emoji": emoji,
        "stop_loss_price": stop_loss_price, "action_note": action_note, "is_defensive": is_defensive
    }

def calculate_portfolio_beta(symbols):
    try:
        spy_data = yf.download("SPY", period="6mo", progress=False)
        spy = spy_data["Close"] if not isinstance(spy_data.columns, pd.MultiIndex) else spy_data["Close"]["SPY"]
        betas = []
        for s in symbols:
            s_data = yf.download(s, period="6mo", progress=False)
            if s_data.empty: continue
            stock = s_data["Close"] if not isinstance(s_data.columns, pd.MultiIndex) else s_data["Close"][s]
            combined = pd.DataFrame({"spy": spy, "stock": stock}).dropna()
            if len(combined) > 30:
                returns = combined.pct_change().dropna()
                cov_val = returns['stock'].cov(returns['spy'])
                var_val = returns['spy'].var()
                if var_val != 0: betas.append(cov_val / var_val)
        return float(np.mean(betas)) if betas else 1.0
    except:
        return 1.0

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        df = pd.read_csv(PORTFOLIO_FILE)
        results = []
        for _, row in df.iterrows():
            res = analyze_stock(row["Symbol"], float(row["Shares"]), float(row["Avg Cost/Share"]))
            if res: results.append(res)
        
        results.sort(key=lambda x: x["risk_score"], reverse=True)
        total_val = sum(r["market_value"] for r in results)
        total_gain = sum(r["unrealized_gain"] for r in results)
        total_gain_pct = (total_gain / (total_val - total_gain)) * 100 if total_val > total_gain else 0
        p_beta = calculate_portfolio_beta([r["symbol"] for r in results])
        
        # Counts
        sell_count = sum(1 for r in results if r["signal_type"] in ["SELL"])
        trim_count = sum(1 for r in results if r["signal_type"] in ["TRIM", "TRIM_EXTENDED"])
        watch_count = sum(1 for r in results if r["signal_type"] == "WATCH")
        buy_count = sum(1 for r in results if r["signal_type"] == "BUY_DIP")
        hold_count = len(results) - sell_count - trim_count - watch_count - buy_count
        
        portfolio_warnings = []
        if sell_count >= 3: portfolio_warnings.append(f"⚠️ {sell_count} positions need selling")
        if p_beta > 1.3 and total_gain_pct < 0: portfolio_warnings.append(f"⚠️ High risk (Beta {p_beta:.2f})")
        if total_gain_pct < -10: portfolio_warnings.append(f"⚠️ Portfolio down {total_gain_pct:.1f}%")
        
        try:
            with open(STATE_FILE) as f: prev_state = json.load(f)
        except: prev_state = {}
        
        new_alerts = []
        for r in results:
            prev_signal = prev_state.get(r["symbol"], {}).get("signal_type", "HOLD")
            if r["signal_type"] in ["SELL", "TRIM", "TRIM_EXTENDED"] and prev_signal not in ["SELL", "TRIM", "TRIM_EXTENDED"]:
                new_alerts.append(f"{r['emoji']} {r['symbol']} → {r['signal_type']}")
        
        current_state = {r["symbol"]: {"signal_type": r["signal_type"], "price": r["current_price"], "gain_pct": r["unrealized_gain_pct"]} for r in results}
        with open(STATE_FILE, "w") as f: json.dump(current_state, f, indent=2)
        
        # Email Subject
        if sell_count > 0: subject = f"🔴 PORTFOLIO: {sell_count} SELL signal(s)"
        elif trim_count > 0: subject = f"🟠 Portfolio: {trim_count} TRIM signal(s)"
        else: subject = f"Portfolio Report: {datetime.now().strftime('%Y-%m-%d')}"
        
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = EMAIL_FROM, EMAIL_TO, subject
        
        html = f"""
        <h2>Portfolio Health Report</h2>
        <p><b>Total Value:</b> ${total_val:,.2f} | <b>Gain:</b> ${total_gain:,.2f} (<span style="color: {'green' if total_gain_pct > 0 else 'red'};">{total_gain_pct:+.1f}%</span>)</p>
        <p><b>Beta:</b> {p_beta:.2f} | {sell_count} SELL | {trim_count} TRIM | {hold_count} HOLD</p>
        """
        if portfolio_warnings: html += f"<div style='background:#fff3cd;padding:10px;'><b>WARNINGS:</b><br>{'<br>'.join(portfolio_warnings)}</div>"
        if new_alerts: html += f"<div style='background:#ffcccc;padding:10px;'><b>ALERTS:</b><br>{'<br>'.join(new_alerts)}</div><hr>"
        
        for r in results:
            color = {"SELL": "#ffcccc", "TRIM": "#ffe5cc", "TRIM_EXTENDED": "#ffe5cc", "WATCH": "#fff9cc", "BUY_DIP": "#ccffcc"}.get(r["signal_type"], "#f0f0f0")
            html += f"""
            <div style="background-color: {color}; padding: 10px; margin: 5px 0; border-left: 4px solid gray;">
            <h3 style="margin: 0;">{r['emoji']} {r['symbol']} - {r['signal_type']}</h3>
            Price: ${r['current_price']:.2f} | Gain: {r['unrealized_gain_pct']:+.1f}% | RSI: {r['rsi']:.0f}<br>
            {f"<b>💡 {r['action_note']}</b><br>" if r['action_note'] else ""}
            {f"🛑 Stop Loss: ${r['stop_loss_price']:.2f}<br>" if r['stop_loss_price'] else ""}
            </div>"""
            
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
            s.starttls()
            s.login(EMAIL_FROM, EMAIL_PASSWORD)
            s.send_message(msg)
        print("✅ Report sent.")

    except Exception as e:
        print(f"❌ Error: {e}")
