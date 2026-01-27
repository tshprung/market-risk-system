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
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    avg_gain = gain.iloc[-1]
    avg_loss = loss.iloc[-1]
    
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))

def calculate_days_below_ma(prices, ma_period=50):
    """Count consecutive days price has been below moving average"""
    ma = prices.rolling(ma_period).mean()
    below_ma = prices < ma
    
    # Count consecutive True values from the end
    count = 0
    for val in reversed(below_ma.values):
        if val:
            count += 1
        else:
            break
    return count

def calculate_volatility(prices, period=20):
    """Calculate recent volatility (annualized)"""
    returns = prices.pct_change().dropna()
    vol = returns.tail(period).std() * np.sqrt(252)
    return float(vol)

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
    
    # NEW: Recovery potential metrics
    days_below_ma50 = calculate_days_below_ma(stock["Close"], 50)
    volatility = calculate_volatility(stock["Close"])
    
    market_value = current_price * shares
    unrealized_gain = market_value - (cost_basis * shares)
    gain_pct = (unrealized_gain / (cost_basis * shares)) * 100 if cost_basis > 0 else 0
    
    is_defensive = symbol in DEFENSIVE_TICKERS
    
    signals = []
    signal_type = "HOLD"
    risk_score = 0
    stop_loss_price = None
    action_note = ""
    
    # --- SIGNAL LOGIC (Enhanced) ---
    
    # Critical: Down >20% from peak
    if drawdown < DRAWDOWN_CRITICAL:
        signals.append(f"📉 Down {drawdown*100:.1f}% from peak - CRITICAL")
        risk_score += 3
        
        # Differentiate: Are you still profitable overall?
        if gain_pct > 0:
            signal_type = "TRIM_EXTENDED"  # Up overall but crashed from peak
            action_note = f"Consider trimming - up {gain_pct:.1f}% but extended"
            stop_loss_price = current_price * 0.90  # 10% below current
        else:
            signal_type = "SELL"
            action_note = "Cut losses before further deterioration"
            stop_loss_price = current_price * 0.88  # 12% below current
            
    # Warning: Down 10-20% from peak
    elif drawdown < DRAWDOWN_WARNING:
        signals.append(f"⚠️ Down {drawdown*100:.1f}% from peak")
        risk_score += 2
        
        # EXCEPTION: Defensive dividend stocks with positive gains
        if is_defensive and gain_pct > 0:
            signal_type = "HOLD"
            action_note = "Defensive position - drawdown is normal volatility"
        elif gain_pct > EXTENDED_GAIN_THRESHOLD * 100:
            signal_type = "TRIM"
            action_note = f"Take profits - up {gain_pct:.1f}% but showing weakness"
            stop_loss_price = cost_basis * 1.10  # Lock in 10% gain
        else:
            signal_type = "WATCH"
            action_note = "Monitor closely for breakdown"
            stop_loss_price = current_price * 0.90
    
    # Below moving averages
    if current_price < ma_50:
        signals.append(f"Below 50-day MA (${ma_50:.2f})")
        risk_score += 1
        
        # Check if it's been broken for a long time
        if days_below_ma50 > 90:
            signals.append(f"Broken for {days_below_ma50} days - dead money")
            if signal_type == "HOLD":
                signal_type = "WATCH"
        elif days_below_ma50 > 30:
            signals.append(f"Below MA for {days_below_ma50} days")
    
    if ma_200 and current_price < ma_200:
        signals.append(f"Below 200-day MA (${ma_200:.2f}) - bear market")
        risk_score += 1
    
    # RSI extremes
    if rsi < RSI_OVERSOLD:
        signals.append(f"RSI {rsi:.0f} - OVERSOLD (potential bounce)")
        # Only upgrade to BUY_DIP if not already in worse category
        if signal_type not in ["SELL", "TRIM", "TRIM_EXTENDED"]:
            signal_type = "BUY_DIP"
            action_note = "Oversold - could bounce, but confirm trend first"
    elif rsi > RSI_OVERBOUGHT:
        signals.append(f"RSI {rsi:.0f} - OVERBOUGHT")
        risk_score += 1
        if gain_pct > 15 and signal_type == "HOLD":
            signal_type = "TRIM"
            action_note = f"Take profits - up {gain_pct:.1f}% and extended"
    
    # Volume analysis
    if vol_ratio > VOLUME_SPIKE:
        signals.append(f"Volume spike {vol_ratio:.1f}x average")
        risk_score += 1
    
    # Recovery potential score (0-1, higher = better recovery chance)
    recovery_potential = 0.0
    if signal_type in ["SELL", "WATCH", "BUY_DIP"]:
        # High volatility = more likely to bounce
        if volatility > 0.4:
            recovery_potential += 0.3
        # Recently broken = more likely to recover than long-term broken
        if days_below_ma50 < 30:
            recovery_potential += 0.4
        elif days_below_ma50 < 60:
            recovery_potential += 0.2
        # Oversold = potential bounce
        if rsi < RSI_OVERSOLD:
            recovery_potential += 0.3
        
        recovery_potential = min(recovery_potential, 1.0)
    
    # Healthy positions
    if signal_type == "HOLD" and gain_pct > 5:
        if current_price > ma_50 and (ma_200 is None or current_price > ma_200):
            if 40 < rsi < 60:
                signals.append("✅ Uptrend intact, healthy")
                action_note = "Strong position - hold"
    
    # Determine emoji
    emoji_map = {
        "SELL": "🔴",
        "TRIM": "🟠", 
        "TRIM_EXTENDED": "🟠",
        "WATCH": "🟡",
        "BUY_DIP": "🟢",
        "STRONG_HOLD": "💎"
    }
    emoji = emoji_map.get(signal_type, "⚪")
    
    return {
        "symbol": symbol,
        "current_price": current_price,
        "shares": shares,
        "cost_basis": cost_basis,
        "market_value": market_value,
        "unrealized_gain": unrealized_gain,
        "unrealized_gain_pct": gain_pct,
        "ma_50": ma_50,
        "ma_200": ma_200,
        "rsi": rsi,
        "drawdown": drawdown,
        "volume_ratio": vol_ratio,
        "days_below_ma50": days_below_ma50,
        "volatility": volatility,
        "recovery_potential": recovery_potential,
        "signals": signals,
        "signal_type": signal_type,
        "risk_score": risk_score,
        "emoji": emoji,
        "stop_loss_price": stop_loss_price,
        "action_note": action_note,
        "is_defensive": is_defensive
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
        sell_count = sum(1 for r in results if r["signal_type"] in ["SELL"])
        trim_count = sum(1 for r in results if r["signal_type"] in ["TRIM", "TRIM_EXTENDED"])
        watch_count = sum(1 for r in results if r["signal_type"] == "WATCH")
        buy_count = sum(1 for r in results if r["signal_type"] == "BUY_DIP")
        hold_count = len(results) - sell_count - trim_count - watch_count - buy_count
        
        # Portfolio-level warnings
        portfolio_warnings = []
        if sell_count >= 3:
            portfolio_warnings.append(f"⚠️ {sell_count} positions need selling - portfolio is struggling")
        if p_beta > 1.3 and total_gain_pct < 0:
            portfolio_warnings.append(f"⚠️ High risk (Beta {p_beta:.2f}) with negative returns - reduce exposure")
        if total_gain_pct < -10:
            portfolio_warnings.append(f"⚠️ Portfolio down {total_gain_pct:.1f}% - consider defensive positions")
        
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
            if r["signal_type"] in ["SELL", "TRIM", "TRIM_EXTENDED"] and prev_signal not in ["SELL", "TRIM", "TRIM_EXTENDED"]:
                new_alerts.append(f"{r['emoji']} {r['symbol']} → {r['signal_type']}")
        
        # Save current state
        current_state = {
            r["symbol"]: {
                "signal_type": r["signal_type"],
                "price": r["current_price"],
                "gain_pct": r["unrealized_gain_pct"]
            } for r in results
        }
        with open(STATE_FILE, "w") as f:
            json.dump(current_state, f, indent=2)
        
        # Email subject
        if sell_count > 0:
            subject = f"🔴 PORTFOLIO: {sell_count} SELL signal(s)"
        elif trim_count > 0:
            subject = f"🟠 Portfolio: {trim_count} TRIM signal(s)"
        elif buy_count > 0:
            subject = f"🟢 Portfolio: {buy_count} buy opportunity"
        else:
            subject = f"Portfolio: {hold_count} healthy positions"
        
        # Build email
        msg = MIMEMultipart()
        msg["From"], msg["To"] = EMAIL_FROM, EMAIL_TO
        msg["Subject"] = subject
        
        html = f"""
<h2>Portfolio Health Report</h2>
<p><b>Total Value:</b> ${total_val:,.2f} | <b>Gain:</b> ${total_gain:,.2f} (<span style="color: {'green' if total_gain_pct > 0 else 'red'};">{total_gain_pct:+.1f}%</span>)</p>
<p><b>Portfolio Beta:</b> {p_beta:.2f} ({abs(p_beta-1)*100:.0f}% {'more' if p_beta > 1 else 'less'} volatile than market)</p>
<p><b>Positions:</b> {sell_count} SELL | {trim_count} TRIM | {watch_count} WATCH | {hold_count} HOLD | {buy_count} BUY</p>
"""
        
        if portfolio_warnings:
            html += f"""
<div style="background-color: #fff3cd; padding: 10px; border-left: 4px solid orange; margin: 10px 0;">
<b>📊 PORTFOLIO-LEVEL WARNINGS:</b><br>
{"<br>".join(portfolio_warnings)}
</div>
"""
        
        if new_alerts:
            html += f"""
<div style="background-color: #ffcccc; padding: 10px; border-left: 4px solid red; margin: 10px 0;">
<b>🚨 NEW ALERTS TODAY:</b><br>
{"<br>".join(new_alerts)}
</div>
"""
        
        html += "<hr>"
        
        for r in results:
            color_map = {
                "SELL": "#ffcccc",
                "TRIM": "#ffe5cc",
                "TRIM_EXTENDED": "#ffe5cc",
                "WATCH": "#fff9cc",
                "BUY_DIP": "#ccffcc"
            }
            color = color_map.get(r["signal_type"], "#f0f0f0")
            
            html += f"""
<div style="background-color: {color}; padding: 10px; margin: 10px 0; border-left: 4px solid gray;">
<h3 style="margin: 0;">{r['emoji']} {r['symbol']} - {r['signal_type'].replace('_', ' ')}</h3>
<p style="margin: 5px 0;">
<b>Price:</b> ${r['current_price']:.2f} | 
<b>Shares:</b> {r['shares']:.0f} | 
<b>Value:</b> ${r['market_value']:,.2f}<br>
<b>Cost Basis:</b> ${r['cost_basis']:.2f} | 
<b>Gain:</b> <span style="color: {'green' if r['unrealized_gain_pct'] > 0 else 'red'};">${r['unrealized_gain']:,.2f} ({r['unrealized_gain_pct']:+.1f}%)</span><br>
<b>MA50:</b> ${r['ma_50']:.2f} | 
<b>RSI:</b> {r['rsi']:.0f} | 
<b>Drawdown:</b> {r['drawdown']*100:.1f}%
</p>
"""
            
            if r['action_note']:
                html += f"<p style='margin: 5px 0; font-weight: bold; color: #d9534f;'>💡 {r['action_note']}</p>"
            
            if r['stop_loss_price']:
                html += f"<p style='margin: 5px 0; background-color: #fff3cd; padding: 5px;'><b>🛑 Suggested Stop Loss:</b> ${r['stop_loss_price']:.2f} ({((r['stop_loss_price']/r['current_price'])-1)*100:.1f}% from current)</p>"
            
            if r['recovery_potential'] > 0:
                html += f"<p style='margin: 5px 0;'><b>Recovery Potential:</b> {r['recovery_potential']*100:.0f}% (Days below MA50: {r['days_below_ma50']}, Vol: {r['volatility']*100:.0f}%)</p>"
            
            if r['signals']:
                html += "<p style='margin: 5px 0;'><b>Signals:</b><br>" + "<br>".join(f"• {s}" for s in r['signals']) + "</p>"
            
            html += "</div>"
        
        html += f"""
<hr>
<p style="font-size: 0.9em; color: gray;">
<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</i><br>
<i>Set stop losses in your broker to protect positions automatically.</i>
</p>
"""
        
        msg.attach(MIMEText(html, "html"))
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
            s.starttls()
            s.login(EMAIL_FROM, EMAIL_PASSWORD)
            s.send_message(msg)
        
        print(f"✅ Portfolio report sent")
        print(f"   {sell_count} SELL | {trim_count} TRIM | {watch_count} WATCH | {hold_count} HOLD | {buy_count} BUY")
        print(f"   Total: ${total_val:,.2f} ({total_gain_pct:+.1f}%)")
        
    except FileNotFoundError:
        print(f"❌ Error: {PORTFOLIO_FILE} not found")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
