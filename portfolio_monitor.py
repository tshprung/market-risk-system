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
EXTENDED_GAIN_THRESHOLD = 0.20

# Defensive dividend stocks
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

def get_earnings_date(symbol):
    """Get next earnings date for a stock"""
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        if calendar is not None and not calendar.empty:
            if 'Earnings Date' in calendar.index:
                e_date = calendar.loc['Earnings Date'].values[0]
            else:
                e_date = calendar.iloc[0, 0]
                
            earnings_date = pd.Timestamp(e_date).tz_localize(None)
            days_until = (earnings_date - datetime.now()).days
            return earnings_date.strftime('%Y-%m-%d'), days_until
    except:
        pass
    return None, None

def calculate_scaling_targets(current_price, cost_basis, gain_pct, signal_type):
    """Calculate profit-taking targets for winners"""
    targets = []
    if gain_pct < 10:
        return targets
    
    if 20 <= gain_pct < 50:
        targets.append({
            "price": cost_basis * 1.50,
            "gain_pct": 50,
            "action": "Sell 25%",
            "reason": "Lock early profits"
        })
        targets.append({
            "price": cost_basis * 2.00,
            "gain_pct": 100,
            "action": "Sell 25%",
            "reason": "Secure double"
        })
    elif gain_pct >= 50:
        targets.append({
            "price": current_price * 1.20,
            "gain_pct": gain_pct * 1.20,
            "action": "Sell 30%",
            "reason": "Take chips off table"
        })
        targets.append({
            "price": current_price * 1.40,
            "gain_pct": gain_pct * 1.40,
            "action": "Sell 30%",
            "reason": "Lock major gains"
        })
    return targets

def calculate_expected_return(current_price, target_price, shares, probability=0.3):
    """Calculate expected value of reaching a target"""
    potential_gain = (target_price - current_price) * shares
    expected_value = potential_gain * probability
    return potential_gain, expected_value

def calculate_days_below_ma(prices, ma_period=50):
    """Count consecutive days price has been below moving average"""
    if len(prices) < ma_period: return 0
    ma = prices.rolling(ma_period).mean()
    below_ma = prices < ma
    count = 0
    for val in reversed(below_ma.values):
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
    return float(vol)

def analyze_stock(symbol, shares, cost_basis, prev_state=None):
    stock_raw = yf.download(symbol, period="1y", progress=False)
    if stock_raw.empty: return None
    
    if isinstance(stock_raw.columns, pd.MultiIndex):
        stock = pd.DataFrame({
            "Close": stock_raw["Close"][symbol],
            "Volume": stock_raw["Volume"][symbol]
        })
    else:
        stock = stock_raw
        
    current_price = float(stock["Close"].iloc[-1])
    volume = stock["Volume"]
    
    ma_50 = float(stock["Close"].rolling(50).mean().iloc[-1])
    ma_200 = float(stock["Close"].rolling(200).mean().iloc[-1]) if len(stock) >= 200 else None
    rsi = calculate_rsi(stock["Close"])
    
    peak_price = float(stock["Close"].rolling(60).max().iloc[-1])
    drawdown = (current_price / peak_price) - 1
    
    avg_vol = float(volume.tail(20).mean())
    curr_vol = float(volume.iloc[-1])
    vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
    
    days_below_ma50 = calculate_days_below_ma(stock["Close"], 50)
    volatility = calculate_volatility(stock["Close"])
    
    days_held = None
    if prev_state and symbol in prev_state:
        first_seen = prev_state[symbol].get('first_seen')
        if first_seen:
            days_held = (datetime.now() - datetime.fromisoformat(first_seen)).days
    
    earnings_date, days_to_earnings = get_earnings_date(symbol)
    
    market_value = current_price * shares
    unrealized_gain = market_value - (cost_basis * shares)
    gain_pct = (unrealized_gain / (cost_basis * shares)) * 100 if cost_basis > 0 else 0
    
    is_defensive = symbol in DEFENSIVE_TICKERS
    
    signals = []
    signal_type = "HOLD"
    risk_score = 0
    stop_loss_price = None
    action_note = ""
    
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
            action_note = "Defensive position - normal volatility"
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
    
    scaling_targets = calculate_scaling_targets(current_price, cost_basis, gain_pct, signal_type)
    
    expected_returns = []
    if gain_pct > 10:
        for multiplier, label in [(1.5, "50% gain"), (2.0, "100% gain"), (3.0, "200% gain")]:
            target = cost_basis * multiplier
            if target > current_price:
                potential, expected = calculate_expected_return(current_price, target, shares, probability=0.3)
                expected_returns.append({
                    "target": target, "label": label, "potential_gain": potential, "expected_value": expected
                })
    
    return {
        "symbol": symbol, "current_price": current_price, "shares": shares, "cost_basis": cost_basis,
        "market_value": market_value, "unrealized_gain": unrealized_gain, "unrealized_gain_pct": gain_pct,
        "ma_50": ma_50, "ma_200": ma_200, "rsi": rsi, "drawdown": drawdown, "volume_ratio": vol_ratio,
        "days_below_ma50": days_below_ma50, "volatility": volatility, "recovery_potential": recovery_potential,
        "signals": signals, "signal_type": signal_type, "risk_score": risk_score, "emoji": emoji,
        "stop_loss_price": stop_loss_price, "action_note": action_note, "is_defensive": is_defensive,
        "scaling_targets": scaling_targets, "expected_returns": expected_returns, "days_held": days_held,
        "earnings_date": earnings_date, "days_to_earnings": days_to_earnings
    }

def calculate_portfolio_beta(symbols):
    try:
        spy_df = yf.download("SPY", period="6mo", progress=False)
        spy = spy_df["Close"] if not isinstance(spy_df.columns, pd.MultiIndex) else spy_df["Close"]["SPY"]
        betas = []
        for s in symbols:
            s_df = yf.download(s, period="6mo", progress=False)
            if s_df.empty: continue
            stock = s_df["Close"] if not isinstance(s_df.columns, pd.MultiIndex) else s_df["Close"][s]
            combined = pd.DataFrame({"spy": spy, "stock": stock}).dropna()
            if len(combined) > 30:
                returns = combined.pct_change().dropna()
                betas.append(returns['stock'].cov(returns['spy']) / returns['spy'].var())
        return float(np.mean(betas)) if betas else 1.0
    except:
        return 1.0

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        df = pd.read_csv(PORTFOLIO_FILE)
        try:
            with open(STATE_FILE) as f: prev_state = json.load(f)
        except: prev_state = {}
        
        results = []
        for _, row in df.iterrows():
            res = analyze_stock(row["Symbol"], float(row["Shares"]), float(row["Avg Cost/Share"]), prev_state)
            if res: results.append(res)
        
        results.sort(key=lambda x: x["risk_score"], reverse=True)
        
        total_val = sum(r["market_value"] for r in results)
        total_gain = sum(r["unrealized_gain"] for r in results)
        total_gain_pct = (total_gain / (total_val - total_gain)) * 100 if total_val > 0 else 0
        p_beta = calculate_portfolio_beta([r["symbol"] for r in results])
        
        sell_count = sum(1 for r in results if r["signal_type"] == "SELL")
        trim_count = sum(1 for r in results if r["signal_type"] in ["TRIM", "TRIM_EXTENDED"])
        watch_count = sum(1 for r in results if r["signal_type"] == "WATCH")
        buy_count = sum(1 for r in results if r["signal_type"] == "BUY_DIP")
        hold_count = len(results) - sell_count - trim_count - watch_count - buy_count
        
        earnings_warnings = [f"{r['symbol']} earnings in {r['days_to_earnings']} days" for r in results if r['days_to_earnings'] and 0 < r['days_to_earnings'] <= 7]
        
        portfolio_warnings = []
        if sell_count >= 3: portfolio_warnings.append(f"⚠️ {sell_count} positions need selling")
        if p_beta > 1.3 and total_gain_pct < 0: portfolio_warnings.append(f"⚠️ High risk (Beta {p_beta:.2f})")
        if total_gain_pct < -10: portfolio_warnings.append(f"⚠️ Portfolio down {total_gain_pct:.1f}%")
        portfolio_warnings.extend(earnings_warnings)
        
        new_alerts = []
        for r in results:
            prev_sig = prev_state.get(r["symbol"], {}).get("signal_type", "HOLD")
            if r["signal_type"] in ["SELL", "TRIM", "TRIM_EXTENDED"] and prev_sig not in ["SELL", "TRIM", "TRIM_EXTENDED"]:
                new_alerts.append(f"{r['emoji']} {r['symbol']} → {r['signal_type']}")
        
        current_state = {}
        for r in results:
            first_seen = prev_state.get(r["symbol"], {}).get('first_seen', datetime.now().isoformat())
            current_state[r["symbol"]] = {"signal_type": r["signal_type"], "price": r["current_price"], "gain_pct": r["unrealized_gain_pct"], "first_seen": first_seen}
        
        with open(STATE_FILE, "w") as f: json.dump(current_state, f, indent=2)
        
        # Build Email
        subject = f"🔴 Portfolio: {sell_count} SELL / {trim_count} TRIM" if (sell_count + trim_count) > 0 else "Portfolio: Healthy"
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = EMAIL_FROM, EMAIL_TO, subject
        
        html = f"""
<h2>Portfolio Health Report</h2>
<p><b>Total Value:</b> ${total_val:,.2f} | <b>Gain:</b> ${total_gain:,.2f} (<span style="color: {'green' if total_gain_pct > 0 else 'red'};">{total_gain_pct:+.1f}%</span>)</p>
<p><b>Portfolio Beta:</b> {p_beta:.2f} | <b>Signals:</b> {sell_count} SELL | {trim_count} TRIM | {buy_count} BUY | {hold_count} HOLD</p>
"""
        
        if portfolio_warnings:
            html += f"<div style='background-color:#fff3cd;padding:10px;border-left:4px solid orange;margin:10px 0;'><b>📊 WARNINGS:</b><br>{'<br>'.join(portfolio_warnings)}</div>"
        
        if new_alerts:
            html += f"<div style='background-color:#ffcccc;padding:10px;border-left:4px solid red;margin:10px 0;'><b>🚨 NEW ALERTS:</b><br>{'<br>'.join(new_alerts)}</div>"
        
        for r in results:
            bg = {"SELL":"#ffcccc","TRIM":"#ffe5cc","TRIM_EXTENDED":"#ffe5cc","WATCH":"#fff9cc","BUY_DIP":"#ccffcc"}.get(r["signal_type"], "#f0f0f0")
            html += f"""
<div style="background-color: {bg}; padding: 10px; margin: 10px 0; border-left: 4px solid gray;">
<h3 style="margin: 0;">{r['emoji']} {r['symbol']} - {r['signal_type'].replace('_', ' ')}</h3>
<p><b>Price:</b> ${r['current_price']:.2f} | <b>Shares:</b> {r['shares']:.0f} | <b>Value:</b> ${r['market_value']:,.2f}<br>
<b>Gain:</b> <span style="color: {'green' if r['unrealized_gain_pct'] > 0 else 'red'};">${r['unrealized_gain']:,.2f} ({r['unrealized_gain_pct']:+.1f}%)</span><br>
<b>RSI:</b> {r['rsi']:.0f} | <b>Drawdown:</b> {r['drawdown']*100:.1f}%</p>
"""
            if r['action_note']: 
                html += f"<p style='color:#d9534f;font-weight:bold;'>💡 {r['action_note']}</p>"
            if r['stop_loss_price']: 
                html += f"<p style='background:#fff3cd;padding:5px;'><b>🛑 Stop Loss:</b> ${r['stop_loss_price']:.2f} ({((r['stop_loss_price']/r['current_price'])-1)*100:.1f}% from current)</p>"
            if r['scaling_targets']:
                html += "<p style='background:#d4edda;padding:5px;'><b>📈 Profit Targets:</b><br>" + "".join([f"• {t['action']} at ${t['price']:.2f} (+{t['gain_pct']:.0f}%) - {t['reason']}<br>" for t in r['scaling_targets']]) + "</p>"
            if r['expected_returns']:
                html += "<p><b>💰 Expected Returns (30% probability):</b><br>"
                for exp in r['expected_returns']:
                    html += f"• {exp['label']}: Target ${exp['target']:.2f}, Potential ${exp['potential_gain']:,.0f}, Expected ${exp['expected_value']:,.0f}<br>"
                html += "</p>"
            if r['days_held']:
                held_text = f"<b>Held:</b> {r['days_held']} days"
                if r['earnings_date'] and r['days_to_earnings']:
                    held_text += f" | <b>Earnings:</b> {r['earnings_date']} (in {r['days_to_earnings']} days)"
                html += f"<p style='font-size:0.9em;color:#666;'>{held_text}</p>"
            if r['recovery_potential'] > 0: 
                html += f"<p><b>Recovery Potential:</b> {r['recovery_potential']*100:.0f}% (Below MA50: {r['days_below_ma50']} days, Vol: {r['volatility']*100:.0f}%)</p>"
            if r['signals']:
                html += "<p><b>Signals:</b><br>" + "<br>".join(f"• {s}" for s in r['signals']) + "</p>"
            html += "</div>"
        
        html += "<hr><p style='font-size:0.9em;color:gray;'><i>Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M UTC') + "</i></p>"
        
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
            s.starttls()
            s.login(EMAIL_FROM, EMAIL_PASSWORD)
            s.send_message(msg)
        print("✅ Portfolio report sent successfully")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
