"""
Portfolio Monitor - Long-Term Mean Reversion Strategy Support
Alerts on: delisting risk, sharp drops (buy opportunities), real threats only
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

# Configuration
EMAIL_FROM = "tshprung@gmail.com"
EMAIL_TO = "tshprung@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
PORTFOLIO_FILE = "portfolio.csv"
STATE_FILE = "portfolio_state.json"

# Thresholds for REAL threats only
SHARP_DROP_1DAY = -0.15  # -15% in 1 day = alert
SHARP_DROP_1WEEK = -0.25  # -25% in 1 week = alert
DELISTING_RISK_PRICE = 1.00  # Stock below $1 = delisting risk
VOLUME_SURGE = 5.0  # 5x volume spike = something happening
BANKRUPTCY_THRESHOLD = -0.70  # -70% from peak = investigate

def calculate_financial_health(symbol):
    """Check bankruptcy/delisting risk"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Financial health indicators
        cash = info.get('totalCash', 0)
        debt = info.get('totalDebt', 0)
        revenue = info.get('totalRevenue', 0)
        market_cap = info.get('marketCap', 0)
        current_price = info.get('currentPrice', 0)
        
        # Risk factors
        risks = []
        risk_score = 0
        
        # Price-based delisting risk
        if current_price < DELISTING_RISK_PRICE:
            risks.append(f"Price ${current_price:.2f} below $1 - NASDAQ delisting risk")
            risk_score += 30
        
        # Debt to cash ratio
        if cash > 0 and debt > 0:
            debt_ratio = debt / cash
            if debt_ratio > 5:
                risks.append(f"High debt: {debt_ratio:.1f}x cash")
                risk_score += 20
            elif debt_ratio > 2:
                risks.append(f"Moderate debt: {debt_ratio:.1f}x cash")
                risk_score += 10
        
        # No revenue (speculative)
        if revenue == 0:
            risks.append("No revenue - pure speculation")
            risk_score += 15
        
        # Market cap risk
        if market_cap < 100e6:  # <$100M
            risks.append(f"Micro cap: ${market_cap/1e6:.0f}M")
            risk_score += 10
        
        return {
            'risk_score': min(risk_score, 100),
            'risks': risks,
            'cash': cash,
            'debt': debt,
            'revenue': revenue,
            'market_cap': market_cap
        }
    except:
        return {'risk_score': 0, 'risks': [], 'cash': 0, 'debt': 0, 'revenue': 0, 'market_cap': 0}

def analyze_stock(symbol, shares, cost_basis):
    """Analyze stock for mean reversion opportunities and real threats"""
    
    # Fetch data
    try:
        stock = yf.download(symbol, period="1y", progress=False)
        if stock.empty:
            return None
    except:
        return None
    
    current_price = float(stock["Close"].iloc[-1])
    current_value = current_price * shares
    total_gain = current_value - (cost_basis * shares)
    gain_pct = (current_price / cost_basis - 1) * 100
    
    # Financial health check
    health = calculate_financial_health(symbol)
    
    # Price movement analysis
    price_1d_ago = float(stock["Close"].iloc[-2]) if len(stock) >= 2 else current_price
    price_1w_ago = float(stock["Close"].iloc[-5]) if len(stock) >= 5 else current_price
    price_1m_ago = float(stock["Close"].iloc[-21]) if len(stock) >= 21 else current_price
    
    drop_1d = (current_price / price_1d_ago - 1)
    drop_1w = (current_price / price_1w_ago - 1)
    drop_1m = (current_price / price_1m_ago - 1)
    
    # Find historical peak (1 year)
    peak_price = float(stock["Close"].max())
    peak_date = stock["Close"].idxmax()
    drawdown_from_peak = (current_price / peak_price - 1)
    
    # Volume analysis
    avg_volume = float(stock["Volume"].rolling(20).mean().iloc[-1])
    current_volume = float(stock["Volume"].iloc[-1])
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    
    # Calculate if stock is at good reversion level
    # Was it stable at peak for a while?
    peak_area = stock["Close"].iloc[-252:] if len(stock) >= 252 else stock["Close"]
    stable_high_series = peak_area.quantile(0.80)
    stable_high = float(stable_high_series.iloc[0]) if isinstance(stable_high_series, pd.Series) else float(stable_high_series)
    reversion_potential = (stable_high / current_price - 1) * 100
    
    # Signal generation
    signals = []
    signal_type = "HOLD"
    actions = []
    
    # 🚨 CRITICAL ALERTS (Real threats)
    if health['risk_score'] >= 50:
        signal_type = "🚨 URGENT"
        signals.append(f"DELISTING RISK ({health['risk_score']}%)")
        actions.append(f"🚨 INVESTIGATE IMMEDIATELY: {', '.join(health['risks'])}")
    
    # Sharp drops (potential buying opportunities OR danger)
    if drop_1d <= SHARP_DROP_1DAY:
        signals.append(f"📉 SHARP DROP 1-day: {drop_1d*100:.1f}%")
        if health['risk_score'] < 30:
            actions.append(f"💰 Consider averaging down if fundamentals intact")
        else:
            actions.append(f"⚠️ Check news - drop + high risk = danger")
    
    if drop_1w <= SHARP_DROP_1WEEK:
        signals.append(f"📉 SEVERE DROP 1-week: {drop_1w*100:.1f}%")
        if health['risk_score'] < 30:
            actions.append(f"💰 Strong averaging down opportunity if thesis holds")
        else:
            actions.append(f"⚠️ HIGH RISK - verify company viability")
    
    # Massive drawdown from peak
    if drawdown_from_peak <= BANKRUPTCY_THRESHOLD:
        signals.append(f"💀 DOWN {abs(drawdown_from_peak)*100:.0f}% from peak")
        actions.append(f"⚠️ Verify company survival - extreme drop")
    
    # Volume surge (something happening)
    if volume_ratio >= VOLUME_SURGE:
        signals.append(f"📢 VOLUME SURGE: {volume_ratio:.1f}x normal")
        actions.append(f"📰 Check news - unusual activity")
    
    # Mean reversion opportunity
    if reversion_potential > 50 and health['risk_score'] < 30:
        signals.append(f"📊 REVERSION POTENTIAL: +{reversion_potential:.0f}%")
        actions.append(f"💡 Stock down {abs(drawdown_from_peak)*100:.0f}% from stable ${stable_high:.2f} area")
        if gain_pct < -20:
            actions.append(f"💰 Currently down {abs(gain_pct):.1f}% - good averaging zone")
    
    # No urgent issues
    if not signals:
        signal_type = "✅ STABLE"
        signals.append("No urgent issues")
    elif "🚨" in str(signals):
        signal_type = "🚨 URGENT"
    elif "📉" in str(signals):
        signal_type = "⚠️ ALERT"
    
    return {
        'symbol': symbol,
        'current_price': current_price,
        'cost_basis': cost_basis,
        'shares': shares,
        'gain_pct': gain_pct,
        'total_gain': total_gain,
        'current_value': current_value,
        'signal_type': signal_type,
        'signals': signals,
        'actions': actions,
        'health': health,
        'peak_price': peak_price,
        'peak_date': peak_date.strftime('%Y-%m-%d') if hasattr(peak_date, 'strftime') else str(peak_date),
        'drawdown_from_peak': drawdown_from_peak * 100,
        'reversion_target': stable_high,
        'reversion_potential': reversion_potential,
        'drop_1d': drop_1d * 100,
        'drop_1w': drop_1w * 100,
        'drop_1m': drop_1m * 100,
        'volume_ratio': volume_ratio
    }

def load_portfolio():
    """Load portfolio from CSV"""
    df = pd.read_csv(PORTFOLIO_FILE)
    holdings = []
    for _, row in df.iterrows():
        holdings.append({
            'symbol': row['Symbol'],
            'shares': float(row['Shares']),
            'cost_basis': float(row['Avg Cost/Share'])
        })
    return holdings

def generate_email_report(results):
    """Generate HTML email report"""
    
    # Sort by urgency
    urgent = [r for r in results if '🚨' in r['signal_type']]
    alerts = [r for r in results if '⚠️' in r['signal_type'] and r not in urgent]
    stable = [r for r in results if r not in urgent and r not in alerts]
    
    total_value = sum(r['current_value'] for r in results)
    total_gain = sum(r['total_gain'] for r in results)
    total_gain_pct = (total_gain / (total_value - total_gain)) * 100 if (total_value - total_gain) != 0 else 0
    
    html = f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h2>📊 Portfolio Monitor - Long-Term Strategy</h2>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M ET')}</p>
    
    <div style="background: #f0f0f0; padding: 15px; margin: 20px 0; border-radius: 5px;">
        <h3>Portfolio Summary</h3>
        <p><strong>Total Value:</strong> ${total_value:,.2f}</p>
        <p><strong>Total Gain:</strong> ${total_gain:,.2f} ({total_gain_pct:+.1f}%)</p>
        <p><strong>Urgent Issues:</strong> {len(urgent)} | <strong>Alerts:</strong> {len(alerts)} | <strong>Stable:</strong> {len(stable)}</p>
    </div>
    """
    
    # Urgent section
    if urgent:
        html += "<h3>🚨 URGENT ATTENTION NEEDED</h3>"
        for r in urgent:
            html += f"""
            <div style="background: #ffe6e6; padding: 15px; margin: 10px 0; border-left: 5px solid #cc0000; border-radius: 5px;">
                <h4>{r['symbol']}: ${r['current_price']:.2f} | {r['gain_pct']:+.1f}%</h4>
                <p><strong>Risk Score:</strong> {r['health']['risk_score']}/100</p>
                <p><strong>Signals:</strong> {', '.join(r['signals'])}</p>
                <p><strong>⚠️ ACTIONS:</strong></p>
                <ul>{''.join(f"<li>{a}</li>" for a in r['actions'])}</ul>
                <p><strong>Financial:</strong> Cash ${r['health']['cash']/1e6:.0f}M | Debt ${r['health']['debt']/1e6:.0f}M | Revenue ${r['health']['revenue']/1e6:.0f}M</p>
            </div>
            """
    
    # Alerts section
    if alerts:
        html += "<h3>⚠️ ALERTS (Opportunities or Concerns)</h3>"
        for r in alerts:
            html += f"""
            <div style="background: #fff3cd; padding: 15px; margin: 10px 0; border-left: 5px solid #ff9800; border-radius: 5px;">
                <h4>{r['symbol']}: ${r['current_price']:.2f} | {r['gain_pct']:+.1f}%</h4>
                <p><strong>Peak:</strong> ${r['peak_price']:.2f} ({r['peak_date']}) | <strong>Drawdown:</strong> {r['drawdown_from_peak']:.1f}%</p>
                <p><strong>Signals:</strong> {', '.join(r['signals'])}</p>
                {f"<p><strong>💡 Mean Reversion Target:</strong> ${r['reversion_target']:.2f} (+{r['reversion_potential']:.0f}% upside)</p>" if r['reversion_potential'] > 50 else ""}
                <p><strong>Actions:</strong></p>
                <ul>{''.join(f"<li>{a}</li>" for a in r['actions'])}</ul>
            </div>
            """
    
    # Stable section (brief)
    if stable:
        html += "<h3>✅ STABLE HOLDINGS</h3>"
        html += "<table style='width:100%; border-collapse: collapse;'>"
        html += "<tr style='background: #e0e0e0;'><th>Symbol</th><th>Price</th><th>Gain</th><th>Status</th></tr>"
        for r in stable:
            html += f"""
            <tr style='border-bottom: 1px solid #ddd;'>
                <td><strong>{r['symbol']}</strong></td>
                <td>${r['current_price']:.2f}</td>
                <td style='color: {"green" if r["gain_pct"] > 0 else "red"};'>{r['gain_pct']:+.1f}%</td>
                <td>{r['signals'][0]}</td>
            </tr>
            """
        html += "</table>"
    
    html += """
    <div style="margin-top: 30px; padding: 15px; background: #f9f9f9; border-radius: 5px;">
        <h4>📌 Strategy Reminder</h4>
        <p><strong>Your Approach:</strong> Long-term mean reversion</p>
        <ul>
            <li>✅ Hold quality stocks through volatility</li>
            <li>💰 Average down on sharp drops if fundamentals intact</li>
            <li>🚨 Exit only on delisting/bankruptcy risk</li>
            <li>⏳ Wait years for reversion to stable levels</li>
        </ul>
    </div>
    </body></html>
    """
    
    return html

def send_email(subject, html_body):
    """Send email alert"""
    if not EMAIL_PASSWORD:
        print("No email password set")
        return
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = subject
        msg.attach(MIMEText(html_body, 'html'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"✅ Email sent: {subject}")
    except Exception as e:
        print(f"❌ Email failed: {e}")

def main():
    print(f"📊 Portfolio Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)
    
    # Load portfolio
    holdings = load_portfolio()
    print(f"Analyzing {len(holdings)} holdings...")
    
    # Analyze each stock
    results = []
    for holding in holdings:
        print(f"  • {holding['symbol']}...", end=" ")
        result = analyze_stock(holding['symbol'], holding['shares'], holding['cost_basis'])
        if result:
            results.append(result)
            print(f"{result['signal_type']}")
        else:
            print("FAILED")
    
    if not results:
        print("❌ No results")
        return
    
    # Count urgency
    urgent_count = len([r for r in results if '🚨' in r['signal_type']])
    alert_count = len([r for r in results if '⚠️' in r['signal_type']])
    
    print(f"\n📊 Summary: {urgent_count} urgent | {alert_count} alerts | {len(results)-urgent_count-alert_count} stable")
    
    # Generate report
    html = generate_email_report(results)
    
    # Send email (always send, even if all stable)
    subject = f"📊 Portfolio: "
    if urgent_count > 0:
        subject += f"🚨 {urgent_count} URGENT"
    elif alert_count > 0:
        subject += f"⚠️ {alert_count} ALERTS"
    else:
        subject += "✅ All Stable"
    
    send_email(subject, html)
    
    # Save state
    state = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    print("✅ Done")

if __name__ == "__main__":
    main()
