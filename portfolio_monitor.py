"""
Portfolio Monitor - Fixed Version
Handles zero cost_basis errors and provides stock analysis
"""

import yfinance as yf
import json
from datetime import datetime, timedelta
import os

# Portfolio holdings (update with your actual holdings)
HOLDINGS = [
    {"symbol": "BSX", "shares": 100, "cost_basis": 91.62},
    {"symbol": "INTU", "shares": 100, "cost_basis": 434.09},
    {"symbol": "SAP.DE", "shares": 30, "cost_basis": 168.0},
    {"symbol": "ABSI", "shares": 700, "cost_basis": 3.77},
    {"symbol": "LI", "shares": 800, "cost_basis": 17.15},
    {"symbol": "ATGE", "shares": 90, "cost_basis": 95.0},
    {"symbol": "ENPH", "shares": 300, "cost_basis": 29.8},
    {"symbol": "KMB", "shares": 386, "cost_basis": 99.0},  # Average
    {"symbol": "MOH", "shares": 78, "cost_basis": 151.0},  # Average
    {"symbol": "CE", "shares": 300, "cost_basis": 41.8},
    {"symbol": "RGTI", "shares": 966.2, "cost_basis": 31.5},  # Average
    {"symbol": "TTD", "shares": 1860, "cost_basis": 48.7},  # Average
    {"symbol": "PAGS", "shares": 1500, "cost_basis": 8.92},
    {"symbol": "JD", "shares": 870, "cost_basis": 31.5},  # Average
]

def safe_division(numerator, denominator, default=0):
    """Safely divide, return default if denominator is zero"""
    if denominator == 0 or denominator is None:
        return default
    try:
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default

def analyze_stock(symbol, shares, cost_basis):
    """
    Analyze a single stock position
    Returns dict with current price, gain/loss, and recommendation
    """
    try:
        # Fetch current data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        
        if hist.empty:
            return {
                "symbol": symbol,
                "shares": shares,
                "cost_basis": cost_basis,
                "error": "No data available"
            }
        
        current_price = float(hist['Close'].iloc[-1])
        
        # Calculate gains (with zero protection)
        if cost_basis is None or cost_basis <= 0:
            gain_pct = 0
            gain_amount = 0
            print(f"  ⚠️  {symbol}: Invalid cost_basis ({cost_basis}), using current price as baseline")
            cost_basis = current_price  # Use current price as baseline
        else:
            gain_pct = safe_division(current_price - cost_basis, cost_basis, 0) * 100
            gain_amount = (current_price - cost_basis) * shares
        
        position_value = current_price * shares
        
        # Get volatility (5-day standard deviation)
        volatility = float(hist['Close'].pct_change().std() * 100) if len(hist) > 1 else 0
        
        # 5-day performance
        if len(hist) >= 5:
            five_day_change = safe_division(
                current_price - float(hist['Close'].iloc[0]),
                float(hist['Close'].iloc[0]),
                0
            ) * 100
        else:
            five_day_change = 0
        
        # Simple recommendation logic
        recommendation = "HOLD"
        if gain_pct > 20 and volatility > 5:
            recommendation = "🎯 TAKE PROFITS"
        elif gain_pct < -15:
            recommendation = "⚠️ REVIEW"
        elif five_day_change < -5:
            recommendation = "⚠️ WATCH"
        elif gain_pct > 10 and volatility < 3:
            recommendation = "✅ STABLE"
        
        return {
            "symbol": symbol,
            "shares": shares,
            "cost_basis": cost_basis,
            "current_price": current_price,
            "position_value": position_value,
            "gain_pct": gain_pct,
            "gain_amount": gain_amount,
            "volatility": volatility,
            "five_day_change": five_day_change,
            "recommendation": recommendation,
            "error": None
        }
        
    except Exception as e:
        return {
            "symbol": symbol,
            "shares": shares,
            "cost_basis": cost_basis,
            "error": str(e)
        }

def generate_portfolio_summary(results):
    """Generate overall portfolio statistics"""
    total_value = sum(r.get('position_value', 0) for r in results if r.get('error') is None)
    total_cost = sum(r.get('cost_basis', 0) * r.get('shares', 0) for r in results if r.get('error') is None and r.get('cost_basis'))
    
    total_gain = safe_division(total_value - total_cost, total_cost, 0) * 100 if total_cost > 0 else 0
    
    # Count recommendations
    take_profits = sum(1 for r in results if '🎯' in r.get('recommendation', ''))
    warnings = sum(1 for r in results if '⚠️' in r.get('recommendation', ''))
    stable = sum(1 for r in results if '✅' in r.get('recommendation', ''))
    
    return {
        "total_value": total_value,
        "total_cost": total_cost,
        "total_gain_pct": total_gain,
        "total_gain_amount": total_value - total_cost,
        "take_profits_count": take_profits,
        "warnings_count": warnings,
        "stable_count": stable
    }

def main():
    """Main portfolio monitoring function"""
    print("=" * 80)
    print(f"📊 Portfolio Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print()
    
    print(f"Analyzing {len(HOLDINGS)} holdings...")
    
    results = []
    for holding in HOLDINGS:
        print(f"  • {holding['symbol']}...", end=" ", flush=True)
        
        result = analyze_stock(
            holding['symbol'],
            holding['shares'],
            holding.get('cost_basis', 0)  # Default to 0 if missing
        )
        
        results.append(result)
        
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
        else:
            print(result.get('recommendation', 'HOLD'))
    
    print()
    print("=" * 80)
    print("PORTFOLIO SUMMARY")
    print("=" * 80)
    
    summary = generate_portfolio_summary(results)
    
    print(f"\n💰 Total Portfolio Value: ${summary['total_value']:,.2f}")
    print(f"📈 Total Gain/Loss: ${summary['total_gain_amount']:,.2f} ({summary['total_gain_pct']:+.2f}%)")
    print()
    
    # Detailed holdings table
    print("HOLDINGS DETAIL:")
    print("-" * 80)
    print(f"{'Stock':<8} {'Shares':<8} {'Cost':<10} {'Current':<10} {'Value':<12} {'Gain %':<10} {'5d %':<8} {'Rec'}")
    print("-" * 80)
    
    # Sort by gain_pct (worst first)
    sorted_results = sorted(
        [r for r in results if r.get('error') is None],
        key=lambda x: x.get('gain_pct', 0)
    )
    
    for r in sorted_results:
        gain_pct = r.get('gain_pct', 0)
        gain_color = "🟢" if gain_pct > 0 else "🔴" if gain_pct < -5 else "🟡"
        
        print(f"{r['symbol']:<8} "
              f"{r['shares']:<8.0f} "
              f"${r['cost_basis']:<9.2f} "
              f"${r['current_price']:<9.2f} "
              f"${r['position_value']:<11.2f} "
              f"{gain_color}{gain_pct:+6.2f}% "
              f"{r.get('five_day_change', 0):+7.2f}% "
              f"{r.get('recommendation', 'HOLD')}")
    
    print("-" * 80)
    print()
    
    # Action items
    print("🎯 ACTION ITEMS:")
    print()
    
    if summary['take_profits_count'] > 0:
        print(f"✓ {summary['take_profits_count']} position(s) ready to take profits:")
        for r in results:
            if '🎯' in r.get('recommendation', ''):
                print(f"  - {r['symbol']}: +{r['gain_pct']:.1f}% gain (consider selling 30-50%)")
        print()
    
    if summary['warnings_count'] > 0:
        print(f"⚠️  {summary['warnings_count']} position(s) need attention:")
        for r in results:
            if '⚠️' in r.get('recommendation', ''):
                reason = "down -15%+" if r['gain_pct'] < -15 else "dropped -5%+ this week"
                print(f"  - {r['symbol']}: {reason} (review fundamentals)")
        print()
    
    if summary['stable_count'] > 0:
        print(f"✅ {summary['stable_count']} position(s) performing well and stable")
        print()
    
    # Save to JSON for other scripts to use
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "holdings": results
    }
    
    with open("portfolio_status.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("=" * 80)
    print("✅ Portfolio analysis complete. Data saved to portfolio_status.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
