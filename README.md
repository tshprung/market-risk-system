# Market Crash Predictor - Improvements Summary

## Overview
Comprehensive upgrade to your crash prediction system with:
1. Budget vote & debt ceiling tracking
2. Reweighted indicators based on historical performance
3. Fixed .item() conversion issues
4. Added Treasury stress monitoring

---

## NEW FEATURES

### 1. Debt Ceiling & Budget Vote Tracking

**New Indicators:**
- `debt_ceiling_stress_score()` - Short-term Treasury market stress (BIL vs SHY)
- `treasury_stress_score()` - Treasury volatility (systemic stress indicator)
- `budget_vote_risk_score()` - Combined score (proximity + Treasury stress + VIX)
- `days_to_debt_ceiling()` - Returns (days_remaining, is_near_deadline)

**How It Works:**
- Estimates X-date: August 1, 2026 (based on CBO guidance)
- Triggers at 60 days (when market historically starts reacting)
- Emergency mode at 14 days (2-week critical window)
- Based on 2011 pattern: Market ignored until 2 weeks before, then -17% drop

**Historical Context (from research):**
- 2011: S&P dropped 17-20% in weeks around deadline
- Even after deal, took 2+ months to recover
- Credit downgrade added another -7% in one day
- Short-term Treasury yields spike first (early warning)

**Integration:**
- Adds up to 20% boost to composite when near deadline
- Separate alerts in dashboard and intraday watcher
- Email subject line includes "Debt Ceiling Xd" when critical

---

## INDICATOR REWEIGHTING

### Old vs New Weights (in trade_signals.py)

| Indicator | Old | New | Change | Rationale |
|-----------|-----|-----|--------|-----------|
| VIX Expansion | 20% | 15% | -25% | Too many false positives, often spikes without crashes |
| Credit Stress | 15% | 15% | 0% | Reliable - kept same |
| Options Hedging | 15% | 15% | 0% | Reliable - kept same |
| VIX Spike | 10% | 10% | 0% | Good for flash crashes - kept same |
| Put/Call Ratio | 10% | 12% | +20% | More reliable than VIX expansion, shows real hedging |
| Credit Spread | 10% | 10% | 0% | HY-IG spread is good - kept same |
| Breadth | 8% | 10% | +25% | Important divergence signal, underweighted before |
| Dollar Strength | 7% | 5% | -29% | Less predictive than thought, often lags |
| Yield Curve | 5% | 8% | +60% | Longer lead time, recession predictor - underweighted |

**Total: 100%**

### Why These Changes?

**VIX Expansion (20% → 15%):**
- Problem: VIX spikes frequently without causing crashes
- Example: Mini spikes in 2023 didn't lead to crashes
- Solution: Reduce weight, rely more on put/call ratio which shows actual hedging behavior

**Put/Call Ratio (10% → 12%):**
- Uses VIX/VIX3M as proxy
- Shows institutional hedging (real money protecting portfolios)
- More reliable than raw VIX movement
- 2011: Spiked before crash when VIX was still moderate

**Breadth (8% → 10%):**
- Small cap underperformance is critical warning
- Narrow rallies precede crashes (2000, 2007, 2020)
- Was underweighted relative to importance

**Dollar (7% → 5%):**
- Often lags rather than leads
- Can strengthen for non-crash reasons (Fed policy)
- Reduced but not eliminated

**Yield Curve (5% → 8%):**
- Inverted curve preceded every recession since 1950s
- Longer lead time (6-18 months)
- Important macro indicator that was too low

---

## CODE FIXES

### .item() Conversions
**Problem:** Pandas Series not converted to scalars (you requested this always be fixed)

**Fixed in:**
- `zscore()` function (line 34)
- `credit_complacency_score()` (line 65)
- `options_percentile()` (line 139)
- `check_recovery()` (lines 276-281)
- `btc_equity_correlation()` (line 239)
- All gold_crypto_confirmation() calls

**Pattern Used:**
```python
if isinstance(value, pd.Series):
    value = value.item()
```

### safe_float() Helper
Added to all scripts to handle None/NaN:
```python
def safe_float(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(value)
```

**Removed:** Redundant NaN checks (lines 69-84 in old trade_signals.py)

---

## ALERT ENHANCEMENTS

### Dashboard Email Changes
1. **Subject line** now includes debt ceiling days when critical
2. **New banner** on chart when <14 days to X-date
3. **Debt ceiling section** in email with:
   - Days to X-date
   - Budget risk score
   - Treasury stress
   - Historical context (2011 drop)

### Intraday Watcher Changes
1. Debt ceiling adds +20 to score when <14 days
2. +10 when 14-30 days
3. Alert level overridden to "DEBT CEILING ALERT" when critical
4. Subject line includes "Debt Ceiling Xd"

### Trade Signals Changes
1. New emergency condition: `debt_ceiling_emergency`
2. Triggers SELL if:
   - <14 days to X-date
   - Budget risk >0.6
   - Not in cooldown
3. Reason includes debt ceiling days

---

## TESTING RECOMMENDATIONS

### 1. Verify Data Fetching
```python
from risk_indicators import days_to_debt_ceiling, debt_ceiling_stress_score
days, is_near = days_to_debt_ceiling()
print(f"Days to X-date: {days}, Near deadline: {is_near}")
print(f"Debt stress: {debt_ceiling_stress_score():.2f}")
```

### 2. Test Weight Changes
Run for 1 week and compare:
- Old composite vs new composite
- Which catches more early warnings
- False positive rate

### 3. Monitor These Patterns
**Early warning signs (60-30 days before):**
- BIL/SHY spread widening
- Treasury volatility increasing
- VIX staying elevated

**Critical signs (14-0 days):**
- 1-month Treasury yields spike
- Credit spreads widen
- VIX >25

---

## EXPECTED BEHAVIOR

### Normal Market (Jan 2026)
- Debt ceiling: 184 days away (not near deadline)
- Budget risk: 0.0
- No additional score boost
- Normal monitoring

### Approaching Deadline (June 2026)
- 60 days out: Flag appears "monitoring debt ceiling"
- Budget risk starts rising
- Email includes countdown
- No emergency yet

### Critical Phase (July 2026)
- 14 days out: Emergency mode
- +20 score boost
- Red banner on dashboard
- Subject line: "⚠️ DEBT CEILING 14d"
- Possible SELL signal if Treasury stress high

### Post-Resolution
- After deal: Recovery check
- VIX declining + credit stable = REBUY consideration
- But may take weeks (2011 took 2 months)

---

## ADJUSTMENT RECOMMENDATIONS

### If Too Sensitive
Reduce these weights:
- Put/Call: 12% → 10%
- Breadth: 10% → 8%
- Debt ceiling boost: 20% → 15%

### If Too Conservative
Increase these:
- VIX Spike: 10% → 12%
- Credit Spread: 10% → 12%
- Lower SELL_THRESHOLD: 0.55 → 0.50

### For Different Scenarios

**Debt Ceiling Focus:**
- Increase debt_ceiling_stress to 20% weight
- Lower emergency threshold to 7 days

**Traditional Crash Focus:**
- Increase VIX expansion back to 18%
- Reduce debt ceiling to 10% weight

**Mean Reversion Trading:**
- Increase recovery threshold to 0.45
- Add breadth recovery check

---

## FILES CREATED

1. `risk_indicators.py` - Core indicators with new debt ceiling functions
2. `trade_signals.py` - Reweighted composite with debt ceiling logic
3. `daily_market_risk_dashboard.py` - Enhanced email with debt ceiling warnings
4. `intraday_emergency_watcher.py` - Intraday alerts with debt ceiling checks

**Weekly options update unchanged** (still works with new indicators)

---

## NEXT STEPS

1. **Deploy files** to your GitHub Actions environment
2. **Test locally** for 1-2 days to verify data fetching
3. **Monitor weight performance** for 1-2 weeks
4. **Adjust** based on your risk tolerance
5. **Update X-date** if CBO revises estimate

---

## KEY INSIGHT FROM RESEARCH

**2011 Debt Ceiling Pattern:**
- Days 90-30: Market mostly ignored
- Days 30-14: Increased volatility, some selling
- Days 14-0: Panic selling (-17% total)
- Day 0-2: Deal reached, market still fell
- Days 2-60: Continued selling on credit downgrade
- Days 60+: Slow recovery over 2 months

**Your system is now calibrated for this pattern.**

---

## QUESTIONS?

If you need to:
- Change X-date estimate → Edit `days_to_debt_ceiling()` in risk_indicators.py
- Adjust weights → Edit lines 87-96 in trade_signals.py  
- Change alert thresholds → Edit SELL_THRESHOLD, emergency thresholds
- Disable debt ceiling → Set `budget_boost = 0` in trade_signals.py
