---
name: "jimi-market-analysis"
description: "JIMI market analysis methodology for quantitative edge discovery and trade assessment in ETH/USDT and related crypto markets."
status: active
version: "v1"
---

# JIMI Market Analysis

## Purpose
A quantitative market analysis methodology for identifying edges in ETH/USDT and related crypto markets by evaluating evidence-based state models, regime-adaptive strategies, and statistical discipline.

## Version History
| Version | Date | Change | Reason |
|---------|------|--------|--------|
| 0.1 | 2026-09-18 | Created via /learn | Distilled from JIMI framework docs, scanner code, and reports |

## Prerequisites
- Access to JIMI scanner environment (`jimi_audit/`)
- Market data feeds: OHLCV, Derivatives (OI, funding, LS ratio), Order Flow (taker flow, OBI)
- Python 3.x with scanner dependencies installed
- Understanding of ICS (Integrated Conviction Score) framework
- Cron jobs: scanner runs on schedule, report generation automated

## Architecture Overview

### Scanner Pipeline Flow
```
Data Fetch → Indicator Population → Regime Detection → Direction Resolution
→ Strategy Evaluation (22+ modules) → Ensemble Gate → ICS Scoring → Report
```

### Strategy System
22+ strategy modules evaluated through a unified ensemble gate:
- S01-S06: Breakout, momentum, mean-reversion, liquidity-based
- S07-S12: Structural, flow, derivatives-driven
- S13-S22: Composite, cross-market, regime-adaptive

### Regime Detection
HMM-based classification with four states:
- **Bull:** Trend-following strategies favored
- **Bear:** Defensive/short strategies favored
- **Range:** Mean-reversion strategies favored
- **Stress:** Reduced exposure, high-conviction only

### Report Generation
Standardized 15-minute scan reports with mandatory verdict taxonomy.

---

## Core Procedures

### Procedure 1: Running a Market Scan

1. Execute scanner: `python3 jimi_audit/scripts/scanner.py`
2. Inspect output (`latest_scan.json`) for:
   - Indicator status (derivatives, order flow, structure)
   - Strategy signals (per-strategy conviction)
   - ICS score vs. 0.50 threshold
   - Regime classification
3. Check for staleness flags — stale data invalidates signals
4. Verify no pipeline errors in scanner output

### Procedure 2: Evaluating Strategies

**Signal Assessment:**
- Check `multi_strategy` output for ensemble conviction
- Apply M-Series Synergy Protocol (MSSP) to combine Liquidity, Micro-Flow, and Structural modules
- Ensure signals bypass gates only if specifically permitted by regime

**Performance Evaluation (per THINKING.md §6-§11):**
- Win rate (WR), Profit Factor (PF), Sample size (N)
- Require N ≥ 30 before drawing conclusions
- Check distribution: median vs mean (§8)
- Test for selection bias (§10)
- Validate out-of-sample (§11)

**When to Retire a Strategy:**
- WR < 40% over 50+ trades
- PF < 0.8 consistently
- Strategy contradicts current regime for 30+ days
- New evidence invalidates the underlying hypothesis (§13)

### Procedure 3: Generating Reports

**15-Min Scan Report Protocol:**
1. Run scanner → capture output
2. Compare against previous scan (`scan_history.json`)
3. Report flips (direction changes crossing thresholds)
4. Use exact verdict labels:
   - `STRONG SIGNAL` — ICS > 0.70, regime-aligned, high conviction
   - `WATCH` — ICS 0.50-0.70, mixed signals
   - `HOLD` — maintain current position
   - `AVOID` — ICS < 0.30 or regime conflict
   - `NO_SIGNAL` — insufficient data or no edge detected

**Report Sections:**
- Market state (price, volume, trend)
- Regime classification + confidence
- Derivatives snapshot (OI, funding, LS ratio)
- Top strategy signals with conviction scores
- Risk flags (staleness, divergence, extreme readings)
- Verdict + recommended action

### Procedure 4: Regime Detection

1. HMM classifier outputs regime + confidence
2. Cross-check with manual indicators:
   - Trend: EMA structure, higher highs/lows
   - Volatility: ATR expansion/contraction
   - Breadth: participation across timeframes
3. If regime confidence < 60%, treat as transitional
4. Regime changes trigger strategy re-evaluation

**Interaction Effects (per THINKING.md §3):**
A strategy that works in Bull may fail in Range. Always ask: "Under what regime does this edge hold?"

### Procedure 5: Optimization Cycle

**Experimental Discipline (per THINKING.md §6):**
1. Freeze current production strategy
2. Create challenger with single change
3. Run both against identical signals
4. Isolate incremental population (Champion REJECT / Challenger TRADE)
5. Compare: WR, PF, N, drawdown
6. Only promote if evidence supports

**Statistical Requirements:**
- N ≥ 30 trades minimum
- DSR (Deflated Sharpe Ratio) > 1.0
- Walk-forward WR > 50%
- Out-of-sample validation required

**When NOT to Optimize:**
- Less than 2 weeks since last change
- N < 30 for current configuration
- User says experimentation is finished (§14) — freeze and monitor

---

## Key Concepts

### Moneytaur State Model
```
Liquidity Created → Fuel Stored → Engine Ignites → Move → Exhaust
```
Understanding where price sits in this cycle determines strategy selection.

### Hard Gates vs Soft Features (THINKING.md §5)
- Hard gate: unacceptable condition (e.g., stale data, N < 10)
- Soft feature: contributes to quality (e.g., volume profile, trend alignment)
- Do not turn every predictor into a rejection rule

### Proxy Variables (THINKING.md §4)
Breadth ↑, Liquidity ↑, Performance ↑ — does breadth cause performance? Or is it a proxy for liquidity? Always control for the stronger variable.

### Small-N Discipline (THINKING.md §9)
N = 4, PF = 12 is NOT evidence. It's a question worth investigating.

---

## Known Limitations
- Data latency: stale data flags must be respected, never ignored
- Small sample sizes in new regime conditions
- Correlation ≠ causation in cross-market signals
- Historical backtests don't account for execution slippage perfectly
- Order book depth changes invalidate some structural signals

## Anti-Patterns
- **Cherry-picking:** Selecting favorable examples from a larger dataset
- **Overfitting:** Tuning thresholds until historical performance improves (§11)
- **Descriptive ≠ Predictive:** A pattern that describes past data isn't necessarily an edge
- **Ignoring regime:** Applying a Bull strategy in a Range market
- **Stacking proxies:** Multiple filters all measuring the same underlying phenomenon
- **Treating forward return as PnL:** Theoretical returns ≠ realized execution (§7)
- **Continuous optimization:** Changing thresholds every week against the same data

## References
For detailed strategy protocols, optimization framework, and reporting taxonomy, see:
- `JIMI_FRAMEWORK.md` — full framework documentation
- `OPTIMIZATION_FRAMEWORK.md` — detailed optimization procedures
- `STRATEGY_PROTOCOLS.md` — per-strategy signal definitions
- `REPORTING.md` — report generation standards
- `THINKING.md` — analytical reasoning framework

## Usage Log
| Date | Task | Result | Notes |
|------|------|--------|-------|
| 2026-09-18 | Skill created | ✅ | Distilled from 10 JIMI source files via /learn |