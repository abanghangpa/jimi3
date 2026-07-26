# S01 Failed Breakout v2 — Forensic Analysis Report

*Date: 2026-07-26*
*Protocol: 8-Agent Forensic*
*Data: 88,735 bars (Jan 2024 – Jul 13 2026)*

---

## Executive Summary

**VERDICT: FAIL — No edge found in any regime, quality level, or parameter combination.**

| Metric | Value |
|---|---|
| Total events | 2,548 (82.7/month) |
| After S01 filters | 763 (24.8/month) |
| MC p-value | 0.159 |
| Bootstrap CI | [-0.030%, +0.096%] — includes zero |
| Best regime (2026) | +0.184%, p=0.082 — marginal, not significant |
| WR | 50.8% — coin flip |

---

## 8-Agent Results

### Agent 1: Forensics
- **2,548** deduplicated false breakout events detected
- LONG: 1,262, SHORT: 1,286 (balanced)
- Quality: 307 high (≥0.5), 2,110 medium (0.2-0.5), 131 low (<0.2)
- July 2026: 32 events

### Agent 2: Non-Indicator — Raw Edge
**No raw edge found.**

| Direction | Horizon | n | Effective Mean | p-value | WR | Gate |
|---|---|---|---|---|---|---|
| LONG | 1h | 1,262 | -0.031% | 0.203 | 52.4% | FAIL |
| LONG | 4h | 1,262 | -0.015% | 0.752 | 52.9% | FAIL |
| SHORT | 1h | 1,286 | -0.004% | 0.870 | 51.9% | FAIL |
| SHORT | 4h | 1,286 | -0.080% | 0.077 | 51.2% | FAIL |

Quality breakdown: all FAIL. Higher quality doesn't predict better returns.

### Agent 3: S01 v2 Trigger
- **763 signals** after session + EMA200 + volume filters
- 1h: -0.027%, p=0.32, WR=48.2% — **FAIL**
- 4h: -0.025%, p=0.69, WR=49.5% — **FAIL**

### Agent 4: Regime
**Only 2026 shows marginal edge:**

| Regime | n | 4h Mean | p-value | Gate |
|---|---|---|---|---|
| HIGH vol | 291 | +0.024% | 0.839 | FAIL |
| LOW vol | 188 | +0.057% | 0.559 | FAIL |
| MID vol | 284 | -0.128% | 0.138 | FAIL |
| BEAR | 420 | +0.053% | 0.549 | FAIL |
| BULL | 343 | -0.120% | 0.136 | FAIL |
| **2026** | **151** | **+0.184%** | **0.082** | **MARGINAL** |

EMA200 distance: no edge in any band.

### Agent 5: Frequency & Structural TP/SL
- Structural TP hit rate: **41.0%** (using swing levels)
- Fixed 2% TP hit rate: **12.0%**
- Structural TP is 3.4x better than fixed — but still no edge because SL also gets hit

### Agent 6: Co-occurrence
- Volume: 1.09x average (neutral)
- LS ratio: 2.163 (long-crowded)
- Direction: 52% LONG

### Agent 7: Sensitivity
**No parameter combination produces a passing signal.** Quality threshold, bars held, direction — none create edge.

### Agent 8: Monte Carlo
- n=2,548, mean=+0.033%, WR=50.8%
- MC p=0.159 — **NOT SIGNIFICANT**
- Bootstrap CI: [-0.030%, +0.096%] — includes zero

---

## Root Cause Analysis

### Why S01 Has No Edge

1. **15m timeframe is too noisy for swing level detection.** Swing highs/lows computed from 48 bars (12 hours) are frequently retested, creating many false breakout candidates that aren't meaningful liquidity traps.

2. **0.1% breakout threshold is too small.** At ETH ~$3,500, 0.1% = $3.50. This is within normal bid-ask bounce. Real false breakouts need 0.3-0.5% penetration.

3. **No volume confirmation on the breakout itself.** S01 checks volume *regime* but not whether the breakout bar had a volume spike. Real breakouts (that succeed) have high volume; false breakouts have low volume. This is the wrong filter direction.

4. **Session filter is too restrictive.** 763 filtered vs 2,548 raw = 70% of events filtered out. But the session filter doesn't add edge (compare: filtered 4h = -0.025% vs raw 4h = +0.033%).

5. **The strategy detects "failed breakout" but not "trap reversal."** A failed breakout is just the first condition. The *trap* (trapped traders forced to close) is what creates the edge. S01 doesn't detect the trap — it just detects the failure.

6. **2026 marginal edge is regime-specific noise.** 151 events with p=0.082 is not enough to claim edge, especially when all other eras show nothing.

---

## Research Context

### What Literature Says vs What S01 Does

| Research | S01 | Gap |
|---|---|---|
| Brunnermeier & Pedersen: Stop-loss cascades create edge when *many* stops are clustered | S01 doesn't check stop clustering | Missing key filter |
| Wyckoff: Springs/upthrusts require *prior accumulation/distribution* | S01 doesn't check Wyckoff phases | Missing context |
| Cespa & Foucault: Liquidity traps work when *informed traders* set the trap | S01 has no informed trader detection | Missing smart money filter |
| Real false breakouts: 0.3-0.5% penetration, low volume, quick reversal | S01 uses 0.1% threshold, any volume | Threshold too small |

---

## Recommendation

### **KILL S01 Failed Breakout v2.**

The strategy has no edge in any configuration. The concept is sound but the implementation is too noisy — it detects "price crossed a level" rather than "trapped traders are about to be squeezed."

### If Rework Is Desired

A correct implementation would need:
1. **Higher timeframe swing levels** (4h or daily, not 15m)
2. **0.3%+ penetration threshold** (real false breakouts, not noise)
3. **Volume divergence**: breakout bar has LOW volume (not real breakout)
4. **Stop clustering detection**: where are the stops? (requires orderbook data)
5. **Wyckoff context**: spring after accumulation = high conviction
6. **Reversal confirmation**: entry only after first reversal candle closes back inside

**Estimated effort:** 2-3 days. Requires 4h timeframe integration + orderbook data.

**Verdict:** Not worth the investment. Focus on strategies with proven edge (S20 v8b, S13, S22).

---

## Files
- `reports/s01_failed_breakout_forensic.json` (VPS)
- `reports/s01_failed_breakout_forensic.md` (local + VPS)
- `scripts/s01_8agent_v2.py` (VPS)
- Commit: pending
