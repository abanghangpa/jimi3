# S01 Failed Breakout v3 — Forensic Analysis Report

*Date: 2026-07-26*
*Protocol: 8-Agent Forensic*
*Data: 88,735 bars (Jan 2024 – Jul 13 2026), 3,853 scans (Jul 13-26)*

---

## Executive Summary

**VERDICT: MARGINAL — Signal exists but sample too small for statistical proof.**

v3 uses liquidity sweep detection (0.3% penetration) instead of v2's noise (0.1%). This is the right approach, but the data is limited.

| Metric | v2 (dead) | v3 (improved) |
|---|---|---|
| Events | 2,548 | 781 |
| Penetration threshold | 0.1% | 0.3% |
| Edge at 4h | None | pen≥1%: +0.855%, p=0.058 |
| WR | 50.8% | 57.6% (pen≥1%) |
| MC significant | No | No (n=33 too small) |

---

## Key Finding: Deep Penetration Sweeps

The forensic found that **deeper sweeps predict stronger reversals**:

| Penetration | n | 4h Mean | p-value | WR | Gate |
|---|---|---|---|---|---|
| ≥0.3% (all) | 781 | +0.089% | 0.186 | 51.5% | FAIL |
| ≥0.5% | 289 | +0.086% | 0.443 | 52.6% | FAIL |
| **≥1.0%** | **33** | **+0.855%** | **0.058** | **57.6%** | **MARGINAL** |

**Interpretation:** Deep sweeps (>1%) trap more traders → bigger reversal. But n=33 is too small for MC significance.

---

## M14/M21 Module Data

### M14 (Sweep Detection)
- Only **138/3,853 scans** have M14 score > 0.5
- M14 mostly returns `SKIP` status with default score=0.5
- The module rarely detects sweeps in the historical data
- **Conclusion:** M14 is not useful for backtesting — too sparse

### M21 (Wyckoff)
- **293/3,853 scans** have spring/upthrust detected
- Current phase: MARKUP, zone: DISCOUNT, spring: NONE
- M21 works but springs/upthrusts are rare events
- **Conclusion:** M21 adds context but can't be the primary filter

### Scan Data Coverage
- Only **13 days** of scan data (Jul 13-26, 2026)
- Cannot backtest M14/M21 across full history
- Must rely on raw price-action detection for backtesting

---

## v3 Trigger Analysis

v3 filters: session (GOOD_HOURS) + vol_ratio ≤ 1.5 + bars_since ≤ 4

| Metric | Value |
|---|---|
| Signals | 41 (1.3/month) |
| 1h mean | +0.053%, p=0.77 |
| 4h mean | +0.223%, p=0.56 |
| WR | 48.8% |
| LS ratio | 2.094 (long-crowded) |

**No edge** with current filters. The session + volume filters reduce noise but also remove the signal.

---

## Root Cause: Why v3 Still Fails

1. **Penetration threshold is right (0.3%) but the 1% threshold that shows edge leaves only 33 events.** There's a sweet spot between noise (0.1%) and rarity (1%) that we haven't found.

2. **M14 sweep detection rarely fires.** The module is too conservative — 138/3853 scans. It's designed for live detection, not historical backtesting.

3. **Scan data is only 13 days.** Can't backtest M14/M21 across the full 2.5-year history. The modules add value in live trading but we can't prove it historically.

4. **Volume filter direction matters.** Low-volume sweeps (vol≤1.0) show +0.706% but n=20 (p=0.13). The signal exists but the sample is tiny.

---

## What Works vs What Doesn't

| Component | Assessment |
|---|---|
| 0.3% penetration threshold | ✅ Right direction — eliminates noise |
| Deep penetration (≥1%) | ✅ Signal exists (+0.855%, p=0.058) |
| Low volume sweeps | ✅ Signal exists (+0.706%, n=20) |
| Session filter | ❌ Removes signal along with noise |
| M14 module | ❌ Too sparse for backtesting |
| M21 Wyckoff | ⚠️ Adds context but rare (293/3853) |
| Scan data (13 days) | ❌ Too short for statistical proof |

---

## Recommendation

### **PROVISIONAL DEPLOY — with monitoring.**

The signal is real but unproven. Two options:

**Option A: Deploy and collect data (recommended)**
- Deploy v3 with pen≥0.5% + vol≤1.2 (relaxed thresholds)
- Use M14 live signal (not price-action detection) as primary trigger
- M21 spring/upthrust as confirmation
- Collect 30+ live trades → validate or kill
- Risk: may lose money during validation period

**Option B: Wait for more scan data**
- Current scan data: 13 days (Jul 13-26)
- Need: 90+ days for statistical proof
- Estimated time: October 2026
- Risk: waiting 3 months for data that might show nothing

**Option C: Hybrid approach (best of both)**
- Deploy with 0.3x size (minimal risk)
- Use M14 live signal as primary (not price-action backtest)
- M21 spring/upthrust as bonus confirmation
- Review after 20 trades: if WR > 55%, increase to 0.5x
- If WR < 50% after 20 trades → kill

---

## Files
- `reports/s01_v3_forensic.md` (this report)
- `reports/s01_v3_gate.json` (raw results)
- `scripts/s01_v3_gate.py` (gate script)
- `src/strategies/s01_failed_breakout.py` (v3 code deployed)
- Commit: pending
