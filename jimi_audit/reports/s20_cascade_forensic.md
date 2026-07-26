# S20 Liquidation Cascade — Forensic Analysis Report

*Date: 2026-07-26*
*Protocol: 8-Agent Forensic + Research Review*
*Data: 88,735 bars (Jan 2024 – Jul 13 2026), 3,403 with OI coverage (3.8%)*

---

## Executive Summary

**VERDICT: KILL — Strategy has no edge. v6.1 is curve-fitted noise.**

The S20 Liquidation Cascade strategy is conceptually sound but empirically dead. The 8-agent forensic reveals:

1. **The trigger fires on 79.6% of all bars** (70,675/88,735) — it's not detecting cascades, it's detecting "market exists"
2. **Win rate: 50.2%** — statistically indistinguishable from coin flip
3. **Monte Carlo p-value: 0.32** — not significant by any standard
4. **OI data coverage: 3.8%** — the strategy operates on 96.2% interpolated/missing data
5. **The one real signal is backwards**: OI *drops* (liquidations happening) predict mean reversion, not continuation

---

## Research Context

### Key Papers

1. **"Anatomy of a Crypto Cascade: Minute-Level Evidence"** (SSRN 6579278, 2026)
   - Findings: Cascades are characterized by OI dropping >5% in 15min, price displacement >3%, volume spike >5x
   - Duration: 2-15 minutes (NOT 15m bars)
   - Key insight: The cascade happens *inside* a single 15m bar — by the time the bar closes, it's over

2. **"Explainable Patterns in Crypto Microstructure"** (arXiv 2602.00776, 2026)
   - Liquidation events are self-limiting: they exhaust within minutes
   - Post-cascade: mean reversion (price bounces) within 30-60 minutes
   - OI *drop* is the signal, not OI *surge*

3. **"Anatomy of Oct 2025 Liquidation Cascade"** (ResearchGate, 2025)
   - Macro trigger (NFP miss) + overleveraged positioning = cascade
   - Cross-exchange contagion within 5 minutes
   - Recovery within 2 hours

### What Research Says vs What S20 Does

| Research Finding | S20 Implementation | Gap |
|---|---|---|
| Cascade = OI *drop* >5% in minutes | S20 looks for OI *surge* >1.5% | **Direction is wrong** |
| Cascade duration: 2-15 minutes | S20 uses 15m bars (already too slow) | **Timeframe too slow** |
| Post-cascade = mean reversion | S20 trades continuation (LONG) | **Direction is wrong** |
| Volume spike >5x | S20 doesn't check volume | **Missing critical filter** |
| Requires overleveraged positioning | S20 checks LS ratio <0.7 (too loose) | **Threshold too loose** |

---

## 8-Agent Forensic Results

### Agent 1: Forensics — Data Coverage

| Metric | Value | Assessment |
|---|---|---|
| OI coverage | 3,403/88,735 (3.8%) | **CRITICAL: 96% of bars have no OI data** |
| OI ROC > 1% events | 73 | Very rare |
| OI ROC > 1.5% events | 28 | Extremely rare |
| July 2026 OI coverage | 520/1,183 bars | 44% (improved but still sparse) |

**Diagnosis:** The strategy's OI-based triggers can only fire on 3.8% of bars. When it does fire on the other 96.2%, it's using stale/wrong OI data (the `pd.merge_asof` with 2h tolerance propagates the last known value forward).

### Agent 2: Non-Indicator — Raw Signal Edge

**PASSING signals (p < 0.10):**

| Signal | Horizon | n | Mean Return | p-value | Direction |
|---|---|---|---|---|---|
| OI surge 1% + price disp 1% | 4h | 24 | +0.606% | 0.055 | LONG |
| OI drop 1% + price disp 0.5% | 4h | 65 | +0.459% | 0.035 | LONG |
| OI drop (all) | 4h | 96 | +0.324% | 0.043 | LONG |

**Critical finding:** OI *drops* (liquidations happening) predict POSITIVE returns at 4h. This is **mean reversion** — after liquidations exhaust, price bounces. S20 trades this as LONG continuation, which is correct direction but wrong logic (it looks for OI *surges*, not drops).

### Agent 3: Indicator — S20 Trigger Logic

| Metric | Value |
|---|---|
| Total cascade signals | 70,675 (79.6% of all bars!) |
| July 2026 signals | 577 |
| Win rate | 50.2% |
| Mean return (4h) | +0.007% |

**The trigger is broken.** The conditions:
- OI ROC > 1.5% → Only 28 events pass this
- LS ratio < 0.7 → Only 44 events fail this
- Price not crashing → 14,657 events fail this

But `all_pass = 70,675` means **the strategy fires on everything** because the conditions are checked with `or` logic (any cascade detection returns a signal), and the OI ROC check uses `<=` (greater than threshold) while the data has 96% stale values that happen to pass.

### Agent 4: Regime — Edge by Regime

| Regime | 4h Mean | p-value | Assessment |
|---|---|---|---|
| HIGH vol | +0.039% | 0.001 | Marginal edge |
| LOW vol | -0.019% | 0.003 | **Negative** |
| BULL | +0.011% | 0.096 | Marginal |
| BEAR | +0.002% | 0.833 | No edge |
| **2026** | **-0.064%** | **<0.0001** | **CATASTROPHIC** |

**2026 is killing the strategy.** The regime shifted — what worked in 2024_H1 (+0.038%, p=0.0002) and 2025_H2 (+0.035%, p=0.002) is now deeply negative in 2026 (-0.064%, p<0.0001).

### Agent 5: Gate — Whale Watch Dependency

- 70.9% of cascade events are within 2% of EMA200
- Whale watch confirmation adds no filtering value — most events already near EMA200
- The dependency on whale_watch is a bottleneck that further reduces an already-dead signal

### Agent 6: Co-occurrence

- Zero co-occurrence events found (FR>0.0005+LS>1.5 = 0 events)
- The LS ratio condition is so loose it never actually filters
- Funding during cascade: mean 0.000016 (near zero) — cascades happen in neutral funding environments

### Agent 7: Sensitivity

- **No passing configurations found** in the threshold sweep
- Every combination of OI ROC and LS ratio thresholds produces noise
- The signal itself has no edge to extract

### Agent 8: Monte Carlo

| Metric | Value |
|---|---|
| Events | 70,662 |
| Actual mean | +0.0067% |
| Actual WR | 50.2% |
| MC p-value (mean) | 0.3243 |
| Bootstrap CI (mean) | [-0.003%, +0.017%] |
| **Significant** | **NO** |

---

## Root Cause Analysis

### Why v6.1 is Dead

1. **Wrong signal direction**: S20 looks for OI *surges* (new positions opening). Research shows cascades are OI *drops* (forced liquidations). The strategy is looking at the wrong side of the trade.

2. **Timeframe too slow**: Cascades happen in 2-15 minutes. By the time a 15m bar closes, the cascade is over and mean reversion has started. S20 enters *after* the cascade, when the bounce is already happening — but it enters as if the cascade is still going.

3. **Trigger fires on everything**: The OI ROC > 1.5% condition should filter to 28 events, but stale OI data (96% interpolated) makes it fire on 70,675 bars. The strategy is not detecting cascades — it's detecting "market exists."

4. **Whale watch dependency adds noise**: It doesn't filter for cascade quality — it filters for "whale activity," which is a different signal entirely.

5. **2026 regime shift**: Whatever marginal edge existed in 2024-2025 has evaporated. The market microstructure changed.

### Why v7 (23% WR) Was Even Worse

v7 added more aggressive thresholds but kept the wrong signal direction. Tightening a broken trigger just reduces the sample size while keeping the noise.

---

## Recommendation

### **KILL S20 permanently.**

The concept of trading liquidation cascades is valid, but the implementation is fundamentally wrong in every dimension:
- Wrong signal direction (surge vs drop)
- Wrong timeframe (15m vs tick/1m)
- Wrong trade logic (continuation vs mean reversion)
- Wrong data (96% interpolated OI)

### If Rework Is Desired (v8)

Based on research, a correct implementation would:

1. **Signal**: OI *drop* >3% in 15m (not surge)
2. **Direction**: LONG (mean reversion after liquidation exhaust)
3. **Timeframe**: Need tick/1m data for proper cascade detection
4. **Entry**: 15-30 minutes after cascade starts (when OI stabilizes)
5. **TP**: 0.5-1.5% (mean reversion target, not trend-following)
6. **SL**: Below cascade low
7. **Filters**: Volume spike >3x, funding rate extreme (>0.05% or <-0.05%)
8. **Regime**: Only in HIGH vol regimes

**Estimated effort:** Complete rewrite, 2-3 days, requires tick data pipeline.

**Verdict:** Not worth it. Kill and focus on strategies with proven edge.

---

## Files Generated

- `cascade_8agent.py` — 8-agent forensic script (deployed to VPS)
- `reports/cascade_8agent_forensic.json` — raw results (on VPS)
- `reports/s20_cascade_forensic.md` — this report

## Next Steps

1. Disable S20 in executor
2. Move to next strategy for forensic analysis
3. Consider: S01 failed_breakout (regime-specific edge), S06 liquidity_grab (v7 just deployed), or S11 cross_asset (highest per-trade return but low sample)
