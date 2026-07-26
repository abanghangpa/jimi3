# Strategy Dependency Map

*Created: 2026-07-26*
*Purpose: Track which strategies/modules depend on what, so modifications don't silently break things.*

---

## Dependency Rules

1. **Before modifying a booster** → re-run dependent strategy's gate
2. **After modifying a booster** → run ALL dependent gates, compare before/after
3. **Never modify a booster and dependent in the same commit** — separate changes
4. **Snapshot booster version** when deploying a dependent strategy

---

## Strategy → Booster Dependencies

### S19 orderbook_imbalance v6
| Booster | Usage | Validated With |
|---|---|---|
| cross_asset | Scenario A co-filter (68.0% WR, PF 2.12) | Gate pass 2026-07-22 |
| whale_watch | Scenario B co-filter (68.5% WR, PF 2.18) | Gate pass 2026-07-22 |
| momentum_v3 | Scenario C co-filter (72.7% WR, PF 2.67) | Gate pass 2026-07-22 |
| regime | Direction bonus (BULL+LONG, BEAR+SHORT) | Built-in |

**Impact if booster changes:**
- cross_asset signal change → Scenario A breaks
- whale_watch signal change → Scenario B breaks
- momentum_v3 signal change → Scenario C breaks
- regime change → conviction scoring shifts

---

### S21 trade_flow v4.2
| Booster | Usage | Validated With |
|---|---|---|
| taker_flow | Scenario A confirmation (73.9% WR, PF 4.67) | Gate pass 2026-07-24 |
| orderbook_imbalance | Co-filter | Gate pass 2026-07-24 |
| whale_watch | Co-filter | Gate pass 2026-07-24 |
| funding_arb | Co-filter | Gate pass 2026-07-24 |

**Impact if booster changes:**
- taker_flow signal change → Scenario A breaks (primary scenario)
- OBI/whale/funding change → co-filter quality degrades

---

### S01 failed_breakout v3
| Booster | Usage | Validated With |
|---|---|---|
| M14 (sweep detection) | Primary sweep detection | Full backtest 2026-07-26 |
| M21 (Wyckoff) | Phase + zone context (ACCUMULATION required) | Full backtest 2026-07-26 |
| M5 (structural) | TP levels | Structural TP 41% hit rate |
| derivatives | LS ratio, funding rate for positioning | Integrated |
| taker_summary | Taker flow direction for divergence | Integrated |
| regime | Direction filter | Integrated |

**Impact if booster changes:**
- M14 threshold change → sweep detection changes → all signals affected
- M21 phase change → ACCUMULATION filter changes → signals rejected/accepted
- M5 level change → TP targets shift → R:R changes

---

### S04 positioning_fade v4
| Booster | Usage | Validated With |
|---|---|---|
| M9 (vol regime) | Regime filter (RANGING/HIGH_VOL/CRISIS only) | Gate pass 2026-07-14 |
| derivatives | LS z-score for extreme detection | Integrated |
| whale_watch | Confirmation bonus | Gate pass 2026-07-14 |

**Impact if booster changes:**
- M9 regime change → BAD_REGIMES list changes → signals filtered differently
- whale_watch change → confirmation bonus affected

---

### S05 kill_zone v2
| Booster | Usage | Validated With |
|---|---|---|
| M21 (Wyckoff) | kill_zone, phase, zone detection | Built-in |

**Impact if booster changes:**
- M21 kill_zone change → session detection breaks
- M21 phase change → phase-based logic affected

---

### S06 liquidity_grab v7
| Booster | Usage | Validated With |
|---|---|---|
| regime | Scenario A requires BEAR regime | Gate pass 2026-07-24 |
| M5 (orderbook) | OB ratio for bid/ask imbalance | Integrated |

**Impact if booster changes:**
- regime change → BEAR filter changes → Scenario A breaks
- M5 OB change → imbalance detection shifts

---

### S08 regime_switch
| Booster | Usage | Validated With |
|---|---|---|
| M9 (vol regime) | Vol regime scoring | Built-in |
| M22 (inflation) | Inflation regime scoring | Built-in |
| M23 (macro) | Macro regime scoring | Built-in |
| cascade_risk | Cascade severity scoring | Built-in |

**Impact if booster changes:**
- Any module change → regime scoring shifts → direction/conviction affected

---

### S20 liquidation_cascade v8b
| Booster | Usage | Validated With |
|---|---|---|
| derivatives | OI ROC for cascade detection | Gate pass 2026-07-26 |
| regime | Regime context | Integrated |

**Impact if booster changes:**
- derivatives OI change → cascade detection breaks (core signal)
- regime change → context shifts

---

## Module → Dependent Strategies

### M14 (Sweep Detection)
**Used by:** S01, S16
**Risk level:** HIGH — primary signal for S01

### M21 (Wyckoff Phase)
**Used by:** S01, S05
**Risk level:** HIGH — context filter for S01, primary for S05

### M5 (Structural Levels)
**Used by:** S01, S16
**Risk level:** MEDIUM — TP targets, not signal triggers

### M9 (Vol Regime)
**Used by:** S04, S08, S10
**Risk level:** HIGH — regime filter for multiple strategies

### Regime Classifier (V5)
**Used by:** S06, S19, S20 + all strategies via REGIME_STRATEGY_GATE
**Risk level:** CRITICAL — affects ALL strategies

### Derivatives (OI, LS, FR)
**Used by:** S01, S04, S13, S14, S19, S20, S21, S25
**Risk level:** CRITICAL — data source for most strategies

### Taker Summary
**Used by:** S01, S07, S21
**Risk level:** MEDIUM — confirmation signal

### Whale Watch (S14)
**Used by:** S04, S19, S21
**Risk level:** HIGH — co-filter for multiple strategies

### Cross Asset (S11)
**Used by:** S19
**Risk level:** MEDIUM — co-filter for S19

### Taker Flow (S07)
**Used by:** S21
**Risk level:** MEDIUM — primary scenario for S21

### Funding Arb (S13)
**Used by:** S21
**Risk level:** LOW — co-filter only

### Orderbook Imbalance (S19)
**Used by:** S21
**Risk level:** LOW — co-filter only

---

## Modification Protocol

### Before Modifying Any Booster

1. **Identify dependents** — check this file
2. **Run dependent gates** — get baseline metrics (WR, p, n)
3. **Save baseline** — commit gate results

### After Modifying a Booster

1. **Re-run dependent gates** — compare to baseline
2. **If metrics degrade >5%** — investigate, may need to adjust dependent
3. **If metrics improve** — update dependent's validation date in this file
4. **Commit booster + dependent updates together** — atomic change

### Emergency: Booster Broke a Dependent

1. **Revert booster** to last known good version (.bak file)
2. **Re-run dependent gate** — confirm fix
3. **Investigate** why the change broke the dependent
4. **Re-design** the dependency to be more robust

---

## Version Snapshots

| Booster | Current Version | Last Validated | Dependents |
|---|---|---|---|
| M14 | m14_sweep.py (default config) | 2026-07-26 | S01 |
| M21 | m21_wyckoff.py (default config) | 2026-07-26 | S01, S05 |
| M9 | scanner_executor.py | 2026-07-24 | S04, S08, S10 |
| Regime V5 | regime_classifier_v5.py | 2026-07-24 | ALL |
| whale_watch | s14_whale_watch.py | 2026-07-22 | S04, S19, S21 |
| cross_asset | s11_cross_asset.py | 2026-07-22 | S19 |
| taker_flow | s07_taker_flow.py | 2026-07-24 | S21 |
| funding_arb | s13_funding_arb.py | 2026-07-22 | S21 |
| OBI | s19_orderbook_imbalance.py | 2026-07-22 | S21 |

---

## Risk Matrix

| If We Modify... | Affected Strategies | Risk Level | Re-test Required |
|---|---|---|---|
| Regime Classifier V5 | ALL | CRITICAL | ALL gates |
| Derivatives data | S01, S04, S13, S14, S19, S20, S21, S25 | CRITICAL | 8 gates |
| M14 | S01, S16 | HIGH | 2 gates |
| M21 | S01, S05 | HIGH | 2 gates |
| M9 | S04, S08, S10 | HIGH | 3 gates |
| whale_watch | S04, S19, S21 | HIGH | 3 gates |
| cross_asset | S19 | MEDIUM | 1 gate |
| taker_flow | S21 | MEDIUM | 1 gate |
| M5 | S01, S16 | MEDIUM | 2 gates |
| taker_summary | S01, S07, S21 | MEDIUM | 3 gates |
| funding_arb | S21 | LOW | 1 gate |
| OBI | S21 | LOW | 1 gate |
