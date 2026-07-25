# MEMORY.md — JIMI Framework Knowledge

## TP/SL Rules (learned 2026-07-02)
- TP=SL=$15 (1:1 R:R) does NOT work — 46% WR, coin flip
- $15 is noise-level for ETH ($1,600 asset, $66 daily range)
- Wide SL ($30) is required to survive intraday noise before $15 TP is hit
- Production config: TP=$15 SL=$30 → 74% WR, +$3.30/trade expected
- 70% WR with equal TP/SL is NOT achievable at $15 granularity

## Signal Quality (learned 2026-07-02)
- Only 2 strategies are profitable: orderbook_imbalance (+0.254 avg_RR) and trade_flow (+0.214)
- regime_switch is the worst: 31.9% WR, fires 100% of scans, should be disabled or capped
- SHORT signals outperform LONG in downtrend periods
- Higher conviction correlates with higher WR (43% for 0.7+ vs 37.6% for 0.5-0.7)
- Simulated direction accuracy (price moved right way) is very different from actual trade outcomes (TP hit before SL)

## Data Quality (learned 2026-07-02)
- Filter fields (ensemble_passes, sweep_blocked, m20_blocked) are NOT persisted in scan files
- Without these fields, filter analysis is meaningless — always check filter_data_quality first
- strategy_signals.jsonl has actual fired signals with entry/SL/TP — use this for outcome analysis, not scan files

## Cron Job Config
- JIMI Deep Analysis: 08:10 UTC, model=free-proxy/qwen/qwen3.6-27b
- Model was changed from openrouter/free due to rate limiting

## TP/SL Update (2026-07-02)
- Changed from TP=$15 SL=$30 to TP=$12 SL=$36 (1:3 R:R)
- Backtest: 79.8% WR, $2.30 EV/trade (vs 72.1% WR, $2.44 EV with old config)
- Rationale: wider SL survives noise, tighter TP hits more often
- Also deployed: strategy-specific volume gating (orderbook_imbalance, trade_flow, cross_asset require vol_ratio > 0.12-0.15)
- Also deployed: EMA200 + vol_ratio in scan output and signal logging

## Session Updates (2026-07-02)
### TP/SL
- TP=$12 SL=$36 (1:3 R:R) replacing TP=$15 SL=$30
- TP: use liquidity pool if beyond $12 minimum
- SL: always enforce $36 minimum

### Volume Gating
- orderbook_imbalance, trade_flow: 0.15
- cross_asset: 0.12

### Scanner Fixes
- EMA200 + vol_ratio in output
- Flip prevention (1h window)
- Entry/SL/TP always shown
- Power of 3 phase-direction fix
- BaseStrategy cfg init fix


## Strategy Optimization Bug (discovered 2026-07-05)
The scanner optimization on 2026-07-05 broke strategy TP/SL multipliers. All strategies reported as "PF ≥ 2.0" are actually losing money because the optimization degraded their R:R ratios.

**Root cause:** _calc_levels() in base.py was called with wrong tp_mults/sl_mult values.
- whale_watch: tp_mults changed from (1.5,2.5,4.0) to (0.3,1.5,2.5), sl_mult from 1.0 to 2.0
- This made R:R go from 1.5:1 to 0.15:1 (10x worse)

**Fix plan:** Revert all strategies to .bak_pre_opt versions, verify R:R ≥ 1.0, re-backtest with 15m data.

**Lesson:** Always verify R:R ratio before deploying. Small TP/SL multiplier changes destroy edge completely.

## momentum_v3 Exhaustion Filter (2026-07-07)
- State filter (not standalone), pairs with event triggers
- v2: 9 weighted signals (decel, vol_div, percentile, OI_div, RSI, RSI_div, MACD_hist, wave, BB)
- Scoring: 0.0-1.2, threshold 0.30+
- Best combos: positioning_fade (9t, 77.8% WR, PF=2.02), funding_arb (6t, 100% WR, PF=inf)
- Trade count still too low to lock in — need 20+ trades


## Strategy Evaluation (2026-07-07)

**Architecture:** Group A (event/trigger) + Group B (state filters)
- A fires alone → trade at 1.0x
- A + B confirms → trade at 1.5x

**Enabled strategies:**
- Group A: orderbook_imbalance, positioning_fade, trade_flow
- Group B: funding_arb, cross_asset, momentum_v3, momentum_v2, squeeze_breakout

**Target hits (75% WR, 2.0 PF):**
- cross_asset + FR<0: 79.2% WR, PF=2.54 (24 trades)
- positioning_fade + exhaustion: 77.8% WR, PF=2.02 (9 trades)
- OBI + squeeze_breakout: 80% WR, PF=3.16 (20 trades)
- funding_arb + momentum_v3 + session: 80% WR, PF=4.87 (5 trades)

**Key lessons:**
- Disabled strategies are valuable as co-occurrence/context filters
- Session filtering is the strongest single quality filter
- LONG bias exists in most strategies
- Volume dead zone (1.0-1.5x) hurts most strategies
- EMA200 proximity: 1-3% above = best WR

## positioning_fade + Group B (2026-07-07)
- +cross_asset: 85.7% WR, PF=14.26 (7 trades) — best Group B filter
- +momentum_v2: 47.1% WR, PF=1.24 (87 trades) — solid, profitable
- +momentum_v3: 77.8% WR, PF=2.02 (9 trades) — exhaustion confirms
- Standalone: losing (39% WR) — needs Group B confirmation

### Trading System Evaluation (2026-07-12)

**Key Findings:**
- bb_mom6 is dead (negative returns across all regimes, even with extreme positioning confluence)
- failed_breakout is regime-specific (works in ranging/chop, fails in trending bears)
- positioning_fade + whale_watch only work in bearish/stress regimes
- trade_flow is the strongest by event count (6/9 scenarios PASS)
- 16 remaining strategies are fundamentally dead (0 signals with synthetic data)

**Architecture:**
- RegimeClassifier v2 uses multi-signal approach (derivatives + vol + macro + taker + cascade)
- Regime filters protect strategies from unfavorable conditions
- Position sizing capped by margin (80% of available)
- Executor runs6 strategies with data-driven regime filters

**Lessons:**
- Pooling events across eras hides regime-specific behavior (bb_mom6 looked mediocre pooled, negative in all regimes when stratified)
-13 events is not enough for a gate claim (bb_mom6 earlier result was noise)
- "Buy the paint" principle: use direct data when available, don't build inference engines
- Regime testing is mandatory before claiming an edge

**Files:**
- config/isolation_gate_results.json (regime-specific results)
- reports/scenario_gating.json (10 scenario results)
- reports/all_strats_scenario_gate.json (16 strategy results)
- reports/trade_flow_gate.json (trade_flow results)
- reports/bb_mom6_confluence_backtest.json (bb_mom6 confluence results)


### System Cleanup & State Reset (2026-07-14)

**Problem:** Executor crashed at 17:36 UTC on Jul 13 with `total_pnl` KeyError, then capital corrupted to -$1,064. Memory files were outdated — showed $-30 capital, executor "RUNNING" (actually DEAD), and old strategy statuses.

**Root Cause:** `load_state()` default dict was missing `pnl_total` key. When executor loaded an older state file, the KeyError cascaded into capital corruption.

**Fixes Applied:**
1. **Regime matrix cleaned** — removed 15 killed strategies, kept only 7 gate-passed (trade_flow, funding_arb, judas_sweep, orderbook_imbalance, positioning_fade, whale_watch, cross_asset)
2. **Executor configs updated** — trade_flow and orderbook_imbalance now direction-agnostic (was LONG-only), RISK_PCT reduced from 3% to 2%
3. **State files reset** — capital reset to $200, clean slate, all required keys present (total_pnl, pnl_total, total_fees)
4. **load_state() patched** — added missing `pnl_total` key to prevent future KeyErrors
5. **4 strategies enabled:** trade_flow, funding_arb, judas_sweep, orderbook_imbalance

**Current Gate Status (verified 2026-07-14):**
- PASS: trade_flow (623e, p=0.003, +0.214%), funding_arb (226e, p=0.054, +0.21%), judas_sweep (1895e, p=0.040, +0.103%), orderbook_imbalance (847e, p=0.001, +0.254%), positioning_fade (512e, p=0.045, +0.18%), whale_watch (340e, p=0.038, +0.16%), cross_asset (82e, p=0.0, +0.661%)
- KILLED (15): bb_mom6, momentum_v3, squeeze_breakout, scalp_v2, power_of_3, macro_surprise, liquidation_cascade, taker_flow, liquidity_grab, cascade, mtf_confluence, vol_rotation, kill_zone, structural_break, regime_switch

**Executor Status:** STOPPED (needs manual restart after fixes verified)
**Smart Proxy:** RUNNING but g4f returning 404 for qwen-3.6-27b model


### Strategy Overhaul Session (2026-07-14)

**Objective:** Audit all strategies, fix what's fixable, kill what's dead.

**Starting state:** 4 strategies enabled, capital -$1,064, executor dead.
**Ending state:** 13 strategies enabled, capital $200, executor running dry-run.

**Strategies upgraded (v2/v3):**
- S04 positioning_fade v3: rolling L/S stats, time-in-position filter, gate validated
- S07 taker_flow v3: df_15m fallback, regime-aware EMA, freshness check
- S13 funding_arb v6: 72h cumulative FR, SHORT support, no FR cap
- S19 orderbook_imbalance v3: SHORT direction, persistence, spoofing detection
- S20 liquidation_cascade v3: lower thresholds, OI shock, price-level awareness
- S21 trade_flow v3: z-score thresholds, flow acceleration, session filter
- S24 forced_movement v2: lower thresholds, basis widening, OI fallback

**Strategies built from scratch:**
- S01 failed_breakout v2: independent detection from df_15m, quality grading
- S02 squeeze_breakout v2: independent ATR/BB squeeze, Q>=0.80, 63% WR, p=0.0049
- S06 liquidity_grab v2: OB collector + independent S/R + persistence + spoofing
- S10 structural_break v2: independent BOS/ChoCH detection
- S14 whale_watch v2: real Etherscan on-chain data, contrarian L/S

**Executor improvements:**
- Kill zone session bonus: +0.08 London/NY overlap, +0.05 active sessions
- Regime matrix cleaned: removed 15 killed strategies
- Capital reset to $200 (was -$1,064 from total_pnl KeyError)
- RISK_PCT reduced from 3% to 2%

**Data collectors deployed:**
- fm-collector.timer: funding/OI/basis from Bybit every 5 min
- fm-liq-stream.service: Bybit liquidation websocket
- ob-collector.timer: orderbook snapshots every 60s

**Strategies killed (confirmed dead):**
- cross_asset: extended backtest (5000 bars, 1784 events) showed -0.018% mean, p=0.311
- bb_mom6: -0.714% mean return
- regime_switch: 31.9% WR, fires 100%
- scalp_v2, power_of_3, mtf_confluence, momentum_v3: synthetic confirmed dead
- vol_rotation, kill_zone, cascade, macro_surprise: no edge or missing data

**Key lessons:**
- 82 events is NOT enough for a gate claim (cross_asset was noise)
- Static thresholds don't adapt to market conditions (use z-score)
- Pipeline dependencies cause silent failures (always have fallback)
- Session timing is a filter, not a strategy
- Squeeze detection works when using ATR/BB compression (not module-dependent)


### Strategy Rework Session (2026-07-22)

**18 commits in one session.** Major strategy upgrades + new gate system.

**Strategy reworks (research-backed):**
- S19 orderbook_imbalance v5: trade-based OBI (Nittur Anantha 2025), concave conviction (Bieganowski 2026), VWAP deviation, spread filter, asymmetric entry
- S20 liquidation_cascade v5: bidirectional (added LONG cascade), regime-adaptive thresholds, 30min cooldown
- S06 liquidity_grab v6: killed v5 "ride the flow" (12.5% WR), replaced with volume spike + S/R break + momentum confirmation

**New systems deployed:**
- **RegimeClassifierV4**: daily timeframe regime detection, multi-signal (derivatives + vol + macro + taker + cascade)
- **Conditional Directional Gate**: blocks LONG+BEAR and SHORT+BULL (PF: LONG+BEAR=0.31, SHORT+BULL=0.29 — catastrophic)
- **DirectionalConsensusGate**: 5-metric veto (EMA200 distance, slope, regime, momentum, swing bias). BLOCK at 2.0+ weight, FLIP at 3.0+
- **Execution quality tracking**: daily_execution_report.py + execution_tracker.py
- **Quant analysis suite**: stress tests, regime experiments, conditional directional model

**Key data points:**
- Event-driven backtest: 392 trades replayed through real strategy logic
- LONG+BEAR = 0.31 PF, SHORT+BULL = 0.29 PF (both portfolio killers)
- LONG+BULL = 3.51 PF, SHORT+BEAR = 3.31 PF (both profitable)

**Lessons:**
- Research papers provide better signal than pure backtesting (trade-based OBI > quote-based OBI)
- Bidirectional strategies need regime awareness (LONG cascade works in bull, SHORT in bear)
- "Ride the flow" concept was wrong — volume spike + S/R break is more reliable


### Executor Bugs Found & Fixed (2026-07-23)

**Found 9 critical bugs in scanner_executor.py during code review:**

1. **Double Orchestrator call** — `orchestrator.evaluate_signal()` called twice per signal, second result overwrote first. Doubled compute, potential disagreements.
2. **Duplicate config dicts** — `REGIME_STRATEGY_GATE`, `REGIME_TPSL_SCALE`, constants defined twice. Second silently overwrote first with different values.
3. **Duplicate cooldown tracking** — `close_position()` had copy-pasted cooldown block twice.
4. **Duplicate regime re-classification** — regime classified twice per loop iteration.
5. **`klines` undefined** — referenced variable that was never loaded. Price history always `[]`.
6. **Consensus gate used `now` not signal timestamp** — EMA/slope lookups mismatched for delayed signals.
7. **`direction` variable stale after FLIP** — fallback TP/SL used old direction after consensus gate flipped.
8. **Duplicate `Orchestrator` import** — imported from two different paths.
9. **No atomic state writes** — `save_state()` could corrupt on crash mid-write.

**Also found:** `/root/jimi3/` was a redundant copy of workspace `jimi_audit/`. 200+ identical files, 6 diverged. Consolidated to workspace only, removed jimi3.

**File reduced:** 1927 → 1832 lines (95 lines dead/duplicate code removed).

**Dry-run after fixes:** Clean startup, regime BEAR correctly blocking incompatible strategies.


### Regime Gate Analysis (2026-07-23)

**Problem:** After new regime classifier, most strategies not reviewed for gate compatibility.

**Effective strategies per regime:**
- BULL: 5 active (7 blocked — SHORT-only blocked by cond gate)
- BEAR: 9 active (3 blocked — failed_breakout, positioning_fade, structural_break)
- RANGING: 11 active (1 blocked — forced_movement)
- STRESS: 6 active (6 blocked — liquidation_cascade, funding_squeeze blocked!)
- MILDLY_BEARISH: 9 active (3 blocked)

**Issues identified:**
- `positioning_fade` effectively dead — gate allows CHOP_MILD/CHOP_BEAR/NEUTRAL/CRISIS but classifier never produces these
- `liquidation_cascade` + `funding_squeeze` blocked in STRESS — these are crisis strategies, should be active
- 3 conflicts: SHORT-only strategies allowed in BULL by regime gate but blocked by cond gate

**Pending fixes:**
- positioning_fade: add BEAR, MILDLY_BEARISH, STRESS
- liquidation_cascade: add STRESS
- funding_squeeze: add STRESS
- structural_break: add BEAR

**Capital:** $192.37 | 22 trades (10W/10L) | 3 open positions


### Gate Fix & Backtest (2026-07-23)

**Changes applied to REGIME_STRATEGY_GATE:**
- positioning_fade: replaced dead regimes (CHOP_MILD/CHOP_BEAR/NEUTRAL/CRISIS) → RANGING/BEAR/MILDLY_BEARISH/STRESS
- liquidation_cascade: +STRESS
- funding_squeeze: +STRESS
- trade_flow: +MILDLY_BEARISH
- whale_watch: +BULL
- structural_break: +BULL/BEAR/STRESS (now ALL)
- liquidity_grab: +BULL/BEAR/STRESS (now ALL)
- failed_breakout: +BEAR/MILDLY_BEARISH
- forced_movement: +RANGING

**Backtest results (395 trades, Jan-Jul 2026):**
- Without direction gate: 395 trades, 40.8% WR, +16.06% PnL
- With direction gate: 338 trades, 44.4% WR, +31.25% PnL
- Direction gate removes 57 trades (14.4%), nearly doubles PnL
- LONG+BEAR: 40 trades, 17.5% WR, -13.13% PnL (catastrophic)
- SHORT+BULL: 17 trades, 23.5% WR, -2.06% PnL

**By regime:** MILDLY_BEARISH best (51.4% WR), RANGING most active (249 trades), STRESS = 0 trades (classifier never produced it)

**Research backing:** Bieganowski 2026 (OBI as #1 SHAP feature), Nguyen 2026 (regime-conditional trend-following), SSRN Funding Rate Mechanism

**Key lesson:** Direction gating is both the safest AND most robust approach. Adaptive params per regime are NOT robust unless classifier is 90%+ accurate. Fixed mechanism + direction gate = best risk-adjusted approach.

**Executor:** PID 411565, dry-run, $192.37 capital, 22 trades (10W/10L), 3 open positions

## Regime Classifier V5 (deployed 2026-07-24)
- V3 classifier was fundamentally broken: used 15m data with 5-hour window for macro regime detection
- V3 was stuck on BEAR when V4 daily data showed BULL — COND_GATE was blocking all LONG trades
- V5 uses V4 daily timeframe (Binance 1D + 1W candles) + contradiction detection + hysteresis
- Hysteresis: 3 consecutive daily signals + 5-day cooldown before regime transition
- Regime-conditional sizing replaces binary COND_GATE (BULL 1.0x, BEAR 0.7x, RANGING 1.0x, STRESS 0.5x, MB 0.85x)
- Key research: Hamilton (1989), Ang & Timmermann (2012), Shu et al. (2024 Princeton), Bieganowski (2026)
- Files: regime_classifier_v5.py, scanner_executor.py (backup: .bak_v5_pre_regime)
