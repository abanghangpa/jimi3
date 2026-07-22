# Backtesting Framework — Updated 2026-07-23

## ADDENDUM: Lessons from Conditional Directional Model Development

*Added after the OB v5 + regime × direction analysis session.*

---

### NEW RULE: Event-Driven Backtest Required

Random signal simulation is **meaningless**. We proved this:

| Approach | Result | Valid? |
|----------|--------|--------|
| Random signals | -99% return | No |
| Event-driven (replay real OHLCV) | +168% return | Yes |

**Requirement:** Every backtest must replay actual price bars through the same feature computation and strategy logic the executor uses. No random entries. No synthetic fills.

If you can't replay through real logic, the backtest is a simulation, not a backtest. Label it accordingly.

---

### NEW RULE: Regime × Direction Matrix Is Mandatory

Before any conclusion about direction or regime, compute this matrix:

```
Regime          LONG PF    SHORT PF    LONG n    SHORT n
MILDLY_BEARISH  ...        ...         ...       ...
RANGING         ...        ...         ...       ...
BULL            ...        ...         ...       ...
BEAR            ...        ...         ...       ...
```

**Why:** Aggregates hide structure. We found:
- LONG + BEAR: PF 0.31 (portfolio killer)
- SHORT + BULL: PF 0.29 (portfolio killer)
- SHORT + BEAR: PF 3.31 (true edge)
- LONG + BULL: PF 3.51 (true edge)

Aggregated SHORT PF was 1.49 and LONG PF was 0.89 — suggesting "disable LONG." But the matrix revealed LONG works fine in BULL. The problem was direction × regime disagreement, not direction itself.

**Don't say "direction filter" until you've shown the matrix.**

---

### NEW RULE: Min Hold Duration Test

Always test minimum hold filters:

```
Duration        PF
1-5 bars        0.90  ← noise
6-15 bars       1.48
16-30 bars      1.15
31-48 bars      1.57
```

Quick trades (1-5 bars) are microstructure noise. Entry signal may be correct but needs time to develop.

**Test:** Min 6 bars, min 12 bars, min 24 bars. If min 6 dramatically improves PF, the entry timing is fine but exits are too aggressive.

**Implementation:** No TP before bar 6. Trailing stop activates after bar 6.

---

### NEW RULE: Execution Cost Threshold

Every backtest must compute the **breakeven total cost**:

```
Breakeven = Fee + Spread + Slippage
ETH perps on Binance: Fee ~0.04% + Spread ~0.02% = 0.06% base
Slippage budget: PF-dependent
```

**Stress test:** Apply 1.5x, 2x, 3x slippage to all exits. If PF drops below 1.0 at 1.5x slippage, the strategy is **execution-sensitive**.

Our finding: PF 1.72 → 0.764 at 1.5x slippage. This means the edge is execution-limited, not prediction-limited.

**Report:**
```
Cost threshold: 0.15% per trade
Actual cost (estimated): 0.06-0.09%
Buffer: 0.06-0.09%
Verdict: MARGINAL — execution quality is critical
```

---

### NEW RULE: Walk-Forward Is Primary Validation

Hold-out validation (70/30 split) is the minimum. Walk-forward is the standard.

**Walk-forward protocol:**
1. Train on first N months
2. Test on remaining months
3. Report Train PF → Test PF
4. Compute PF decay: (Test - Train) / Train

**Interpretation:**
| PF Decay | Verdict |
|----------|---------|
| > 0% | Excellent (OOS better than IS) |
| 0% to -20% | Good (expected slight degradation) |
| -20% to -50% | Warning (possible overfitting) |
| < -50% | Likely overfit |

Our finding: Train PF 1.60 → Test PF 2.35 (+47%). This is unusual and very positive. But always consider: small test sample, favorable market conditions, or genuine robustness.

---

### NEW RULE: Stress Tests Required

Before any production claim, run these 7 stress tests:

| Test | What | Threshold |
|------|------|-----------|
| 1. Remove top 10 winners | Dependency on outliers | PF > 1.3 |
| 2. Slippage 1.5x | Execution sensitivity | PF > 1.0 |
| 3. Fee +0.10% | Cost sensitivity | PF > 1.2 |
| 4. Skip 15% trades | Missed fills | P(Exp>0) > 80% |
| 5. ±2 bar entry shift | Timing fragility | PF > 1.0 |
| 6. Combined worst case | All stress combined | PF > 0.8 |
| 7. Regime-specific | Per-regime breakdown | All regimes positive |

**If slippage 1.5x breaks the strategy:** Focus on execution quality, not signal optimization.

---

### NEW RULE: Bootstrap Confidence Intervals

Every claim must include 95% CI from bootstrap (5,000 resamples):

```
Expectancy: +0.16% [CI: +0.05%, +0.27%]
PF: 1.72 [CI: 1.20, 2.46]
```

**If CI crosses zero:** The edge is not statistically significant regardless of point estimate.

---

### NEW RULE: Monte Carlo Required

Run Monte Carlo (10,000 sims) at 30-day and 90-day horizons:

```
30-day:  P50=+6.8%  P(loss)=6.7%  MaxDD P50=2.5%
90-day:  P50=+22.4% P(loss)=0.3%  MaxDD P95=6.7%
```

**Interpret P(loss) conservatively.** Monte Carlo assumes future ≈ historical. It does not capture:
- Regime shifts
- Exchange changes
- Volatility shocks
- Liquidity changes

Report as: "Historical P(loss) under simulation assumptions" not "Probability of future loss."

---

### NEW: Research-Backed Feature Engineering

From the literature review (see REGIME_RESEARCH.md, LOB_RESEARCH.md):

**Regime detection:**
- Use **daily** timeframe, not 15m (Shu et al. 2024, Princeton)
- Statistical Jump Model with jump penalty for persistence
- Ensemble voting across multiple signal categories

**Order book imbalance:**
- **Trade-based OBI** > quote-based OBI (Nittur Anantha 2025)
- Concave conviction: `sqrt()` scaling, not linear (Bieganowski 2026)
- VWAP deviation is a top-3 feature (SHAP analysis, Binance Futures)
- Wide spread = skip signal (adverse selection)

---

### Updated Minimum Requirements

| Requirement | Old | New |
|------------|-----|-----|
| Isolation gate events | 500+ | 500+ |
| Post-optimization trades | 20+ | 50+ |
| Validation | Hold-out 70/30 | Walk-forward |
| Statistical test | t-test | Bootstrap CI + Monte Carlo |
| Stress tests | None | 7 tests required |
| Execution cost | Not checked | 0.15% breakeven computed |
| Regime × Direction | Optional | Mandatory |
| Min hold filter | Not tested | Always test 6/12/24 bars |

---

## Original Framework (Below)

_The original framework content remains valid. This addendum supersedes where conflicts exist._

