# JIMI Optimization Framework — Architecture Draft

*Version: 0.1 | Date: 2026-07-24*

---

## Problem Statement

Current optimization is ad-hoc: manual param sweeps, no overfitting prevention, no systematic strategy selection. The S20 v6 revalidation exposed the core issue — 1109 signals filtered to 52 requires statistical rigor to claim edge. Without it, we're curve-fitting.

**Goals:**
1. Automate parameter optimization with overfitting guards
2. Dynamically select which strategies to enable per regime
3. Adapt ensemble weights based on recent performance
4. Validate every claim with walk-forward and deflated Sharpe
5. Size positions using Kelly, not gut feel

---

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │         ORCHESTRATOR (existing)       │
                    │   Signal routing, trade execution     │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────────────┐
                    │        OPTIMIZATION FRAMEWORK        │
                    │                                      │
    ┌───────────────┼───────────────┬──────────────────┐  │
    │               │               │                  │  │
┌───┴───┐    ┌──────┴──────┐  ┌────┴────┐    ┌────────┴┐ │
│VALIDA-│    │   SEARCH    │  │ SELECTOR│    │ ENSEMBLE │ │
│ TOR   │    │   AGENT     │  │  AGENT  │    │  AGENT   │ │
│AGENT  │    │             │  │         │    │          │ │
└───┬───┘    └──────┬──────┘  └────┬────┘    └────────┬┘ │
    │               │              │                   │  │
    │        ┌──────┴──────┐  ┌────┴────┐              │  │
    │        │   REGIME    │  │  RISK   │              │  │
    │        │   AGENT     │  │  AGENT  │              │  │
    │        └─────────────┘  └─────────┘              │  │
    │                                                   │  │
    └───────────────────────────────────────────────────┘  │
                    │                                      │
                    └──────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────────────┐
                    │     LIVE EXECUTION (existing)        │
                    │   scanner_executor.py, engine.py     │
                    └─────────────────────────────────────┘
```

---

## Agent Specifications

### 1. Validator Agent (Priority: CRITICAL)

**Purpose:** Prevent overfitting. Every strategy/param change must pass statistical validation before deployment.

**Methods:**
- **Walk-Forward Analysis (WFA):** Rolling train/test windows. Train on N bars, test on M bars, slide forward.
- **Deflated Sharpe Ratio (DSR):** Bailey & Lopez de Prado (2014). Adjusts Sharpe for multiple testing.
- **Monte Carlo Permutation:** Shuffle trade labels, recompute metrics. If real result < 95th percentile of random, reject.
- **Combinatorial Purged Cross-Validation (CPCV):** De Prado (2018). K-fold CV with purge gap to prevent leakage.

**Inputs:**
- Strategy signals (timestamp, direction, entry, SL, TP, outcome)
- OHLCV bars
- Number of trials tested (for DSR)

**Outputs:**
- `validation_report.json`: WR, PF, Sharpe, MaxDD, DSR, CPCV score, pass/fail
- Gate decision: PASS / FAIL / CONDITIONAL

**Thresholds:**
| Metric | Minimum | Target |
|---|---|---|
| DSR | > 1.0 | > 2.0 |
| Walk-Forward WR | > 50% | > 60% |
| CPCV Score | > 0.55 | > 0.65 |
| Monte Carlo p-value | < 0.05 | < 0.01 |
| Sample Size | > 30 trades | > 100 trades |

**Walk-Forward Config:**
```python
WFA_CONFIG = {
    "train_bars": 4000,    # ~42 days of 15m
    "test_bars": 960,      # ~10 days of 15m
    "step_bars": 960,      # slide by test size
    "min_test_trades": 5,  # minimum trades per test window
    "purge_bars": 16,      # 4h purge gap between train/test
}
```

**Deflated Sharpe Ratio:**
```python
def deflated_sharpe(sharpe, n_trials, n_trades, skew, kurtosis):
    """
    Bailey & Lopez de Prado (2014)
    Adjusts observed Sharpe for multiple testing bias.
    """
    e_max_sharpe = expected_max_sharpe(n_trials, n_trades, skew, kurtosis)
    dsr = (sharpe - e_max_sharpe) / std_sharpe(n_trades, skew, kurtosis)
    return dsr  # > 1.96 = significant at 95%
```

---

### 2. Search Agent (Priority: HIGH)

**Purpose:** Find optimal parameters for each strategy. Replace grid search with Bayesian Optimization.

**Method:** Bayesian Optimization with Gaussian Process (GP) surrogate.
- Snoek et al. (2012), "Practical Bayesian Optimization"
- Library: `scikit-optimize` or `optuna`

**Parameters to optimize per strategy:**
| Parameter | Range | Type |
|---|---|---|
| TP multiplier | 0.5 – 5.0 | Continuous |
| SL multiplier | 0.5 – 5.0 | Continuous |
| Conviction threshold | 0.3 – 0.9 | Continuous |
| Cooldown (bars) | 1 – 48 | Integer |
| Volume threshold | 0.05 – 0.50 | Continuous |
| OI ROC threshold | 0.005 – 0.030 | Continuous |
| LS ratio threshold | 0.3 – 2.0 | Continuous |

**Objective function:** Maximize risk-adjusted return:
```python
def objective(params):
    # Run walk-forward backtest with these params
    results = walk_forward_backtest(strategy, params, ohlcv, signals)
    # Penalize overfitting
    dsr = deflated_sharpe(results.sharpe, n_trials, results.n_trades, results.skew, results.kurtosis)
    # Penalize low sample size
    sample_penalty = min(results.n_trades / 100, 1.0)
    return dsr * sample_penalty * results.expectancy
```

**Budget:** 50-200 trials per strategy (depending on parameter space dimensionality).

**Output:**
- `search_results.json`: best params, confidence interval, convergence plot

---

### 3. Selector Agent (Priority: HIGH)

**Purpose:** Dynamically select which strategies to enable. Strategies that degrade get disabled automatically.

**Method:** Thompson Sampling (Bayesian Multi-Armed Bandit)
- Each strategy = one arm
- Reward = trade PnL (normalized)
- Beta distribution for binary outcomes (win/loss)
- Gamma-Poisson for continuous rewards (PnL magnitude)

**Why Thompson Sampling over UCB:**
- Naturally balances exploration/exploitation
- Handles non-stationary rewards (strategies degrade)
- Computationally cheap (sample from posterior)

**Implementation:**
```python
class StrategySelector:
    def __init__(self, strategies):
        # Beta prior for each strategy: alpha=wins, beta=losses
        self.posteriors = {s: Beta(alpha=2, beta=2) for s in strategies}
    
    def select(self, regime, n_select=5):
        """Select top-N strategies for current regime."""
        samples = {}
        for s, dist in self.posteriors.items():
            if s in REGIME_STRATEGY_GATE[regime]['allowed']:
                samples[s] = dist.sample()
        return sorted(samples, key=samples.get, reverse=True)[:n_select]
    
    def update(self, strategy, pnl):
        """Update posterior after trade."""
        if pnl > 0:
            self.posteriors[strategy].alpha += 1
        else:
            self.posteriors[strategy].beta += 1
```

**Regime-aware selection:**
- Separate posteriors per regime (BULL, BEAR, RANGING, STRESS, MILDLY_BEARISH)
- Prevents a strategy good in BULL from being selected in BEAR

**Decay factor:**
- Recent trades weighted higher (exponential decay, half-life = 50 trades)
- Prevents stale performance from dominating

---

### 4. Regime Agent (Priority: MEDIUM)

**Purpose:** Adaptive regime detection. Current RegimeClassifierV4 uses fixed thresholds. This agent learns optimal thresholds.

**Method:** Hidden Markov Model (HMM) + online learning
- Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series"
- Observable: returns, volatility, OI change, funding rate, taker ratio
- Hidden states: BULL, BEAR, RANGING, STRESS, MILDLY_BEARISH

**Online learning:**
- Update transition matrix every 24h with new data
- Retrain emission distributions weekly
- Alert on regime transition probability > 0.7

**Inputs:**
- 15m OHLCV returns
- Derivatives: OI, funding rate, L/S ratio, taker flow
- Macro: DXY, VIX, 10Y yield (daily)

**Output:**
- `regime_state.json`: current regime, confidence, transition probabilities

**Improvement over current:**
- Current: fixed thresholds (oi_roc > 0.015 = LONG cascade)
- Proposed: learned thresholds that adapt to market microstructure changes

---

### 5. Ensemble Agent (Priority: MEDIUM)

**Purpose:** Optimize how strategies combine. Current ensemble uses fixed weights.

**Method:** Stacking (Wolpert, 1992) + dynamic weight adjustment

**Layer 1 — Strategy signals:**
- Each strategy produces: direction, conviction, entry/SL/TP

**Layer 2 — Meta-learner:**
- Input: strategy signals + regime + market features
- Output: final direction, adjusted conviction, position size
- Model: Logistic Regression (interpretable) or LightGBM (if enough data)

**Dynamic weight adjustment:**
```python
def update_weights(strategy, regime, outcome):
    """Exponential moving average of strategy performance."""
    alpha = 0.1  # learning rate
    weight = WEIGHTS[regime][strategy]
    reward = 1 if outcome == 'WIN' else -1
    WEIGHTS[regime][strategy] = weight * (1 - alpha) + reward * alpha
    # Normalize
    total = sum(WEIGHTS[regime].values())
    WEIGHTS[regime] = {k: v/total for k, v in WEIGHTS[regime].items()}
```

**Co-occurrence matrix:**
- Track which strategies fire together
- Identify complementary pairs (S20 + whale_watch = 59.5% WR)
- Penalize redundant pairs (strategies that always agree = no added value)

---

### 6. Risk Agent (Priority: MEDIUM)

**Purpose:** Optimal position sizing. Current: fixed RISK_PCT (2%).

**Method:** Fractional Kelly Criterion
- Kelly (1956), "A New Interpretation of Information Rate"
- MacLean et al. (2011), "The Kelly Capital Growth Investment Criterion"

**Formula:**
```python
def kelly_fraction(win_rate, avg_win, avg_loss):
    """Full Kelly fraction."""
    if avg_loss == 0:
        return 0
    b = avg_win / abs(avg_loss)  # win/loss ratio
    f = (win_rate * b - (1 - win_rate)) / b
    return max(f, 0)  # never go negative

def position_size(capital, kelly_f, fraction=0.25):
    """Fractional Kelly (0.25x = conservative)."""
    return capital * kelly_f * fraction
```

**Regime-adjusted Kelly:**
| Regime | Kelly Fraction |
|---|---|
| BULL | 0.35x |
| RANGING | 0.25x |
| BEAR | 0.15x |
| STRESS | 0.10x |
| MILDLY_BEARISH | 0.20x |

**Drawdown circuit breaker:**
- DD > 15%: reduce Kelly to 0.5x
- DD > 25%: reduce Kelly to 0.25x
- DD > 35%: pause trading

---

## Data Flow

```
1. OHLCV + Derivatives → Regime Agent → regime_state.json
2. regime_state → Selector Agent → enabled_strategies[]
3. enabled_strategies → Scanner → strategy_signals
4. strategy_signals + regime_state → Ensemble Agent → final_signal
5. final_signal → Risk Agent → position_size
6. final_signal + position_size → Orchestrator → trade execution
7. trade outcomes → Validator Agent → validation_report
8. validation_report → Search Agent → param updates
9. trade outcomes → Selector Agent → posterior updates
10. trade outcomes → Ensemble Agent → weight updates
```

---

## Dependency Management Protocol

*Added: 2026-07-26*

### Problem

Strategies use other strategies/modules as boosters (confirmation signals). Modifying a booster can silently break dependents. Example: changing whale_watch breaks S19 orderbook_imbalance, S21 trade_flow, and S04 positioning_fade.

### Dependency Map

Stored in `config/strategy_dependencies.json` (machine-readable) and `reports/strategy_dependencies.md` (human-readable).

**Risk levels:**
- **CRITICAL:** Regime classifier (ALL strategies), derivatives data (8 strategies)
- **HIGH:** whale_watch (3 strategies), M14 (2 strategies), M21 (2 strategies), M9 (3 strategies)
- **MEDIUM:** cross_asset, taker_flow, M5, taker_summary
- **LOW:** funding_arb, OBI, momentum_v3

### Modification Protocol

#### Step 0: Pre-Flight Check
```bash
# Run BEFORE any modification
python3 scripts/dependency_gate_check.py <component>

# Examples:
python3 scripts/dependency_gate_check.py whale_watch
python3 scripts/dependency_gate_check.py m14_sweep
python3 scripts/dependency_gate_check.py regime_classifier
```

This shows all strategies that depend on the component, with their current baseline metrics.

#### Step 1: Snapshot Baseline
```bash
# Run dependent gates to capture before-state
python3 scripts/dependency_gate_check.py <component> --run-gates
```

Save the output — this is your regression baseline.

#### Step 2: Make the Modification
- Modify the booster strategy/module
- Commit with descriptive message
- Include `[dep: <affected strategies>]` in commit message

#### Step 3: Post-Flight Check
```bash
# Re-run dependent gates
python3 scripts/dependency_gate_check.py <component> --run-gates
```

#### Step 4: Evaluate

| Result | Meaning | Action |
|---|---|---|
| ✅ PASS | No regression | Deploy |
| ⚠️ WARN | Metrics degraded >5% | Investigate, may need to adjust dependent |
| ❌ FAIL | Edge lost (MC sig gone, WR <50%) | Revert booster, investigate |

#### Step 5: Atomic Commit
If all pass, commit booster + dependent updates together:
```bash
git add -A
git commit -m "modify <booster>: <what changed> [dep: s19, s21, s04]"
```

### Emergency Revert

If a booster modification breaks a dependent:
1. Revert booster to `.bak` file
2. Re-run dependent gate — confirm fix
3. Investigate root cause
4. Redesign dependency to be more robust

### Integration with 8-Agent Forensic

When running forensic on a strategy that is used as a booster:
- **Agent 0 (new):** Dependency check — list all dependents, run their gates
- **Agent 8 (Monte Carlo):** After forensic, verify dependents still pass

### Integration with Optimization Framework

- **Validator Agent:** Before deploying optimized params, run dependency check
- **Search Agent:** Include dependent strategy performance in objective function
- **Selector Agent:** Don't disable a strategy that others depend on without checking

---

## Implementation Roadmap

### Phase 0: Dependency Safety (DONE)
- [x] Create dependency map (`config/strategy_dependencies.json`)
- [x] Create dependency checker (`scripts/dependency_gate_check.py`)
- [x] Embed protocol in optimization framework
- [ ] Add Agent 0 (dependency check) to 8-agent forensic template

### Phase 1: Validator Agent (Week 1-2)
- [ ] Implement walk-forward engine
- [ ] Implement deflated Sharpe ratio
- [ ] Implement Monte Carlo permutation test
- [ ] Gate all strategies through validator
- [ ] **Deliverable:** Every strategy claim backed by DSR > 1.0

### Phase 2: Search Agent (Week 3-4)
- [ ] Integrate Optuna for Bayesian optimization
- [ ] Define param spaces for top 5 strategies
- [ ] Run optimization with walk-forward objective
- [ ] **Deliverable:** Optimized params for S20, trade_flow, orderbook_imbalance, whale_watch, funding_arb

### Phase 3: Selector Agent (Week 5-6)
- [ ] Implement Thompson Sampling with regime-aware posteriors
- [ ] Integrate with executor (auto-enable/disable strategies)
- [ ] Backtest selector vs static strategy set
- [ ] **Deliverable:** Dynamic strategy selection, ~20% fewer losing trades

### Phase 4: Risk + Ensemble Agents (Week 7-8)
- [ ] Implement fractional Kelly
- [ ] Implement dynamic ensemble weights
- [ ] Co-occurrence matrix for strategy pairs
- [ ] **Deliverable:** Optimal position sizing, complementary strategy pairs identified

### Phase 5: Regime Agent (Week 9-10)
- [ ] Implement HMM-based regime detection
- [ ] Online learning for threshold adaptation
- [ ] Compare vs RegimeClassifierV4
- [ ] **Deliverable:** Adaptive regime detection with transition alerts

---

## Statistical Rigor Requirements

Every strategy claim must include:
1. **Sample size** (n trades)
2. **Walk-forward WR** (not in-sample)
3. **Deflated Sharpe** (adjusted for multiple testing)
4. **Bootstrap 95% CI** on WR and PF
5. **Regime breakdown** (not pooled)

**Minimum thresholds for deployment:**
- n ≥ 30 trades (minimum for statistical power)
- DSR > 1.0 (95% confidence after multiple testing correction)
- Walk-forward WR > 50%
- No regime with WR < 35%

---

## Key Research References

1. Bailey, D. & Lopez de Prado, M. (2014). "The Probability of Backtest Overfitting." *Journal of Computational Finance.*
2. De Prado, M. (2018). "Advances in Financial Machine Learning." *Wiley.* Chapters 12-14 (walk-forward, CV, feature importance).
3. Harvey, C. & Liu, Y. (2015). "Backtesting." *Journal of Portfolio Management.*
4. Snoek, J., Larochelle, H., & Adams, R. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms." *NeurIPS.*
5. Bubeck, S. & Cesa-Bianchi, N. (2012). "Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems." *Foundations and Trends in ML.*
6. Thompson, W. (1933). "On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples." *Biometrika.*
7. Kelly, J. (1956). "A New Interpretation of Information Rate." *Bell System Technical Journal.*
8. Hamilton, J. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series." *Econometrica.*
9. Wolpert, D. (1992). "Stacked Generalization." *Neural Networks.*
10. Ang, A. & Timmermann, A. (2012). "Regime Changes and Financial Markets." *Annual Review of Financial Economics.*

---

## Design Decisions (2026-07-24)

| Question | Decision | Rationale |
|---|---|---|
| Q1: Data sufficiency | **Hybrid** — portfolio-level walk-forward for gate decisions, trade-level Monte Carlo for CIs | 52 signals too few for per-trade walk-forward. Portfolio aggregation provides statistical power. |
| Q2: Regime stability | **Weekly retrain + event-driven override** | Weekly baseline. HMM triggers immediate update on regime transition (prob > 0.7). |
| Q3: Ensemble complexity | **Weighted vote now → Logistic regression at 200+ trades → LightGBM at 500+** | Start simple, upgrade as data accumulates. |
| Q4: Execution latency | **Manual** — optimize, review, deploy | Safety over speed. No auto-deploy. |
| Q5: Selector cold start | **Seed with backtest priors + aggressive updates (alpha=0.3)** | Backtest seeds give head start. High learning rate lets live results override in 10-15 trades. |

---

*Next step: Start Phase 1 (Validator Agent) with hybrid walk-forward + Monte Carlo approach.*
