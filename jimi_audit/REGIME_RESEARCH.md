# Regime Detection Research Summary

*Prepared 2026-07-22 for JIMI Framework improvement*

---

## Current JIMI RegimeClassifier — Problems Identified

The current implementation (lines 512–720 in `scanner_executor.py`) has these structural issues:

1. **Hardcoded thresholds everywhere** — FR > 0.000030 = BULL, LS > 2.2 = LONG_CROWDED, etc. These were manually tuned and keep getting adjusted (comments show multiple rounds of "LOWERED thresholds"). No principled basis.

2. **No regime persistence** — Classifier can flip every 60s poll cycle. A single scan showing bearish signals can switch from BULL to BEAR instantly. Real regimes are persistent by definition.

3. **Flat additive scoring** — bull_score, bear_score, stress_score are summed with arbitrary weights (0.2, 0.3, 0.5, 0.8, 1.0). No normalization, no decay, no confidence calibration.

4. **No temporal memory** — Uses only the last 20 data points (WINDOW=20). Doesn't track how long the current regime has lasted or how stable it is.

5. **Missing microstructure signals** — No order flow imbalance, no VWAP deviation, no volume profile, no bid-ask spread dynamics. These are critical for crypto perps.

6. **No backtested validation** — Thresholds are tuned by hand against recent data. No walk-forward validation, no regime accuracy metric.

---

## Key Research Papers

### 1. Statistical Jump Models (JM) — Princeton, 2024
**"Downside Risk Reduction Using Regime-Switching Signals"**
*Shu, Yu, Mulvey — Princeton ORFE*
📄 https://arxiv.org/html/2402.05272v2

**Key insight:** JMs outperform HMMs for regime detection in trading. They cluster temporal features (returns, volatility) while imposing a **jump penalty** that controls regime persistence. Higher penalty = fewer transitions = more stable regimes.

**What JIMI should steal:**
- **Jump penalty hyperparameter** — controls how often regime can change. Tunable via cross-validation on strategy Sharpe ratio. This directly solves the "no persistence" problem.
- **Feature set** — They use only return-derived features (risk + return measures). Simple but effective. We can add our derivatives/microstructure features on top.
- **Time-series cross-validation** — Select the jump penalty that maximizes strategy performance, not statistical fit.

### 2. Ensemble-HMM Voting — Cornell/IIT, 2025
**"A forest of opinions: Multi-model ensemble-HMM voting framework"**
*Gupta, Kapoor, Gupta, Natesan*
📄 https://doi.org/10.3934/DSFE.2025019

**Key insight:** No single model reliably detects regimes. They combine:
- Bagging (Random Forest)
- Boosting (XGBoost, CatBoost)
- HMM
- **Hybrid voting classifiers** that integrate HMM with ensemble models

**What JIMI should steal:**
- **Ensemble voting** — Instead of one threshold-based classifier, run 3-4 lightweight classifiers and take majority vote. More robust to any single signal being wrong.
- **Feature engineering** — They use macroeconomic + technical indicators together. JIMI already has derivatives data; add vol structure features.

### 3. Volatility-Volume-Gap Classifier — SSRN, 2025
**"A Validated Volatility-Volume-Gap Classifier for Regime Detection"**
📄 https://papers.ssrn.com/sol3/Delivery.cfm/6750442.pdf?abstractid=6750442

**Key insight:** Regime classification should incorporate microstructure features:
- Volume profile (HVN/LVN zones)
- Gap detection (fair value gaps)
- Institutional order flow patterns
- Volatility clustering (GARCH-like)

**What JIMI should steal:**
- **Vol clustering** — High vol tends to persist. Use rolling vol ratio + vol trend as a regime feature, not just a threshold.
- **Volume-price divergence** — Price up + volume down = weak trend = likely regime shift.

### 4. SAE — Survivability-Aware Execution (2026)
**"Execution Is the New Attack Surface"**
*Borjigin et al.*
📄 https://arxiv.org/html/2603.10092v1

**Key insight:** Market-state constraints should be enforced at the execution layer, not just the signal layer. They use:
- Trust-state conditioned budgeting
- Projection-based exposure limits
- Cooldown and order-rate limiting

**What JIMI should steal:**
- **Regime-conditioned execution** — Already partially implemented (REGIME_TPSL_SCALE). But should also affect position sizing and max exposure per regime.

### 5. SSRN — "What Are Market Regimes? Definitional Chaos, Validation Failure"
📄 https://papers.ssrn.com/sol3/Delivery.cfm/6493762.pdf?abstractid=6493762

**Key insight:** Most regime detection research suffers from:
- Inconsistent definitions (what IS a regime?)
- No out-of-sample validation
- Look-ahead bias in feature selection

**Warning for JIMI:** Our current thresholds are tuned on recent data. Without walk-forward validation, we're likely overfitting to Jul 2025 conditions.

---

## Recommended Architecture Improvements

### A. Replace flat scoring with Jump Model approach
```
Features: [FR, LS, OI_ROC, vol_ratio, taker_flow, EMA_slope]
→ Cluster into K regimes (K=4: BULL, BEAR, RANGING, STRESS)
→ Apply jump penalty λ to control persistence
→ Online inference (update each scan cycle)
```

### B. Add regime persistence tracking
```python
# Track regime duration and stability
self.regime_history = []  # [(timestamp, regime, confidence)]
self.regime_duration = 0  # bars in current regime
self.regime_stability = 0.0  # % of recent classifications agreeing

# Only allow regime switch if:
# 1. New regime signal is strong enough (confidence > threshold)
# 2. Signal persists for N consecutive scans (e.g., 3-5)
# 3. Or jump penalty allows it
```

### C. Add microstructure features
- **Order flow imbalance** (buy vs sell volume ratio)
- **VWAP deviation** (price distance from VWAP)
- **Spread dynamics** (bid-ask widening = stress)
- **Large trade detection** (whale activity)

### D. Ensemble voting
```python
class EnsembleRegimeClassifier:
    classifiers = [
        DerivativesClassifier(),   # current FR/LS/OI approach
        VolStructureClassifier(),  # ATR, vol clustering, squeeze
        FlowClassifier(),          # taker flow, CVD, OI flow
        MacroClassifier(),         # calendar events, macro regime
    ]
    
    def classify(self, scan_data):
        votes = [c.classify(scan_data) for c in self.classifiers]
        # Majority vote with confidence weighting
        return weighted_vote(votes)
```

### E. Walk-forward validation
- Split historical data into train/validation windows
- Tune jump penalty and thresholds on train
- Validate on out-of-sample data
- Track regime accuracy: did BULL regime actually produce positive returns?

---

## Priority Implementation Order

1. **Regime persistence** (jump penalty or hysteresis) — biggest bang for buck, fixes the flipping problem
2. **Ensemble voting** — robustness improvement, easy to implement
3. **Microstructure features** — needs new data pipeline (order flow, VWAP)
4. **Walk-forward validation** — needs historical backtest infrastructure
5. **Full Jump Model** — most research-backed but most complex

---

## References

1. Shu, Y., Yu, C., & Mulvey, J. M. (2024). "Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach." *arXiv:2402.05272*
2. Gupta, R., Kapoor, S., Gupta, H., & Natesan, S. (2025). "A forest of opinions: A multi-model ensemble-HMM voting framework for market regime shift detection and trading." *Data Science in Finance and Economics, 5(4), 466-501.*
3. Borjigin, A., et al. (2026). "Execution Is the New Attack Surface: Survivability-Aware Agentic Crypto Trading." *arXiv:2603.10092*
4. Nystrup, P., et al. (2020). "Regime-aware asset allocation." (HMM sensitivity to mis-estimation)
5. Hamilton, J. D. (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica, 57, 357-384.*
6. Ang, A., & Timmermann, A. (2012). "Regime changes and financial markets." *Annual Review of Financial Economics, 4, 313-337.*
