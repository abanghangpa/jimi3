# Research Summary: Order Book Imbalance & Liquidity Sweep

*Prepared 2026-07-22 for JIMI strategy rework*

---

## 1. Order Book Imbalance (OBI)

### Key Papers

**Nittur Anantha et al. (2025) — "Order Book Filtration and Directional Signal Extraction at High Frequency"**
📄 arXiv:2507.22712

**Core findings:**
- Raw OBI is **contaminated by flickering liquidity** — orders placed and canceled within milliseconds
- **Structural filtration** (by order lifetime, update count, inter-update delay) significantly improves directional signal clarity
- **Trade-based OBI** (computed from actual trades, not just quotes) shows **stronger causal alignment** with future price movements than quote-based OBI
- Key distinction: associative correlation ≠ causal prediction. Filtration improves correlation but trade-based OBI improves causation

**What JIMI should steal:**
- **Trade-based OBI**: Instead of just measuring bid/ask volume imbalance (quotes), measure the imbalance of **actual executed trades**. This is the "OBI by trades" concept from the paper.
- **Fleet filter**: Ignore OB snapshots that persist < 3 seconds (flickering quotes). JIMI already has persistence filter but should extend to sub-minute levels.

---

**Bieganowski & Ślepaczuk (2026) — "Explainable Patterns in Cryptocurrency Microstructure"**
📄 arXiv:2602.00776

**Core findings (Binance Futures, 5 cryptos, 2022-2025, 1-second data):**
1. **Order flow imbalance** has a **monotone but concave** effect on returns — strong signal at moderate imbalance, diminishing at extremes
2. **VWAP-to-mid deviations** show **asymmetric effects** — price above VWAP = selling pressure building (mean reversion), price below VWAP = buying pressure
3. **Spreads** are associated with **diminished predictability** — wide spreads = noise, not signal
4. **Taker execution** (market orders) outperformed **maker execution** (limit orders) in the backtest
5. These patterns are **universal across crypto assets** (BTC, LTC, ETC, ENJ, ROSE)

**What JIMI should steal:**
- **Concave conviction function**: Don't scale conviction linearly with OBI. Use `sqrt()` or log — moderate imbalance = strong signal, extreme imbalance = possible spoof/manipulation
- **VWAP deviation as primary feature**: Not just a bonus — it's one of the top 3 features in the SHAP analysis
- **Spread filter**: Wide spread = skip the signal (adverse selection risk)
- **Taker-side execution**: Market orders, not limit orders, when following OBI signals

---

**Cont et al. (2014) — "The Price Impact of Order Book Events"**

**Core findings:**
- Linear relationship between order flow imbalance and price changes
- Slope inversely proportional to market depth (deeper book = less price impact per unit imbalance)
- **Price impact is asymmetric**: buy imbalance has larger impact than sell imbalance (in equity markets)

---

## 2. Liquidity Sweep / Stop Hunt

### Academic Gap

There is **no rigorous academic paper** specifically on "liquidity sweep" or "stop hunt" detection in crypto futures. This is a practitioner concept from ICT/Smart Money literature, not academic finance.

**What we know from microstructure theory:**

**Kyle (1985) — "Informed Trading and Market Making"**
- Informed traders **strategically accumulate** before revealing their position
- They push price through key levels to **trigger stop losses** (liquidity), then reverse
- The "sweep" is the informed trader's execution strategy

**Glosten & Milgrom (1985) — "Bid-Ask Spread as Adverse Selection"**
- Market makers widen spreads when they suspect informed flow
- After a sweep, spreads typically widen briefly, then narrow as the reversal begins

### Practitioner Knowledge (Empirical)

From TradingView/community indicators and futures trading forums:

**Liquidity Sweep Mechanism:**
1. Price approaches a key level (swing high/low, S/R)
2. Price **exceeds the level by a small amount** (0.1-0.5%) — this triggers clustered stop losses
3. **Volume spikes** as stops are hit (liquidity grab)
4. **Wick rejection** — price closes back inside the level
5. **Order flow reversal** — taker flow flips direction
6. **Reversal trade** — enter in the opposite direction of the sweep

**Key detection features:**
- **Wick ratio**: (high - close) / (high - low) for bearish sweep, (close - low) / (high - low) for bullish sweep. High wick ratio = rejection
- **Volume absorption**: High volume but price didn't move through (absorption)
- **OI change**: OI spike during sweep = new positions opened (stops being hit), OI drop = liquidations
- **Taker flow shift**: Taker ratio flips during/after the sweep

---

## 3. Recommended Strategy Reworks

### orderbook_imbalance v5 (research-backed)

**Changes from v4:**
1. **Trade-based OBI**: Use actual executed trade imbalance, not just bid/ask quote imbalance
2. **Concave conviction**: `conviction = base + sqrt(obi_strength) * scale` — not linear
3. **VWAP deviation**: Primary signal, not just a bonus
4. **Spread filter**: Skip when spread > 2x average (adverse selection)
5. **Fleet filter**: Already have persistence filter, extend to ignore < 3s snapshots
6. **Asymmetric entry**: Buy imbalance at VWAP discount, sell imbalance at VWAP premium

### liquidity_grab v7 (sweep detection)

**Complete rework — new mechanism:**
1. **Sweep detection**: Price exceeded S/R level by 0.1-0.5%, then closed back inside
2. **Wick rejection**: Wick ratio > 0.6 (strong rejection)
3. **Volume absorption**: Volume spike > 1.5x average during the sweep
4. **Taker flow reversal**: Taker ratio flips direction after sweep
5. **Enter on reversal**: After confirmation, enter counter-sweep direction
6. **Tight SL**: Beyond the sweep high/low (0.5-1.0 ATR)
7. **TP**: Next S/R level in the reversal direction

---

## References

1. Nittur Anantha, A., Jain, S., & Maiti, P. (2025). "Order Book Filtration and Directional Signal Extraction at High Frequency." *arXiv:2507.22712*
2. Bieganowski, B. & Ślepaczuk, R. (2026). "Explainable Patterns in Cryptocurrency Microstructure." *arXiv:2602.00776*
3. Cont, R., Kukanov, A., & Stoikov, S. (2014). "The Price Impact of Order Book Events." *Journal of Financial Markets, 17, 53-81*
4. Kyle, A.S. (1985). "Continuous Auctions and Insider Trading." *Econometrica, 53(6), 1315-1335*
5. Glosten, L.R. & Milgrom, P.R. (1985). "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders." *Journal of Financial Economics, 14(1), 71-100*
6. Kolm, P.N. & Ritter, G. (2023). "Deep Order Flow Imbalance." *arXiv:2307.13716*
7. Stoikov, S. (2018). "The Micro-Price: A High Frequency Estimator of Future Prices." *SSRN*
