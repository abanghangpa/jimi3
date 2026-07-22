# JIMI Framework — Live Scanner Report
**Timestamp:** 2026-07-13 20:45:00 UTC  
**ETH/USDT Price:** $1,765.00  
**Scan ID:** scan_20260713_205049  

---

## 🎯 EXECUTIVE SUMMARY

| Metric | Value | Status |
|--------|-------|--------|
| **ICS Score** | 0.542 | ⚠️ Moderate |
| **Resolved Direction** | **SHORT** (dir_resolver) | ⚠️ **CONFLICT** |
| **Ensemble Consensus** | LONG (2/2 agree) | ⚠️ **CONFLICT** |
| **Swing Bias (Daily)** | BEARISH | ✅ Aligns SHORT |
| **Phase0 (Macro Health)** | **0.039** | 🔴 **DEATH ZONE** (<0.15) |
| **M9 Vol Regime** | NEUTRAL | ⚪ Neutral |
| **M20 Failed Breakout** | **BLOCKED** (stale, 22 bars) | 🔴 Blocks SHORT |
| **M22 Inflation Regime** | STAGFLATION_LITE (MEDIUM) | ⚠️ Size mult 0.85x |
| **M23 PPI Session** | STAGFLATION_HOT, claims today | ⚠️ Decay 0.7x |

**Bottom Line:** **Conflicted signal — do not trade.** Direction resolver says SHORT (structure + swing bias), but ensemble votes LONG (orderbook + trade flow). M20 stale block prevents SHORT entry. Phase0 in death zone (0.039) makes ALL setups low-probability. Best action: **WAIT** for Phase0 recovery >0.15 or clean M20 resolution.

---

## 📊 CORE MARKET STATE

| Component | Value | Interpretation |
|-----------|-------|----------------|
| **Price** | $1,765.00 | — |
| **EMA 200 (1h)** | $1,780.27 | Price below → bearish LT structure |
| **Swing Bias (1D)** | BEARISH | Lower highs/lows on daily |
| **Trend (1D)** | NEUTRAL (-0.084) | No clear trend |
| **Phase0** | **0.039** | 🔴 **DEATH ZONE** — macro context extremely weak |
| **ATR (15m)** | 5.52 | Normal volatility |
| **RSI (15m)** | 43.8 | Mildly oversold |
| **VWAP Dist** | -1.36% | Price below VWAP (bearish) |
| **Vol Ratio** | 0.19 | Low volume (dry) |

---

## 🧭 DIRECTION RESOLUTION & CONFLICT

### Direction Resolver (Primary Logic)
```
Resolved: SHORT (size_mult: 0.844)
Action: TRADEABLE
Reason: "NEUTRAL + BEARISH structure → SHORT"
Inputs:
  - M9 Regime: NEUTRAL (0.5)
  - M13 Bias: BEARISH (0.9)
  - M7 Score: 0.372 (FAIL)
  - Swing Bias: BEARISH
  - Trend: NEUTRAL
  - Long Target Score: 0.819
  - Short Target Score: 0.968  ← SHORT has better targets
```

### Ensemble (Strategy Consensus) — **CONTRADICTS RESOLVER**
```
Consensus: LONG
Agreeing: 2/2 strategies
  1. orderbook_imbalance (conv=0.75, wt=1.2, weighted=0.90)
  2. trade_flow (conv=0.66, wt=1.2, weighted=0.79)
Weighted Score: 1.69 | Conviction: 0.85 | PASSES
Reason: "Ensemble: 2 strategies agree LONG (weighted=1.69)"
```

### Conflict Resolution Module
```
Conflict Type: REGIME_CONFLICT
Severity: LOW
Summary: "SHORT bias conflicted: 2 signals against vs 4 supporting. LOW severity."

Factors FOR SHORT (4):
  ✅ M13 structure: BEARISH
  ✅ Daily swing bias: BEARISH
  ✅ M3 VWAP PASS — price below VWAP zone
  ✅ M9 regime: NEUTRAL (favorable for shorts)
  ✅ Spot flow: sellers dominant (supports short)

Factors AGAINST SHORT (2):
  ❌ Phase0=0.039 (death zone <0.15) — weak macro context
  ❌ M2 EMA confluence FAIL — multi-TF trend disagreement

Precaution: "Phase0 death zone (0.039) — macro context is weak."
Action Plan: Key level $1,765–$1,765 (support_level). Watch for GENUINE_BREAKDOWN.
```

---

## 📈 MODULE SCORECARD

| Module | Status | Score | Key Detail |
|--------|--------|-------|------------|
| **M1 MACD** | — | 0.552 | Neutral |
| **M2 EMA Confluence** | **FAIL** | 0.15 | Multi-TF trend disagreement |
| **M3 VWAP** | PASS | 0.482 | Price below VWAP |
| **M4 CVD (15m)** | PASS | 0.036 | Weak bullish CVD |
| **M5 Volume Profile** | PASS | 0.949 | Strong HVN structure |
| **M8 Funding** | PASS | 0.500 | Neutral |
| **M9 Vol Regime** | PASS | 0.500 | NEUTRAL |
| **M10 Macro (ETH/BTC)** | PASS | 0.631 | Moderate |
| **M11** | FAIL | 0.150 | — |
| **M12 Orderbook** | FAIL | 0.350 | No edge |
| **M14 Sweep** | SKIP | 0.500 | — |
| **M17 Resistance Quality** | PASS | 0.542 | Moderate resistance above |
| **M20 Failed Breakout** | PASS | 1.000 | **BLOCKED** (stale 22 bars @ $1,776.54) |
| **M21 Wyckoff** | PASS | 0.250 | MARKUP phase, DISCOUNT zone |
| **M22 Inflation Regime** | PASS | 0.350 | STAGFLATION_LITE, size_mult=0.85 |
| **M23 PPI Session** | PASS | 0.651 | STAGFLATION_HOT, claims today, decay=0.7 |
| **M72 BTC Dom** | PASS | 0.620 | BTC_DOMINANT (55.98%) |

---

## 💰 DERIVATIVES & ORDER FLOW

### Core Derivatives (Binance Aggregate)
| Metric | Value | Signal |
|--------|-------|--------|
| **Open Interest** | $3.92B | High |
| **OI ROC (1h)** | -0.138%/hr | Slow deleveraging |
| **Long/Short Ratio** | 1.998 | **CROWDED_LONG** (z=2.07) |
| **Top Trader L/S** | 2.013 | Extreme long bias |
| **Whale Signal** | NEUTRAL | — |
| **Futures Taker Ratio** | 0.605 | **SELLERS_DOMINANT** |
| **Funding Rate** | 0.002% | Near zero |
| **OI/Price Divergence** | NONE | — |

### Cross-Exchange Activity
| Exchange | OI Share | L/S Ratio | Funding | Signal |
|----------|----------|-----------|---------|--------|
| OKX | 64.7% | 1.645 | FALLING | Most bearish L/S |
| Binance | 21.1% | 1.652 | **RISING** | Bearish funding trend |
| Bybit | 7.0% | 2.071 | **RISING** | Most bullish L/S |
| HTX | 6.4% | — | STABLE | — |
| Phemex | 0.8% | — | **RISING** | Bearish funding trend |

**Exchange Score:** 0.61 (PASS) — Bearish funding trends on 4/6 exchanges
**Spot Score:** 0.45 (PASS) — BACKWARDATION (-0.0499%), strong bid support (ratio 2.08)

### Intrabar CVD (4b)
- **Divergence:** NONE
- **CVD Slope (12):** N/A
- **Status:** SKIP

### Taker Flow Summary (15m)
| Metric | Value | Regime |
|--------|-------|--------|
| Raw Taker Ratio | 0.273 | **SELLING_SPIKE** |
| 4h Avg | 0.474 | Selling pressure |
| 12h Avg | 0.462 | Confirmed selling |
| Momentum | +0.012 | Accelerating slightly |
| Percentile | 46.5 | Mid-range |

---

## 🎯 SQUEEZE & BREAKOUT ANALYSIS

### M18 Squeeze Detection
| Parameter | Value |
|-----------|-------|
| **Type** | LONG_SQUEEZE |
| **Status** | **PENDING** (not triggered) |
| **Direction** | SHORT |
| **Score** | 0.197 (weak) |
| **Compression** | 30 bars (deep) |
| **Dry Bars** | 14 / Doji: 11 |
| **Trigger** | TAKER_SPIKE → SHORT |
| **Entry** | $1,763.35 (wait for 15m close below) |
| **TP** | $1,749.55 (-0.78%) |
| **SL** | $1,768.88 (+0.31%) |
| **RR** | 2.5 : 1 |
| **Lifecycle** | TIGHTENING |
| **Failed Breakout** | YES — down 13 bars ago (spike $1,763.33 → close $1,767.71) |
| **Quality** | 0.197 (low) |
| **ICS Boost** | +0.10 |
| **Size Mult** | 0.80 |

### M19 Breakout Confirmation
| Filter | Passed | Detail |
|--------|--------|--------|
| CVD Flip | ✅ | CVD flipped (M4=NONE, M4b=NONE) |
| BTC Confluence | ✅ | BTC stable/bearish |
| Volume Surge | ❌ | Vol 0.12x MA20 (need 1.5x) |
| OI Expansion | ❌ | OI -0.14%/hr |
| Liquidity Hold | ✅ | Swept $1,773: held below 3 bars |
| Spot Flow | ✅ | 75% weighted SHORT-aligned |
| Funding Stay | ✅ | +0.002% (longs pay) |
| **Total** | **5/7** | **CONFIRMED** (score 0.714) |

---

## 📍 LIQUIDITY MAP (M15)

### Key Clusters (Distance from $1,765)
| Level | Type | Strength | Source | Cascade Risk | Status |
|-------|------|----------|--------|--------------|--------|
| **$1,777.75** | SHORT_STOP | 56.8 | S/R 75 @ $1,765 | **HIGH** | 🔴 Unswept |
| **$1,784.96** | SHORT_STOP | 93.6 | S/R 122 @ $1,773 | **HIGH** | 🔴 Unswept |
| **$1,792.18** | SHORT_STOP | 79.2 | S/R 110 @ $1,780 | **HIGH** | 🔴 Unswept |
| $1,759.93 | LONG_STOP | 38.8 | Swing L $1,763 | MED | Unswept |
| $1,746.70 | LONG_STOP | 71.0 | Rejection L $1,750 | MED | Fresh (10 bars) |
| $1,741.45 | LONG_STOP | 82.2 | Swing L $1,750 | MED | Fresh (10 bars) |
| $1,765.00 | **BID_WALL** | 100 | OB (1,553 ETH) | LOW | Current |
| $1,765.00 | **ASK_WALL** | 100 | OB (3,669 ETH) | LOW | Current |

**High Cascade Zones:** 3 (all above — short stops stacked $1,778–$1,792)

---

## 🏗️ WYCKOFF & STRUCTURE (M21)

| Component | Value |
|-----------|-------|
| **Phase** | MARKUP (confidence 0.7) |
| **Zone** | DISCOUNT (position 0.39) |
| **EQ** | $1,779.72 |
| **Range** | $1,713.44 – $1,846.00 |
| **Kill Zone** | OFF (mult 0.7) |
| **Spring/Upthrust** | NONE |
| **HH/HL/LH/LL** | HH ✅, HL ✅, LH ❌, LL ❌ |

**Implication:** Markup phase penalizes SHORT. Selling in discount zone. Outside kill zone.

---

## 🌍 MACRO & CASCADE ANALYSIS

### Current Macro Phase: 🔥 **CPI_WEEK** (Week 2: 8th–14th)
- **Biggest movers:** CPI/PPI — primary ETH catalysts
- **Next major:** PBoC LPR Decision (6d 4h)
- **Regime:** NEUTRAL

### M22 Inflation Regime Aggregator
| Dimension | Signal | Label | Weight |
|-----------|--------|-------|--------|
| Inflation | 0.65 | **INFLATION_HOT** (CPI 4.2%, PPI 8.8%) | — |
| Labor | 0.00 | NORMAL (Unemp 4.2%) | — |
| Policy | 0.00 | HOLDING (Fed 3.60%) | — |
| Global | 0.00 | NO_DATA | — |
| **Composite** | **0.35** | **STAGFLATION_LITE** | MEDIUM |
| **Size Mult** | **0.85x** | — | — |

### M23 PPI Session (Jul 10 release, 3 days ago)
- **Regime:** STAGFLATION_HOT
- **US Session:** FLAT (-0.30%), SMALL magnitude
- **Asia:** CONTINUATION (UP → UP), gap held, 25% recovery
- **UK:** CONFIRMED (UP +0.82% vs Asia)
- **Claims:** Today! (Rising trend: 222 vs 212, +4.7%)
- **Decay:** 0.70x (3 days post-release)
- **Expected ETH Move:** -1.25% (PPI hot + CPI hot, Fed HOLD)

### Cascade Meta (Combined Score: 0.56 → HOLD)
| Cascade | Weight | Score | Signal | Expected Move | Confidence |
|---------|--------|-------|--------|---------------|------------|
| **US_INFLATION** | 0.35 | 0.62 | BUY | -0.65% | MEDIUM |
| **US_LABOR** | 0.25 | 0.50 | BUY | +1.22% | LOW |
| **CHINA_MACRO** | 0.10 | 0.50 | HOLD | 0.00% | LOW |
| EU_MACRO | 0.08 | 0.50 | SKIP | — | — |
| UK_MACRO | 0.05 | 0.50 | SKIP | — | — |
| JAPAN_MACRO | 0.05 | 0.50 | SKIP | — | — |

**US_INFLATION Detail:** PPI HOT (confirmation), but primary CPI denied → HOLD. Decay 0.70x.
**US_LABOR Detail:** NFP 10 days ago (NEUTRAL), Unemployment NEUTRAL → BUY (decay 0.20x).

---

## ⚔️ M20 FAILED BREAKOUT — THE BLOCKER

```
Status: PASS (score 1.0) → BLOCKED by filter
Level: $1,776.54 (resistance)
Age: 22 bars (very stale)
Quality: Low
Reason: "M20 stale block: breakout failure is 22 bars old (very stale)"
Contrarian Direction: LONG (failed down breakout → bounce bias)
```

**Critical:** The M20 filter **blocks SHORT entries** because a failed downside breakout from 22 bars ago creates a contrarian LONG bias. This directly contradicts the direction resolver's SHORT call.

---

## 📋 ENTRY / TRADE PLAN

### If Forcing SHORT (Against Ensemble, Blocked by M20)
| Parameter | Value |
|-----------|-------|
| **Entry** | Wait for squeeze trigger: 15m close below **$1,763.35** |
| **Stop Loss** | $1,768.88 (above coil high $1,778.32 + buffer) |
| **Take Profit** | $1,749.55 (ATR-based, -0.78%) |
| **Risk/Reward** | 2.5 : 1 |
| **Position Size** | Base × 0.844 (dir) × 0.80 (squeeze) × 0.85 (M22) × 0.70 (M23) = **~0.40x** |
| **Max Risk** | ~0.31% of account per 1R |

### If Following Ensemble LONG (Contrarian to Structure)
| Parameter | Value |
|-----------|-------|
| **Entry** | No clear trigger — would need breakout above $1,776–$1,778 |
| **Stop** | Below $1,759 (long stop cluster) |
| **Target** | $1,773 (magnet) → $1,780 (EQ) → $1,799 (HVN) |
| **Risk** | Fighting bearish structure, Phase0 death zone, M20 contrarian |

---

## 🚦 GATEKEEPER & VETO STATUS

| Gate | Status | Detail |
|------|--------|--------|
| **Regime Block (M9)** | PASS | NEUTRAL not in block list |
| **Sweep Filter** | PASS | Not blocked |
| **M20 Filter** | **BLOCKED** | Stale failed breakout (22 bars) |
| **Ensemble** | PASS | 2/2 agree LONG (weighted 1.69) |
| **Coherence** | — | Not computed in this scan |
| **Vetoes** | — | Not computed in this scan |

---

## 🎯 KEY LEVELS TO WATCH

| Level | Type | Significance |
|-------|------|--------------|
| **$1,763.35** | Squeeze Entry | 15m close below → triggers SHORT |
| **$1,765.00** | Current Price / BID/ASK Wall | Heavy order book clustering |
| **$1,765.39** | S/R Resistance | 71 strength, 75 touches |
| **$1,768.97** | S/R Resistance | 99 strength, 104 touches |
| **$1,772.55** | S/R Resistance | 117 strength (top cluster) |
| **$1,776.54** | **M20 Failed Breakout Level** | 22 bars old, blocks SHORT |
| **$1,777.75** | Short Stop Cluster | 56.8 str, cascade HIGH |
| **$1,784.96** | Short Stop Cluster | 93.6 str, cascade HIGH |
| **$1,738.63** | Downside Magnet | 1.57 str, unswept |
| **$1,714.77** | Major Gap | 2.85% below |

---

## 📅 UPCOMING CATALYSTS (Next 7 Days)

| Time | Event | Impact | Note |
|------|-------|--------|------|
| **17h 09m** | Michigan Consumer Sentiment | LOW | — |
| **1d 16h** | US Retail Sales | MEDIUM | Leads PCE by 2–4 weeks |
| **2d 16h** | Jobless Claims | MEDIUM | Rising trend (222 → watch) |
| **3d 16h** | US Housing Starts | LOW | — |
| **4d 06h** | **China GDP (QoQ)** | MEDIUM | China activity pulse |
| **4d 09h** | **UK CPI** | MEDIUM | BoE policy input |
| **6d 04h** | **PBoC LPR Decision** | **HIGH** | China easing signal → ETH lead 1–2w |

---

## 🏁 FINAL VERDICT

### **DO NOT TRADE — CONFLICTED & LOW PROBABILITY**

| Reason | Weight |
|--------|--------|
| **Phase0 Death Zone (0.039)** | 🔴 Critical — macro context invalidates all setups |
| **Direction Resolver vs Ensemble Conflict** | 🔴 Critical — SHORT vs LONG |
| **M20 Stale Block Active** | 🔴 Critical — blocks SHORT entry |
| **M22 Size Mult 0.85x** | ⚠️ Moderate — stagflation reduces position |
| **M23 Decay 0.70x** | ⚠️ Moderate — post-PPI fade |
| **Crowded Longs (L/S z=2.07)** | ⚠️ Moderate — squeeze risk but stale |
| **Low Volume (vol_ratio 0.19)** | ⚠️ Minor — dry market |

### **Recommended Action: WAIT**
1. **Wait for Phase0 > 0.15** (macro health recovery) — currently 0.039
2. **Wait for M20 block to clear** (staleness > threshold or price breaks $1,776.54 cleanly)
3. **Monitor squeeze trigger** at $1,763.35 — if hit with Phase0 recovered, re-evaluate
4. **Watch PBoC LPR (6 days)** — potential China easing could shift macro regime

### **If You Must Trade (Not Recommended):**
- **SHORT only** if: Phase0 > 0.15, M20 clears, 15m closes below $1,763.35 with volume
- **Size:** 0.4x base (stacked penalties: 0.844 × 0.80 × 0.85 × 0.70)
- **Stop:** $1,768.88 (above coil high)
- **Target:** $1,749.55

---

*Report generated from scan_20260713_205049.json | JIMI Framework v3 | 2026-07-13 20:50 UTC*