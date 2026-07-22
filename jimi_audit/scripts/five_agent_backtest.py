#!/usr/bin/env python3
"""
5-Agent Framework Backtest Engine
Simulates the full orchestrator pipeline over historical data.

Agents:
1. Structure Agent   — derives edge from price action + OI + funding
2. Regime Classifier — classifies BULL/BEAR/RANGING/STRESS from OI + FR
3. Validation Agent  — checks isolation gate (static config)
4. Risk Agent        — position sizing with DD circuit breaker
5. Execution Agent   — simplified (no orderbook, uses ATR-based slippage)

Data sources:
- Binance 15m candles (fetched live)
- OI history CSV (headerless: ts,oi,volume)
- Funding history CSV (headerless: exchange,ts,rate,collected_at)
- isolation_gate_results.json (static)
- backtest_benchmarks.json (static)
"""

import json, os, sys, math, statistics, csv, time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

OI_CSV = os.path.join(BASE, "data", "forced_movement", "oi_history.csv")
FUNDING_CSV = os.path.join(BASE, "data", "forced_movement", "funding_history.csv")
GATE_FILE = os.path.join(BASE, "config", "isolation_gate_results.json")
BENCH_FILE = os.path.join(BASE, "config", "backtest_benchmarks.json")
OUTPUT_DIR = os.path.join(BASE, "data", "5agent_backtest")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════

def load_candles(symbol="ETHUSDT", interval="15m", limit=1500):
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    r.raise_for_status()
    candles = []
    for c in r.json():
        candles.append({
            "ts": c[0], "open": float(c[1]), "high": float(c[2]),
            "low": float(c[3]), "close": float(c[4]), "volume": float(c[5]),
        })
    return candles


def load_oi_data():
    if not os.path.exists(OI_CSV):
        return []
    rows = []
    with open(OI_CSV) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    rows.append({"ts": int(parts[0]), "oi": float(parts[1])})
                except (ValueError, IndexError):
                    continue
    return sorted(rows, key=lambda x: x["ts"])


def load_funding_data():
    if not os.path.exists(FUNDING_CSV):
        return {}
    by_ts = {}
    with open(FUNDING_CSV) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    by_ts[int(parts[1])] = float(parts[2])
                except (ValueError, IndexError):
                    continue
    return by_ts


def load_gate():
    if not os.path.exists(GATE_FILE):
        return {}
    with open(GATE_FILE) as f:
        return json.load(f)


def load_benchmarks():
    if not os.path.exists(BENCH_FILE):
        return {}
    with open(BENCH_FILE) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# AGENT 1: STRUCTURE AGENT
# ═══════════════════════════════════════════════════════════════

class StructureAgent:
    """
    Assesses market microstructure from price action + OI + funding.
    Returns: direction (LONG/SHORT/NEUTRAL), conviction (0-1), edge_type
    """

    def assess(self, candles, idx, oi_window, funding_by_ts):
        if idx < 20:
            return {"direction": "NEUTRAL", "conviction": 0, "edge_type": "insufficient_data"}

        bull_score = 0.0
        bear_score = 0.0
        components = {}

        price = candles[idx]["close"]

        # 1. OI divergence
        if len(oi_window) >= 6:
            cur_oi = oi_window[-1]["oi"]
            ref_oi = oi_window[-6]["oi"]
            if ref_oi > 0:
                oi_roc = (cur_oi - ref_oi) / ref_oi
                price_change = (candles[idx]["close"] - candles[idx-5]["close"]) / candles[idx-5]["close"]

                # OI dropping + price dropping = trapped longs liquidating
                if oi_roc < -0.005 and price_change < -0.003:
                    bear_score += min(abs(oi_roc) * 20, 2.0)
                    components["oi_divergence"] = "TRAPPED_LONGS"
                # OI dropping + price rising = trapped shorts covering
                elif oi_roc < -0.005 and price_change > 0.003:
                    bull_score += min(abs(oi_roc) * 20, 2.0)
                    components["oi_divergence"] = "TRAPPED_SHORTS"
                # OI surging + price rising = new longs entering
                elif oi_roc > 0.005 and price_change > 0.003:
                    bull_score += min(abs(oi_roc) * 15, 1.5)
                    components["oi_divergence"] = "NEW_LONGS"
                # OI surging + price dropping = new shorts entering
                elif oi_roc > 0.005 and price_change < -0.003:
                    bear_score += min(abs(oi_roc) * 15, 1.5)
                    components["oi_divergence"] = "NEW_SHORTS"
                else:
                    components["oi_divergence"] = "NEUTRAL"

        # 2. Funding rate (contrarian)
        candle_ts = candles[idx]["ts"]
        nearest_fr = None
        min_diff = float('inf')
        for fts, rate in funding_by_ts.items():
            diff = abs(fts - candle_ts)
            if diff < min_diff:
                min_diff = diff
                nearest_fr = rate

        if nearest_fr is not None:
            if nearest_fr > 0.0001:
                bear_score += 1.0  # longs crowded = contrarian short
                components["funding"] = "LONG_CROWDED"
            elif nearest_fr < -0.0001:
                bull_score += 1.0  # shorts crowded = contrarian long
                components["funding"] = "SHORT_CROWDED"
            else:
                components["funding"] = "NEUTRAL"

        # 3. Volume profile
        avg_vol = sum(c["volume"] for c in candles[max(0,idx-20):idx]) / min(20, idx)
        cur_vol = candles[idx]["volume"]
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1

        if vol_ratio > 2.0:
            # High volume = conviction move
            price_change = candles[idx]["close"] - candles[idx-1]["close"]
            if price_change > 0:
                bull_score += 0.5
            else:
                bear_score += 0.5
            components["volume"] = f"HIGH ({vol_ratio:.1f}x)"
        else:
            components["volume"] = "NORMAL"

        # 4. Price structure (swing bias)
        highs = [candles[j]["high"] for j in range(max(0,idx-48), idx)]
        lows = [candles[j]["low"] for j in range(max(0,idx-48), idx)]
        swing_high = max(highs)
        swing_low = min(lows)
        range_pos = (price - swing_low) / (swing_high - swing_low) if swing_high > swing_low else 0.5

        if range_pos > 0.8:
            bear_score += 0.3  # near resistance
            components["structure"] = "NEAR_RESISTANCE"
        elif range_pos < 0.2:
            bull_score += 0.3  # near support
            components["structure"] = "NEAR_SUPPORT"
        else:
            components["structure"] = "MID_RANGE"

        # Consensus
        total = bull_score + bear_score
        if total == 0:
            return {"direction": "NEUTRAL", "conviction": 0, "edge_type": "none", "components": components}

        if bull_score > bear_score:
            direction = "LONG"
            conviction = min(0.95, bull_score / max(total, 1) * (total / 4.0))
        else:
            direction = "SHORT"
            conviction = min(0.95, bear_score / max(total, 1) * (total / 4.0))

        edge_type = max(components.values(), key=lambda x: len(str(x))) if components else "unknown"

        return {
            "direction": direction,
            "conviction": round(conviction, 3),
            "edge_type": edge_type,
            "components": components,
            "bull_score": round(bull_score, 2),
            "bear_score": round(bear_score, 2),
        }


# ═══════════════════════════════════════════════════════════════
# AGENT 2: REGIME CLASSIFIER
# ═══════════════════════════════════════════════════════════════

class RegimeClassifier:
    """
    Multi-signal regime classifier.
    Regimes: BULL, BEAR, RANGING, STRESS, MILDLY_BEARISH
    """

    def __init__(self):
        self.regime = "RANGING"
        self.confidence = 0.5

    def classify(self, oi_window, funding_by_ts, candle_ts):
        if len(oi_window) < 3:
            return "RANGING", 0.5

        # Build deriv window with FR
        deriv_window = []
        for d in oi_window[-20:]:
            nearest_fr = 0
            min_diff = float('inf')
            for fts, rate in funding_by_ts.items():
                diff = abs(fts - d["ts"])
                if diff < min_diff:
                    min_diff = diff
                    nearest_fr = rate
            deriv_window.append({"oi": d["oi"], "fr": nearest_fr, "ls": 2.0})

        latest = deriv_window[-1]
        fr = latest["fr"]
        oi = latest["oi"]

        if len(deriv_window) >= 2:
            prev_oi = deriv_window[-2]["oi"]
            oi_roc = ((oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
        else:
            oi_roc = 0

        avg_fr = sum(d["fr"] for d in deriv_window) / len(deriv_window)
        fr_values = [d["fr"] for d in deriv_window]
        fr_std = (sum((f - avg_fr)**2 for f in fr_values) / max(len(fr_values)-1, 1)) ** 0.5

        bull_score = 0.0
        bear_score = 0.0
        stress_score = 0.0

        if fr > 0.000030:
            bull_score += 1
        elif fr < -0.000010:
            bear_score += 1

        if oi_roc < -3:
            stress_score += 2
        elif oi_roc > 5:
            bull_score += 0.5

        if avg_fr > 0.000015:
            bull_score += 0.5
        elif avg_fr < -0.000005:
            bear_score += 0.5

        if fr_std > 0.00003:
            stress_score += 1

        if stress_score > 2:
            regime = "STRESS"
            confidence = min(0.9, 0.5 + stress_score * 0.1)
        elif bull_score > bear_score + 0.5:
            regime = "BULL"
            confidence = min(0.9, 0.5 + (bull_score - bear_score) * 0.15)
        elif bear_score > bull_score + 0.5:
            regime = "BEAR"
            confidence = min(0.9, 0.5 + (bear_score - bull_score) * 0.15)
        elif bear_score > bull_score and bear_score >= 1.0:
            regime = "MILDLY_BEARISH"
            confidence = min(0.8, 0.5 + (bear_score - bull_score) * 0.1)
        else:
            regime = "RANGING"
            confidence = 0.5

        self.regime = regime
        self.confidence = confidence
        return regime, confidence


# ═══════════════════════════════════════════════════════════════
# AGENT 3: VALIDATION AGENT (Gate + Monitor)
# ═══════════════════════════════════════════════════════════════

class ValidationAgent:
    """
    Checks isolation gate + live performance monitor.
    """

    def __init__(self, gate_data, bench_data):
        self.gate = gate_data
        self.bench = bench_data.get("benchmarks", {})
        self.live_stats = bench_data.get("live_stats", {})
        self.paused = bench_data.get("paused", {})

    def is_gate_passed(self, strategy, regime=None):
        """Check isolation gate. Supports regime-specific override."""
        entry = self.gate.get(strategy)
        if not entry:
            return False, "no gate result"
        if entry.get("passed"):
            return True, "passed"

        # Check regime-specific breakdown
        if regime:
            breakdown = entry.get("regime_breakdown", {})
            regime_lower = regime.lower()
            if regime_lower in breakdown:
                r = breakdown[regime_lower]
                if r.get("passed"):
                    return True, f"passed ({regime_lower})"

        return False, "gate failed"

    def is_monitor_paused(self, strategy):
        """Check if strategy is paused by live performance monitor."""
        if strategy in self.paused:
            return True, self.paused[strategy].get("reason", "paused")

        # Check live vs backtest degradation
        live = self.live_stats.get(strategy, {})
        bench = self.bench.get(strategy, {})

        if not live or not bench:
            return False, ""

        live_trades = live.get("trades", 0)
        if live_trades < 5:
            return False, ""

        live_wr = live.get("wins", 0) / live_trades if live_trades > 0 else 0
        bench_wr = bench.get("wr", 0)

        # Pause if live WR drops below 50% of backtest WR
        if bench_wr > 0 and live_wr < bench_wr * 0.5:
            return True, f"WR degraded: {live_wr:.1%} vs {bench_wr:.1%} backtest"

        return False, ""


# ═══════════════════════════════════════════════════════════════
# AGENT 4: RISK AGENT
# ═══════════════════════════════════════════════════════════════

class RiskAgent:
    """
    Survival-first position sizing.
    """

    def __init__(self, initial_capital=200.0):
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_portfolio_heat = 0.06
        self.max_directional_heat = 0.04
        self.max_strategy_heat = 0.02
        self.max_drawdown_pct = 0.25
        self.dd_reduce_threshold = 0.15
        self.kelly_fraction = 0.25
        self.max_leverage = 25
        self.min_leverage = 5

    def evaluate(self, signal, state, regime):
        capital = state["capital"]
        open_positions = state.get("open_positions", [])
        peak = state.get("peak_capital", capital)
        self.peak_capital = max(self.peak_capital, peak)

        # DD circuit breaker
        drawdown = (self.peak_capital - capital) / self.peak_capital if self.peak_capital > 0 else 0
        if drawdown >= self.max_drawdown_pct:
            return {"approved": False, "reason": f"DD BREAKER {drawdown*100:.1f}%", "size": 0, "leverage": 0}

        # Portfolio heat
        total_risk = sum(abs(p.get("risk", 0)) for p in open_positions)
        if total_risk >= self.max_portfolio_heat:
            return {"approved": False, "reason": f"PORTFOLIO HEAT {total_risk*100:.1f}%", "size": 0, "leverage": 0}

        # Position sizing
        entry = signal["entry"]
        sl = signal["sl"]
        sl_distance_pct = abs(entry - sl) / entry if entry > 0 else 0
        if sl_distance_pct <= 0:
            return {"approved": False, "reason": "Invalid SL", "size": 0, "leverage": 0}

        # Kelly sizing
        conviction = signal.get("conviction", 0.5)
        rr_ratio = signal.get("tp_pct", 2.0) / signal.get("sl_pct", 1.0) if signal.get("sl_pct", 0) > 0 else 1.5
        kelly_pct = conviction - (1 - conviction) / rr_ratio
        kelly_pct = max(0, kelly_pct) * self.kelly_fraction

        # Available heat
        available_heat = self.max_portfolio_heat - total_risk
        max_risk_amount = capital * min(available_heat, self.max_directional_heat)

        # DD scaling
        size_mult = 1.0
        if drawdown >= self.dd_reduce_threshold:
            size_mult = 0.5

        # Regime scaling
        regime_scale = {"BULL": 1.0, "BEAR": 0.9, "RANGING": 0.85, "STRESS": 0.7, "MILDLY_BEARISH": 0.9}
        size_mult *= regime_scale.get(regime, 0.85)

        # Final size
        risk_amount = max_risk_amount * size_mult
        position_size = risk_amount / sl_distance_pct if sl_distance_pct > 0 else 0
        position_size = min(position_size, capital * 0.3)  # max 30% of capital

        leverage = min(int(position_size / (capital * 0.1)) + 1, self.max_leverage)
        leverage = max(leverage, self.min_leverage)

        if position_size < 1:  # min $1 position
            return {"approved": False, "reason": "Position too small", "size": 0, "leverage": 0}

        return {
            "approved": True,
            "size": round(position_size, 2),
            "leverage": leverage,
            "risk_amount": round(risk_amount, 2),
            "drawdown": round(drawdown * 100, 2),
            "reason": "OK",
        }


# ═══════════════════════════════════════════════════════════════
# AGENT 5: EXECUTION AGENT (simplified)
# ═══════════════════════════════════════════════════════════════

class ExecutionAgent:
    """
    Simplified execution validation (no orderbook data).
    Uses ATR-based slippage estimation and signal freshness.
    """

    def __init__(self):
        self.max_signal_age_bars = 2  # 30min on 15m
        self.max_slippage_pct = 0.15

    def validate(self, signal, candles, idx):
        entry = signal["entry"]
        direction = signal["direction"]

        # Signal freshness (should be from current bar)
        # In backtest, signals are always fresh

        # Price drift check (simulate: entry should be close to current price)
        price = candles[idx]["close"]
        drift = abs(price - entry) / entry * 100 if entry > 0 else 0
        if drift > 0.5:
            return {"execute": False, "reason": f"PRICE DRIFT {drift:.2f}%", "adjusted_entry": price}

        # ATR-based slippage estimate
        atr_vals = []
        for j in range(max(1, idx-14), idx):
            h = candles[j]["high"]
            l = candles[j]["low"]
            pc = candles[j-1]["close"]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            atr_vals.append(tr)
        atr = sum(atr_vals) / len(atr_vals) if atr_vals else 0

        slippage_est = (atr / price * 100) * 0.1 if price > 0 else 0  # 10% of ATR
        slippage_est = min(slippage_est, self.max_slippage_pct)

        adjusted_entry = entry * (1 + slippage_est / 100) if direction == "LONG" else entry * (1 - slippage_est / 100)

        return {
            "execute": True,
            "adjusted_entry": round(adjusted_entry, 2),
            "slippage_est": round(slippage_est, 4),
            "reason": "OK",
        }


# ═══════════════════════════════════════════════════════════════
# STRATEGY SIGNAL GENERATORS
# ═══════════════════════════════════════════════════════════════

STRATEGIES = {
    "liquidation_cascade": {
        "allowed_regimes": ["BULL", "BEAR", "STRESS", "RANGING", "MILDLY_BEARISH"],
        "tp_pct": 1.5, "sl_pct": 1.5, "hold_bars": 8,
    },
    "orderbook_imbalance": {
        "allowed_regimes": ["BULL", "BEAR", "STRESS", "RANGING", "MILDLY_BEARISH"],
        "tp_pct": 2.0, "sl_pct": 0.75, "hold_bars": 8,
    },
    "taker_flow": {
        "allowed_regimes": ["BULL", "BEAR", "STRESS", "RANGING", "MILDLY_BEARISH"],
        "tp_pct": 2.0, "sl_pct": 1.5, "hold_bars": 12,
    },
    "trade_flow": {
        "allowed_regimes": ["BULL", "BEAR", "STRESS"],
        "tp_pct": 2.0, "sl_pct": 1.5, "hold_bars": 12,
    },
    "squeeze_breakout": {
        "allowed_regimes": ["RANGING", "BULL", "BEAR"],
        "tp_pct": 2.0, "sl_pct": 1.0, "hold_bars": 8,
    },
    "judas_sweep": {
        "allowed_regimes": ["BULL", "BEAR", "STRESS", "RANGING", "MILDLY_BEARISH"],
        "tp_pct": 1.5, "sl_pct": 1.0, "hold_bars": 6,
    },
    "liquidity_grab": {
        "allowed_regimes": ["RANGING", "MILDLY_BEARISH"],
        "tp_pct": 2.0, "sl_pct": 1.5, "hold_bars": 12,
    },
    "forced_movement": {
        "allowed_regimes": ["STRESS", "BEAR", "MILDLY_BEARISH"],
        "tp_pct": 2.0, "sl_pct": 1.5, "hold_bars": 8,
    },
    "failed_breakout": {
        "allowed_regimes": ["RANGING"],
        "tp_pct": 1.5, "sl_pct": 1.0, "hold_bars": 8,
    },
    "positioning_fade": {
        "allowed_regimes": ["RANGING", "BEAR", "STRESS", "MILDLY_BEARISH"],
        "tp_pct": 1.5, "sl_pct": 1.0, "hold_bars": 8,
    },
}


def generate_signals(candles, idx, oi_window, funding_by_ts):
    """Generate signals for all strategies at candle index."""
    signals = []
    if idx < 20:
        return signals

    price = candles[idx]["close"]

    # ATR
    atr_vals = []
    for j in range(max(1, idx-14), idx):
        h = candles[j]["high"]
        l = candles[j]["low"]
        pc = candles[j-1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        atr_vals.append(tr)
    atr = sum(atr_vals) / len(atr_vals) if atr_vals else 0
    if not atr:
        return signals

    # OI data
    oi_roc = 0
    if len(oi_window) >= 6:
        cur_oi = oi_window[-1]["oi"]
        ref_oi = oi_window[-6]["oi"]
        if ref_oi > 0:
            oi_roc = (cur_oi - ref_oi) / ref_oi

    price_change_5 = (candles[idx]["close"] - candles[idx-5]["close"]) / candles[idx-5]["close"]
    price_change_10 = (candles[idx]["close"] - candles[idx-10]["close"]) / candles[idx-10]["close"]

    # Volume
    avg_vol = sum(c["volume"] for c in candles[max(0,idx-20):idx]) / min(20, idx)
    vol_ratio = candles[idx]["volume"] / avg_vol if avg_vol > 0 else 1

    # Funding
    candle_ts = candles[idx]["ts"]
    nearest_fr = 0
    min_diff = float('inf')
    for fts, rate in funding_by_ts.items():
        diff = abs(fts - candle_ts)
        if diff < min_diff:
            min_diff = diff
            nearest_fr = rate

    # BB
    closes = [candles[j]["close"] for j in range(max(0, idx-20), idx+1)]
    sma20 = sum(closes[-20:]) / min(20, len(closes))
    std20 = (sum((x - sma20)**2 for x in closes[-20:]) / min(20, len(closes))) ** 0.5
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20

    # Swing levels
    highs = [candles[j]["high"] for j in range(max(0,idx-48), idx)]
    lows = [candles[j]["low"] for j in range(max(0,idx-48), idx)]
    swing_high = max(highs) if highs else price
    swing_low = min(lows) if lows else price

    # ── liquidation_cascade ──
    if abs(oi_roc) > 0.005:
        if oi_roc < -0.005 and price_change_5 < -0.003:
            signals.append(_make_signal("liquidation_cascade", "SHORT", price, atr, 0.6, 1.5, 1.5))
        elif oi_roc < -0.005 and price_change_5 > 0.003:
            signals.append(_make_signal("liquidation_cascade", "LONG", price, atr, 0.6, 1.5, 1.5))
        elif oi_roc > 0.01 and abs(price_change_5) > 0.005:
            d = "LONG" if price_change_5 > 0 else "SHORT"
            signals.append(_make_signal("liquidation_cascade", d, price, atr, 0.5, 1.5, 1.5))

    # ── orderbook_imbalance ──
    if vol_ratio > 1.5 and abs(price_change_5) > 0.003:
        d = "LONG" if price_change_5 > 0 else "SHORT"
        conv = min(0.5 + (vol_ratio - 1) * 0.2 + abs(oi_roc) * 10, 0.8)
        signals.append(_make_signal("orderbook_imbalance", d, price, atr, conv, 2.0, 0.75))

    # ── taker_flow ──
    if abs(oi_roc) > 0.003 and vol_ratio > 1.3:
        d = "LONG" if price_change_5 > 0 else "SHORT"
        conv = min(0.5 + abs(oi_roc) * 10 + (vol_ratio - 1) * 0.15, 0.75)
        signals.append(_make_signal("taker_flow", d, price, atr, conv, 2.0, 1.5))

    # ── trade_flow ──
    if abs(price_change_10) > 0.005 and vol_ratio > 1.2:
        d = "LONG" if price_change_10 > 0 else "SHORT"
        conv = min(0.5 + abs(price_change_10) * 10, 0.7)
        signals.append(_make_signal("trade_flow", d, price, atr, conv, 2.0, 1.5))

    # ── squeeze_breakout ──
    if std20 > 0:
        bb_width = (upper_bb - lower_bb) / sma20
        if bb_width < 0.02:  # tight squeeze
            if price > upper_bb:
                signals.append(_make_signal("squeeze_breakout", "LONG", price, atr, 0.6, 2.0, 1.0))
            elif price < lower_bb:
                signals.append(_make_signal("squeeze_breakout", "SHORT", price, atr, 0.6, 2.0, 1.0))

    # ── judas_sweep ──
    if abs(price - swing_high) / price < 0.002 or abs(price - swing_low) / price < 0.002:
        if price_change_5 < -0.003:  # swept high then dropped
            signals.append(_make_signal("judas_sweep", "SHORT", price, atr, 0.55, 1.5, 1.0))
        elif price_change_5 > 0.003:  # swept low then bounced
            signals.append(_make_signal("judas_sweep", "LONG", price, atr, 0.55, 1.5, 1.0))

    # ── liquidity_grab ──
    if abs(price - swing_low) / price < 0.003 and price_change_5 > 0.002:
        signals.append(_make_signal("liquidity_grab", "LONG", price, atr, 0.5, 2.0, 1.5))
    elif abs(price - swing_high) / price < 0.003 and price_change_5 < -0.002:
        signals.append(_make_signal("liquidity_grab", "SHORT", price, atr, 0.5, 2.0, 1.5))

    # ── forced_movement ──
    if abs(oi_roc) > 0.01 and abs(price_change_5) > 0.008:
        d = "LONG" if price_change_5 > 0 else "SHORT"
        signals.append(_make_signal("forced_movement", d, price, atr, 0.65, 2.0, 1.5))

    # ── failed_breakout ──
    if price > swing_high * 0.998 and price_change_5 < -0.002:
        signals.append(_make_signal("failed_breakout", "SHORT", price, atr, 0.5, 1.5, 1.0))
    elif price < swing_low * 1.002 and price_change_5 > 0.002:
        signals.append(_make_signal("failed_breakout", "LONG", price, atr, 0.5, 1.5, 1.0))

    # ── positioning_fade ──
    if nearest_fr > 0.00015 and price_change_10 < -0.005:
        signals.append(_make_signal("positioning_fade", "SHORT", price, atr, 0.5, 1.5, 1.0))
    elif nearest_fr < -0.00015 and price_change_10 > 0.005:
        signals.append(_make_signal("positioning_fade", "LONG", price, atr, 0.5, 1.5, 1.0))

    return signals


def _make_signal(strategy, direction, price, atr, conviction, tp_mult, sl_mult):
    if direction == "LONG":
        sl = price - atr * sl_mult
        tp1 = price + atr * tp_mult
    else:
        sl = price + atr * sl_mult
        tp1 = price - atr * tp_mult

    return {
        "strategy": strategy,
        "direction": direction,
        "entry": price,
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "conviction": conviction,
        "tp_pct": abs(tp1 - price) / price * 100,
        "sl_pct": abs(price - sl) / price * 100,
    }


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

class Orchestrator:
    """
    Wires all 5 agents together.
    """

    def __init__(self, initial_capital=200.0):
        self.structure = StructureAgent()
        self.regime_clf = RegimeClassifier()
        self.gate_data = load_gate()
        self.bench_data = load_benchmarks()
        self.validation = ValidationAgent(self.gate_data, self.bench_data)
        self.risk = RiskAgent(initial_capital)
        self.execution = ExecutionAgent()

        self.stats = {
            "total_evaluated": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "rejection_reasons": defaultdict(int),
        }

    def evaluate(self, signal, candles, idx, oi_window, funding_by_ts, state):
        """
        Full 5-agent evaluation pipeline.
        Returns: decision dict with approved/rejected + details
        """
        self.stats["total_evaluated"] += 1
        strategy = signal["strategy"]
        direction = signal["direction"]
        candle_ts = candles[idx]["ts"]

        result = {
            "strategy": strategy,
            "direction": direction,
            "agents": {},
            "approved": False,
            "reason": "",
        }

        # ── STAGE 1: Structure Agent ──
        structure = self.structure.assess(candles, idx, oi_window, funding_by_ts)
        result["agents"]["structure"] = structure

        if structure["direction"] != "NEUTRAL" and structure["direction"] != direction:
            if structure["conviction"] > 0.6:
                result["reason"] = f"STRUCTURE_DISAGREES: {structure['direction']} ({structure['conviction']:.2f})"
                self.stats["total_rejected"] += 1
                self.stats["rejection_reasons"]["STRUCTURE_DISAGREES"] += 1
                return result

        # ── STAGE 2: Regime Classifier ──
        regime, confidence = self.regime_clf.classify(oi_window, funding_by_ts, candle_ts)
        result["agents"]["regime"] = {"regime": regime, "confidence": confidence}

        # Check regime gate
        cfg = STRATEGIES.get(strategy, {})
        allowed_regimes = cfg.get("allowed_regimes", [])
        if allowed_regimes and regime not in allowed_regimes:
            result["reason"] = f"REGIME_BLOCKED: {regime} not in {allowed_regimes}"
            self.stats["total_rejected"] += 1
            self.stats["rejection_reasons"]["REGIME_BLOCKED"] += 1
            return result

        # ── STAGE 3: Validation Agent (Gate + Monitor) ──
        gate_ok, gate_reason = self.validation.is_gate_passed(strategy, regime)
        result["agents"]["gate"] = {"passed": gate_ok, "reason": gate_reason}

        if not gate_ok:
            result["reason"] = f"GATE_BLOCKED: {gate_reason}"
            self.stats["total_rejected"] += 1
            self.stats["rejection_reasons"]["GATE_BLOCKED"] += 1
            return result

        monitor_paused, monitor_reason = self.validation.is_monitor_paused(strategy)
        result["agents"]["monitor"] = {"paused": monitor_paused, "reason": monitor_reason}

        if monitor_paused:
            result["reason"] = f"MONITOR_PAUSED: {monitor_reason}"
            self.stats["total_rejected"] += 1
            self.stats["rejection_reasons"]["MONITOR_PAUSED"] += 1
            return result

        # ── STAGE 4: Risk Agent ──
        risk_result = self.risk.evaluate(signal, state, regime)
        result["agents"]["risk"] = risk_result

        if not risk_result["approved"]:
            result["reason"] = f"RISK_REJECTED: {risk_result['reason']}"
            self.stats["total_rejected"] += 1
            self.stats["rejection_reasons"]["RISK_REJECTED"] += 1
            return result

        # ── STAGE 5: Execution Agent ──
        exec_result = self.execution.validate(signal, candles, idx)
        result["agents"]["execution"] = exec_result

        if not exec_result["execute"]:
            result["reason"] = f"EXECUTION_REJECTED: {exec_result['reason']}"
            self.stats["total_rejected"] += 1
            self.stats["rejection_reasons"]["EXECUTION_REJECTED"] += 1
            return result

        # ── APPROVED ──
        result["approved"] = True
        result["reason"] = "APPROVED"
        result["position"] = {
            "strategy": strategy,
            "direction": direction,
            "fill_price": exec_result["adjusted_entry"],
            "sl": signal["sl"],
            "tp": signal["tp1"],
            "size": risk_result["size"],
            "leverage": risk_result["leverage"],
            "hold_bars": cfg.get("hold_bars", 8),
            "conviction": signal["conviction"],
            "regime": regime,
        }

        self.stats["total_approved"] += 1
        return result


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def run_5agent_backtest(candles, oi_data, funding_by_ts, initial_capital=200.0):
    """
    Run the full 5-agent backtest over historical data.
    """
    orchestrator = Orchestrator(initial_capital)

    state = {
        "capital": initial_capital,
        "peak_capital": initial_capital,
        "open_positions": [],
        "closed_trades": [],
        "total_pnl": 0,
        "total_fees": 0,
        "trades_count": 0,
        "wins": 0,
        "losses": 0,
        "timeouts": 0,
    }

    all_decisions = []
    trades = []

    WINDOW = 20
    fee_rate = 0.0004  # 0.04% round trip

    for i in range(WINDOW + 10, len(candles)):
        candle_ts = candles[i]["ts"]
        price = candles[i]["close"]

        # Get OI window
        oi_window = [d for d in oi_data if abs(d["ts"] - candle_ts) < 7200000]
        if len(oi_window) < 3:
            continue

        # ── Check open positions for exits ──
        closed_positions = []
        for pos in state["open_positions"]:
            pos["bars_held"] += 1
            exited = False
            exit_price = price
            outcome = "TIMEOUT"

            if pos["direction"] == "LONG":
                if candles[i]["low"] <= pos["sl"]:
                    exit_price = pos["sl"]
                    outcome = "LOSS"
                    exited = True
                elif candles[i]["high"] >= pos["tp"]:
                    exit_price = pos["tp"]
                    outcome = "WIN"
                    exited = True
            else:
                if candles[i]["high"] >= pos["sl"]:
                    exit_price = pos["sl"]
                    outcome = "LOSS"
                    exited = True
                elif candles[i]["low"] <= pos["tp"]:
                    exit_price = pos["tp"]
                    outcome = "WIN"
                    exited = True

            if not exited and pos["bars_held"] >= pos["hold_bars"]:
                exited = True
                outcome = "TIMEOUT"
                exit_price = price

            if exited:
                if pos["direction"] == "LONG":
                    pnl_pct = (exit_price - pos["fill_price"]) / pos["fill_price"] * 100
                else:
                    pnl_pct = (pos["fill_price"] - exit_price) / pos["fill_price"] * 100

                pnl_pct -= fee_rate * 100
                pnl_dollar = pos["size"] * pnl_pct / 100 * pos.get("leverage", 1)

                state["capital"] += pnl_dollar
                state["peak_capital"] = max(state["peak_capital"], state["capital"])
                state["total_pnl"] += pnl_dollar
                state["trades_count"] += 1

                if outcome == "WIN":
                    state["wins"] += 1
                elif outcome == "LOSS":
                    state["losses"] += 1
                else:
                    state["timeouts"] += 1

                trade = {
                    "strategy": pos["strategy"],
                    "direction": pos["direction"],
                    "entry": pos["fill_price"],
                    "exit": round(exit_price, 2),
                    "sl": pos["sl"],
                    "tp": pos["tp"],
                    "size": pos["size"],
                    "leverage": pos.get("leverage", 1),
                    "outcome": outcome,
                    "pnl_pct": round(pnl_pct, 4),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "regime": pos.get("regime", "UNKNOWN"),
                    "bars_held": pos["bars_held"],
                    "entry_bar": pos.get("entry_bar", i),
                    "exit_bar": i,
                }
                trades.append(trade)
                closed_positions.append(pos)

        for pos in closed_positions:
            state["open_positions"].remove(pos)

        # ── Generate signals ──
        signals = generate_signals(candles, i, oi_window, funding_by_ts)

        # ── Evaluate each signal through 5 agents ──
        for sig in signals:
            # Skip if already have position in this strategy
            if any(p["strategy"] == sig["strategy"] for p in state["open_positions"]):
                continue

            # Skip if already have max positions (3)
            if len(state["open_positions"]) >= 3:
                continue

            decision = orchestrator.evaluate(sig, candles, i, oi_window, funding_by_ts, state)

            if decision["approved"]:
                pos = decision["position"]
                pos["entry_bar"] = i
                pos["bars_held"] = 0
                state["open_positions"].append(pos)

            all_decisions.append({
                "bar": i,
                "timestamp": candle_ts,
                "price": price,
                "decision": decision,
            })

    # Close remaining positions at end
    for pos in state["open_positions"]:
        price = candles[-1]["close"]
        if pos["direction"] == "LONG":
            pnl_pct = (price - pos["fill_price"]) / pos["fill_price"] * 100
        else:
            pnl_pct = (pos["fill_price"] - price) / pos["fill_price"] * 100
        pnl_pct -= fee_rate * 100
        pnl_dollar = pos["size"] * pnl_pct / 100 * pos.get("leverage", 1)
        state["capital"] += pnl_dollar
        state["total_pnl"] += pnl_dollar
        state["timeouts"] += 1
        trades.append({
            "strategy": pos["strategy"], "direction": pos["direction"],
            "entry": pos["fill_price"], "exit": round(price, 2),
            "outcome": "TIMEOUT", "pnl_pct": round(pnl_pct, 4),
            "pnl_dollar": round(pnl_dollar, 2), "regime": pos.get("regime", "UNKNOWN"),
        })

    return state, trades, all_decisions, orchestrator


# ═══════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════

def print_report(state, trades, orchestrator):
    print("\n" + "=" * 60)
    print("5-AGENT BACKTEST RESULTS")
    print("=" * 60)

    print(f"\n📊 Portfolio:")
    print(f"  Initial Capital:  $200.00")
    print(f"  Final Capital:    ${state['capital']:.2f}")
    print(f"  Total PnL:        ${state['total_pnl']:.2f} ({state['total_pnl']/200*100:.1f}%)")
    print(f"  Peak Capital:     ${state['peak_capital']:.2f}")

    dd = (state['peak_capital'] - state['capital']) / state['peak_capital'] * 100 if state['peak_capital'] > 0 else 0
    print(f"  Max Drawdown:     {dd:.1f}%")

    print(f"\n📈 Trades:")
    print(f"  Total:            {state['trades_count']}")
    print(f"  Wins:             {state['wins']}")
    print(f"  Losses:           {state['losses']}")
    print(f"  Timeouts:         {state['timeouts']}")

    wr = state['wins'] / state['trades_count'] * 100 if state['trades_count'] > 0 else 0
    print(f"  Win Rate:         {wr:.1f}%")

    gross_profit = sum(t["pnl_dollar"] for t in trades if t["pnl_dollar"] > 0)
    gross_loss = abs(sum(t["pnl_dollar"] for t in trades if t["pnl_dollar"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    print(f"  Profit Factor:    {pf:.2f}")

    print(f"\n🤖 Agent Stats:")
    print(f"  Evaluated:        {orchestrator.stats['total_evaluated']}")
    print(f"  Approved:         {orchestrator.stats['total_approved']}")
    print(f"  Rejected:         {orchestrator.stats['total_rejected']}")

    if orchestrator.stats['rejection_reasons']:
        print(f"\n  Rejection breakdown:")
        for reason, count in sorted(orchestrator.stats['rejection_reasons'].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    # Per-strategy breakdown
    if trades:
        print(f"\n📋 Per Strategy:")
        strategies = set(t["strategy"] for t in trades)
        for strat in sorted(strategies):
            st = [t for t in trades if t["strategy"] == strat]
            s_wins = sum(1 for t in st if t["outcome"] == "WIN")
            s_losses = sum(1 for t in st if t["outcome"] == "LOSS")
            s_pnl = sum(t["pnl_dollar"] for t in st)
            s_wr = s_wins / len(st) * 100 if st else 0
            print(f"  {strat}: {len(st)}T {s_wins}W/{s_losses}L WR={s_wr:.0f}% PnL=${s_pnl:.2f}")

    # Per-regime breakdown
    if trades:
        print(f"\n🌡️ Per Regime:")
        regimes = set(t.get("regime", "UNKNOWN") for t in trades)
        for regime in sorted(regimes):
            rt = [t for t in trades if t.get("regime") == regime]
            r_wins = sum(1 for t in rt if t["outcome"] == "WIN")
            r_pnl = sum(t["pnl_dollar"] for t in rt)
            r_wr = r_wins / len(rt) * 100 if rt else 0
            print(f"  {regime}: {len(rt)}T {r_wins}W WR={r_wr:.0f}% PnL=${r_pnl:.2f}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("5-AGENT FRAMEWORK BACKTEST")
    print("=" * 60)

    print("\n📊 Loading data...")
    candles = load_candles(limit=1500)
    print(f"  Candles: {len(candles)} (15m)")

    oi_data = load_oi_data()
    print(f"  OI data: {len(oi_data)} points")

    funding_by_ts = load_funding_data()
    print(f"  Funding: {len(funding_by_ts)} points")

    gate = load_gate()
    print(f"  Gate entries: {len(gate)}")

    if not candles or len(oi_data) < 10:
        print("❌ Insufficient data")
        return

    # Run backtest
    print("\n🔄 Running 5-agent backtest...")
    state, trades, decisions, orchestrator = run_5agent_backtest(candles, oi_data, funding_by_ts)

    # Print report
    print_report(state, trades, orchestrator)

    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candles": len(candles),
        "oi_points": len(oi_data),
        "funding_points": len(funding_by_ts),
        "state": {k: v for k, v in state.items() if k != "open_positions"},
        "trades": trades,
        "agent_stats": dict(orchestrator.stats["rejection_reasons"]),
        "total_evaluated": orchestrator.stats["total_evaluated"],
        "total_approved": orchestrator.stats["total_approved"],
        "total_rejected": orchestrator.stats["total_rejected"],
    }

    output_path = os.path.join(OUTPUT_DIR, "backtest_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {output_path}")

    # Also save trade log
    trade_path = os.path.join(OUTPUT_DIR, "trades.json")
    with open(trade_path, "w") as f:
        json.dump(trades, f, indent=2)
    print(f"💾 Trade log: {trade_path}")


if __name__ == "__main__":
    main()
