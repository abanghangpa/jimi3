#!/usr/bin/env python3
"""
PROPER BACKTEST: Actual Scanner Strategies + V2 Agents
Tests the real strategy implementations against historical data.

Compares v1 (old agents) vs v2 (new agents) on same data.
"""

import json, os, sys, time, csv, math, statistics
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def fetch_candles(symbol="ETHUSDT", interval="15m", limit=1500):
    """Fetch candles from Binance."""
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    r.raise_for_status()
    candles = []
    for c in r.json():
        candles.append({
            "ts": c[0], "open": float(c[1]), "high": float(c[2]),
            "low": float(c[3]), "close": float(c[4]), "volume": float(c[5]),
            "taker_buy_vol": float(c[9]),  # Taker buy base asset volume
            "trades": int(c[8]),  # Number of trades
        })
    return candles


def load_oi_data():
    """Load OI history. Format: ts,oi,volume (no header)."""
    path = os.path.join(BASE, "data", "forced_movement", "oi_history.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    rows.append({"ts": int(parts[0]), "oi": float(parts[1])})
                except: pass
    return sorted(rows, key=lambda x: x["ts"])


def load_funding_data():
    """Load funding history. Format: exchange,ts,rate,collected_at."""
    path = os.path.join(BASE, "data", "forced_movement", "funding_history.csv")
    if not os.path.exists(path):
        return {}
    by_ts = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    by_ts[int(parts[1])] = float(parts[2])
                except: pass
    return by_ts


def load_gate():
    path = os.path.join(BASE, "config", "isolation_gate_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# ACTUAL STRATEGY IMPLEMENTATIONS (from src/strategies/)
# ═══════════════════════════════════════════════════════════════

def compute_atr(highs, lows, closes, period=14):
    """Compute ATR."""
    trs = []
    for i in range(1, len(highs)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        trs.append(tr)
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else 0
    return sum(trs[-period:]) / period


def check_trade_flow(candles, idx, df_15m_available=False):
    """
    S21: Trade Flow Momentum v3 — actual implementation.
    Uses taker ratio z-score + trend alignment.
    """
    if idx < 60:
        return None

    price = candles[idx]["close"]
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]
    taker_buys = [c["taker_buy_vol"] for c in candles]

    atr = compute_atr(highs, lows, closes)

    # Session filter
    ts_ms = candles[idx]["ts"]
    hour = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    BAD_HOURS = {4, 5, 6, 19, 20, 22, 23}
    if hour in BAD_HOURS:
        return None

    # Compute taker ratio
    taker_ratios = []
    for i in range(len(candles)):
        if volumes[i] > 0:
            taker_ratios.append(taker_buys[i] / volumes[i])
        else:
            taker_ratios.append(0.5)

    current_ratio = taker_ratios[idx]

    # Z-score (60-bar lookback)
    lookback = taker_ratios[max(0, idx-60):idx]
    if len(lookback) < 20:
        return None
    mean_ratio = np.mean(lookback)
    std_ratio = np.std(lookback)
    if std_ratio == 0:
        return None
    z_score = (current_ratio - mean_ratio) / std_ratio

    # Trend alignment (v4 upgrade)
    if idx >= 4:
        mom_1h = (closes[idx] - closes[idx-4]) / closes[idx-4]
    else:
        mom_1h = 0
    if idx >= 16:
        mom_4h = (closes[idx] - closes[idx-16]) / closes[idx-16]
    else:
        mom_4h = 0

    # Direction from z-score
    if z_score > 1.5:
        direction = "LONG"
        conviction = min(0.5 + abs(z_score) * 0.1, 0.8)
    elif z_score < -1.5:
        direction = "SHORT"
        conviction = min(0.5 + abs(z_score) * 0.1, 0.8)
    else:
        return None

    # Block counter-trend in strong moves
    if mom_1h > 0.015 and direction == "SHORT":
        return None
    if mom_1h < -0.015 and direction == "LONG":
        return None

    # Volume confirmation
    avg_vol = np.mean(volumes[max(0, idx-20):idx])
    vol_ratio = volumes[idx] / avg_vol if avg_vol > 0 else 1
    if vol_ratio < 1.2:
        conviction *= 0.7

    if conviction < 0.4:
        return None

    if direction == "LONG":
        sl = price - atr * 1.5
        tp1 = price + atr * 2.0
    else:
        sl = price + atr * 1.5
        tp1 = price - atr * 2.0

    return {
        "strategy": "trade_flow", "direction": direction,
        "entry": price, "sl": round(sl, 2), "tp1": round(tp1, 2),
        "conviction": round(conviction, 3),
        "tp_pct": abs(tp1 - price) / price * 100,
        "sl_pct": abs(price - sl) / price * 100,
        "hold_bars": 12,
    }


def check_taker_flow(candles, idx):
    """
    S07: Taker Flow — actual implementation.
    Uses taker ratio + trend alignment.
    """
    if idx < 60:
        return None

    price = candles[idx]["close"]
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]
    taker_buys = [c["taker_buy_vol"] for c in candles]

    atr = compute_atr(highs, lows, closes)

    # Compute taker ratio
    taker_ratios = []
    for i in range(len(candles)):
        if volumes[i] > 0:
            taker_ratios.append(taker_buys[i] / volumes[i])
        else:
            taker_ratios.append(0.5)

    current_ratio = taker_ratios[idx]

    # Z-score
    lookback = taker_ratios[max(0, idx-60):idx]
    if len(lookback) < 20:
        return None
    mean_ratio = np.mean(lookback)
    std_ratio = np.std(lookback)
    if std_ratio == 0:
        return None
    z_score = (current_ratio - mean_ratio) / std_ratio

    # Direction
    if z_score > 2.0:
        direction = "LONG"
        conviction = min(0.5 + abs(z_score) * 0.08, 0.75)
    elif z_score < -2.0:
        direction = "SHORT"
        conviction = min(0.5 + abs(z_score) * 0.08, 0.75)
    else:
        return None

    # Trend alignment (v4 upgrade)
    if idx >= 4:
        mom_1h = (closes[idx] - closes[idx-4]) / closes[idx-4]
    else:
        mom_1h = 0

    if mom_1h > 0.015 and direction == "SHORT":
        return None
    if mom_1h < -0.015 and direction == "LONG":
        return None

    # Flow acceleration
    if idx >= 5:
        recent_ratios = taker_ratios[idx-5:idx+1]
        accel = recent_ratios[-1] - recent_ratios[0]
        if direction == "LONG" and accel > 0:
            conviction *= 1.1
        elif direction == "SHORT" and accel < 0:
            conviction *= 1.1

    if conviction < 0.4:
        return None

    if direction == "LONG":
        sl = price - atr * 1.5
        tp1 = price + atr * 2.0
    else:
        sl = price + atr * 1.5
        tp1 = price - atr * 2.0

    return {
        "strategy": "taker_flow", "direction": direction,
        "entry": price, "sl": round(sl, 2), "tp1": round(tp1, 2),
        "conviction": round(conviction, 3),
        "tp_pct": abs(tp1 - price) / price * 100,
        "sl_pct": abs(price - sl) / price * 100,
        "hold_bars": 8,
    }


def check_orderbook_imbalance(candles, idx):
    """
    S19: Orderbook Imbalance — actual implementation.
    Uses volume imbalance + trend adjustment.
    """
    if idx < 20:
        return None

    price = candles[idx]["close"]
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]
    taker_buys = [c["taker_buy_vol"] for c in candles]

    atr = compute_atr(highs, lows, closes)

    # Volume imbalance (taker buy vs sell ratio)
    if volumes[idx] == 0:
        return None
    buy_ratio = taker_buys[idx] / volumes[idx]
    sell_ratio = 1 - buy_ratio

    # Imbalance score
    imbalance = buy_ratio - sell_ratio  # -1 to +1

    if abs(imbalance) < 0.2:
        return None

    # Volume spike confirmation
    avg_vol = np.mean(volumes[max(0, idx-20):idx])
    vol_ratio = volumes[idx] / avg_vol if avg_vol > 0 else 1

    if vol_ratio < 1.5:
        return None  # Need volume spike

    # Direction from imbalance
    if imbalance > 0.2:
        direction = "LONG"
        conviction = min(0.5 + abs(imbalance) * 0.5 + (vol_ratio - 1) * 0.2, 0.85)
    elif imbalance < -0.2:
        direction = "SHORT"
        conviction = min(0.5 + abs(imbalance) * 0.5 + (vol_ratio - 1) * 0.2, 0.85)
    else:
        return None

    # Trend adjustment (v4 upgrade)
    if idx >= 4:
        mom_1h = (closes[idx] - closes[idx-4]) / closes[idx-4]
    else:
        mom_1h = 0

    if mom_1h < -0.015 and direction == "LONG":
        conviction *= 0.5
    if mom_1h > 0.015 and direction == "SHORT":
        conviction *= 0.5

    if conviction < 0.4:
        return None

    if direction == "LONG":
        sl = price - atr * 0.75
        tp1 = price + atr * 2.0
    else:
        sl = price + atr * 0.75
        tp1 = price - atr * 2.0

    return {
        "strategy": "orderbook_imbalance", "direction": direction,
        "entry": price, "sl": round(sl, 2), "tp1": round(tp1, 2),
        "conviction": round(conviction, 3),
        "tp_pct": abs(tp1 - price) / price * 100,
        "sl_pct": abs(price - sl) / price * 100,
        "hold_bars": 8,
    }


def check_squeeze_breakout(candles, idx):
    """
    S02: Squeeze Breakout — actual implementation.
    Uses BB squeeze + breakout confirmation.
    """
    if idx < 25:
        return None

    price = candles[idx]["close"]
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    atr = compute_atr(highs, lows, closes)

    # BB(20, 2.0)
    lookback = closes[max(0, idx-20):idx+1]
    sma20 = np.mean(lookback[-20:])
    std20 = np.std(lookback[-20:])
    if std20 == 0 or sma20 == 0:
        return None

    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20

    # Squeeze: BB width < 2%
    if bb_width > 0.02:
        return None

    # Breakout: price breaks above/below BB
    if price > upper_bb:
        direction = "LONG"
        conviction = 0.6
    elif price < lower_bb:
        direction = "SHORT"
        conviction = 0.6
    else:
        return None

    # Volume confirmation
    volumes = [c["volume"] for c in candles]
    avg_vol = np.mean(volumes[max(0, idx-20):idx])
    vol_ratio = volumes[idx] / avg_vol if avg_vol > 0 else 1
    if vol_ratio > 1.5:
        conviction *= 1.2

    conviction = min(conviction, 0.85)

    if direction == "LONG":
        sl = price - atr * 1.0
        tp1 = price + atr * 2.0
    else:
        sl = price + atr * 1.0
        tp1 = price - atr * 2.0

    return {
        "strategy": "squeeze_breakout", "direction": direction,
        "entry": price, "sl": round(sl, 2), "tp1": round(tp1, 2),
        "conviction": round(conviction, 3),
        "tp_pct": abs(tp1 - price) / price * 100,
        "sl_pct": abs(price - sl) / price * 100,
        "hold_bars": 8,
    }


def check_judas_sweep(candles, idx):
    """
    S22: Judas Sweep — actual implementation.
    Uses daily/session H/L sweep + rejection wick.
    """
    if idx < 48:
        return None

    price = candles[idx]["close"]
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]

    atr = compute_atr(highs, lows, closes)

    # Find session high/low (last 48 bars = 12h)
    session_high = max(highs[idx-48:idx])
    session_low = min(lows[idx-48:idx])

    # Check if current bar swept the high or low
    bar_high = highs[idx]
    bar_low = lows[idx]

    # Sweep high then reject (wick above, close below)
    if bar_high > session_high and price < session_high:
        # Rejection wick
        wick_up = bar_high - max(closes[idx], candles[idx]["open"])
        body = abs(closes[idx] - candles[idx]["open"])
        if body > 0 and wick_up / body > 1.5:  # wick > 1.5x body
            direction = "SHORT"
            conviction = 0.55
        else:
            return None
    # Sweep low then reject (wick below, close above)
    elif bar_low < session_low and price > session_low:
        wick_down = min(closes[idx], candles[idx]["open"]) - bar_low
        body = abs(closes[idx] - candles[idx]["open"])
        if body > 0 and wick_down / body > 1.5:
            direction = "LONG"
            conviction = 0.55
        else:
            return None
    else:
        return None

    # Volume confirmation
    avg_vol = np.mean(volumes[max(0, idx-20):idx])
    vol_ratio = volumes[idx] / avg_vol if avg_vol > 0 else 1
    if vol_ratio > 1.5:
        conviction *= 1.15

    conviction = min(conviction, 0.8)

    if direction == "LONG":
        sl = price - atr * 1.5
        tp1 = price + atr * 2.5
    else:
        sl = price + atr * 1.5
        tp1 = price - atr * 2.5

    return {
        "strategy": "judas_sweep", "direction": direction,
        "entry": price, "sl": round(sl, 2), "tp1": round(tp1, 2),
        "conviction": round(conviction, 3),
        "tp_pct": abs(tp1 - price) / price * 100,
        "sl_pct": abs(price - sl) / price * 100,
        "hold_bars": 6,
    }


def check_liquidation_cascade(candles, idx, oi_data, funding_by_ts):
    """
    S20: Liquidation Cascade v3 — actual implementation.
    Uses OI shock + OI estimate + funding divergence.
    """
    if idx < 20:
        return None

    price = candles[idx]["close"]
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    atr = compute_atr(highs, lows, closes)
    if atr == 0:
        return None

    candle_ts = candles[idx]["ts"]

    # Get OI window
    oi_window = [d for d in oi_data if abs(d["ts"] - candle_ts) < 7200000]
    if len(oi_window) < 6:
        return None

    # OI ROC
    cur_oi = oi_window[-1]["oi"]
    ref_oi = oi_window[-6]["oi"]
    if ref_oi > 0:
        oi_roc = (cur_oi - ref_oi) / ref_oi
    else:
        oi_roc = 0

    # Price momentum
    if idx >= 5:
        price_change = (closes[idx] - closes[idx-5]) / closes[idx-5]
    else:
        price_change = 0

    signals = []

    # SOURCE 1: OI shock (OI ROC > 0.5% + price move)
    if abs(oi_roc) > 0.005:
        if oi_roc < -0.005 and price_change < -0.003:
            signals.append({"direction": "SHORT", "strength": min(abs(oi_roc) * 15, 0.8), "source": "oi_shock"})
        elif oi_roc < -0.005 and price_change > 0.003:
            signals.append({"direction": "LONG", "strength": min(abs(oi_roc) * 15, 0.8), "source": "oi_shock"})
        elif oi_roc > 0.01 and abs(price_change) > 0.005:
            d = "LONG" if price_change > 0 else "SHORT"
            signals.append({"direction": d, "strength": min(abs(oi_roc) * 10, 0.7), "source": "oi_surge"})

    # SOURCE 2: OI estimate (OI ROC > 0.3% + volume spike)
    if not signals and abs(oi_roc) > 0.003:
        volumes = [c["volume"] for c in candles]
        avg_vol = np.mean(volumes[max(0, idx-20):idx])
        vol_ratio = volumes[idx] / avg_vol if avg_vol > 0 else 1
        if vol_ratio > 1.5:
            d = "LONG" if price_change > 0 else "SHORT"
            signals.append({"direction": d, "strength": min(abs(oi_roc) * 10 + (vol_ratio-1)*0.2, 0.7), "source": "oi_estimate"})

    # SOURCE 3: Funding divergence
    if not signals:
        nearest_fr = 0
        min_diff = float('inf')
        for fts, rate in funding_by_ts.items():
            diff = abs(fts - candle_ts)
            if diff < min_diff:
                min_diff = diff
                nearest_fr = rate

        if abs(nearest_fr) > 0.0001 and idx >= 10:
            price_change_10 = (closes[idx] - closes[idx-10]) / closes[idx-10]
            if nearest_fr > 0.0001 and price_change_10 < -0.005:
                signals.append({"direction": "SHORT", "strength": min(abs(nearest_fr) * 5000, 0.6), "source": "funding_div"})
            elif nearest_fr < -0.0001 and price_change_10 > 0.005:
                signals.append({"direction": "LONG", "strength": min(abs(nearest_fr) * 5000, 0.6), "source": "funding_div"})

    if not signals:
        return None

    best = max(signals, key=lambda x: x["strength"])
    direction = best["direction"]
    conviction = min(0.45 + best["strength"] * 0.35, 0.85)

    if conviction < 0.45:
        return None

    if direction == "LONG":
        sl = price - atr * 1.5
        tp1 = price + atr * 1.5
    else:
        sl = price + atr * 1.5
        tp1 = price - atr * 1.5

    return {
        "strategy": "liquidation_cascade", "direction": direction,
        "entry": price, "sl": round(sl, 2), "tp1": round(tp1, 2),
        "conviction": round(conviction, 3),
        "tp_pct": abs(tp1 - price) / price * 100,
        "sl_pct": abs(price - sl) / price * 100,
        "hold_bars": 8,
    }


# ═══════════════════════════════════════════════════════════════
# REGIME CLASSIFIER (v1 vs v2)
# ═══════════════════════════════════════════════════════════════

def classify_regime_v1(oi_window, funding_by_ts, candle_ts):
    """v1: Original regime classifier (slow — FR/OI/LS only)."""
    if len(oi_window) < 3:
        return "RANGING", 0.5

    deriv_window = []
    for d in oi_window[-20:]:
        nearest_fr = 0
        min_diff = float('inf')
        for fts, rate in funding_by_ts.items():
            diff = abs(fts - d["ts"])
            if diff < min_diff:
                min_diff = diff
                nearest_fr = rate
        deriv_window.append({"oi": d["oi"], "fr": nearest_fr})

    if len(deriv_window) < 3:
        return "RANGING", 0.5

    latest = deriv_window[-1]
    fr = latest["fr"]
    oi = latest["oi"]

    if len(deriv_window) >= 2:
        prev_oi = deriv_window[-2]["oi"]
        oi_roc = ((oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
    else:
        oi_roc = 0

    avg_fr = sum(d["fr"] for d in deriv_window) / len(deriv_window)

    bull_score = 0.0
    bear_score = 0.0
    stress_score = 0.0

    if fr > 0.000030: bull_score += 1
    elif fr < -0.000010: bear_score += 1
    if oi_roc < -3: stress_score += 2
    elif oi_roc > 5: bull_score += 0.5
    if avg_fr > 0.000015: bull_score += 0.5
    elif avg_fr < -0.000005: bear_score += 0.5

    if stress_score > 2:
        return "STRESS", min(0.9, 0.5 + stress_score * 0.1)
    elif bull_score > bear_score + 0.5:
        return "BULL", min(0.9, 0.5 + (bull_score - bear_score) * 0.15)
    elif bear_score > bull_score + 0.5:
        return "BEAR", min(0.9, 0.5 + (bear_score - bull_score) * 0.15)
    elif bear_score > bull_score and bear_score >= 1.0:
        return "MILDLY_BEARISH", min(0.8, 0.5 + (bear_score - bull_score) * 0.1)
    else:
        return "RANGING", 0.5


def classify_regime_v2(oi_window, funding_by_ts, candle_ts, price_history, idx):
    """v2: Upgraded regime classifier (fast — adds price momentum override)."""
    regime, confidence = classify_regime_v1(oi_window, funding_by_ts, candle_ts)

    # Momentum override
    if idx >= 4 and len(price_history) > idx:
        mom_1h = (price_history[idx] - price_history[idx-4]) / price_history[idx-4]
    else:
        mom_1h = 0
    if idx >= 16 and len(price_history) > idx:
        mom_4h = (price_history[idx] - price_history[idx-16]) / price_history[idx-16]
    else:
        mom_4h = 0

    if mom_1h < -0.015:
        regime = "BEAR"
        confidence = min(0.9, 0.6 + abs(mom_1h) * 10)
    elif mom_1h > 0.015:
        regime = "BULL"
        confidence = min(0.9, 0.6 + mom_1h * 10)
    elif mom_4h < -0.03:
        regime = "STRESS"
        confidence = min(0.9, 0.6 + abs(mom_4h) * 5)
    elif mom_4h > 0.03:
        regime = "BULL"
        confidence = min(0.9, 0.6 + mom_4h * 5)
    elif mom_1h < -0.008:
        regime = "MILDLY_BEARISH"
        confidence = 0.55
    elif mom_1h > 0.008:
        regime = "BULL"
        confidence = 0.55

    return regime, confidence


# ═══════════════════════════════════════════════════════════════
# REGIME-STRATEGY GATE
# ═══════════════════════════════════════════════════════════════

REGIME_GATE = {
    "liquidation_cascade": ["BULL", "BEAR", "STRESS", "RANGING", "MILDLY_BEARISH"],
    "orderbook_imbalance": ["BULL", "BEAR", "STRESS", "RANGING", "MILDLY_BEARISH"],
    "taker_flow": ["BULL", "BEAR", "STRESS", "RANGING", "MILDLY_BEARISH"],
    "trade_flow": ["BULL", "BEAR", "STRESS"],
    "squeeze_breakout": ["RANGING", "BULL", "BEAR"],
    "judas_sweep": ["BULL", "BEAR", "STRESS", "RANGING", "MILDLY_BEARISH"],
}


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def run_backtest(candles, oi_data, funding_by_ts, gate_data, use_v2_agents=False, initial_capital=200.0):
    """
    Run backtest with actual strategies and agent framework.
    
    Args:
        use_v2_agents: If True, use v2 agents (momentum regime, cooldown, dedup, trend filter)
                       If False, use v1 agents (original)
    """
    capital = initial_capital
    peak_capital = initial_capital
    positions = []
    trades = []
    fee_rate = 0.0004

    # v2 agent state
    signal_history = []  # (ts, strategy, direction)
    strategy_trades = defaultdict(list)  # strategy -> [(ts, direction, outcome, entry)]
    last_entry_time = 0
    dedup_window = 3600  # 1h
    cooldown_hours = 4
    max_consecutive_losses = 3
    entry_price_tolerance = 0.005

    # Counters
    regime_blocks = 0
    dedup_blocks = 0
    cooldown_blocks = 0
    trend_blocks = 0
    gate_blocks = 0
    signals_generated = 0
    signals_evaluated = 0

    # Regime tracking
    prev_regime = "RANGING"

    price_history = [c["close"] for c in candles]

    for i in range(60, len(candles)):
        price = candles[i]["close"]
        candle_ts = candles[i]["ts"]

        # ── Check exits ──
        closed = []
        for pos in positions:
            pos["bars_held"] = pos.get("bars_held", 0) + 1
            exited = False
            outcome = "TIMEOUT"
            exit_price = price

            if pos["direction"] == "LONG":
                if candles[i]["low"] <= pos["sl"]:
                    exit_price = pos["sl"]; outcome = "LOSS"; exited = True
                elif candles[i]["high"] >= pos["tp"]:
                    exit_price = pos["tp"]; outcome = "WIN"; exited = True
            else:
                if candles[i]["high"] >= pos["sl"]:
                    exit_price = pos["sl"]; outcome = "LOSS"; exited = True
                elif candles[i]["low"] <= pos["tp"]:
                    exit_price = pos["tp"]; outcome = "WIN"; exited = True

            if not exited and pos.get("bars_held", 0) >= pos.get("hold_bars", 8):
                exited = True; outcome = "TIMEOUT"; exit_price = price

            if exited:
                if pos["direction"] == "LONG":
                    pnl_pct = (exit_price - pos["entry"]) / pos["entry"] * 100
                else:
                    pnl_pct = (pos["entry"] - exit_price) / pos["entry"] * 100
                pnl_pct -= fee_rate * 100
                pnl_dollar = pos["size"] * pnl_pct / 100 * pos.get("leverage", 10)

                capital += pnl_dollar
                peak_capital = max(peak_capital, capital)

                if use_v2_agents:
                    strategy_trades[pos["strategy"]].append({
                        "timestamp": candle_ts / 1000,
                        "direction": pos["direction"],
                        "outcome": outcome,
                        "entry_price": pos["entry"],
                    })

                trades.append({
                    "strategy": pos["strategy"], "direction": pos["direction"],
                    "entry": pos["entry"], "exit": round(exit_price, 2),
                    "outcome": outcome, "pnl_pct": round(pnl_pct, 4),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "regime": pos.get("regime", "?"), "bars_held": pos.get("bars_held", 0),
                })
                closed.append(pos)

        for pos in closed:
            positions.remove(pos)

        # ── Classify regime ──
        oi_window = [d for d in oi_data if abs(d["ts"] - candle_ts) < 7200000]

        if use_v2_agents:
            regime, confidence = classify_regime_v2(oi_window, funding_by_ts, candle_ts, price_history, i)
        else:
            regime, confidence = classify_regime_v1(oi_window, funding_by_ts, candle_ts)

        # ── Generate signals from all strategies ──
        signals = []

        tf = check_trade_flow(candles, i)
        if tf: signals.append(tf)

        taker = check_taker_flow(candles, i)
        if taker: signals.append(taker)

        obi = check_orderbook_imbalance(candles, i)
        if obi: signals.append(obi)

        sqz = check_squeeze_breakout(candles, i)
        if sqz: signals.append(sqz)

        judas = check_judas_sweep(candles, i)
        if judas: signals.append(judas)

        liq = check_liquidation_cascade(candles, i, oi_data, funding_by_ts)
        if liq: signals.append(liq)

        signals_generated += len(signals)

        # ── Evaluate each signal ──
        for sig in signals:
            signals_evaluated += 1
            strategy = sig["strategy"]
            direction = sig["direction"]

            # Skip if already have position in this strategy
            if any(p["strategy"] == strategy for p in positions):
                continue
            # Max 3 positions
            if len(positions) >= 3:
                continue

            # ── Regime gate ──
            allowed = REGIME_GATE.get(strategy, [])
            if allowed and regime not in allowed:
                regime_blocks += 1
                continue

            # ── Isolation gate ──
            gate_entry = gate_data.get(strategy, {})
            gate_passed = gate_entry.get("passed", False)
            if not gate_passed:
                rb = gate_entry.get("regime_breakdown", {})
                if rb and regime.lower() in rb:
                    gate_passed = rb[regime.lower()].get("passed", False)
            if not gate_passed:
                gate_blocks += 1
                continue

            # ── V2: Signal dedup ──
            if use_v2_agents:
                now_ts = candle_ts / 1000
                recent_sigs = [(ts, s, d) for ts, s, d in signal_history if (now_ts - ts) < dedup_window]
                signal_history = recent_sigs
                is_dup = any(s == strategy and d == direction for _, s, d in recent_sigs)
                if is_dup:
                    dedup_blocks += 1
                    continue
                signal_history.append((now_ts, strategy, direction))

                # Rate limit: no new positions within 30 min
                if last_entry_time and (now_ts - last_entry_time) < 1800:
                    continue

            # ── V2: Per-strategy cooldown ──
            if use_v2_agents:
                strat_history = strategy_trades.get(strategy, [])
                now_ts = candle_ts / 1000
                recent_losses = [t for t in strat_history
                                 if t["outcome"] == "LOSS" and (now_ts - t["timestamp"]) < cooldown_hours * 3600]
                # Same direction loss in last 4h
                same_dir_loss = any(t["direction"] == direction for t in recent_losses)
                if same_dir_loss:
                    cooldown_blocks += 1
                    continue
                # 3 consecutive losses
                last_3 = strat_history[-3:] if len(strat_history) >= 3 else strat_history
                if len(last_3) >= 3 and all(t["outcome"] == "LOSS" for t in last_3):
                    last_loss_ts = last_3[-1]["timestamp"]
                    if (now_ts - last_loss_ts) < 24 * 3600:
                        cooldown_blocks += 1
                        continue
                # Same entry price as last loss
                for t in reversed(strat_history):
                    if t["outcome"] == "LOSS" and t["entry_price"] > 0:
                        price_diff = abs(price - t["entry_price"]) / t["entry_price"]
                        if price_diff < entry_price_tolerance:
                            cooldown_blocks += 1
                            continue
                        break

            # ── V2: Trend alignment filter ──
            if use_v2_agents:
                if i >= 4:
                    mom_1h = (price_history[i] - price_history[i-4]) / price_history[i-4]
                else:
                    mom_1h = 0
                if mom_1h < -0.015 and direction == "LONG":
                    trend_blocks += 1
                    continue
                if mom_1h > 0.015 and direction == "SHORT":
                    trend_blocks += 1
                    continue

            # ── Open position ──
            pos = {
                "strategy": strategy, "direction": direction,
                "entry": price, "sl": sig["sl"], "tp": sig["tp1"],
                "size": 10, "leverage": 10,
                "hold_bars": sig.get("hold_bars", 8),
                "conviction": sig["conviction"], "regime": regime,
                "bars_held": 0, "entry_bar": i,
            }
            positions.append(pos)
            last_entry_time = candle_ts / 1000

    # Close remaining
    for pos in positions:
        price = candles[-1]["close"]
        if pos["direction"] == "LONG":
            pnl_pct = (price - pos["entry"]) / pos["entry"] * 100
        else:
            pnl_pct = (pos["entry"] - price) / pos["entry"] * 100
        pnl_pct -= fee_rate * 100
        pnl_dollar = pos["size"] * pnl_pct / 100 * pos.get("leverage", 10)
        capital += pnl_dollar
        trades.append({"strategy": pos["strategy"], "direction": pos["direction"],
                       "entry": pos["entry"], "exit": round(price, 2),
                       "outcome": "TIMEOUT", "pnl_pct": round(pnl_pct, 4),
                       "pnl_dollar": round(pnl_dollar, 2), "regime": pos.get("regime", "?")})

    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    losses = sum(1 for t in trades if t["outcome"] == "LOSS")
    timeouts = sum(1 for t in trades if t["outcome"] == "TIMEOUT")
    gross_profit = sum(t["pnl_dollar"] for t in trades if t["pnl_dollar"] > 0)
    gross_loss = abs(sum(t["pnl_dollar"] for t in trades if t["pnl_dollar"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0

    return {
        "capital": round(capital, 2), "peak_capital": round(peak_capital, 2),
        "pnl": round(capital - initial_capital, 2), "pnl_pct": round((capital - initial_capital) / initial_capital * 100, 1),
        "max_dd": round(dd, 1), "trades": len(trades),
        "wins": wins, "losses": losses, "timeouts": timeouts,
        "wr": round(wins / len(trades) * 100, 1) if trades else 0,
        "pf": round(pf, 2),
        "signals_generated": signals_generated, "signals_evaluated": signals_evaluated,
        "regime_blocks": regime_blocks, "gate_blocks": gate_blocks,
        "dedup_blocks": dedup_blocks, "cooldown_blocks": cooldown_blocks,
        "trend_blocks": trend_blocks,
        "trade_details": trades,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PROPER BACKTEST: Actual Strategies + V1 vs V2 Agents")
    print("=" * 60)

    # Load data
    print("\n📊 Loading data...")
    candles = fetch_candles(limit=1500)
    print(f"  Candles: {len(candles)} (15m)")

    oi_data = load_oi_data()
    print(f"  OI data: {len(oi_data)} points")

    funding_by_ts = load_funding_data()
    print(f"  Funding: {len(funding_by_ts)} points")

    gate_data = load_gate()
    print(f"  Gate entries: {len(gate_data)}")

    if len(candles) < 100 or len(oi_data) < 10:
        print("❌ Insufficient data")
        return

    # Run v1 backtest
    print("\n🔄 Running V1 backtest (original agents)...")
    v1 = run_backtest(candles, oi_data, funding_by_ts, gate_data, use_v2_agents=False)

    # Run v2 backtest
    print("🔄 Running V2 backtest (upgraded agents)...")
    v2 = run_backtest(candles, oi_data, funding_by_ts, gate_data, use_v2_agents=True)

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON: V1 vs V2")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'V1 (old)':<15} {'V2 (new)':<15} {'Change':<10}")
    print("-" * 65)
    for metric, v1_val, v2_val in [
        ("Capital", f"${v1['capital']}", f"${v2['capital']}"),
        ("PnL", f"${v1['pnl']}", f"${v2['pnl']}"),
        ("PnL %", f"{v1['pnl_pct']}%", f"{v2['pnl_pct']}%"),
        ("Max DD", f"{v1['max_dd']}%", f"{v2['max_dd']}%"),
        ("Trades", str(v1['trades']), str(v2['trades'])),
        ("Wins", str(v1['wins']), str(v2['wins'])),
        ("Losses", str(v1['losses']), str(v2['losses'])),
        ("Win Rate", f"{v1['wr']}%", f"{v2['wr']}%"),
        ("Profit Factor", str(v1['pf']), str(v2['pf'])),
    ]:
        print(f"  {metric:<25} {v1_val:<15} {v2_val:<15}")

    print(f"\n🛡️ V2 Agent Filter Stats:")
    print(f"  Signals generated:     {v2['signals_generated']}")
    print(f"  Signals evaluated:     {v2['signals_evaluated']}")
    print(f"  Regime blocks:         {v2['regime_blocks']}")
    print(f"  Gate blocks:           {v2['gate_blocks']}")
    print(f"  Dedup blocks:          {v2['dedup_blocks']}")
    print(f"  Cooldown blocks:       {v2['cooldown_blocks']}")
    print(f"  Trend filter blocks:   {v2['trend_blocks']}")

    # Per strategy comparison
    print(f"\n📋 Per Strategy (V1):")
    for strat in sorted(set(t["strategy"] for t in v1["trade_details"])):
        st = [t for t in v1["trade_details"] if t["strategy"] == strat]
        sw = sum(1 for t in st if t["outcome"] == "WIN")
        sl = sum(1 for t in st if t["outcome"] == "LOSS")
        sp = sum(t["pnl_dollar"] for t in st)
        print(f"  {strat}: {len(st)}T {sw}W/{sl}L PnL=${sp:.2f}")

    print(f"\n📋 Per Strategy (V2):")
    for strat in sorted(set(t["strategy"] for t in v2["trade_details"])):
        st = [t for t in v2["trade_details"] if t["strategy"] == strat]
        sw = sum(1 for t in st if t["outcome"] == "WIN")
        sl = sum(1 for t in st if t["outcome"] == "LOSS")
        sp = sum(t["pnl_dollar"] for t in st)
        print(f"  {strat}: {len(st)}T {sw}W/{sl}L PnL=${sp:.2f}")

    # Per regime
    print(f"\n🌡️ Per Regime (V2):")
    for regime in sorted(set(t.get("regime", "?") for t in v2["trade_details"])):
        rt = [t for t in v2["trade_details"] if t.get("regime") == regime]
        rw = sum(1 for t in rt if t["outcome"] == "WIN")
        rp = sum(t["pnl_dollar"] for t in rt)
        print(f"  {regime}: {len(rt)}T {rw}W PnL=${rp:.2f}")

    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candles": len(candles), "oi_points": len(oi_data),
        "v1": {k: v for k, v in v1.items() if k != "trade_details"},
        "v2": {k: v for k, v in v2.items() if k != "trade_details"},
        "v1_trades": v1["trade_details"],
        "v2_trades": v2["trade_details"],
    }
    out_path = os.path.join(BASE, "data", "5agent_backtest", "proper_backtest_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Results: {out_path}")


if __name__ == "__main__":
    main()
