#!/usr/bin/env python3
"""
Squeeze Breakout Backtest — validates volatility compression → expansion.

Detects squeeze from:
1. ATR contraction (ATR at 20-bar low relative to recent ATR)
2. Range compression (high-low range shrinking over N bars)
3. Bollinger Band squeeze (bandwidth at multi-bar low)

Fires when compression releases (expansion begins).
Direction from taker flow + EMA200.

Usage:
    python3 backtest_squeeze.py [--bars 5000]
"""

import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "eth_15m_merged.csv")
RESULTS_FILE = os.path.join(BASE_DIR, "data", "squeeze_backtest.json")


def load_eth_data(max_bars=None):
    """Load ETH 15m data from CSV."""
    import csv
    opens, highs, lows, closes, volumes, taker_buy = [], [], [], [], [], []
    timestamps = []

    with open(DATA_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                opens.append(float(row['Open']))
                highs.append(float(row['High']))
                lows.append(float(row['Low']))
                closes.append(float(row['Close']))
                volumes.append(float(row['Volume']))
                taker_buy.append(float(row['Taker buy base asset volume']))
                timestamps.append(row.get('Open time', ''))
            except (ValueError, KeyError):
                continue

    if max_bars:
        opens = opens[-max_bars:]
        highs = highs[-max_bars:]
        lows = lows[-max_bars:]
        closes = closes[-max_bars:]
        volumes = volumes[-max_bars:]
        taker_buy = taker_buy[-max_bars:]
        timestamps = timestamps[-max_bars:]

    return {
        "open": np.array(opens), "high": np.array(highs),
        "low": np.array(lows), "close": np.array(closes),
        "volume": np.array(volumes), "taker_buy": np.array(taker_buy),
        "ts": timestamps, "n": len(closes)
    }


def detect_squeeze(d, idx, atr_period=14, squeeze_window=20, bb_period=20, bb_std=2.0):
    """
    Detect if current bar is in a squeeze (volatility compression).

    Returns dict with squeeze info or None.
    """
    if idx < max(atr_period, squeeze_window, bb_period) + 10:
        return None

    closes = d["close"]
    highs = d["high"]
    lows = d["low"]
    volumes = d["volume"]

    # ── 1. ATR COMPRESSION ──
    # Compute ATR series
    tr = np.maximum(highs[1:] - lows[1:],
                   np.maximum(np.abs(highs[1:] - closes[:-1]),
                             np.abs(lows[1:] - closes[:-1])))
    tr = np.insert(tr, 0, highs[0] - lows[0])  # prepend first bar

    atr_window = 14
    atr_series = np.convolve(tr, np.ones(atr_window)/atr_window, mode='valid')
    atr_offset = idx - len(atr_series) + 1  # offset between idx and atr_series index

    if idx - atr_offset < squeeze_window:
        return None

    current_atr = atr_series[idx - atr_offset]
    recent_atr = atr_series[max(0, idx - atr_offset - squeeze_window):idx - atr_offset + 1]
    atr_min = np.min(recent_atr)
    atr_max = np.max(recent_atr)
    atr_pctile = (current_atr - atr_min) / (atr_max - atr_min) if atr_max > atr_min else 0.5

    # Squeeze if ATR is in bottom 20% of recent range
    atr_squeeze = atr_pctile < 0.20

    # ── 2. RANGE COMPRESSION ──
    bar_range = highs[idx] - lows[idx]
    avg_range = np.mean(np.abs(highs[max(0,idx-squeeze_window):idx] - lows[max(0,idx-squeeze_window):idx]))
    range_ratio = bar_range / avg_range if avg_range > 0 else 1.0

    # Also check if recent ranges are contracting
    recent_ranges = np.abs(highs[max(0,idx-squeeze_window):idx+1] - lows[max(0,idx-squeeze_window):idx+1])
    range_trend = np.polyfit(range(len(recent_ranges)), recent_ranges, 1)[0]  # slope
    range_contracting = range_trend < 0  # negative slope = contracting

    # ── 3. BOLLINGER BAND SQUEEZE ──
    seg = closes[max(0, idx-bb_period+1):idx+1]
    if len(seg) < bb_period:
        return None
    sma = np.mean(seg)
    std = np.std(seg)
    bandwidth = (2 * bb_std * std) / sma if sma > 0 else 0

    # Historical bandwidth
    bw_history = []
    for j in range(max(0, idx-100), idx+1):
        s = closes[max(0, j-bb_period+1):j+1]
        if len(s) >= bb_period:
            sm = np.mean(s)
            st = np.std(s)
            if sm > 0:
                bw_history.append(2 * bb_std * st / sm)

    if len(bw_history) < 20:
        return None

    bw_min = np.min(bw_history)
    bw_max = np.max(bw_history)
    bw_pctile = (bandwidth - bw_min) / (bw_max - bw_min) if bw_max > bw_min else 0.5
    bb_squeeze = bw_pctile < 0.20

    # ── COMBINE ──
    squeeze_signals = sum([atr_squeeze, range_contracting, bb_squeeze])

    if squeeze_signals < 2:
        return None  # need at least 2 of 3

    # Squeeze quality: how compressed is it?
    compression_score = (1 - atr_pctile) * 0.4 + (1 - bw_pctile) * 0.4 + min(1, squeeze_window / 50) * 0.2

    # Count compression bars (how long has ATR been below median?)
    compression_bars = 0
    for j in range(idx, max(0, idx-100), -1):
        a_idx = j - atr_offset
        if a_idx >= 0 and a_idx < len(atr_series):
            a_pctile = (atr_series[a_idx] - atr_min) / (atr_max - atr_min) if atr_max > atr_min else 0.5
            if a_pctile < 0.50:
                compression_bars += 1
            else:
                break
        else:
            break

    return {
        "is_squeeze": True,
        "squeeze_quality": float(compression_score),
        "compression_bars": compression_bars,
        "atr_pctile": float(atr_pctile),
        "bw_pctile": float(bw_pctile),
        "range_contracting": range_contracting,
        "current_atr": float(current_atr),
        "bandwidth": float(bandwidth),
    }


def detect_squeeze_release(d, idx, prev_squeeze):
    """
    Detect if a squeeze is releasing (expansion begins).
    Release = ATR expanding after compression + volume surge.
    """
    if not prev_squeeze or not prev_squeeze.get("is_squeeze"):
        return False

    closes = d["close"]
    highs = d["high"]
    lows = d["low"]
    volumes = d["volume"]

    # Current bar expansion
    bar_range = highs[idx] - lows[idx]
    avg_range = np.mean(np.abs(highs[max(0,idx-20):idx] - lows[max(0,idx-20):idx]))
    range_expansion = bar_range / avg_range if avg_range > 0 else 1.0

    # Volume surge
    vol_ma = np.mean(volumes[max(0,idx-20):idx])
    vol_ratio = volumes[idx] / vol_ma if vol_ma > 0 else 1.0

    # Release if: range expanding (>1.2x average) AND volume surge (>1.3x)
    return range_expansion > 1.2 and vol_ratio > 1.3


def compute_direction(d, idx, ema_period=200):
    """Compute direction from taker flow + EMA200."""
    closes = d["close"]
    taker_buy = d["taker_buy"]
    volumes = d["volume"]

    # Taker ratio
    taker_ratio = taker_buy[idx] / volumes[idx] if volumes[idx] > 0 else 0.5

    # EMA200
    if idx >= ema_period:
        ema = np.mean(closes[idx-ema_period:idx])
    else:
        ema = np.mean(closes[:idx+1])

    price = closes[idx]

    # Direction from taker flow
    direction = None
    if taker_ratio > 0.55:
        direction = 'LONG'
    elif taker_ratio < 0.45:
        direction = 'SHORT'
    else:
        return None, taker_ratio, ema

    # EMA200 filter (3% band)
    dist = (price - ema) / ema if ema > 0 else 0
    if direction == 'LONG' and dist < -0.03:
        return None, taker_ratio, ema
    if direction == 'SHORT' and dist > 0.03:
        return None, taker_ratio, ema

    return direction, taker_ratio, ema


def compute_forward_returns(d, idx, horizons=[1, 4, 16, 24]):
    """Compute forward returns."""
    closes = d["close"]
    returns = {}
    for h in horizons:
        if idx + h < len(closes):
            ret = (closes[idx + h] - closes[idx]) / closes[idx] * 100
            returns[f"ret_{h}bar"] = float(ret)
        else:
            returns[f"ret_{h}bar"] = None
    return returns


def isolation_gate(signals):
    """Run isolation gate on signals."""
    from scipy import stats

    long_rets = []
    short_rets = []

    for sig in signals:
        if sig["direction"] is None:
            continue
        ret_4 = sig["forward_returns"].get("ret_4bar")
        if ret_4 is None:
            continue

        if sig["direction"] == "LONG":
            long_rets.append(ret_4)
        else:
            short_rets.append(-ret_4)

    all_rets = long_rets + short_rets

    if len(all_rets) < 20:
        return {"passed": False, "reason": f"too few signals ({len(all_rets)})"}

    mean_ret = float(np.mean(all_rets))
    std_ret = float(np.std(all_rets))
    t_stat, p_value = stats.ttest_1samp(all_rets, 0)

    direction_correct = mean_ret > 0
    effect_size_ok = abs(mean_ret) > 0.10
    passed = p_value < 0.1 and direction_correct and effect_size_ok

    return {
        "passed": passed,
        "events": len(all_rets),
        "mean_return_pct": round(mean_ret, 4),
        "std_return_pct": round(std_ret, 4),
        "p_value": round(float(p_value), 6),
        "t_stat": round(float(t_stat), 4),
        "direction_correct": direction_correct,
        "effect_size_ok": effect_size_ok,
        "long_signals": len(long_rets),
        "short_signals": len(short_rets),
        "long_mean": round(float(np.mean(long_rets)), 4) if long_rets else 0,
        "short_mean": round(float(np.mean(short_rets)), 4) if short_rets else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=10000)
    args = parser.parse_args()

    print(f"=== Squeeze Breakout Backtest ({args.bars} bars) ===")

    # Load data
    print("\nLoading ETH 15m data...")
    d = load_eth_data(args.bars)
    print(f"  {d['n']} bars, ${d['close'][-1]:.2f}")

    # Run strategy
    print("\nDetecting squeezes and releases...")
    signals = []
    in_squeeze = False
    squeeze_info = None
    squeeze_start = 0

    for idx in range(250, d["n"] - 24):  # leave room for ATR warmup + forward returns
        # Check for squeeze
        sq = detect_squeeze(d, idx)

        if sq and not in_squeeze:
            in_squeeze = True
            squeeze_info = sq
            squeeze_start = idx

        # Check for release
        if in_squeeze:
            released = detect_squeeze_release(d, idx, squeeze_info)
            if released:
                # Compute direction
                direction, taker, ema = compute_direction(d, idx)
                fwd = compute_forward_returns(d, idx)

                signals.append({
                    "idx": idx,
                    "price": float(d["close"][idx]),
                    "direction": direction,
                    "taker": float(taker),
                    "ema": float(ema),
                    "squeeze_quality": squeeze_info["squeeze_quality"],
                    "compression_bars": squeeze_info["compression_bars"],
                    "atr_pctile": squeeze_info["atr_pctile"],
                    "bw_pctile": squeeze_info["bw_pctile"],
                    "forward_returns": fwd,
                })

                in_squeeze = False
                squeeze_info = None

            # Squeeze timeout (if no release in 50 bars, abandon)
            elif idx - squeeze_start > 50:
                in_squeeze = False
                squeeze_info = None

    print(f"  Total squeeze releases detected: {len(signals)}")
    fired = [s for s in signals if s["direction"] is not None]
    print(f"  With direction: {len(fired)}")

    if fired:
        longs = [s for s in fired if s["direction"] == "LONG"]
        shorts = [s for s in fired if s["direction"] == "SHORT"]
        print(f"  LONG: {len(longs)}, SHORT: {len(shorts)}")
        avg_quality = np.mean([s["squeeze_quality"] for s in fired])
        avg_bars = np.mean([s["compression_bars"] for s in fired])
        print(f"  Avg squeeze quality: {avg_quality:.3f}")
        print(f"  Avg compression bars: {avg_bars:.1f}")

    # Run isolation gate
    print("\n=== Isolation Gate ===")
    gate = isolation_gate(signals)
    print(f"  Passed: {gate['passed']}")
    print(f"  Events: {gate.get('events', 0)}")
    print(f"  Mean return: {gate.get('mean_return_pct', 0):.4f}%")
    print(f"  p-value: {gate.get('p_value', 1):.6f}")
    print(f"  Direction correct: {gate.get('direction_correct', False)}")
    print(f"  Effect size OK: {gate.get('effect_size_ok', False)}")
    print(f"  Long signals: {gate.get('long_signals', 0)} (mean: {gate.get('long_mean', 0):.4f}%)")
    print(f"  Short signals: {gate.get('short_signals', 0)} (mean: {gate.get('short_mean', 0):.4f}%)")

    # Breakdown by squeeze quality
    if fired:
        print("\n=== Quality Breakdown ===")
        high_q = [s for s in fired if s["squeeze_quality"] > 0.6]
        low_q = [s for s in fired if s["squeeze_quality"] <= 0.6]
        for label, group in [("High quality (>0.6)", high_q), ("Low quality (<=0.6)", low_q)]:
            if group:
                rets = []
                for s in group:
                    r = s["forward_returns"].get("ret_4bar")
                    if r is not None:
                        if s["direction"] == "LONG":
                            rets.append(r)
                        else:
                            rets.append(-r)
                if rets:
                    print(f"  {label}: {len(group)} signals, mean={np.mean(rets):.4f}%, p={stats.ttest_1samp(rets, 0)[1]:.4f}")

    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bars_analyzed": d["n"],
        "squeeze_releases": len(signals),
        "with_direction": len(fired),
        "gate": gate,
        "config": {
            "atr_period": 14, "squeeze_window": 20, "bb_period": 20,
            "release_range_threshold": 1.2, "release_vol_threshold": 1.3,
            "taker_long": 0.55, "taker_short": 0.45, "ema_band": 0.03,
        },
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
