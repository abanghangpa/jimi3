#!/usr/bin/env python3
"""
Synthetic Isolation Gate Runner
Runs the isolation gate on 20 synthetic datasets for specified strategies.

Per BACKTEST_FRAMEWORK.md:
1. Log every event (no filtering)
2. Compute unconditional forward returns at 1-bar, 4-bar, 16-bar, 24-bar
3. Split on key variable
4. t-test the two buckets
5. Gate passes if: p < 0.1, correct direction, effect size > costs

Usage:
    python3 synthetic_isolation_gate.py [--strategy NAME] [--sets DIR] [--output FILE]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats


# === INDICATOR FUNCTIONS ===

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1)
    return 100 - (100 / (1 + rs))

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_bb(series, period=20, std_dev=2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower

def calc_vol_ratio(volume, period=20):
    ma = volume.rolling(period).mean()
    return volume / ma.replace(0, 1)


# === STRATEGY DETECTION LOGIC ===
# Each returns a Series of booleans (True = event detected) and optionally a direction Series

def detect_momentum_v3(df):
    """Momentum exhaustion filter: RSI + MACD + BB + volume divergence."""
    close = df["close"]
    rsi = calc_rsi(close, 14)
    ema12 = calc_ema(close, 12)
    ema26 = calc_ema(close, 26)
    macd = ema12 - ema26
    macd_signal = calc_ema(macd, 9)
    macd_hist = macd - macd_signal
    bb_upper, bb_mid, bb_lower = calc_bb(close, 20, 2.0)
    vol_ratio = calc_vol_ratio(df["volume"], 20)

    # Exhaustion score: count of aligned signals
    score = pd.Series(0, index=df.index, dtype=float)

    # RSI overbought/oversold
    score += (rsi > 70).astype(float)  # overbought
    score += (rsi < 30).astype(float)  # oversold

    # MACD histogram reversal
    macd_hist_diff = macd_hist.diff()
    score += ((macd_hist > 0) & (macd_hist_diff < 0)).astype(float)  # bullish exhaustion
    score += ((macd_hist < 0) & (macd_hist_diff > 0)).astype(float)  # bearish exhaustion

    # BB touch
    score += (close >= bb_upper).astype(float)
    score += (close <= bb_lower).astype(float)

    # Volume spike
    score += (vol_ratio > 1.5).astype(float)

    # Event: score >= 2 (at least 2 signals aligned)
    events = score >= 2
    direction = pd.Series("NONE", index=df.index)
    direction[rsi > 70] = "SHORT"
    direction[rsi < 30] = "LONG"

    return events, direction


def detect_squeeze_breakout(df):
    """BB inside KC squeeze + expansion breakout."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    bb_upper, bb_mid, bb_lower = calc_bb(close, 20, 2.0)
    atr = calc_atr(high, low, close, 10)
    kc_upper = bb_mid + 1.5 * atr
    kc_lower = bb_mid - 1.5 * atr

    # Squeeze: BB inside KC
    squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    # Release: was squeezing, now not
    was_squeezing = squeeze.shift(1).fillna(False)
    released = was_squeezing & ~squeeze

    # Direction from close relative to mid
    direction = pd.Series("NONE", index=df.index)
    direction[released & (close > bb_mid)] = "LONG"
    direction[released & (close < bb_mid)] = "SHORT"

    events = released & (direction != "NONE")
    return events, direction


def detect_funding_arb(df):
    """Funding rate divergence proxy: taker ratio extreme + OI divergence."""
    close = df["close"]
    taker_ratio = df["taker_buy_volume"] / df["volume"].replace(0, 1)
    vol_ratio = calc_vol_ratio(df["volume"], 20)

    # Extreme taker imbalance (proxy for funding pressure)
    taker_ma = taker_ratio.rolling(50).mean()
    taker_std = taker_ratio.rolling(50).std()
    taker_zscore = (taker_ratio - taker_ma) / taker_std.replace(0, 1)

    # Extreme taker + volume spike = potential arb opportunity
    events = (taker_zscore.abs() > 2.0) & (vol_ratio > 1.2)

    direction = pd.Series("NONE", index=df.index)
    direction[taker_zscore > 2.0] = "LONG"  # heavy selling -> mean reversion up
    direction[taker_zscore < -2.0] = "SHORT"  # heavy buying -> mean reversion down

    return events, direction


def detect_failed_breakout(df):
    """Sweep beyond level + reclaim (failed breakout)."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Swing highs/lows (5-bar fractal)
    swing_high = pd.Series(np.nan, index=df.index)
    swing_low = pd.Series(np.nan, index=df.index)
    for i in range(3, len(df) - 3):
        if high.iloc[i] > high.iloc[i-3:i].max() and high.iloc[i] > high.iloc[i+1:i+4].max():
            swing_high.iloc[i] = high.iloc[i]
        if low.iloc[i] < low.iloc[i-3:i].min() and low.iloc[i] < low.iloc[i+1:i+4].min():
            swing_low.iloc[i] = low.iloc[i]

    swing_high = swing_high.ffill()
    swing_low = swing_low.ffill()

    # Sweep: price goes above swing high then closes below (bull trap / failed breakout short)
    sweep_high = (high > swing_high) & (close < swing_high)
    # Sweep: price goes below swing low then closes above (bear trap / failed breakout long)
    sweep_low = (low < swing_low) & (close > swing_low)

    events = sweep_high | sweep_low
    direction = pd.Series("NONE", index=df.index)
    direction[sweep_high] = "SHORT"
    direction[sweep_low] = "LONG"

    return events, direction


# === STRATEGY REGISTRY ===

STRATEGIES = {
    "momentum_v3": {
        "detect": detect_momentum_v3,
        "key_variable": "exhaustion_aligned",
        "description": "RSI + MACD + BB + volume exhaustion",
    },
    "squeeze_breakout": {
        "detect": detect_squeeze_breakout,
        "key_variable": "squeeze_release",
        "description": "BB/KC squeeze compression + expansion breakout",
    },
    "funding_arb": {
        "detect": detect_funding_arb,
        "key_variable": "taker_extreme",
        "description": "Taker ratio extreme + volume spike mean reversion",
    },
    "failed_breakout": {
        "detect": detect_failed_breakout,
        "key_variable": "sweep_reclaim",
        "description": "Swing level sweep + close back inside",
    },
}


# === ISOLATION GATE ===

def run_isolation_gate(df, events, direction, strategy_name, round_trip_cost=0.001):
    """
    Run the isolation gate on detected events.

    Returns dict with:
    - events: number of events detected
    - horizons: forward return stats at each horizon
    - t_test: t-test results
    - gate_passed: bool
    - gate_reason: string
    """
    close = df["close"].values
    n = len(close)
    event_indices = np.where(events)[0]

    if len(event_indices) < 5:
        return {
            "events": len(event_indices),
            "horizons": {},
            "t_test": {},
            "gate_passed": False,
            "gate_reason": f"Too few events ({len(event_indices)} < 5)",
        }

    horizons = [1, 4, 16, 24]
    results = {}

    for h in horizons:
        # Compute forward returns
        forward_returns = []
        for idx in event_indices:
            if idx + h < n:
                ret = (close[idx + h] - close[idx]) / close[idx]
                forward_returns.append(ret)

        if len(forward_returns) < 3:
            results[f"{h}bar"] = {"mean": None, "n": len(forward_returns), "note": "too few"}
            continue

        fr = np.array(forward_returns)
        mean_ret = np.mean(fr)
        std_ret = np.std(fr, ddof=1)

        # Non-event forward returns for comparison
        non_event_mask = np.ones(n, dtype=bool)
        non_event_mask[event_indices] = False
        non_event_indices = np.where(non_event_mask)[0]
        non_event_returns = []
        for idx in non_event_indices:
            if idx + h < n:
                non_event_returns.append((close[idx + h] - close[idx]) / close[idx])

        ne = np.array(non_event_returns) if non_event_returns else np.array([0.0])

        # t-test: event returns vs non-event returns
        if len(fr) > 1 and len(ne) > 1:
            t_stat, p_value = stats.ttest_ind(fr, ne, equal_var=False)
        else:
            t_stat, p_value = 0.0, 1.0

        results[f"{h}bar"] = {
            "mean_return_pct": round(mean_ret * 100, 4),
            "std_pct": round(std_ret * 100, 4),
            "n": len(forward_returns),
            "non_event_mean_pct": round(np.mean(ne) * 100, 4),
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_value), 4),
        }

    # Determine best horizon (lowest p-value with correct direction)
    best_horizon = None
    best_p = 1.0
    best_mean = 0.0

    for h_str, r in results.items():
        if r.get("mean_return_pct") is None:
            continue
        p = r.get("p_value", 1.0)
        mean = r.get("mean_return_pct", 0)
        if p < best_p:
            best_p = p
            best_horizon = h_str
            best_mean = mean

    # Gate criteria
    direction_correct = best_mean > 0  # should be positive for profitable strategy
    p_value_ok = best_p < 0.1
    effect_size_ok = abs(best_mean) > round_trip_cost * 100  # convert to pct

    gate_passed = direction_correct and p_value_ok and effect_size_ok
    reasons = []
    if not direction_correct:
        reasons.append(f"direction backwards (mean={best_mean:.4f}%)")
    if not p_value_ok:
        reasons.append(f"p={best_p:.4f} > 0.1")
    if not effect_size_ok:
        reasons.append(f"effect {abs(best_mean):.4f}% < costs {round_trip_cost*100:.2f}%")

    return {
        "events": len(event_indices),
        "horizons": results,
        "best_horizon": best_horizon,
        "best_p": best_p,
        "best_mean_pct": best_mean,
        "direction_correct": direction_correct,
        "effect_size_ok": effect_size_ok,
        "gate_passed": gate_passed,
        "gate_reason": "; ".join(reasons) if reasons else "PASS",
    }


def run_strategy_on_sets(strategy_name, sets_dir, output_file):
    """Run isolation gate for one strategy across all 20 synthetic sets."""
    if strategy_name not in STRATEGIES:
        print(f"Unknown strategy: {strategy_name}")
        print(f"Available: {', '.join(STRATEGIES.keys())}")
        return

    strat = STRATEGIES[strategy_name]
    detect_fn = strat["detect"]

    all_results = {}
    set_files = sorted([f for f in os.listdir(sets_dir) if f.startswith("synthetic_set_") and f.endswith(".csv")])

    print(f"\n{'='*70}")
    print(f"  SYNTHETIC ISOLATION GATE: {strategy_name}")
    print(f"  {strat['description']}")
    print(f"  Sets: {len(set_files)}")
    print(f"{'='*70}\n")

    pass_count = 0
    fail_count = 0
    direction_ok_count = 0

    for set_file in set_files:
        set_id = set_file.replace("synthetic_set_", "").replace(".csv", "")
        df = pd.read_csv(os.path.join(sets_dir, set_file))

        # Rename columns to expected format
        col_map = {"Open time": "timestamp", "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume", "Taker buy base asset volume": "taker_buy_volume"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Ensure numeric
        for col in ["open", "high", "low", "close", "volume", "taker_buy_volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Detect events
        events, direction = detect_fn(df)
        n_events = events.sum()

        # Run gate
        gate_result = run_isolation_gate(df, events, direction, strategy_name)
        all_results[f"set_{set_id}"] = gate_result

        status = "PASS" if gate_result["gate_passed"] else "FAIL"
        dir_ok = "DIR_OK" if gate_result.get("direction_correct") else "DIR_BAD"
        if gate_result["gate_passed"]:
            pass_count += 1
        else:
            fail_count += 1
        if gate_result.get("direction_correct"):
            direction_ok_count += 1

        events_str = f"n={gate_result['events']}"
        mean_str = f"mean={gate_result.get('best_mean_pct', 0):.4f}%"
        p_str = f"p={gate_result.get('best_p', 1):.4f}"
        reason = gate_result.get("gate_reason", "")[:40]

        print(f"  Set {set_id}: {status:4s} | {dir_ok:7s} | {events_str:8s} | {mean_str:16s} | {p_str:12s} | {reason}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {strategy_name}")
    print(f"  Sets passed: {pass_count}/{len(set_files)}")
    print(f"  Direction correct: {direction_ok_count}/{len(set_files)}")
    print(f"  Gate criterion: >= 14/20 correct direction AND mean p < 0.1")
    print(f"{'='*70}")

    # Determine overall result
    all_p_values = [r.get("best_p", 1.0) for r in all_results.values() if r.get("best_p") is not None]
    mean_p = np.mean(all_p_values) if all_p_values else 1.0

    if direction_ok_count >= 14 and mean_p < 0.1:
        overall = "SYNTHETIC GATE PASS"
    elif direction_ok_count <= 6 or mean_p > 0.15:
        overall = "SYNTHETIC GATE FAIL"
    else:
        overall = "BORDERLINE — needs real data validation"

    print(f"\n  Overall: {overall}")
    print(f"  Mean p-value: {mean_p:.4f}")
    print(f"  Direction correct: {direction_ok_count}/20")

    # Save results
    output = {
        "strategy": strategy_name,
        "description": strat["description"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sets_tested": len(set_files),
        "sets_passed": pass_count,
        "direction_correct": direction_ok_count,
        "mean_p_value": round(mean_p, 4),
        "overall_result": overall,
        "per_set": all_results,
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved: {output_file}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Run isolation gate on synthetic data")
    parser.add_argument("--strategy", default="all", help="Strategy name or 'all'")
    parser.add_argument("--sets", default="data/synthetic", help="Directory with synthetic CSV files")
    parser.add_argument("--output-dir", default="reports/synthetic_gates", help="Output directory for results")
    args = parser.parse_args()

    strategies = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]

    for strat_name in strategies:
        output_file = os.path.join(args.output_dir, f"{strat_name}_synthetic_gate.json")
        run_strategy_on_sets(strat_name, args.sets, output_file)
        print()


if __name__ == "__main__":
    main()
