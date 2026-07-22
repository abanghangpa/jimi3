#!/usr/bin/env python3
"""
Extended Isolation Gate Runner v2
Tests all 11 ungated strategies on extended synthetic data (with derivatives, L2, session, MTF).

Strategies to test:
1. scalp_v2 — tight TP scalping on momentum
2. power_of_3 — Wyckoff accumulation/distribution phases
3. macro_surprise — funding rate extreme mean reversion
4. liquidation_cascade — OI crash + price cascade detection
5. judas_sweep — liquidity sweep + reversal
6. taker_flow — taker ratio extreme directional
7. vol_rotation — volume regime shift
8. kill_zone — session timing filter (not a strategy, a filter)
9. liquidity_grab — bid/ask depth imbalance
10. cascade — OI + price cascade (extended)
11. mtf_confluence — multi-timeframe alignment
12. structural_break — price structure break
13. regime_switch — regime transition detection
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats


# === INDICATORS ===

def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, 1)))

def calc_atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def calc_bb(s, p=20, std=2.0):
    m = s.rolling(p).mean()
    sd = s.rolling(p).std()
    return m + std * sd, m, m - std * sd

def calc_vol_ratio(v, p=20):
    return v / v.rolling(p).mean().replace(0, 1)


# === DETECTION FUNCTIONS ===

def detect_scalp_v2(df):
    """Scalp v2: tight momentum entries on RSI + volume confirmation."""
    c = df["close"]
    rsi = calc_rsi(c, 7)  # fast RSI
    vol_ratio = calc_vol_ratio(df["volume"], 10)
    ema5 = calc_ema(c, 5)
    ema13 = calc_ema(c, 13)

    # LONG: RSI < 25 (oversold) + volume spike + EMA crossover starting
    long_events = (rsi < 25) & (vol_ratio > 1.3) & (c > ema5)
    # SHORT: RSI > 75 (overbought) + volume spike + EMA crossover starting
    short_events = (rsi > 75) & (vol_ratio > 1.3) & (c < ema5)

    events = long_events | short_events
    direction = pd.Series("NONE", index=df.index)
    direction[long_events] = "LONG"
    direction[short_events] = "SHORT"
    return events, direction


def detect_power_of_3(df):
    """Power of 3: Accumulation -> Markup -> Distribution -> Markdown phases."""
    c = df["close"]
    v = df["volume"]

    # Detect phases by volume + price structure
    ema20 = calc_ema(c, 20)
    vol_ma = v.rolling(20).mean()

    # Accumulation: low vol, price near EMA, tight range
    range_20 = (c.rolling(20).max() - c.rolling(20).min()) / c * 100
    accum = (range_20 < 2.0) & (v < vol_ma * 0.8) & ((c - ema20).abs() / c < 0.01)

    # Markup: price > EMA, expanding volume
    markup = (c > ema20) & (v > vol_ma * 1.2) & (c.pct_change(4) > 0.005)

    # Distribution: high vol, price near highs, range expanding
    dist = (c > c.rolling(50).quantile(0.9)) & (v > vol_ma * 1.5) & (range_20 > 3.0)

    # Markdown: price < EMA, expanding volume
    markdown = (c < ema20) & (v > vol_ma * 1.2) & (c.pct_change(4) < -0.005)

    # Events: transition from accum->markup (LONG) or dist->markdown (SHORT)
    long_events = markup & markup.shift(1).fillna(False).apply(lambda x: not x)  # just started markup
    short_events = markdown & markdown.shift(1).fillna(False).apply(lambda x: not x)

    events = long_events | short_events
    direction = pd.Series("NONE", index=df.index)
    direction[long_events] = "LONG"
    direction[short_events] = "SHORT"
    return events, direction


def detect_macro_surprise(df):
    """Macro surprise: funding rate extreme predicts mean reversion."""
    if "funding_rate" not in df.columns:
        return pd.Series(False, index=df.index), pd.Series("NONE", index=df.index)

    fr = df["funding_rate"]
    fr_ma = fr.rolling(48).mean()  # 12h MA
    fr_std = fr.rolling(48).std()
    fr_zscore = (fr - fr_ma) / fr_std.replace(0, 1)

    vol_ratio = calc_vol_ratio(df["volume"], 20)

    # Extreme negative funding + volume spike = LONG (shorts overcrowded)
    long_events = (fr_zscore < -2.0) & (vol_ratio > 1.0)
    # Extreme positive funding + volume spike = SHORT (longs overcrowded)
    short_events = (fr_zscore > 2.0) & (vol_ratio > 1.0)

    events = long_events | short_events
    direction = pd.Series("NONE", index=df.index)
    direction[long_events] = "LONG"
    direction[short_events] = "SHORT"
    return events, direction


def detect_liquidation_cascade(df):
    """Liquidation cascade: OI crash + sharp price move + volume spike."""
    if "open_interest" not in df.columns:
        return pd.Series(False, index=df.index), pd.Series("NONE", index=df.index)

    oi = df["open_interest"]
    c = df["close"]
    v = df["volume"]

    # OI dropping sharply (liquidations)
    oi_change = oi.pct_change(4)  # 1h change
    vol_ratio = calc_vol_ratio(v, 20)
    price_move = c.pct_change(4).abs()

    # Cascade: OI dropping > 5% + price moving sharply + volume spike
    cascade = (oi_change < -0.05) & (price_move > 0.01) & (vol_ratio > 1.5)

    # Direction from price
    direction = pd.Series("NONE", index=df.index)
    direction[cascade & (c.pct_change(4) < 0)] = "SHORT"  # price crashing = more downside
    direction[cascade & (c.pct_change(4) > 0)] = "LONG"  # price bouncing = recovery

    events = cascade & (direction != "NONE")
    return events, direction


def detect_judas_sweep(df):
    """Judas sweep: price sweeps a level then reverses (liquidity grab)."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # Recent swing high/low (10-bar)
    swing_h = h.rolling(10).max().shift(1)
    swing_l = l.rolling(10).min().shift(1)

    # Sweep high then close below = SHORT
    sweep_high = (h > swing_h) & (c < swing_h) & ((h - c) > (c - l) * 1.5)
    # Sweep low then close above = LONG
    sweep_low = (l < swing_l) & (c > swing_l) & ((c - l) > (h - c) * 1.5)

    events = sweep_high | sweep_low
    direction = pd.Series("NONE", index=df.index)
    direction[sweep_high] = "SHORT"
    direction[sweep_low] = "LONG"
    return events, direction


def detect_taker_flow(df):
    """Taker flow: extreme taker buy/sell ratio predicts direction."""
    if "taker_buy_ratio" not in df.columns:
        return pd.Series(False, index=df.index), pd.Series("NONE", index=df.index)

    tr = df["taker_buy_ratio"]
    tr_ma = tr.rolling(50).mean()
    tr_std = tr.rolling(50).std()
    tr_zscore = (tr - tr_ma) / tr_std.replace(0, 1)

    vol_ratio = calc_vol_ratio(df["volume"], 20)

    # Extreme taker buying + volume = LONG (momentum)
    long_events = (tr_zscore > 2.0) & (vol_ratio > 1.2)
    # Extreme taker selling + volume = SHORT
    short_events = (tr_zscore < -2.0) & (vol_ratio > 1.2)

    events = long_events | short_events
    direction = pd.Series("NONE", index=df.index)
    direction[long_events] = "LONG"
    direction[short_events] = "SHORT"
    return events, direction


def detect_vol_rotation(df):
    """Volume rotation: shift from low to high vol regime."""
    v = df["volume"]
    vol_ratio = calc_vol_ratio(v, 20)
    vol_ratio_ma = vol_ratio.rolling(10).mean()

    # Volume expanding from compression
    expanding = (vol_ratio > 1.5) & (vol_ratio_ma.shift(5) < 0.8)

    c = df["close"]
    ema20 = calc_ema(c, 20)

    direction = pd.Series("NONE", index=df.index)
    direction[expanding & (c > ema20)] = "LONG"
    direction[expanding & (c < ema20)] = "SHORT"

    events = expanding & (direction != "NONE")
    return events, direction


def detect_kill_zone(df):
    """Kill zone: session-based timing filter.
    NOT a strategy — this is a FILTER that should be combined with others.
    But we test it standalone to see if session timing alone has edge.
    """
    if "session" not in df.columns:
        return pd.Series(False, index=df.index), pd.Series("NONE", index=df.index)

    # Kill zones: London open (07-10 UTC), NY open (13-16 UTC)
    hour = df["hour_of_day"] if "hour_of_day" in df.columns else pd.Series(12, index=df.index)
    in_kill_zone = ((hour >= 7) & (hour <= 10)) | ((hour >= 13) & (hour <= 16))

    # Direction from recent price action
    c = df["close"]
    ema20 = calc_ema(c, 20)

    direction = pd.Series("NONE", index=df.index)
    direction[in_kill_zone & (c > ema20)] = "LONG"
    direction[in_kill_zone & (c < ema20)] = "SHORT"

    events = in_kill_zone & (direction != "NONE")
    return events, direction


def detect_liquidity_grab(df):
    """Liquidity grab: bid/ask imbalance + price reversal."""
    if "bid_ask_imbalance" not in df.columns:
        return pd.Series(False, index=df.index), pd.Series("NONE", index=df.index)

    imb = df["bid_ask_imbalance"]
    c = df["close"]
    vol_ratio = calc_vol_ratio(df["volume"], 20)

    # Extreme bid imbalance + volume = LONG (buying pressure)
    long_events = (imb > 0.3) & (vol_ratio > 1.2)
    # Extreme ask imbalance + volume = SHORT (selling pressure)
    short_events = (imb < -0.3) & (vol_ratio > 1.2)

    events = long_events | short_events
    direction = pd.Series("NONE", index=df.index)
    direction[long_events] = "LONG"
    direction[short_events] = "SHORT"
    return events, direction


def detect_cascade(df):
    """Cascade: OI + price + volume all moving together (extended liquidation)."""
    if "open_interest" not in df.columns:
        return pd.Series(False, index=df.index), pd.Series("NONE", index=df.index)

    oi = df["open_interest"]
    c = df["close"]
    v = df["volume"]

    oi_change = oi.pct_change(8)  # 2h change
    price_change = c.pct_change(8)
    vol_ratio = calc_vol_ratio(v, 20)

    # Cascade: OI + price + volume all moving same direction
    # Bullish cascade: OI up, price up, volume up
    bull_cascade = (oi_change > 0.03) & (price_change > 0.005) & (vol_ratio > 1.3)
    # Bearish cascade: OI down, price down, volume up (liquidations)
    bear_cascade = (oi_change < -0.03) & (price_change < -0.005) & (vol_ratio > 1.3)

    events = bull_cascade | bear_cascade
    direction = pd.Series("NONE", index=df.index)
    direction[bull_cascade] = "LONG"
    direction[bear_cascade] = "SHORT"
    return events, direction


def detect_mtf_confluence(df, df_1h=None):
    """Multi-timeframe confluence: 15m and 1h trends aligned.
    Uses 15m data with rolling 1h equivalent (4-bar EMA).
    """
    c = df["close"]

    # 15m trend
    ema15_12 = calc_ema(c, 12)
    ema15_26 = calc_ema(c, 26)
    trend_15m = ema15_12 > ema15_26

    # 1h trend (approximate with 4x longer EMAs)
    ema1h_12 = calc_ema(c, 48)  # 12 * 4
    ema1h_26 = calc_ema(c, 104)  # 26 * 4
    trend_1h = ema1h_12 > ema1h_26

    vol_ratio = calc_vol_ratio(df["volume"], 20)

    # Confluence: both timeframes aligned + volume
    long_events = trend_15m & trend_1h & (vol_ratio > 1.0) & (c > ema15_12)
    short_events = ~trend_15m & ~trend_1h & (vol_ratio > 1.0) & (c < ema15_12)

    # Only fire on fresh alignment (not sustained)
    long_fresh = long_events & ~long_events.shift(1).fillna(False)
    short_fresh = short_events & ~short_events.shift(1).fillna(False)

    events = long_fresh | short_fresh
    direction = pd.Series("NONE", index=df.index)
    direction[long_fresh] = "LONG"
    direction[short_fresh] = "SHORT"
    return events, direction


def detect_structural_break(df):
    """Structural break: break of key price structure (support/resistance)."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # Key levels: rolling 50-bar high/low
    resistance = h.rolling(50).max().shift(1)
    support = l.rolling(50).min().shift(1)

    vol_ratio = calc_vol_ratio(df["volume"], 20)

    # Break above resistance with volume = LONG
    break_up = (c > resistance) & (vol_ratio > 1.3)
    # Break below support with volume = SHORT
    break_down = (c < support) & (vol_ratio > 1.3)

    # Fresh breaks only
    break_up_fresh = break_up & ~break_up.shift(1).fillna(False)
    break_down_fresh = break_down & ~break_down.shift(1).fillna(False)

    events = break_up_fresh | break_down_fresh
    direction = pd.Series("NONE", index=df.index)
    direction[break_up_fresh] = "LONG"
    direction[break_down_fresh] = "SHORT"
    return events, direction


def detect_regime_switch(df):
    """Regime switch: transition between vol regimes."""
    c = df["close"]
    v = df["volume"]

    # Vol regime: ATR percentile
    atr = calc_atr(df["high"], df["low"], df["close"], 14)
    atr_pctl = atr.rolling(100, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)

    # Regime switch: vol going from low to high or high to low
    vol_expanding = (atr_pctl > 0.8) & (atr_pctl.shift(10) < 0.5)
    vol_contracting = (atr_pctl < 0.2) & (atr_pctl.shift(10) > 0.5)

    # Direction from price trend
    ema20 = calc_ema(c, 20)
    direction = pd.Series("NONE", index=df.index)
    direction[vol_expanding & (c > ema20)] = "LONG"
    direction[vol_expanding & (c < ema20)] = "SHORT"
    direction[vol_contracting] = "NONE"  # contracting vol = no direction

    events = vol_expanding & (direction != "NONE")
    return events, direction


# === STRATEGY REGISTRY ===

STRATEGIES = {
    "scalp_v2": {"detect": detect_scalp_v2, "desc": "Fast RSI + volume scalping"},
    "power_of_3": {"detect": detect_power_of_3, "desc": "Wyckoff accum/markup/dist/markdown"},
    "macro_surprise": {"detect": detect_macro_surprise, "desc": "Funding rate extreme mean reversion"},
    "liquidation_cascade": {"detect": detect_liquidation_cascade, "desc": "OI crash + price cascade"},
    "judas_sweep": {"detect": detect_judas_sweep, "desc": "Liquidity sweep + reversal"},
    "taker_flow": {"detect": detect_taker_flow, "desc": "Taker ratio extreme directional"},
    "vol_rotation": {"detect": detect_vol_rotation, "desc": "Volume regime shift"},
    "kill_zone": {"detect": detect_kill_zone, "desc": "Session timing filter"},
    "liquidity_grab": {"detect": detect_liquidity_grab, "desc": "Bid/ask imbalance + reversal"},
    "cascade": {"detect": detect_cascade, "desc": "OI + price + volume cascade"},
    "mtf_confluence": {"detect": detect_mtf_confluence, "desc": "Multi-timeframe alignment"},
    "structural_break": {"detect": detect_structural_break, "desc": "Support/resistance break"},
    "regime_switch": {"detect": detect_regime_switch, "desc": "Vol regime transition"},
}


# === ISOLATION GATE ===

def run_isolation_gate(df, events, direction, strategy_name, round_trip_cost=0.001):
    close = df["close"].values
    n = len(close)
    event_indices = np.where(events)[0]

    if len(event_indices) < 5:
        return {"events": len(event_indices), "gate_passed": False, "gate_reason": f"Too few ({len(event_indices)})"}

    horizons = [1, 4, 16, 24]
    results = {}

    for h in horizons:
        forward_returns = []
        for idx in event_indices:
            if idx + h < n:
                forward_returns.append((close[idx + h] - close[idx]) / close[idx])

        if len(forward_returns) < 3:
            results[f"{h}bar"] = {"mean_return_pct": None, "n": len(forward_returns)}
            continue

        fr = np.array(forward_returns)
        mean_ret = np.mean(fr)

        non_event_mask = np.ones(n, dtype=bool)
        non_event_mask[event_indices] = False
        non_event_indices = np.where(non_event_mask)[0]
        ne_returns = []
        for idx in non_event_indices:
            if idx + h < n:
                ne_returns.append((close[idx + h] - close[idx]) / close[idx])
        ne = np.array(ne_returns) if ne_returns else np.array([0.0])

        if len(fr) > 1 and len(ne) > 1:
            t_stat, p_value = stats.ttest_ind(fr, ne, equal_var=False)
        else:
            t_stat, p_value = 0.0, 1.0

        results[f"{h}bar"] = {
            "mean_return_pct": round(mean_ret * 100, 4),
            "n": len(forward_returns),
            "p_value": round(float(p_value), 4),
        }

    # Best horizon
    best_horizon, best_p, best_mean = None, 1.0, 0.0
    for h_str, r in results.items():
        if r.get("mean_return_pct") is None:
            continue
        if r["p_value"] < best_p:
            best_p = r["p_value"]
            best_horizon = h_str
            best_mean = r["mean_return_pct"]

    direction_correct = best_mean > 0
    p_ok = best_p < 0.1
    effect_ok = abs(best_mean) > round_trip_cost * 100

    gate_passed = direction_correct and p_ok and effect_ok
    reasons = []
    if not direction_correct:
        reasons.append(f"dir backwards ({best_mean:+.4f}%)")
    if not p_ok:
        reasons.append(f"p={best_p:.4f}")
    if not effect_ok:
        reasons.append(f"effect {abs(best_mean):.4f}% < costs")

    return {
        "events": len(event_indices),
        "horizons": results,
        "best_horizon": best_horizon,
        "best_p": best_p,
        "best_mean_pct": best_mean,
        "direction_correct": direction_correct,
        "gate_passed": gate_passed,
        "gate_reason": "; ".join(reasons) if reasons else "PASS",
    }


def run_strategy_on_sets(strategy_name, sets_dir, output_dir):
    strat = STRATEGIES[strategy_name]
    detect_fn = strat["detect"]

    set_files = sorted([f for f in os.listdir(sets_dir) if f.startswith("synthetic_set_") and f.endswith(".csv") and "_1h" not in f and "_4h" not in f])

    print(f"\n{'='*70}")
    print(f"  {strategy_name}: {strat['desc']}")
    print(f"  Sets: {len(set_files)}")
    print(f"{'='*70}\n")

    all_results = {}
    pass_count = 0
    direction_ok_count = 0

    for set_file in set_files:
        set_id = set_file.replace("synthetic_set_", "").replace(".csv", "")
        df = pd.read_csv(os.path.join(sets_dir, set_file))

        # Ensure numeric
        for col in ["open", "high", "low", "close", "volume", "taker_buy_volume",
                     "funding_rate", "open_interest", "bid_depth", "ask_depth",
                     "bid_ask_imbalance", "taker_buy_ratio", "hour_of_day"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        events, direction = detect_fn(df)
        gate_result = run_isolation_gate(df, events, direction, strategy_name)
        all_results[f"set_{set_id}"] = gate_result

        status = "PASS" if gate_result["gate_passed"] else "FAIL"
        dir_ok = "OK" if gate_result.get("direction_correct") else "BAD"
        if gate_result["gate_passed"]:
            pass_count += 1
        if gate_result.get("direction_correct"):
            direction_ok_count += 1

        print(f"  Set {set_id}: {status:4s} | {dir_ok:3s} | n={gate_result['events']:>5} | mean={gate_result.get('best_mean_pct',0):>+8.4f}% | p={gate_result.get('best_p',1):.4f} | {gate_result['gate_reason'][:35]}")

    # Summary
    all_p = [r.get("best_p", 1.0) for r in all_results.values() if r.get("best_p") is not None]
    mean_p = np.mean(all_p) if all_p else 1.0

    if direction_ok_count >= 14 and mean_p < 0.1:
        overall = "SYNTHETIC GATE PASS"
    elif direction_ok_count <= 6 or mean_p > 0.15:
        overall = "SYNTHETIC GATE FAIL"
    else:
        overall = "BORDERLINE"

    print(f"\n  SUMMARY: {pass_count}/{len(set_files)} passed | {direction_ok_count}/20 correct dir | mean p={mean_p:.4f}")
    print(f"  RESULT: {overall}")

    output = {
        "strategy": strategy_name,
        "description": strat["desc"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sets_tested": len(set_files),
        "sets_passed": pass_count,
        "direction_correct": direction_ok_count,
        "mean_p_value": round(mean_p, 4),
        "overall_result": overall,
        "per_set": all_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{strategy_name}_synthetic_gate.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all")
    parser.add_argument("--sets", default="data/synthetic_v2")
    parser.add_argument("--output-dir", default="reports/synthetic_gates_v2")
    args = parser.parse_args()

    strategies = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]

    print(f"{'='*70}")
    print(f"  EXTENDED SYNTHETIC ISOLATION GATE v2")
    print(f"  Strategies: {len(strategies)} | Sets: 20")
    print(f"  Data: OHLCV + funding_rate + OI + L2 + session + MTF")
    print(f"{'='*70}")

    summary = {}
    for strat_name in strategies:
        result = run_strategy_on_sets(strat_name, args.sets, args.output_dir)
        summary[strat_name] = result["overall_result"]

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    for name, result in summary.items():
        icon = "PASS" if "PASS" in result else "FAIL" if "FAIL" in result else "??"
        print(f"  [{icon:4s}] {name:25s} -> {result}")


if __name__ == "__main__":
    main()
