#!/usr/bin/env python3
"""
Extended Synthetic Data Generator — v2
Adds derivatives, L2, multi-timeframe, and session data to OHLCV.

New columns:
- funding_rate: mean-reverting, bounded, with spikes
- open_interest: correlated with price + volume
- bid_depth / ask_depth: L2 order book depth
- bid_ask_imbalance: (bid - ask) / (bid + ask)
- taker_buy_ratio: taker_buy_volume / volume
- hour_of_day: 0-23 (for session/kill-zone filtering)
- session: ASIA/EU/US/OVERLAP
- 1h OHLCV aggregated from 15m
- 4h OHLCV aggregated from 15m

Usage:
    python3 generate_synthetic_v2.py [--output-dir DIR] [--seed 42] [--bars 1500]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


# === REGIME DEFINITIONS (same as v1) ===
REGIMES = {
    "trending_bull": {"drift_range": (0.0002, 0.0008), "vol_range": (0.003, 0.008), "description": "Higher highs, shallow pullbacks"},
    "trending_bear": {"drift_range": (-0.0008, -0.0002), "vol_range": (0.003, 0.008), "description": "Lower lows, weak bounces"},
    "ranging": {"drift_range": (-0.0001, 0.0001), "vol_range": (0.002, 0.005), "description": "Mean-reverting, tight range", "mean_revert": True},
    "high_vol": {"drift_range": (-0.0005, 0.0005), "vol_range": (0.01, 0.025), "description": "Breakouts, wide bars, volume spikes"},
    "crisis": {"drift_range": (-0.003, 0.003), "vol_range": (0.02, 0.05), "description": "Gap-like moves, liquidation cascades"},
}

CONDITIONS = {
    "normal": {"spread_mult": 1.0, "vol_mult": 1.0, "description": "Baseline"},
    "thin_liquidity": {"spread_mult": 2.5, "vol_mult": 0.6, "description": "Wide spreads, erratic wicks"},
    "whale": {"spread_mult": 1.0, "vol_mult": 1.0, "whale_bars": (3, 6), "description": "Large single-bar moves"},
    "news": {"spread_mult": 1.2, "vol_mult": 1.3, "regime_transition": True, "description": "Gap moves, transitions"},
}

TRAPS = {
    "none": {"description": "Clean price action"},
    "bull_trap": {"description": "False breakout above resistance"},
    "bear_trap": {"description": "False breakdown below support"},
    "liquidation_cascade": {"description": "Chain of stop-hunts"},
}


# === PRICE GENERATION (reused from v1) ===

def generate_base_price(rng, n_bars, regime_params, condition_params):
    drift = rng.uniform(*regime_params["drift_range"])
    vol = rng.uniform(*regime_params["vol_range"]) * condition_params.get("vol_mult", 1.0)
    if regime_params.get("mean_revert"):
        theta = 0.05
        center = 100.0
        prices = [center]
        for i in range(1, n_bars):
            noise = rng.normal(0, vol)
            dp = theta * (center - prices[-1]) + noise
            prices.append(max(prices[-1] + dp, prices[-1] * 0.95))
        return np.array(prices)
    else:
        returns = rng.normal(drift, vol, n_bars)
        return 100.0 * np.exp(np.cumsum(returns))


def add_whale_moves(rng, prices, n_whale_bars):
    n = len(prices)
    whale_indices = rng.choice(range(50, n - 50), size=n_whale_bars, replace=False)
    for idx in whale_indices:
        direction = rng.choice([-1, 1])
        magnitude = rng.uniform(0.02, 0.05)
        shift = prices[idx] * magnitude * direction
        prices[idx:] += shift
    return prices, whale_indices


def add_regime_transition(rng, prices, n_bars):
    transition_bar = rng.integers(n_bars // 3, 2 * n_bars // 3)
    direction = rng.choice([-1, 1])
    magnitude = rng.uniform(0.001, 0.003)
    for i in range(transition_bar, n_bars):
        prices[i] *= (1 + direction * magnitude) ** (i - transition_bar)
    return prices, transition_bar


def inject_trap(rng, prices, trap_type, n_bars):
    trap_events = []
    if trap_type == "none":
        return prices, trap_events
    trap_start = rng.integers(n_bars // 4, 3 * n_bars // 4)
    if trap_type == "bull_trap":
        resistance = prices[trap_start] * 1.005
        trap_len = rng.integers(3, 6)
        for i in range(trap_len):
            if trap_start + i < n_bars:
                prices[trap_start + i] = resistance * (1 + rng.uniform(0.001, 0.003))
        for i in range(trap_len, trap_len + 8):
            if trap_start + i < n_bars:
                drop = (i - trap_len) * rng.uniform(0.002, 0.004)
                prices[trap_start + i] = resistance * (1 - drop)
        trap_events.append({"type": "bull_trap", "bar": int(trap_start)})
    elif trap_type == "bear_trap":
        support = prices[trap_start] * 0.995
        trap_len = rng.integers(3, 6)
        for i in range(trap_len):
            if trap_start + i < n_bars:
                prices[trap_start + i] = support * (1 - rng.uniform(0.001, 0.003))
        for i in range(trap_len, trap_len + 8):
            if trap_start + i < n_bars:
                bounce = (i - trap_len) * rng.uniform(0.002, 0.004)
                prices[trap_start + i] = support * (1 + bounce)
        trap_events.append({"type": "bear_trap", "bar": int(trap_start)})
    elif trap_type == "liquidation_cascade":
        cascade_len = rng.integers(8, 15)
        direction = rng.choice([-1, 1])
        for i in range(cascade_len):
            if trap_start + i < n_bars:
                accel = (i + 1) ** 1.5 * rng.uniform(0.001, 0.002)
                prices[trap_start + i] *= (1 + direction * accel)
        for i in range(cascade_len, cascade_len + 5):
            if trap_start + i < n_bars:
                prices[trap_start + i] *= (1 - direction * 0.003)
        trap_events.append({"type": "liquidation_cascade", "bar": int(trap_start), "direction": "down" if direction < 0 else "up"})
    return prices, trap_events


# === NEW: DERIVATIVES GENERATION ===

def generate_funding_rate(rng, prices, n_bars, regime):
    """Generate realistic funding rates.
    - Mean-reverting around 0.01% (8h rate)
    - Spikes during high vol / crisis
    - Correlated with price trend (bullish = positive funding)
    """
    base_rate = 0.0001  # 0.01% per 8h = baseline
    funding = np.zeros(n_bars)

    # Compute price returns for correlation
    returns = np.diff(np.log(prices), prepend=np.log(prices[0]))

    for i in range(n_bars):
        # Base: mean-revert to 0.01%
        mean_rev = 0.1 * (base_rate - funding[max(0, i-1)])
        # Trend component: positive when price rising
        trend = returns[i] * 0.5
        # Noise
        noise = rng.normal(0, 0.00005)

        # Spike during crisis/high vol
        if regime == "crisis":
            spike_prob = 0.02
        elif regime == "high_vol":
            spike_prob = 0.01
        else:
            spike_prob = 0.003

        spike = 0
        if rng.random() < spike_prob:
            spike = rng.choice([-1, 1]) * rng.uniform(0.0003, 0.001)  # 0.03% to 0.1%

        funding[i] = funding[max(0, i-1)] + mean_rev + trend + noise + spike
        funding[i] = np.clip(funding[i], -0.003, 0.003)  # -0.3% to +0.3% bounds

    return funding


def generate_open_interest(rng, prices, volume, n_bars, regime):
    """Generate open interest.
    - Correlated with volume (more volume = more OI)
    - Drops during price crashes (liquidations)
    - Rises during trending markets
    """
    # Base OI from volume
    vol_normalized = volume / np.mean(volume)
    oi_base = vol_normalized * 1e6  # scale to ~1M contracts

    # Add trend component
    returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
    oi_trend = np.cumsum(returns) * 1e5  # OI grows with sustained moves

    # Drop OI during sharp moves (liquidations)
    oi = oi_base + oi_trend
    for i in range(1, n_bars):
        if abs(returns[i]) > 0.01:  # 1% move in one bar
            oi[i] *= 0.9  # 10% OI drop (liquidation)
        # Mean revert
        oi[i] = oi[i] * 0.99 + np.mean(oi[max(0, i-20):i]) * 0.01

    # Add noise
    oi += rng.normal(0, np.mean(oi) * 0.02, n_bars)
    oi = np.maximum(oi, 1e4)  # minimum OI

    return oi


def generate_orderbook_depth(rng, prices, volume, n_bars, condition):
    """Generate L2 order book depth.
    - Bid/ask depth correlated with volume
    - Imbalance indicates direction pressure
    - Thin liquidity = wider spreads, lower depth
    """
    spread_mult = CONDITIONS.get(condition, {}).get("spread_mult", 1.0)

    # Base depth from volume
    vol_normalized = volume / np.mean(volume)

    # Bid depth: resting buy orders
    bid_depth = vol_normalized * rng.uniform(0.8, 1.2, n_bars) * 1000
    # Ask depth: resting sell orders
    ask_depth = vol_normalized * rng.uniform(0.8, 1.2, n_bars) * 1000

    # Imbalance: random with occasional directional bias
    # During trends, one side is heavier
    returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
    for i in range(n_bars):
        if returns[i] > 0.002:  # price rising -> more bid support
            bid_depth[i] *= 1.3
            ask_depth[i] *= 0.8
        elif returns[i] < -0.002:  # price falling -> more ask support
            ask_depth[i] *= 1.3
            bid_depth[i] *= 0.8

    # Thin liquidity: reduce depth
    if spread_mult > 1.5:
        bid_depth *= 0.5
        ask_depth *= 0.5

    # Add noise
    bid_depth += rng.normal(0, bid_depth * 0.1)
    ask_depth += rng.normal(0, ask_depth * 0.1)
    bid_depth = np.maximum(bid_depth, 10)
    ask_depth = np.maximum(ask_depth, 10)

    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)

    return bid_depth, ask_depth, imbalance


# === SESSION / TIMING ===

def generate_session_data(n_bars, base_time):
    """Generate session labels and hour-of-day for each bar."""
    hours = []
    sessions = []
    for i in range(n_bars):
        t = base_time + timedelta(minutes=15 * i)
        h = t.hour
        hours.append(h)
        # Session classification (UTC)
        if 0 <= h < 7:
            sessions.append("ASIA")
        elif 7 <= h < 13:
            sessions.append("EU")
        elif 13 <= h < 21:
            sessions.append("US")
        else:
            sessions.append("OVERLAP")  # Asia/EU overlap
    return hours, sessions


# === MULTI-TIMEFRAME AGGREGATION ===

def aggregate_timeframe(df, tf_minutes):
    """Aggregate 15m bars into higher timeframe."""
    bars_per_tf = tf_minutes // 15
    n = len(df)
    agg_data = []

    for i in range(0, n, bars_per_tf):
        chunk = df.iloc[i:i+bars_per_tf]
        if len(chunk) == 0:
            continue
        agg_data.append({
            "timestamp": chunk.iloc[0]["timestamp"],
            "open": chunk.iloc[0]["open"],
            "high": chunk["high"].max(),
            "low": chunk["low"].min(),
            "close": chunk.iloc[-1]["close"],
            "volume": chunk["volume"].sum(),
            "taker_buy_volume": chunk["taker_buy_volume"].sum(),
        })

    return pd.DataFrame(agg_data)


# === OHLCV CONSTRUCTION ===

def prices_to_ohlcv(rng, prices, spread_mult=1.0):
    n = len(prices)
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)
    volumes = np.zeros(n)
    taker_buy_vols = np.zeros(n)

    for i in range(n):
        close = prices[i]
        if i == 0:
            open_price = close
        else:
            gap = rng.normal(0, 0.001) * close
            open_price = closes[i-1] + gap

        wick_up = rng.exponential(0.002) * close * spread_mult
        wick_down = rng.exponential(0.002) * close * spread_mult
        high = max(open_price, close) + wick_up
        low = min(open_price, close) - wick_down
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        opens[i] = open_price
        highs[i] = high
        lows[i] = low
        closes[i] = close

        base_vol = rng.lognormal(mean=10, sigma=1)
        volumes[i] = base_vol
        taker_buy_vols[i] = base_vol * rng.uniform(0.35, 0.65)

    return opens, highs, lows, closes, volumes, taker_buy_vols


# === MAIN GENERATION ===

def generate_set_v2(set_id, regime, condition, trap, n_bars, rng):
    regime_params = REGIMES[regime]
    condition_params = CONDITIONS[condition]

    # 1. Price
    prices = generate_base_price(rng, n_bars, regime_params, condition_params)

    # 2. Condition overlays
    whale_indices = []
    transition_bar = None
    if condition == "whale":
        n_whale = rng.integers(*condition_params["whale_bars"])
        prices, whale_indices = add_whale_moves(rng, prices, n_whale)
    elif condition == "news":
        prices, transition_bar = add_regime_transition(rng, prices, n_bars)

    # 3. Traps
    prices, trap_events = inject_trap(rng, prices, trap, n_bars)

    # 4. OHLCV
    spread_mult = condition_params.get("spread_mult", 1.0)
    opens, highs, lows, closes, volumes, taker_buy_vols = prices_to_ohlcv(rng, prices, spread_mult)

    # 5. NEW: Derivatives
    funding_rate = generate_funding_rate(rng, prices, n_bars, regime)
    open_interest = generate_open_interest(rng, prices, volumes, n_bars, regime)
    bid_depth, ask_depth, imbalance = generate_orderbook_depth(rng, prices, volumes, n_bars, condition)

    # 6. Session data
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    hours, sessions = generate_session_data(n_bars, base_time)

    # 7. Taker ratio
    taker_ratio = taker_buy_vols / np.maximum(volumes, 1)

    # 8. Build DataFrame
    timestamps = [base_time + timedelta(minutes=15*i) for i in range(n_bars)]
    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in timestamps],
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes, "taker_buy_volume": taker_buy_vols,
        "taker_buy_ratio": taker_ratio,
        "funding_rate": funding_rate,
        "open_interest": open_interest,
        "bid_depth": bid_depth, "ask_depth": ask_depth,
        "bid_ask_imbalance": imbalance,
        "hour_of_day": hours,
        "session": sessions,
    })

    # 9. Multi-timeframe
    df_1h = aggregate_timeframe(df, 60)
    df_4h = aggregate_timeframe(df, 240)

    # 10. Metadata
    metadata = {
        "set_id": set_id, "regime": regime, "condition": condition, "trap": trap,
        "n_bars": n_bars, "seed": int(rng.bit_generator._seed_seq.entropy),
        "regime_description": regime_params["description"],
        "condition_description": condition_params["description"],
        "trap_description": TRAPS[trap]["description"],
        "whale_bars": [int(x) for x in whale_indices],
        "transition_bar": int(transition_bar) if transition_bar else None,
        "trap_events": trap_events,
        "columns": list(df.columns),
        "derived_timeframes": ["1h", "4h"],
        "funding_rate_stats": {
            "mean": float(np.mean(funding_rate)),
            "std": float(np.std(funding_rate)),
            "min": float(np.min(funding_rate)),
            "max": float(np.max(funding_rate)),
        },
        "oi_stats": {
            "mean": float(np.mean(open_interest)),
            "std": float(np.std(open_interest)),
        },
        "imbalance_stats": {
            "mean": float(np.mean(imbalance)),
            "std": float(np.std(imbalance)),
        },
    }

    return df, df_1h, df_4h, metadata


# === THE 20 SETS ===
SET_DEFINITIONS = [
    {"regime": "trending_bull", "condition": "normal", "trap": "none"},
    {"regime": "trending_bear", "condition": "normal", "trap": "none"},
    {"regime": "ranging", "condition": "normal", "trap": "none"},
    {"regime": "high_vol", "condition": "normal", "trap": "none"},
    {"regime": "crisis", "condition": "normal", "trap": "none"},
    {"regime": "ranging", "condition": "thin_liquidity", "trap": "none"},
    {"regime": "ranging", "condition": "whale", "trap": "none"},
    {"regime": "ranging", "condition": "news", "trap": "none"},
    {"regime": "trending_bull", "condition": "normal", "trap": "bull_trap"},
    {"regime": "trending_bull", "condition": "normal", "trap": "bear_trap"},
    {"regime": "trending_bull", "condition": "normal", "trap": "liquidation_cascade"},
    {"regime": "trending_bear", "condition": "normal", "trap": "bull_trap"},
    {"regime": "trending_bear", "condition": "normal", "trap": "bear_trap"},
    {"regime": "trending_bear", "condition": "normal", "trap": "liquidation_cascade"},
    {"regime": "high_vol", "condition": "whale", "trap": "liquidation_cascade"},
    {"regime": "crisis", "condition": "thin_liquidity", "trap": "bear_trap"},
    {"regime": "trending_bull", "condition": "news", "trap": "bull_trap"},
    {"regime": "ranging", "condition": "whale", "trap": "bull_trap"},
    {"regime": "trending_bear", "condition": "news", "trap": "liquidation_cascade"},
    {"regime": "high_vol", "condition": "thin_liquidity", "trap": "none"},
]


def main():
    parser = argparse.ArgumentParser(description="Extended synthetic data generator v2")
    parser.add_argument("--output-dir", default="data/synthetic_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bars", type=int, default=1500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    master_rng = np.random.default_rng(args.seed)
    all_metadata = []

    print(f"Generating 20 extended synthetic datasets ({args.bars} bars each)")
    print(f"Output: {args.output_dir}/")
    print(f"Columns: OHLCV + funding_rate + OI + L2 depth + session + taker_ratio")
    print()

    for i, set_def in enumerate(SET_DEFINITIONS):
        set_seed = int(master_rng.integers(0, 2**31))
        rng = np.random.default_rng(set_seed)

        df, df_1h, df_4h, metadata = generate_set_v2(
            set_id=i, regime=set_def["regime"], condition=set_def["condition"],
            trap=set_def["trap"], n_bars=args.bars, rng=rng,
        )

        # Save
        df.to_csv(os.path.join(args.output_dir, f"synthetic_set_{i:02d}.csv"), index=False)
        df_1h.to_csv(os.path.join(args.output_dir, f"synthetic_set_{i:02d}_1h.csv"), index=False)
        df_4h.to_csv(os.path.join(args.output_dir, f"synthetic_set_{i:02d}_4h.csv"), index=False)
        all_metadata.append(metadata)

        fr_stats = metadata["funding_rate_stats"]
        print(f"  Set {i:02d}: {set_def['regime']:15s} + {set_def['condition']:15s} + {set_def['trap']:20s} | FR mean={fr_stats['mean']:+.5f} | OI mean={metadata['oi_stats']['mean']:.0f}")

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "master_seed": args.seed, "bars_per_set": args.bars, "version": "v2", "columns": "OHLCV+funding_rate+OI+L2_depth+session+taker_ratio", "sets": all_metadata}, f, indent=2, default=str)

    print(f"\nDone. {len(all_metadata)} sets generated with extended data.")
    print(f"15m: {args.output_dir}/synthetic_set_XX.csv")
    print(f"1h:  {args.output_dir}/synthetic_set_XX_1h.csv")
    print(f"4h:  {args.output_dir}/synthetic_set_XX_4h.csv")


if __name__ == "__main__":
    main()
