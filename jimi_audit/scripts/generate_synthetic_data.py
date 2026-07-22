#!/usr/bin/env python3
"""
Synthetic Data Generator — 20-Set Framework
Per BACKTEST_FRAMEWORK.md: Synthetic Data Protocol

Generates 20 OHLCV datasets representing various regimes, market conditions, and trap scenarios.
Each dataset: 1500 bars of 15m data with realistic OHLCV + taker_buy_volume.

Usage:
    python3 generate_synthetic_data.py [--output-dir DIR] [--seed SEED] [--bars N]

Output:
    20 CSV files: synthetic_set_00.csv through synthetic_set_19.csv
    metadata.json: parameters and regime labels for each set
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


# === REGIME DEFINITIONS ===
REGIMES = {
    "trending_bull": {
        "drift_range": (0.0002, 0.0008),
        "vol_range": (0.003, 0.008),
        "description": "Higher highs, shallow pullbacks",
    },
    "trending_bear": {
        "drift_range": (-0.0008, -0.0002),
        "vol_range": (0.003, 0.008),
        "description": "Lower lows, weak bounces",
    },
    "ranging": {
        "drift_range": (-0.0001, 0.0001),
        "vol_range": (0.002, 0.005),
        "description": "Mean-reverting, tight range",
        "mean_revert": True,
    },
    "high_vol": {
        "drift_range": (-0.0005, 0.0005),
        "vol_range": (0.01, 0.025),
        "description": "Breakouts, wide bars, volume spikes",
    },
    "crisis": {
        "drift_range": (-0.003, 0.003),
        "vol_range": (0.02, 0.05),
        "description": "Gap-like moves, liquidation cascades",
    },
}

# === MARKET CONDITION OVERLAYS ===
CONDITIONS = {
    "normal": {
        "spread_mult": 1.0,
        "vol_mult": 1.0,
        "description": "Baseline spread, normal volume",
    },
    "thin_liquidity": {
        "spread_mult": 2.5,
        "vol_mult": 0.6,
        "description": "Wide spreads, erratic wicks",
    },
    "whale": {
        "spread_mult": 1.0,
        "vol_mult": 1.0,
        "whale_bars": (3, 6),
        "description": "Large single-bar moves, volume spikes",
    },
    "news": {
        "spread_mult": 1.2,
        "vol_mult": 1.3,
        "regime_transition": True,
        "description": "Gap moves, regime transitions",
    },
}

# === TRAP SCENARIOS ===
TRAPS = {
    "none": {"description": "Clean price action — baseline"},
    "bull_trap": {"description": "False breakout above resistance, sharp reversal"},
    "bear_trap": {"description": "False breakdown below support, sharp reversal"},
    "liquidation_cascade": {"description": "Chain of stop-hunts cascading through levels"},
}


def generate_base_price(rng, n_bars, regime_params, condition_params):
    """Generate base log-returns using GBM with regime parameters."""
    drift = rng.uniform(*regime_params["drift_range"])
    vol = rng.uniform(*regime_params["vol_range"]) * condition_params.get("vol_mult", 1.0)

    # For ranging regime, add mean reversion
    if regime_params.get("mean_revert"):
        # Ornstein-Uhlenbeck process
        theta = 0.05  # mean reversion speed
        center = 100.0
        prices = [center]
        for i in range(1, n_bars):
            noise = rng.normal(0, vol)
            dp = theta * (center - prices[-1]) + noise
            prices.append(max(prices[-1] + dp, prices[-1] * 0.95))
        return np.array(prices)
    else:
        # Standard GBM
        returns = rng.normal(drift, vol, n_bars)
        prices = 100.0 * np.exp(np.cumsum(returns))
        return prices


def add_whale_moves(rng, prices, n_whale_bars):
    """Inject large single-bar moves (whale activity)."""
    n = len(prices)
    whale_indices = rng.choice(range(50, n - 50), size=n_whale_bars, replace=False)
    for idx in whale_indices:
        direction = rng.choice([-1, 1])
        magnitude = rng.uniform(0.02, 0.05)  # 2-5% move
        # Apply to price series
        shift = prices[idx] * magnitude * direction
        prices[idx:] += shift
    return prices, whale_indices


def add_regime_transition(rng, prices, n_bars):
    """Inject a regime transition mid-dataset (news/event driven)."""
    transition_bar = rng.integers(n_bars // 3, 2 * n_bars // 3)
    # Reverse drift direction at transition
    direction = rng.choice([-1, 1])
    magnitude = rng.uniform(0.001, 0.003)
    for i in range(transition_bar, n_bars):
        prices[i] *= (1 + direction * magnitude) ** (i - transition_bar)
    return prices, transition_bar


def inject_trap(rng, prices, trap_type, n_bars):
    """Inject trap scenarios at random points."""
    trap_events = []

    if trap_type == "none":
        return prices, trap_events

    # Find a suitable location for the trap (not at edges)
    trap_start = rng.integers(n_bars // 4, 3 * n_bars // 4)

    if trap_type == "bull_trap":
        # Create resistance level, push above it, then reverse
        resistance = prices[trap_start] * 1.005  # 0.5% above current
        trap_len = rng.integers(3, 6)
        for i in range(trap_len):
            if trap_start + i < n_bars:
                prices[trap_start + i] = resistance * (1 + rng.uniform(0.001, 0.003))
        # Sharp reversal
        for i in range(trap_len, trap_len + 8):
            if trap_start + i < n_bars:
                drop = (i - trap_len) * rng.uniform(0.002, 0.004)
                prices[trap_start + i] = resistance * (1 - drop)
        trap_events.append({
            "type": "bull_trap",
            "bar": int(trap_start),
            "resistance": float(resistance),
            "trap_bars": int(trap_len),
        })

    elif trap_type == "bear_trap":
        # Create support level, push below it, then reverse
        support = prices[trap_start] * 0.995
        trap_len = rng.integers(3, 6)
        for i in range(trap_len):
            if trap_start + i < n_bars:
                prices[trap_start + i] = support * (1 - rng.uniform(0.001, 0.003))
        # Sharp reversal
        for i in range(trap_len, trap_len + 8):
            if trap_start + i < n_bars:
                bounce = (i - trap_len) * rng.uniform(0.002, 0.004)
                prices[trap_start + i] = support * (1 + bounce)
        trap_events.append({
            "type": "bear_trap",
            "bar": int(trap_start),
            "support": float(support),
            "trap_bars": int(trap_len),
        })

    elif trap_type == "liquidation_cascade":
        # Chain of accelerating moves through levels
        cascade_len = rng.integers(8, 15)
        direction = rng.choice([-1, 1])
        for i in range(cascade_len):
            if trap_start + i < n_bars:
                accel = (i + 1) ** 1.5 * rng.uniform(0.001, 0.002)
                prices[trap_start + i] *= (1 + direction * accel)
        # Recovery bounce
        for i in range(cascade_len, cascade_len + 5):
            if trap_start + i < n_bars:
                prices[trap_start + i] *= (1 - direction * 0.003)
        trap_events.append({
            "type": "liquidation_cascade",
            "bar": int(trap_start),
            "direction": "down" if direction < 0 else "up",
            "cascade_bars": int(cascade_len),
        })

    return prices, trap_events


def prices_to_ohlcv(rng, prices, spread_mult=1.0):
    """Convert price series to realistic OHLCV bars."""
    n = len(prices)
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)
    volumes = np.zeros(n)
    taker_buy_vols = np.zeros(n)

    for i in range(n):
        close = prices[i]
        # Generate open with small gap from previous close
        if i == 0:
            open_price = close
        else:
            gap = rng.normal(0, 0.001) * close
            open_price = closes[i - 1] + gap

        # Generate high/low with realistic wicks
        bar_range = abs(close - open_price) + rng.exponential(0.003) * close * spread_mult
        wick_up = rng.exponential(0.002) * close * spread_mult
        wick_down = rng.exponential(0.002) * close * spread_mult

        high = max(open_price, close) + wick_up
        low = min(open_price, close) - wick_down

        # Ensure OHLCV consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        opens[i] = open_price
        highs[i] = high
        lows[i] = low
        closes[i] = close

        # Volume: base volume with some noise
        base_vol = rng.lognormal(mean=10, sigma=1)
        volumes[i] = base_vol
        # Taker buy: roughly 40-60% of total volume
        taker_buy_ratio = rng.uniform(0.35, 0.65)
        taker_buy_vols[i] = base_vol * taker_buy_ratio

    return opens, highs, lows, closes, volumes, taker_buy_vols


def generate_set(set_id, regime, condition, trap, n_bars, rng):
    """Generate one synthetic dataset."""
    regime_params = REGIMES[regime]
    condition_params = CONDITIONS[condition]
    trap_params = TRAPS[trap]

    # 1. Base price process
    prices = generate_base_price(rng, n_bars, regime_params, condition_params)

    # 2. Condition overlays
    whale_indices = []
    transition_bar = None

    if condition == "whale":
        n_whale = rng.integers(*condition_params["whale_bars"])
        prices, whale_indices = add_whale_moves(rng, prices, n_whale)
    elif condition == "news":
        prices, transition_bar = add_regime_transition(rng, prices, n_bars)

    # 3. Trap injection
    prices, trap_events = inject_trap(rng, prices, trap, n_bars)

    # 4. Generate OHLCV
    spread_mult = condition_params.get("spread_mult", 1.0)
    opens, highs, lows, closes, volumes, taker_buy_vols = prices_to_ohlcv(rng, prices, spread_mult)

    # 5. Generate timestamps (15m intervals)
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(minutes=15 * i) for i in range(n_bars)]

    # 6. Build DataFrame
    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in timestamps],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "taker_buy_volume": taker_buy_vols,
    })

    # 7. Metadata
    metadata = {
        "set_id": set_id,
        "regime": regime,
        "condition": condition,
        "trap": trap,
        "n_bars": n_bars,
        "seed": int(rng.bit_generator._seed_seq.entropy),
        "regime_description": regime_params["description"],
        "condition_description": condition_params["description"],
        "trap_description": trap_params["description"],
        "whale_bars": [int(x) for x in whale_indices],
        "transition_bar": int(transition_bar) if transition_bar else None,
        "trap_events": trap_events,
        "price_stats": {
            "start": float(closes[0]),
            "end": float(closes[-1]),
            "return_pct": float((closes[-1] / closes[0] - 1) * 100),
            "max": float(np.max(closes)),
            "min": float(np.min(closes)),
        },
        "sanity_checks": {
            "mean_return_pct": float(np.mean(np.diff(np.log(closes))) * 100),
            "vol_pct": float(np.std(np.diff(np.log(closes))) * 100),
            "autocorr_1": float(np.corrcoef(np.diff(np.log(closes))[:-1], np.diff(np.log(closes))[1:])[0, 1]),
        },
    }

    return df, metadata


# === THE 20 SETS ===
SET_DEFINITIONS = [
    # Set 0-4: Each regime with normal condition, no trap (baseline)
    {"regime": "trending_bull", "condition": "normal", "trap": "none"},
    {"regime": "trending_bear", "condition": "normal", "trap": "none"},
    {"regime": "ranging", "condition": "normal", "trap": "none"},
    {"regime": "high_vol", "condition": "normal", "trap": "none"},
    {"regime": "crisis", "condition": "normal", "trap": "none"},
    # Set 5-8: Ranging regime with each condition, no trap
    {"regime": "ranging", "condition": "thin_liquidity", "trap": "none"},
    {"regime": "ranging", "condition": "whale", "trap": "none"},
    {"regime": "ranging", "condition": "news", "trap": "none"},
    # Set 9-11: Each trap on trending bull
    {"regime": "trending_bull", "condition": "normal", "trap": "bull_trap"},
    {"regime": "trending_bull", "condition": "normal", "trap": "bear_trap"},
    {"regime": "trending_bull", "condition": "normal", "trap": "liquidation_cascade"},
    # Set 12-14: Each trap on trending bear
    {"regime": "trending_bear", "condition": "normal", "trap": "bull_trap"},
    {"regime": "trending_bear", "condition": "normal", "trap": "bear_trap"},
    {"regime": "trending_bear", "condition": "normal", "trap": "liquidation_cascade"},
    # Set 15-17: Regime + condition combos
    {"regime": "high_vol", "condition": "whale", "trap": "liquidation_cascade"},
    {"regime": "crisis", "condition": "thin_liquidity", "trap": "bear_trap"},
    {"regime": "trending_bull", "condition": "news", "trap": "bull_trap"},
    # Set 18-19: Complex combos
    {"regime": "ranging", "condition": "whale", "trap": "bull_trap"},
    {"regime": "trending_bear", "condition": "news", "trap": "liquidation_cascade"},
    {"regime": "high_vol", "condition": "thin_liquidity", "trap": "none"},
]


def validate_set(df, metadata, real_eth_stats=None):
    """Validate a synthetic set against sanity checks."""
    issues = []
    stats = metadata["sanity_checks"]

    # Check mean return within 2x of real ETH (if provided)
    if real_eth_stats:
        if abs(stats["mean_return_pct"]) > 2 * abs(real_eth_stats["mean_return_pct"]):
            issues.append(f"Mean return {stats['mean_return_pct']:.4f}% > 2x real {real_eth_stats['mean_return_pct']:.4f}%")

    # Check volatility is reasonable
    if stats["vol_pct"] < 0.01:
        issues.append(f"Volatility too low: {stats['vol_pct']:.4f}%")
    if stats["vol_pct"] > 10:
        issues.append(f"Volatility too high: {stats['vol_pct']:.4f}%")

    # Check OHLCV consistency
    violations = 0
    for i in range(len(df)):
        if df.iloc[i]["high"] < max(df.iloc[i]["open"], df.iloc[i]["close"]):
            violations += 1
        if df.iloc[i]["low"] > min(df.iloc[i]["open"], df.iloc[i]["close"]):
            violations += 1
    if violations > 0:
        issues.append(f"OHLCV consistency violations: {violations}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Generate 20 synthetic OHLCV datasets")
    parser.add_argument("--output-dir", default="data/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Master seed for reproducibility")
    parser.add_argument("--bars", type=int, default=1500, help="Bars per dataset (15m)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    master_rng = np.random.default_rng(args.seed)
    all_metadata = []

    print(f"Generating 20 synthetic datasets ({args.bars} bars each)")
    print(f"Output: {args.output_dir}/")
    print()

    for i, set_def in enumerate(SET_DEFINITIONS):
        set_seed = int(master_rng.integers(0, 2**31))
        rng = np.random.default_rng(set_seed)

        df, metadata = generate_set(
            set_id=i,
            regime=set_def["regime"],
            condition=set_def["condition"],
            trap=set_def["trap"],
            n_bars=args.bars,
            rng=rng,
        )

        # Validate
        issues = validate_set(df, metadata)
        metadata["validation_issues"] = issues

        # Save
        filename = f"synthetic_set_{i:02d}.csv"
        df.to_csv(os.path.join(args.output_dir, filename), index=False)
        all_metadata.append(metadata)

        status = "OK" if not issues else f"WARN: {'; '.join(issues)}"
        print(f"  Set {i:02d}: {set_def['regime']:15s} + {set_def['condition']:15s} + {set_def['trap']:20s} -> {filename} [{status}]")

    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "master_seed": args.seed,
            "bars_per_set": args.bars,
            "sets": all_metadata,
        }, f, indent=2)

    print()
    print(f"Done. {len(all_metadata)} sets generated.")
    print(f"Metadata: {meta_path}")
    print()
    print("Next steps:")
    print("  1. Run isolation gate on each set")
    print("  2. Record pass/fail per set")
    print(f"  3. Gate passes if >= 14/20 sets have correct direction AND mean p < 0.1")


if __name__ == "__main__":
    main()
