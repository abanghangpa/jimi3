#!/usr/bin/env python3
"""
JIMI Outcome Tracker — CLI Entry Point
=======================================

Matches fired signals against actual OHLCV price data to determine
real trade outcomes (TP hit, SL hit, timeout).

Usage:
    python run_outcome_tracker.py [options]

Examples:
    python run_outcome_tracker.py
    python run_outcome_tracker.py --strategy trade_flow
    python run_outcome_tracker.py --ohlcv data/eth_15m_extended.csv --max-bars 192
    python run_outcome_tracker.py --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.outcome_resolver import (
    OutcomeResolver,
    load_ohlcv,
    load_signals_jsonl,
    save_outcomes,
    save_summary,
)

DEFAULT_SIGNALS = "/root/.openclaw/workspace/jimi_audit/data/strategy_signals.jsonl"
DEFAULT_OHLCV = "/root/.openclaw/workspace/jimi_audit/data/eth_15m_merged.csv"
DEFAULT_OUTPUT_DIR = "/root/.openclaw/workspace/jimi_audit/data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="JIMI Outcome Tracker — Resolve signal outcomes from OHLCV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--signals", "-s",
        default=DEFAULT_SIGNALS,
        help=f"Path to signals JSONL (default: {DEFAULT_SIGNALS})",
    )
    parser.add_argument(
        "--ohlcv", "-o",
        default=DEFAULT_OHLCV,
        help=f"Path to OHLCV CSV (default: {DEFAULT_OHLCV})",
    )
    parser.add_argument(
        "--output-dir", "-d",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--strategy",
        help="Filter to specific strategy (e.g., trade_flow)",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=96,
        help="Max bars to hold a trade before timeout (default: 96 = 24h for 15m)",
    )
    parser.add_argument(
        "--output-name",
        help="Output filename (default: resolved_outcomes_<strategy>.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate synthetic signals and OHLCV for testing",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def generate_synthetic_data(n_signals=50, n_candles=500):
    """Generate synthetic OHLCV + signals for dry-run testing."""
    import random
    from src.outcome_resolver import Candle, Signal

    rng = random.Random(42)
    base_ts = 1700000000.0
    bar_secs = 900

    # Generate OHLCV
    candles = []
    price = 3000.0
    for i in range(n_candles):
        ts = base_ts + i * bar_secs
        change = rng.gauss(0, 10)
        o = price
        h = o + abs(rng.gauss(0, 8))
        l = o - abs(rng.gauss(0, 8))
        c = o + change
        candles.append(Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=rng.uniform(100, 1000)))
        price = c

    # Generate signals
    signals = []
    for i in range(n_signals):
        candle_idx = rng.randint(10, n_candles - 100)
        c = candles[candle_idx]
        direction = rng.choice(["LONG", "SHORT"])
        entry = c.close
        risk = rng.uniform(5, 20)

        if direction == "LONG":
            sl = entry - risk
            tp = entry + risk * rng.uniform(1.0, 2.5)
        else:
            sl = entry + risk
            tp = entry - risk * rng.uniform(1.0, 2.5)

        rr = abs(tp - entry) / risk

        from datetime import datetime
        ts_dt = datetime.fromtimestamp(c.timestamp)

        signals.append(Signal(
            line_idx=i,
            timestamp=ts_dt.strftime("%Y-%m-%d %H:%M:%S"),
            entry_time=c.timestamp,
            strategy="dry_run_test",
            direction=direction,
            entry=round(entry, 2),
            sl=round(sl, 2),
            tp=round(tp, 2),
            rr=round(rr, 2),
            conviction=round(rng.uniform(0.3, 0.95), 2),
            price=round(c.close, 2),
        ))

    return candles, signals


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("jimi.outcome.cli")

    if args.dry_run:
        logger.info("DRY RUN: Generating synthetic data ...")
        candles, signals = generate_synthetic_data()
        strategy_name = "dry_run_test"
    else:
        # Load OHLCV
        ohlcv_path = Path(args.ohlcv)
        if not ohlcv_path.exists():
            logger.error("OHLCV file not found: %s", ohlcv_path)
            logger.info("Available ETH data: eth_15m_merged.csv, eth_15m_extended.csv")
            sys.exit(1)
        candles = load_ohlcv(ohlcv_path)

        # Load signals
        signals_path = Path(args.signals)
        if not signals_path.exists():
            logger.error("Signals file not found: %s", signals_path)
            sys.exit(1)
        signals = load_signals_jsonl(signals_path, strategy_filter=args.strategy)
        strategy_name = args.strategy or "all"

    if not signals:
        logger.error("No signals found")
        sys.exit(1)

    logger.info("Loaded %d candles, %d signals", len(candles), len(signals))

    # Resolve
    resolver = OutcomeResolver(candles, max_bars=args.max_bars)
    outcomes = resolver.resolve_all(signals)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name:
        out_name = args.output_name
    else:
        out_name = f"resolved_outcomes_{strategy_name}.jsonl"

    outcomes_path = save_outcomes(outcomes, output_dir / out_name)
    summary_path = save_summary(outcomes, output_dir / f"outcome_summary_{strategy_name}.txt")

    # Print summary
    with open(summary_path) as f:
        print(f.read())

    logger.info("Done. Outcomes: %s | Summary: %s", outcomes_path, summary_path)


if __name__ == "__main__":
    main()
