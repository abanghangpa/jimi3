#!/usr/bin/env python3
"""
JIMI Validator Agent — CLI Entry Point
=======================================

Usage:
    python run_validation.py <strategy_name> [options]

Examples:
    python run_validation.py trade_flow_v4.2
    python run_validation.py funding_arb_v7.1 --data /path/to/signals.jsonl
    python run_validation.py trade_flow_v4.2 --no-wf --mc-perms 5000
    python run_validation.py trade_flow_v4.2 --report-dir ./reports
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validator_agent import (
    ValidatorAgent,
    ValidationThresholds,
    WFA_CONFIG,
    load_trades_jsonl,
    save_report,
    print_summary,
)

DEFAULT_DATA_PATH = "/root/.openclaw/workspace/jimi_audit/data/strategy_signals.jsonl"
DEFAULT_REPORT_DIR = "/root/.openclaw/workspace/jimi_audit/reports"


def parse_args():
    parser = argparse.ArgumentParser(
        description="JIMI Validator Agent — Statistical validation for strategy claims",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Thresholds (defaults):
  DSR          > 1.0 (min)  |  > 2.0 (target)
  WF WR        > 50% (min)  |  > 60% (target)
  CPCV Score   > 0.55 (min) |  > 0.65 (target)
  MC p-value   < 0.05 (min) |  < 0.01 (target)
  Sample Size  > 30 (min)   |  > 100 (target)

Strategies:
  trade_flow_v4.2     First validation target
  funding_arb_v7.1    Second validation target
        """,
    )

    parser.add_argument(
        "strategy",
        help="Strategy name to validate (e.g., trade_flow_v4.2)",
    )
    parser.add_argument(
        "--data", "-d",
        default=DEFAULT_DATA_PATH,
        help=f"Path to strategy signals JSONL (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--outcomes",
        help="Path to resolved outcomes JSONL (from outcome tracker)",
    )
    parser.add_argument(
        "--report-dir", "-r",
        default=DEFAULT_REPORT_DIR,
        help=f"Output directory for reports (default: {DEFAULT_REPORT_DIR})",
    )
    parser.add_argument(
        "--no-wf",
        action="store_true",
        help="Skip walk-forward analysis",
    )
    parser.add_argument(
        "--no-dsr",
        action="store_true",
        help="Skip deflated Sharpe ratio",
    )
    parser.add_argument(
        "--no-mc",
        action="store_true",
        help="Skip Monte Carlo permutation test",
    )
    parser.add_argument(
        "--no-cpcv",
        action="store_true",
        help="Skip combinatorial purged cross-validation",
    )
    parser.add_argument(
        "--mc-perms",
        type=int,
        default=10_000,
        help="Number of Monte Carlo permutations (default: 10000)",
    )
    parser.add_argument(
        "--n-strategies",
        type=int,
        default=20,
        help="Number of strategies tested (for DSR deflation, default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--train-bars",
        type=int,
        default=WFA_CONFIG["train_bars"],
        help=f"Walk-forward training window in bars (default: {WFA_CONFIG['train_bars']})",
    )
    parser.add_argument(
        "--test-bars",
        type=int,
        default=WFA_CONFIG["test_bars"],
        help=f"Walk-forward test window in bars (default: {WFA_CONFIG['test_bars']})",
    )
    parser.add_argument(
        "--step-bars",
        type=int,
        default=WFA_CONFIG["step_bars"],
        help=f"Walk-forward step size in bars (default: {WFA_CONFIG['step_bars']})",
    )
    parser.add_argument(
        "--purge-bars",
        type=int,
        default=WFA_CONFIG["purge_bars"],
        help=f"Purge window in bars (default: {WFA_CONFIG['purge_bars']})",
    )
    parser.add_argument(
        "--bar-secs",
        type=int,
        default=WFA_CONFIG["bar_duration_secs"],
        help=f"Seconds per bar for WFA time conversion (default: {WFA_CONFIG['bar_duration_secs']})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate synthetic data for testing instead of loading from file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def generate_synthetic_trades(n: int = 200, seed: int = 42):
    """Generate synthetic trade data for dry-run testing.

    Produces trades spanning enough time for walk-forward analysis.
    Uses 15-min bar equivalent timestamps (900s per bar).
    With default WFA config (train=4000, test=960, step=960),
    we need at least 4000+16+960 = 4976 bars = ~51.8 days minimum.
    We generate ~7000 bars (~73 days) worth of trades.
    """
    import random
    rng = random.Random(seed)

    regimes = ["trending", "ranging", "volatile", "quiet"]
    sides = ["long", "short"]
    BAR_SECS = 900  # 15-minute bars
    base_ts = 1700000000.0

    trades_data = []
    ts = base_ts
    # Spread trades across ~7000 bars (about 73 days)
    total_span = 7000 * BAR_SECS
    avg_gap = total_span / n

    for i in range(n):
        entry_time = ts
        duration = rng.uniform(BAR_SECS, BAR_SECS * 8)  # 1 to 8 bars
        exit_time = entry_time + duration

        # Regime-dependent edge
        regime = rng.choice(regimes)
        if regime == "trending":
            win_prob = 0.62
        elif regime == "volatile":
            win_prob = 0.55
        elif regime == "ranging":
            win_prob = 0.48
        else:  # quiet
            win_prob = 0.50

        win = rng.random() < win_prob
        if win:
            pnl = rng.uniform(0.01, 0.15)
        else:
            pnl = -rng.uniform(0.01, 0.12)

        trades_data.append({
            "strategy": "dry_run_test",
            "entry_time": entry_time,
            "exit_time": exit_time,
            "pnl": round(pnl, 6),
            "side": rng.choice(sides),
            "symbol": "BTC/USDT",
            "regime": regime,
        })
        ts += rng.uniform(avg_gap * 0.5, avg_gap * 1.5)

    return trades_data


def main():
    args = parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("jimi.validator.cli")

    # Load or generate data
    if args.dry_run:
        logger.info("DRY RUN: Generating synthetic trades ...")
        from src.validator_agent import Trade
        synthetic = generate_synthetic_trades(n=200, seed=args.seed)
        trades = [
            Trade(
                entry_time=r["entry_time"],
                exit_time=r["exit_time"],
                pnl=r["pnl"],
                side=r["side"],
                symbol=r["symbol"],
                regime=r["regime"],
                strategy=r["strategy"],
            )
            for r in synthetic
        ]
        strategy_name = f"{args.strategy}_dryrun"
    else:
        logger.info("Loading trades for strategy: %s", args.strategy)
        data_path = Path(args.data)
        if not data_path.exists():
            logger.error("Data file not found: %s", data_path)
            logger.info("Tip: Use --dry-run to test with synthetic data")
            sys.exit(1)

        trades = load_trades_jsonl(
            data_path,
            strategy_filter=args.strategy,
            outcomes_path=args.outcomes,
        )
        strategy_name = args.strategy

    if not trades:
        logger.error("No trades found for strategy '%s'", args.strategy)
        sys.exit(1)

    logger.info("Loaded %d trades", len(trades))

    # Configure walk-forward
    wfa_config = {
        "train_bars": args.train_bars,
        "test_bars": args.test_bars,
        "step_bars": args.step_bars,
        "min_test_trades": 5,
        "purge_bars": args.purge_bars,
        "bar_duration_secs": args.bar_secs,
    }

    # Build validator
    validator = ValidatorAgent(
        wfa_config=wfa_config,
        n_mc_permutations=args.mc_perms,
        n_strategies_tested=args.n_strategies,
        random_seed=args.seed,
    )

    # Run validation
    logger.info("Starting validation ...")
    report = validator.validate(
        strategy_name=strategy_name,
        trades=trades,
        run_wf=not args.no_wf,
        run_dsr=not args.no_dsr,
        run_mc=not args.no_mc,
        run_cpcv=not args.no_cpcv,
    )

    # Print summary
    print_summary(report)

    # Save report
    report_path = save_report(report, args.report_dir)
    logger.info("Report saved: %s", report_path)

    # Exit code
    sys.exit(0 if report.pass_gate else 1)


if __name__ == "__main__":
    main()
