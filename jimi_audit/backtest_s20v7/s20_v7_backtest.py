#!/usr/bin/env python3
"""
S20 Liquidation Cascade v7 — Backtest Script
==============================================
Runs on VPS with REAL data:
  - OHLCV: /root/.openclaw/workspace/jimi_audit/data/history/ETHUSDT_15m.csv
  - Derivatives: derivatives_collected.csv + derivatives_backfilled.csv
  - Signals: strategy_signals.jsonl (liquidation_cascade fired signals)

Produces:
  1. Trade-by-trade log
  2. Performance metrics (WR, PF, mean return, Sharpe)
  3. Monte Carlo (1000 iterations) for statistical confidence
  4. Breakdown by OI ROC level and volatility regime
  5. Equity curve data
"""

import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# Add strategy module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s20_liquidation_cascade_v7 import (
    S20Config,
    S20LiquidationCascadeV7,
    SignalState,
    TradeSignal,
    VolatilityRegimeClassifier,
    OIROCCalculator,
    merge_derivatives_sources,
)

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════

BASE = "/root/.openclaw/workspace/jimi_audit/data"
OHLCV_PATH = os.path.join(BASE, "history", "ETHUSDT_15m.csv")
DERIV_COLLECTED = os.path.join(BASE, "derivatives_history", "derivatives_collected.csv")
DERIV_BACKFILLED = os.path.join(BASE, "derivatives_history", "derivatives_backfilled.csv")
SIGNALS_PATH = os.path.join(BASE, "strategy_signals.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s20_v7_backtest_results.json")


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

@dataclass
class OHLCVBar:
    timestamp: datetime
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_ohlcv(path: str) -> List[OHLCVBar]:
    """Load OHLCV data. Timestamps are Unix milliseconds."""
    bars = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_ms = int(row['ts'])
            # Convert Unix ms to datetime (UTC)
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            bars.append(OHLCVBar(
                timestamp=dt,
                ts_ms=ts_ms,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
            ))
    return bars


def load_fired_signals(path: str) -> Dict[str, List[dict]]:
    """
    Load fired liquidation_cascade signals from JSONL.
    Returns dict keyed by timestamp string → list of signal dicts.
    """
    signals = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                if (d.get("strategy") == "liquidation_cascade" and 
                    d.get("fired") == True):
                    ts = d["timestamp"]
                    signals[ts].append(d)
            except (json.JSONDecodeError, KeyError):
                continue
    return dict(signals)


def find_closest_derivatives(
    ts: datetime, 
    deriv_data: Dict[str, Dict[str, float]], 
    max_delta_minutes: int = 30
) -> Optional[Dict[str, float]]:
    """
    Find the closest derivatives data point to a given timestamp.
    Handles the mismatch between OHLCV (15m bars) and derivatives (variable intervals).
    """
    ts_str = ts.strftime("%Y-%m-%d %H:%M")
    
    # Try exact match first
    if ts_str in deriv_data:
        return deriv_data[ts_str]
    
    # Try ±15 min windows
    best_match = None
    best_delta = max_delta_minutes + 1
    
    for delta_min in [0, -15, 15, -30, 30]:
        check_ts = ts + timedelta(minutes=delta_min)
        check_str = check_ts.strftime("%Y-%m-%d %H:%M")
        if check_str in deriv_data:
            abs_delta = abs(delta_min)
            if abs_delta < best_delta:
                best_delta = abs_delta
                best_match = deriv_data[check_str]
    
    return best_match


# ═══════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class TradeResult:
    """Result of a single simulated trade."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    tp_price: float
    sl_price: float
    pnl_pct: float
    exit_reason: str  # 'TP', 'SL', 'HOLD', 'EOD'
    conviction: float
    oi_roc: float
    signal_type: str
    vol_regime: str
    hold_bars_actual: int
    expected_return_pct: float


def simulate_trade(
    signal: TradeSignal,
    bars: List[OHLCVBar],
    start_idx: int,
) -> Optional[TradeResult]:
    """
    Simulate a trade from signal entry through TP/SL/hold exit.
    
    Entry at signal price on the signal bar.
    Check each subsequent bar for TP, SL, or max hold.
    """
    if start_idx >= len(bars):
        return None
    
    entry_price = signal.entry_price
    tp_price = signal.tp_price
    sl_price = signal.sl_price
    
    # Check bars from entry+1 to entry+hold_bars
    max_idx = min(start_idx + signal.hold_bars, len(bars))
    
    for i in range(start_idx + 1, max_idx):
        bar = bars[i]
        
        if signal.direction == "SHORT":
            # Check SL (price went up against us)
            if bar.high >= sl_price:
                return TradeResult(
                    entry_time=signal.timestamp,
                    exit_time=bar.timestamp,
                    direction="SHORT",
                    entry_price=entry_price,
                    exit_price=sl_price,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    pnl_pct=(entry_price - sl_price) / entry_price * 100,  # Negative for SL
                    exit_reason="SL",
                    conviction=signal.conviction,
                    oi_roc=signal.oi_roc,
                    signal_type=signal.signal_type,
                    vol_regime=signal.vol_regime,
                    hold_bars_actual=i - start_idx,
                    expected_return_pct=signal.expected_return_pct,
                )
            
            # Check TP (price went down in our favor)
            if bar.low <= tp_price:
                return TradeResult(
                    entry_time=signal.timestamp,
                    exit_time=bar.timestamp,
                    direction="SHORT",
                    entry_price=entry_price,
                    exit_price=tp_price,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    pnl_pct=(entry_price - tp_price) / entry_price * 100,  # Positive for TP
                    exit_reason="TP",
                    conviction=signal.conviction,
                    oi_roc=signal.oi_roc,
                    signal_type=signal.signal_type,
                    vol_regime=signal.vol_regime,
                    hold_bars_actual=i - start_idx,
                    expected_return_pct=signal.expected_return_pct,
                )
    
    # Hold timeout — exit at close of last bar
    exit_idx = max_idx - 1
    exit_bar = bars[exit_idx]
    exit_price = exit_bar.close
    
    pnl_pct = (entry_price - exit_price) / entry_price * 100
    
    return TradeResult(
        entry_time=signal.timestamp,
        exit_time=exit_bar.timestamp,
        direction="SHORT",
        entry_price=entry_price,
        exit_price=exit_price,
        tp_price=tp_price,
        sl_price=sl_price,
        pnl_pct=pnl_pct,
        exit_reason="HOLD",
        conviction=signal.conviction,
        oi_roc=signal.oi_roc,
        signal_type=signal.signal_type,
        vol_regime=signal.vol_regime,
        hold_bars_actual=signal.hold_bars,
        expected_return_pct=signal.expected_return_pct,
    )


# ═══════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════

def compute_stats(trades: List[TradeResult]) -> dict:
    """Compute comprehensive performance statistics."""
    if not trades:
        return {"error": "No trades"}
    
    pnls = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    n = len(pnls)
    n_wins = len(wins)
    n_losses = len(losses)
    wr = n_wins / n if n > 0 else 0
    
    mean_pnl = sum(pnls) / n if n > 0 else 0
    median_pnl = sorted(pnls)[n // 2] if n > 0 else 0
    
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Sharpe-like ratio (annualized, assuming 15m bars)
    if n > 1:
        var = sum((p - mean_pnl) ** 2 for p in pnls) / (n - 1)
        std = math.sqrt(var)
        # Annualize: ~35040 bars/year at 15m
        sharpe = (mean_pnl / std) * math.sqrt(35040) if std > 0 else 0
    else:
        std = 0
        sharpe = 0
    
    # Max drawdown (cumulative)
    cum_pnl = 0
    peak = 0
    max_dd = 0
    for p in pnls:
        cum_pnl += p
        if cum_pnl > peak:
            peak = cum_pnl
        dd = peak - cum_pnl
        if dd > max_dd:
            max_dd = dd
    
    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t.exit_reason] += 1
    
    # Average hold bars
    avg_hold = sum(t.hold_bars_actual for t in trades) / n if n > 0 else 0
    
    return {
        "total_trades": n,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": round(wr, 4),
        "mean_pnl_pct": round(mean_pnl, 4),
        "median_pnl_pct": round(median_pnl, 4),
        "std_pnl_pct": round(std, 4),
        "profit_factor": round(pf, 4),
        "gross_profit_pct": round(gross_profit, 4),
        "gross_loss_pct": round(gross_loss, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "avg_hold_bars": round(avg_hold, 1),
        "exit_reasons": dict(exit_reasons),
    }


def compute_breakdown(trades: List[TradeResult]) -> dict:
    """Breakdown by OI ROC level and volatility regime."""
    
    # By OI ROC level
    by_oi_roc = defaultdict(list)
    for t in trades:
        if t.oi_roc < -0.015:
            by_oi_roc["oi_roc_lt_-0.015"].append(t)
        elif t.oi_roc < -0.01:
            by_oi_roc["oi_roc_-0.015_to_-0.01"].append(t)
        else:
            by_oi_roc["oi_roc_gt_-0.01"].append(t)
    
    # By signal type
    by_signal_type = defaultdict(list)
    for t in trades:
        by_signal_type[t.signal_type].append(t)
    
    # By vol regime
    by_regime = defaultdict(list)
    for t in trades:
        by_regime[t.vol_regime].append(t)
    
    # By exit reason
    by_exit = defaultdict(list)
    for t in trades:
        by_exit[t.exit_reason].append(t)
    
    result = {}
    for label, group in [
        ("by_oi_roc_level", by_oi_roc),
        ("by_signal_type", by_signal_type),
        ("by_vol_regime", by_regime),
        ("by_exit_reason", by_exit),
    ]:
        result[label] = {}
        for key, tlist in group.items():
            result[label][key] = compute_stats(tlist)
    
    return result


# ═══════════════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════════════

def monte_carlo_analysis(
    trades: List[TradeResult], n_iterations: int = 1000
) -> dict:
    """
    Monte Carlo simulation: shuffle trade order to test robustness.
    Returns distribution of key metrics across random reorderings.
    """
    if len(trades) < 2:
        return {"error": "Need at least 2 trades for Monte Carlo"}
    
    pnls = [t.pnl_pct for t in trades]
    n = len(pnls)
    
    # Track distributions
    mc_wrs = []
    mc_pfs = []
    mc_means = []
    mc_max_dds = []
    mc_cum_pnls = []
    
    for _ in range(n_iterations):
        shuffled = pnls[:] 
        random.shuffle(shuffled)
        
        # Win rate
        wins = sum(1 for p in shuffled if p > 0)
        mc_wrs.append(wins / n)
        
        # Profit factor
        gross_profit = sum(p for p in shuffled if p > 0)
        gross_loss = abs(sum(p for p in shuffled if p <= 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        mc_pfs.append(min(pf, 100))  # Cap at 100 for visualization
        
        # Mean
        mc_means.append(sum(shuffled) / n)
        
        # Max drawdown
        cum = 0
        peak = 0
        max_dd = 0
        for p in shuffled:
            cum += p
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd
        mc_max_dds.append(max_dd)
        
        # Cumulative PnL
        mc_cum_pnls.append(sum(shuffled))
    
    def percentile(data, p):
        s = sorted(data)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]
    
    return {
        "iterations": n_iterations,
        "n_trades": n,
        "win_rate": {
            "mean": round(sum(mc_wrs) / n_iterations, 4),
            "p5": round(percentile(mc_wrs, 5), 4),
            "p25": round(percentile(mc_wrs, 25), 4),
            "p50": round(percentile(mc_wrs, 50), 4),
            "p75": round(percentile(mc_wrs, 75), 4),
            "p95": round(percentile(mc_wrs, 95), 4),
        },
        "profit_factor": {
            "mean": round(sum(mc_pfs) / n_iterations, 4),
            "p5": round(percentile(mc_pfs, 5), 4),
            "p50": round(percentile(mc_pfs, 50), 4),
            "p95": round(percentile(mc_pfs, 95), 4),
        },
        "mean_pnl_pct": {
            "mean": round(sum(mc_means) / n_iterations, 4),
            "p5": round(percentile(mc_means, 5), 4),
            "p50": round(percentile(mc_means, 50), 4),
            "p95": round(percentile(mc_means, 95), 4),
        },
        "max_drawdown_pct": {
            "mean": round(sum(mc_max_dds) / n_iterations, 4),
            "p5": round(percentile(mc_max_dds, 5), 4),
            "p50": round(percentile(mc_max_dds, 50), 4),
            "p95": round(percentile(mc_max_dds, 95), 4),
        },
        "cumulative_pnl_pct": {
            "mean": round(sum(mc_cum_pnls) / n_iterations, 4),
            "p5": round(percentile(mc_cum_pnls, 5), 4),
            "p50": round(percentile(mc_cum_pnls, 50), 4),
            "p95": round(percentile(mc_cum_pnls, 95), 4),
        },
    }


# ═══════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════

def run_backtest():
    """Run full backtest with real data."""
    
    print("=" * 60)
    print("S20 Liquidation Cascade v7 — Backtest")
    print("=" * 60)
    
    # ─── Load Data ─────────────────────────────────────────────
    print("\n[1/6] Loading OHLCV data...")
    bars = load_ohlcv(OHLCV_PATH)
    print(f"  Loaded {len(bars)} bars: {bars[0].timestamp} to {bars[-1].timestamp}")
    
    print("\n[2/6] Loading derivatives data...")
    deriv_data = merge_derivatives_sources(DERIV_COLLECTED, DERIV_BACKFILLED)
    # Count OI availability
    oi_count = sum(1 for v in deriv_data.values() if v.get("oi", 0) > 0)
    print(f"  Loaded {len(deriv_data)} derivatives records, {oi_count} with OI")
    
    print("\n[3/6] Loading fired signals...")
    fired_signals = load_fired_signals(SIGNALS_PATH)
    total_fired = sum(len(v) for v in fired_signals.values())
    print(f"  Loaded {total_fired} fired liquidation_cascade signals at {len(fired_signals)} timestamps")
    
    # ─── Precompute Regimes ────────────────────────────────────
    print("\n[4/6] Computing volatility regimes (rolling percentile)...")
    closes = [b.close for b in bars]
    timestamps = [b.timestamp for b in bars]
    
    regime_classifier = VolatilityRegimeClassifier(
        lookback=96,  # 24h at 15m
        low_pct=0.33,
        high_pct=0.67,
    )
    regime_classifier.compute(closes, timestamps)
    
    # Count regimes
    regime_counts = defaultdict(int)
    for ts in timestamps:
        r = regime_classifier.get_regime(ts)
        if r:
            regime_counts[r] += 1
    print(f"  Regime distribution: {dict(regime_counts)}")
    
    # ─── Run Strategy ──────────────────────────────────────────
    print("\n[5/6] Running strategy over historical data...")
    
    config = S20Config()
    strategy = S20LiquidationCascadeV7(config)
    oi_calc = OIROCCalculator(lookback_bars=4)
    
    # Build timestamp → bar index mapping
    bar_index = {b.timestamp: i for i, b in enumerate(bars)}
    
    # Track trade results
    trade_results: List[TradeResult] = []
    
    # Track signal stats
    signals_evaluated = 0
    signals_rejected_oi = 0
    signals_rejected_regime = 0
    signals_rejected_cooldown = 0
    signals_generated = 0
    
    # Process each bar — feed OI data and check for signals
    for i, bar in enumerate(bars):
        # Get derivatives data for this bar
        deriv = find_closest_derivatives(bar.timestamp, deriv_data)
        
        if deriv and deriv.get("oi", 0) > 0:
            # Feed OI to calculator
            oi_calc.add(bar.timestamp, deriv["oi"])
            
            # Compute OI ROC
            oi_roc = oi_calc.compute_roc(bar.timestamp, deriv["oi"])
            
            # Get regime
            regime = regime_classifier.get_regime(bar.timestamp)
            
            if oi_roc is not None and regime is not None:
                signals_evaluated += 1
                
                # Build signal state
                state = SignalState(
                    timestamp=bar.timestamp,
                    price=bar.close,
                    oi=deriv["oi"],
                    oi_roc=oi_roc,
                    vol_regime=regime,
                    rolling_vol=None,
                    vol_percentile=regime_classifier.get_percentile(bar.timestamp),
                    data_source="collected" if deriv.get("oi", 0) > 0 else "backfilled",
                )
                
                # Evaluate strategy
                signal = strategy.evaluate(state)
                
                if signal:
                    signals_generated += 1
                    
                    # Simulate trade
                    result = simulate_trade(signal, bars, i)
                    if result:
                        trade_results.append(result)
                else:
                    # Track why signal was rejected
                    if oi_roc >= config.oi_roc_borderline:
                        pass  # Not a signal threshold
                    elif regime != "MID":
                        signals_rejected_regime += 1
                    else:
                        signals_rejected_cooldown += 1
    
    print(f"  Signals evaluated: {signals_evaluated}")
    print(f"  Signals generated: {signals_generated}")
    print(f"  Trades completed: {len(trade_results)}")
    print(f"  Rejected (regime): {signals_rejected_regime}")
    print(f"  Rejected (cooldown): {signals_rejected_cooldown}")
    
    # ─── Compute Statistics ────────────────────────────────────
    print("\n[6/6] Computing statistics...")
    
    stats = compute_stats(trade_results)
    breakdown = compute_breakdown(trade_results)
    mc = monte_carlo_analysis(trade_results, n_iterations=1000)
    
    # ─── Trade Log ─────────────────────────────────────────────
    trade_log = []
    for t in trade_results:
        trade_log.append({
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat(),
            "direction": t.direction,
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2),
            "tp_price": round(t.tp_price, 2),
            "sl_price": round(t.sl_price, 2),
            "pnl_pct": round(t.pnl_pct, 4),
            "exit_reason": t.exit_reason,
            "conviction": round(t.conviction, 3),
            "oi_roc": round(t.oi_roc, 6),
            "signal_type": t.signal_type,
            "vol_regime": t.vol_regime,
            "hold_bars": t.hold_bars_actual,
        })
    
    # ─── Output ────────────────────────────────────────────────
    results = {
        "strategy": "S20_Liquidation_Cascade_v7",
        "backtest_date": datetime.now(timezone.utc).isoformat(),
        "data_range": {
            "ohlcv_start": bars[0].timestamp.isoformat(),
            "ohlcv_end": bars[-1].timestamp.isoformat(),
            "ohlcv_bars": len(bars),
            "derivatives_records": len(deriv_data),
            "derivatives_with_oi": oi_count,
        },
        "signal_stats": {
            "signals_evaluated": signals_evaluated,
            "signals_generated": signals_generated,
            "trades_completed": len(trade_results),
            "rejected_regime": signals_rejected_regime,
            "rejected_cooldown": signals_rejected_cooldown,
        },
        "performance": stats,
        "breakdown": breakdown,
        "monte_carlo": mc,
        "config": {
            "oi_roc_primary": config.oi_roc_primary,
            "oi_roc_borderline": config.oi_roc_borderline,
            "regime_filter": config.regime_filter,
            "direction": config.direction,
            "tp_pct": config.tp_pct,
            "sl_pct": config.sl_pct,
            "hold_bars": config.hold_bars,
            "cooldown_minutes": config.cooldown_minutes,
        },
        "trade_log": trade_log,
    }
    
    # Save to file
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {OUTPUT_PATH}")
    
    # ─── Print Summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nData: {bars[0].timestamp.strftime('%Y-%m-%d')} to {bars[-1].timestamp.strftime('%Y-%m-%d')}")
    print(f"Derivatives records with OI: {oi_count}")
    print(f"Total trades: {stats.get('total_trades', 0)}")
    
    if stats.get('total_trades', 0) > 0:
        print(f"\n--- Performance ---")
        print(f"Win Rate:       {stats['win_rate']*100:.1f}%")
        print(f"Profit Factor:  {stats['profit_factor']:.2f}")
        print(f"Mean PnL:       {stats['mean_pnl_pct']:.3f}%")
        print(f"Median PnL:     {stats['median_pnl_pct']:.3f}%")
        print(f"Std Dev:        {stats['std_pnl_pct']:.3f}%")
        print(f"Sharpe Ratio:   {stats['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:   {stats['max_drawdown_pct']:.3f}%")
        print(f"Avg Hold Bars:  {stats['avg_hold_bars']:.1f}")
        print(f"Exit Reasons:   {stats['exit_reasons']}")
        
        print(f"\n--- Monte Carlo ({mc.get('iterations', 0)} iterations) ---")
        if 'win_rate' in mc:
            wr = mc['win_rate']
            print(f"Win Rate:    p5={wr['p5']*100:.1f}%  p50={wr['p50']*100:.1f}%  p95={wr['p95']*100:.1f}%")
            pf = mc['profit_factor']
            print(f"PF:          p5={pf['p5']:.2f}  p50={pf['p50']:.2f}  p95={pf['p95']:.2f}")
            mp = mc['mean_pnl_pct']
            print(f"Mean PnL:    p5={mp['p5']:.3f}%  p50={mp['p50']:.3f}%  p95={mp['p95']:.3f}%")
            dd = mc['max_drawdown_pct']
            print(f"Max DD:      p5={dd['p5']:.3f}%  p50={dd['p50']:.3f}%  p95={dd['p95']:.3f}%")
            cp = mc['cumulative_pnl_pct']
            print(f"Cum PnL:     p5={cp['p5']:.2f}%  p50={cp['p50']:.2f}%  p95={cp['p95']:.2f}%")
        
        print(f"\n--- Breakdown by OI ROC Level ---")
        for level, s in breakdown.get('by_oi_roc_level', {}).items():
            print(f"  {level}: n={s.get('total_trades',0)}, WR={s.get('win_rate',0)*100:.1f}%, "
                  f"PF={s.get('profit_factor',0):.2f}, mean={s.get('mean_pnl_pct',0):.3f}%")
        
        print(f"\n--- Breakdown by Signal Type ---")
        for stype, s in breakdown.get('by_signal_type', {}).items():
            print(f"  {stype}: n={s.get('total_trades',0)}, WR={s.get('win_rate',0)*100:.1f}%, "
                  f"PF={s.get('profit_factor',0):.2f}, mean={s.get('mean_pnl_pct',0):.3f}%")
        
        print(f"\n--- Breakdown by Vol Regime ---")
        for regime, s in breakdown.get('by_vol_regime', {}).items():
            print(f"  {regime}: n={s.get('total_trades',0)}, WR={s.get('win_rate',0)*100:.1f}%, "
                  f"PF={s.get('profit_factor',0):.2f}, mean={s.get('mean_pnl_pct',0):.3f}%")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    random.seed(42)  # Reproducible Monte Carlo
    run_backtest()
