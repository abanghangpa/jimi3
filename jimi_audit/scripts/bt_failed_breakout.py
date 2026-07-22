#!/usr/bin/env python3
"""
Failed Breakout Comprehensive Backtest
=======================================
Tests the M20 failed_breakout strategy with:
1. Standalone baseline
2. Parameter sweep (TP, SL, hold, conviction)
3. With sweep magnitude filter
4. With funding rate filter
5. Combined filters

Data: ETH/USDT 15m, April 2025 - July 2026 (37,845 bars)
Target: WR >= 75%, PF >= 2.0
"""

import csv, json, sys, os
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
DATA_FILE = "/root/.openclaw/workspace/jimi_audit/eth_15m_6m.csv"
DERIVATIVES_DIR = "/root/.openclaw/workspace/jimi_audit/data/derivatives_history"
OUTPUT_FILE = "/root/.openclaw/workspace/jimi_audit/reports/failed_breakout_backtest.json"

# Strategy defaults (from scanner_executor)
DEFAULT_TP_PCT = 2.5
DEFAULT_SL_PCT = 1.0
DEFAULT_HOLD_HOURS = 32
DEFAULT_MIN_CONV = 0.7
LEVERAGE = 25
FEE_RATE = 0.0002
SLIPPAGE = 0.001

# ── Load Data ───────────────────────────────────────────────────────────
def load_ohlcv(filepath):
    """Load 15m OHLCV data."""
    bars = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append({
                'ts': datetime.strptime(row['Open time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume']),
                'trades': int(row.get('Number of trades', 0)),
                'taker_buy_vol': float(row.get('Taker buy base asset volume', 0)),
            })
    return bars

def compute_atr(bars, period=14):
    """Compute ATR."""
    atrs = [0] * len(bars)
    for i in range(1, len(bars)):
        tr = max(
            bars[i]['high'] - bars[i]['low'],
            abs(bars[i]['high'] - bars[i-1]['close']),
            abs(bars[i]['low'] - bars[i-1]['close'])
        )
        if i < period:
            atrs[i] = tr
        else:
            atrs[i] = (atrs[i-1] * (period - 1) + tr) / period
    return atrs

def compute_bb(bars, period=20, std_mult=2.0):
    """Compute Bollinger Bands."""
    closes = [b['close'] for b in bars]
    upper = [0] * len(bars)
    lower = [0] * len(bars)
    mid = [0] * len(bars)
    for i in range(period - 1, len(bars)):
        window = closes[i - period + 1:i + 1]
        m = np.mean(window)
        s = np.std(window)
        mid[i] = m
        upper[i] = m + std_mult * s
        lower[i] = m - std_mult * s
    return upper, mid, lower

def load_derivatives():
    """Load funding rate data if available."""
    fr_data = {}
    fr_file = os.path.join(DERIVATIVES_DIR, "funding_rate.json")
    if os.path.exists(fr_file):
        with open(fr_file) as f:
            data = json.load(f)
            for entry in data:
                ts = entry.get('timestamp') or entry.get('ts')
                if ts:
                    fr_data[ts] = entry.get('funding_rate', 0)
    return fr_data

# ── Signal Detection ────────────────────────────────────────────────────
def detect_failed_breakout(bars, i, bb_upper, bb_lower, atrs, config):
    """
    Detect failed breakout pattern at bar i.
    Returns: (direction, conviction, entry, sl, tp) or None
    """
    if i < config['lookback'] + 5:
        return None

    close = bars[i]['close']
    high = bars[i]['high']
    low = bars[i]['low']
    atr = atrs[i]
    if atr <= 0:
        return None

    lookback = config['lookback']
    min_conv = config['min_conv']

    # Scan for recent breakout attempt
    for j in range(max(0, i - lookback), i):
        bb_up = bb_upper[j]
        bb_lo = bb_lower[j]
        if bb_up == 0 or bb_lo == 0:
            continue

        # Upside breakout attempt (price broke above BB upper)
        if bars[j]['high'] > bb_up:
            # Check if it failed: price returned below the breakout level
            breakout_level = bb_up
            return_pct = (breakout_level - bars[i]['close']) / breakout_level * 100
            if return_pct >= config['failure_return_pct']:
                # Compute conviction
                wick_ratio = (bars[j]['high'] - max(bars[j]['open'], bars[j]['close'])) / max(bars[j]['high'] - bars[j]['low'], 0.01)
                vol_ratio = bars[j]['volume'] / max(np.mean([bars[k]['volume'] for k in range(max(0,j-10), j)]), 1)
                taker_ratio = bars[j]['taker_buy_vol'] / max(bars[j]['volume'], 0.01)

                quality = 0.5
                if wick_ratio >= 0.4: quality += 0.15
                if vol_ratio >= 1.0: quality += 0.1
                if taker_ratio < 0.45: quality += 0.15  # sellers active
                reversal_strength = min(return_pct / 1.0, 1.0) * 0.3
                conv = min(quality + reversal_strength, 1.0)

                if conv >= min_conv:
                    # SHORT signal: failed upside breakout
                    entry = close * (1 - SLIPPAGE)
                    sl = breakout_level + atr * 0.5
                    tp = entry - (entry * config['tp_pct'] / 100)
                    return ('SHORT', conv, entry, sl, tp, j)

        # Downside breakout attempt (price broke below BB lower)
        if bars[j]['low'] < bb_lo:
            breakout_level = bb_lo
            return_pct = (bars[i]['close'] - breakout_level) / breakout_level * 100
            if return_pct >= config['failure_return_pct']:
                wick_ratio = (min(bars[j]['open'], bars[j]['close']) - bars[j]['low']) / max(bars[j]['high'] - bars[j]['low'], 0.01)
                vol_ratio = bars[j]['volume'] / max(np.mean([bars[k]['volume'] for k in range(max(0,j-10), j)]), 1)
                taker_ratio = bars[j]['taker_buy_vol'] / max(bars[j]['volume'], 0.01)

                quality = 0.5
                if wick_ratio >= 0.4: quality += 0.15
                if vol_ratio >= 1.0: quality += 0.1
                if taker_ratio > 0.55: quality += 0.15  # buyers active
                reversal_strength = min(return_pct / 1.0, 1.0) * 0.3
                conv = min(quality + reversal_strength, 1.0)

                if conv >= min_conv:
                    entry = close * (1 + SLIPPAGE)
                    sl = breakout_level - atr * 0.5
                    tp = entry + (entry * config['tp_pct'] / 100)
                    return ('LONG', conv, entry, sl, tp, j)

    return None

# ── Backtest Engine ─────────────────────────────────────────────────────
def run_backtest(bars, config, fr_filter=False, sweep_filter=False):
    """Run backtest with given config and optional filters."""
    atrs = compute_atr(bars)
    bb_upper, bb_mid, bb_lower = compute_bb(bars)
    fr_data = load_derivatives() if fr_filter else {}

    trades = []
    open_pos = None
    equity = 200.0
    peak_equity = 200.0

    for i in range(50, len(bars)):
        # Check open position
        if open_pos:
            bar = bars[i]
            held_hours = (bar['ts'] - open_pos['opened_at']).total_seconds() / 3600

            # Check TP/SL
            hit_tp = False
            hit_sl = False
            exit_price = None

            if open_pos['direction'] == 'LONG':
                if bar['high'] >= open_pos['tp']:
                    hit_tp = True
                    exit_price = open_pos['tp']
                elif bar['low'] <= open_pos['sl']:
                    hit_sl = True
                    exit_price = open_pos['sl']
            else:  # SHORT
                if bar['low'] <= open_pos['tp']:
                    hit_tp = True
                    exit_price = open_pos['tp']
                elif bar['high'] >= open_pos['sl']:
                    hit_sl = True
                    exit_price = open_pos['sl']

            # Timeout
            if not hit_tp and not hit_sl and held_hours >= open_pos['hold_hours']:
                exit_price = bar['close']

            if exit_price is not None:
                if open_pos['direction'] == 'LONG':
                    pnl_pct = (exit_price - open_pos['entry']) / open_pos['entry']
                else:
                    pnl_pct = (open_pos['entry'] - exit_price) / open_pos['entry']

                pnl_pct -= FEE_RATE * 2  # entry + exit fees
                pnl_dollar = equity * pnl_pct
                equity += pnl_dollar

                trades.append({
                    'direction': open_pos['direction'],
                    'entry': open_pos['entry'],
                    'exit': exit_price,
                    'sl': open_pos['sl'],
                    'tp': open_pos['tp'],
                    'pnl_pct': round(pnl_pct * 100, 4),
                    'pnl_dollar': round(pnl_dollar, 2),
                    'outcome': 'WIN' if hit_tp else ('LOSS' if hit_sl else 'TIMEOUT'),
                    'conviction': open_pos['conviction'],
                    'held_hours': round(held_hours, 1),
                    'opened_at': open_pos['opened_at'].isoformat(),
                    'closed_at': bar['ts'].isoformat(),
                    'breakout_bar': open_pos.get('breakout_bar', -1),
                })

                if equity > peak_equity:
                    peak_equity = equity
                open_pos = None

        # Look for new signals (only if no open position)
        if open_pos is None:
            signal = detect_failed_breakout(bars, i, bb_upper, bb_lower, atrs, config)
            if signal:
                direction, conv, entry, sl, tp, breakout_idx = signal

                # Apply sweep magnitude filter
                if sweep_filter:
                    atr = atrs[i]
                    if atr > 0:
                        sweep_depth = abs(bars[breakout_idx]['high'] - bb_upper[breakout_idx]) if direction == 'SHORT' else abs(bb_lower[breakout_idx] - bars[breakout_idx]['low'])
                        sweep_atr = sweep_depth / atr
                        if sweep_atr < 0.08:  # 0.08 ATR minimum sweep
                            continue

                # Apply funding rate filter
                if fr_filter:
                    ts_key = bars[i]['ts'].strftime('%Y-%m-%d %H:%M:%S')
                    fr = fr_data.get(ts_key, None)
                    if fr is not None and abs(fr) < 0.00008:
                        continue  # Skip if FR too low

                # Check minimum SL distance
                sl_pct = abs(entry - sl) / entry
                if sl_pct < 0.003:  # 0.3% minimum
                    continue

                open_pos = {
                    'direction': direction,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'hold_hours': config['hold_hours'],
                    'conviction': conv,
                    'opened_at': bars[i]['ts'],
                    'breakout_bar': breakout_idx,
                }

    return compute_results(trades, equity, peak_equity, config)

def compute_results(trades, final_equity, peak_equity, config):
    """Compute backtest statistics."""
    if not trades:
        return {'config': config, 'total_trades': 0, 'error': 'No trades generated'}

    wins = [t for t in trades if t['outcome'] == 'WIN']
    losses = [t for t in trades if t['outcome'] == 'LOSS']
    timeouts = [t for t in trades if t['outcome'] == 'TIMEOUT']

    gross_profit = sum(t['pnl_dollar'] for t in wins)
    gross_loss = abs(sum(t['pnl_dollar'] for t in losses + timeouts))

    wr = len(wins) / len(trades) * 100 if trades else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    equity_curve = [200.0]
    for t in trades:
        equity_curve.append(equity_curve[-1] + t['pnl_dollar'])
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Monthly breakdown
    monthly = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
    for t in trades:
        month = t['opened_at'][:7]
        if t['outcome'] == 'WIN':
            monthly[month]['wins'] += 1
        else:
            monthly[month]['losses'] += 1
        monthly[month]['pnl'] += t['pnl_dollar']

    return {
        'config': config,
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'timeouts': len(timeouts),
        'win_rate': round(wr, 2),
        'profit_factor': round(pf, 3),
        'total_pnl_pct': round((final_equity - 200) / 200 * 100, 2),
        'final_equity': round(final_equity, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'avg_win_pnl': round(np.mean([t['pnl_dollar'] for t in wins]), 2) if wins else 0,
        'avg_loss_pnl': round(np.mean([t['pnl_dollar'] for t in losses + timeouts]), 2) if losses + timeouts else 0,
        'avg_held_hours': round(np.mean([t['held_hours'] for t in trades]), 1),
        'monthly': {k: dict(v) for k, v in sorted(monthly.items())},
        'meets_target': wr >= 75 and pf >= 2.0,
        'trades': trades[-10:],  # Last 10 trades for inspection
    }

# ── Main ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading data...")
    bars = load_ohlcv(DATA_FILE)
    print(f"Loaded {len(bars)} bars: {bars[0]['ts']} to {bars[-1]['ts']}")

    results = {}

    # ── Test 1: Baseline (default config) ───────────────────────────────
    print("\n" + "="*60)
    print("TEST 1: Baseline (default config)")
    print("="*60)
    config = {
        'tp_pct': DEFAULT_TP_PCT, 'sl_pct': DEFAULT_SL_PCT,
        'hold_hours': DEFAULT_HOLD_HOURS, 'min_conv': DEFAULT_MIN_CONV,
        'lookback': 48, 'failure_return_pct': 0.3,
    }
    r = run_backtest(bars, config)
    print(f"  Trades: {r['total_trades']} | WR: {r.get('win_rate',0)}% | PF: {r.get('profit_factor',0)} | PnL: {r.get('total_pnl_pct',0)}% | DD: {r.get('max_drawdown_pct',0)}%")
    results['baseline'] = r

    # ── Test 2: Parameter sweep ─────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 2: Parameter Sweep")
    print("="*60)
    best = None
    best_key = None
    sweep_results = []

    for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for sl in [0.5, 0.75, 1.0, 1.5, 2.0]:
            for hold in [8, 12, 16, 24, 32]:
                for conv in [0.5, 0.6, 0.7, 0.8]:
                    cfg = {
                        'tp_pct': tp, 'sl_pct': sl,
                        'hold_hours': hold, 'min_conv': conv,
                        'lookback': 48, 'failure_return_pct': 0.3,
                    }
                    r = run_backtest(bars, cfg)
                    wr = r.get('win_rate', 0)
                    pf = r.get('profit_factor', 0)
                    trades = r.get('total_trades', 0)

                    if trades >= 5:  # Minimum trades
                        score = (wr / 100) * pf if pf != float('inf') else 0
                        sweep_results.append({
                            'tp': tp, 'sl': sl, 'hold': hold, 'conv': conv,
                            'trades': trades, 'wr': wr, 'pf': pf,
                            'pnl': r.get('total_pnl_pct', 0),
                            'dd': r.get('max_drawdown_pct', 0),
                            'score': round(score, 3),
                            'meets_target': wr >= 75 and pf >= 2.0,
                        })
                        if best is None or (wr >= 75 and pf >= 2.0 and score > best.get('score', 0)):
                            best = r
                            best_key = f"tp{tp}_sl{sl}_h{hold}_c{conv}"

    # Sort by score
    sweep_results.sort(key=lambda x: x['score'], reverse=True)
    print(f"\n  Tested {len(sweep_results)} configs ({sum(1 for s in sweep_results if s['meets_target'])} meet target)")
    print("\n  Top 10:")
    for s in sweep_results[:10]:
        marker = "✅" if s['meets_target'] else "  "
        print(f"  {marker} TP={s['tp']}% SL={s['sl']}% Hold={s['hold']}h Conv={s['conv']} | {s['trades']}T {s['wr']}% WR {s['pf']}PF | Score={s['score']}")

    results['sweep'] = {
        'total_configs': len(sweep_results),
        'meets_target': sum(1 for s in sweep_results if s['meets_target']),
        'top_10': sweep_results[:10],
        'best_config': best_key,
    }

    if best:
        results['best_sweep'] = best

    # ── Test 3: With sweep magnitude filter ─────────────────────────────
    print("\n" + "="*60)
    print("TEST 3: Best config + Sweep Magnitude Filter (0.08 ATR min)")
    print("="*60)
    if best:
        r = run_backtest(bars, best['config'], sweep_filter=True)
        print(f"  Trades: {r['total_trades']} | WR: {r.get('win_rate',0)}% | PF: {r.get('profit_factor',0)} | PnL: {r.get('total_pnl_pct',0)}%")
        results['with_sweep_filter'] = r

    # ── Test 4: With funding rate filter ────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: Best config + Funding Rate Filter (|FR|>=0.00008)")
    print("="*60)
    if best:
        r = run_backtest(bars, best['config'], fr_filter=True)
        print(f"  Trades: {r['total_trades']} | WR: {r.get('win_rate',0)}% | PF: {r.get('profit_factor',0)} | PnL: {r.get('total_pnl_pct',0)}%")
        results['with_fr_filter'] = r

    # ── Test 5: Combined filters ────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 5: Best config + Both Filters Combined")
    print("="*60)
    if best:
        r = run_backtest(bars, best['config'], fr_filter=True, sweep_filter=True)
        print(f"  Trades: {r['total_trades']} | WR: {r.get('win_rate',0)}% | PF: {r.get('profit_factor',0)} | PnL: {r.get('total_pnl_pct',0)}%")
        results['with_both_filters'] = r

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, r in results.items():
        if isinstance(r, dict) and 'total_trades' in r:
            marker = "✅" if r.get('meets_target') else "❌"
            print(f"  {marker} {name}: {r['total_trades']}T {r.get('win_rate',0)}% WR {r.get('profit_factor',0)}PF {r.get('total_pnl_pct',0)}% PnL DD={r.get('max_drawdown_pct',0)}%")

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")
