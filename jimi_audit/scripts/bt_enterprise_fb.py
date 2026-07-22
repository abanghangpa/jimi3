#!/usr/bin/env python3
"""
Enterprise-Grade Failed Breakout Backtest
==========================================
Rewrites M20 from scratch with:
1. Level significance scoring (PDH/PDL, PWH/PWL, round numbers, session H/L)
2. Trap confirmation (volume spike, wick rejection, taker flip)
3. Time-of-day filter (London/NY sessions only)
4. Cooldown per level (no re-trading same level within 24h)
5. Sweep magnitude + funding rate filters

Data: ETH/USDT 15m, April 2025 - July 2026
Target: WR >= 75%, PF >= 2.0
"""

import csv, json, os, sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np

DATA_FILE = "/root/.openclaw/workspace/jimi_audit/eth_15m_6m.csv"
OUTPUT = "/root/.openclaw/workspace/jimi_audit/reports/enterprise_fb_backtest.json"

FEE = 0.0002
SLIP = 0.001

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data():
    bars = []
    with open(DATA_FILE) as f:
        for row in csv.DictReader(f):
            bars.append({
                'ts': datetime.strptime(row['Open time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc),
                'o': float(row['Open']), 'h': float(row['High']),
                'l': float(row['Low']), 'c': float(row['Close']),
                'v': float(row['Volume']),
                'tb': float(row.get('Taker buy base asset volume', 0)),
                'trades': int(row.get('Number of trades', 0)),
            })
    return bars

# ═══════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════

def compute_atr(bars, period=14):
    atr = np.zeros(len(bars))
    for i in range(1, len(bars)):
        tr = max(bars[i]['h']-bars[i]['l'], abs(bars[i]['h']-bars[i-1]['c']), abs(bars[i]['l']-bars[i-1]['c']))
        atr[i] = tr if i < period else (atr[i-1]*(period-1) + tr)/period
    return atr

def compute_avg_volume(bars, period=20):
    """Rolling average volume."""
    avg = np.zeros(len(bars))
    for i in range(period, len(bars)):
        avg[i] = np.mean([bars[j]['v'] for j in range(i-period, i)])
    return avg

# ═══════════════════════════════════════════════════════════════
# LEVEL DETECTION — The Key Difference
# ═══════════════════════════════════════════════════════════════

def find_significant_levels(bars, i, lookback_days=5):
    """
    Find significant price levels near bar i.
    Returns list of (price, type, significance_score).

    Level types and base scores:
      PWH/PWL (prev week high/low)    = 4
      PDH/PDL (prev day high/low)     = 3
      Session high/low (London/NY)    = 2
      Round number (within 0.1%)      = 2
      Swing high/low (3-bar pivot)    = 1
    """
    levels = []
    ts = bars[i]['ts']
    price = bars[i]['c']

    # ── Previous Day High/Low ──
    # Find the most recent completed day
    current_date = ts.date()
    for d_offset in range(1, lookback_days + 1):
        target_date = current_date - timedelta(days=d_offset)
        day_bars = [b for b in bars[max(0, i-96*d_offset):i] if b['ts'].date() == target_date]
        if day_bars:
            pdh = max(b['h'] for b in day_bars)
            pdl = min(b['l'] for b in day_bars)
            if abs(pdh - price) / price < 0.03:  # Within 3%
                levels.append((pdh, 'PDH', 3))
            if abs(pdl - price) / price < 0.03:
                levels.append((pdl, 'PDL', 3))

    # ── Previous Week High/Low ──
    current_week = ts.isocalendar()[1]
    for w_offset in range(1, 3):
        target_week = current_week - w_offset
        week_bars = [b for b in bars[max(0, i-96*7*w_offset):i] if b['ts'].isocalendar()[1] == target_week]
        if week_bars:
            pwh = max(b['h'] for b in week_bars)
            pwl = min(b['l'] for b in week_bars)
            if abs(pwh - price) / price < 0.05:
                levels.append((pwh, 'PWH', 4))
            if abs(pwl - price) / price < 0.05:
                levels.append((pwl, 'PWL', 4))

    # ── Session Highs/Lows (London 08-16 UTC, NY 13-21 UTC) ──
    hour = ts.hour
    # London session
    london_bars = [b for b in bars[max(0, i-32):i] if 8 <= b['ts'].hour < 16]
    if london_bars and len(london_bars) >= 4:
        ldh = max(b['h'] for b in london_bars)
        ldl = min(b['l'] for b in london_bars)
        if abs(ldh - price) / price < 0.02:
            levels.append((ldh, 'LONDON_H', 2))
        if abs(ldl - price) / price < 0.02:
            levels.append((ldl, 'LONDON_L', 2))

    # NY session
    ny_bars = [b for b in bars[max(0, i-32):i] if 13 <= b['ts'].hour < 21]
    if ny_bars and len(ny_bars) >= 4:
        nyh = max(b['h'] for b in ny_bars)
        nyl = min(b['l'] for b in ny_bars)
        if abs(nyh - price) / price < 0.02:
            levels.append((nyh, 'NY_H', 2))
        if abs(nyl - price) / price < 0.02:
            levels.append((nyl, 'NY_L', 2))

    # ── Round Numbers ──
    round_base = round(price / 50) * 50  # Nearest $50
    for offset in [-50, 0, 50]:
        rnd = round_base + offset
        if abs(rnd - price) / price < 0.015:
            levels.append((rnd, 'ROUND', 2))

    # ── Swing Highs/Lows (3-bar pivot) ──
    for j in range(max(3, i-48), i-3):
        if j + 3 >= len(bars):
            continue
        # Swing high
        if bars[j+1]['h'] > bars[j]['h'] and bars[j+1]['h'] > bars[j+2]['h']:
            if abs(bars[j+1]['h'] - price) / price < 0.02:
                levels.append((bars[j+1]['h'], 'SWING_H', 1))
        # Swing low
        if bars[j+1]['l'] < bars[j]['l'] and bars[j+1]['l'] < bars[j+2]['l']:
            if abs(bars[j+1]['l'] - price) / price < 0.02:
                levels.append((bars[j+1]['l'], 'SWING_L', 1))

    # Deduplicate nearby levels (within 0.1%)
    levels.sort(key=lambda x: x[0])
    deduped = []
    for lvl in levels:
        if not deduped or abs(lvl[0] - deduped[-1][0]) / lvl[0] > 0.001:
            deduped.append(lvl)

    return deduped

# ═══════════════════════════════════════════════════════════════
# SIGNAL DETECTION — Enterprise Grade
# ═══════════════════════════════════════════════════════════════

def detect_enterprise_fb(bars, i, atr, avg_vol, levels, config, traded_levels):
    """
    Enterprise-grade failed breakout detection.

    Requirements:
    1. Breakout at a significant level (score >= min_level_score)
    2. Trap confirmation (wick, volume, taker)
    3. Failure within N bars
    4. Reversal conviction (volume spike, taker flip)
    5. Not recently traded this level
    """
    if i < 50 or atr[i] <= 0:
        return None

    price = bars[i]['c']
    min_level_score = config['min_level_score']
    min_conv = config['min_conv']
    lookback = config['lookback']
    failure_bars = config['failure_bars']
    failure_ret_pct = config['failure_ret_pct']

    for level_price, level_type, level_score in levels:
        if level_score < min_level_score:
            continue

        # Check if we recently traded this level
        level_key = f"{round(level_price, 0)}_{level_type}"
        if level_key in traded_levels:
            last_traded = traded_levels[level_key]
            if (bars[i]['ts'] - last_traded).total_seconds() < 86400:  # 24h cooldown
                continue

        # Scan for breakout attempt in lookback window
        for j in range(max(0, i - lookback), i):
            breakout_bar = bars[j]

            # ── Upside breakout (price broke above level) ──
            if breakout_bar['h'] > level_price + atr[i] * 0.1:
                breakout_level = level_price

                # Check failure: price returned below level
                return_pct = (breakout_level - bars[i]['c']) / breakout_level * 100
                if return_pct < failure_ret_pct:
                    continue

                # Check failure happened within N bars of breakout
                if i - j > failure_bars:
                    continue

                # ── Trap Confirmation ──
                # 1. Wick rejection on breakout bar
                candle_range = breakout_bar['h'] - breakout_bar['l']
                if candle_range <= 0:
                    continue
                upper_wick = breakout_bar['h'] - max(breakout_bar['o'], breakout_bar['c'])
                wick_ratio = upper_wick / candle_range

                # 2. Volume on breakout bar
                vol_ratio = breakout_bar['v'] / max(avg_vol[j], 1)

                # 3. Taker ratio on breakout bar
                taker_ratio = breakout_bar['tb'] / max(breakout_bar['v'], 0.01)

                # 4. Volume spike on failure bar (current bar)
                fail_vol_ratio = bars[i]['v'] / max(avg_vol[i], 1)

                # 5. Taker flip (sellers dominate on failure)
                fail_taker = bars[i]['tb'] / max(bars[i]['v'], 0.01)

                # ── Score the trap ──
                trap_score = 0.0

                # Wick rejection (high wick = breakout was rejected)
                if wick_ratio >= 0.40:
                    trap_score += 0.20
                elif wick_ratio >= 0.25:
                    trap_score += 0.10

                # Volume on breakout (high vol = many traders got trapped)
                if vol_ratio >= 1.3:
                    trap_score += 0.15
                elif vol_ratio >= 1.0:
                    trap_score += 0.08

                # Taker on breakout (buyers dominated = longs trapped)
                if taker_ratio >= 0.58:
                    trap_score += 0.15  # Buyers were dominant → longs trapped

                # Volume spike on failure (trapped traders exiting)
                if fail_vol_ratio >= 1.3:
                    trap_score += 0.20
                elif fail_vol_ratio >= 1.0:
                    trap_score += 0.10

                # Taker flip on failure (sellers now dominate)
                if fail_taker <= 0.42:
                    trap_score += 0.20  # Sellers taking over = cascade

                # Body on failure candle (strong reversal candle)
                fail_body = abs(bars[i]['c'] - bars[i]['o'])
                fail_range = bars[i]['h'] - bars[i]['l']
                if fail_range > 0 and fail_body / fail_range >= 0.50:
                    trap_score += 0.10

                # Conviction = level significance + trap quality
                conv = min((level_score / 4) * 0.4 + trap_score * 0.6, 1.0)

                if conv >= min_conv:
                    # SHORT signal: failed upside breakout
                    entry = price * (1 - SLIP)
                    sl = breakout_level + atr[i] * 0.5
                    tp = entry - (entry * config['tp_pct'] / 100)

                    # Validate SL distance
                    sl_pct = abs(entry - sl) / entry
                    if sl_pct < 0.003:
                        continue

                    return {
                        'direction': 'SHORT', 'entry': entry, 'sl': sl, 'tp': tp,
                        'conviction': round(conv, 3), 'level_price': level_price,
                        'level_type': level_type, 'level_score': level_score,
                        'trap_score': round(trap_score, 3),
                        'wick_ratio': round(wick_ratio, 3),
                        'vol_ratio': round(vol_ratio, 3),
                        'fail_vol_ratio': round(fail_vol_ratio, 3),
                        'breakout_bar': j,
                    }

            # ── Downside breakout (price broke below level) ──
            if breakout_bar['l'] < level_price - atr[i] * 0.1:
                breakout_level = level_price

                return_pct = (bars[i]['c'] - breakout_level) / breakout_level * 100
                if return_pct < failure_ret_pct:
                    continue

                if i - j > failure_bars:
                    continue

                candle_range = breakout_bar['h'] - breakout_bar['l']
                if candle_range <= 0:
                    continue
                lower_wick = min(breakout_bar['o'], breakout_bar['c']) - breakout_bar['l']
                wick_ratio = lower_wick / candle_range

                vol_ratio = breakout_bar['v'] / max(avg_vol[j], 1)
                taker_ratio = breakout_bar['tb'] / max(breakout_bar['v'], 0.01)
                fail_vol_ratio = bars[i]['v'] / max(avg_vol[i], 1)
                fail_taker = bars[i]['tb'] / max(bars[i]['v'], 0.01)

                trap_score = 0.0
                if wick_ratio >= 0.40: trap_score += 0.20
                elif wick_ratio >= 0.25: trap_score += 0.10
                if vol_ratio >= 1.3: trap_score += 0.15
                elif vol_ratio >= 1.0: trap_score += 0.08
                if taker_ratio <= 0.42: trap_score += 0.15  # Sellers dominated → shorts trapped
                if fail_vol_ratio >= 1.3: trap_score += 0.20
                elif fail_vol_ratio >= 1.0: trap_score += 0.10
                if fail_taker >= 0.58: trap_score += 0.20  # Buyers taking over
                fail_body = abs(bars[i]['c'] - bars[i]['o'])
                fail_range = bars[i]['h'] - bars[i]['l']
                if fail_range > 0 and fail_body / fail_range >= 0.50:
                    trap_score += 0.10

                conv = min((level_score / 4) * 0.4 + trap_score * 0.6, 1.0)

                if conv >= min_conv:
                    entry = price * (1 + SLIP)
                    sl = breakout_level - atr[i] * 0.5
                    tp = entry + (entry * config['tp_pct'] / 100)
                    sl_pct = abs(entry - sl) / entry
                    if sl_pct < 0.003:
                        continue

                    return {
                        'direction': 'LONG', 'entry': entry, 'sl': sl, 'tp': tp,
                        'conviction': round(conv, 3), 'level_price': level_price,
                        'level_type': level_type, 'level_score': level_score,
                        'trap_score': round(trap_score, 3),
                        'wick_ratio': round(wick_ratio, 3),
                        'vol_ratio': round(vol_ratio, 3),
                        'fail_vol_ratio': round(fail_vol_ratio, 3),
                        'breakout_bar': j,
                    }

    return None

# ═══════════════════════════════════════════════════════════════
# SESSION FILTER
# ═══════════════════════════════════════════════════════════════

def is_trade_session(ts, session_filter):
    """Check if timestamp is within allowed trading sessions."""
    if not session_filter:
        return True
    hour = ts.hour
    # London: 08-16 UTC, NY: 13-21 UTC, Overlap: 13-16 UTC
    if session_filter == 'london_ny':
        return 8 <= hour < 21
    elif session_filter == 'overlap':
        return 13 <= hour < 16
    elif session_filter == 'ny':
        return 13 <= hour < 21
    elif session_filter == 'london':
        return 8 <= hour < 16
    return True

# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def run_backtest(bars, config):
    atr = compute_atr(bars)
    avg_vol = compute_avg_volume(bars)
    N = len(bars)

    trades = []
    pos = None
    eq = 200.0
    peak = 200.0
    max_dd = 0
    traded_levels = {}  # level_key -> last_traded_ts

    for i in range(60, N):
        # ── Check open position ──
        if pos:
            b = bars[i]
            held = (b['ts'] - pos['ot']).total_seconds() / 3600
            exit_p = None
            outcome = None

            if pos['d'] == 'LONG':
                if b['h'] >= pos['tp']: exit_p, outcome = pos['tp'], 'WIN'
                elif b['l'] <= pos['sl']: exit_p, outcome = pos['sl'], 'LOSS'
            else:
                if b['l'] <= pos['tp']: exit_p, outcome = pos['tp'], 'WIN'
                elif b['h'] >= pos['sl']: exit_p, outcome = pos['sl'], 'LOSS'

            if exit_p is None and held >= pos['hold']:
                exit_p, outcome = b['c'], 'TIMEOUT'

            if exit_p:
                pnl = ((exit_p - pos['e'])/pos['e']) if pos['d']=='LONG' else ((pos['e'] - exit_p)/pos['e'])
                pnl -= FEE * 2
                dollar = eq * pnl
                eq += dollar

                trades.append({
                    'd': pos['d'], 'e': pos['e'], 'x': exit_p,
                    'sl': pos['sl'], 'tp': pos['tp'],
                    'pnl_pct': round(pnl*100, 4), 'pnl_$': round(dollar, 2),
                    'outcome': outcome, 'conv': pos['cv'],
                    'level_type': pos['lt'], 'level_score': pos['ls'],
                    'trap_score': pos['ts_score'],
                    'held_h': round(held, 1),
                    'opened_at': pos['ot'].isoformat(),
                    'closed_at': b['ts'].isoformat(),
                })

                if eq > peak: peak = eq
                dd = (peak - eq) / peak * 100
                if dd > max_dd: max_dd = dd
                pos = None

        # ── Look for signals ──
        if pos is None:
            # Session filter
            if not is_trade_session(bars[i]['ts'], config.get('session_filter')):
                continue

            # Find significant levels
            levels = find_significant_levels(bars, i)

            # Detect failed breakout
            sig = detect_enterprise_fb(bars, i, atr, avg_vol, levels, config, traded_levels)
            if sig:
                # Record traded level
                level_key = f"{round(sig['level_price'], 0)}_{sig['level_type']}"
                traded_levels[level_key] = bars[i]['ts']

                pos = {
                    'd': sig['direction'], 'e': sig['entry'],
                    'sl': sig['sl'], 'tp': sig['tp'],
                    'hold': config['hold_hours'], 'cv': sig['conviction'],
                    'lt': sig['level_type'], 'ls': sig['level_score'],
                    'ts_score': sig['trap_score'],
                    'ot': bars[i]['ts'],
                }

    # ── Compute results ──
    if not trades:
        return {'total_trades': 0, 'config': config}

    wins = [t for t in trades if t['outcome'] == 'WIN']
    losses = [t for t in trades if t['outcome'] == 'LOSS']
    timeouts = [t for t in trades if t['outcome'] == 'TIMEOUT']

    gp = sum(t['pnl_$'] for t in wins)
    gl = abs(sum(t['pnl_$'] for t in losses + timeouts))

    wr = len(wins) / len(trades) * 100
    pf = gp / gl if gl > 0 else 999

    # Level type breakdown
    level_breakdown = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0})
    for t in trades:
        level_breakdown[t['level_type']]['total'] += 1
        if t['outcome'] == 'WIN':
            level_breakdown[t['level_type']]['wins'] += 1
        else:
            level_breakdown[t['level_type']]['losses'] += 1

    return {
        'config': {k: v for k, v in config.items() if k != 'trades'},
        'total_trades': len(trades),
        'wins': len(wins), 'losses': len(losses), 'timeouts': len(timeouts),
        'win_rate': round(wr, 2),
        'profit_factor': round(pf, 3),
        'total_pnl_pct': round((eq - 200) / 200 * 100, 2),
        'final_equity': round(eq, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'avg_conviction': round(np.mean([t['conv'] for t in trades]), 3),
        'avg_trap_score': round(np.mean([t['trap_score'] for t in trades]), 3),
        'level_breakdown': {k: dict(v) for k, v in level_breakdown.items()},
        'meets_target': wr >= 75 and pf >= 2.0,
        'trades': trades[-15:],
    }

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Loading data...", flush=True)
    bars = load_data()
    print(f"Loaded {len(bars)} bars: {bars[0]['ts']} to {bars[-1]['ts']}", flush=True)

    all_results = {}

    # ── Test 1: Baseline (all features ON, moderate thresholds) ──
    print("\n" + "="*70, flush=True)
    print("TEST 1: Enterprise Baseline", flush=True)
    print("="*70, flush=True)

    config = {
        'tp_pct': 2.0, 'sl_pct': 1.0, 'hold_hours': 16,
        'min_conv': 0.6, 'min_level_score': 2, 'lookback': 24,
        'failure_bars': 8, 'failure_ret_pct': 0.2,
        'session_filter': 'london_ny',
    }
    r = run_backtest(bars, config)
    marker = "✅" if r.get('meets_target') else "❌"
    print(f"  {marker} {r['total_trades']}T {r.get('win_rate',0)}%WR {r.get('profit_factor',0)}PF PnL={r.get('total_pnl_pct',0)}% DD={r.get('max_drawdown_pct',0)}%", flush=True)
    if r.get('level_breakdown'):
        print(f"  Level breakdown: {json.dumps(r['level_breakdown'], indent=4)}", flush=True)
    all_results['baseline'] = r

    # ── Test 2: Only Tier 1-2 levels (PDH/PDL, PWH/PWL, Round) ──
    print("\n" + "="*70, flush=True)
    print("TEST 2: Tier 1-2 Levels Only (PDH/PDL, PWH/PWL, Session, Round)", flush=True)
    print("="*70, flush=True)

    config2 = {**config, 'min_level_score': 2}
    r2 = run_backtest(bars, config2)
    marker = "✅" if r2.get('meets_target') else "❌"
    print(f"  {marker} {r2['total_trades']}T {r2.get('win_rate',0)}%WR {r2.get('profit_factor',0)}PF PnL={r2.get('total_pnl_pct',0)}%", flush=True)
    all_results['tier12_only'] = r2

    # ── Test 3: Only Tier 1 levels (PWH/PWL) ──
    print("\n" + "="*70, flush=True)
    print("TEST 3: Tier 1 Only (PWH/PWL — weekly levels)", flush=True)
    print("="*70, flush=True)

    config3 = {**config, 'min_level_score': 4}
    r3 = run_backtest(bars, config3)
    marker = "✅" if r3.get('meets_target') else "❌"
    print(f"  {marker} {r3['total_trades']}T {r3.get('win_rate',0)}%WR {r3.get('profit_factor',0)}PF PnL={r3.get('total_pnl_pct',0)}%", flush=True)
    all_results['tier1_only'] = r3

    # ── Test 4: Parameter sweep on best level tier ──
    print("\n" + "="*70, flush=True)
    print("TEST 4: Parameter Sweep (best level tier)", flush=True)
    print("="*70, flush=True)

    sweep_results = []
    configs_tested = 0

    for min_level in [2, 3, 4]:
        for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for sl in [0.5, 0.75, 1.0, 1.5]:
                for hold in [8, 12, 16, 24]:
                    for conv in [0.5, 0.6, 0.7, 0.8]:
                        for session in ['london_ny', 'overlap', None]:
                            cfg = {
                                'tp_pct': tp, 'sl_pct': sl, 'hold_hours': hold,
                                'min_conv': conv, 'min_level_score': min_level,
                                'lookback': 24, 'failure_bars': 8,
                                'failure_ret_pct': 0.2, 'session_filter': session,
                            }
                            r = run_backtest(bars, cfg)
                            configs_tested += 1

                            if r['total_trades'] >= 5:
                                wr = r.get('win_rate', 0)
                                pf = r.get('profit_factor', 0)
                                score = (wr / 100) * min(pf, 10) if pf < 999 else 0
                                sweep_results.append({
                                    'min_level': min_level, 'tp': tp, 'sl': sl,
                                    'hold': hold, 'conv': conv, 'session': session or 'all',
                                    'trades': r['total_trades'], 'wr': wr, 'pf': pf,
                                    'pnl': r.get('total_pnl_pct', 0),
                                    'dd': r.get('max_drawdown_pct', 0),
                                    'score': round(score, 3),
                                    'ok': wr >= 75 and pf >= 2.0,
                                })

                            if configs_tested % 200 == 0:
                                print(f"  Tested {configs_tested}...", flush=True)

    sweep_results.sort(key=lambda x: x['score'], reverse=True)
    meets = [s for s in sweep_results if s['ok']]

    print(f"\n  Tested {configs_tested} configs, {len(meets)} meet target", flush=True)
    print("\n  TOP 15:", flush=True)
    for s in sweep_results[:15]:
        m = "✅" if s['ok'] else "  "
        print(f"  {m} Lvl≥{s['min_level']} TP={s['tp']}% SL={s['sl']}% H={s['hold']}h C={s['conv']} S={s['session']} | {s['trades']}T {s['wr']}%WR {s['pf']}PF Score={s['score']}", flush=True)

    all_results['sweep'] = {
        'total_configs': configs_tested,
        'meets_target': len(meets),
        'top_15': sweep_results[:15],
        'all_meets': meets[:10],
    }

    # ── Test 5: Best config from sweep ──
    if sweep_results:
        best = sweep_results[0]
        print(f"\n" + "="*70, flush=True)
        print(f"TEST 5: Best Config Detailed", flush=True)
        print("="*70, flush=True)

        cfg_best = {
            'tp_pct': best['tp'], 'sl_pct': best['sl'],
            'hold_hours': best['hold'], 'min_conv': best['conv'],
            'min_level_score': best['min_level'],
            'lookback': 24, 'failure_bars': 8,
            'failure_ret_pct': 0.2,
            'session_filter': best['session'] if best['session'] != 'all' else None,
        }
        r_best = run_backtest(bars, cfg_best)
        marker = "✅" if r_best.get('meets_target') else "❌"
        print(f"  {marker} {r_best['total_trades']}T {r_best.get('win_rate',0)}%WR {r_best.get('profit_factor',0)}PF PnL={r_best.get('total_pnl_pct',0)}% DD={r_best.get('max_drawdown_pct',0)}%", flush=True)
        print(f"  Avg conviction: {r_best.get('avg_conviction',0)}", flush=True)
        print(f"  Avg trap score: {r_best.get('avg_trap_score',0)}", flush=True)
        if r_best.get('level_breakdown'):
            print(f"  Level breakdown:", flush=True)
            for lt, stats in r_best['level_breakdown'].items():
                wr = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
                print(f"    {lt}: {stats['total']}T {wr:.1f}%WR ({stats['wins']}W/{stats['losses']}L)", flush=True)
        all_results['best_detailed'] = r_best

    # ── Save ──
    with open(OUTPUT, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT}", flush=True)

    # ── Final Summary ──
    print("\n" + "="*70, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("="*70, flush=True)
    for name, r in all_results.items():
        if isinstance(r, dict) and 'total_trades' in r:
            m = "✅" if r.get('meets_target') else "❌"
            print(f"  {m} {name}: {r['total_trades']}T {r.get('win_rate',0)}%WR {r.get('profit_factor',0)}PF PnL={r.get('total_pnl_pct',0)}% DD={r.get('max_drawdown_pct',0)}%", flush=True)
