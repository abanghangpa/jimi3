#!/usr/bin/env python3
"""
Combo Strategy Backtest
========================
Tests failed_breakout as confirmation module paired with event triggers:

1. OBI + failed_breakout confirmation
2. Liquidity_grab + failed_breakout confirmation
3. OBI + liquidity_grab + failed_breakout (triple)

With filters: EMA200 trend, session overlap, volume >= 1.0

Data: ETH/USDT 15m, April 2025 - July 2026
Target: WR >= 75%, PF >= 2.0
"""

import csv, json, sys
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

DATA_FILE = "/root/.openclaw/workspace/jimi_audit/eth_15m_6m.csv"
OUTPUT = "/root/.openclaw/workspace/jimi_audit/reports/combo_backtest.json"
FEE = 0.0002
SLIP = 0.001

# ═══════════════════════════════════════════════════════════════
# DATA & INDICATORS
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
            })
    return bars

def compute_ema(data, period):
    ema = np.zeros(len(data))
    ema[period-1] = np.mean(data[:period])
    m = 2 / (period + 1)
    for i in range(period, len(data)):
        ema[i] = data[i] * m + ema[i-1] * (1 - m)
    return ema

def compute_atr(bars, period=14):
    atr = np.zeros(len(bars))
    for i in range(1, len(bars)):
        tr = max(bars[i]['h']-bars[i]['l'], abs(bars[i]['h']-bars[i-1]['c']), abs(bars[i]['l']-bars[i-1]['c']))
        atr[i] = tr if i < period else (atr[i-1]*(period-1) + tr)/period
    return atr

def compute_avg_vol(bars, period=20):
    avg = np.zeros(len(bars))
    for i in range(period, len(bars)):
        avg[i] = np.mean([bars[j]['v'] for j in range(i-period, i)])
    return avg

def compute_bb(bars, period=20, std_mult=2.0):
    closes = np.array([b['c'] for b in bars])
    upper = np.zeros(len(bars))
    lower = np.zeros(len(bars))
    for i in range(period-1, len(bars)):
        w = closes[i-period+1:i+1]
        m = np.mean(w)
        s = np.std(w)
        upper[i] = m + std_mult * s
        lower[i] = m - std_mult * s
    return upper, lower

# ═══════════════════════════════════════════════════════════════
# SIGNAL DETECTORS
# ═══════════════════════════════════════════════════════════════

def detect_obi(bars, i, taker_threshold, avg_vol, ema200, require_trend, min_vol_ratio):
    """Detect orderbook imbalance (taker volume proxy)."""
    taker = bars[i]['tb'] / max(bars[i]['v'], 0.01)
    vol_r = bars[i]['v'] / max(avg_vol[i], 1) if avg_vol[i] > 0 else 0

    if vol_r < min_vol_ratio:
        return None

    direction = None
    if taker >= taker_threshold:
        direction = 'LONG'
        if require_trend and bars[i]['c'] < ema200[i]:
            return None
    elif taker <= (1 - taker_threshold):
        direction = 'SHORT'
        if require_trend and bars[i]['c'] > ema200[i]:
            return None

    if direction:
        conv = 0.5 + abs(taker - 0.5) * 2
        conv = min(conv, 1.0)
        return {'type': 'OBI', 'direction': direction, 'conviction': conv, 'taker': round(taker, 4)}
    return None

def detect_liquidity_grab(bars, i, atr, lookback=20, depth_min=0.001, depth_max=0.02):
    """Detect liquidity grab (sweep of swing level + reclaim)."""
    if i < lookback + 5:
        return None

    price = bars[i]['c']

    # Find recent swing levels
    for j in range(max(3, i-lookback*3), i-3):
        # Swing high
        if j+3 < len(bars) and bars[j+1]['h'] > bars[j]['h'] and bars[j+1]['h'] > bars[j+2]['h']:
            level = bars[j+1]['h']
            # Check if current bar or recent bar swept above then failed
            for k in range(max(j+1, i-lookback), i+1):
                if bars[k]['h'] > level:
                    sweep_depth = (bars[k]['h'] - level) / level
                    if depth_min <= sweep_depth <= depth_max:
                        # Check if it reclaimed (closed below)
                        if bars[k]['c'] < level:
                            wick = bars[k]['h'] - max(bars[k]['o'], bars[k]['c'])
                            rng = bars[k]['h'] - bars[k]['l']
                            wick_ratio = wick / rng if rng > 0 else 0
                            if wick_ratio >= 0.3:
                                return {'type': 'LG', 'direction': 'SHORT', 'conviction': 0.6 + wick_ratio * 0.3,
                                        'level': round(level, 2), 'sweep_depth': round(sweep_depth*100, 3)}

        # Swing low
        if j+3 < len(bars) and bars[j+1]['l'] < bars[j]['l'] and bars[j+1]['l'] < bars[j+2]['l']:
            level = bars[j+1]['l']
            for k in range(max(j+1, i-lookback), i+1):
                if bars[k]['l'] < level:
                    sweep_depth = (level - bars[k]['l']) / level
                    if depth_min <= sweep_depth <= depth_max:
                        if bars[k]['c'] > level:
                            wick = min(bars[k]['o'], bars[k]['c']) - bars[k]['l']
                            rng = bars[k]['h'] - bars[k]['l']
                            wick_ratio = wick / rng if rng > 0 else 0
                            if wick_ratio >= 0.3:
                                return {'type': 'LG', 'direction': 'LONG', 'conviction': 0.6 + wick_ratio * 0.3,
                                        'level': round(level, 2), 'sweep_depth': round(sweep_depth*100, 3)}
    return None

def detect_failed_breakout(bars, i, bb_upper, bb_lower, atr, lookback=24, min_conv=0.5):
    """Detect failed breakout (price broke BB then reversed)."""
    if i < lookback + 5:
        return None

    price = bars[i]['c']

    for j in range(max(0, i-lookback), i):
        # Upside breakout fail
        if bars[j]['h'] > bb_upper[j] and bb_upper[j] > 0:
            ret = (bb_upper[j] - price) / bb_upper[j] * 100
            if ret >= 0.2:
                wick = (bars[j]['h'] - max(bars[j]['o'], bars[j]['c'])) / max(bars[j]['h'] - bars[j]['l'], 0.01)
                vol_r = bars[j]['v'] / max(np.mean([bars[k]['v'] for k in range(max(0,j-10),j)]), 1)
                taker = bars[j]['tb'] / max(bars[j]['v'], 0.01)
                conv = 0.5
                if wick >= 0.4: conv += 0.15
                if vol_r >= 1.0: conv += 0.1
                if taker >= 0.58: conv += 0.15
                conv += min(ret/1.0, 1.0) * 0.1
                conv = min(conv, 1.0)
                if conv >= min_conv:
                    return {'type': 'FB', 'direction': 'SHORT', 'conviction': round(conv, 3)}

        # Downside breakout fail
        if bars[j]['l'] < bb_lower[j] and bb_lower[j] > 0:
            ret = (price - bb_lower[j]) / bb_lower[j] * 100
            if ret >= 0.2:
                wick = (min(bars[j]['o'], bars[j]['c']) - bars[j]['l']) / max(bars[j]['h'] - bars[j]['l'], 0.01)
                vol_r = bars[j]['v'] / max(np.mean([bars[k]['v'] for k in range(max(0,j-10),j)]), 1)
                taker = bars[j]['tb'] / max(bars[j]['v'], 0.01)
                conv = 0.5
                if wick >= 0.4: conv += 0.15
                if vol_r >= 1.0: conv += 0.1
                if taker <= 0.42: conv += 0.15
                conv += min(ret/1.0, 1.0) * 0.1
                conv = min(conv, 1.0)
                if conv >= min_conv:
                    return {'type': 'FB', 'direction': 'LONG', 'conviction': round(conv, 3)}
    return None

# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def run_combo_backtest(bars, config):
    N = len(bars)
    closes = np.array([b['c'] for b in bars])
    atr = compute_atr(bars)
    avg_vol = compute_avg_vol(bars)
    ema200 = compute_ema(closes, 200)
    bb_upper, bb_lower = compute_bb(bars)

    trades = []
    pos = None
    eq = 200.0
    peak = 200.0
    max_dd = 0

    tp_pct = config['tp_pct']
    sl_pct = config['sl_pct']
    hold_h = config['hold_hours']
    session = config.get('session')
    require_trend = config.get('require_trend', True)
    min_vol_ratio = config.get('min_vol_ratio', 1.0)
    taker_t = config.get('taker_threshold', 0.65)
    combo_mode = config.get('combo_mode', 'OBI_FB')  # OBI_FB, LG_FB, OBI_LG_FB
    min_combo_conv = config.get('min_combo_conv', 0.6)

    for i in range(210, N):
        # Check position
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
            if exit_p is None and held >= hold_h:
                exit_p, outcome = b['c'], 'TIMEOUT'
            if exit_p:
                pnl = ((exit_p-pos['e'])/pos['e']) if pos['d']=='LONG' else ((pos['e']-exit_p)/pos['e'])
                pnl -= FEE*2
                eq += eq * pnl
                trades.append({'o': outcome, 'pnl': round(pnl*100,4), '$': round(eq*pnl,2), 'd': pos['d'], 'held': round(held,1)})
                if eq > peak: peak = eq
                dd = (peak-eq)/peak*100
                if dd > max_dd: max_dd = dd
                pos = None

        # Look for combo signals
        if pos is None:
            ts = bars[i]['ts']
            if session:
                h = ts.hour
                if session == 'ol' and not (13 <= h < 16): continue
                elif session == 'ln' and not (8 <= h < 21): continue

            # Detect signals
            obi = detect_obi(bars, i, taker_t, avg_vol, ema200, require_trend, min_vol_ratio)
            lg = detect_liquidity_grab(bars, i, atr)
            fb = detect_failed_breakout(bars, i, bb_upper, bb_lower, atr)

            # Apply combo logic
            direction = None
            combo_conv = 0
            combo_name = ""

            if combo_mode == 'OBI_FB':
                # OBI + failed_breakout same direction
                if obi and fb and obi['direction'] == fb['direction']:
                    direction = obi['direction']
                    combo_conv = (obi['conviction'] + fb['conviction']) / 2
                    combo_name = f"OBI+FB"

            elif combo_mode == 'LG_FB':
                # Liquidity_grab + failed_breakout same direction
                if lg and fb and lg['direction'] == fb['direction']:
                    direction = lg['direction']
                    combo_conv = (lg['conviction'] + fb['conviction']) / 2
                    combo_name = f"LG+FB"

            elif combo_mode == 'OBI_LG_FB':
                # Triple: OBI + liquidity_grab + failed_breakout
                if obi and lg and fb:
                    dirs = [obi['direction'], lg['direction'], fb['direction']]
                    if len(set(dirs)) == 1:  # All same direction
                        direction = dirs[0]
                        combo_conv = (obi['conviction'] + lg['conviction'] + fb['conviction']) / 3
                        combo_name = f"OBI+LG+FB"

            elif combo_mode == 'OBI_LG':
                # OBI + liquidity_grab
                if obi and lg and obi['direction'] == lg['direction']:
                    direction = obi['direction']
                    combo_conv = (obi['conviction'] + lg['conviction']) / 2
                    combo_name = f"OBI+LG"

            elif combo_mode == 'OBI_ONLY':
                # OBI standalone (baseline)
                if obi:
                    direction = obi['direction']
                    combo_conv = obi['conviction']
                    combo_name = "OBI"

            elif combo_mode == 'LG_ONLY':
                # Liquidity_grab standalone
                if lg:
                    direction = lg['direction']
                    combo_conv = lg['conviction']
                    combo_name = "LG"

            elif combo_mode == 'FB_ONLY':
                # Failed_breakout standalone (baseline)
                if fb:
                    direction = fb['direction']
                    combo_conv = fb['conviction']
                    combo_name = "FB"

            # Execute signal
            if direction and combo_conv >= min_combo_conv:
                e = closes[i] * (1+SLIP if direction == 'LONG' else 1-SLIP)
                sl_d = e * sl_pct / 100
                tp_d = e * tp_pct / 100
                if direction == 'LONG':
                    sl_p, tp_p = e - sl_d, e + tp_d
                else:
                    sl_p, tp_p = e + sl_d, e - tp_d

                # Min SL check
                if abs(e - sl_p) / e < 0.003:
                    continue

                pos = {'d': direction, 'e': e, 'sl': sl_p, 'tp': tp_p, 'ot': ts, 'combo': combo_name}

    # Results
    if not trades:
        return {'total_trades': 0, 'config': {k:v for k,v in config.items() if k != 'bars'}}

    wins = sum(1 for t in trades if t['o']=='WIN')
    losses = sum(1 for t in trades if t['o']=='LOSS')
    timeouts = sum(1 for t in trades if t['o']=='TIMEOUT')
    gp = sum(t['$'] for t in trades if t['o']=='WIN')
    gl = abs(sum(t['$'] for t in trades if t['o']!='WIN'))
    wr = wins/len(trades)*100
    pf = gp/gl if gl > 0 else 999

    return {
        'config': {k: v for k, v in config.items() if k not in ['bars']},
        'total_trades': len(trades), 'wins': wins, 'losses': losses, 'timeouts': timeouts,
        'win_rate': round(wr, 2), 'profit_factor': round(pf, 3),
        'total_pnl_pct': round((eq-200)/200*100, 2), 'final_equity': round(eq, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'avg_held': round(np.mean([t['held'] for t in trades]), 1),
        'meets_target': wr >= 75 and pf >= 2.0,
    }

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Loading data...", flush=True)
    bars = load_data()
    print(f"Loaded {len(bars)} bars", flush=True)

    all_results = {}

    # Base config
    base = {
        'tp_pct': 1.0, 'sl_pct': 1.5, 'hold_hours': 12,
        'session': 'ol', 'require_trend': True, 'min_vol_ratio': 1.0,
        'taker_threshold': 0.65, 'min_combo_conv': 0.55,
    }

    # ── Test each combo mode ──
    modes = [
        ('FB_ONLY', 'Failed Breakout Only (baseline)'),
        ('OBI_ONLY', 'OBI Only (baseline)'),
        ('LG_ONLY', 'Liquidity Grab Only (baseline)'),
        ('OBI_FB', 'OBI + Failed Breakout'),
        ('LG_FB', 'Liquidity Grab + Failed Breakout'),
        ('OBI_LG', 'OBI + Liquidity Grab'),
        ('OBI_LG_FB', 'Triple: OBI + Liquidity Grab + Failed Breakout'),
    ]

    print("\n" + "="*70, flush=True)
    print("COMBO BACKTEST: All Modes", flush=True)
    print("="*70, flush=True)

    for mode, desc in modes:
        cfg = {**base, 'combo_mode': mode}
        r = run_combo_backtest(bars, cfg)
        m = "✅" if r.get('meets_target') else "❌"
        print(f"  {m} {mode:15s} | {r['total_trades']:4d}T {r.get('win_rate',0):5.1f}%WR {r.get('profit_factor',0):5.2f}PF PnL={r.get('total_pnl_pct',0):7.2f}% DD={r.get('max_drawdown_pct',0):5.1f}% AvgHold={r.get('avg_held',0):4.1f}h", flush=True)
        all_results[mode] = r

    # ── Sweep best combo modes ──
    print("\n" + "="*70, flush=True)
    print("SWEEP: Best Combo Modes", flush=True)
    print("="*70, flush=True)

    sweep = []
    tested = 0

    for mode in ['OBI_FB', 'LG_FB', 'OBI_LG', 'OBI_LG_FB']:
        for taker in [0.60, 0.65, 0.70, 0.75, 0.80]:
            for tp in [0.5, 1.0, 1.5, 2.0]:
                for sl in [0.5, 0.75, 1.0, 1.5]:
                    for hold in [8, 12, 16]:
                        for min_conv in [0.5, 0.6, 0.7]:
                            for vol_min in [0.5, 1.0, 1.5]:
                                cfg = {
                                    'tp_pct': tp, 'sl_pct': sl, 'hold_hours': hold,
                                    'session': 'ol', 'require_trend': True,
                                    'min_vol_ratio': vol_min, 'taker_threshold': taker,
                                    'combo_mode': mode, 'min_combo_conv': min_conv,
                                }
                                r = run_combo_backtest(bars, cfg)
                                tested += 1

                                if r['total_trades'] >= 5:
                                    wr = r.get('win_rate', 0)
                                    pf = r.get('profit_factor', 0)
                                    score = (wr/100) * min(pf, 10) if pf < 999 else 0
                                    sweep.append({
                                        'mode': mode, 'taker': taker, 'tp': tp, 'sl': sl,
                                        'hold': hold, 'min_conv': min_conv, 'vol': vol_min,
                                        'trades': r['total_trades'], 'wr': wr, 'pf': pf,
                                        'pnl': r.get('total_pnl_pct', 0), 'dd': r.get('max_drawdown_pct', 0),
                                        'score': round(score, 3), 'ok': wr >= 75 and pf >= 2.0,
                                    })

                                if tested % 500 == 0:
                                    print(f"  {tested}...", flush=True)

    sweep.sort(key=lambda x: x['score'], reverse=True)
    meets = [s for s in sweep if s['ok']]

    print(f"\n  Tested {tested} configs, {len(meets)} meet target", flush=True)

    if meets:
        print(f"\n  ✅ MEETS TARGET ({len(meets)}):", flush=True)
        for s in meets[:20]:
            print(f"  ✅ {s['mode']:12s} T>={s['taker']} TP={s['tp']}% SL={s['sl']}% H={s['hold']}h C>={s['min_conv']} V>={s['vol']} | {s['trades']}T {s['wr']}%WR {s['pf']}PF PnL={s['pnl']}%", flush=True)

    print(f"\n  TOP 15 by score:", flush=True)
    for s in sweep[:15]:
        m = "✅" if s['ok'] else "  "
        print(f"  {m} {s['mode']:12s} T>={s['taker']} TP={s['tp']}% SL={s['sl']}% H={s['hold']}h C>={s['min_conv']} V>={s['vol']} | {s['trades']}T {s['wr']}%WR {s['pf']}PF Score={s['score']}", flush=True)

    all_results['sweep'] = {
        'tested': tested, 'meets_target': len(meets),
        'top_15': sweep[:15], 'all_meets': meets[:30],
    }

    # Save
    with open(OUTPUT, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT}", flush=True)
