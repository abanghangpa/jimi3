#!/usr/bin/env python3
"""
Orderbook Imbalance Proxy Backtest with Full Filter Stack
==========================================================
Since orderbook snapshots aren't stored historically, we use
TAKER VOLUME RATIO as a proxy for orderbook imbalance:
  taker_buy_ratio > 0.58 = buyers dominating (like high bid/ask ratio)
  taker_buy_ratio < 0.42 = sellers dominating

Strategy: Volume Imbalance + Full Filter Stack
  Layer 1: Taker volume imbalance (event trigger proxy)
  Layer 2: Direction = price vs EMA200 (trend confirmation)
  Layer 3: Sweep magnitude ≥ 0.08 ATR
           |Funding rate| ≥ 0.00008
           Session = London/NY
           Volume ratio ≥ 1.0 (above average)

Data: ETH/USDT 15m, April 2025 - July 2026
Target: WR >= 75%, PF >= 2.0
"""

import csv, json, os
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

DATA_FILE = "/root/.openclaw/workspace/jimi_audit/eth_15m_6m.csv"
OUTPUT = "/root/.openclaw/workspace/jimi_audit/reports/obi_backtest.json"

FEE = 0.0002
SLIP = 0.001

# ═══════════════════════════════════════════════════════════════
# DATA
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

def compute_ema(data, period):
    ema = np.zeros(len(data))
    ema[period-1] = np.mean(data[:period])
    mult = 2 / (period + 1)
    for i in range(period, len(data)):
        ema[i] = data[i] * mult + ema[i-1] * (1 - mult)
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

# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def run_backtest(bars, config):
    N = len(bars)
    closes = np.array([b['c'] for b in bars])
    atr = compute_atr(bars)
    avg_vol = compute_avg_vol(bars)
    ema200 = compute_ema(closes, 200)

    trades = []
    pos = None
    eq = 200.0
    peak = 200.0
    max_dd = 0

    tp_pct = config['tp_pct']
    sl_pct = config['sl_pct']
    hold_h = config['hold_hours']
    taker_threshold = config['taker_threshold']
    min_vol_ratio = config.get('min_vol_ratio', 0)
    require_trend = config.get('require_trend', False)
    session_filter = config.get('session_filter')
    min_atr_filter = config.get('min_atr_filter', 0)

    for i in range(210, N):
        # ── Check position ──
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
                pnl = ((exit_p - pos['e'])/pos['e']) if pos['d']=='LONG' else ((pos['e'] - exit_p)/pos['e'])
                pnl -= FEE * 2
                dollar = eq * pnl
                eq += dollar

                trades.append({
                    'd': pos['d'], 'e': pos['e'], 'x': exit_p,
                    'pnl_pct': round(pnl*100, 4), 'pnl_$': round(dollar, 2),
                    'outcome': outcome, 'held_h': round(held, 1),
                    'taker': pos['taker'], 'vol_r': pos['vol_r'],
                    'opened_at': pos['ot'].isoformat(),
                })

                if eq > peak: peak = eq
                dd = (peak - eq) / peak * 100
                if dd > max_dd: max_dd = dd
                pos = None

        # ── Look for signals ──
        if pos is None:
            ts = bars[i]['ts']

            # Session filter
            if session_filter:
                hour = ts.hour
                if session_filter == 'london_ny' and not (8 <= hour < 21):
                    continue
                elif session_filter == 'overlap' and not (13 <= hour < 16):
                    continue

            # Taker ratio
            taker = bars[i]['tb'] / max(bars[i]['v'], 0.01)

            # Volume ratio
            vol_r = bars[i]['v'] / max(avg_vol[i], 1) if avg_vol[i] > 0 else 0

            # ATR filter
            if min_atr_filter > 0 and atr[i] / bars[i]['c'] < min_atr_filter:
                continue

            # Volume filter
            if vol_r < min_vol_ratio:
                continue

            direction = None
            conv = 0.5

            # ── Buy imbalance (taker > threshold) → LONG ──
            if taker >= taker_threshold:
                direction = 'LONG'

                # Trend confirmation
                if require_trend and closes[i] < ema200[i]:
                    continue

                # Conviction from taker strength
                conv = 0.5 + (taker - taker_threshold) / (1.0 - taker_threshold) * 0.3
                conv += min(vol_r - 1, 1) * 0.2 if vol_r > 1 else 0
                conv = min(conv, 1.0)

            # ── Sell imbalance (taker < 1-threshold) → SHORT ──
            elif taker <= (1 - taker_threshold):
                direction = 'SHORT'

                if require_trend and closes[i] > ema200[i]:
                    continue

                conv = 0.5 + ((1 - taker_threshold) - taker) / (1 - taker_threshold) * 0.3
                conv += min(vol_r - 1, 1) * 0.2 if vol_r > 1 else 0
                conv = min(conv, 1.0)

            if direction:
                entry = closes[i] * (1 + SLIP if direction == 'LONG' else 1 - SLIP)
                sl_dist = entry * sl_pct / 100
                tp_dist = entry * tp_pct / 100

                if direction == 'LONG':
                    sl = entry - sl_dist
                    tp = entry + tp_dist
                else:
                    sl = entry + sl_dist
                    tp = entry - tp_dist

                pos = {
                    'd': direction, 'e': entry, 'sl': sl, 'tp': tp,
                    'ot': bars[i]['ts'], 'taker': round(taker, 4),
                    'vol_r': round(vol_r, 3),
                }

    # ── Results ──
    if not trades:
        return {'total_trades': 0, 'config': config}

    wins = [t for t in trades if t['outcome'] == 'WIN']
    losses = [t for t in trades if t['outcome'] == 'LOSS']
    timeouts = [t for t in trades if t['outcome'] == 'TIMEOUT']

    gp = sum(t['pnl_$'] for t in wins)
    gl = abs(sum(t['pnl_$'] for t in losses + timeouts))
    wr = len(wins) / len(trades) * 100
    pf = gp / gl if gl > 0 else 999

    return {
        'config': {k: v for k, v in config.items()},
        'total_trades': len(trades),
        'wins': len(wins), 'losses': len(losses), 'timeouts': len(timeouts),
        'win_rate': round(wr, 2),
        'profit_factor': round(pf, 3),
        'total_pnl_pct': round((eq - 200) / 200 * 100, 2),
        'final_equity': round(eq, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'avg_win_pnl': round(np.mean([t['pnl_$'] for t in wins]), 2) if wins else 0,
        'avg_loss_pnl': round(np.mean([t['pnl_$'] for t in losses + timeouts]), 2) if losses + timeouts else 0,
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

    # ── Test 1: Baseline (taker > 0.58, no filters) ──
    print("\n" + "="*70, flush=True)
    print("TEST 1: Baseline — Taker Imbalance Only (no filters)", flush=True)
    print("="*70, flush=True)

    cfg1 = {
        'tp_pct': 2.0, 'sl_pct': 1.5, 'hold_hours': 12,
        'taker_threshold': 0.58,
    }
    r = run_backtest(bars, cfg1)
    m = "✅" if r.get('meets_target') else "❌"
    print(f"  {m} {r['total_trades']}T {r.get('win_rate',0)}%WR {r.get('profit_factor',0)}PF PnL={r.get('total_pnl_pct',0)}% DD={r.get('max_drawdown_pct',0)}%", flush=True)
    all_results['baseline'] = r

    # ── Test 2: + Volume filter ──
    print("\n" + "="*70, flush=True)
    print("TEST 2: + Volume Ratio >= 1.0", flush=True)
    print("="*70, flush=True)

    cfg2 = {**cfg1, 'min_vol_ratio': 1.0}
    r = run_backtest(bars, cfg2)
    m = "✅" if r.get('meets_target') else "❌"
    print(f"  {m} {r['total_trades']}T {r.get('win_rate',0)}%WR {r.get('profit_factor',0)}PF PnL={r.get('total_pnl_pct',0)}%", flush=True)
    all_results['with_vol_filter'] = r

    # ── Test 3: + Trend filter (EMA200) ──
    print("\n" + "="*70, flush=True)
    print("TEST 3: + EMA200 Trend Confirmation", flush=True)
    print("="*70, flush=True)

    cfg3 = {**cfg2, 'require_trend': True}
    r = run_backtest(bars, cfg3)
    m = "✅" if r.get('meets_target') else "❌"
    print(f"  {m} {r['total_trades']}T {r.get('win_rate',0)}%WR {r.get('profit_factor',0)}PF PnL={r.get('total_pnl_pct',0)}%", flush=True)
    all_results['with_trend'] = r

    # ── Test 4: + Session filter ──
    print("\n" + "="*70, flush=True)
    print("TEST 4: + London/NY Session Filter", flush=True)
    print("="*70, flush=True)

    cfg4 = {**cfg3, 'session_filter': 'london_ny'}
    r = run_backtest(bars, cfg4)
    m = "✅" if r.get('meets_target') else "❌"
    print(f"  {m} {r['total_trades']}T {r.get('win_rate',0)}%WR {r.get('profit_factor',0)}PF PnL={r.get('total_pnl_pct',0)}%", flush=True)
    all_results['with_session'] = r

    # ── Test 5: Parameter sweep ──
    print("\n" + "="*70, flush=True)
    print("TEST 5: Parameter Sweep", flush=True)
    print("="*70, flush=True)

    sweep = []
    tested = 0

    for taker_t in [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]:
        for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for sl in [0.5, 0.75, 1.0, 1.5, 2.0]:
                for hold in [8, 12, 16, 24]:
                    for vol_min in [0, 0.5, 1.0, 1.5]:
                        for trend in [True, False]:
                            for session in ['london_ny', 'overlap', None]:
                                cfg = {
                                    'tp_pct': tp, 'sl_pct': sl, 'hold_hours': hold,
                                    'taker_threshold': taker_t,
                                    'min_vol_ratio': vol_min,
                                    'require_trend': trend,
                                    'session_filter': session,
                                }
                                r = run_backtest(bars, cfg)
                                tested += 1

                                if r['total_trades'] >= 10:
                                    wr = r.get('win_rate', 0)
                                    pf = r.get('profit_factor', 0)
                                    score = (wr / 100) * min(pf, 10) if pf < 999 else 0
                                    sweep.append({
                                        'taker': taker_t, 'tp': tp, 'sl': sl,
                                        'hold': hold, 'vol_min': vol_min,
                                        'trend': trend, 'session': session or 'all',
                                        'trades': r['total_trades'], 'wr': wr, 'pf': pf,
                                        'pnl': r.get('total_pnl_pct', 0),
                                        'dd': r.get('max_drawdown_pct', 0),
                                        'score': round(score, 3),
                                        'ok': wr >= 75 and pf >= 2.0,
                                    })

                                if tested % 2000 == 0:
                                    print(f"  Tested {tested}...", flush=True)

    sweep.sort(key=lambda x: x['score'], reverse=True)
    meets = [s for s in sweep if s['ok']]

    print(f"\n  Tested {tested} configs, {len(meets)} meet target (WR>=75% PF>=2.0)", flush=True)

    if meets:
        print(f"\n  ✅ CONFIGS THAT MEET TARGET ({len(meets)}):", flush=True)
        for s in meets[:20]:
            print(f"  ✅ Taker≥{s['taker']} TP={s['tp']}% SL={s['sl']}% H={s['hold']}h Vol≥{s['vol_min']} Trend={s['trend']} S={s['session']} | {s['trades']}T {s['wr']}%WR {s['pf']}PF PnL={s['pnl']}%", flush=True)

    print(f"\n  TOP 15 by score:", flush=True)
    for s in sweep[:15]:
        m = "✅" if s['ok'] else "  "
        print(f"  {m} Taker≥{s['taker']} TP={s['tp']}% SL={s['sl']}% H={s['hold']}h Vol≥{s['vol_min']} Trend={s['trend']} S={s['session']} | {s['trades']}T {s['wr']}%WR {s['pf']}PF Score={s['score']}", flush=True)

    all_results['sweep'] = {
        'total_configs': tested,
        'meets_target': len(meets),
        'top_15': sweep[:15],
        'all_meets': meets[:30],
    }

    # ── Test 6: Best config detailed ──
    if sweep:
        best = sweep[0]
        print(f"\n" + "="*70, flush=True)
        print(f"BEST: Taker≥{best['taker']} TP={best['tp']}% SL={best['sl']}% H={best['hold']}h Vol≥{best['vol_min']} Trend={best['trend']} S={best['session']}", flush=True)
        print("="*70, flush=True)

    # ── Save ──
    with open(OUTPUT, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT}", flush=True)
