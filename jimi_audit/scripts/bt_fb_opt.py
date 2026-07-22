#!/usr/bin/env python3
"""Failed Breakout Backtest - Optimized"""
import csv, json, os, sys
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

DATA_FILE = "/root/.openclaw/workspace/jimi_audit/eth_15m_6m.csv"
OUTPUT = "/root/.openclaw/workspace/jimi_audit/reports/failed_breakout_backtest.json"

# Load data
print("Loading data...", flush=True)
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
N = len(bars)
print(f"Loaded {N} bars", flush=True)

# Precompute ATR
atr = np.zeros(N)
for i in range(1, N):
    tr = max(bars[i]['h']-bars[i]['l'], abs(bars[i]['h']-bars[i-1]['c']), abs(bars[i]['l']-bars[i-1]['c']))
    atr[i] = tr if i < 14 else (atr[i-1]*13 + tr)/14

# Precompute BB
closes = np.array([b['c'] for b in bars])
bb_mid = np.zeros(N)
bb_up = np.zeros(N)
bb_lo = np.zeros(N)
for i in range(19, N):
    w = closes[i-19:i+1]
    m = np.mean(w)
    s = np.std(w)
    bb_mid[i] = m
    bb_up[i] = m + 2*s
    bb_lo[i] = m - 2*s

print("Indicators computed. Running backtests...", flush=True)

FEE = 0.0002
SLIP = 0.001

def backtest(tp_pct, sl_pct, hold_h, min_conv, lookback=48, fail_ret=0.3):
    trades = []
    pos = None
    eq = 200.0
    peak = 200.0
    max_dd = 0

    for i in range(50, N):
        # Check position
        if pos:
            b = bars[i]
            held = (b['ts'] - pos['ot']).total_seconds() / 3600
            exit_p = None
            outcome = None

            if pos['d'] == 'L':
                if b['h'] >= pos['tp']: exit_p, outcome = pos['tp'], 'W'
                elif b['l'] <= pos['sl']: exit_p, outcome = pos['sl'], 'L'
            else:
                if b['l'] <= pos['tp']: exit_p, outcome = pos['tp'], 'W'
                elif b['h'] >= pos['sl']: exit_p, outcome = pos['sl'], 'L'

            if exit_p is None and held >= hold_h:
                exit_p, outcome = b['c'], 'T'

            if exit_p:
                pnl = ((exit_p - pos['e'])/pos['e']) if pos['d']=='L' else ((pos['e'] - exit_p)/pos['e'])
                pnl -= FEE*2
                dollar = eq * pnl
                eq += dollar
                trades.append({'d': pos['d'], 'e': pos['e'], 'x': exit_p, 'pnl': round(pnl*100,4), '$': round(dollar,2), 'o': outcome, 'cv': pos['cv'], 'h': round(held,1)})
                if eq > peak: peak = eq
                dd = (peak-eq)/peak*100
                if dd > max_dd: max_dd = dd
                pos = None

        # Signal detection
        if pos is None and atr[i] > 0:
            for j in range(max(0, i-lookback), i):
                # Upside breakout fail
                if bars[j]['h'] > bb_up[j] and bb_up[j] > 0:
                    ret = (bb_up[j] - bars[i]['c']) / bb_up[j] * 100
                    if ret >= fail_ret:
                        wick = (bars[j]['h'] - max(bars[j]['o'], bars[j]['c'])) / max(bars[j]['h'] - bars[j]['l'], 0.01)
                        vol_r = bars[j]['v'] / max(np.mean([bars[k]['v'] for k in range(max(0,j-10),j)]), 1)
                        taker = bars[j]['tb'] / max(bars[j]['v'], 0.01)
                        cv = 0.5
                        if wick >= 0.4: cv += 0.15
                        if vol_r >= 1.0: cv += 0.1
                        if taker < 0.45: cv += 0.15
                        cv += min(ret/1.0, 1.0) * 0.3
                        cv = min(cv, 1.0)
                        if cv >= min_conv:
                            e = bars[i]['c'] * (1-SLIP)
                            sl_p = bb_up[j] + atr[i]*0.5
                            tp_p = e - e*tp_pct/100
                            sl_d = abs(e-sl_p)/e
                            if sl_d >= 0.003:
                                pos = {'d':'S','e':e,'sl':sl_p,'tp':tp_p,'cv':cv,'ot':bars[i]['ts']}
                                break

                # Downside breakout fail
                if bars[j]['l'] < bb_lo[j] and bb_lo[j] > 0:
                    ret = (bars[i]['c'] - bb_lo[j]) / bb_lo[j] * 100
                    if ret >= fail_ret:
                        wick = (min(bars[j]['o'], bars[j]['c']) - bars[j]['l']) / max(bars[j]['h'] - bars[j]['l'], 0.01)
                        vol_r = bars[j]['v'] / max(np.mean([bars[k]['v'] for k in range(max(0,j-10),j)]), 1)
                        taker = bars[j]['tb'] / max(bars[j]['v'], 0.01)
                        cv = 0.5
                        if wick >= 0.4: cv += 0.15
                        if vol_r >= 1.0: cv += 0.1
                        if taker > 0.55: cv += 0.15
                        cv += min(ret/1.0, 1.0) * 0.3
                        cv = min(cv, 1.0)
                        if cv >= min_conv:
                            e = bars[i]['c'] * (1+SLIP)
                            sl_p = bb_lo[j] - atr[i]*0.5
                            tp_p = e + e*tp_pct/100
                            sl_d = abs(e-sl_p)/e
                            if sl_d >= 0.003:
                                pos = {'d':'L','e':e,'sl':sl_p,'tp':tp_p,'cv':cv,'ot':bars[i]['ts']}
                                break

    if not trades:
        return None
    wins = sum(1 for t in trades if t['o']=='W')
    losses = sum(1 for t in trades if t['o']=='L')
    timeouts = sum(1 for t in trades if t['o']=='T')
    gp = sum(t['$'] for t in trades if t['o']=='W')
    gl = abs(sum(t['$'] for t in trades if t['o']!='W'))
    wr = wins/len(trades)*100
    pf = gp/gl if gl > 0 else 999
    return {
        'trades': len(trades), 'wins': wins, 'losses': losses, 'timeouts': timeouts,
        'wr': round(wr,2), 'pf': round(pf,3),
        'pnl': round((eq-200)/200*100,2), 'eq': round(eq,2), 'dd': round(max_dd,2),
        'avg_h': round(np.mean([t['h'] for t in trades]),1),
    }

# ── Test 1: Baseline ──
print("\n=== BASELINE ===", flush=True)
r = backtest(2.5, 1.0, 32, 0.7)
if r: print(f"  {r['trades']}T {r['wr']}%WR {r['pf']}PF PnL={r['pnl']}% DD={r['dd']}%", flush=True)
else: print("  No trades", flush=True)

# ── Test 2: Focused sweep (reduced) ──
print("\n=== PARAMETER SWEEP ===", flush=True)
results = []
configs_tested = 0
for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for sl in [0.5, 0.75, 1.0, 1.5]:
        for hold in [8, 12, 16, 24]:
            for conv in [0.5, 0.6, 0.7]:
                r = backtest(tp, sl, hold, conv)
                configs_tested += 1
                if r and r['trades'] >= 5:
                    score = (r['wr']/100) * r['pf'] if r['pf'] < 999 else 0
                    results.append({**r, 'tp':tp, 'sl':sl, 'hold':hold, 'conv':conv, 'score':round(score,3), 'ok': r['wr']>=75 and r['pf']>=2.0})
                if configs_tested % 100 == 0:
                    print(f"  Tested {configs_tested} configs...", flush=True)

results.sort(key=lambda x: x['score'], reverse=True)
meets = [r for r in results if r['ok']]
print(f"\n  Tested {configs_tested} configs, {len(meets)} meet target (WR>=75% PF>=2.0)", flush=True)
print("\n  TOP 15:", flush=True)
for r in results[:15]:
    m = "✅" if r['ok'] else "  "
    print(f"  {m} TP={r['tp']}% SL={r['sl']}% Hold={r['hold']}h Conv={r['conv']} | {r['trades']}T {r['wr']}%WR {r['pf']}PF PnL={r['pnl']}% DD={r['dd']}% Score={r['score']}", flush=True)

best = results[0] if results else None

# ── Test 3: Best + filters ──
if best:
    print(f"\n=== BEST CONFIG: TP={best['tp']}% SL={best['sl']}% Hold={best['hold']}h Conv={best['conv']} ===", flush=True)
    # Baseline with best config
    r = backtest(best['tp'], best['sl'], best['hold'], best['conv'])
    print(f"  Baseline: {r['trades']}T {r['wr']}%WR {r['pf']}PF", flush=True)

# ── Save ──
output = {
    'baseline': backtest(2.5, 1.0, 32, 0.7),
    'sweep_total': configs_tested,
    'sweep_meets_target': len(meets),
    'top_15': results[:15],
    'best': best,
}
with open(OUTPUT, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to {OUTPUT}", flush=True)
