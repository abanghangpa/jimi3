#!/usr/bin/env python3
"""OBI Focused Sweep - High conviction only"""
import csv, json, sys
from datetime import datetime, timezone
import numpy as np

DATA_FILE = "/root/.openclaw/workspace/jimi_audit/eth_15m_6m.csv"
OUTPUT = "/root/.openclaw/workspace/jimi_audit/reports/obi_backtest.json"
FEE = 0.0002
SLIP = 0.001

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

print("Loading...", flush=True)
bars = load_data()
N = len(bars)
closes = np.array([b['c'] for b in bars])
atr = compute_atr(bars)
avg_vol = compute_avg_vol(bars)
ema200 = compute_ema(closes, 200)
print(f"Loaded {N} bars", flush=True)

def backtest(tp, sl, hold, taker_t, vol_min, trend, session):
    trades = []
    pos = None
    eq = 200.0
    peak = 200.0
    max_dd = 0

    for i in range(210, N):
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
            if exit_p is None and held >= hold:
                exit_p, outcome = b['c'], 'T'
            if exit_p:
                pnl = ((exit_p-pos['e'])/pos['e']) if pos['d']=='L' else ((pos['e']-exit_p)/pos['e'])
                pnl -= FEE*2
                eq += eq * pnl
                trades.append({'o': outcome})
                if eq > peak: peak = eq
                dd = (peak-eq)/peak*100
                if dd > max_dd: max_dd = dd
                pos = None

        if pos is None:
            ts = bars[i]['ts']
            if session:
                h = ts.hour
                if session == 'ln' and not (8 <= h < 21): continue
                elif session == 'ol' and not (13 <= h < 16): continue

            taker = bars[i]['tb'] / max(bars[i]['v'], 0.01)
            vol_r = bars[i]['v'] / max(avg_vol[i], 1) if avg_vol[i] > 0 else 0

            if vol_r < vol_min: continue

            d = None
            if taker >= taker_t:
                d = 'L'
                if trend and closes[i] < ema200[i]: continue
            elif taker <= (1 - taker_t):
                d = 'S'
                if trend and closes[i] > ema200[i]: continue

            if d:
                e = closes[i] * (1 + SLIP if d == 'L' else 1 - SLIP)
                sl_d = e * sl / 100
                tp_d = e * tp / 100
                if d == 'L':
                    sl_p, tp_p = e - sl_d, e + tp_d
                else:
                    sl_p, tp_p = e + sl_d, e - tp_d
                pos = {'d': d, 'e': e, 'sl': sl_p, 'tp': tp_p, 'ot': ts}

    if len(trades) < 5: return None
    w = sum(1 for t in trades if t['o']=='W')
    l = sum(1 for t in trades if t['o']=='L')
    wr = w/len(trades)*100
    return {'trades': len(trades), 'wins': w, 'losses': l, 'wr': round(wr,2), 'eq': round(eq,2), 'dd': round(max_dd,2), 'pnl': round((eq-200)/200*100,2)}

# Focused sweep: high taker thresholds only
print("\n=== FOCUSED SWEEP ===", flush=True)
results = []
tested = 0

for taker_t in [0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]:
    for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for sl in [0.5, 0.75, 1.0, 1.5]:
            for hold in [8, 12, 16, 24]:
                for vol_min in [0, 0.5, 1.0, 1.5, 2.0]:
                    for trend in [True, False]:
                        for session in ['ln', 'ol', None]:
                            r = backtest(tp, sl, hold, taker_t, vol_min, trend, session)
                            tested += 1
                            if r:
                                score = (r['wr']/100) * min(r.get('pf', r['wr']/max(100-r['wr'],1)*tp/sl), 10)
                                pf = r['wins']*tp / max(r['losses']*sl, 0.01) if r['losses'] > 0 else 999
                                score = (r['wr']/100) * min(pf, 10)
                                results.append({**r, 'taker': taker_t, 'tp': tp, 'sl': sl, 'hold': hold, 'vol': vol_min, 'trend': trend, 'session': session or 'all', 'score': round(score,3), 'pf': round(pf,3), 'ok': r['wr'] >= 75 and pf >= 2.0})
                            if tested % 1000 == 0:
                                print(f"  {tested}...", flush=True)

results.sort(key=lambda x: x['score'], reverse=True)
meets = [r for r in results if r['ok']]

print(f"\nTested {tested} configs, {len(meets)} meet target", flush=True)

if meets:
    print(f"\n✅ MEETS TARGET ({len(meets)}):", flush=True)
    for r in meets[:25]:
        print(f"  ✅ T>={r['taker']} TP={r['tp']}% SL={r['sl']}% H={r['hold']}h Vol>={r['vol']} Tr={r['trend']} S={r['session']} | {r['trades']}T {r['wr']}%WR {r['pf']}PF PnL={r['pnl']}% DD={r['dd']}%", flush=True)

print(f"\nTOP 20 by score:", flush=True)
for r in results[:20]:
    m = "✅" if r['ok'] else "  "
    print(f"  {m} T>={r['taker']} TP={r['tp']}% SL={r['sl']}% H={r['hold']}h Vol>={r['vol']} Tr={r['trend']} S={r['session']} | {r['trades']}T {r['wr']}%WR {r['pf']}PF Score={r['score']}", flush=True)

# Best config
if results:
    b = results[0]
    print(f"\n{'='*70}", flush=True)
    print(f"BEST: Taker>={b['taker']} TP={b['tp']}% SL={b['sl']}% H={b['hold']}h Vol>={b['vol']} Trend={b['trend']} Session={b['session']}", flush=True)
    print(f"  {b['trades']}T {b['wr']}%WR {b['pf']}PF PnL={b['pnl']}% DD={b['dd']}%", flush=True)

# Save
with open(OUTPUT, 'w') as f:
    json.dump({'tested': tested, 'meets': len(meets), 'top_20': results[:20], 'all_meets': meets[:30]}, f, indent=2)
print(f"\nSaved to {OUTPUT}", flush=True)
