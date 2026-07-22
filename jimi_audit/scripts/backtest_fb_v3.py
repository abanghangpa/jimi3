#!/usr/bin/env python3
"""Failed Breakout v2 — Focused Sweep (smaller, faster)"""
import csv, json, sys, os
from datetime import datetime, timezone
import numpy as np

BASE = '/root/.openclaw/workspace/jimi_audit'
DATA_FILE = f'{BASE}/eth_15m_6m.csv'
FEE = 0.0002
SLIP = 0.001

print("Loading...", flush=True)
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
closes = np.array([b['c'] for b in bars])
highs = np.array([b['h'] for b in bars])
lows = np.array([b['l'] for b in bars])
volumes = np.array([b['v'] for b in bars])

atr = np.zeros(N)
for i in range(1, N):
    tr = max(bars[i]['h']-bars[i]['l'], abs(bars[i]['h']-bars[i-1]['c']), abs(bars[i]['l']-bars[i-1]['c']))
    atr[i] = tr if i < 14 else (atr[i-1]*13 + tr)/14

avg_vol = np.zeros(N)
for i in range(20, N):
    avg_vol[i] = np.mean(volumes[i-20:i])

ema200 = np.zeros(N)
ema200[199] = np.mean(closes[:200])
m_ema = 2 / 201
for i in range(200, N):
    ema200[i] = closes[i] * m_ema + ema200[i-1] * (1 - m_ema)

print(f"Loaded {N} bars", flush=True)


def detect_fb(idx, lookback=48, return_pct=0.3, min_vol=1.0, min_wick=0.4):
    if idx < lookback + 10:
        return None, 0
    
    price = closes[idx]
    swing_high = max(highs[idx-lookback:idx])
    swing_low = min(lows[idx-lookback:idx])
    
    # Failed SHORT breakout → LONG
    for j in range(max(idx-8, 200), idx):
        if highs[j] > swing_high:
            return_dist = (swing_high - price) / swing_high * 100
            if return_dist >= return_pct:
                bb = bars[j]
                bb_range = bb['h'] - bb['l']
                wick = bb['h'] - max(bb['c'], bb['o'])
                wick_r = wick / bb_range if bb_range > 0 else 0
                vol_r = volumes[j] / max(avg_vol[j], 1)
                taker_sell = 1 - (bb['tb'] / max(bb['v'], 0.01))
                
                conv = 0.0
                if wick_r >= min_wick: conv += 0.3
                if vol_r >= min_vol: conv += 0.2
                if taker_sell >= 0.52: conv += 0.2
                if price < swing_high: conv += 0.2
                if closes[idx] < bars[idx]['o']: conv += 0.1
                
                if conv >= 0.5:
                    return 'LONG', conv
            break
    
    # Failed LONG breakout → SHORT
    for j in range(max(idx-8, 200), idx):
        if lows[j] < swing_low:
            return_dist = (price - swing_low) / swing_low * 100
            if return_dist >= return_pct:
                bb = bars[j]
                bb_range = bb['h'] - bb['l']
                wick = min(bb['c'], bb['o']) - bb['l']
                wick_r = wick / bb_range if bb_range > 0 else 0
                vol_r = volumes[j] / max(avg_vol[j], 1)
                taker_buy = bb['tb'] / max(bb['v'], 0.01)
                
                conv = 0.0
                if wick_r >= min_wick: conv += 0.3
                if vol_r >= min_vol: conv += 0.2
                if taker_buy >= 0.52: conv += 0.2
                if price > swing_low: conv += 0.2
                if closes[idx] > bars[idx]['o']: conv += 0.1
                
                if conv >= 0.5:
                    return 'SHORT', conv
            break
    
    return None, 0


def run_bt(tp, sl, hold, conv, lb, ret, vol, wick, trend, session):
    trades = []
    capital = 200.0
    peak = capital
    max_dd = 0
    last_bar = -999
    
    for i in range(210, N):
        if session == 'ol' and not (13 <= bars[i]['ts'].hour < 16): continue
        if session == 'ln' and not (8 <= bars[i]['ts'].hour < 21): continue
        
        d, c = detect_fb(i, lb, ret, vol, wick)
        if not d or c < conv: continue
        if i - last_bar < 12: continue
        
        if trend:
            if d == 'LONG' and closes[i] < ema200[i]: continue
            if d == 'SHORT' and closes[i] > ema200[i]: continue
        
        entry = closes[i] * (1 + SLIP if d == 'LONG' else 1 - SLIP)
        sl_p = entry * (1 - sl/100) if d == 'LONG' else entry * (1 + sl/100)
        tp_p = entry * (1 + tp/100) if d == 'LONG' else entry * (1 - tp/100)
        last_bar = i
        
        outcome = None
        for j in range(i+1, min(i+hold*4+1, N)):
            if d == 'LONG':
                if highs[j] >= tp_p: outcome = 'W'; exit_p = tp_p; break
                if lows[j] <= sl_p: outcome = 'L'; exit_p = sl_p; break
            else:
                if lows[j] <= tp_p: outcome = 'W'; exit_p = tp_p; break
                if highs[j] >= sl_p: outcome = 'L'; exit_p = sl_p; break
        if not outcome:
            exit_p = closes[min(i+hold*4, N-1)]
            outcome = 'T'
        
        pnl = ((exit_p-entry)/entry*100) if d=='LONG' else ((entry-exit_p)/entry*100)
        pnl -= FEE*2
        size = capital * 0.10 * 25
        capital += size * pnl / 100
        if capital > peak: peak = capital
        dd = (peak-capital)/peak*100 if peak>0 else 0
        if dd > max_dd: max_dd = dd
        trades.append({'o': outcome, 'pnl': pnl, 'ts': bars[i]['ts'].isoformat()[:7]})
    
    if len(trades) < 8: return None
    w = sum(1 for t in trades if t['o']=='W')
    l = sum(1 for t in trades if t['o']=='L')
    tt = sum(1 for t in trades if t['o']=='T')
    wr = w/len(trades)*100
    avg_w = np.mean([t['pnl'] for t in trades if t['o']=='W']) if w else 0
    avg_l = np.mean([abs(t['pnl']) for t in trades if t['o']=='L']) if l else 0
    pf = w*avg_w / max(l*avg_l, 0.01) if l else 999
    
    monthly = {}
    for t in trades:
        m = t['ts']
        if m not in monthly: monthly[m] = 0
        monthly[m] += t['pnl']
    bad_m = sum(1 for v in monthly.values() if v < 0)
    
    return {'trades': len(trades), 'wins': w, 'losses': l, 'timeouts': tt,
            'wr': round(wr,1), 'pf': round(pf,2), 'pnl': round((capital-200)/200*100,2),
            'dd': round(max_dd,1), 'avg_w': round(avg_w,2), 'avg_l': round(avg_l,2),
            'bad_m': bad_m, 'months': len(monthly), 'monthly': monthly}


print("\nFocused sweep...", flush=True)
results = []
tested = 0

for tp in [1.0, 1.5, 2.0, 2.5]:
    for sl in [0.3, 0.5, 0.75]:
        if sl >= tp: continue
        for hold in [4, 8, 12, 16]:
            for conv in [0.5, 0.6, 0.7, 0.8]:
                for lb in [24, 48]:
                    for ret in [0.2, 0.3, 0.5]:
                        for vol in [0.8, 1.0]:
                            for wick in [0.3, 0.4]:
                                for trend in [None, True]:
                                    for sess in [None, 'ol']:
                                        r = run_bt(tp, sl, hold, conv, lb, ret, vol, wick, trend, sess)
                                        tested += 1
                                        if r:
                                            results.append({**r, 'tp': tp, 'sl': sl, 'hold': hold,
                                                           'conv': conv, 'lb': lb, 'ret': ret,
                                                           'vol': vol, 'wick': wick,
                                                           'trend': trend, 'sess': sess or 'all'})

print(f"Tested {tested}", flush=True)

# Sort by PF * WR
results.sort(key=lambda x: x['pf'] * (x['wr']/100), reverse=True)

ent = [r for r in results if r['wr']>=65 and r['pf']>=2.0 and r['dd']<25 and r['bad_m']<=3 and r['trades']>=15]
good = [r for r in results if r['wr']>=55 and r['pf']>=1.5 and r['trades']>=15]
ok = [r for r in results if r['wr']>=50 and r['pf']>=1.2 and r['trades']>=10]

print(f"\nEnterprise (WR>=65, PF>=2.0, DD<25, bad_m<=3): {len(ent)}")
print(f"Good (WR>=55, PF>=1.5, T>=15): {len(good)}")
print(f"OK (WR>=50, PF>=1.2, T>=10): {len(ok)}")

for label, subset in [("ENTERPRISE", ent), ("GOOD", good)]:
    if not subset: continue
    print(f"\n{'='*110}")
    print(f"{label} RESULTS ({len(subset)} configs)")
    print(f"{'='*110}")
    print(f"  {'#':>2s} {'TP':>4s} {'SL':>4s} {'RR':>4s} {'H':>3s} {'Cv':>3s} {'LB':>3s} {'Rt':>4s} {'Vl':>3s} {'Wk':>3s} {'Tr':>2s} {'Se':>2s} | {'N':>4s} {'W':>3s} {'L':>3s} {'T':>3s} {'WR':>5s} {'PF':>5s} {'PnL':>7s} {'DD':>5s} {'AvW':>5s} {'AvL':>5s} {'BM':>3s}")
    print("  " + "-" * 105)
    for i, r in enumerate(subset[:25]):
        rr = round(r['tp']/r['sl'], 1)
        tr = 'T' if r['trend'] else '-'
        print(f"  {i+1:>2d} {r['tp']:>4.1f} {r['sl']:>4.2f} {rr:>4.1f} {r['hold']:>3d} {r['conv']:>3.1f} {r['lb']:>3d} {r['ret']:>4.1f} {r['vol']:>3.1f} {r['wick']:>3.1f} {tr:>2s} {r['sess']:>2s} | {r['trades']:>4d} {r['wins']:>3d} {r['losses']:>3d} {r['timeouts']:>3d} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>+7.1f} {r['dd']:>5.1f} {r['avg_w']:>5.2f} {r['avg_l']:>5.2f} {r['bad_m']:>3d}")

# Best detail
if ent:
    b = ent[0]
    print(f"\n{'='*80}")
    print("BEST CONFIG")
    print(f"{'='*80}")
    print(f"  TP={b['tp']}% SL={b['sl']}% Hold={b['hold']}h Conv>={b['conv']}")
    print(f"  Lookback={b['lb']} Return>={b['ret']}% Vol>={b['vol']}x Wick>={b['wick']}")
    print(f"  Trend={'EMA200' if b['trend'] else 'None'} Session={b['sess']}")
    print(f"  R:R=1:{b['tp']/b['sl']:.1f}")
    print(f"  {b['trades']}T {b['wr']}%WR {b['pf']}PF PnL={b['pnl']}% DD={b['dd']}%")
    print(f"\n  Monthly:")
    for m, v in sorted(b['monthly'].items()):
        print(f"    {m}: PnL={v:+.1f}%")

print("\nDone")
