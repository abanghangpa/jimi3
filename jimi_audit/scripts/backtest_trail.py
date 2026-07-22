#!/usr/bin/env python3
"""
Backtest positioning_fade + trade_flow with trailing stop
Same approach that rescued OBI: tight SL, trail to breakeven
"""
import csv, json, sys, os
from datetime import datetime, timezone
import numpy as np

BASE = '/root/.openclaw/workspace/jimi_audit'
FEE = 0.0002
SLIP = 0.001

print("Loading...", flush=True)
bars = []
with open(f'{BASE}/eth_15m_6m.csv') as f:
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

ema200 = np.zeros(N)
ema200[199] = np.mean(closes[:200])
m_ema = 2 / 201
for i in range(200, N):
    ema200[i] = closes[i] * m_ema + ema200[i-1] * (1 - m_ema)

ts_map = {}
for i, b in enumerate(bars):
    ts_map[b['ts'].strftime('%Y-%m-%d %H:%M:%S')] = i

print(f"Loaded {N} bars", flush=True)

event_signals = {}
with open(f'{BASE}/data/strategy_signals.jsonl') as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('fired'):
                s = d.get('strategy')
                if s not in event_signals:
                    event_signals[s] = []
                event_signals[s].append(d)
        except:
            pass


def sim_trail(strat_name, tp_pct, sl_pct, hold_hours, min_conv, trend, sess,
              trail_trigger, trail_offset, dedup=4):
    """Simulate with trailing stop: move SL to breakeven+offset after trail_trigger% profit"""
    sigs = event_signals.get(strat_name, [])
    if not sigs: return None
    
    trades = []
    capital = 200.0
    peak = capital
    max_dd = 0
    last_bar = -999
    
    for sig in sigs:
        ts = sig.get('timestamp', '')
        d = sig.get('direction')
        p = sig.get('entry') or sig.get('price', 0)
        conv = sig.get('conviction', 0)
        if not d or not p: continue
        if conv < min_conv: continue
        
        idx = ts_map.get(ts, -1)
        if idx < 0 or idx >= N - hold_hours * 4: continue
        if idx - last_bar < dedup: continue
        
        h = bars[idx]['ts'].hour
        if sess == 'ol' and not (13 <= h < 16): continue
        if sess == 'ln' and not (8 <= h < 21): continue
        if trend:
            if d == 'LONG' and closes[idx] < ema200[idx]: continue
            if d == 'SHORT' and closes[idx] > ema200[idx]: continue
        
        entry = p * (1 + SLIP if d == 'LONG' else 1 - SLIP)
        sl_p = entry * (1 - sl_pct/100) if d == 'LONG' else entry * (1 + sl_pct/100)
        tp_p = entry * (1 + tp_pct/100) if d == 'LONG' else entry * (1 - tp_pct/100)
        last_bar = idx
        trailed = False
        
        outcome = None
        exit_p = None
        for j in range(idx+1, min(idx+hold_hours*4+1, N)):
            # Trailing stop logic
            if not trailed and trail_trigger > 0:
                if d == 'LONG':
                    profit = (highs[j] - entry) / entry * 100
                    if profit >= trail_trigger:
                        sl_p = entry * (1 + trail_offset/100)
                        trailed = True
                else:
                    profit = (entry - lows[j]) / entry * 100
                    if profit >= trail_trigger:
                        sl_p = entry * (1 - trail_offset/100)
                        trailed = True
            
            # Check exit
            if d == 'LONG':
                if highs[j] >= tp_p: outcome = 'W'; exit_p = tp_p; break
                if lows[j] <= sl_p:
                    exit_p = sl_p
                    outcome = 'W' if trailed and sl_p >= entry else 'L'
                    break
            else:
                if lows[j] <= tp_p: outcome = 'W'; exit_p = tp_p; break
                if highs[j] >= sl_p:
                    exit_p = sl_p
                    outcome = 'W' if trailed and sl_p <= entry else 'L'
                    break
        
        if not outcome:
            exit_p = closes[min(idx+hold_hours*4, N-1)]
            outcome = 'T'
        
        pnl = ((exit_p-entry)/entry*100) if d=='LONG' else ((entry-exit_p)/entry*100)
        pnl -= FEE*2
        size = capital * 0.10 * 25
        capital += size * pnl / 100
        if capital > peak: peak = capital
        dd = (peak-capital)/peak*100 if peak>0 else 0
        if dd > max_dd: max_dd = dd
        trades.append({'o': outcome, 'pnl': pnl, 'ts': bars[idx]['ts'].isoformat()[:7]})
    
    if len(trades) < 5: return None
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


for strat_name in ['positioning_fade', 'trade_flow']:
    sigs = event_signals.get(strat_name, [])
    if len(sigs) < 10:
        print(f"\n{strat_name}: {len(sigs)} signals, skipping")
        continue
    
    print(f"\n{'='*110}")
    print(f"  {strat_name} ({len(sigs)} signals) — TRAILING STOP")
    print(f"{'='*110}")
    
    results = []
    tested = 0
    
    for tp in [1.0, 1.5, 2.0, 2.5]:
        for sl in [0.3, 0.5, 0.75]:
            if sl >= tp: continue
            for hold in [8, 12, 16]:
                for conv in [0.3, 0.4, 0.5]:
                    for trail_trigger in [0.2, 0.3, 0.5]:
                        for trail_offset in [0.0, 0.1]:
                            for trend in [None, True]:
                                for sess in [None, 'ol']:
                                    r = sim_trail(strat_name, tp, sl, hold, conv, trend, sess,
                                                 trail_trigger, trail_offset)
                                    tested += 1
                                    if r:
                                        results.append({**r, 'tp': tp, 'sl': sl, 'hold': hold,
                                                       'conv': conv, 'trend': trend, 'sess': sess or 'all',
                                                       'trail_trigger': trail_trigger,
                                                       'trail_offset': trail_offset})
    
    print(f"  Tested {tested}", flush=True)
    
    results.sort(key=lambda x: x['pf'] * (x['wr']/100), reverse=True)
    
    ent = [r for r in results if r['wr']>=65 and r['pf']>=2.0 and r['dd']<25 and r['bad_m']<=3 and r['trades']>=10]
    good = [r for r in results if r['wr']>=55 and r['pf']>=1.5 and r['trades']>=10]
    ok = [r for r in results if r['wr']>=50 and r['pf']>=1.2 and r['trades']>=8]
    
    print(f"  Enterprise: {len(ent)}, Good: {len(good)}, OK: {len(ok)}", flush=True)
    
    for label, subset in [("ENTERPRISE", ent), ("GOOD", good), ("OK", ok)]:
        if not subset: continue
        print(f"\n  {label} ({len(subset)}):", flush=True)
        print(f"  {'#':>2s} {'TP':>4s} {'SL':>4s} {'RR':>4s} {'H':>3s} {'Cv':>3s} {'Trig':>5s} {'Off':>4s} {'Tr':>2s} {'Se':>2s} | {'N':>4s} {'W':>3s} {'L':>3s} {'T':>3s} {'WR':>5s} {'PF':>5s} {'PnL':>7s} {'DD':>5s} {'BM':>3s}", flush=True)
        print("  " + "-" * 100, flush=True)
        for i, r in enumerate(subset[:20]):
            rr = round(r['tp']/r['sl'], 1)
            tr = 'T' if r['trend'] else '-'
            print(f"  {i+1:>2d} {r['tp']:>4.1f} {r['sl']:>4.2f} {rr:>4.1f} {r['hold']:>3d} {r['conv']:>3.1f} {r['trail_trigger']:>5.2f} {r['trail_offset']:>4.1f} {tr:>2s} {r['sess']:>2s} | {r['trades']:>4d} {r['wins']:>3d} {r['losses']:>3d} {r['timeouts']:>3d} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>+7.1f} {r['dd']:>5.1f} {r['bad_m']:>3d}", flush=True)
    
    best = ent if ent else (good if good else ok)
    if best:
        b = best[0]
        print(f"\n  BEST: TP={b['tp']}% SL={b['sl']}% Hold={b['hold']}h Conv>={b['conv']} Trail@{b['trail_trigger']}% Offset={b['trail_offset']}% Trend={'T' if b['trend'] else '-'} Sess={b['sess']}", flush=True)
        print(f"    {b['trades']}T {b['wr']}%WR {b['pf']}PF PnL={b['pnl']}% DD={b['dd']}%", flush=True)
        print(f"    Monthly:", flush=True)
        for m, v in sorted(b['monthly'].items()):
            print(f"      {m}: PnL={v:+.1f}%", flush=True)
    else:
        print(f"  No viable configs", flush=True)

print("\nDone")
