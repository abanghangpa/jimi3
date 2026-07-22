#!/usr/bin/env python3
"""Failed Breakout v4 — Fixed breakout detection"""
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

# ============================================================
# PRE-COMPUTE signals with FIXED breakout detection
# ============================================================
print("Pre-computing signals...", flush=True)

LB = 48
signals = []

for i in range(LB + 20, N):
    price = closes[i]
    atr_val = atr[i]
    if atr_val <= 0: continue
    
    # Swing high/low from BEFORE the breakout window
    # Lookback: bars [i-LB-8, i-8) for the swing levels
    # Breakout attempt: bars [i-8, i)
    if i < LB + 8: continue
    
    # Use a PRIOR period for swing levels (not including recent 8 bars)
    prior_high = np.max(highs[i-LB-8:i-8])
    prior_low = np.min(lows[i-LB-8:i-8])
    
    # Check for breakout above prior_high in recent 8 bars, then failure
    broke_above = False
    breakout_bar = -1
    for j in range(i-8, i):
        if highs[j] > prior_high:
            broke_above = True
            breakout_bar = j
            break
    
    if broke_above and breakout_bar >= 0:
        # Check if price returned below prior_high (failure)
        return_dist = (prior_high - price) / prior_high * 100
        if return_dist >= 0.2:  # Lowered threshold
            bb = bars[breakout_bar]
            bb_range = bb['h'] - bb['l']
            wick = bb['h'] - max(bb['c'], bb['o'])
            wick_r = wick / bb_range if bb_range > 0 else 0
            vol_r = volumes[breakout_bar] / max(avg_vol[breakout_bar], 1)
            taker_sell = 1 - (bb['tb'] / max(bb['v'], 0.01))
            
            conv = 0.0
            if wick_r >= 0.3: conv += 0.3
            if vol_r >= 0.8: conv += 0.2
            if taker_sell >= 0.50: conv += 0.2
            if price < prior_high: conv += 0.2
            if closes[i] < bars[i]['o']: conv += 0.1
            
            if conv >= 0.4:
                signals.append((i, 'LONG', conv, price, atr_val))
    
    # Check for breakout below prior_low in recent 8 bars, then failure
    broke_below = False
    breakout_bar = -1
    for j in range(i-8, i):
        if lows[j] < prior_low:
            broke_below = True
            breakout_bar = j
            break
    
    if broke_below and breakout_bar >= 0:
        return_dist = (price - prior_low) / prior_low * 100
        if return_dist >= 0.2:
            bb = bars[breakout_bar]
            bb_range = bb['h'] - bb['l']
            wick = min(bb['c'], bb['o']) - bb['l']
            wick_r = wick / bb_range if bb_range > 0 else 0
            vol_r = volumes[breakout_bar] / max(avg_vol[breakout_bar], 1)
            taker_buy = bb['tb'] / max(bb['v'], 0.01)
            
            conv = 0.0
            if wick_r >= 0.3: conv += 0.3
            if vol_r >= 0.8: conv += 0.2
            if taker_buy >= 0.50: conv += 0.2
            if price > prior_low: conv += 0.2
            if closes[i] > bars[i]['o']: conv += 0.1
            
            if conv >= 0.4:
                signals.append((i, 'SHORT', conv, price, atr_val))

print(f"Found {len(signals)} raw signals", flush=True)
if signals:
    longs = sum(1 for s in signals if s[1] == 'LONG')
    shorts = sum(1 for s in signals if s[1] == 'SHORT')
    print(f"  LONG: {longs}, SHORT: {shorts}", flush=True)
    convs = [s[2] for s in signals]
    print(f"  Conviction: min={min(convs):.2f}, max={max(convs):.2f}, avg={np.mean(convs):.2f}", flush=True)

# ============================================================
# FAST SWEEP
# ============================================================
def sim(signals_subset, tp_pct, sl_pct, hold_hours, min_conv, trend_filter, session_filter, dedup=12):
    trades = []
    capital = 200.0
    peak = capital
    max_dd = 0
    last_bar = -999
    
    for idx, direction, conv, entry_price, atr_val in signals_subset:
        if conv < min_conv: continue
        if idx - last_bar < dedup: continue
        
        h = bars[idx]['ts'].hour
        if session_filter == 'ol' and not (13 <= h < 16): continue
        if session_filter == 'ln' and not (8 <= h < 21): continue
        
        if trend_filter:
            if direction == 'LONG' and closes[idx] < ema200[idx]: continue
            if direction == 'SHORT' and closes[idx] > ema200[idx]: continue
        
        entry = entry_price * (1 + SLIP if direction == 'LONG' else 1 - SLIP)
        sl_p = entry * (1 - sl_pct/100) if direction == 'LONG' else entry * (1 + sl_pct/100)
        tp_p = entry * (1 + tp_pct/100) if direction == 'LONG' else entry * (1 - tp_pct/100)
        last_bar = idx
        
        outcome = None
        exit_p = None
        for j in range(idx+1, min(idx+hold_hours*4+1, N)):
            if direction == 'LONG':
                if highs[j] >= tp_p: outcome = 'W'; exit_p = tp_p; break
                if lows[j] <= sl_p: outcome = 'L'; exit_p = sl_p; break
            else:
                if lows[j] <= tp_p: outcome = 'W'; exit_p = tp_p; break
                if highs[j] >= sl_p: outcome = 'L'; exit_p = sl_p; break
        if not outcome:
            exit_p = closes[min(idx+hold_hours*4, N-1)]
            outcome = 'T'
        
        pnl = ((exit_p-entry)/entry*100) if direction=='LONG' else ((entry-exit_p)/entry*100)
        pnl -= FEE*2
        size = capital * 0.10 * 25
        capital += size * pnl / 100
        if capital > peak: peak = capital
        dd = (peak-capital)/peak*100 if peak>0 else 0
        if dd > max_dd: max_dd = dd
        trades.append({'o': outcome, 'pnl': pnl, 'ts': bars[idx]['ts'].isoformat()[:7]})
    
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


print("\nSweeping...", flush=True)
results = []
tested = 0

for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for sl in [0.3, 0.5, 0.75]:
        if sl >= tp: continue
        for hold in [4, 8, 12, 16]:
            for conv in [0.4, 0.5, 0.6, 0.7, 0.8]:
                for trend in [None, True]:
                    for sess in [None, 'ol', 'ln']:
                        r = sim(signals, tp, sl, hold, conv, trend, sess)
                        tested += 1
                        if r:
                            results.append({**r, 'tp': tp, 'sl': sl, 'hold': hold,
                                           'conv': conv, 'trend': trend, 'sess': sess or 'all'})

print(f"Tested {tested} configs", flush=True)

results.sort(key=lambda x: x['pf'] * (x['wr']/100), reverse=True)

ent = [r for r in results if r['wr']>=65 and r['pf']>=2.0 and r['dd']<25 and r['bad_m']<=3 and r['trades']>=15]
good = [r for r in results if r['wr']>=55 and r['pf']>=1.5 and r['trades']>=15]

print(f"\nEnterprise (WR>=65, PF>=2.0, DD<25, bad_m<=3): {len(ent)}")
print(f"Good (WR>=55, PF>=1.5, T>=15): {len(good)}")

for label, subset in [("ENTERPRISE", ent), ("GOOD", good)]:
    if not subset: continue
    print(f"\n{'='*100}")
    print(f"{label} RESULTS ({len(subset)} configs)")
    print(f"{'='*100}")
    print(f"  {'#':>2s} {'TP':>4s} {'SL':>4s} {'RR':>4s} {'H':>3s} {'Cv':>3s} {'Tr':>2s} {'Se':>2s} | {'N':>4s} {'W':>3s} {'L':>3s} {'T':>3s} {'WR':>5s} {'PF':>5s} {'PnL':>7s} {'DD':>5s} {'AvW':>5s} {'AvL':>5s} {'BM':>3s}")
    print("  " + "-" * 95)
    for i, r in enumerate(subset[:30]):
        rr = round(r['tp']/r['sl'], 1)
        tr = 'T' if r['trend'] else '-'
        print(f"  {i+1:>2d} {r['tp']:>4.1f} {r['sl']:>4.2f} {rr:>4.1f} {r['hold']:>3d} {r['conv']:>3.1f} {tr:>2s} {r['sess']:>2s} | {r['trades']:>4d} {r['wins']:>3d} {r['losses']:>3d} {r['timeouts']:>3d} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>+7.1f} {r['dd']:>5.1f} {r['avg_w']:>5.2f} {r['avg_l']:>5.2f} {r['bad_m']:>3d}")

best = ent if ent else good
if best:
    b = best[0]
    print(f"\n{'='*80}")
    print("BEST CONFIG")
    print(f"{'='*80}")
    print(f"  TP={b['tp']}% SL={b['sl']}% Hold={b['hold']}h Conv>={b['conv']}")
    print(f"  Trend={'EMA200' if b['trend'] else 'None'} Session={b['sess']}")
    print(f"  R:R=1:{b['tp']/b['sl']:.1f}")
    print(f"  {b['trades']}T {b['wr']}%WR {b['pf']}PF PnL={b['pnl']}% DD={b['dd']}%")
    print(f"\n  Monthly:")
    for m, v in sorted(b['monthly'].items()):
        print(f"    {m}: PnL={v:+.1f}%")

print("\nDone")
