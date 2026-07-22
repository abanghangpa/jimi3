#!/usr/bin/env python3
"""
Failed Breakout MTF — 4H detection + 15m entry
================================================
- 4H: Detect failed breakout pattern (significant levels, trapped participants)
- 15m: Entry timing on the reversal confirmation
"""
import csv, json, sys, os
from datetime import datetime, timezone
import numpy as np

BASE = '/root/.openclaw/workspace/jimi_audit'
DATA_FILE = f'{BASE}/eth_15m_6m.csv'
FEE = 0.0002
SLIP = 0.001

print("Loading 15m data...", flush=True)

bars_15m = []
with open(DATA_FILE) as f:
    for row in csv.DictReader(f):
        bars_15m.append({
            'ts': datetime.strptime(row['Open time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc),
            'o': float(row['Open']), 'h': float(row['High']),
            'l': float(row['Low']), 'c': float(row['Close']),
            'v': float(row['Volume']),
            'tb': float(row.get('Taker buy base asset volume', 0)),
        })

N15 = len(bars_15m)
print(f"Loaded {N15} bars (15m)", flush=True)

# ============================================================
# Aggregate 15m → 4H
# ============================================================
print("Aggregating to 4H...", flush=True)

bars_4h = []
i = 0
while i < N15:
    ts = bars_15m[i]['ts']
    # 4H boundary: 0, 4, 8, 12, 16, 20 UTC
    hour = ts.hour
    bucket = (hour // 4) * 4
    bucket_start = ts.replace(hour=bucket, minute=0, second=0, microsecond=0)
    
    o = bars_15m[i]['o']
    h = bars_15m[i]['h']
    l = bars_15m[i]['l']
    c = bars_15m[i]['c']
    v = bars_15m[i]['v']
    tb = bars_15m[i]['tb']
    count = 1
    
    i += 1
    while i < N15:
        next_ts = bars_15m[i]['ts']
        next_hour = next_ts.hour
        next_bucket = (next_hour // 4) * 4
        if next_bucket != bucket or next_ts.date() != ts.date():
            # Also handle day boundary
            if next_ts.hour == 0 and bucket == 20:
                break
            if next_bucket != bucket:
                break
        h = max(h, bars_15m[i]['h'])
        l = min(l, bars_15m[i]['l'])
        c = bars_15m[i]['c']
        v += bars_15m[i]['v']
        tb += bars_15m[i]['tb']
        count += 1
        i += 1
    
    bars_4h.append({
        'ts': bucket_start,
        'o': o, 'h': h, 'l': l, 'c': c,
        'v': v, 'tb': tb, 'count': count,
        'idx_start': i - count,  # index in 15m array
        'idx_end': i,
    })

N4 = len(bars_4h)
print(f"Aggregated to {N4} bars (4H)", flush=True)

# 4H technicals
closes_4h = np.array([b['c'] for b in bars_4h])
highs_4h = np.array([b['h'] for b in bars_4h])
lows_4h = np.array([b['l'] for b in bars_4h])
volumes_4h = np.array([b['v'] for b in bars_4h])

# ATR on 4H
atr_4h = np.zeros(N4)
for i in range(1, N4):
    tr = max(highs_4h[i]-lows_4h[i], abs(highs_4h[i]-closes_4h[i-1]), abs(lows_4h[i]-closes_4h[i-1]))
    atr_4h[i] = tr if i < 14 else (atr_4h[i-1]*13 + tr)/14

# Avg volume on 4H
avg_vol_4h = np.zeros(N4)
for i in range(20, N4):
    avg_vol_4h[i] = np.mean(volumes_4h[i-20:i])

# EMA200 on 4H
ema200_4h = np.zeros(N4)
if N4 > 200:
    ema200_4h[199] = np.mean(closes_4h[:200])
    m_ema = 2 / 201
    for i in range(200, N4):
        ema200_4h[i] = closes_4h[i] * m_ema + ema200_4h[i-1] * (1 - m_ema)

# 15m technicals
closes_15m = np.array([b['c'] for b in bars_15m])
highs_15m = np.array([b['h'] for b in bars_15m])
lows_15m = np.array([b['l'] for b in bars_15m])
volumes_15m = np.array([b['v'] for b in bars_15m])

atr_15m = np.zeros(N15)
for i in range(1, N15):
    tr = max(bars_15m[i]['h']-bars_15m[i]['l'], abs(bars_15m[i]['h']-bars_15m[i-1]['c']), abs(bars_15m[i]['l']-bars_15m[i-1]['c']))
    atr_15m[i] = tr if i < 14 else (atr_15m[i-1]*13 + tr)/14

avg_vol_15m = np.zeros(N15)
for i in range(20, N15):
    avg_vol_15m[i] = np.mean(volumes_15m[i-20:i])

# ============================================================
# Detect failed breakout on 4H
# ============================================================
print("Detecting 4H failed breakouts...", flush=True)

fb_signals_4h = []  # (4h_idx, direction, conviction, breakout_level, breakout_bar_4h)

for i in range(60, N4):
    price = closes_4h[i]
    atr_val = atr_4h[i]
    if atr_val <= 0: continue
    
    # Swing high/low from prior period (excluding last 2 bars)
    prior_high = np.max(highs_4h[i-48:i-2])
    prior_low = np.min(lows_4h[i-48:i-2])
    
    # Check for failed SHORT breakout → LONG signal
    # Look in last 2-4 bars for breakout above prior_high
    for j in range(max(i-4, 48), i):
        if highs_4h[j] > prior_high:
            # Breakout found, check if it failed (price returned below)
            return_dist = (prior_high - price) / prior_high * 100
            if return_dist >= 0.3:
                bb = bars_4h[j]
                bb_range = bb['h'] - bb['l']
                wick = bb['h'] - max(bb['c'], bb['o'])
                wick_r = wick / bb_range if bb_range > 0 else 0
                vol_r = volumes_4h[j] / max(avg_vol_4h[j], 1)
                taker_sell = 1 - (bb['tb'] / max(bb['v'], 0.01))
                
                conv = 0.0
                # Wick rejection on 4H is much more significant
                if wick_r >= 0.3: conv += 0.3
                # Volume on breakout (high vol = more trapped)
                if vol_r >= 1.0: conv += 0.25
                # Taker flip
                if taker_sell >= 0.50: conv += 0.2
                # Price below breakout level
                if price < prior_high: conv += 0.15
                # Current bar bearish
                if closes_4h[i] < bars_4h[i]['o']: conv += 0.1
                
                if conv >= 0.5:
                    fb_signals_4h.append((i, 'LONG', conv, prior_high, j))
            break
    
    # Check for failed LONG breakout → SHORT signal
    for j in range(max(i-4, 48), i):
        if lows_4h[j] < prior_low:
            return_dist = (price - prior_low) / prior_low * 100
            if return_dist >= 0.3:
                bb = bars_4h[j]
                bb_range = bb['h'] - bb['l']
                wick = min(bb['c'], bb['o']) - bb['l']
                wick_r = wick / bb_range if bb_range > 0 else 0
                vol_r = volumes_4h[j] / max(avg_vol_4h[j], 1)
                taker_buy = bb['tb'] / max(bb['v'], 0.01)
                
                conv = 0.0
                if wick_r >= 0.3: conv += 0.3
                if vol_r >= 1.0: conv += 0.25
                if taker_buy >= 0.50: conv += 0.2
                if price > prior_low: conv += 0.15
                if closes_4h[i] > bars_4h[i]['o']: conv += 0.1
                
                if conv >= 0.5:
                    fb_signals_4h.append((i, 'SHORT', conv, prior_low, j))
            break

print(f"Found {len(fb_signals_4h)} 4H failed breakout signals", flush=True)
if fb_signals_4h:
    longs = sum(1 for s in fb_signals_4h if s[1] == 'LONG')
    shorts = sum(1 for s in fb_signals_4h if s[1] == 'SHORT')
    convs = [s[2] for s in fb_signals_4h]
    print(f"  LONG: {longs}, SHORT: {shorts}", flush=True)
    print(f"  Conviction: min={min(convs):.2f}, max={max(convs):.2f}, avg={np.mean(convs):.2f}", flush=True)

# ============================================================
# Entry on 15m: wait for reversal confirmation
# ============================================================
print("\nBuilding 15m entries from 4H signals...", flush=True)

# For each 4H signal, find the best 15m entry point
# Entry criteria on 15m:
# 1. Price in direction of signal
# 2. Volume confirmation
# 3. Taker in signal direction

mtf_signals = []  # (15m_idx, direction, conviction_4h, entry_price)

for sig_4h_idx, direction, conv_4h, breakout_level, breakout_bar in fb_signals_4h:
    # The 4H signal is detected at bar sig_4h_idx
    # Look for 15m entry in the NEXT 4H bar(s) — up to 2 bars (8 hours)
    entry_start = bars_4h[sig_4h_idx]['idx_start']  # Start of current 4H bar in 15m
    entry_end_idx = min(sig_4h_idx + 3, N4)  # Look in next 2 x 4H bars
    entry_end = bars_4h[entry_end_idx - 1]['idx_end'] if entry_end_idx <= N4 else N15
    
    best_entry = None
    best_conv = 0
    
    for idx_15m in range(entry_start, min(entry_end, N15)):
        price_15m = closes_15m[idx_15m]
        vol_r = volumes_15m[idx_15m] / max(avg_vol_15m[idx_15m], 1)
        taker = bars_15m[idx_15m]['tb'] / max(bars_15m[idx_15m]['v'], 0.01)
        
        # Entry confirmation
        entry_conv = 0.0
        
        if direction == 'LONG':
            # Price should be bouncing (close > open)
            if closes_15m[idx_15m] > bars_15m[idx_15m]['o']:
                entry_conv += 0.3
            # Volume spike
            if vol_r >= 1.2:
                entry_conv += 0.3
            # Taker buying
            if taker >= 0.53:
                entry_conv += 0.2
            # Price above breakout level (reclaim)
            if price_15m > breakout_level:
                entry_conv += 0.2
        else:
            if closes_15m[idx_15m] < bars_15m[idx_15m]['o']:
                entry_conv += 0.3
            if vol_r >= 1.2:
                entry_conv += 0.3
            if taker <= 0.47:
                entry_conv += 0.2
            if price_15m < breakout_level:
                entry_conv += 0.2
        
        if entry_conv > best_conv:
            best_conv = entry_conv
            best_entry = idx_15m
        
        # Take first good entry
        if entry_conv >= 0.5:
            mtf_signals.append((idx_15m, direction, conv_4h, price_15m, entry_conv))
            break
    
    # If no 15m confirmation found, use the 4H close as entry
    if best_entry is not None and best_conv >= 0.3:
        if not any(s[0] >= entry_start and s[0] < entry_end for s in mtf_signals):
            mtf_signals.append((best_entry, direction, conv_4h, closes_15m[best_entry], best_conv))

print(f"Generated {len(mtf_signals)} MTF entry signals", flush=True)

# ============================================================
# Sweep TP/SL on MTF signals
# ============================================================
def sim_mtf(signals, tp_pct, sl_pct, hold_hours, min_conv_4h, min_conv_entry, trend_4h, dedup_bars=48):
    trades = []
    capital = 200.0
    peak = capital
    max_dd = 0
    last_bar = -999
    
    for idx_15m, direction, conv_4h, entry_price, conv_entry in signals:
        if conv_4h < min_conv_4h: continue
        if conv_entry < min_conv_entry: continue
        if idx_15m - last_bar < dedup_bars: continue
        
        # 4H trend filter
        if trend_4h:
            # Find which 4H bar this 15m bar belongs to
            bar_4h_idx = 0
            for k in range(len(bars_4h)):
                if bars_4h[k]['idx_start'] <= idx_15m < bars_4h[k]['idx_end']:
                    bar_4h_idx = k
                    break
            if bar_4h_idx >= 200:
                if direction == 'LONG' and closes_4h[bar_4h_idx] < ema200_4h[bar_4h_idx]:
                    continue
                if direction == 'SHORT' and closes_4h[bar_4h_idx] > ema200_4h[bar_4h_idx]:
                    continue
        
        entry = entry_price * (1 + SLIP if direction == 'LONG' else 1 - SLIP)
        sl_p = entry * (1 - sl_pct/100) if direction == 'LONG' else entry * (1 + sl_pct/100)
        tp_p = entry * (1 + tp_pct/100) if direction == 'LONG' else entry * (1 - tp_pct/100)
        last_bar = idx_15m
        
        outcome = None
        exit_p = None
        for j in range(idx_15m+1, min(idx_15m+hold_hours*4+1, N15)):
            if direction == 'LONG':
                if highs_15m[j] >= tp_p: outcome = 'W'; exit_p = tp_p; break
                if lows_15m[j] <= sl_p: outcome = 'L'; exit_p = sl_p; break
            else:
                if lows_15m[j] <= tp_p: outcome = 'W'; exit_p = tp_p; break
                if highs_15m[j] >= sl_p: outcome = 'L'; exit_p = sl_p; break
        if not outcome:
            exit_p = closes_15m[min(idx_15m+hold_hours*4, N15-1)]
            outcome = 'T'
        
        pnl = ((exit_p-entry)/entry*100) if direction=='LONG' else ((entry-exit_p)/entry*100)
        pnl -= FEE*2
        size = capital * 0.10 * 25
        capital += size * pnl / 100
        if capital > peak: peak = capital
        dd = (peak-capital)/peak*100 if peak>0 else 0
        if dd > max_dd: max_dd = dd
        trades.append({'o': outcome, 'pnl': pnl, 'ts': bars_15m[idx_15m]['ts'].isoformat()[:7]})
    
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


print("\nSweeping MTF configs...", flush=True)
results = []
tested = 0

for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for sl in [0.3, 0.5, 0.75, 1.0]:
        if sl >= tp: continue
        for hold in [8, 12, 16, 24, 32]:
            for min_conv_4h in [0.5, 0.6, 0.7, 0.8]:
                for min_entry in [0.3, 0.5]:
                    for trend in [None, True]:
                        r = sim_mtf(mtf_signals, tp, sl, hold, min_conv_4h, min_entry, trend)
                        tested += 1
                        if r:
                            results.append({**r, 'tp': tp, 'sl': sl, 'hold': hold,
                                           'min_conv_4h': min_conv_4h, 'min_entry': min_entry,
                                           'trend': trend})

print(f"Tested {tested} configs", flush=True)

results.sort(key=lambda x: x['pf'] * (x['wr']/100), reverse=True)

ent = [r for r in results if r['wr']>=65 and r['pf']>=2.0 and r['dd']<25 and r['bad_m']<=3 and r['trades']>=10]
good = [r for r in results if r['wr']>=55 and r['pf']>=1.5 and r['trades']>=10]
ok = [r for r in results if r['wr']>=50 and r['pf']>=1.2 and r['trades']>=8]

print(f"\nEnterprise (WR>=65, PF>=2.0, DD<25, bad_m<=3): {len(ent)}")
print(f"Good (WR>=55, PF>=1.5, T>=10): {len(good)}")
print(f"OK (WR>=50, PF>=1.2, T>=8): {len(ok)}")

for label, subset in [("ENTERPRISE", ent), ("GOOD", good), ("OK", ok)]:
    if not subset: continue
    print(f"\n{'='*110}")
    print(f"{label} RESULTS ({len(subset)} configs)")
    print(f"{'='*110}")
    print(f"  {'#':>2s} {'TP':>4s} {'SL':>4s} {'RR':>4s} {'H':>3s} {'C4':>3s} {'Ce':>3s} {'Tr':>2s} | {'N':>4s} {'W':>3s} {'L':>3s} {'T':>3s} {'WR':>5s} {'PF':>5s} {'PnL':>7s} {'DD':>5s} {'AvW':>5s} {'AvL':>5s} {'BM':>3s}")
    print("  " + "-" * 105)
    for i, r in enumerate(subset[:30]):
        rr = round(r['tp']/r['sl'], 1)
        tr = 'T' if r['trend'] else '-'
        print(f"  {i+1:>2d} {r['tp']:>4.1f} {r['sl']:>4.2f} {rr:>4.1f} {r['hold']:>3d} {r['min_conv_4h']:>3.1f} {r['min_entry']:>3.1f} {tr:>2s} | {r['trades']:>4d} {r['wins']:>3d} {r['losses']:>3d} {r['timeouts']:>3d} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>+7.1f} {r['dd']:>5.1f} {r['avg_w']:>5.2f} {r['avg_l']:>5.2f} {r['bad_m']:>3d}")

best = ent if ent else (good if good else ok)
if best:
    b = best[0]
    print(f"\n{'='*80}")
    print("BEST MTF CONFIG")
    print(f"{'='*80}")
    print(f"  TP={b['tp']}% SL={b['sl']}% Hold={b['hold']}h")
    print(f"  4H Conv>={b['min_conv_4h']}, 15m Entry Conv>={b['min_entry']}")
    print(f"  4H Trend={'EMA200' if b['trend'] else 'None'}")
    print(f"  R:R=1:{b['tp']/b['sl']:.1f}")
    print(f"  {b['trades']}T {b['wr']}%WR {b['pf']}PF PnL={b['pnl']}% DD={b['dd']}%")
    print(f"\n  Monthly:")
    for m, v in sorted(b['monthly'].items()):
        print(f"    {m}: PnL={v:+.1f}%")

print("\nDone")
