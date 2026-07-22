#!/usr/bin/env python3
"""
Failed Breakout v5 — Liquidity Grab Focus
===========================================
Key insight: Don't just detect "broke level and came back."
Detect: "swept liquidity beyond level, then reversed with intent."

What matters:
1. The sweep WENT BEYOND the level (grabbed stops)
2. The reversal has MOMENTUM (not just drifting back)
3. There's CONFLUENCE (volume spike, taker flip, wick rejection)

Focus on the INTENT of the move, not just the mechanics.
"""
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
# Pre-compute liquidity levels (swing highs/lows)
# ============================================================
print("Computing liquidity levels...", flush=True)

# Multi-timeframe swing levels
# 4H swings (48 x 15m = 12h lookback)
swing_high_4h = np.zeros(N)
swing_low_4h = np.zeros(N)
for i in range(48, N):
    swing_high_4h[i] = np.max(highs[i-48:i])
    swing_low_4h[i] = np.min(lows[i-48:i])

# 1D swings (96 x 15m = 24h lookback)
swing_high_1d = np.zeros(N)
swing_low_1d = np.zeros(N)
for i in range(96, N):
    swing_high_1d[i] = np.max(highs[i-96:i])
    swing_low_1d[i] = np.min(lows[i-96:i])

# ============================================================
# Detect liquidity grab signals
# ============================================================
print("Detecting liquidity grabs...", flush=True)

signals = []

for i in range(100, N):
    price = closes[i]
    atr_val = atr[i]
    if atr_val <= 0: continue
    
    current = bars[i]
    h = current['h']
    l = current['l']
    c = current['c']
    o = current['o']
    v = current['v']
    tb = current['tb']
    
    # === LIQUIDITY GRAB LONG ===
    # Price swept below a significant level, then closed above it
    # This is a stop hunt on the downside → reversal up
    
    # Check if current bar or recent bars swept below swing low
    for lookback, level_name in [(48, '4H'), (96, '1D')]:
        if i < lookback + 10: continue
        
        swing_low = swing_low_1d[i] if lookback == 96 else swing_low_4h[i]
        swing_high = swing_high_1d[i] if lookback == 96 else swing_high_4h[i]
        
        # LONG: swept below swing_low, closed above
        swept_below = l < swing_low
        closed_above = c > swing_low
        
        if swept_below and closed_above:
            # How far below did it sweep?
            sweep_depth = (swing_low - l) / atr_val  # in ATR multiples
            
            # Did it close as a bullish candle?
            bullish_close = c > o
            
            # Volume spike?
            vol_r = v / max(avg_vol[i], 1)
            
            # Taker buying?
            taker_buy = tb / max(v, 0.01)
            
            # Wick rejection (long lower wick = buyers defending)
            wick_low = min(o, c) - l
            bar_range = h - l
            wick_ratio = wick_low / bar_range if bar_range > 0 else 0
            
            # Conviction scoring
            conv = 0.0
            
            # Sweep depth (deeper sweep = more stops grabbed = stronger signal)
            if sweep_depth >= 0.5: conv += 0.2
            if sweep_depth >= 1.0: conv += 0.1
            
            # Bullish close after sweep
            if bullish_close: conv += 0.2
            
            # Volume spike on sweep
            if vol_r >= 1.2: conv += 0.15
            if vol_r >= 1.5: conv += 0.05
            
            # Taker flip to buying
            if taker_buy >= 0.53: conv += 0.15
            
            # Wick rejection
            if wick_ratio >= 0.4: conv += 0.1
            if wick_ratio >= 0.6: conv += 0.05
            
            # Bonus: swept below 1D level (bigger liquidity)
            if lookback == 96: conv += 0.1
            
            if conv >= 0.5:
                signals.append((i, 'LONG', conv, c, atr_val, level_name, sweep_depth))
        
        # SHORT: swept above swing_high, closed below
        swept_above = h > swing_high
        closed_below = c < swing_high
        
        if swept_above and closed_below:
            sweep_depth = (h - swing_high) / atr_val
            bearish_close = c < o
            vol_r = v / max(avg_vol[i], 1)
            taker_sell = 1 - (tb / max(v, 0.01))
            wick_high = h - max(o, c)
            bar_range = h - l
            wick_ratio = wick_high / bar_range if bar_range > 0 else 0
            
            conv = 0.0
            if sweep_depth >= 0.5: conv += 0.2
            if sweep_depth >= 1.0: conv += 0.1
            if bearish_close: conv += 0.2
            if vol_r >= 1.2: conv += 0.15
            if vol_r >= 1.5: conv += 0.05
            if taker_sell >= 0.53: conv += 0.15
            if wick_ratio >= 0.4: conv += 0.1
            if wick_ratio >= 0.6: conv += 0.05
            if lookback == 96: conv += 0.1
            
            if conv >= 0.5:
                signals.append((i, 'SHORT', conv, c, atr_val, level_name, sweep_depth))

print(f"Found {len(signals)} liquidity grab signals", flush=True)
if signals:
    longs = sum(1 for s in signals if s[1] == 'LONG')
    shorts = sum(1 for s in signals if s[1] == 'SHORT')
    convs = [s[2] for s in signals]
    depths = [s[6] for s in signals]
    print(f"  LONG: {longs}, SHORT: {shorts}", flush=True)
    print(f"  Conviction: min={min(convs):.2f}, max={max(convs):.2f}, avg={np.mean(convs):.2f}", flush=True)
    print(f"  Sweep depth: min={min(depths):.2f} ATR, max={max(depths):.2f} ATR, avg={np.mean(depths):.2f} ATR", flush=True)
    
    # By level type
    by_level = {}
    for s in signals:
        lvl = s[5]
        if lvl not in by_level: by_level[lvl] = 0
        by_level[lvl] += 1
    for lvl, cnt in sorted(by_level.items()):
        print(f"  {lvl}: {cnt} signals", flush=True)

# ============================================================
# Sweep TP/SL
# ============================================================
def sim(signals_subset, tp_pct, sl_pct, hold_hours, min_conv, trend_filter, session_filter, dedup=8):
    trades = []
    capital = 200.0
    peak = capital
    max_dd = 0
    last_bar = -999
    
    for idx, direction, conv, entry_price, atr_val, level_name, sweep_depth in signals_subset:
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


print("\nSweeping...", flush=True)
results = []
tested = 0

for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for sl in [0.3, 0.5, 0.75, 1.0]:
        if sl >= tp: continue
        for hold in [4, 8, 12, 16, 24]:
            for conv in [0.5, 0.6, 0.7, 0.8]:
                for trend in [None, True]:
                    for sess in [None, 'ol', 'ln']:
                        r = sim(signals, tp, sl, hold, conv, trend, sess)
                        tested += 1
                        if r:
                            results.append({**r, 'tp': tp, 'sl': sl, 'hold': hold,
                                           'conv': conv, 'trend': trend, 'sess': sess or 'all'})

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
    print(f"  {'#':>2s} {'TP':>4s} {'SL':>4s} {'RR':>4s} {'H':>3s} {'Cv':>3s} {'Tr':>2s} {'Se':>2s} | {'N':>4s} {'W':>3s} {'L':>3s} {'T':>3s} {'WR':>5s} {'PF':>5s} {'PnL':>7s} {'DD':>5s} {'AvW':>5s} {'AvL':>5s} {'BM':>3s}")
    print("  " + "-" * 105)
    for i, r in enumerate(subset[:30]):
        rr = round(r['tp']/r['sl'], 1)
        tr = 'T' if r['trend'] else '-'
        print(f"  {i+1:>2d} {r['tp']:>4.1f} {r['sl']:>4.2f} {rr:>4.1f} {r['hold']:>3d} {r['conv']:>3.1f} {tr:>2s} {r['sess']:>2s} | {r['trades']:>4d} {r['wins']:>3d} {r['losses']:>3d} {r['timeouts']:>3d} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>+7.1f} {r['dd']:>5.1f} {r['avg_w']:>5.2f} {r['avg_l']:>5.2f} {r['bad_m']:>3d}")

best = ent if ent else (good if good else ok)
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
