#!/usr/bin/env python3
"""
CROSS-ASSET DIVERGENCE — 4H BACKTEST
Hypothesis: ETH/BTC ratio z-score extremes predict ETH mean reversion.
Data: ETH 1h (Binance) + BTC 1h (Binance), 2017-2026.
"""
import json, sys, time, math
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

t0 = time.time()

# Load data
with open('/root/.openclaw/workspace/jimi_audit/data/eth_full_1h.json') as f:
    eth_raw = json.load(f)
with open('/root/.openclaw/workspace/jimi_audit/data/btc_1h.json') as f:
    btc_raw = json.load(f)

print(f"ETH: {len(eth_raw)} candles")
print(f"BTC: {len(btc_raw)} candles")

# Align by timestamp
eth_ts = {c[0]: i for i, c in enumerate(eth_raw)}
btc_ts = {c[0]: i for i, c in enumerate(btc_raw)}
common_ts = sorted(set(eth_ts.keys()) & set(btc_ts.keys()))
print(f"Common: {len(common_ts)} candles ({datetime.fromtimestamp(common_ts[0]/1000,tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(common_ts[-1]/1000,tz=timezone.utc).strftime('%Y-%m-%d')})")

eth_c = np.array([float(eth_raw[eth_ts[ts]][4]) for ts in common_ts])
eth_h = np.array([float(eth_raw[eth_ts[ts]][2]) for ts in common_ts])
eth_l = np.array([float(eth_raw[eth_ts[ts]][3]) for ts in common_ts])
eth_v = np.array([float(eth_raw[eth_ts[ts]][5]) for ts in common_ts])
btc_c = np.array([float(btc_raw[btc_ts[ts]][4]) for ts in common_ts])
ts_arr = np.array(common_ts)

N = len(common_ts)

# Resample to 4h
def resample(arr, ts, step=4):
    out = []
    i = 0
    while i < len(arr):
        out.append(arr[i])
        i += step
    return np.array(out)

def resample_4h_full(c, h, l, v, ts, step=4):
    co, ho, lo, vo, tso = [], [], [], [], []
    i = 0
    while i < len(c):
        end = min(i+step, len(c))
        co.append(c[i])
        ho.append(max(h[i:end]))
        lo.append(min(l[i:end]))
        vo.append(sum(v[i:end]))
        tso.append(ts[i])
        i = step
    return np.array(co), np.array(ho), np.array(lo), np.array(vo), np.array(tso)

eth4_c, eth4_h, eth4_l, eth4_v, eth4_ts = resample_4h_full(eth_c, eth_h, eth_l, eth_v, ts_arr)
btc4_c = resample(btc_c, ts_arr)
N4 = len(eth4_c)
print(f"4h candles: {N4}")

# ETH/BTC ratio
ratio = eth4_c / np.maximum(btc4_c, 1)

# Z-score with multiple lookbacks
def rolling_z(data, lb):
    z = np.zeros(len(data))
    for i in range(lb, len(data)):
        w = data[i-lb:i+1]
        mu, sigma = np.mean(w), np.std(w)
        z[i] = (data[i] - mu) / sigma if sigma > 0 else 0
    return z

z_50 = rolling_z(ratio, 50)    # ~8 days
z_100 = rolling_z(ratio, 100)  # ~17 days
z_200 = rolling_z(ratio, 200)  # ~33 days

# Volume ratio
vol_avg = np.zeros(N4)
for i in range(20, N4):
    vol_avg[i] = np.mean(eth4_v[max(0,i-20):i])
vol_ratio = eth4_v / np.maximum(vol_avg, 1)

# EMA
def ema(data, p):
    out = np.zeros(len(data))
    out[0] = data[0]; k = 2/(p+1)
    for i in range(1, len(data)):
        out[i] = data[i]*k + out[i-1]*(1-k)
    return out

ema50 = ema(eth4_c, 50)
ema200 = ema(eth4_c, 200)

def detect(i, z_threshold, lookback_type='100'):
    if i < 250: return None
    
    z = {'50': z_50, '100': z_100, '200': z_200}[lookback_type][i]
    price = eth4_c[i]
    
    if abs(z) < z_threshold: return None
    if vol_ratio[i] < 0.8: return None
    
    # Mean reversion direction
    if z > z_threshold: direction = 'SHORT'  # ETH overvalued
    elif z < -z_threshold: direction = 'LONG'  # ETH undervalued
    else: return None
    
    # Trend filter: only fade if not in strong trend against
    if direction == 'LONG' and price < ema200[i] * 0.92:
        return None  # Too far below EMA, might keep falling
    if direction == 'SHORT' and price > ema200[i] * 1.08:
        return None  # Too far above EMA, might keep rising
    
    z_str = min(abs(z) / 5.0, 0.3)
    vol_str = min((vol_ratio[i] - 1.0) / 3.0, 0.2)
    conv = min(0.4 + z_str + vol_str, 0.85)
    
    return {'direction': direction, 'conviction': conv, 'price': price, 'z': z}


def backtest(tp_pct, sl_pct, hold, z_thresh=2.0, z_type='100', init=200):
    cap = float(init); pk = cap; dd = 0.0
    wins = 0; losses = 0; total = 0; gp = 0.0; gl = 0.0
    monthly = defaultdict(lambda: {'t': 0, 'w': 0, 'pnl': 0.0})
    
    i = 250
    while i < N4 - 1:
        sig = detect(i, z_thresh, z_type)
        if sig is None: i += 1; continue
        
        entry = eth4_c[i+1]
        if sig['direction'] == 'LONG':
            tp_p = entry*(1+tp_pct); sl_p = entry*(1-sl_pct)
        else:
            tp_p = entry*(1-tp_pct); sl_p = entry*(1+sl_pct)
        
        sl_d = abs(entry - sl_p)
        if sl_d == 0: i += 1; continue
        sz = min(cap*0.02/sl_d, cap*25/entry)
        if sz <= 0: i += 1; continue
        
        closed = False
        for j in range(i+1, min(i+1+hold, N4)):
            hit = False; ep = 0
            if sig['direction'] == 'LONG':
                if eth4_h[j] >= tp_p: hit=True; ep=tp_p
                elif eth4_l[j] <= sl_p: hit=True; ep=sl_p
            else:
                if eth4_l[j] <= tp_p: hit=True; ep=tp_p
                elif eth4_h[j] >= sl_p: hit=True; ep=sl_p
            if hit:
                pnl = (ep-entry)*sz if sig['direction']=='LONG' else (entry-ep)*sz
                pnl -= entry*sz*0.001*2
                cap += pnl; total += 1
                mk = datetime.fromtimestamp(eth4_ts[i]/1000,tz=timezone.utc).strftime('%Y-%m')
                monthly[mk]['t'] += 1; monthly[mk]['pnl'] += pnl
                if pnl > 0: wins+=1; gp+=pnl; monthly[mk]['w'] += 1
                else: losses+=1; gl+=abs(pnl)
                if cap>pk:pk=cap
                d=(pk-cap)/pk*100
                if d>dd:dd=d
                closed=True; i=j+1; break
        if not closed:
            j=min(i+hold, N4-1); ep=eth4_c[j]
            pnl=(ep-entry)*sz if sig['direction']=='LONG' else (entry-ep)*sz
            pnl-=entry*sz*0.001*2
            cap+=pnl; total+=1
            mk = datetime.fromtimestamp(eth4_ts[i]/1000,tz=timezone.utc).strftime('%Y-%m')
            monthly[mk]['t'] += 1; monthly[mk]['pnl'] += pnl
            if pnl>0:wins+=1;gp+=pnl;monthly[mk]['w']+=1
            else:losses+=1;gl+=abs(pnl)
            if cap>pk:pk=cap
            d=(pk-cap)/pk*100
            if d>dd:dd=d
            i=j+1
    
    wr = wins/total*100 if total>0 else 0
    pf = gp/gl if gl>0 else float('inf')
    exp = (cap-init)/init/total*100 if total>0 else 0
    return {'cap':round(cap,2),'dd':round(dd,1),'trades':total,'wins':wins,
            'wr':round(wr,1),'pf':round(pf,2),'exp':round(exp,4),'monthly':dict(monthly)}


print(f"\n=== CROSS-ASSET 4H BACKTEST ===")
print(f"Data: {N4} 4h bars ({datetime.fromtimestamp(eth4_ts[0]/1000,tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(eth4_ts[-1]/1000,tz=timezone.utc).strftime('%Y-%m-%d')})")
print(f"Round-trip cost: 0.10%\n")

# Z-score distribution
for z_name, z_arr in [('50', z_50), ('100', z_100), ('200', z_200)]:
    valid = z_arr[250:]
    print(f"Z-{z_name}: mean={np.mean(valid):.3f}, std={np.std(valid):.3f}, "
          f"extreme(>2): {sum(1 for z in valid if abs(z)>2)} ({sum(1 for z in valid if abs(z)>2)/len(valid)*100:.1f}%)")
print()

# Grid search
configs = [
    (1.0, 0.75, 6, "conservative"),
    (1.5, 1.0, 8, "standard"),
    (2.0, 1.5, 12, "wide"),
    (1.0, 0.5, 4, "tight_fast"),
    (2.5, 1.5, 16, "very_wide"),
    (1.5, 0.75, 6, "medium"),
]

for z_type in ['50', '100', '200']:
    print(f"\n--- Z-Score Lookback: {z_type} bars ---")
    for tp, sl, hold, label in configs:
        for z_thresh in [1.5, 2.0, 2.5]:
            r = backtest(tp/100, sl/100, hold, z_thresh, z_type)
            profitable = r['pf'] > 1.0 and r['wr'] > 50 and r['exp'] > 0.10
            sign = "✅" if profitable else "❌"
            if r['trades'] >= 5:
                print(f"{sign} {label:12s} z>{z_thresh} TP={tp}% SL={sl}% hold={hold} | "
                      f"trades={r['trades']:4d} WR={r['wr']:5.1f}% PF={r['pf']:5.2f} "
                      f"cap=${r['cap']:8.2f} dd={r['dd']:5.1f}% exp={r['exp']:.4f}%")

# === ISOLATION GATE on best found ===
print(f"\n=== ISOLATION GATE (best configs) ===")
best_configs = [
    (1.0, 0.75, 6, 2.0, '100'),
    (1.5, 1.0, 8, 2.0, '100'),
    (1.0, 0.5, 4, 1.5, '50'),
    (2.0, 1.5, 12, 2.0, '100'),
]

for tp, sl, hold, zt, zt_name in best_configs:
    full = backtest(tp/100, sl/100, hold, zt, zt_name)
    n = full['trades']
    if n >= 30:
        p_hat = full['wins'] / n
        se = math.sqrt(0.5 * 0.5 / n)
        z_stat = (p_hat - 0.5) / se if se > 0 else 0
        p_val = 0.5 * (1 - math.erf(abs(z_stat) / math.sqrt(2)))
        above = full['exp'] > 0.10
        sign = "✅" if p_val < 0.1 and above else "❌"
        print(f"{sign} z>{zt}({zt_name}) TP={tp}% SL={sl}% hold={hold} | n={n} WR={full['wr']}% "
              f"exp={full['exp']:.4f}% z={z_stat:.2f} p={p_val:.4f} edge={full['exp']-0.10:+.4f}%")
    else:
        print(f"⚠️  z>{zt}({zt_name}) TP={tp}% SL={sl}% hold={hold} | only {n} events")

print(f"\nTime: {time.time()-t0:.1f}s")
