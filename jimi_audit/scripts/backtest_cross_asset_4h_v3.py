#!/usr/bin/env python3
"""CROSS-ASSET 4H -- vectorized z-score, writes results to file."""
import json, time, math, sys
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

t0 = time.time()
OUT = '/tmp/cross_asset_results.txt'
open(OUT, 'w').close()

def log(msg):
    print(msg); sys.stdout.flush()
    with open(OUT, 'a') as f: f.write(msg + '\n')

with open('/root/.openclaw/workspace/jimi_audit/data/eth_full_1h.json') as f:
    eth_raw = json.load(f)
with open('/root/.openclaw/workspace/jimi_audit/data/btc_1h.json') as f:
    btc_raw = json.load(f)

eth_ts = {c[0]: i for i, c in enumerate(eth_raw)}
btc_ts = {c[0]: i for i, c in enumerate(btc_raw)}
common_ts = sorted(set(eth_ts.keys()) & set(btc_ts.keys()))

eth_c = np.array([float(eth_raw[eth_ts[ts]][4]) for ts in common_ts])
eth_h = np.array([float(eth_raw[eth_ts[ts]][2]) for ts in common_ts])
eth_l = np.array([float(eth_raw[eth_ts[ts]][3]) for ts in common_ts])
eth_v = np.array([float(eth_raw[eth_ts[ts]][5]) for ts in common_ts])
btc_c = np.array([float(btc_raw[btc_ts[ts]][4]) for ts in common_ts])
ts_arr = np.array(common_ts)

log("Common 1h: %d (%s to %s)" % (len(common_ts),
    datetime.fromtimestamp(common_ts[0]/1000,tz=timezone.utc).strftime('%Y-%m-%d'),
    datetime.fromtimestamp(common_ts[-1]/1000,tz=timezone.utc).strftime('%Y-%m-%d')))

step = 4
idx = np.arange(0, len(eth_c), step)
eth4_c = eth_c[idx]
eth4_h = np.array([max(eth_h[i:min(i+step, len(eth_h))]) for i in idx])
eth4_l = np.array([min(eth_l[i:min(i+step, len(eth_l))]) for i in idx])
eth4_v = np.array([sum(eth_v[i:min(i+step, len(eth_v))]) for i in idx])
btc4_c = btc_c[idx]
eth4_ts = ts_arr[idx]
N4 = len(eth4_c)
log("4h candles: %d" % N4)

ratio = eth4_c / np.maximum(btc4_c, 1)

def rolling_z_np(data, lb):
    n = len(data)
    z = np.zeros(n)
    for i in range(lb, n):
        w = data[i-lb:i+1]
        mu = np.mean(w)
        sigma = np.std(w)
        z[i] = (data[i] - mu) / sigma if sigma > 0 else 0
    return z

z50 = rolling_z_np(ratio, 50)
z100 = rolling_z_np(ratio, 100)
z200 = rolling_z_np(ratio, 200)

vol_avg = np.zeros(N4)
for i in range(20, N4):
    vol_avg[i] = np.mean(eth4_v[max(0,i-20):i])
vol_ratio = eth4_v / np.maximum(vol_avg, 1)

def ema_np(data, p):
    out = np.zeros(len(data))
    out[0] = data[0]; k = 2.0/(p+1)
    for i in range(1, len(data)):
        out[i] = data[i]*k + out[i-1]*(1-k)
    return out

ema200 = ema_np(eth4_c, 200)

log("Z-50 extremes: %d" % np.sum(np.abs(z50[250:])>2))
log("Z-100 extremes: %d" % np.sum(np.abs(z100[250:])>2))
log("Z-200 extremes: %d" % np.sum(np.abs(z200[250:])>2))

def backtest(tp_pct, sl_pct, hold, zt, z_type):
    z_arr = {'50': z50, '100': z100, '200': z200}[z_type]
    cap = 200.0; pk = cap; dd = 0.0
    wins = 0; total = 0; gp = 0.0; gl = 0.0
    i = 250
    while i < N4 - 1:
        z = z_arr[i]
        if abs(z) < zt or vol_ratio[i] < 0.8:
            i += 1; continue
        price = eth4_c[i]
        direction = 'SHORT' if z > zt else 'LONG'
        if direction == 'LONG' and price < ema200[i] * 0.92: i += 1; continue
        if direction == 'SHORT' and price > ema200[i] * 1.08: i += 1; continue
        entry = eth4_c[i+1]
        if direction == 'LONG': tp_p = entry*(1+tp_pct); sl_p = entry*(1-sl_pct)
        else: tp_p = entry*(1-tp_pct); sl_p = entry*(1+sl_pct)
        sl_d = abs(entry - sl_p)
        if sl_d == 0: i += 1; continue
        sz = min(cap*0.02/sl_d, cap*25/entry)
        if sz <= 0: i += 1; continue
        closed = False
        for j in range(i+1, min(i+1+hold, N4)):
            hit = False; ep = 0
            if direction == 'LONG':
                if eth4_h[j] >= tp_p: hit=True; ep=tp_p
                elif eth4_l[j] <= sl_p: hit=True; ep=sl_p
            else:
                if eth4_l[j] <= tp_p: hit=True; ep=tp_p
                elif eth4_h[j] >= sl_p: hit=True; ep=sl_p
            if hit:
                pnl = (ep-entry)*sz if direction=='LONG' else (entry-ep)*sz
                pnl -= entry*sz*0.001*2
                cap += pnl; total += 1
                if pnl > 0: wins += 1; gp += pnl
                else: gl += abs(pnl)
                if cap > pk: pk = cap
                d = (pk-cap)/pk*100
                if d > dd: dd = d
                closed = True; i = j+1; break
        if not closed:
            j = min(i+hold, N4-1); ep = eth4_c[j]
            pnl = (ep-entry)*sz if direction=='LONG' else (entry-ep)*sz
            pnl -= entry*sz*0.001*2
            cap += pnl; total += 1
            if pnl > 0: wins += 1; gp += pnl
            else: gl += abs(pnl)
            if cap > pk: pk = cap
            d = (pk-cap)/pk*100
            if d > dd: dd = d
            i = j+1
    wr = wins/total*100 if total > 0 else 0
    pf = gp/gl if gl > 0 else float('inf')
    exp = (cap-200)/200/total*100 if total > 0 else 0
    return {'cap':round(cap,2),'dd':round(dd,1),'trades':total,'wins':wins,
            'wr':round(wr,1),'pf':round(pf,2),'exp':round(exp,4)}

log("\n=== CROSS-ASSET 4H BACKTEST ===")
log("4h bars: %d, cost: 0.10%%" % N4)

configs = [
    (1.0, 0.75, 6, "conservative"),
    (1.5, 1.0, 8, "standard"),
    (2.0, 1.5, 12, "wide"),
    (1.0, 0.5, 4, "tight_fast"),
    (2.5, 1.5, 16, "very_wide"),
]

for zt_name in ['50', '100', '200']:
    log("\n--- Z-%s ---" % zt_name)
    for tp, sl, hold, label in configs:
        for zt in [1.5, 2.0, 2.5]:
            r = backtest(tp/100, sl/100, hold, zt, zt_name)
            if r['trades'] >= 3:
                profitable = r['pf'] > 1.0 and r['wr'] > 50 and r['exp'] > 0.10
                sign = "+" if profitable else "-"
                log("%s %12s z>%s TP=%s SL=%s h=%s | n=%3d WR=%5.1f%% PF=%5.2f $%8.2f dd=%5.1f%% exp=%+.4f%%" % (
                    sign, label, zt, tp, sl, hold, r['trades'], r['wr'], r['pf'], r['cap'], r['dd'], r['exp']))

log("\n=== ISOLATION GATE ===")
for zt_name in ['50', '100', '200']:
    for zt in [1.5, 2.0, 2.5]:
        r = backtest(1.0/100, 0.75/100, 6, zt, zt_name)
        n = r['trades']
        if n >= 30:
            p_hat = r['wins']/n
            se = math.sqrt(0.5*0.5/n)
            z_stat = (p_hat-0.5)/se if se > 0 else 0
            p_val = 0.5*(1-math.erf(abs(z_stat)/math.sqrt(2)))
            above = r['exp'] > 0.10
            sign = "PASS" if p_val < 0.1 and above else "FAIL"
            log("%s z%s>z=%s n=%d WR=%s%% exp=%.4f%% z=%.2f p=%.4f" % (sign, zt_name, zt, n, r['wr'], r['exp'], z_stat, p_val))

log("\nTime: %.1fs" % (time.time()-t0))
log("DONE")
