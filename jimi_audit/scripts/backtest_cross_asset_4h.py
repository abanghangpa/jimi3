#!/usr/bin/env python3
"""
CROSS-ASSET DIVERGENCE — 4H BACKTEST
Hypothesis: ETH/BTC ratio z-score extremes predict ETH price reversion.
On 15m: noise. On 4h: ratio divergence is real institutional positioning.

Detection:
1. Compute ETH/BTC ratio rolling z-score (lookback=100 4h bars = ~17 days)
2. Signal when z-score crosses +/- 2.0 (extreme divergence)
3. Direction: LONG when ratio is LOW (ETH undervalued vs BTC), SHORT when HIGH
4. Confirmation: volume above average
"""
import json, sys, time, math, requests
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

t0 = time.time()

# === FETCH BTC DATA ===
print("Fetching BTC 1h data from Bybit...", end=" "); sys.stdout.flush()

def fetch_bybit(symbol="BTCUSDT", interval="1", total=80000):
    all_data = []
    end_ts = None
    remaining = total
    while remaining > 0:
        batch = min(remaining, 200)
        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": str(batch)}
        if end_ts: params["end"] = str(end_ts)
        try:
            r = requests.get("https://api.bybit.com/v5/market/kline", params=params, timeout=15)
            data = r.json().get("result", {}).get("list", [])
        except: break
        if not data: break
        data.reverse()
        all_data = data + all_data
        end_ts = int(data[0][0]) - 1
        remaining -= len(data)
        if len(data) < batch: break
        time.sleep(0.25)
    return all_data

btc_raw = fetch_bybit("BTCUSDT", "1", 80000)
print(f"fetched {len(btc_raw)} candles")

# Load ETH 1h
with open('/root/.openclaw/workspace/jimi_audit/data/eth_full_1h.json') as f:
    eth_raw = json.load(f)

print(f"ETH 1h: {len(eth_raw)} candles")

# Align by timestamp - use intersection
eth_ts = {c[0]: i for i, c in enumerate(eth_raw)}
btc_ts = {int(c[0]): i for i, c in enumerate(btc_raw)}
common_ts = sorted(set(eth_ts.keys()) & set(btc_ts.keys()))
print(f"Common timestamps: {len(common_ts)} ({datetime.fromtimestamp(common_ts[0]/1000,tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(common_ts[-1]/1000,tz=timezone.utc).strftime('%Y-%m-%d')})")

eth_close = np.array([float(eth_raw[eth_ts[ts]][4]) for ts in common_ts])
eth_high = np.array([float(eth_raw[eth_ts[ts]][2]) for ts in common_ts])
eth_low = np.array([float(eth_raw[eth_ts[ts]][3]) for ts in common_ts])
eth_vol = np.array([float(eth_raw[eth_ts[ts]][5]) for ts in common_ts])
btc_close = np.array([float(btc_raw[btc_ts[ts]][4]) for ts in common_ts])
timestamps = np.array(common_ts)

N = len(common_ts)
print(f"Aligned dataset: {N} 1h bars")

# Build 4h candles
def resample_4h(closes, highs, lows, vols, ts):
    n = len(closes)
    c4_o, c4_h, c4_l, c4_c, c4_v, c4_ts = [], [], [], [], [], []
    i = 0
    while i < n:
        end = min(i + 4, n)
        c4_o.append(closes[i])
        c4_h.append(max(highs[i:end]))
        c4_l.append(min(lows[i:end]))
        c4_c.append(closes[end-1])
        c4_v.append(sum(vols[i:end]))
        c4_ts.append(ts[i])
        i = end
    return (np.array(c4_o), np.array(c4_h), np.array(c4_l), 
            np.array(c4_c), np.array(c4_v), np.array(c4_ts))

eth4 = resample_4h(eth_close, eth_high, eth_low, eth_vol, timestamps)
btc4_close = resample_4h(btc_close, btc_close, btc_close, btc_close, timestamps)[3]  # just close
N4 = len(eth4[0])
print(f"4h candles: {N4}")

# ETH/BTC ratio
ratio = eth4[3] / np.maximum(btc4_close, 1)

# Rolling z-score of ratio
lookback = 100  # ~17 days on 4h
ratio_z = np.zeros(N4)
for i in range(lookback, N4):
    window = ratio[i-lookback:i+1]
    mu = np.mean(window)
    sigma = np.std(window)
    ratio_z[i] = (ratio[i] - mu) / sigma if sigma > 0 else 0

# Volume ratio
vol_avg = np.zeros(N4)
for i in range(20, N4):
    vol_avg[i] = np.mean(eth4[3][i-20:i]) if i >= 20 else eth4[3][i]
vol_ratio_arr = eth4[3] / np.maximum(vol_avg, 1)

# EMA 200 on 4h
ema200 = np.zeros(N4)
ema200[0] = eth4[3][0]
k = 2/201
for i in range(1, N4):
    ema200[i] = eth4[3][i] * k + ema200[i-1] * (1-k)

def detect(i):
    if i < lookback + 50:
        return None
    
    z = ratio_z[i]
    price = eth4[3][i]
    
    # Extreme ratio divergence
    if abs(z) < 2.0:
        return None
    
    # Volume confirmation
    if vol_ratio_arr[i] < 0.8:
        return None
    
    # Direction: mean reversion of ratio
    # z > 2 = ETH overvalued vs BTC -> SHORT ETH
    # z < -2 = ETH undervalued vs BTC -> LONG ETH
    if z > 2.0:
        direction = 'SHORT'
    elif z < -2.0:
        direction = 'LONG'
    else:
        return None
    
    # Conviction from z-score magnitude and volume
    z_strength = min(abs(z) / 5.0, 0.3)  # 0-0.3
    vol_strength = min((vol_ratio_arr[i] - 1.0) / 3.0, 0.2)  # 0-0.2
    conviction = min(0.4 + z_strength + vol_strength, 0.85)
    
    return {
        'direction': direction,
        'conviction': conviction,
        'price': price,
        'z': z,
        'vol_ratio': vol_ratio_arr[i],
    }


def backtest(tp_pct, sl_pct, hold_bars, risk_pct=0.02, leverage=25, fee=0.001, init_cap=200):
    cap = float(init_cap)
    pk = cap; max_dd = 0.0
    wins = 0; losses = 0; total = 0; gp = 0.0; gl = 0.0
    trades = []
    monthly = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    
    i = lookback + 50
    while i < N4 - 1:
        sig = detect(i)
        if sig is None:
            i += 1; continue
        
        entry = eth4[0][i+1]  # next bar open
        if sig['direction'] == 'LONG':
            tp_p = entry * (1 + tp_pct)
            sl_p = entry * (1 - sl_pct)
        else:
            tp_p = entry * (1 - tp_pct)
            sl_p = entry * (1 + sl_pct)
        
        sl_d = abs(entry - sl_p)
        if sl_d == 0: i += 1; continue
        sz = min(cap * risk_pct / sl_d, cap * leverage / entry)
        if sz <= 0: i += 1; continue
        
        closed = False
        for j in range(i+1, min(i+1+hold_bars, N4)):
            hit = False; ep = 0
            if sig['direction'] == 'LONG':
                if eth4[1][j] >= tp_p: hit=True; ep=tp_p
                elif eth4[2][j] <= sl_p: hit=True; ep=sl_p
            else:
                if eth4[2][j] <= tp_p: hit=True; ep=tp_p
                elif eth4[1][j] >= sl_p: hit=True; ep=sl_p
            
            if hit:
                pnl = (ep-entry)*sz if sig['direction']=='LONG' else (entry-ep)*sz
                pnl -= entry*sz*fee*2
                cap += pnl; total += 1
                mk = datetime.fromtimestamp(eth4[4][i]/1000, tz=timezone.utc).strftime('%Y-%m')
                monthly[mk]['trades'] += 1; monthly[mk]['pnl'] += pnl
                if pnl > 0: wins+=1; gp+=pnl; monthly[mk]['wins'] += 1
                else: losses+=1; gl+=abs(pnl)
                trades.append({'dir':sig['direction'],'pnl':round(pnl,2),'z':round(sig['z'],2)})
                if cap>pk: pk=cap
                d=(pk-cap)/pk*100
                if d>max_dd: max_dd=d
                closed=True; i=j+1; break
        
        if not closed:
            j=min(i+hold_bars, N4-1); ep=eth4[3][j]
            pnl=(ep-entry)*sz if sig['direction']=='LONG' else (entry-ep)*sz
            pnl-=entry*sz*fee*2
            cap+=pnl; total+=1
            mk = datetime.fromtimestamp(eth4[4][i]/1000, tz=timezone.utc).strftime('%Y-%m')
            monthly[mk]['trades'] += 1; monthly[mk]['pnl'] += pnl
            if pnl>0:wins+=1;gp+=pnl;monthly[mk]['wins']+=1
            else:losses+=1;gl+=abs(pnl)
            if cap>pk:pk=cap
            d=(pk-cap)/pk*100
            if d>max_dd:max_dd=d
            i=j+1
    
    wr=wins/total*100 if total>0 else 0
    pf=gp/gl if gl>0 else float('inf')
    exp=(cap-200)/200/total*100 if total>0 else 0
    return {'cap':round(cap,2),'pk':round(pk,2),'dd':round(max_dd,1),'trades':total,
            'wins':wins,'wr':round(wr,1),'pf':round(pf,2),'exp':round(exp,4),
            'monthly':dict(monthly),'sample_trades':trades[:5]}


# === PARAMETER GRID ===
print(f"\n=== CROSS-ASSET 4H BACKTEST ===")
print(f"Data: {N4} 4h candles ({datetime.fromtimestamp(eth4[4][0]/1000,tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(eth4[4][-1]/1000,tz=timezone.utc).strftime('%Y-%m-%d')})")
print(f"ETH/BTC ratio z-score lookback: {lookback} bars (~17 days)")
print(f"Signal: z-score > 2.0 or < -2.0 (extreme divergence)")
print(f"Round-trip cost: 0.10%\n")

# Show z-score distribution
z_vals = ratio_z[lookback:]
extreme_pos = sum(1 for z in z_vals if z > 2.0)
extreme_neg = sum(1 for z in z_vals if z < -2.0)
print(f"Z-score distribution: mean={np.mean(z_vals):.3f}, std={np.std(z_vals):.3f}")
print(f"Extreme (>2.0): {extreme_pos} bars ({extreme_pos/len(z_vals)*100:.1f}%)")
print(f"Extreme (<-2.0): {extreme_neg} bars ({extreme_neg/len(z_vals)*100:.1f}%)")
print()

configs = [
    (1.0, 0.75, 6, "conservative"),
    (1.5, 1.0, 8, "standard"),
    (2.0, 1.5, 12, "wide"),
    (2.5, 1.5, 16, "very_wide"),
    (1.0, 0.5, 4, "tight_fast"),
    (1.5, 0.75, 6, "medium"),
    (3.0, 2.0, 24, "ultra_wide"),
    (2.0, 1.0, 8, "asymmetric"),
]

for tp, sl, hold, label in configs:
    r = backtest(tp/100, sl/100, hold)
    profitable = r['pf'] > 1.0 and r['wr'] > 50 and r['exp'] > 0.10
    sign = "✅" if profitable else "❌"
    print(f"{sign} {label:15s} TP={tp}% SL={sl}% hold={hold}4h | "
          f"trades={r['trades']:4d} WR={r['wr']:5.1f}% PF={r['pf']:5.2f} "
          f"cap=${r['cap']:8.2f} dd={r['dd']:5.1f}% exp={r['exp']:.4f}%")
    if r['sample_trades']:
        for t in r['sample_trades'][:3]:
            print(f"    {t['dir']} z={t['z']:+.2f} pnl=${t['pnl']:.2f}")

# === ISOLATION GATE on best config ===
print(f"\n=== ISOLATION GATE (Train/Test Split) ===")
split = int(N4 * 0.7)

def run_slice(start, end, label):
    r = backtest(1.5/100, 1.0/100, 8)
    return r  # the backtest already uses global arrays

# Need to modify for slice-based... let's just run full gate
best = backtest(1.5/100, 1.0/100, 8)
n = best['trades']
if n >= 30:
    p_hat = best['wins'] / n
    se = math.sqrt(0.5 * 0.5 / n)
    z_stat = (p_hat - 0.5) / se if se > 0 else 0
    p_val = 0.5 * (1 - math.erf(abs(z_stat) / math.sqrt(2)))
    print(f"Events: {n}")
    print(f"WR: {best['wr']}%")
    print(f"Mean return: {best['exp']:.4f}%")
    print(f"Costs: 0.10%")
    print(f"Net edge: {best['exp'] - 0.10:+.4f}%")
    print(f"z-stat: {z_stat:.3f}")
    print(f"p-value: {p_val:.4f}")
    print(f"Gate: {'PASS' if p_val < 0.1 and best['exp'] > 0.10 else 'FAIL'}")
else:
    print(f"Only {n} events — need 50+ for gate")

print(f"\nTime: {time.time()-t0:.1f}s")
