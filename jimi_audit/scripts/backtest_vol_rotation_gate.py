#!/usr/bin/env python3
"""
VOL ROTATION 4H — ISOLATION GATE (Train/Test Split)
Following BACKTEST_FRAMEWORK.md protocol:
- Train on first 70% of data
- Test on last 30% (out-of-sample)
- Report both separately
"""
import json, sys, time, math
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

t0 = time.time()

with open('/root/.openclaw/workspace/jimi_audit/data/eth_full_1h.json') as f:
    raw = json.load(f)

N = len(raw)
o1h = [float(c[1]) for c in raw]
h1h = [float(c[2]) for c in raw]
l1h = [float(c[3]) for c in raw]
c1h = [float(c[4]) for c in raw]
v1h = [float(c[5]) for c in raw]
ts1h = [c[0] for c in raw]

# Build 4h candles
c4h_o, c4h_h, c4h_l, c4h_c, c4h_v, c4h_ts = [], [], [], [], [], []
i = 0
while i < N:
    end = min(i + 4, N)
    c4h_o.append(o1h[i])
    c4h_h.append(max(h1h[i:end]))
    c4h_l.append(min(l1h[i:end]))
    c4h_c.append(c1h[end - 1])
    c4h_v.append(sum(v1h[i:end]))
    c4h_ts.append(ts1h[i])
    i = end

N4 = len(c4h_o)

# Indicators
atr_period = 14
atr = [0.0] * N4
for i in range(1, N4):
    tr = max(c4h_h[i] - c4h_l[i], abs(c4h_h[i] - c4h_c[i-1]), abs(c4h_l[i] - c4h_c[i-1]))
    atr[i] = tr if i < atr_period else (atr[i-1] * (atr_period - 1) + tr) / atr_period

atr_pct = [0.5] * N4
for i in range(100, N4):
    window = atr[i-100:i+1]
    atr_pct[i] = sum(1 for x in window if x <= atr[i]) / len(window)

vol_ratio = [1.0] * N4
for i in range(20, N4):
    avg = np.mean(c4h_v[i-20:i])
    vol_ratio[i] = c4h_v[i] / avg if avg > 0 else 1.0

ema200 = [0.0] * N4; ema200[0] = c4h_c[0]; k200 = 2/201
for i in range(1, N4): ema200[i] = c4h_c[i] * k200 + ema200[i-1] * (1-k200)

ema50 = [0.0] * N4; ema50[0] = c4h_c[0]; k50 = 2/51
for i in range(1, N4): ema50[i] = c4h_c[i] * k50 + ema50[i-1] * (1-k50)

rsi = [50.0] * N4
for i in range(15, N4):
    gains, losses = [], []
    for j in range(i-13, i+1):
        d = c4h_c[j] - c4h_c[j-1]
        gains.append(max(d, 0)); losses.append(max(-d, 0))
    ag = np.mean(gains); al = np.mean(losses)
    rsi[i] = 100 - 100/(1 + ag/al) if al > 0 else 100

def detect(i, start_bar):
    if i < max(start_bar + 200, 300):
        return None
    recent_compressed = any(atr_pct[i-j] < 0.30 for j in range(1, 11))
    expanding = atr_pct[i] > 0.50
    vol_confirm = vol_ratio[i] > 1.2
    if not (recent_compressed and expanding and vol_confirm):
        return None
    price = c4h_c[i]
    ema_up = price > ema200[i] and ema50[i] > ema200[i]
    ema_down = price < ema200[i] and ema50[i] < ema200[i]
    rsi_bull = 40 < rsi[i] < 70
    rsi_bear = 30 < rsi[i] < 60
    recent_move = (c4h_c[i] - c4h_c[i-3]) / c4h_c[i-3]
    if ema_up and rsi_bull and recent_move > 0: d = 'LONG'
    elif ema_down and rsi_bear and recent_move < 0: d = 'SHORT'
    else: return None
    vs = min((vol_ratio[i]-1.0)/2.0, 0.3)
    ats = min(atr_pct[i]/2.0, 0.25)
    ts = min(abs(recent_move)*10, 0.25)
    return {'dir': d, 'conv': min(0.3+vs+ats+ts, 0.85), 'price': price}

def run_slice(start_bar, end_bar, label):
    cap = 200.0; pk = cap; dd = 0.0
    wins = 0; losses = 0; total = 0; gp = 0.0; gl = 0.0
    trades = []
    i = start_bar
    while i < end_bar - 1:
        sig = detect(i, start_bar)
        if sig is None:
            i += 1; continue
        entry = c4h_o[i+1]
        tp_p = entry * 1.01 if sig['dir'] == 'LONG' else entry * 0.99
        sl_p = entry * 0.995 if sig['dir'] == 'LONG' else entry * 1.005
        sl_d = abs(entry - sl_p)
        if sl_d == 0: i += 1; continue
        sz = min(cap * 0.02 / sl_d, cap * 25 / entry)
        if sz <= 0: i += 1; continue
        closed = False
        for j in range(i+1, min(i+5, end_bar)):
            hit = False; ep = 0
            if sig['dir'] == 'LONG':
                if c4h_h[j] >= tp_p: hit=True; ep=tp_p
                elif c4h_l[j] <= sl_p: hit=True; ep=sl_p
            else:
                if c4h_l[j] <= tp_p: hit=True; ep=tp_p
                elif c4h_h[j] >= sl_p: hit=True; ep=sl_p
            if hit:
                pnl = (ep-entry)*sz if sig['dir']=='LONG' else (entry-ep)*sz
                pnl -= entry*sz*0.001*2
                cap += pnl; total += 1
                if pnl > 0: wins+=1; gp+=pnl
                else: losses+=1; gl+=abs(pnl)
                trades.append({'dir':sig['dir'],'pnl':round(pnl,2)})
                if cap>pk: pk=cap
                d=(pk-cap)/pk*100
                if d>dd: dd=d
                closed=True; i=j+1; break
        if not closed:
            j=min(i+4, end_bar-1); ep=c4h_c[j]
            pnl=(ep-entry)*sz if sig['dir']=='LONG' else (entry-ep)*sz
            pnl-=entry*sz*0.001*2
            cap+=pnl; total+=1
            if pnl>0:wins+=1;gp+=pnl
            else:losses+=1;gl+=abs(pnl)
            if cap>pk:pk=cap
            d=(pk-cap)/pk*100
            if d>dd:dd=d
            i=j+1
    wr=wins/total*100 if total>0 else 0
    pf=gp/gl if gl>0 else float('inf')
    mean_ret=(cap-200)/200/total*100 if total>0 else 0
    return {'label':label,'cap':round(cap,2),'dd':round(dd,1),'trades':total,
            'wins':wins,'wr':round(wr,1),'pf':round(pf,2),'mean_ret':round(mean_ret,4)}

# Split: 70% train, 30% test
split = int(N4 * 0.7)
print(f"=== VOL ROTATION 4H — ISOLATION GATE ===")
print(f"Total: {N4} 4h bars")
print(f"Train: bars 0-{split} ({datetime.fromtimestamp(c4h_ts[0]/1000,tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(c4h_ts[split]/1000,tz=timezone.utc).strftime('%Y-%m-%d')})")
print(f"Test:  bars {split}-{N4} ({datetime.fromtimestamp(c4h_ts[split]/1000,tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(c4h_ts[-1]/1000,tz=timezone.utc).strftime('%Y-%m-%d')})")
print(f"Round-trip cost: 0.10%\n")

train = run_slice(200, split, "IN-SAMPLE (70%)")
test = run_slice(split, N4, "OUT-OF-SAMPLE (30%)")
full = run_slice(200, N4, "FULL")

for r in [train, test, full]:
    above_cost = r['mean_ret'] > 0.10
    sign = "✅" if above_cost and r['pf'] > 1.0 else "❌"
    print(f"{sign} {r['label']:20s} | trades={r['trades']:4d} WR={r['wr']:5.1f}% PF={r['pf']:5.2f} "
          f"cap=${r['cap']:8.2f} dd={r['dd']:5.1f}% mean_ret={r['mean_ret']:.4f}%")
    print(f"   Edge over costs: {r['mean_ret'] - 0.10:+.4f}%")

# Statistical test on full sample
print(f"\n=== STATISTICAL TEST ===")
n = full['trades']
if n >= 50:
    p_hat = full['wins'] / n
    se = math.sqrt(0.5 * 0.5 / n)
    z = (p_hat - 0.5) / se if se > 0 else 0
    p_val = 0.5 * (1 - math.erf(abs(z) / math.sqrt(2)))
    print(f"Events: {n}")
    print(f"WR: {full['wr']}% (null: 50%)")
    print(f"z-stat: {z:.3f}")
    print(f"p-value: {p_val:.4f}")
    print(f"Mean return: {full['mean_ret']:.4f}%")
    print(f"Costs: 0.10%")
    print(f"Net edge: {full['mean_ret'] - 0.10:+.4f}%")
    
    # OOS check
    oos_pf_drift = abs(train['pf'] - test['pf'])
    print(f"\n=== OVERFIT CHECK ===")
    print(f"IS PF: {train['pf']}")
    print(f"OOS PF: {test['pf']}")
    print(f"PF drift: {oos_pf_drift:.2f} (threshold: 0.5)")
    print(f"Overfit: {'YES' if oos_pf_drift > 0.5 else 'NO'}")
    
    # Gate verdict
    gate_pass = (p_val < 0.1 and full['mean_ret'] > 0.10 and 
                 test['pf'] > 0.5 and oos_pf_drift <= 0.5)
    print(f"\n=== GATE VERDICT: {'PASS ✅' if gate_pass else 'FAIL ❌'} ===")
    reasons = []
    if p_val >= 0.1: reasons.append(f"p={p_val:.4f} >= 0.1")
    if full['mean_ret'] <= 0.10: reasons.append(f"mean_ret={full['mean_ret']:.4f}% <= costs")
    if test['pf'] <= 0.5: reasons.append(f"OOS PF={test['pf']} <= 0.5")
    if oos_pf_drift > 0.5: reasons.append(f"PF drift={oos_pf_drift:.2f} > 0.5")
    if reasons:
        for r in reasons: print(f"  - {r}")
else:
    print(f"Only {n} events — need 50+ for gate")

print(f"\nTime: {time.time()-t0:.1f}s")
