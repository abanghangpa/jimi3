#!/usr/bin/env python3
"""Failed Breakout v7 — Optimized. Pre-computes everything."""
import csv, json, sys, os
from datetime import datetime, timezone
import numpy as np

BASE = '/root/.openclaw/workspace/jimi_audit'
DATA_FILE = f'{BASE}/eth_15m_6m.csv'
FEE = 0.0002; SLIP = 0.001

print('Loading...', flush=True)
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
opens = np.array([b['o'] for b in bars])
volumes = np.array([b['v'] for b in bars])
taker_buys = np.array([b['tb'] for b in bars])

atr = np.zeros(N)
for i in range(1, N):
    tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
    atr[i] = tr if i < 14 else (atr[i-1]*13 + tr)/14

avg_vol = np.zeros(N)
for i in range(20, N):
    avg_vol[i] = np.mean(volumes[i-20:i])

ema200 = np.zeros(N)
ema200[199] = np.mean(closes[:200])
m_ema = 2 / 201
for i in range(200, N):
    ema200[i] = closes[i] * m_ema + ema200[i-1] * (1 - m_ema)

cvd_delta = 2 * taker_buys - volumes
cvd_cum = np.cumsum(cvd_delta)

print(f'Loaded {N} bars ({bars[0]["ts"].strftime("%Y-%m-%d")} to {bars[-1]["ts"].strftime("%Y-%m-%d")})', flush=True)

# === Structural map ===
print('Building structural map...', flush=True)
swing_highs = []
swing_lows = []
for i in range(3, N-3):
    if highs[i] > max(highs[i-3:i]) and highs[i] > max(highs[i+1:i+4]):
        swing_highs.append((i, highs[i]))
    if lows[i] < min(lows[i-3:i]) and lows[i] < min(lows[i+1:i+4]):
        swing_lows.append((i, lows[i]))
print(f'  Swings: {len(swing_highs)}H {len(swing_lows)}L')

EQ_TOL = 0.002
def find_eq(swings):
    clusters = []; used = set()
    for i, (idx, price) in enumerate(swings):
        if i in used: continue
        cluster = [(idx, price)]
        for j in range(i+1, len(swings)):
            if j in used: continue
            if abs(price - swings[j][1]) / price < EQ_TOL:
                cluster.append((j, swings[j][1])); used.add(j)
        if len(cluster) >= 2:
            clusters.append({'price': np.mean([p for _,p in cluster]), 'count': len(cluster)})
        used.add(i)
    return clusters

eq_highs = find_eq(swing_highs)
eq_lows = find_eq(swing_lows)
print(f'  Equal: {len(eq_highs)}H {len(eq_lows)}L')

# Pre-compute session/day H/L
sess_h = np.full(N, np.nan); sess_l = np.full(N, np.nan)
day_h = np.full(N, np.nan); day_l = np.full(N, np.nan)
for i in range(96, N):
    sh = max(0, i - 32)
    sess_h[i] = np.max(highs[sh:i+1]); sess_l[i] = np.min(lows[sh:i+1])
    day_h[i] = np.max(highs[i-96:i+1]); day_l[i] = np.min(lows[i-96:i+1])

# Pre-compute CVD divergence arrays
print('Pre-computing CVD...', flush=True)
cvd_div_h = np.zeros(N, dtype=bool); cvd_div_l = np.zeros(N, dtype=bool)
cvd_str_h = np.zeros(N); cvd_str_l = np.zeros(N)
for i in range(20, N):
    if highs[i] > np.max(highs[i-20:i]):
        cm = np.max(cvd_cum[i-20:i])
        if cvd_cum[i] < cm:
            cvd_div_h[i] = True
            cvd_str_h[i] = (cm - cvd_cum[i]) / (abs(cm) + 1e-8)
    if lows[i] < np.min(lows[i-20:i]):
        cm = np.min(cvd_cum[i-20:i])
        if cvd_cum[i] > cm:
            cvd_div_l[i] = True
            cvd_str_l[i] = (cvd_cum[i] - cm) / (abs(cm) + 1e-8)

def lvl_score(price, bar, d):
    s = 0.0
    eq = eq_highs if d == 'h' else eq_lows
    for e in eq:
        if abs(e['price'] - price) / price < EQ_TOL:
            s += min(e['count']/5, 1.0) * 0.3; break
    if d == 'h':
        if not np.isnan(sess_h[bar]) and atr[bar] > 0 and abs(sess_h[bar]-price)/atr[bar] < 0.5: s += 0.2
        if not np.isnan(day_h[bar]) and atr[bar] > 0 and abs(day_h[bar]-price)/atr[bar] < 0.5: s += 0.15
    else:
        if not np.isnan(sess_l[bar]) and atr[bar] > 0 and abs(sess_l[bar]-price)/atr[bar] < 0.5: s += 0.2
        if not np.isnan(day_l[bar]) and atr[bar] > 0 and abs(day_l[bar]-price)/atr[bar] < 0.5: s += 0.15
    return min(s, 1.0)

# === Signal detection ===
print('Detecting signals...', flush=True)
COOLDOWN = 8; MAX_RECLAIM = 5; LOOKBACK = 200
signals = []

for i in range(96, N - MAX_RECLAIM - 16):
    if atr[i] == 0 or avg_vol[i] == 0: continue
    hour = bars[i]['ts'].hour
    session = 'asia' if 0 <= hour < 7 else ('london' if 7 <= hour < 13 else ('ny' if 13 <= hour < 21 else 'late'))

    best_sig = None; best_sc = 0

    # SHORT: sweep above recent swing iighs (top 5 nearest)
    rsh = [(idx, p) for idx, p in swing_highs if i-LOOKBACK < idx < i-3 and p > closes[i]]
    rsh.sort(key=lambda x: abs(x[1] - closes[i]))
    for li, lp in rsh[:5]:
        if highs[i] <= lp: continue
        sd = (highs[i] - lp) / atr[i] if atr[i] > 0 else 0
        if sd < 0.1: continue
        rb = None
        for k in range(MAX_RECLAIM+1):
            ci = i+k
            if ci >= N: break
            if closes[ci] < lp: rb = ci; break
        if rb is None: continue
        sp = rb - i
        sc = min(sd/1.5,1.0)*0.25 + max(0,(1-sp/MAX_RECLAIM))*0.20
        if cvd_div_h[i]: sc += min(cvd_str_h[i],1.0)*0.25
        sc += lvl_score(lp, i, 'h')*0.15
        vr = volumes[i]/avg_vol[i] if avg_vol[i] > 0 else 0
        sc += min(vr/2.0,1.0)*0.15
        if sc > best_sc:
            best_sc = sc
            best_sig = {'dir':'SHORT','entry':closes[rb],'sl':highs[i]+atr[i]*0.3,
                'sweep_bar':i,'reclaim_bar':rb,'reclaim_speed':sp,
                'sweep_depth':round(sd,3),'cvd_div':bool(cvd_div_h[i]),
                'cvd_str':round(float(cvd_str_h[i]),3),'vol_ratio':round(vr,2),
                'score':round(sc,3),'bar':i,'ts':str(bars[i]['ts']),'hour':hour,'session':session}

    # LONG: sweep below recent swing lows
    rsl = [(idx, p) for idx, p in swing_lows if i-LOOKBACK < idx < i-3 and p < closes[i]]
    rsl.sort(key=lambda x: abs(x[1] - closes[i]))
    for li, lp in rsl[:5]:
        if lows[i] >= lp: continue
        sd = (lp - lows[i]) / atr[i] if atr[i] > 0 else 0
        if sd < 0.1: continue
        rb = None
        for k in range(MAX_RECLAIM+1):
            ci = i+k
            if ci >= N: break
            if closes[ci] > lp: rb = ci; break
        if rb is None: continue
        sp = rb - i
        sc = min(sd/1.5,1.0)*0.25 + max(0,(1-sp/MAX_RECLAIM))*0.20
        if cvd_div_l[i]: sc += min(cvd_str_l[i],1.0)*0.25
        sc += lvl_score(lp, i, 'l')*0.15
        vr = volumes[i]/avg_vol[i] if avg_vol[i] > 0 else 0
        sc += min(vr/2.0,1.0)*0.15
        if sc > best_sc:
            best_sc = sc
            best_sig = {'dir':'LONG','entry':closes[rb],'sl':lows[i]-atr[i]*0.3,
                'sweep_bar':i,'reclaim_bar':rb,'reclaim_speed':sp,
                'sweep_depth':round(sd,3),'cvd_div':bool(cvd_div_l[i]),
                'cvd_str':round(float(cvd_str_l[i]),3),'vol_ratio':round(vr,2),
                'score':round(sc,3),'bar':i,'ts':str(bars[i]['ts']),'hour':hour,'session':session}

    if best_sig and best_sc >= 0.25:
        if not signals or (i - signals[-1]['bar']) >= COOLDOWN:
            signals.append(best_sig)

print(f'Found {len(signals)} signals', flush=True)
if signals:
    dirs = {}
    for s in signals: dirs[s['dir']] = dirs.get(s['dir'],0)+1
    print(f'  Directions: {dirs}')
    sc = [s['score'] for s in signals]
    print(f'  Scores: min={min(sc):.3f} max={max(sc):.3f} avg={np.mean(sc):.3f}')
    sp = {}
    for s in signals: sp[s['reclaim_speed']] = sp.get(s['reclaim_speed'],0)+1
    print(f'  Reclaim speeds: {sorted(sp.items())}')
    cc = sum(1 for s in signals if s['cvd_div'])
    print(f'  CVD divergence: {cc}/{len(signals)} ({cc/len(signals)*100:.1f}%)')
    se = {}
    for s in signals: se[s['session']] = se.get(s['session'],0)+1
    print(f'  Sessions: {se}')

si = int(len(signals)*0.7)
train = signals[:si]; test = signals[si:]
print(f'Hold-out: {len(train)} train, {len(test)} test')

def sim(sigs, tp, sl, hold, ms, trend, sess, spd, cvd):
    trades = []
    for s in sigs:
        if s['score'] < ms: continue
        if sess and s['session'] == 'asia': continue
        if spd is not None and s['reclaim_speed'] > spd: continue
        if cvd and not s['cvd_div']: continue
        if trend:
            if s['dir']=='LONG' and closes[s['bar']]<ema200[s['bar']]: continue
            if s['dir']=='SHORT' and closes[s['bar']]>ema200[s['bar']]: continue
        i = s['reclaim_bar']; entry = s['entry']
        if s['dir']=='LONG':
            tp_p=entry*(1+tp/100); sl_p=max(entry*(1-sl/100),s['sl'])
        else:
            tp_p=entry*(1-tp/100); sl_p=min(entry*(1+sl/100),s['sl'])
        oc='TIMEOUT'; ex=entry
        for j in range(i+1,min(i+hold,N)):
            if s['dir']=='LONG':
                if highs[j]>=tp_p: oc='WIN'; ex=tp_p; break
                if lows[j]<=sl_p: oc='LOSS'; ex=sl_p; break
            else:
                if lows[j]<=tp_p: oc='WIN'; ex=tp_p; break
                if highs[j]>=sl_p: oc='LOSS'; ex=sl_p; break
        if oc=='TIMEOUT': ex=closes[min(i+hold-1,N-1)]
        pnl=((ex-entry)/entry*100) if s['dir']=='LONG' else ((entry-ex)/entry*100)
        pnl-=FEE*2*100+SLIP*100
        trades.append({'pnl':round(pnl,4),'oc':oc,'ts':s['ts'],'session':s['session']})
    if len(trades)<3: return None
    n=len(trades); w=sum(1 for t in trades if t['oc']=='WIN')
    wins=[t['pnl'] for t in trades if t['oc']=='WIN']
    losses=[t['pnl'] for t in trades if t['oc'] in('LOSS','TIMEOUT') and t['pnl']<0]
    wp=sum(wins) if wins else 0; lp=abs(sum(losses)) if losses else 0
    pf=wp/lp if lp>0 else 999; pnl=sum(t['pnl'] for t in trades)
    monthly={}
    for t in trades:
        m=t['ts'][:7]
        if m not in monthly: monthly[m]=0
        monthly[m]=round(monthly[m]+t['pnl'],2)
    eq=100;pk=100;mdd=0
    for t in trades:
        eq+=t['pnl']
        if eq>pk:pk=eq
        dd=(pk-eq)/pk*100
        if dd>mdd:mdd=dd
    return {'trades':n,'wins':w,'wr':round(w/n*100,1),'pf':round(pf,2),
            'pnl':round(pnl,1),'dd':round(mdd,1),'bad_m':sum(1 for v in monthly.values() if v<0),
            'monthly':{k:v for k,v in sorted(monthly.items())}}

print('Sweeping...', flush=True)
results=[]; tested=0
for tp in [1.0,1.5,2.0,2.5,3.0,4.0]:
    for sl in [0.3,0.5,0.75,1.0,1.5]:
        if sl>=tp: continue
        for hold in [4,8,12,16,24,32]:
            for ms in [0.25,0.35,0.45,0.55,0.65]:
                for trend in [False,True]:
                    for sess in [False,True]:
                        for sp in [None,1,2,3]:
                            for cvd in [False,True]:
                                r=sim(signals,tp,sl,hold,ms,trend,sess,sp,cvd)
                                tested+=1
                                if r: results.append({**r,'tp':tp,'sl':sl,'hold':hold,'score':ms,'trend':trend,'sess':sess,'speed':sp,'cvd':cvd})

print(f'Tested {tested} configs', flush=True)
results.sort(key=lambda x: x['pf']*(x['wr']/100)*(x['trades']**0.3), reverse=True)

ent=[r for r in results if r['wr']>=65 and r['pf']>=2.0 and r['dd']<25 and r['bad_m']<=3 and r['trades']>=10]
good=[r for r in results if r['wr']>=55 and r['pf']>=1.5 and r['trades']>=8]
ok=[r for r in results if r['wr']>=50 and r['pf']>=1.2 and r['trades']>=5]
profit=[r for r in results if r['pf']>1.0 and r['trades']>=5]

print(f'\nEnterprise: {len(ent)}  Good: {len(good)}  OK: {len(ok)}  Profitable: {len(profit)}')

for label,sub in [('ENTERPRISE',ent),('GOOD',good),('OK',ok),('PROFITABLE',profit)]:
    if not sub: continue
    print(f'\n{"="*130}')
    print(f'{label} ({len(sub)} configs)')
    print(f'{"="*130}')
    print(f'  {"#":>2} {"TP":>4} {"SL":>4} {"RR":>4} {"H":>3} {"Sc":>4} {"Tr":>2} {"Se":>2} {"Sp":>3} {"CV":>2} | {"N":>4} {"W":>3} {"WR":>5} {"PF":>5} {"PnL":>7} {"DD":>5} {"BM":>3}')
    print('  '+'-'*120)
    for i,r in enumerate(sub[:20]):
        rr=round(r['tp']/r['sl'],1)
        print(f'  {i+1:>2} {r["tp"]:>4.1f} {r["sl"]:>4.2f} {rr:>4.1f} {r["hold"]:>3} {r["score"]:>4.2f} {"T" if r["trend"] else "-":>2} {"S" if r["sess"] else "-":>2} {str(r["speed"]) if r["speed"] else "-":>3} {"C" if r["cvd"] else "-":>2} | {r["trades"]:>4} {r["wins"]:>3} {r["wr"]:>5.1f} {r["pf"]:>5.2f} {r["pnl"]:>+7.1f} {r["dd"]:>5.1f} {r["bad_m"]:>3}')

best=ent or good or ok or profit
if best:
    b=best[0]
    print(f'\nBEST: TP={b["tp"]}% SL={b["sl"]}% H={b["hold"]}h Sc>={b["score"]} Trend={"EMA200" if b["trend"] else "-"} Sess={"skip-asia" if b["sess"] else "all"} Speed={b["speed"]} CVD={b["cvd"]}')
    print(f'  {b["trades"]}T {b["wr"]}%WR {b["pf"]}PF PnL={b["pnl"]}% DD={b["dd"]}%')
    for m,v in sorted(b['monthly'].items()): print(f'    {m}: {v:+.1f}%')
    print(f'\nHOLD-OUT:')
    rt=sim(train,b['tp'],b['sl'],b['hold'],b['score'],b['trend'],b['sess'],b['speed'],b['cvd'])
    rv=sim(test,b['tp'],b['sl'],b['hold'],b['score'],b['trend'],b['sess'],b['speed'],b['cvd'])
    if rt: print(f'  Train: {rt["trades"]}T {rt["wr"]}%WR {rt["pf"]}PF PnL={rt["pnl"]}%')
    else: print('  Train: no trades')
    if rv: print(f'  Test:  {rv["trades"]}T {rv["wr"]}%WR {rv["pf"]}PF PnL={rv["pnl"]}%')
    else: print('  Test: no trades')
    if rt and rv:
        pd=abs(rt['pf']-rv['pf'])
        print(f'  PF drift: {pd:.2f} ({"STABLE" if pd<0.5 else "UNSTABLE"})')
else:
    print('\nNo configs met any tier. Top 5:')
    for i,r in enumerate(results[:5]):
        print(f'  {i+1}. TP={r["tp"]} SL={r["sl"]} H={r["hold"]} | {r["trades"]}T {r["wr"]}%WR {r["pf"]}PF PnL={r["pnl"]}%')

def conv(o):
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return float(o)
    if isinstance(o,np.ndarray): return o.tolist()
    return o
out={'version':'v7','signals':len(signals),'tested':tested,
     'enterprise':len(ent),'good':len(good),'ok':len(ok),'profitable':len(profit),
     'best':best[0] if best else None,'top5':results[:5]}
with open(f'{BASE}/reports/fb_v7_backtest.json','w') as f:
    json.dump(out,f,indent=2,default=conv)
print('\nDone')
