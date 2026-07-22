#!/usr/bin/env python3
"""
Backtest tune: orderbook_imbalance + momentum_v3
Constraint: SL < TP
"""
import csv, json, sys, os
from datetime import datetime, timezone, timedelta
import numpy as np

BASE = '/root/.openclaw/workspace/jimi_audit'
DATA_FILE = f'{BASE}/eth_15m_6m.csv'
FEE = 0.0002
SLIP = 0.001

print("Loading data...", flush=True)

# Load ETH bars
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

# ATR
atr = np.zeros(N)
for i in range(1, N):
    tr = max(bars[i]['h']-bars[i]['l'], abs(bars[i]['h']-bars[i-1]['c']), abs(bars[i]['l']-bars[i-1]['c']))
    atr[i] = tr if i < 14 else (atr[i-1]*13 + tr)/14

# Avg volume
avg_vol = np.zeros(N)
for i in range(20, N):
    avg_vol[i] = np.mean(volumes[i-20:i])

# EMA200
ema200 = np.zeros(N)
ema200[199] = np.mean(closes[:200])
m = 2 / 201
for i in range(200, N):
    ema200[i] = closes[i] * m + ema200[i-1] * (1 - m)

print(f"Loaded {N} bars", flush=True)

# Derivatives for momentum_v3
deriv = {}
try:
    with open(f'{BASE}/data/derivatives_history/derivatives_collected.csv') as f:
        for row in csv.DictReader(f):
            dt = datetime.fromisoformat(row['timestamp'])
            dt_floor = dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)
            k = dt_floor.strftime('%Y-%m-%d %H:%M:%S')
            deriv[k] = {'ls_ratio': float(row['ls_ratio']), 'funding_rate': float(row['funding_rate']), 'oi': float(row.get('oi', 0))}
    oi_keys = sorted(deriv.keys())
    for i in range(4, len(oi_keys)):
        curr = deriv[oi_keys[i]]['oi']
        prev = deriv[oi_keys[i-4]]['oi']
        deriv[oi_keys[i]]['oi_roc_1h'] = (curr - prev) / prev if prev > 0 else 0
    print(f"Loaded {len(deriv)} derivative points", flush=True)
except:
    print("No derivatives data, momentum_v3 will be limited", flush=True)

# Event signals for momentum_v3
event_signals = {}
try:
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
    print("Event signals:", flush=True)
    for s, sigs in sorted(event_signals.items(), key=lambda x: -len(x[1])):
        print(f"  {s}: {len(sigs)}", flush=True)
except:
    print("No event signals", flush=True)

eth_map = {}
for i, b in enumerate(bars):
    eth_map[b['ts'].strftime('%Y-%m-%d %H:%M:%S')] = i

def find_deriv(ts, max_min=30):
    if ts in deriv: return deriv[ts]
    dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    for off in range(1, max_min + 1):
        for d in [dt - timedelta(minutes=off), dt + timedelta(minutes=off)]:
            k = d.strftime('%Y-%m-%d %H:%M:%S')
            if k in deriv: return deriv[k]
    return None

def detect_exhaustion(idx, min_signals=2):
    if idx < 80: return None
    mom_5 = (closes[idx] - closes[idx - 5]) / closes[idx - 5]
    mom_10 = (closes[idx] - closes[idx - 10]) / closes[idx - 10]
    accel = mom_5 - mom_10 / 2
    decel = (mom_5 > 0 and accel < 0) or (mom_5 < 0 and accel > 0)
    vol_recent = np.mean(volumes[idx - 5:idx])
    vol_prior = np.mean(volumes[idx - 15:idx - 5])
    vol_change = (vol_recent - vol_prior) / vol_prior if vol_prior > 0 else 0
    vol_div = abs(mom_5) > 0.005 and vol_change < -0.1
    moves = [abs(closes[j + 5] - closes[j]) / closes[j] for j in range(idx - 80, idx - 5)]
    current_move = abs(closes[idx] - closes[idx - 5]) / closes[idx - 5]
    percentile = sum(1 for m in moves if m < current_move) / len(moves) * 100
    extreme = percentile > 85
    ts = eth_keys[idx] if idx < len(eth_keys) else ''
    dv = find_deriv(ts)
    oi_roc = dv.get('oi_roc_1h', 0) if dv else 0
    oi_div = abs(mom_5) > 0.005 and oi_roc < -0.02
    count = sum([decel, vol_div, extreme, oi_div])
    if count < min_signals: return None
    direction = 'SHORT' if mom_5 > 0 else 'LONG' if mom_5 < 0 else None
    if not direction: return None
    return {'direction': direction, 'mom_5': mom_5, 'count': count}

# ============================================================
# PART 1: orderbook_imbalance — TP/SL sweep with SL < TP
# ============================================================
print("\n" + "=" * 80, flush=True)
print("PART 1: orderbook_imbalance — TP/SL sweep (SL < TP)", flush=True)
print("=" * 80, flush=True)

def backtest_obi(tp, sl, hold, taker_t, vol_min, trend, session):
    trades = []
    pos = None
    eq = 200.0
    peak = 200.0
    max_dd = 0

    for i in range(210, N):
        if pos:
            b = bars[i]
            held = (b['ts'] - pos['ot']).total_seconds() / 3600
            exit_p = None
            outcome = None
            if pos['d'] == 'L':
                if b['h'] >= pos['tp']: exit_p, outcome = pos['tp'], 'W'
                elif b['l'] <= pos['sl']: exit_p, outcome = pos['sl'], 'L'
            else:
                if b['l'] <= pos['tp']: exit_p, outcome = pos['tp'], 'W'
                elif b['h'] >= pos['sl']: exit_p, outcome = pos['sl'], 'L'
            if exit_p is None and held >= hold:
                exit_p, outcome = b['c'], 'T'
            if exit_p:
                pnl = ((exit_p-pos['e'])/pos['e']) if pos['d']=='L' else ((pos['e']-exit_p)/pos['e'])
                pnl -= FEE*2
                eq += eq * pnl
                trades.append({'o': outcome})
                if eq > peak: peak = eq
                dd = (peak-eq)/peak*100
                if dd > max_dd: max_dd = dd
                pos = None

        if pos is None:
            ts = bars[i]['ts']
            if session:
                h = ts.hour
                if session == 'ln' and not (8 <= h < 21): continue
                elif session == 'ol' and not (13 <= h < 16): continue

            taker = bars[i]['tb'] / max(bars[i]['v'], 0.01)
            vol_r = bars[i]['v'] / max(avg_vol[i], 1) if avg_vol[i] > 0 else 0
            if vol_r < vol_min: continue

            d = None
            if taker >= taker_t:
                d = 'L'
                if trend and closes[i] < ema200[i]: continue
            elif taker <= (1 - taker_t):
                d = 'S'
                if trend and closes[i] > ema200[i]: continue

            if d:
                e = closes[i] * (1 + SLIP if d == 'L' else 1 - SLIP)
                sl_d = e * sl / 100
                tp_d = e * tp / 100
                if d == 'L':
                    sl_p, tp_p = e - sl_d, e + tp_d
                else:
                    sl_p, tp_p = e + sl_d, e - tp_d
                pos = {'d': d, 'e': e, 'sl': sl_p, 'tp': tp_p, 'ot': ts}

    if len(trades) < 3: return None
    w = sum(1 for t in trades if t['o']=='W')
    l = sum(1 for t in trades if t['o']=='L')
    t_count = sum(1 for t in trades if t['o']=='T')
    wr = w/len(trades)*100
    pf = w * tp / max(l * sl, 0.01) if l > 0 else 999
    return {'trades': len(trades), 'wins': w, 'losses': l, 'timeouts': t_count,
            'wr': round(wr,2), 'pf': round(pf,2), 'eq': round(eq,2), 'dd': round(max_dd,2),
            'pnl': round((eq-200)/200*100,2)}

# Sweep — SL < TP only
obi_results = []
tested = 0

for taker_t in [0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75]:
    for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for sl in [0.3, 0.5, 0.75]:  # SL must be < TP
            if sl >= tp: continue
            for hold in [8, 12, 16, 24]:
                for vol_min in [0, 0.5, 1.0, 1.5]:
                    for trend in [True, False]:
                        for session in ['ln', 'ol', None]:
                            r = backtest_obi(tp, sl, hold, taker_t, vol_min, trend, session)
                            tested += 1
                            if r:
                                obi_results.append({**r, 'taker': taker_t, 'tp': tp, 'sl': sl,
                                                    'hold': hold, 'vol': vol_min, 'trend': trend,
                                                    'session': session or 'all'})
                            if tested % 2000 == 0:
                                print(f"  OBI {tested}...", flush=True)

obi_results.sort(key=lambda x: x.get('pf', 0) * (x.get('wr', 0)/100), reverse=True)
obi_ok = [r for r in obi_results if r['wr'] >= 70 and r['pf'] >= 1.5 and r['trades'] >= 5]

print(f"\nOBI: Tested {tested}, {len(obi_ok)} meet target (WR>=70%, PF>=1.5, T>=5)", flush=True)
print(f"\n{'='*100}", flush=True)
print(f"{'OBI TOP 15 (SL < TP)':^100s}", flush=True)
print(f"{'='*100}", flush=True)
print(f"  {'#':>2s} {'TP':>4s} {'SL':>4s} {'RR':>4s} {'Hold':>4s} {'Taker':>5s} {'Vol':>4s} {'Trnd':>4s} {'Sess':>3s} | {'Tr':>4s} {'W':>3s} {'L':>3s} {'WR':>5s} {'PF':>5s} {'PnL%':>7s} {'DD%':>5s}", flush=True)
print("  " + "-" * 95, flush=True)
for i, r in enumerate(obi_results[:15]):
    rr = round(r['tp']/r['sl'], 1)
    ok = " *" if r['wr'] >= 70 and r['pf'] >= 1.5 else ""
    print(f"  {i+1:>2d} {r['tp']:>4.1f} {r['sl']:>4.2f} {rr:>4.1f} {r['hold']:>4d} {r['taker']:>5.2f} {r['vol']:>4.1f} {'T' if r['trend'] else 'F':>4s} {r['session']:>3s} | {r['trades']:>4d} {r['wins']:>3d} {r['losses']:>3d} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>+7.1f} {r['dd']:>5.1f}{ok}", flush=True)

# ============================================================
# PART 2: momentum_v3 — TP/SL sweep with SL < TP
# ============================================================
print(f"\n{'='*80}", flush=True)
print("PART 2: momentum_v3 — TP/SL sweep (SL < TP)", flush=True)
print(f"{'='*80}", flush=True)

eth_keys = [b['ts'].strftime('%Y-%m-%d %H:%M:%S') for b in bars]

def sim_mv3(event_sigs, min_signals=2, tp_atr=1.0, sl_atr=0.5, hold=16):
    trades = []
    capital = 200.0
    peak = capital
    max_dd = 0
    used = set()
    for sig in event_sigs:
        ts = sig['timestamp']
        d = sig['direction']
        p = sig.get('entry') or sig.get('price', 0)
        if not d or not p: continue
        if ts in used: continue
        used.add(ts)
        idx = eth_map.get(ts, -1)
        if idx < 0 or idx < 80 or idx >= len(eth_keys) - hold: continue
        exc = detect_exhaustion(idx, min_signals=min_signals)
        if not exc: continue
        if exc['direction'] != d: continue
        atr_val = atr[idx] if idx < len(atr) else 0
        if atr_val <= 0: continue
        if d == 'LONG':
            tp_p = p + tp_atr * atr_val; sl_p = p - sl_atr * atr_val
        else:
            tp_p = p - tp_atr * atr_val; sl_p = p + sl_atr * atr_val
        outcome = None
        for j in range(idx + 1, min(idx + hold + 1, len(eth_keys))):
            if d == 'LONG':
                if highs[j] >= tp_p: outcome = 'W'; exit_p = tp_p; break
                if lows[j] <= sl_p: outcome = 'L'; exit_p = sl_p; break
            else:
                if lows[j] <= tp_p: outcome = 'W'; exit_p = tp_p; break
                if highs[j] >= sl_p: outcome = 'L'; exit_p = sl_p; break
        if outcome is None:
            exit_p = closes[min(idx + hold, len(eth_keys) - 1)]
            outcome = 'W' if ((d == 'LONG' and exit_p > p) or (d == 'SHORT' and exit_p < p)) else 'L'
        pnl_pct = ((exit_p - p) / p * 100) if d == 'LONG' else ((p - exit_p) / p * 100)
        size = capital * 0.10 * 25
        net_pnl = size * pnl_pct / 100 - size * 0.0005 * 2
        capital += net_pnl
        if capital > peak: peak = capital
        dd = (peak - capital) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        trades.append({'time': ts, 'dir': d, 'outcome': outcome, 'pnl_dollar': net_pnl, 'pnl_pct': pnl_pct})
    if not trades: return None
    wins = [t for t in trades if t['outcome'] == 'W']
    losses = [t for t in trades if t['outcome'] == 'L']
    tw = sum(t['pnl_dollar'] for t in wins)
    tl = sum(abs(t['pnl_dollar']) for t in losses)
    pf = tw / tl if tl > 0 else float('inf')
    return {'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
            'wr': round(len(wins)/len(trades)*100, 1), 'pf': round(pf, 2),
            'pnl': round(sum(t['pnl_dollar'] for t in trades), 2),
            'max_dd': round(max_dd, 2)}

mv3_results = []
for strat_name in ['orderbook_imbalance', 'trade_flow', 'squeeze_breakout', 'positioning_fade']:
    sigs = event_signals.get(strat_name, [])
    if len(sigs) < 5:
        print(f"  {strat_name}: only {len(sigs)} signals, skipping", flush=True)
        continue

    print(f"\n  {strat_name} ({len(sigs)} signals):", flush=True)
    print(f"  {'MinSig':>6s} {'TP':>4s} {'SL':>4s} {'RR':>4s} {'Hold':>4s} | {'Tr':>4s} {'W':>3s} {'L':>3s} {'WR':>5s} {'PF':>5s} {'PnL$':>8s} {'DD%':>5s}", flush=True)
    print("  " + "-" * 65, flush=True)

    for min_sig in [2, 3]:
        for tp_atr in [0.5, 0.75, 1.0, 1.5, 2.0]:
            for sl_atr in [0.25, 0.3, 0.5]:  # SL < TP
                if sl_atr >= tp_atr: continue
                for hold in [8, 12, 16, 24]:
                    m = sim_mv3(sigs, min_signals=min_sig, tp_atr=tp_atr, sl_atr=sl_atr, hold=hold)
                    if m and m['trades'] >= 3:
                        rr = round(tp_atr/sl_atr, 1)
                        hit = " *" if m['pf'] >= 2.0 and m['wr'] >= 70 else ""
                        print(f"  {min_sig:>6d} {tp_atr:>4.2f} {sl_atr:>4.2f} {rr:>4.1f} {hold:>4d} | {m['trades']:>4d} {m['wins']:>3d} {m['losses']:>3d} {m['wr']:>5.1f} {m['pf']:>5.2f} ${m['pnl']:>+7.2f} {m['max_dd']:>5.1f}{hit}", flush=True)
                        mv3_results.append({**m, 'strategy': strat_name, 'min_sig': min_sig,
                                           'tp_atr': tp_atr, 'sl_atr': sl_atr, 'hold': hold,
                                           'rr': rr})

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*80}", flush=True)
print("SUMMARY — Recommended configs (SL < TP)", flush=True)
print(f"{'='*80}", flush=True)

print("\norderbook_imbalance — Best config with SL < TP:", flush=True)
if obi_results:
    b = obi_results[0]
    rr = b['tp']/b['sl']
    print(f"  TP={b['tp']}% SL={b['sl']}% Hold={b['hold']}h Taker>={b['taker']} Vol>={b['vol']} Trend={b['trend']} Session={b['session']}", flush=True)
    print(f"  {b['trades']}T {b['wr']}%WR {b['pf']}PF R:R=1:{rr:.1f} PnL={b['pnl']}% DD={b['dd']}%", flush=True)

print("\nmomentum_v3 — Best config with SL < TP:", flush=True)
if mv3_results:
    best_mv3 = sorted(mv3_results, key=lambda x: x.get('pf', 0) * (x.get('wr', 0)/100), reverse=True)
    b = best_mv3[0]
    print(f"  Strategy={b['strategy']} TP={b['tp_atr']}ATR SL={b['sl_atr']}ATR Hold={b['hold']}h MinExc={b['min_sig']}", flush=True)
    print(f"  {b['trades']}T {b['wr']}%WR {b['pf']}PF R:R=1:{b['rr']:.1f} PnL=${b['pnl']:.2f} DD={b['max_dd']}%", flush=True)

print("\nDone", flush=True)
