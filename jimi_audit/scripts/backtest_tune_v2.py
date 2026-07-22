#!/usr/bin/env python3
"""
Backtest tune v2: Alternative logic for orderbook_imbalance + momentum_v3
SL < TP constraint with different approaches
"""
import csv, json, sys, os
from datetime import datetime, timezone, timedelta
import numpy as np

BASE = '/root/.openclaw/workspace/jimi_audit'
DATA_FILE = f'{BASE}/eth_15m_6m.csv'
FEE = 0.0002
SLIP = 0.001

print("Loading data...", flush=True)

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

# EMA50 for trend
ema50 = np.zeros(N)
ema50[49] = np.mean(closes[:50])
m50 = 2 / 51
for i in range(50, N):
    ema50[i] = closes[i] * m50 + ema50[i-1] * (1 - m50)

print(f"Loaded {N} bars", flush=True)

# ============================================================
# APPROACH 1: OBI with trailing stop (move SL to breakeven)
# ============================================================
print("\n" + "=" * 90, flush=True)
print("APPROACH 1: OBI — Trailing Stop (move SL to breakeven after X% in profit)", flush=True)
print("=" * 90, flush=True)

def backtest_obi_trailing(tp, sl, hold, taker_t, vol_min, trend, session, trail_trigger, trail_offset):
    """OBI with trailing stop: once profit hits trail_trigger%, move SL to entry + trail_offset%"""
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

            # Check if SL got trailed to breakeven
            if pos.get('trailed'):
                # SL is at breakeven or better
                if pos['d'] == 'L':
                    if b['l'] <= pos['sl']: exit_p, outcome = pos['sl'], 'W' if pos['sl'] >= pos['e'] else 'L'
                    elif b['h'] >= pos['tp']: exit_p, outcome = pos['tp'], 'W'
                else:
                    if b['h'] >= pos['sl']: exit_p, outcome = pos['sl'], 'W' if pos['sl'] <= pos['e'] else 'L'
                    elif b['l'] <= pos['tp']: exit_p, outcome = pos['tp'], 'W'
            else:
                # Original SL
                if pos['d'] == 'L':
                    if b['h'] >= pos['tp']: exit_p, outcome = pos['tp'], 'W'
                    elif b['l'] <= pos['sl']: exit_p, outcome = pos['sl'], 'L'
                    else:
                        # Check trailing trigger
                        profit_pct = (b['h'] - pos['e']) / pos['e'] * 100
                        if profit_pct >= trail_trigger:
                            pos['sl'] = pos['e'] * (1 + trail_offset / 100)  # Move SL to entry + offset
                            pos['trailed'] = True
                else:
                    if b['l'] <= pos['tp']: exit_p, outcome = pos['tp'], 'W'
                    elif b['h'] >= pos['sl']: exit_p, outcome = pos['sl'], 'L'
                    else:
                        profit_pct = (pos['e'] - b['l']) / pos['e'] * 100
                        if profit_pct >= trail_trigger:
                            pos['sl'] = pos['e'] * (1 - trail_offset / 100)
                            pos['trailed'] = True

            if exit_p is None and held >= hold:
                exit_p = b['c']
                outcome = 'W' if ((pos['d']=='L' and exit_p > pos['e']) or (pos['d']=='S' and exit_p < pos['e'])) else 'L'

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
                pos = {'d': d, 'e': e, 'sl': sl_p, 'tp': tp_p, 'ot': ts, 'trailed': False}

    if len(trades) < 3: return None
    w = sum(1 for t in trades if t['o']=='W')
    l = sum(1 for t in trades if t['o']=='L')
    wr = w/len(trades)*100
    pf = w * tp / max(l * sl, 0.01) if l > 0 else 999
    return {'trades': len(trades), 'wins': w, 'losses': l,
            'wr': round(wr,2), 'pf': round(pf,2), 'eq': round(eq,2), 'dd': round(max_dd,2),
            'pnl': round((eq-200)/200*100,2)}

# Sweep trailing stop params
print(f"\n  {'TP':>4s} {'SL':>4s} {'RR':>4s} {'Trig':>5s} {'Off':>4s} {'Hold':>4s} {'Taker':>5s} {'Vol':>4s} | {'Tr':>4s} {'W':>3s} {'L':>3s} {'WR':>5s} {'PF':>5s} {'PnL%':>7s} {'DD%':>5s}", flush=True)
print("  " + "-" * 90, flush=True)

trail_results = []
for tp in [1.0, 1.5, 2.0]:
    for sl in [0.3, 0.5, 0.75]:
        if sl >= tp: continue
        for trail_trigger in [0.3, 0.5, 0.75]:
            for trail_offset in [0.0, 0.1, 0.2]:
                for taker_t in [0.62, 0.65, 0.70]:
                    for vol_min in [1.0, 1.5]:
                        for hold in [12, 16]:
                            r = backtest_obi_trailing(tp, sl, hold, taker_t, vol_min, True, 'ol', trail_trigger, trail_offset)
                            if r and r['trades'] >= 5:
                                rr = round(tp/sl, 1)
                                ok = r['wr'] >= 60 and r['pf'] >= 1.3
                                if ok or r['pf'] >= 1.1:
                                    trail_results.append({**r, 'tp': tp, 'sl': sl, 'rr': rr,
                                                         'trail_trigger': trail_trigger, 'trail_offset': trail_offset,
                                                         'taker': taker_t, 'vol': vol_min, 'hold': hold})

trail_results.sort(key=lambda x: x.get('pf', 0) * (x.get('wr', 0)/100), reverse=True)
for i, r in enumerate(trail_results[:20]):
    ok = " *" if r['wr'] >= 60 and r['pf'] >= 1.3 else ""
    print(f"  {r['tp']:>4.1f} {r['sl']:>4.2f} {r['rr']:>4.1f} {r['trail_trigger']:>5.2f} {r['trail_offset']:>4.1f} {r['hold']:>4d} {r['taker']:>5.2f} {r['vol']:>4.1f} | {r['trades']:>4d} {r['wins']:>3d} {r['losses']:>3d} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>+7.1f} {r['dd']:>5.1f}{ok}", flush=True)

# ============================================================
# APPROACH 2: OBI with dual timeframe confirmation
# ============================================================
print("\n" + "=" * 90, flush=True)
print("APPROACH 2: OBI — Dual confirmation (taker + volume + momentum direction)", flush=True)
print("=" * 90, flush=True)

def backtest_obi_momentum(tp, sl, hold, taker_t, vol_min, mom_period, mom_threshold):
    """OBI with momentum confirmation: price must be moving in signal direction"""
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
                exit_p = b['c']
                outcome = 'W' if ((pos['d']=='L' and exit_p > pos['e']) or (pos['d']=='S' and exit_p < pos['e'])) else 'L'
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
            taker = bars[i]['tb'] / max(bars[i]['v'], 0.01)
            vol_r = bars[i]['v'] / max(avg_vol[i], 1) if avg_vol[i] > 0 else 0
            if vol_r < vol_min: continue

            # Momentum confirmation
            if i < mom_period + 10: continue
            mom = (closes[i] - closes[i - mom_period]) / closes[i - mom_period]

            d = None
            if taker >= taker_t:
                if mom > mom_threshold:  # Price already moving up
                    d = 'L'
                    if closes[i] < ema200[i]: continue
            elif taker <= (1 - taker_t):
                if mom < -mom_threshold:  # Price already moving down
                    d = 'S'
                    if closes[i] > ema200[i]: continue

            if d:
                e = closes[i] * (1 + SLIP if d == 'L' else 1 - SLIP)
                sl_d = e * sl / 100
                tp_d = e * tp / 100
                if d == 'L':
                    sl_p, tp_p = e - sl_d, e + tp_d
                else:
                    sl_p, tp_p = e + sl_d, e - tp_d
                pos = {'d': d, 'e': e, 'sl': sl_p, 'tp': tp_p, 'ot': bars[i]['ts']}

    if len(trades) < 3: return None
    w = sum(1 for t in trades if t['o']=='W')
    l = sum(1 for t in trades if t['o']=='L')
    wr = w/len(trades)*100
    pf = w * tp / max(l * sl, 0.01) if l > 0 else 999
    return {'trades': len(trades), 'wins': w, 'losses': l,
            'wr': round(wr,2), 'pf': round(pf,2), 'eq': round(eq,2), 'dd': round(max_dd,2),
            'pnl': round((eq-200)/200*100,2)}

print(f"\n  {'TP':>4s} {'SL':>4s} {'RR':>4s} {'Mom':>4s} {'Thr':>5s} {'Taker':>5s} {'Vol':>4s} | {'Tr':>4s} {'W':>3s} {'L':>3s} {'WR':>5s} {'PF':>5s} {'PnL%':>7s} {'DD%':>5s}", flush=True)
print("  " + "-" * 80, flush=True)

mom_results = []
for tp in [1.0, 1.5, 2.0]:
    for sl in [0.3, 0.5, 0.75]:
        if sl >= tp: continue
        for mom_period in [4, 8, 12]:
            for mom_threshold in [0.001, 0.002, 0.003]:
                for taker_t in [0.58, 0.62, 0.65, 0.70]:
                    for vol_min in [0.5, 1.0]:
                        r = backtest_obi_momentum(tp, sl, 16, taker_t, vol_min, mom_period, mom_threshold)
                        if r and r['trades'] >= 5:
                            rr = round(tp/sl, 1)
                            ok = r['wr'] >= 60 and r['pf'] >= 1.3
                            if ok or (r['pf'] >= 1.1 and r['wr'] >= 50):
                                mom_results.append({**r, 'tp': tp, 'sl': sl, 'rr': rr,
                                                   'mom_period': mom_period, 'mom_threshold': mom_threshold,
                                                   'taker': taker_t, 'vol': vol_min})

mom_results.sort(key=lambda x: x.get('pf', 0) * (x.get('wr', 0)/100), reverse=True)
for i, r in enumerate(mom_results[:20]):
    ok = " *" if r['wr'] >= 60 and r['pf'] >= 1.3 else ""
    print(f"  {r['tp']:>4.1f} {r['sl']:>4.2f} {r['rr']:>4.1f} {r['mom_period']:>4d} {r['mom_threshold']:>5.3f} {r['taker']:>5.2f} {r['vol']:>4.1f} | {r['trades']:>4d} {r['wins']:>3d} {r['losses']:>3d} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>+7.1f} {r['dd']:>5.1f}{ok}", flush=True)

# ============================================================
# APPROACH 3: OBI with ATR-based SL (dynamic)
# ============================================================
print("\n" + "=" * 90, flush=True)
print("APPROACH 3: OBI — ATR-based SL (dynamic stop based on volatility)", flush=True)
print("=" * 90, flush=True)

def backtest_obi_atr(tp_pct, atr_mult, hold, taker_t, vol_min):
    """OBI with ATR-based SL: SL = ATR * atr_mult"""
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
                exit_p = b['c']
                outcome = 'W' if ((pos['d']=='L' and exit_p > pos['e']) or (pos['d']=='S' and exit_p < pos['e'])) else 'L'
            if exit_p:
                pnl = ((exit_p-pos['e'])/pos['e']) if pos['d']=='L' else ((pos['e']-exit_p)/pos['e'])
                pnl -= FEE*2
                eq += eq * pnl
                trades.append({'o': outcome, 'sl_dist': pos['sl_dist']})
                if eq > peak: peak = eq
                dd = (peak-eq)/peak*100
                if dd > max_dd: max_dd = dd
                pos = None

        if pos is None:
            taker = bars[i]['tb'] / max(bars[i]['v'], 0.01)
            vol_r = bars[i]['v'] / max(avg_vol[i], 1) if avg_vol[i] > 0 else 0
            if vol_r < vol_min: continue

            d = None
            if taker >= taker_t:
                d = 'L'
                if closes[i] < ema200[i]: continue
            elif taker <= (1 - taker_t):
                d = 'S'
                if closes[i] > ema200[i]: continue

            if d:
                e = closes[i] * (1 + SLIP if d == 'L' else 1 - SLIP)
                atr_val = atr[i]
                sl_d = atr_val * atr_mult
                tp_d = e * tp_pct / 100
                sl_pct_actual = sl_d / e * 100
                if sl_pct_actual >= tp_pct: continue  # SL must be < TP
                if d == 'L':
                    sl_p, tp_p = e - sl_d, e + tp_d
                else:
                    sl_p, tp_p = e + sl_d, e - tp_d
                pos = {'d': d, 'e': e, 'sl': sl_p, 'tp': tp_p, 'ot': bars[i]['ts'], 'sl_dist': sl_pct_actual}

    if len(trades) < 3: return None
    w = sum(1 for t in trades if t['o']=='W')
    l = sum(1 for t in trades if t['o']=='L')
    wr = w/len(trades)*100
    avg_sl = np.mean([t['sl_dist'] for t in trades]) if trades else 0
    pf = w * tp_pct / max(l * avg_sl, 0.01) if l > 0 else 999
    return {'trades': len(trades), 'wins': w, 'losses': l,
            'wr': round(wr,2), 'pf': round(pf,2), 'eq': round(eq,2), 'dd': round(max_dd,2),
            'pnl': round((eq-200)/200*100,2), 'avg_sl': round(avg_sl,3)}

print(f"\n  {'TP%':>4s} {'ATRx':>5s} {'AvgSL':>5s} {'Hold':>4s} {'Taker':>5s} {'Vol':>4s} | {'Tr':>4s} {'W':>3s} {'L':>3s} {'WR':>5s} {'PF':>5s} {'PnL%':>7s} {'DD%':>5s}", flush=True)
print("  " + "-" * 85, flush=True)

atr_results = []
for tp_pct in [1.0, 1.5, 2.0]:
    for atr_mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
        for taker_t in [0.58, 0.62, 0.65, 0.70]:
            for vol_min in [0.5, 1.0, 1.5]:
                for hold in [12, 16, 24]:
                    r = backtest_obi_atr(tp_pct, atr_mult, hold, taker_t, vol_min)
                    if r and r['trades'] >= 5:
                        ok = r['wr'] >= 60 and r['pf'] >= 1.3
                        if ok or (r['pf'] >= 1.1 and r['wr'] >= 50):
                            atr_results.append({**r, 'tp_pct': tp_pct, 'atr_mult': atr_mult,
                                               'taker': taker_t, 'vol': vol_min, 'hold': hold})

atr_results.sort(key=lambda x: x.get('pf', 0) * (x.get('wr', 0)/100), reverse=True)
for i, r in enumerate(atr_results[:20]):
    ok = " *" if r['wr'] >= 60 and r['pf'] >= 1.3 else ""
    print(f"  {r['tp_pct']:>4.1f} {r['atr_mult']:>5.2f} {r['avg_sl']:>5.2f} {r['hold']:>4d} {r['taker']:>5.2f} {r['vol']:>4.1f} | {r['trades']:>4d} {r['wins']:>3d} {r['losses']:>3d} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>+7.1f} {r['dd']:>5.1f}{ok}", flush=True)

# ============================================================
# APPROACH 4: momentum_v3 — pure filter (no standalone SL/TP)
# Confirm direction only, boost Group A size
# ============================================================
print("\n" + "=" * 90, flush=True)
print("APPROACH 4: momentum_v3 — Direction filter only (no own SL/TP, just confirms Group A)", flush=True)
print("=" * 90, flush=True)

# Load event signals
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
except:
    pass

eth_keys = [b['ts'].strftime('%Y-%m-%d %H:%M:%S') for b in bars]
eth_map_ts = {}
for i, k in enumerate(eth_keys):
    eth_map_ts[k] = i

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
    percentile = sum(1 for m2 in moves if m2 < current_move) / len(moves) * 100
    extreme = percentile > 85
    count = sum([decel, vol_div, extreme])
    if count < min_signals: return None
    direction = 'SHORT' if mom_5 > 0 else 'LONG' if mom_5 < 0 else None
    if not direction: return None
    return {'direction': direction, 'mom_5': mom_5, 'count': count}

def sim_mv3_filter(event_sigs, min_exc=2, tp_pct=1.0, sl_pct=0.5, hold=16):
    """Test momentum_v3 as a direction filter for Group A signals"""
    trades = []
    capital = 200.0
    peak = capital
    max_dd = 0
    used = set()
    confirmed = 0
    rejected = 0

    for sig in event_sigs:
        ts = sig['timestamp']
        d = sig['direction']
        p = sig.get('entry') or sig.get('price', 0)
        if not d or not p: continue
        if ts in used: continue
        used.add(ts)
        idx = eth_map_ts.get(ts, -1)
        if idx < 0 or idx < 80: continue

        # momentum_v3 confirmation
        exc = detect_exhaustion(idx, min_signals=min_exc)
        if not exc:
            rejected += 1
            continue
        if exc['direction'] != d:
            rejected += 1
            continue
        confirmed += 1

        # Use strategy SL/TP (SL < TP)
        e = p * (1 + SLIP if d == 'LONG' else 1 - SLIP)
        if d == 'LONG':
            tp_p = e * (1 + tp_pct/100)
            sl_p = e * (1 - sl_pct/100)
        else:
            tp_p = e * (1 - tp_pct/100)
            sl_p = e * (1 + sl_pct/100)

        # Simulate trade
        outcome = None
        for j in range(idx + 1, min(idx + hold * 4 + 1, N)):
            if d == 'LONG':
                if highs[j] >= tp_p: outcome = 'W'; exit_p = tp_p; break
                if lows[j] <= sl_p: outcome = 'L'; exit_p = sl_p; break
            else:
                if lows[j] <= tp_p: outcome = 'W'; exit_p = tp_p; break
                if highs[j] >= sl_p: outcome = 'L'; exit_p = sl_p; break
        if outcome is None:
            exit_p = closes[min(idx + hold * 4, N - 1)]
            outcome = 'W' if ((d == 'LONG' and exit_p > e) or (d == 'SHORT' and exit_p < e)) else 'L'

        pnl_pct = ((exit_p - e) / e * 100) if d == 'LONG' else ((e - exit_p) / e * 100)
        size = capital * 0.10 * 25
        net_pnl = size * pnl_pct / 100 - size * 0.0005 * 2
        capital += net_pnl
        if capital > peak: peak = capital
        dd = (peak - capital) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        trades.append({'outcome': outcome, 'pnl': net_pnl})

    if not trades: return None
    wins = sum(1 for t in trades if t['outcome'] == 'W')
    losses = sum(1 for t in trades if t['outcome'] == 'L')
    tw = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    tl = sum(abs(t['pnl']) for t in trades if t['pnl'] < 0)
    pf = tw / tl if tl > 0 else float('inf')
    return {'trades': len(trades), 'wins': wins, 'losses': losses,
            'wr': round(wins/len(trades)*100, 1), 'pf': round(pf, 2),
            'pnl': round(sum(t['pnl'] for t in trades), 2),
            'max_dd': round(max_dd, 2), 'confirmed': confirmed, 'rejected': rejected}

for strat_name in ['orderbook_imbalance', 'trade_flow', 'positioning_fade', 'squeeze_breakout']:
    sigs = event_signals.get(strat_name, [])
    if len(sigs) < 10: continue

    print(f"\n  {strat_name} ({len(sigs)} signals):", flush=True)
    print(f"  {'MinExc':>6s} {'TP%':>4s} {'SL%':>4s} {'RR':>4s} {'Hold':>4s} | {'Tr':>4s} {'W':>3s} {'L':>3s} {'WR':>5s} {'PF':>5s} {'PnL$':>8s} {'DD%':>5s} {'Conf':>4s} {'Rej':>4s}", flush=True)
    print("  " + "-" * 85, flush=True)

    for min_exc in [1, 2, 3]:
        for tp_pct in [0.5, 0.75, 1.0, 1.5, 2.0]:
            for sl_pct in [0.25, 0.3, 0.5]:
                if sl_pct >= tp_pct: continue
                for hold in [8, 12, 16]:
                    m = sim_mv3_filter(sigs, min_exc=min_exc, tp_pct=tp_pct, sl_pct=sl_pct, hold=hold)
                    if m and m['trades'] >= 3:
                        rr = round(tp_pct/sl_pct, 1)
                        hit = " *" if m['pf'] >= 1.5 and m['wr'] >= 55 else ""
                        print(f"  {min_exc:>6d} {tp_pct:>4.2f} {sl_pct:>4.2f} {rr:>4.1f} {hold:>4d} | {m['trades']:>4d} {m['wins']:>3d} {m['losses']:>3d} {m['wr']:>5.1f} {m['pf']:>5.2f} ${m['pnl']:>+7.2f} {m['max_dd']:>5.1f} {m['confirmed']:>4d} {m['rejected']:>4d}{hit}", flush=True)

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*90}", flush=True)
print("FINAL SUMMARY — Best configs found", flush=True)
print(f"{'='*90}", flush=True)

print("\n--- orderbook_imbalance ---", flush=True)
if trail_results:
    b = trail_results[0]
    print(f"  [Trailing Stop] TP={b['tp']}% SL={b['sl']}% Trail@{b['trail_trigger']}% Offset={b['trail_offset']}%", flush=True)
    print(f"    {b['trades']}T {b['wr']}%WR {b['pf']}PF PnL={b['pnl']}% DD={b['dd']}%", flush=True)
if mom_results:
    b = mom_results[0]
    print(f"  [Momentum Confirm] TP={b['tp']}% SL={b['sl']}% Mom={b['mom_period']}bar Threshold={b['mom_threshold']}", flush=True)
    print(f"    {b['trades']}T {b['wr']}%WR {b['pf']}PF PnL={b['pnl']}% DD={b['dd']}%", flush=True)
if atr_results:
    b = atr_results[0]
    print(f"  [ATR-based SL] TP={b['tp_pct']}% SL={b['atr_mult']}xATR(~{b['avg_sl']}%)", flush=True)
    print(f"    {b['trades']}T {b['wr']}%WR {b['pf']}PF PnL={b['pnl']}% DD={b['dd']}%", flush=True)

print("\n--- momentum_v3 ---", flush=True)
print("  Tested as direction filter for Group A strategies", flush=True)
print("  See results above per strategy", flush=True)

print("\nDone", flush=True)
