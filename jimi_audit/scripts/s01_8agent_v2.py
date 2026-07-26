"""
8-Agent Forensic Protocol: S01 Failed Breakout v2 (FIXED)
=========================================================
Bug fix: swing levels now computed from bars BEFORE breakout bar (not including it).
Bug fix: deduplication — each failed breakout counted only once.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json, os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'
OUTPUT_FILE = '/root/.openclaw/workspace/jimi_audit/reports/s01_failed_breakout_forensic.json'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print("Loading data...")
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
for c in ['Close','High','Low','Volume']: ohlcv[c] = ohlcv[c].astype(float)
print(f"OHLCV: {len(ohlcv)} bars")

deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

merged = pd.merge_asof(
    ohlcv[['timestamp','Open','High','Low','Close','Volume']],
    deriv[['timestamp','oi','ls_ratio','funding_rate']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)

merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()
merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')
merged['atr'] = (merged['High'] - merged['Low']).rolling(14).mean()

for h in [1, 4, 16, 24]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

vols = merged['vol_20bar'].dropna()
p33, p67 = vols.quantile(0.33), vols.quantile(0.67)
merged['vol_regime'] = 'MID'
merged.loc[merged['vol_20bar'] < p33, 'vol_regime'] = 'LOW'
merged.loc[merged['vol_20bar'] > p67, 'vol_regime'] = 'HIGH'

def get_era(ts):
    if ts < pd.Timestamp('2024-07-01'): return '2024_H1'
    elif ts < pd.Timestamp('2025-01-01'): return '2024_H2'
    elif ts < pd.Timestamp('2025-07-01'): return '2025_H1'
    elif ts < pd.Timestamp('2026-01-01'): return '2025_H2'
    else: return '2026'
merged['era'] = merged['timestamp'].apply(get_era)
merged['is_july_2026'] = (merged['timestamp'] >= '2026-07-01') & (merged['timestamp'] < '2026-08-01')
merged['hour'] = merged['timestamp'].dt.hour
GOOD_HOURS = {9, 10, 11, 12, 14, 15, 16, 18}

round_trip_cost = 0.0010
results = {}

# ═══════════ DETECTION (FIXED) ═══════════
print("Detecting failed breakouts (fixed)...")

highs = merged['High'].values.astype(float)
lows = merged['Low'].values.astype(float)
closes = merged['Close'].values.astype(float)

fb_events = []
seen_breakouts = set()  # deduplicate by (breakout_bar, direction)

for idx in range(56, len(ohlcv)):
    for lb in range(1, min(8, idx)):
        bar_idx = idx - lb
        bar_high = highs[bar_idx]
        bar_low = lows[bar_idx]
        bar_close = closes[bar_idx]
        
        if bar_idx < 48:
            continue
        
        # Swing levels from bars BEFORE breakout bar
        swing_high = float(np.max(highs[bar_idx-48:bar_idx]))
        swing_low = float(np.min(lows[bar_idx-48:bar_idx]))
        
        # Failed breakout ABOVE → SHORT
        if bar_high > swing_high * 1.001 and bar_close < swing_high:
            key = (bar_idx, 'SHORT')
            if key not in seen_breakouts:
                bars_held = 0
                for j in range(bar_idx, idx + 1):
                    if highs[j] > swing_high:
                        bars_held += 1
                    else:
                        break
                if closes[idx] < swing_high and bars_held >= 1:
                    seen_breakouts.add(key)
                    fb_events.append({
                        'idx': idx, 'direction': 'SHORT',
                        'level': swing_high, 'level_type': 'swing_high',
                        'bars_held': bars_held,
                        'quality': min(bars_held / 5, 1.0),
                        'breakout_bar': bar_idx,
                    })
                    break
        
        # Failed breakout BELOW → LONG
        if bar_low < swing_low * 0.999 and bar_close > swing_low:
            key = (bar_idx, 'LONG')
            if key not in seen_breakouts:
                bars_held = 0
                for j in range(bar_idx, idx + 1):
                    if lows[j] < swing_low:
                        bars_held += 1
                    else:
                        break
                if closes[idx] > swing_low and bars_held >= 1:
                    seen_breakouts.add(key)
                    fb_events.append({
                        'idx': idx, 'direction': 'LONG',
                        'level': swing_low, 'level_type': 'swing_low',
                        'bars_held': bars_held,
                        'quality': min(bars_held / 5, 1.0),
                        'breakout_bar': bar_idx,
                    })
                    break

print(f"  Total (deduplicated): {len(fb_events)}")

# ═══════════ AGENT 1: FORENSICS ═══════════
print("\n" + "="*70)
print("AGENT 1: FORENSICS")
print("="*70)

a1 = {}
qualities = [e['quality'] for e in fb_events]
bars = [e['bars_held'] for e in fb_events]
a1['total'] = len(fb_events)
a1['quality_mean'] = float(np.mean(qualities)) if qualities else 0
a1['bars_held_mean'] = float(np.mean(bars)) if bars else 0
long_events = [e for e in fb_events if e['direction'] == 'LONG']
short_events = [e for e in fb_events if e['direction'] == 'SHORT']
a1['long'] = len(long_events)
a1['short'] = len(short_events)
a1['july'] = sum(1 for e in fb_events if merged.iloc[e['idx']]['is_july_2026'])

for q_min, q_max, label in [(0, 0.2, 'low'), (0.2, 0.5, 'med'), (0.5, 1.0, 'high')]:
    n = sum(1 for q in qualities if q_min <= q < q_max)
    a1[f'q_{label}'] = n
    print(f"  Quality {label}: {n}")

print(f"  LONG={len(long_events)}, SHORT={len(short_events)}, July={a1['july']}")
results['agent_1'] = a1

# ═══════════ AGENT 2: NON-INDICATOR ═══════════
print("\n" + "="*70)
print("AGENT 2: NON-INDICATOR — Raw edge")
print("="*70)

a2 = {}
for direction in ['LONG', 'SHORT']:
    de = [e for e in fb_events if e['direction'] == direction]
    if len(de) < 5: continue
    indices = [e['idx'] for e in de]
    for h, label in [(4, '1h'), (16, '4h')]:
        rets = merged.iloc[indices][f'fwd_ret_{h}'].dropna()
        if len(rets) < 3: continue
        mean_r = rets.mean()
        eff = -mean_r if direction == 'SHORT' else mean_r
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets < 0).mean() if direction == 'SHORT' else (rets > 0).mean()
        gate = "PASS" if p < 0.1 and eff > round_trip_cost else "FAIL"
        a2[f'{direction}_{label}'] = {'n': len(rets), 'mean': float(mean_r), 'eff': float(eff), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} {direction:5s} {label}: n={len(rets):4d}, eff={eff*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# Quality buckets
print("\n  --- Quality ---")
for q_min, label in [(0, 'low'), (0.2, 'med'), (0.5, 'high')]:
    qe = [e for e in fb_events if e['quality'] >= q_min]
    if len(qe) < 5: continue
    indices = [e['idx'] for e in qe]
    rets = merged.iloc[indices]['fwd_ret_16'].dropna()
    if len(rets) < 3: continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
    a2[f'q_{label}_4h'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
    print(f"  {'+' if gate=='PASS' else '-'} q>={q_min} 4h: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_2'] = a2

# ═══════════ AGENT 3: INDICATOR (S01 filters) ═══════════
print("\n" + "="*70)
print("AGENT 3: INDICATOR — S01 v2 filters")
print("="*70)

a3 = {}
s01_events = []
for e in fb_events:
    idx = e['idx']
    row = merged.iloc[idx]
    
    # Session
    if row['hour'] not in GOOD_HOURS: continue
    # Quality
    if e['bars_held'] < 1: continue
    # EMA200 (3%)
    ema200 = row['ema200']
    price = row['Close']
    if ema200 > 0:
        dist = (price - ema200) / ema200
        if e['direction'] == 'LONG' and dist < -0.03: continue
        if e['direction'] == 'SHORT' and dist > 0.03: continue
    # Volume (no high vol)
    vr = row['vol_ratio'] if pd.notna(row['vol_ratio']) else 1.0
    if vr > 2.0: continue
    
    s01_events.append(e)

a3['s01_signals'] = len(s01_events)
print(f"  S01 signals: {len(s01_events)}")

if len(s01_events) >= 5:
    indices = [e['idx'] for e in s01_events]
    for h, label in [(4, '1h'), (16, '4h')]:
        rets = merged.iloc[indices][f'fwd_ret_{h}'].dropna()
        if len(rets) < 3: continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        a3[f's01_{label}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} S01 {label}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_3'] = a3

# ═══════════ AGENT 4: REGIME ═══════════
print("\n" + "="*70)
print("AGENT 4: REGIME")
print("="*70)

a4 = {}
test_events = s01_events if len(s01_events) >= 10 else fb_events
indices_all = [e['idx'] for e in test_events]

for col, name in [('vol_regime', 'Vol'), ('trend', 'Trend'), ('era', 'Era')]:
    print(f"\n  --- {name} ---")
    for reg in sorted(merged[col].dropna().unique()):
        ri = [i for i in indices_all if merged.iloc[i][col] == reg]
        if len(ri) < 3: 
            print(f"    {reg}: n={len(ri)}")
            continue
        rets = merged.iloc[ri]['fwd_ret_16'].dropna()
        if len(rets) < 3: continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        a4[f'{col}_{reg}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"    {'+' if gate=='PASS' else '-'} {reg:12s}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# EMA200 distance
print("\n  --- EMA200 ---")
for dmin, dmax, label in [(-0.05, -0.02, 'below'), (-0.02, 0.02, 'near'), (0.02, 0.05, 'above')]:
    di = [i for i in indices_all if dmin <= (merged.iloc[i]['Close'] - merged.iloc[i]['ema200']) / merged.iloc[i]['ema200'] < dmax]
    if len(di) < 3: continue
    rets = merged.iloc[di]['fwd_ret_16'].dropna()
    if len(rets) < 3: continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
    a4[f'ema_{label}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'gate': gate}
    print(f"  {'+' if gate=='PASS' else '-'} {label:10s}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}")

results['agent_4'] = a4

# ═══════════ AGENT 5: GATE ═══════════
print("\n" + "="*70)
print("AGENT 5: GATE — Frequency & structural TP/SL")
print("="*70)

months = max(1, (merged['timestamp'].max() - merged['timestamp'].min()).days / 30)
a5 = {'raw': len(fb_events), 'filtered': len(s01_events),
      'raw_per_month': len(fb_events)/months, 'filtered_per_month': len(s01_events)/months}
print(f"  Raw: {len(fb_events)} ({len(fb_events)/months:.1f}/mo), Filtered: {len(s01_events)} ({len(s01_events)/months:.1f}/mo)")

# Structural TP/SL
tp_hits = {'struct': 0, 'fixed': 0, 'total': 0}
for e in s01_events[:200]:
    idx = e['idx']
    if idx + 16 >= len(merged): continue
    entry = merged.iloc[idx]['Close']
    future = merged.iloc[idx+1:idx+17]['Close'].values
    
    if e['direction'] == 'SHORT':
        stp = e['level'] * 0.99
        ftp = entry * 0.98
        tp_hits['total'] += 1
        if any(c <= stp for c in future): tp_hits['struct'] += 1
        if any(c <= ftp for c in future): tp_hits['fixed'] += 1
    else:
        stp = e['level'] * 1.01
        ftp = entry * 1.02
        tp_hits['total'] += 1
        if any(c >= stp for c in future): tp_hits['struct'] += 1
        if any(c >= ftp for c in future): tp_hits['fixed'] += 1

if tp_hits['total'] > 0:
    a5['struct_tp'] = tp_hits['struct'] / tp_hits['total']
    a5['fixed_tp'] = tp_hits['fixed'] / tp_hits['total']
    print(f"  Structural TP: {tp_hits['struct']}/{tp_hits['total']} ({tp_hits['struct']/tp_hits['total']:.1%})")
    print(f"  Fixed 2% TP: {tp_hits['fixed']}/{tp_hits['total']} ({tp_hits['fixed']/tp_hits['total']:.1%})")

results['agent_5'] = a5

# ═══════════ AGENT 6: CO-OCCURRENCE ═══════════
print("\n" + "="*70)
print("AGENT 6: CO-OCCURRENCE")
print("="*70)

a6 = {}
if len(s01_events) > 0:
    indices = [e['idx'] for e in s01_events]
    vr = merged.iloc[indices]['vol_ratio'].dropna()
    ls = merged.iloc[indices]['ls_ratio'].dropna()
    if len(vr) > 0: a6['vol'] = float(vr.mean()); print(f"  Vol: {vr.mean():.2f}x")
    if len(ls) > 0: a6['ls'] = float(ls.mean()); print(f"  LS: {ls.mean():.3f}")
    long_pct = sum(1 for e in s01_events if e['direction'] == 'LONG') / len(s01_events)
    a6['long_pct'] = long_pct; print(f"  Long%: {long_pct:.0%}")
results['agent_6'] = a6

# ═══════════ AGENT 7: SENSITIVITY ═══════════
print("\n" + "="*70)
print("AGENT 7: SENSITIVITY")
print("="*70)

a7 = {}
for q in [0, 0.1, 0.2, 0.3, 0.5]:
    for bh in [1, 2, 3]:
        evts = [e for e in fb_events if e['quality'] >= q and e['bars_held'] >= bh]
        if len(evts) < 10: continue
        indices = [e['idx'] for e in evts]
        rets = merged.iloc[indices]['fwd_ret_16'].dropna()
        if len(retts) < 5: continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        if gate == "PASS":
            key = f"q{q}_bh{bh}"
            a7[key] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr)}
            print(f"  + q>={q} bh>={bh}: n={len(rets)}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# Direction-specific
for d in ['LONG', 'SHORT']:
    for q in [0.2, 0.5]:
        evts = [e for e in fb_events if e['direction'] == d and e['quality'] >= q and e['bars_held'] >= 2]
        if len(evts) < 5: continue
        indices = [e['idx'] for e in evts]
        rets = merged.iloc[indices]['fwd_ret_16'].dropna()
        if len(retts) < 3: continue
        mean_r = rets.mean()
        eff = -mean_r if d == 'SHORT' else mean_r
        t, p = stats.ttest_1samp(rets, 0)
        wr = (rets < 0).mean() if d == 'SHORT' else (rets > 0).mean()
        gate = "PASS" if p < 0.1 and eff > round_trip_cost else "FAIL"
        if gate == "PASS":
            key = f"{d}_q{q}"
            a7[key] = {'n': len(rets), 'eff': float(eff), 'p': float(p), 'wr': float(wr)}
            print(f"  + {d} q>={q}: n={len(rets)}, eff={eff*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_7'] = a7

# ═══════════ AGENT 8: MONTE CARLO ═══════════
print("\n" + "="*70)
print("AGENT 8: MONTE CARLO")
print("="*70)

a8 = {}
test = s01_events if len(s01_events) >= 10 else fb_events
if len(test) >= 5:
    indices = [e['idx'] for e in test]
    actual = merged.iloc[indices]['fwd_ret_16'].dropna()
    am, aw, n = actual.mean(), (actual > 0).mean(), len(actual)
    print(f"  n={n}, mean={am*100:+.4f}%, WR={aw:.1%}")
    
    np.random.seed(42)
    all_r = merged['fwd_ret_16'].dropna()
    rm = np.array([all_r.sample(n).mean() for _ in range(10000)])
    pm = (rm >= am).mean()
    
    bm = np.array([actual.sample(n, replace=True).mean() for _ in range(10000)])
    ci_lo, ci_hi = np.percentile(bm, 2.5), np.percentile(bm, 97.5)
    
    a8 = {'n': n, 'mean': float(am), 'wr': float(aw), 'mc_p': float(pm),
          'ci': [float(ci_lo), float(ci_hi)], 'sig': pm < 0.05}
    print(f"  MC p: {pm:.4f}, CI: [{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%]")
    print(f"  SIGNIFICANT: {'YES' if pm < 0.05 else 'NO'}")
results['agent_8'] = a8

# ═══════════ VERDICT ═══════════
print("\n" + "="*70)
print("VERDICT")
print("="*70)

best = None
best_m = 0
for k, v in a4.items():
    if v.get('gate') == 'PASS' and v.get('mean', 0) > best_m:
        best_m = v['mean']
        best = k

v = {'strategy': 'S01 Failed Breakout v2', 'raw': len(fb_events), 'filtered': len(s01_events),
     'mc_sig': a8.get('sig', False), 'best_regime': best, 'best_mean': float(best_m) if best_m else None}

if a8.get('sig'):
    v['gate'] = 'PASS'; v['rec'] = 'Deploy with regime filter'
elif best and best_m > 0.003:
    v['gate'] = 'CONDITIONAL'; v['rec'] = f'Deploy only in {best}'
elif len(s01_events) < 10:
    v['gate'] = 'LOW_SAMPLE'; v['rec'] = 'Relax filters'
else:
    v['gate'] = 'FAIL'; v['rec'] = 'No edge'

print(f"  Raw: {v['raw']}, Filtered: {v['filtered']}, Best: {v['best_regime']}")
print(f"  Gate: {v['gate']}, Rec: {v['rec']}")
results['verdict'] = v

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUTPUT_FILE}")
