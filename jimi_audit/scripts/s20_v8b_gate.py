"""
8-Agent Gate: S20 v8b Liquidation Mean Reversion (Simplified)
=============================================================
Tests: OI drop >1.5% → LONG mean reversion
Gate finding: +0.86%, p=0.008, WR=74.3%, n=35
"""

import pandas as pd
import numpy as np
from scipy import stats
import json, os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'
OUTPUT_FILE = '/root/.openclaw/workspace/jimi_audit/reports/s20_v8b_gate.json'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print("Loading data...")
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
for c in ['Close','High','Low','Volume']: ohlcv[c] = ohlcv[c].astype(float)

deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

merged = pd.merge_asof(ohlcv[['timestamp','Open','High','Low','Close','Volume']],
                       deriv[['timestamp','oi','ls_ratio','funding_rate']],
                       on='timestamp', direction='backward', tolerance=pd.Timedelta('2h'))

merged['oi_roc'] = merged['oi'].pct_change(4, fill_method=None)
merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()
merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')
merged['atr'] = (merged['High'] - merged['Low']).rolling(14).mean()
merged['price_disp'] = merged['Close'].pct_change(5)

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

round_trip_cost = 0.0010
results = {}

# ═══════════ AGENT 1: FORENSICS ═══════════
print("\n" + "="*70)
print("AGENT 1: FORENSICS")
print("="*70)
a1 = {}
for thresh in [0.005, 0.01, 0.015, 0.02, 0.03]:
    n = (merged['oi_roc'] < -thresh).sum()
    a1[f'oi_drop_gt_{thresh}'] = int(n)
    print(f"  OI drop > {thresh}: {n}")
july = merged[merged['is_july_2026']]
a1['july_oi_drop_1.5pct'] = int((july['oi_roc'] < -0.015).sum())
print(f"  July 2026 OI drop >1.5%: {a1['july_oi_drop_1.5pct']}")
results['agent_1'] = a1

# ═══════════ AGENT 2: NON-INDICATOR ═══════════
print("\n" + "="*70)
print("AGENT 2: NON-INDICATOR — Raw OI drop signal")
print("="*70)
a2 = {}
configs = [
    ('OI_drop_1pct', -0.01, None, None),
    ('OI_drop_1.5pct', -0.015, None, None),
    ('OI_drop_2pct', -0.02, None, None),
    ('OI_drop_1pct_vol_1.5x', -0.01, 1.5, None),
    ('OI_drop_1.5pct_vol_1.5x', -0.015, 1.5, None),
    ('OI_drop_1pct_vol_2x', -0.01, 2.0, None),
    ('OI_drop_1.5pct_vol_2x', -0.015, 2.0, None),
    ('OI_drop_1pct_fr_0.03pct', -0.01, None, 0.0003),
    ('OI_drop_1.5pct_fr_0.03pct', -0.015, None, 0.0003),
    ('OI_drop_1.5pct_vol_1.5x_fr', -0.015, 1.5, 0.0003),
]
for name, oi_t, vol_t, fr_t in configs:
    mask = merged['oi_roc'] < oi_t
    if vol_t: mask = mask & (merged['vol_ratio'] > vol_t)
    if fr_t: mask = mask & (merged['funding_rate'].abs() > fr_t)
    shifted = mask.shift(1).fillna(False)
    events = merged[shifted]
    for h, label in [(1, '1h'), (16, '4h')]:
        rets = merged.loc[events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 3: continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        key = f"{name}_{label}"
        a2[key] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} {name:30s} {label}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")
results['agent_2'] = a2

# ═══════════ AGENT 3: INDICATOR — v8b trigger ═══════════
print("\n" + "="*70)
print("AGENT 3: INDICATOR — v8b trigger (OI drop >1.5% + price not bounced)")
print("="*70)
a3 = {}

# v8b: OI ROC < -0.015, price_change < 0.02 (not chased)
v8b_mask = (
    (merged['oi_roc'] < -0.015) &
    (merged['price_disp'] < 0.02)  # price not already bounced >2%
)
v8b_events = merged[v8b_mask.shift(1).fillna(False)]
a3['v8b_signals'] = len(v8b_events)
print(f"  v8b signals: {len(v8b_events)}")

if len(v8b_events) > 0:
    for h, label in [(1, '15m'), (4, '1h'), (16, '4h')]:
        rets = merged.loc[v8b_events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 3: continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        a3[f'v8b_{label}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} v8b {label}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")
    july_v8b = v8b_events[merged.loc[v8b_events.index, 'is_july_2026']]
    a3['v8b_july'] = len(july_v8b)
    print(f"  v8b July 2026: {len(july_v8b)}")
results['agent_3'] = a3

# ═══════════ AGENT 4: REGIME ═══════════
print("\n" + "="*70)
print("AGENT 4: REGIME")
print("="*70)
a4 = {}
if len(v8b_events) > 0:
    for col, name in [('vol_regime', 'Vol'), ('trend', 'Trend'), ('era', 'Era')]:
        print(f"\n  --- {name} ---")
        for reg in sorted(merged[col].dropna().unique()):
            evts = v8b_events[merged.loc[v8b_events.index, col] == reg]
            if len(evts) < 3:
                print(f"    {reg}: n={len(evts)}")
                continue
            rets = merged.loc[evts.index, 'fwd_ret_16'].dropna()
            if len(rets) < 3: continue
            mean_r = rets.mean()
            t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
            gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
            a4[f'{col}_{reg}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'gate': gate}
            print(f"    {'+' if gate=='PASS' else '-'} {reg:12s}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}")
results['agent_4'] = a4

# ═══════════ AGENT 5: GATE ═══════════
print("\n" + "="*70)
print("AGENT 5: GATE — Signal frequency")
print("="*70)
months = max(1, (merged['timestamp'].max() - merged['timestamp'].min()).days / 30)
a5 = {'total': len(v8b_events), 'per_month': len(v8b_events)/months}
print(f"  Total: {len(v8b_events)}, Per month: {a5['per_month']:.1f}")
results['agent_5'] = a5

# ═══════════ AGENT 6: CO-OCCURRENCE ═══════════
print("\n" + "="*70)
print("AGENT 6: CO-OCCURRENCE")
print("="*70)
a6 = {}
if len(v8b_events) > 0:
    ls = merged.loc[v8b_events.index, 'ls_ratio'].dropna()
    fr = merged.loc[v8b_events.index, 'funding_rate'].dropna()
    vr = merged.loc[v8b_events.index, 'vol_ratio'].dropna()
    if len(ls)>0: a6['ls_mean'] = float(ls.mean()); print(f"  LS ratio: {ls.mean():.3f}")
    if len(fr)>0: a6['fr_mean'] = float(fr.mean()); print(f"  Funding: {fr.mean():.6f}")
    if len(vr)>0: a6['vol_mean'] = float(vr.mean()); print(f"  Vol ratio: {vr.mean():.2f}x")
    td = merged.loc[v8b_events.index, 'trend'].value_counts()
    a6['trend'] = td.to_dict(); print(f"  Trend: {td.to_dict()}")
results['agent_6'] = a6

# ═══════════ AGENT 7: SENSITIVITY ═══════════
print("\n" + "="*70)
print("AGENT 7: SENSITIVITY")
print("="*70)
a7 = {}
for oi_t in [0.01, 0.012, 0.015, 0.02, 0.025]:
    for vol_t in [None, 1.5, 2.0]:
        mask = merged['oi_roc'] < -oi_t
        if vol_t: mask = mask & (merged['vol_ratio'] > vol_t)
        shifted = mask.shift(1).fillna(False)
        events = merged[shifted]
        rets = merged.loc[events.index, 'fwd_ret_16'].dropna()
        if len(rets) < 5: continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        if gate == "PASS":
            vt = vol_t if vol_t else 'any'
            key = f"OI<{-oi_t}_V>{vt}"
            a7[key] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr)}
            print(f"  + OI<{-oi_t:.3f} V>{vt}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")
results['agent_7'] = a7

# ═══════════ AGENT 8: MONTE CARLO ═══════════
print("\n" + "="*70)
print("AGENT 8: MONTE CARLO")
print("="*70)
a8 = {}
if len(v8b_events) >= 5:
    actual = merged.loc[v8b_events.index, 'fwd_ret_16'].dropna()
    am, aw, n = actual.mean(), (actual>0).mean(), len(actual)
    print(f"  n={n}, mean={am*100:+.4f}%, WR={aw:.1%}")
    
    np.random.seed(42)
    all_r = merged['fwd_ret_16'].dropna()
    rm = np.array([all_r.sample(n).mean() for _ in range(10000)])
    rw = np.array([(all_r.sample(n)>0).mean() for _ in range(10000)])
    pm = (rm >= am).mean()
    
    bm = np.array([actual.sample(n, replace=True).mean() for _ in range(10000)])
    ci_lo, ci_hi = np.percentile(bm, 2.5), np.percentile(bm, 97.5)
    
    a8 = {'n': n, 'mean': float(am), 'wr': float(aw), 'mc_p': float(pm),
          'ci': [float(ci_lo), float(ci_hi)], 'sig': pm < 0.05}
    print(f"  MC p: {pm:.4f}")
    print(f"  CI: [{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%]")
    print(f"  SIGNIFICANT: {'YES' if pm < 0.05 else 'NO'}")
else:
    a8 = {'n': len(v8b_events), 'sig': False}
    print(f"  Too few ({len(v8b_events)})")
results['agent_8'] = a8

# ═══════════ VERDICT ═══════════
print("\n" + "="*70)
print("VERDICT")
print("="*70)
v = {
    'strategy': 'S20 v8b',
    'signals': len(v8b_events),
    'mc_sig': a8.get('sig', False),
}
if a8.get('sig'):
    v['gate'] = 'PASS'
    v['rec'] = 'Deploy 0.5x size, validate with 30+ trades'
elif a3.get('v8b_4h', {}).get('p', 1) < 0.1:
    v['gate'] = 'MARGINAL'
    v['rec'] = 'Deploy 0.3x size, extend validation'
elif len(v8b_events) < 10:
    v['gate'] = 'LOW_SAMPLE'
    v['rec'] = 'Deploy provisionally, OI collector upgrade recommended'
else:
    v['gate'] = 'FAIL'
    v['rec'] = 'Not viable'
print(f"  Signals: {v['signals']}")
print(f"  Gate: {v['gate']}")
print(f"  Rec: {v['rec']}")
results['verdict'] = v

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUTPUT_FILE}")
