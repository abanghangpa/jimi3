"""
8-Agent Gate: S20 v8 Liquidation Mean Reversion
================================================
Tests the v8 signal: OI drop + volume spike + funding extreme + bounce
Research basis: SSRN 6579278, arXiv 2602.00776

Agents:
1. Forensics     — OI drop event identification
2. Non-Indicator — Raw OI drop + volume spike edge
3. Indicator     — Full v8 trigger replication
4. Regime        — Edge by regime
5. Gate          — Signal frequency & data freshness
6. Co-occurrence — Confluence with funding/LS/vol
7. Sensitivity   — Threshold optimization
8. Monte Carlo   — Statistical significance
"""

import pandas as pd
import numpy as np
from scipy import stats
import json, os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'
OUTPUT_FILE = '/root/.openclaw/workspace/jimi_audit/reports/s20_v8_gate.json'

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
print("Loading data...")

ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
ohlcv['Close'] = ohlcv['Close'].astype(float)
ohlcv['High'] = ohlcv['High'].astype(float)
ohlcv['Low'] = ohlcv['Low'].astype(float)
ohlcv['Volume'] = ohlcv['Volume'].astype(float)
print(f"OHLCV: {len(ohlcv)} bars")

deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)
print(f"Derivatives: {len(deriv)} rows")

merged = pd.merge_asof(
    ohlcv[['timestamp','Open','High','Low','Close','Volume']],
    deriv[['timestamp','oi','ls_ratio','funding_rate']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)

# Features
merged['oi_roc'] = merged['oi'].pct_change(4, fill_method=None)
merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')
merged['atr'] = (merged['High'] - merged['Low']).rolling(14).mean()
merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()
merged['price_disp'] = merged['Close'].pct_change(5)

# OI data age (approximate: time since last non-null OI)
merged['oi_valid'] = merged['oi'].notna()
merged['oi_group'] = (~merged['oi_valid']).cumsum()
merged['oi_timestamp'] = merged.groupby('oi_group')['timestamp'].transform('first')
merged['oi_age_sec'] = (merged['timestamp'] - merged['oi_timestamp']).dt.total_seconds()

# Forward returns
for h in [1, 4, 16, 24]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

# Regime
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

print(f"Merged: {len(merged)} rows")


# ═══════════════════════════════════════════════════════════════
# AGENT 1: FORENSICS — OI drop event identification
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 1: FORENSICS — OI drop events")
print("="*70)

a1 = {}
oi_drop_events = merged[merged['oi_roc'] < -0.01]
oi_drop_events_fresh = oi_drop_events[oi_drop_events['oi_age_sec'] < 1800]

a1['oi_drop_gt_1pct'] = len(oi_drop_events)
a1['oi_drop_gt_1pct_fresh'] = len(oi_drop_events_fresh)
print(f"  OI drop > 1%: {len(oi_drop_events)} events")
print(f"  OI drop > 1% (fresh <30min): {len(oi_drop_events_fresh)} events")

# July 2026
july_oi = merged[merged['is_july_2026'] & (merged['oi_roc'] < -0.01)]
a1['july_oi_drop'] = len(july_oi)
print(f"  July 2026 OI drops: {len(july_oi)}")

# By magnitude
for thresh in [0.005, 0.01, 0.015, 0.02, 0.03]:
    n = (merged['oi_roc'] < -thresh).sum()
    a1[f'oi_drop_gt_{thresh}'] = int(n)
    print(f"  OI drop > {thresh}: {n}")

results['agent_1'] = a1


# ═══════════════════════════════════════════════════════════════
# AGENT 2: NON-INDICATOR — Raw OI drop + volume spike
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 2: NON-INDICATOR — OI drop + volume spike edge")
print("="*70)

a2 = {}

configs = [
    ('OI_drop_1pct', {'oi_roc': -0.01}),
    ('OI_drop_1pct_vol_2x', {'oi_roc': -0.01, 'vol_ratio': 2.0}),
    ('OI_drop_1pct_vol_1.5x', {'oi_roc': -0.01, 'vol_ratio': 1.5}),
    ('OI_drop_1pct_fr_0.03pct', {'oi_roc': -0.01, 'fr': 0.0003}),
    ('OI_drop_1pct_vol_2x_fr', {'oi_roc': -0.01, 'vol_ratio': 2.0, 'fr': 0.0003}),
    ('OI_drop_1.5pct', {'oi_roc': -0.015}),
    ('OI_drop_1.5pct_vol_2x', {'oi_roc': -0.015, 'vol_ratio': 2.0}),
    ('OI_drop_2pct', {'oi_roc': -0.02}),
    ('OI_drop_2pct_vol_2x', {'oi_roc': -0.02, 'vol_ratio': 2.0}),
]

for name, filters in configs:
    mask = merged['oi_roc'] < filters['oi_roc']
    if 'vol_ratio' in filters:
        mask = mask & (merged['vol_ratio'] > filters['vol_ratio'])
    if 'fr' in filters:
        mask = mask & (merged['funding_rate'].abs() > filters['fr'])
    
    shifted = mask.shift(1).fillna(False)
    events = merged[shifted]
    
    for h, label in [(4, '1h'), (16, '4h')]:
        rets = merged.loc[events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 3:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        key = f"{name}_{label}"
        a2[key] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
        symbol = '+' if gate == 'PASS' else '-'
        print(f"  {symbol} {name:30s} {label}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_2'] = a2


# ═══════════════════════════════════════════════════════════════
# AGENT 3: INDICATOR — Full v8 trigger replication
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 3: INDICATOR — Full v8 trigger replication")
print("="*70)

a3 = {}

# v8 conditions:
# 1. OI ROC < -0.01 (OI dropping)
# 2. |price_disp| > 0.005 (price displaced)
# 3. vol_ratio > 2.0 (volume spike)
# 4. |funding_rate| > 0.0003 (overleveraged)
# 5. Price bouncing (close > close[-2])
# 6. OI data fresh (<30min)

v8_mask = (
    (merged['oi_roc'] < -0.01) &
    (merged['price_disp'].abs() > 0.005) &
    (merged['vol_ratio'] > 2.0) &
    (merged['funding_rate'].abs() > 0.0003) &
    (merged['Close'] > merged['Close'].shift(2)) &
    (merged['oi_age_sec'] < 1800)
)

v8_events = merged[v8_mask.shift(1).fillna(False)]
a3['v8_signals'] = len(v8_events)
print(f"  v8 signals (all conditions): {len(v8_events)}")

if len(v8_events) > 0:
    for h, label in [(1, '15m'), (4, '1h'), (16, '4h')]:
        rets = merged.loc[v8_events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 3:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        a3[f'v8_{label}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} v8 {label}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

    july_v8 = v8_events[merged.loc[v8_events.index, 'is_july_2026']]
    a3['v8_july_2026'] = len(july_v8)
    print(f"  v8 July 2026 signals: {len(july_v8)}")

# Relax conditions one by one to find what kills signal count
print("\n  --- CONDITION RELAXATION ---")
conditions = {
    'OI_drop': (merged['oi_roc'] < -0.01),
    'price_disp': (merged['price_disp'].abs() > 0.005),
    'vol_2x': (merged['vol_ratio'] > 2.0),
    'fr_extreme': (merged['funding_rate'].abs() > 0.0003),
    'bouncing': (merged['Close'] > merged['Close'].shift(2)),
    'oi_fresh': (merged['oi_age_sec'] < 1800),
}

for skip_name in ['none'] + list(conditions.keys()):
    if skip_name == 'none':
        mask = pd.Series(True, index=merged.index)
        label = 'ALL conditions'
    else:
        mask = pd.Series(True, index=merged.index)
        for cname, cond in conditions.items():
            if cname != skip_name:
                mask = mask & cond
        label = f'WITHOUT {skip_name}'
    
    shifted = mask.shift(1).fillna(False)
    events = merged[shifted]
    rets = merged.loc[events.index, 'fwd_ret_16'].dropna()
    if len(rets) < 3:
        print(f"  {label}: n={len(rets)} (too few)")
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
    gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
    a3[f'relax_{skip_name}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'gate': gate}
    print(f"  {'+' if gate=='PASS' else '-'} {label:25s}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}")

results['agent_3'] = a3


# ═══════════════════════════════════════════════════════════════
# AGENT 4: REGIME — Edge by regime
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 4: REGIME — v8 edge by regime")
print("="*70)

a4 = {}

if len(v8_events) > 0:
    for regime_col, regime_name in [('vol_regime', 'Vol Regime'), ('trend', 'Trend'), ('era', 'Era')]:
        print(f"\n  --- {regime_name} ---")
        for regime in sorted(merged[regime_col].dropna().unique()):
            regime_events = v8_events[merged.loc[v8_events.index, regime_col] == regime]
            if len(regime_events) < 3:
                print(f"    {regime}: n={len(regime_events)} (too few)")
                continue
            rets = merged.loc[regime_events.index, 'fwd_ret_16'].dropna()
            if len(rets) < 3:
                continue
            mean_r = rets.mean()
            t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
            gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
            a4[f'{regime_col}_{regime}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'gate': gate}
            print(f"    {'+' if gate=='PASS' else '-'} {regime:12s}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}")

results['agent_4'] = a4


# ═══════════════════════════════════════════════════════════════
# AGENT 5: GATE — Signal frequency & data quality
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 5: GATE — Signal frequency & data quality")
print("="*70)

a5 = {
    'total_signals': len(v8_events),
    'signals_per_month': len(v8_events) / max(1, (merged['timestamp'].max() - merged['timestamp'].min()).days / 30),
    'oi_coverage_fresh': int((merged['oi_age_sec'] < 1800).sum()),
    'oi_coverage_fresh_pct': float((merged['oi_age_sec'] < 1800).mean()),
}
print(f"  Total v8 signals: {len(v8_events)}")
print(f"  Signals per month: {a5['signals_per_month']:.1f}")
print(f"  Fresh OI coverage: {a5['oi_coverage_fresh']} ({a5['oi_coverage_fresh_pct']:.1%})")

results['agent_5'] = a5


# ═══════════════════════════════════════════════════════════════
# AGENT 6: CO-OCCURRENCE — Confluence analysis
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 6: CO-OCCURRENCE — Confluence with other signals")
print("="*70)

a6 = {}

if len(v8_events) > 0:
    # What's the LS ratio during v8 signals?
    ls_during = merged.loc[v8_events.index, 'ls_ratio'].dropna()
    if len(ls_during) > 0:
        a6['ls_ratio_mean'] = float(ls_during.mean())
        print(f"  LS ratio during v8: mean={ls_during.mean():.3f}")
    
    # What's the trend distribution?
    trend_dist = merged.loc[v8_events.index, 'trend'].value_counts()
    a6['trend_distribution'] = trend_dist.to_dict()
    print(f"  Trend distribution: {trend_dist.to_dict()}")
    
    # What's the vol regime distribution?
    vol_dist = merged.loc[v8_events.index, 'vol_regime'].value_counts()
    a6['vol_regime_distribution'] = vol_dist.to_dict()
    print(f"  Vol regime distribution: {vol_dist.to_dict()}")

results['agent_6'] = a6


# ═══════════════════════════════════════════════════════════════
# AGENT 7: SENSITIVITY — Threshold sweep
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 7: SENSITIVITY — Threshold optimization")
print("="*70)

a7 = {}

# Sweep OI drop threshold
print("\n  --- OI drop threshold sweep ---")
for oi_thresh in [0.005, 0.008, 0.01, 0.012, 0.015, 0.02]:
    for vol_thresh in [1.5, 2.0, 3.0]:
        for fr_thresh in [0.0001, 0.0003, 0.0005]:
            mask = (
                (merged['oi_roc'] < -oi_thresh) &
                (merged['vol_ratio'] > vol_thresh) &
                (merged['funding_rate'].abs() > fr_thresh) &
                (merged['Close'] > merged['Close'].shift(2)) &
                (merged['oi_age_sec'] < 1800)
            )
            shifted = mask.shift(1).fillna(False)
            events = merged[shifted]
            rets = merged.loc[events.index, 'fwd_ret_16'].dropna()
            if len(rets) < 5:
                continue
            mean_r = rets.mean()
            t, p = stats.ttest_1samp(rets, 0)
            wr = (rets > 0).mean()
            gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
            if gate == "PASS":
                key = f"OI<{-oi_thresh}_V>{vol_thresh}_FR>{fr_thresh}"
                a7[key] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr)}
                print(f"  + OI<{-oi_thresh:.3f} V>{vol_thresh} FR>{fr_thresh}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_7'] = a7


# ═══════════════════════════════════════════════════════════════
# AGENT 8: MONTE CARLO — Statistical significance
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 8: MONTE CARLO — Statistical significance")
print("="*70)

a8 = {}

if len(v8_events) >= 5:
    actual_rets = merged.loc[v8_events.index, 'fwd_ret_16'].dropna()
    actual_mean = actual_rets.mean()
    actual_wr = (actual_rets > 0).mean()
    n_events = len(actual_rets)
    
    print(f"  Actual v8 events: {n_events}")
    print(f"  Actual mean return: {actual_mean*100:+.4f}%")
    print(f"  Actual WR: {actual_wr:.1%}")
    
    n_sims = 10000
    all_rets = merged['fwd_ret_16'].dropna()
    
    np.random.seed(42)
    random_means = [all_rets.sample(n=n_events, replace=False).mean() for _ in range(n_sims)]
    random_wrs = [(all_rets.sample(n=n_events, replace=False) > 0).mean() for _ in range(n_sims)]
    
    random_means = np.array(random_means)
    random_wrs = np.array(random_wrs)
    
    p_mean = (random_means >= actual_mean).mean()
    p_wr = (random_wrs >= actual_wr).mean()
    
    boot_means = [actual_rets.sample(n=n_events, replace=True).mean() for _ in range(n_sims)]
    boot_wrs = [(actual_rets.sample(n=n_events, replace=True) > 0).mean() for _ in range(n_sims)]
    
    ci_mean_lo = np.percentile(boot_means, 2.5)
    ci_mean_hi = np.percentile(boot_means, 97.5)
    ci_wr_lo = np.percentile(boot_wrs, 2.5)
    ci_wr_hi = np.percentile(boot_wrs, 97.5)
    
    a8 = {
        'n_events': n_events,
        'actual_mean': float(actual_mean),
        'actual_wr': float(actual_wr),
        'mc_p_mean': float(p_mean),
        'mc_p_wr': float(p_wr),
        'bootstrap_ci_mean': [float(ci_mean_lo), float(ci_mean_hi)],
        'bootstrap_ci_wr': [float(ci_wr_lo), float(ci_wr_hi)],
        'significant': p_mean < 0.05,
    }
    
    print(f"  MC p-value (mean): {p_mean:.4f}")
    print(f"  MC p-value (WR): {p_wr:.4f}")
    print(f"  Bootstrap CI (mean): [{ci_mean_lo*100:+.4f}%, {ci_mean_hi*100:+.4f}%]")
    print(f"  Bootstrap CI (WR): [{ci_wr_lo:.1%}, {ci_wr_hi:.1%}]")
    print(f"  SIGNIFICANT: {'YES' if p_mean < 0.05 else 'NO'}")
else:
    a8 = {'n_events': len(v8_events), 'significant': False, 'note': f"Too few events ({len(v8_events)})"}
    print(f"  Too few events ({len(v8_events)}) for Monte Carlo")

results['agent_8'] = a8


# ═══════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

v8_n = a3.get('v8_signals', 0)
mc_sig = a8.get('significant', False)
v8_4h = a3.get('v8_4h', {})

verdict = {
    'strategy': 'S20 Liquidation Mean Reversion',
    'version': 'v8',
    'v8_signals': v8_n,
    'v8_4h_mean': v8_4h.get('mean'),
    'v8_4h_p': v8_4h.get('p'),
    'v8_4h_wr': v8_4h.get('wr'),
    'mc_significant': mc_sig,
}

if v8_n < 10:
    verdict['gate'] = 'INSUFFICIENT_DATA'
    verdict['recommendation'] = 'Deploy provisionally with OI collector upgrade pending'
elif mc_sig:
    verdict['gate'] = 'PASS'
    verdict['recommendation'] = 'Deploy with 0.5x size, validate with 30+ live trades'
elif v8_4h.get('p', 1) < 0.1 and v8_4h.get('mean', 0) > 0.001:
    verdict['gate'] = 'MARGINAL'
    verdict['recommendation'] = 'Deploy with 0.3x size, extend data window'
else:
    verdict['gate'] = 'FAIL'
    verdict['recommendation'] = 'Not viable at current thresholds'

print(f"  v8 signals: {v8_n}")
print(f"  v8 4h mean: {v8_4h.get('mean', 0)*100:+.4f}%")
print(f"  v8 4h p-value: {v8_4h.get('p', 'N/A')}")
print(f"  v8 4h WR: {v8_4h.get('wr', 0):.1%}")
print(f"  MC significant: {mc_sig}")
print(f"  Gate: {verdict['gate']}")
print(f"  Recommendation: {verdict['recommendation']}")

results['verdict'] = verdict

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUTPUT_FILE}")
