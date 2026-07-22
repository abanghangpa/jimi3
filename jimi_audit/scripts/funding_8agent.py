"""
8-Agent Protocol: Backwards funding signal investigation.
From cascade Agent 8: funding_rate > 0.0005 → -0.396% at 4h (p=0.0004, n=222).
Hypothesis: funding extreme predicts CONTINUATION, not reversal.
"""
import pandas as pd
import numpy as np
from scipy import stats
import json, os

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'

# Load OHLCV
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
print(f"OHLCV: {len(ohlcv)} bars")

# Load derivatives
deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

merged = pd.merge_asof(ohlcv, deriv[['timestamp','oi','ls_ratio','funding_rate']],
                       on='timestamp', direction='backward', tolerance=pd.Timedelta('2h'))

merged['oi_roc'] = merged['oi'].pct_change(4, fill_method=None)
merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')

for h in [1, 4, 16, 24]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

# ═══════════════════════════════════════════════════════
# AGENT 1: FORENSICS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 1: FORENSICS — Backwards funding signal")
print("="*70)

fr = merged['funding_rate'].dropna()
print(f"Funding rate coverage: {len(fr)}/{len(merged)} ({100*len(fr)/len(merged):.1f}%)")
print(f"Funding range: {fr.min():.6f} to {fr.max():.6f}")
print(f"Funding mean: {fr.mean():.6f}")
print(f"Funding std: {fr.std():.6f}")

# Check distribution
for thresh in [0.0001, 0.0003, 0.0005, 0.001, 0.002]:
    n_pos = (fr > thresh).sum()
    n_neg = (fr < -thresh).sum()
    print(f"  |FR| > {thresh}: {n_pos} positive, {n_neg} negative")

# ═══════════════════════════════════════════════════════
# AGENT 2: NON-INDICATOR — Test raw funding extremes
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 2: NON-INDICATOR — Raw funding extreme detection")
print("="*70)

configs = [
    ('FR > 0.0003', merged['funding_rate'] > 0.0003),
    ('FR > 0.0005', merged['funding_rate'] > 0.0005),
    ('FR > 0.001', merged['funding_rate'] > 0.001),
    ('FR > 0.002', merged['funding_rate'] > 0.002),
    ('FR < -0.0003', merged['funding_rate'] < -0.0003),
    ('FR < -0.0005', merged['funding_rate'] < -0.0005),
    ('FR < -0.001', merged['funding_rate'] < -0.001),
    ('FR < -0.002', merged['funding_rate'] < -0.002),
    ('|FR| > 0.0005', merged['funding_rate'].abs() > 0.0005),
    ('|FR| > 0.001', merged['funding_rate'].abs() > 0.001),
]

print("\nTesting at 4h (16-bar) horizon:")
for name, mask in configs:
    shifted = mask.shift(1).fillna(False)
    events = merged[shifted]
    rets = merged.loc[events.index, 'fwd_ret_16'].dropna()
    if len(rets) < 5:
        print(f"  {name}: n={len(rets)} (too few)")
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    gate = "PASS" if p < 0.1 and mean_r > 0.001 else "FAIL"
    dir_label = "LONG" if mean_r > 0 else "SHORT"
    print(f"  {'+' if gate=='PASS' else '-'} {name:20s} n={len(rets):5d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}, dir={dir_label} [{gate}]")

# ═══════════════════════════════════════════════════════
# AGENT 3: COST GATE + AGENT 4: SAMPLE SIZE
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 3: COST GATE + AGENT 4: SAMPLE SIZE")
print("="*70)

round_trip_cost = 0.0010

# Test the backwards signal with direction-aware logic
# Original finding: FR > 0.0005 → -0.396% (BACKWARDS)
# This means: high funding = price drops. So the signal is SHORT.
for direction_filter, dir_name in [
    (merged['funding_rate'] > 0.0005, 'HIGH FR (LONGS PAYING)'),
    (merged['funding_rate'] < -0.0005, 'LOW FR (SHORTS PAYING)'),
]:
    shifted = direction_filter.shift(1).fillna(False)
    events = merged[shifted]
    print(f"\n--- {dir_name} ({len(events)} events) ---")
    for h, label in [(1, '15m'), (4, '1h'), (16, '4h'), (24, '6h')]:
        rets = merged.loc[events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 5:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and abs(mean_r) > round_trip_cost else "FAIL"
        dir_label = "LONG" if mean_r > 0 else "SHORT"
        print(f"  {'+' if gate=='PASS' else '-'} {label}: n={len(rets):5d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}, dir={dir_label} [{gate}]")

# ═══════════════════════════════════════════════════════
# AGENT 5: STRESS TEST
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 5: STRESS TEST — Funding thresholds + filters")
print("="*70)

stress_configs = [
    ('FR>0.0003', 0.0003),
    ('FR>0.0005', 0.0005),
    ('FR>0.0007', 0.0007),
    ('FR>0.001', 0.001),
    ('FR>0.0015', 0.0015),
    ('FR>0.002', 0.002),
]

for name, thresh in stress_configs:
    mask = (merged['funding_rate'] > thresh).shift(1).fillna(False)
    events = merged[mask]
    for h, label in [(4, '1h'), (16, '4h'), (24, '6h')]:
        rets = merged.loc[events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 5:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        gate = "PASS" if p < 0.1 and abs(mean_r) > round_trip_cost else "FAIL"
        dir_label = "L" if mean_r > 0 else "S"
        print(f"  {'+' if gate=='PASS' else '-'} {name:12s} {label}: n={len(rets):5d}, mean={mean_r*100:+.4f}%, p={p:.4f}, dir={dir_label}")

# ═══════════════════════════════════════════════════════
# AGENT 6: REGIME TESTER
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 6: REGIME TESTER")
print("="*70)

vols = merged['vol_20bar'].dropna()
p33, p67 = vols.quantile(0.33), vols.quantile(0.67)
merged['vol_regime'] = 'MID'
merged.loc[merged['vol_20bar'] < p33, 'vol_regime'] = 'LOW'
merged.loc[merged['vol_20bar'] > p67, 'vol_regime'] = 'HIGH'

# High funding signal
hf_mask = (merged['funding_rate'] > 0.0005).shift(1).fillna(False)

for regime_col, regime_name in [('vol_regime', 'Vol Regime'), ('trend', 'Trend')]:
    print(f"\n--- {regime_name} (FR > 0.0005) ---")
    for regime in sorted(merged[regime_col].dropna().unique()):
        regime_events = merged[hf_mask & (merged[regime_col] == regime)]
        if len(regime_events) < 5:
            print(f"  {regime}: n={len(regime_events)} (too few)")
            continue
        for h, label in [(4, '1h'), (16, '4h')]:
            rets = merged.loc[regime_events.index, f'fwd_ret_{h}'].dropna()
            if len(rets) < 5:
                continue
            mean_r = rets.mean()
            t, p = stats.ttest_1samp(rets, 0)
            gate = "PASS" if p < 0.1 and abs(mean_r) > round_trip_cost else "FAIL"
            dir_label = "L" if mean_r > 0 else "S"
            print(f"  {'+' if gate=='PASS' else '-'} {regime:12s} {label}: n={len(rets):5d}, mean={mean_r*100:+.4f}%, p={p:.4f}, dir={dir_label}")

# Calendar era
def get_era(ts):
    if ts < pd.Timestamp('2024-07-01'): return '2024_H1'
    elif ts < pd.Timestamp('2025-01-01'): return '2024_H2'
    elif ts < pd.Timestamp('2025-07-01'): return '2025_H1'
    elif ts < pd.Timestamp('2026-01-01'): return '2025_H2'
    else: return '2026'

merged['era'] = merged['timestamp'].apply(get_era)
print(f"\n--- Calendar Era (FR > 0.0005) ---")
for era in sorted(merged['era'].unique()):
    era_events = merged[hf_mask & (merged['era'] == era)]
    if len(era_events) < 5:
        print(f"  {era}: n={len(era_events)} (too few)")
        continue
    for h, label in [(4, '1h'), (16, '4h')]:
        rets = merged.loc[era_events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 5:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        gate = "PASS" if p < 0.1 and abs(mean_r) > round_trip_cost else "FAIL"
        dir_label = "L" if mean_r > 0 else "S"
        print(f"  {'+' if gate=='PASS' else '-'} {era:10s} {label}: n={len(rets):5d}, mean={mean_r*100:+.4f}%, p={p:.4f}, dir={dir_label}")

# ═══════════════════════════════════════════════════════
# AGENT 7: CONFLUENCE
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 7: CONFLUENCE — Funding + other filters")
print("="*70)

hf_events = merged[hf_mask]
base_rets = merged.loc[hf_events.index, 'fwd_ret_16'].dropna()
if len(base_rets) > 5:
    base_mean = base_rets.mean()
    print(f"Base (FR>0.0005): n={len(base_rets)}, mean={base_mean*100:+.4f}%")

filters = [
    ('+ LS > 1.5', merged['ls_ratio'] > 1.5),
    ('+ LS > 2.0', merged['ls_ratio'] > 2.0),
    ('+ LS < 0.67', merged['ls_ratio'] < 0.67),
    ('+ OI ROC < -0.01', merged['oi_roc'] < -0.01),
    ('+ OI ROC > 0.01', merged['oi_roc'] > 0.01),
    ('+ BULL trend', merged['trend'] == 'BULL'),
    ('+ BEAR trend', merged['trend'] == 'BEAR'),
    ('+ LOW vol', merged['vol_regime'] == 'LOW'),
    ('+ MID vol', merged['vol_regime'] == 'MID'),
    ('+ HIGH vol', merged['vol_regime'] == 'HIGH'),
    ('+ FR > 0.001', merged['funding_rate'] > 0.001),
    ('+ FR > 0.002', merged['funding_rate'] > 0.002),
]

for name, filt in filters:
    filtered = hf_events[filt.loc[hf_events.index]]
    if len(filtered) < 5:
        print(f"  {name}: n={len(filtered)} (too few)")
        continue
    rets = merged.loc[filtered.index, 'fwd_ret_16'].dropna()
    if len(rets) < 5:
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    delta = mean_r - base_mean
    gate = "PASS" if p < 0.1 and abs(mean_r) > 0.001 else "FAIL"
    dir_label = "L" if mean_r > 0 else "S"
    print(f"  {'+' if gate=='PASS' else '-'} {name:20s} n={len(rets):5d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}, delta={delta*100:+.4f}%, dir={dir_label}")

# ═══════════════════════════════════════════════════════
# AGENT 8: ALTERNATIVE DETECTION
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 8: ALTERNATIVE DETECTION — Inverse funding")
print("="*70)

# The backwards result: FR > 0.0005 → price DROPS
# This means: high funding = overleveraged longs → squeeze → SHORT
# Test: does FLIPPING the signal work? (SHORT when FR high)

alt_configs = [
    ('FR > 0.0005 → SHORT', (merged['funding_rate'] > 0.0005), 'SHORT'),
    ('FR > 0.001 → SHORT', (merged['funding_rate'] > 0.001), 'SHORT'),
    ('FR < -0.0005 → LONG', (merged['funding_rate'] < -0.0005), 'LONG'),
    ('FR < -0.001 → LONG', (merged['funding_rate'] < -0.001), 'LONG'),
    ('FR z-score > 2 → SHORT', ((merged['funding_rate'] - merged['funding_rate'].rolling(96).mean()) / merged['funding_rate'].rolling(96).std() > 2), 'SHORT'),
    ('FR z-score < -2 → LONG', ((merged['funding_rate'] - merged['funding_rate'].rolling(96).mean()) / merged['funding_rate'].rolling(96).std() < -2), 'LONG'),
    ('FR divergence (FR up, price down)', (merged['funding_rate'].diff(4) > 0.0003) & (merged['Close'].pct_change(4) < -0.005), 'SHORT'),
    ('FR convergence (FR down, price up)', (merged['funding_rate'].diff(4) < -0.0003) & (merged['Close'].pct_change(4) > 0.005), 'LONG'),
]

print("\nTesting at 4h (16-bar) horizon:")
for name, mask, expected_dir in alt_configs:
    shifted = mask.shift(1).fillna(False)
    events = merged[shifted]
    rets = merged.loc[events.index, 'fwd_ret_16'].dropna()
    if len(rets) < 5:
        print(f"  {name}: n={len(rets)} (too few)")
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()

    # For SHORT signals, negative return = win
    if expected_dir == 'SHORT':
        effective_mean = -mean_r  # flip for SHORT
        effective_wr = 1 - wr
    else:
        effective_mean = mean_r
        effective_wr = wr

    gate = "PASS" if p < 0.1 and effective_mean > 0.001 else "FAIL"
    print(f"  {'+' if gate=='PASS' else '-'} {name:40s} n={len(rets):5d}, mean={mean_r*100:+.4f}%, eff_mean={effective_mean*100:+.4f}%, p={p:.4f}, eff_WR={effective_wr:.1%} [{gate}]")

# ═══════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)
print("Original finding: FR > 0.0005 → -0.396% at 4h (p=0.0004, n=222)")
print("Interpretation: High funding = overleveraged longs = SHORT opportunity")
print("This is NOT a backwards result — it's a correctly-directed SHORT signal.")
print("Protocol: 8-Agent complete — see results above")
