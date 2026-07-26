"""
8-Agent Forensic Protocol: S20 Liquidation Cascade
===================================================
Hypothesis: The strategy concept (OI shock + price displacement = cascade) is sound,
but implementation may fail due to: (a) insufficient liquidation events in data,
(b) wrong thresholds, (c) regime filtering, or (d) whale_watch dependency killing signals.

Research basis:
- "Anatomy of a Crypto Cascade" (SSRN 6579278, 2026): Minute-level OI shock + price displacement
- "Explainable Patterns in Crypto Microstructure" (arXiv 2602.00776, 2026)
- "Anatomy of Oct 2025 Liquidation Cascade" (ResearchGate, 2025)

Agents:
1. Forensics     — Data coverage & event identification
2. Non-Indicator — Raw OI shock + price displacement edge test
3. Indicator     — Strategy's actual trigger conditions
4. Regime        — Edge by regime breakdown
5. Gate          — Isolation gate correctness
6. Co-occurrence — What fires alongside cascade
7. Sensitivity   — TP/SL/conviction threshold sweep
8. Monte Carlo   — Statistical significance
"""

import pandas as pd
import numpy as np
from scipy import stats
import json, os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'
OUTPUT_FILE = '/root/.openclaw/workspace/jimi_audit/reports/cascade_8agent_forensic.json'

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
print("Loading data...")

# OHLCV
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
ohlcv['Close'] = ohlcv['Close'].astype(float)
ohlcv['High'] = ohlcv['High'].astype(float)
ohlcv['Low'] = ohlcv['Low'].astype(float)
ohlcv['Volume'] = ohlcv['Volume'].astype(float)
print(f"OHLCV: {len(ohlcv)} bars, {ohlcv['timestamp'].min()} to {ohlcv['timestamp'].max()}")

# Derivatives
deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)
print(f"Derivatives: {len(deriv)} rows")

# Merge
merged = pd.merge_asof(
    ohlcv[['timestamp','Open','High','Low','Close','Volume']],
    deriv[['timestamp','oi','ls_ratio','funding_rate']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)

# Compute features
merged['oi_roc'] = merged['oi'].pct_change(4, fill_method=None)  # 1h OI ROC
merged['oi_roc_4h'] = merged['oi'].pct_change(16, fill_method=None)  # 4h OI ROC
merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')
merged['atr'] = (merged['High'] - merged['Low']).rolling(14).mean()
merged['atr_pct'] = merged['atr'] / merged['Close']
merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()

# Price displacement (5-bar = 75min)
merged['price_disp'] = merged['Close'].pct_change(5)
merged['price_disp_abs'] = merged['price_disp'].abs()

# Forward returns
for h in [1, 4, 16, 24]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

# Regime (vol-based)
vols = merged['vol_20bar'].dropna()
p33, p67 = vols.quantile(0.33), vols.quantile(0.67)
merged['vol_regime'] = 'MID'
merged.loc[merged['vol_20bar'] < p33, 'vol_regime'] = 'LOW'
merged.loc[merged['vol_20bar'] > p67, 'vol_regime'] = 'HIGH'

# Calendar eras
def get_era(ts):
    if ts < pd.Timestamp('2024-07-01'): return '2024_H1'
    elif ts < pd.Timestamp('2025-01-01'): return '2024_H2'
    elif ts < pd.Timestamp('2025-07-01'): return '2025_H1'
    elif ts < pd.Timestamp('2026-01-01'): return '2025_H2'
    else: return '2026'
merged['era'] = merged['timestamp'].apply(get_era)

# July 2026 filter
merged['is_july_2026'] = (merged['timestamp'] >= '2026-07-01') & (merged['timestamp'] < '2026-08-01')

round_trip_cost = 0.0010  # 0.10%
results = {}

print(f"Merged: {len(merged)} rows, OI coverage: {merged['oi'].notna().sum()} ({100*merged['oi'].notna().mean():.1f}%)")


# ═══════════════════════════════════════════════════════════════
# AGENT 1: FORENSICS — Data coverage & event identification
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 1: FORENSICS — Data coverage & event identification")
print("="*70)

a1 = {}
oi_valid = merged['oi'].notna().sum()
fr_valid = merged['funding_rate'].notna().sum()
ls_valid = merged['ls_ratio'].notna().sum()

a1['oi_coverage'] = f"{oi_valid}/{len(merged)} ({100*oi_valid/len(merged):.1f}%)"
a1['fr_coverage'] = f"{fr_valid}/{len(merged)} ({100*fr_valid/len(merged):.1f}%)"
a1['ls_coverage'] = f"{ls_valid}/{len(merged)} ({100*ls_valid/len(merged):.1f}%)"

# OI ROC distribution
oi_roc = merged['oi_roc'].dropna()
a1['oi_roc_mean'] = float(oi_roc.mean())
a1['oi_roc_std'] = float(oi_roc.std())
for thresh in [0.005, 0.01, 0.015, 0.02, 0.03]:
    n = (oi_roc > thresh).sum()
    a1[f'oi_roc_gt_{thresh}'] = int(n)
    print(f"  OI ROC > {thresh}: {n} events")

# Price displacement distribution
price_disp = merged['price_disp_abs'].dropna()
a1['price_disp_mean'] = float(price_disp.mean())
for thresh in [0.003, 0.005, 0.01, 0.015, 0.02]:
    n = (price_disp > thresh).sum()
    a1[f'price_disp_gt_{thresh}'] = int(n)
    print(f"  |Price disp| > {thresh}: {n} events")

# July 2026 specific
july = merged[merged['is_july_2026']]
a1['july_2026_bars'] = len(july)
a1['july_2026_oi_coverage'] = int(july['oi'].notna().sum())
print(f"  July 2026: {len(july)} bars, OI coverage: {july['oi'].notna().sum()}")

results['agent_1_forensics'] = a1


# ═══════════════════════════════════════════════════════════════
# AGENT 2: NON-INDICATOR — Raw OI shock + price displacement
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 2: NON-INDICATOR — Raw cascade signal edge")
print("="*70)

a2 = {}

# Define "cascade event": OI surge + price displacement in same direction
cascade_configs = [
    ('OI_surge_1pct_price_0.5pct', 0.01, 0.005),
    ('OI_surge_1.5pct_price_0.5pct', 0.015, 0.005),
    ('OI_surge_2pct_price_1pct', 0.02, 0.01),
    ('OI_surge_1pct_price_1pct', 0.01, 0.01),
    ('OI_surge_0.5pct_price_0.3pct', 0.005, 0.003),
    ('OI_drop_1pct_price_0.5pct', -0.01, 0.005),  # OI drop = liquidations happening
]

for name, oi_thresh, price_thresh in cascade_configs:
    if oi_thresh > 0:
        mask = (merged['oi_roc'] > oi_thresh) & (merged['price_disp_abs'] > price_thresh)
    else:
        mask = (merged['oi_roc'] < oi_thresh) & (merged['price_disp_abs'] > price_thresh)
    
    shifted = mask.shift(1).fillna(False)
    events = merged[shifted]
    
    for h, label in [(1, '15m'), (4, '1h'), (16, '4h')]:
        rets = merged.loc[events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 3:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and abs(mean_r) > round_trip_cost else "FAIL"
        dir_label = "L" if mean_r > 0 else "S"
        key = f"{name}_{label}"
        a2[key] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'dir': dir_label, 'gate': gate}
        symbol = '+' if gate == 'PASS' else '-'
        print(f"  {symbol} {name:35s} {label}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}, dir={dir_label}")

# Also test: OI drop (liquidation happening) vs OI surge (new positions)
print("\n  --- OI DIRECTION BREAKDOWN ---")
for oi_dir, label in [('surge', 'OI surge (new positions)'), ('drop', 'OI drop (liquidations)')]:
    if oi_dir == 'surge':
        mask = merged['oi_roc'] > 0.01
    else:
        mask = merged['oi_roc'] < -0.01
    
    shifted = mask.shift(1).fillna(False)
    events = merged[shifted]
    for h, hlabel in [(4, '1h'), (16, '4h')]:
        rets = merged.loc[events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 5:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        gate = "PASS" if p < 0.1 and abs(mean_r) > round_trip_cost else "FAIL"
        a2[f'{oi_dir}_{hlabel}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} {label:30s} {hlabel}: n={len(rets):5d}, mean={mean_r*100:+.4f}%, p={p:.4f}")

results['agent_2_non_indicator'] = a2


# ═══════════════════════════════════════════════════════════════
# AGENT 3: INDICATOR — Strategy's actual trigger conditions
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 3: INDICATOR — Replicating S20 trigger logic")
print("="*70)

a3 = {}

# Replicate _check_long_cascade logic
def check_long_cascade(row, regime):
    """Replicate S20's LONG cascade detection."""
    oi_roc = row.get('oi_roc', 0) or 0
    ls_ratio = row.get('ls_ratio', 1.0) or 1.0
    price_change = row.get('price_disp', 0) or 0
    
    if regime in ("STRESS", "BEAR"):
        oi_surge_threshold = 0.012
        ls_short_threshold = 0.8
    elif regime == "BULL":
        oi_surge_threshold = 0.020
        ls_short_threshold = 0.6
    else:
        oi_surge_threshold = 0.015
        ls_short_threshold = 0.7
    
    # OI must surge
    if oi_roc <= oi_surge_threshold:
        return False
    # LS ratio must indicate shorts crowded (low ratio = more shorts)
    if ls_ratio >= ls_short_threshold:
        return False
    # Price must not be crashing (counter-trend)
    if price_change < -0.005:
        return False
    return True

# Test each row
cascade_signals = []
for idx, row in merged.iterrows():
    regime = row.get('trend', 'RANGING')  # Use trend as proxy for regime
    if check_long_cascade(row, regime):
        cascade_signals.append(idx)

cascade_events = merged.loc[cascade_signals]
a3['total_cascade_signals'] = len(cascade_events)
print(f"  Total cascade signals (LONG): {len(cascade_events)}")

if len(cascade_events) > 0:
    for h, label in [(1, '15m'), (4, '1h'), (16, '4h')]:
        rets = merged.loc[cascade_events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 3:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        a3[f's20_trigger_{label}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} S20 trigger {label}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# July 2026 specific
july_cascade = cascade_events[cascade_events['is_july_2026']]
a3['july_2026_signals'] = len(july_cascade)
print(f"  July 2026 cascade signals: {len(july_cascade)}")

# Why no signals? Check which condition fails most
print("\n  --- CONDITION FAILURE ANALYSIS ---")
fail_counts = {'oi_too_low': 0, 'ls_wrong': 0, 'price_crashing': 0, 'all_pass': 0}
for idx, row in merged.iterrows():
    oi_roc = row.get('oi_roc', 0) or 0
    ls_ratio = row.get('ls_ratio', 1.0) or 1.0
    price_change = row.get('price_disp', 0) or 0
    
    if oi_roc <= 0.015:
        fail_counts['oi_too_low'] += 1
    elif ls_ratio >= 0.7:
        fail_counts['ls_wrong'] += 1
    elif price_change < -0.005:
        fail_counts['price_crashing'] += 1
    else:
        fail_counts['all_pass'] += 1

a3['condition_failures'] = fail_counts
for k, v in fail_counts.items():
    print(f"  {k}: {v} ({100*v/len(merged):.1f}%)")

results['agent_3_indicator'] = a3


# ═══════════════════════════════════════════════════════════════
# AGENT 4: REGIME — Edge by regime breakdown
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 4: REGIME — Edge by regime breakdown")
print("="*70)

a4 = {}

# Use the cascade signals from Agent 3
if len(cascade_events) > 0:
    for regime_col, regime_name in [('vol_regime', 'Vol Regime'), ('trend', 'Trend'), ('era', 'Era')]:
        print(f"\n  --- {regime_name} ---")
        for regime in sorted(merged[regime_col].dropna().unique()):
            regime_events = cascade_events[cascade_events[regime_col] == regime]
            if len(regime_events) < 3:
                print(f"    {regime}: n={len(regime_events)} (too few)")
                continue
            for h, label in [(4, '1h'), (16, '4h')]:
                rets = merged.loc[regime_events.index, f'fwd_ret_{h}'].dropna()
                if len(rets) < 3:
                    continue
                mean_r = rets.mean()
                t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
                gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
                key = f"{regime_col}_{regime}_{label}"
                a4[key] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'gate': gate}
                print(f"    {'+' if gate=='PASS' else '-'} {regime:12s} {label}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}")

results['agent_4_regime'] = a4


# ═══════════════════════════════════════════════════════════════
# AGENT 5: GATE — Isolation gate correctness
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 5: GATE — Whale watch dependency analysis")
print("="*70)

a5 = {}

# The strategy requires whale_watch confirmation. How often does whale_watch fire?
# We can't directly check strategy_signals from historical data, but we can check
# the conditions that would trigger whale_watch

# Whale watch fires on: large trades, on-chain whale activity
# Since we don't have that data, we check: how many cascade events would be lost
# due to whale_watch dependency?

print("  Whale watch dependency: REQUIRED for S20 to fire")
print("  This means cascade signals WITHOUT whale confirmation are DISCARDED")

# Check: how often do cascade events coincide with price being near EMA200?
# (whale_watch tends to fire when price is near EMA200 or at key levels)
if len(cascade_events) > 0:
    ema200_dist = abs(cascade_events['Close'] - cascade_events['ema200']) / cascade_events['ema200']
    a5['cascade_ema200_dist_mean'] = float(ema200_dist.mean())
    a5['cascade_ema200_dist_median'] = float(ema200_dist.median())
    print(f"  Cascade events EMA200 distance: mean={ema200_dist.mean():.2%}, median={ema200_dist.median():.2%}")
    
    # What % of cascade events are near EMA200 (within 2%)?
    near_ema200 = (ema200_dist < 0.02).sum()
    a5['cascade_near_ema200'] = int(near_ema200)
    a5['cascade_near_ema200_pct'] = float(near_ema200 / len(cascade_events))
    print(f"  Cascade events within 2% of EMA200: {near_ema200}/{len(cascade_events)} ({100*near_ema200/len(cascade_events):.1f}%)")

# The real question: how many cascade events exist at ALL vs how many the strategy sees
a5['raw_cascade_events'] = len(cascade_events)
a5['strategy_sees'] = 'unknown (whale_watch data not available)'
print(f"  Raw cascade events: {len(cascade_events)}")
print(f"  Strategy sees: unknown (whale_watch confirmation data not available)")

results['agent_5_gate'] = a5


# ═══════════════════════════════════════════════════════════════
# AGENT 6: CO-OCCURRENCE — What fires alongside cascade
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 6: CO-OCCURRENCE — Cascade + other signals")
print("="*70)

a6 = {}

if len(cascade_events) > 0:
    # High funding rate during cascade
    fr_during = cascade_events['funding_rate'].dropna()
    if len(fr_during) > 0:
        a6['cascade_funding_mean'] = float(fr_during.mean())
        a6['cascade_funding_median'] = float(fr_during.median())
        print(f"  Funding during cascade: mean={fr_during.mean():.6f}, median={fr_during.median():.6f}")
    
    # LS ratio during cascade
    ls_during = cascade_events['ls_ratio'].dropna()
    if len(ls_during) > 0:
        a6['cascade_ls_mean'] = float(ls_during.mean())
        print(f"  LS ratio during cascade: mean={ls_during.mean():.3f}")
    
    # Volume during cascade
    vol_during = cascade_events['vol_ratio'].dropna()
    if len(vol_during) > 0:
        a6['cascade_vol_ratio_mean'] = float(vol_during.mean())
        print(f"  Vol ratio during cascade: mean={vol_during.mean():.2f}x")
    
    # Confluence: cascade + high funding + extreme LS
    for fr_thresh, ls_thresh, name in [
        (0.0005, 1.5, 'FR>0.0005+LS>1.5'),
        (0.001, 2.0, 'FR>0.001+LS>2.0'),
        (-0.0005, 0.67, 'FR<-0.0005+LS<0.67'),
    ]:
        if fr_thresh > 0:
            combo = cascade_events[(cascade_events['funding_rate'] > fr_thresh) & (cascade_events['ls_ratio'] > ls_thresh)]
        else:
            combo = cascade_events[(cascade_events['funding_rate'] < fr_thresh) & (cascade_events['ls_ratio'] < ls_thresh)]
        
        if len(combo) < 3:
            print(f"  {name}: n={len(combo)} (too few)")
            continue
        
        rets = merged.loc[combo.index, 'fwd_ret_16'].dropna()
        if len(rets) < 3:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        a6[f'combo_{name}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} {name}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}")

results['agent_6_cooccurrence'] = a6


# ═══════════════════════════════════════════════════════════════
# AGENT 7: SENSITIVITY — Threshold sweep
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 7: SENSITIVITY — OI ROC + LS ratio threshold sweep")
print("="*70)

a7 = {}

# Sweep OI ROC threshold
print("\n  --- OI ROC threshold sweep (LONG cascade) ---")
for oi_thresh in [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03]:
    for ls_thresh in [0.5, 0.6, 0.7, 0.8]:
        mask = (merged['oi_roc'] > oi_thresh) & (merged['ls_ratio'] < ls_thresh) & (merged['price_disp'] > -0.005)
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
            key = f"OI>{oi_thresh}_LS<{ls_thresh}"
            a7[key] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr)}
            print(f"  + OI>{oi_thresh:.3f} LS<{ls_thresh}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# Sweep conviction threshold (strength-based)
print("\n  --- Conviction threshold sweep ---")
if len(cascade_events) > 0:
    # Recompute strength for cascade events
    for conv_min in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        # Strength = min(abs(oi_roc) * 8, 0.8) + 0.45
        # conv >= conv_min means strength >= (conv_min - 0.45) / 0.40
        strength_min = max(0, (conv_min - 0.45) / 0.40)
        filtered = cascade_events[(cascade_events['oi_roc'].abs() * 8).clip(upper=0.8) >= strength_min]
        if len(filtered) < 3:
            continue
        rets = merged.loc[filtered.index, 'fwd_ret_16'].dropna()
        if len(rets) < 3:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        key = f"conv>={conv_min}"
        a7[key] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} conv>={conv_min}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}")

results['agent_7_sensitivity'] = a7


# ═══════════════════════════════════════════════════════════════
# AGENT 8: MONTE CARLO — Statistical significance
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 8: MONTE CARLO — Statistical significance")
print("="*70)

a8 = {}

if len(cascade_events) >= 5:
    actual_rets = merged.loc[cascade_events.index, 'fwd_ret_16'].dropna()
    actual_mean = actual_rets.mean()
    actual_wr = (actual_rets > 0).mean()
    n_events = len(actual_rets)
    
    print(f"  Actual cascade events: {n_events}")
    print(f"  Actual mean return: {actual_mean*100:+.4f}%")
    print(f"  Actual WR: {actual_wr:.1%}")
    
    # Monte Carlo: shuffle labels 10000 times
    n_sims = 10000
    all_rets = merged['fwd_ret_16'].dropna()
    random_means = []
    random_wrs = []
    
    np.random.seed(42)
    for _ in range(n_sims):
        sample = all_rets.sample(n=n_events, replace=False)
        random_means.append(sample.mean())
        random_wrs.append((sample > 0).mean())
    
    random_means = np.array(random_means)
    random_wrs = np.array(random_wrs)
    
    # p-value: fraction of random means >= actual mean
    p_mean = (random_means >= actual_mean).mean()
    p_wr = (random_wrs >= actual_wr).mean()
    
    # Bootstrap CI
    boot_means = []
    boot_wrs = []
    for _ in range(n_sims):
        sample = actual_rets.sample(n=n_events, replace=True)
        boot_means.append(sample.mean())
        boot_wrs.append((sample > 0).mean())
    
    boot_means = np.array(boot_means)
    boot_wrs = np.array(boot_wrs)
    
    ci_mean_lo = np.percentile(boot_means, 2.5)
    ci_mean_hi = np.percentile(boot_means, 97.5)
    ci_wr_lo = np.percentile(boot_wrs, 2.5)
    ci_wr_hi = np.percentile(boot_wrs, 97.5)
    
    a8['n_events'] = n_events
    a8['actual_mean'] = float(actual_mean)
    a8['actual_wr'] = float(actual_wr)
    a8['mc_p_mean'] = float(p_mean)
    a8['mc_p_wr'] = float(p_wr)
    a8['bootstrap_ci_mean'] = [float(ci_mean_lo), float(ci_mean_hi)]
    a8['bootstrap_ci_wr'] = [float(ci_wr_lo), float(ci_wr_hi)]
    a8['significant'] = p_mean < 0.05
    
    print(f"  MC p-value (mean): {p_mean:.4f}")
    print(f"  MC p-value (WR): {p_wr:.4f}")
    print(f"  Bootstrap CI (mean): [{ci_mean_lo*100:+.4f}%, {ci_mean_hi*100:+.4f}%]")
    print(f"  Bootstrap CI (WR): [{ci_wr_lo:.1%}, {ci_wr_hi:.1%}]")
    print(f"  SIGNIFICANT: {'YES' if p_mean < 0.05 else 'NO'}")
else:
    a8['n_events'] = len(cascade_events)
    a8['significant'] = False
    a8['note'] = f"Too few events ({len(cascade_events)}) for Monte Carlo"
    print(f"  Too few events ({len(cascade_events)}) for Monte Carlo")

results['agent_8_monte_carlo'] = a8


# ═══════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

# Summarize
total_signals = a3.get('total_cascade_signals', 0)
july_signals = a3.get('july_2026_signals', 0)
mc_sig = a8.get('significant', False)

verdict = {
    'strategy': 'S20 Liquidation Cascade',
    'version': 'v6.1 PROVISIONAL',
    'total_cascade_signals': total_signals,
    'july_2026_signals': july_signals,
    'mc_significant': mc_sig,
    'research_papers': [
        'Anatomy of a Crypto Cascade: Minute-Level Evidence (SSRN 6579278, 2026)',
        'Explainable Patterns in Crypto Microstructure (arXiv 2602.00776, 2026)',
        'Anatomy of Oct 2025 Liquidation Cascade (ResearchGate, 2025)',
    ],
}

if total_signals < 10:
    verdict['diagnosis'] = 'INSUFFICIENT DATA: Too few cascade events in historical data to validate edge'
    verdict['recommendation'] = 'KILL or rework with lower thresholds + remove whale_watch dependency'
elif not mc_sig:
    verdict['diagnosis'] = 'NOT SIGNIFICANT: Cascade signal exists but edge is not statistically confirmed'
    verdict['recommendation'] = 'Lower thresholds, extend data window, or KILL'
else:
    verdict['diagnosis'] = 'EDGE EXISTS: Cascade signal is statistically significant'
    verdict['recommendation'] = 'Deploy with current thresholds, monitor for regime drift'

print(f"  Total cascade signals: {total_signals}")
print(f"  July 2026 signals: {july_signals}")
print(f"  MC significant: {mc_sig}")
print(f"  Diagnosis: {verdict['diagnosis']}")
print(f"  Recommendation: {verdict['recommendation']}")

results['verdict'] = verdict

# Save
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUTPUT_FILE}")
