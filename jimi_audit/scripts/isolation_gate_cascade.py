"""
8-Agent Protocol: Isolation Gate for liquidation_cascade
Agents 2-4: Raw detection, cost gate, sample size

Tests the cascade mechanism on REAL data with NO optimization.
Hypothesis: OI crash + price move = cascading liquidations predict continuation.
"""
import pandas as pd
import numpy as np
from scipy import stats
import json
import os

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
REPORT_DIR = '/root/.openclaw/workspace/jimi_audit/reports'
os.makedirs(REPORT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════
# STEP 1: Load data
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("AGENT 1 (Forensics) — Data Loading")
print("=" * 60)

# Load OHLCV
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
print(f"OHLCV: {len(ohlcv)} bars, {ohlcv['timestamp'].min()} to {ohlcv['timestamp'].max()}")

# Check for gaps
ohlcv['gap'] = ohlcv['timestamp'].diff()
gaps = ohlcv[ohlcv['gap'] > pd.Timedelta('20min')]
if len(gaps) > 0:
    print(f"⚠️ {len(gaps)} gaps > 20min found")
    for _, g in gaps.head(5).iterrows():
        print(f"   Gap at {g['timestamp']}: {g['gap']}")
else:
    print("✅ No significant gaps")

# Check for zero/negative prices
bad = ohlcv[(ohlcv['Close'] <= 0) | (ohlcv['Volume'] < 0)]
print(f"{'⚠️' if len(bad) > 0 else '✅'} Bad rows (zero price/neg volume): {len(bad)}")

# Load derivatives (backfilled - longer history)
deriv = pd.read_csv(f'{DATA_DIR}/derivatives_history/derivatives_backfilled.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'])
deriv = deriv.sort_values('timestamp').reset_index(drop=True)
print(f"Derivatives (backfilled): {len(deriv)} rows, {deriv['timestamp'].min()} to {deriv['timestamp'].max()}")

# Load derivatives (collected - higher quality, shorter)
deriv_col = pd.read_csv(f'{DATA_DIR}/derivatives_history/derivatives_collected.csv')
deriv_col['timestamp'] = pd.to_datetime(deriv_col['timestamp'], format='mixed')
deriv_col = deriv_col.sort_values('timestamp').reset_index(drop=True)
print(f"Derivatives (collected): {len(deriv_col)} rows, {deriv_col['timestamp'].min()} to {deriv_col['timestamp'].max()}")

# Load OI history from forced_movement
oi_hist = pd.read_csv(f'{DATA_DIR}/forced_movement/oi_history.csv', header=None,
                       names=['timestamp_ms', 'oi', 'oi_usd'])
oi_hist['timestamp'] = pd.to_datetime(oi_hist['timestamp_ms'], unit='ms')
oi_hist = oi_hist.sort_values('timestamp').reset_index(drop=True)
print(f"OI history: {len(oi_hist)} rows")

# ═══════════════════════════════════════════════════════
# STEP 2: Merge derivatives onto OHLCV
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("AGENT 1 (Forensics) — Data Merge & Quality")
print("=" * 60)

# Merge backfilled derivatives onto OHLCV (asof merge on timestamp)
ohlcv = ohlcv.sort_values('timestamp')
deriv = deriv.sort_values('timestamp')

merged = pd.merge_asof(
    ohlcv, deriv[['timestamp', 'oi', 'funding_rate', 'ls_ratio', 'long_pct', 'short_pct']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)

# Also merge collected derivatives (has taker_ratio)
deriv_col_subset = deriv_col[['timestamp', 'oi', 'funding_rate', 'ls_ratio',
                               'futures_taker_ratio', 'futures_buy_vol', 'futures_sell_vol']].rename(
    columns={'oi': 'oi_collected', 'funding_rate': 'funding_collected',
             'ls_ratio': 'ls_collected', 'futures_taker_ratio': 'taker_ratio'}
)
merged = pd.merge_asof(
    merged.sort_values('timestamp'),
    deriv_col_subset.sort_values('timestamp'),
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)

# Use collected OI when available, fallback to backfilled
merged['oi_final'] = merged['oi_collected'].fillna(merged['oi'])
merged['ls_final'] = merged['ls_collected'].fillna(merged['ls_ratio'])
merged['funding_final'] = merged['funding_collected'].fillna(merged['funding_rate'])

# Coverage
oi_coverage = merged['oi_final'].notna().sum()
ls_coverage = merged['ls_final'].notna().sum()
print(f"OI coverage: {oi_coverage}/{len(merged)} ({100*oi_coverage/len(merged):.1f}%)")
print(f"L/S coverage: {ls_coverage}/{len(merged)} ({100*ls_coverage/len(merged):.1f}%)")

# ═══════════════════════════════════════════════════════
# STEP 3: Compute OI ROC (1h = 4 bars of 15m)
# ═══════════════════════════════════════════════════════
merged['oi_roc_1h'] = merged['oi_final'].pct_change(4)
merged['price_change_1h'] = merged['Close'].pct_change(4)
merged['price_change_4bar'] = merged['Close'].pct_change(4)

# ATR for context
high = merged['High'].values.astype(float)
low = merged['Low'].values.astype(float)
close = merged['Close'].values.astype(float)
tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
merged['atr_14'] = pd.Series(tr).rolling(14).mean().values

# Volume ratio
merged['vol_ma20'] = merged['Volume'].rolling(20).mean()
merged['vol_ratio'] = merged['Volume'] / merged['vol_ma20']

# ═══════════════════════════════════════════════════════
# STEP 4: DETECTION LOGIC (No filtering, no optimization)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("AGENT 2 (Non-Indicator) — Raw Cascade Detection")
print("=" * 60)

# Source A: OI shock — OI dropping + price moving
oi_shock_mask = (
    (merged['oi_roc_1h'].abs() > 0.01) &  # Significant OI change
    (merged['oi_roc_1h'] < -0.01) &        # OI dropping
    (merged['price_change_4bar'].abs() > 0.005)  # Price moving
)

# Source B: OI crash + extreme L/S
oi_extreme_mask = (
    (merged['oi_roc_1h'] < -0.01) &
    ((merged['ls_final'] > 1.8) | (merged['ls_final'] < 0.6))
)

# Source C: Large OI drop (broader)
oi_large_drop_mask = (merged['oi_roc_1h'] < -0.015)

# Combine all detection sources
cascade_mask = oi_shock_mask | oi_extreme_mask | oi_large_drop_mask

# Remove look-ahead: shift mask by 1 bar (detect at close, enter at next open)
cascade_mask_shifted = cascade_mask.shift(1).fillna(False)

events = merged[cascade_mask_shifted].copy()
print(f"\nTotal cascade events detected: {len(events)}")
print(f"  Source A (OI shock): {oi_shock_mask.sum()}")
print(f"  Source B (OI + L/S extreme): {oi_extreme_mask.sum()}")
print(f"  Source C (Large OI drop): {oi_large_drop_mask.sum()}")

# Direction assignment (based on price momentum at detection)
events['detected_direction'] = np.where(
    events['price_change_4bar'] < -0.005, 'LONG',  # Price dropping = long cascade (shorts win)
    np.where(events['price_change_4bar'] > 0.005, 'SHORT', 'NEUTRAL')
)

print(f"\nDirection split:")
print(f"  LONG: {(events['detected_direction'] == 'LONG').sum()}")
print(f"  SHORT: {(events['detected_direction'] == 'SHORT').sum()}")
print(f"  NEUTRAL: {(events['detected_direction'] == 'NEUTRAL').sum()}")

# ═══════════════════════════════════════════════════════
# STEP 5: FORWARD RETURNS (Isolation Gate)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("AGENT 3 (Cost Gate) + AGENT 4 (Sample Size)")
print("=" * 60)

horizons = [1, 4, 16, 24]  # bars
horizon_labels = ['1-bar (15m)', '4-bar (1h)', '16-bar (4h)', '24-bar (6h)']
round_trip_cost = 0.0010  # 0.10%

results = {}

for h, label in zip(horizons, horizon_labels):
    # Compute forward returns
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

    # Get forward returns for cascade events
    event_returns = merged.loc[events.index, f'fwd_ret_{h}'].dropna()

    if len(event_returns) < 10:
        print(f"\n{label}: Too few events ({len(event_returns)})")
        continue

    mean_ret = event_returns.mean()
    median_ret = event_returns.median()
    std_ret = event_returns.std()
    n = len(event_returns)

    # t-test: is mean different from zero?
    t_stat, p_value = stats.ttest_1samp(event_returns, 0)

    # Effect direction
    direction_correct = mean_ret > 0  # We expect continuation

    # Cost gate
    exceeds_costs = abs(mean_ret) > round_trip_cost

    results[label] = {
        'horizon_bars': h,
        'n_events': n,
        'mean_return_pct': round(mean_ret * 100, 4),
        'median_return_pct': round(median_ret * 100, 4),
        'std_pct': round(std_ret * 100, 4),
        't_stat': round(t_stat, 4),
        'p_value': round(p_value, 4),
        'direction_correct': direction_correct,
        'exceeds_costs': exceeds_costs,
        'gate_pass': p_value < 0.1 and direction_correct and exceeds_costs,
    }

    r = results[label]
    gate_emoji = "✅" if r['gate_pass'] else "❌"
    dir_emoji = "✅" if r['direction_correct'] else "❌ BACKWARDS"
    cost_emoji = "✅" if r['exceeds_costs'] else "❌ BELOW COSTS"

    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    print(f"  Events:         {n}")
    print(f"  Mean return:    {r['mean_return_pct']:+.4f}%")
    print(f"  Median return:  {r['median_return_pct']:+.4f}%")
    print(f"  Std dev:        {r['std_pct']:.4f}%")
    print(f"  t-statistic:    {r['t_stat']:.4f}")
    print(f"  p-value:        {r['p_value']:.4f}")
    print(f"  Direction:      {dir_emoji}")
    print(f"  Cost gate:      {cost_emoji} (>{round_trip_cost*100:.2f}%)")
    print(f"  GATE:           {gate_emoji}")

# ═══════════════════════════════════════════════════════
# STEP 6: Split by direction
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DIRECTION-SPLIT ANALYSIS")
print("=" * 60)

for direction in ['LONG', 'SHORT']:
    dir_events = events[events['detected_direction'] == direction]
    if len(dir_events) < 10:
        print(f"\n{direction}: Too few events ({len(dir_events)})")
        continue

    print(f"\n{'─' * 50}")
    print(f"  {direction} signals ({len(dir_events)} events)")
    print(f"{'─' * 50}")

    for h, label in zip(horizons, horizon_labels):
        rets = merged.loc[dir_events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 5:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        print(f"  {label}: mean={mean_r*100:+.4f}%, p={p:.4f}, n={len(rets)}, "
              f"dir={'✅' if mean_r > 0 else '❌'}")

# ═══════════════════════════════════════════════════════
# STEP 7: Split by OI ROC magnitude
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("OI ROC MAGNITUDE SPLIT")
print("=" * 60)

events_with_roc = events.copy()
events_with_roc['oi_roc'] = merged.loc[events.index, 'oi_roc_1h']

for threshold in [-0.01, -0.015, -0.02, -0.03]:
    subset = events_with_roc[events_with_roc['oi_roc'] < threshold]
    if len(subset) < 10:
        print(f"\nOI ROC < {threshold}: Too few ({len(subset)})")
        continue

    print(f"\n{'─' * 50}")
    print(f"  OI ROC < {threshold} ({len(subset)} events)")
    print(f"{'─' * 50}")

    for h, label in zip([4, 16, 24], ['1h', '4h', '6h']):
        rets = merged.loc[subset.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 5:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        print(f"  {label}: mean={mean_r*100:+.4f}%, p={p:.4f}, n={len(rets)}, "
              f"dir={'✅' if mean_r > 0 else '❌'}")

# ═══════════════════════════════════════════════════════
# STEP 8: Regime split (vol tercile)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("AGENT 6 PREVIEW — REGIME SPLIT (Vol Tercile)")
print("=" * 60)

merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()
merged['vol_tercile'] = pd.qcut(merged['vol_20bar'], 3, labels=['LOW', 'MID', 'HIGH'], duplicates='drop')

for tercile in ['LOW', 'MID', 'HIGH']:
    tercile_events = events[merged.loc[events.index, 'vol_tercile'] == tercile]
    if len(tercile_events) < 10:
        print(f"\n{tercile} vol: Too few ({len(tercile_events)})")
        continue

    print(f"\n{'─' * 50}")
    print(f"  {tercile} vol ({len(tercile_events)} events)")
    print(f"{'─' * 50}")

    for h, label in zip([4, 16, 24], ['1h', '4h', '6h']):
        rets = merged.loc[tercile_events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 5:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        print(f"  {label}: mean={mean_r*100:+.4f}%, p={p:.4f}, n={len(rets)}, "
              f"dir={'✅' if mean_r > 0 else '❌'}")

# ═══════════════════════════════════════════════════════
# STEP 9: Calendar-era split
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("AGENT 6 PREVIEW — CALENDAR-ERA SPLIT")
print("=" * 60)

def get_era(ts):
    if ts < pd.Timestamp('2024-07-01'):
        return '2024_H1'
    elif ts < pd.Timestamp('2025-01-01'):
        return '2024_H2'
    elif ts < pd.Timestamp('2025-07-01'):
        return '2025_H1'
    elif ts < pd.Timestamp('2026-01-01'):
        return '2025_H2'
    else:
        return '2026'

events['era'] = events['timestamp'].apply(get_era)

for era in sorted(events['era'].unique()):
    era_events = events[events['era'] == era]
    if len(era_events) < 10:
        print(f"\n{era}: Too few ({len(era_events)})")
        continue

    print(f"\n{'─' * 50}")
    print(f"  {era} ({len(era_events)} events)")
    print(f"{'─' * 50}")

    for h, label in zip([4, 16, 24], ['1h', '4h', '6h']):
        rets = merged.loc[era_events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 5:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        print(f"  {label}: mean={mean_r*100:+.4f}%, p={p:.4f}, n={len(rets)}, "
              f"dir={'✅' if mean_r > 0 else '❌'}")

# ═══════════════════════════════════════════════════════
# SAVE REPORT
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL VERDICT")
print("=" * 60)

report = {
    'strategy': 'liquidation_cascade',
    'date': '2026-07-19',
    'agent': '2-4 (Isolation Gate)',
    'data': {
        'ohlcv_bars': len(ohlcv),
        'date_range': f"{ohlcv['timestamp'].min()} to {ohlcv['timestamp'].max()}",
        'derivatives_rows': len(deriv),
        'oi_coverage_pct': round(100 * oi_coverage / len(merged), 1),
    },
    'detection': {
        'total_events': len(events),
        'oi_shock': int(oi_shock_mask.sum()),
        'oi_extreme_ls': int(oi_extreme_mask.sum()),
        'oi_large_drop': int(oi_large_drop_mask.sum()),
    },
    'isolation_gate': results,
    'look_ahead_bias': 'None (mask shifted by 1 bar)',
    'round_trip_cost': '0.10%',
    'minimum_events': 500,
}

# Overall verdict
gate_results = [r['gate_pass'] for r in results.values()]
any_pass = any(gate_results)
all_fail = not any_pass

if all_fail:
    report['verdict'] = 'KILLED'
    report['reason'] = 'Isolation gate FAILED at all horizons'
elif any_pass:
    report['verdict'] = 'GATE PASS'
    report['reason'] = 'Isolation gate passed at at least one horizon'
else:
    report['verdict'] = 'INCONCLUSIVE'

print(f"\nVerdict: {report['verdict']}")
print(f"Reason: {report['reason']}")

with open(f'{REPORT_DIR}/liquidation_cascade_isolation_gate.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nReport saved to {REPORT_DIR}/liquidation_cascade_isolation_gate.json")
