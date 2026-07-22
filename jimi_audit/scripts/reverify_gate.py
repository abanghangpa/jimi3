"""Re-verify isolation gate with exact same merge as Phase 1."""
import pandas as pd
import numpy as np
from scipy import stats

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'

# Load OHLCV
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
print(f"OHLCV: {len(ohlcv)} bars, {ohlcv['timestamp'].min()} to {ohlcv['timestamp'].max()}")

# Load BOTH derivatives sources
deriv_back = pd.read_csv(f'{DATA_DIR}/derivatives_history/derivatives_backfilled.csv')
deriv_back['timestamp'] = pd.to_datetime(deriv_back['timestamp'])
deriv_back = deriv_back.sort_values('timestamp').reset_index(drop=True)
print(f"Backfilled: {len(deriv_back)} rows, OI non-null: {deriv_back['oi'].notna().sum()}")

deriv_col = pd.read_csv(f'{DATA_DIR}/derivatives_history/derivatives_collected.csv')
deriv_col['timestamp'] = pd.to_datetime(deriv_col['timestamp'], format='mixed')
deriv_col = deriv_col.sort_values('timestamp').reset_index(drop=True)
print(f"Collected: {len(deriv_col)} rows, OI non-null: {deriv_col['oi'].notna().sum()}")

# Merge backfilled first
merged = pd.merge_asof(
    ohlcv, deriv_back[['timestamp', 'oi', 'funding_rate', 'ls_ratio']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)
# Then merge collected
deriv_col_sub = deriv_col[['timestamp', 'oi', 'funding_rate', 'ls_ratio']].rename(
    columns={'oi': 'oi_c', 'funding_rate': 'fr_c', 'ls_ratio': 'ls_c'}
)
merged = pd.merge_asof(
    merged.sort_values('timestamp'),
    deriv_col_sub.sort_values('timestamp'),
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)
# Fill: collected first, then backfilled
merged['oi_final'] = merged['oi_c'].fillna(merged['oi'])
merged['ls_final'] = merged['ls_c'].fillna(merged['ls_ratio'])

print(f"\nAfter dual merge:")
print(f"OI final coverage: {merged['oi_final'].notna().sum()}/{len(merged)} ({100*merged['oi_final'].notna().sum()/len(merged):.1f}%)")
print(f"LS final coverage: {merged['ls_final'].notna().sum()}/{len(merged)} ({100*merged['ls_final'].notna().sum()/len(merged):.1f}%)")

# Compute OI ROC
merged['oi_roc_1h'] = merged['oi_final'].pct_change(4, fill_method=None)
print(f"\nOI ROC valid: {merged['oi_roc_1h'].notna().sum()}")
print(f"OI ROC < -0.015: {(merged['oi_roc_1h'] < -0.015).sum()}")
print(f"OI ROC < -0.01: {(merged['oi_roc_1h'] < -0.01).sum()}")

# Forward returns
for h in [4, 16, 24]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

# Vol regime
merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()

# Run the EXACT same gate as Phase 1
oi_shock_mask = (
    (merged['oi_roc_1h'].abs() > 0.01) &
    (merged['oi_roc_1h'] < -0.01) &
    (merged['Close'].pct_change(4).abs() > 0.005)
)
oi_extreme_mask = (
    (merged['oi_roc_1h'] < -0.01) &
    ((merged['ls_final'] > 1.8) | (merged['ls_final'] < 0.6))
)
oi_large_drop_mask = (merged['oi_roc_1h'] < -0.015)
cascade_mask = oi_shock_mask | oi_extreme_mask | oi_large_drop_mask
cascade_mask_shifted = cascade_mask.shift(1).fillna(False)
events = merged[cascade_mask_shifted]

print(f"\nPhase 1 detection:")
print(f"  Source A (OI shock): {oi_shock_mask.sum()}")
print(f"  Source B (OI + L/S extreme): {oi_extreme_mask.sum()}")
print(f"  Source C (Large OI drop): {oi_large_drop_mask.sum()}")
print(f"  Combined: {len(events)} events")

# Forward returns for events
for h, label in [(4, '1h'), (16, '4h'), (24, '6h')]:
    rets = merged.loc[events.index, f'fwd_ret_{h}'].dropna()
    if len(rets) < 5:
        print(f"  {label}: Too few ({len(rets)})")
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    print(f"  {label}: mean={mean_r*100:+.4f}%, p={p:.4f}, n={len(rets)}, dir={'OK' if mean_r > 0 else 'BACKWARDS'}")

# Now test with collected-only data (what the validation used)
print(f"\n{'='*60}")
print("COLLECTED-ONLY MERGE (for validation)")
print('='*60)

merged2 = pd.merge_asof(
    ohlcv, deriv_col[['timestamp', 'oi', 'ls_ratio', 'funding_rate']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)
merged2['oi_roc'] = merged2['oi'].pct_change(4, fill_method=None)
merged2['vol_20bar'] = merged2['Close'].pct_change().rolling(20).std()

for thresh in [-0.005, -0.01, -0.015]:
    mask = (merged2['oi_roc'] < thresh).shift(1).fillna(False)
    events2 = merged2[mask]
    for h, label in [(4, '1h'), (16, '4h')]:
        rets = merged2.loc[events2.index, f'Close'].shift(-h) / merged2.loc[events2.index, 'Close'] - 1
        rets = rets.dropna()
        if len(rets) < 5:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        print(f"  OI<{thresh} {label}: mean={mean_r*100:+.4f}%, p={p:.4f}, n={len(rets)}, dir={'OK' if mean_r > 0 else 'BACKWARDS'}")
