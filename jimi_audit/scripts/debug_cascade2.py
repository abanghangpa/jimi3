import pandas as pd
import numpy as np

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'

# Check collected derivatives (real data)
deriv_col = pd.read_csv(f'{DATA_DIR}/derivatives_history/derivatives_collected.csv')
deriv_col['timestamp'] = pd.to_datetime(deriv_col['timestamp'], format='mixed')
print(f"Collected derivatives rows: {len(deriv_col)}")
print(f"Date range: {deriv_col['timestamp'].min()} to {deriv_col['timestamp'].max()}")
print(f"Columns: {deriv_col.columns.tolist()}")
print(f"OI non-null: {deriv_col['oi'].notna().sum()}/{len(deriv_col)}")
print(f"LS non-null: {deriv_col['ls_ratio'].notna().sum()}/{len(deriv_col)}")
print(f"OI sample: {deriv_col['oi'].dropna().head(5).tolist()}")
print(f"LS sample: {deriv_col['ls_ratio'].dropna().head(5).tolist()}")
print(f"Funding non-null: {deriv_col['funding_rate'].notna().sum()}/{len(deriv_col)}")

# Load OHLCV
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])

# Merge collected onto OHLCV
merged = pd.merge_asof(
    ohlcv.sort_values('timestamp'),
    deriv_col.sort_values('timestamp')[['timestamp', 'oi', 'ls_ratio', 'funding_rate']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)
print(f"\nMerged with collected data:")
print(f"OI coverage: {merged['oi'].notna().sum()}/{len(merged)} ({100*merged['oi'].notna().sum()/len(merged):.1f}%)")
print(f"LS coverage: {merged['ls_ratio'].notna().sum()}/{len(merged)} ({100*merged['ls_ratio'].notna().sum()/len(merged):.1f}%)")

merged['oi_roc_1h'] = merged['oi'].pct_change(4, fill_method=None)
print(f"\nOI ROC valid: {merged['oi_roc_1h'].notna().sum()}")
print(f"OI ROC < -0.015: {(merged['oi_roc_1h'] < -0.015).sum()}")
print(f"OI ROC < -0.01: {(merged['oi_roc_1h'] < -0.01).sum()}")
print(f"OI ROC < -0.005: {(merged['oi_roc_1h'] < -0.005).sum()}")

merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()
vols = merged['vol_20bar'].dropna()
p33 = vols.quantile(0.33)
p67 = vols.quantile(0.67)
merged['vol_regime'] = 'MID'
merged.loc[merged['vol_20bar'] < p33, 'vol_regime'] = 'LOW'
merged.loc[merged['vol_20bar'] > p67, 'vol_regime'] = 'HIGH'

for thresh in [-0.005, -0.01, -0.015, -0.02]:
    m = (merged['oi_roc_1h'] < thresh) & (merged['vol_regime'] == 'MID')
    print(f"\nOI ROC < {thresh} + MID vol: {m.sum()} events")
    if m.sum() > 0:
        m_shifted = m.shift(1).fillna(False)
        rets = merged.loc[m_shifted[m_shifted].index, 'Close'].shift(-16) / merged.loc[m_shifted[m_shifted].index, 'Close'] - 1
        rets = rets.dropna()
        if len(rets) > 5:
            from scipy import stats
            mean_r = rets.mean()
            t, p = stats.ttest_1samp(rets, 0)
            print(f"  Forward returns: mean={mean_r*100:+.4f}%, p={p:.4f}, n={len(rets)}")

# Also test with LS filter using collected data
for ls_thresh in [1.2, 1.5, 1.8]:
    m = (merged['oi_roc_1h'] < -0.01) & (merged['ls_ratio'] > ls_thresh)
    print(f"\nOI ROC < -0.01 + LS > {ls_thresh}: {m.sum()} events")
