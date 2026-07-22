import pandas as pd
import numpy as np

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'

deriv = pd.read_csv(f'{DATA_DIR}/derivatives_history/derivatives_backfilled.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'])
print(f"Derivatives rows: {len(deriv)}")
print(f"Date range: {deriv['timestamp'].min()} to {deriv['timestamp'].max()}")
print(f"OI non-null: {deriv['oi'].notna().sum()}/{len(deriv)}")
print(f"LS non-null: {deriv['ls_ratio'].notna().sum()}/{len(deriv)}")
print(f"OI sample: {deriv['oi'].dropna().head(5).tolist()}")
print(f"LS sample: {deriv['ls_ratio'].dropna().head(5).tolist()}")

ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
print(f"\nOHLCV rows: {len(ohlcv)}")

merged = pd.merge_asof(
    ohlcv.sort_values('timestamp'),
    deriv.sort_values('timestamp')[['timestamp', 'oi', 'ls_ratio']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)
print(f"\nMerged OI coverage: {merged['oi'].notna().sum()}/{len(merged)} ({100*merged['oi'].notna().sum()/len(merged):.1f}%)")
print(f"Merged LS coverage: {merged['ls_ratio'].notna().sum()}/{len(merged)} ({100*merged['ls_ratio'].notna().sum()/len(merged):.1f}%)")

merged['oi_roc_1h'] = merged['oi'].pct_change(4, fill_method=None)
print(f"\nOI ROC valid: {merged['oi_roc_1h'].notna().sum()}")
print(f"OI ROC < -0.015: {(merged['oi_roc_1h'] < -0.015).sum()}")
print(f"OI ROC < -0.01: {(merged['oi_roc_1h'] < -0.01).sum()}")

merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()
vols = merged['vol_20bar'].dropna()
p33 = vols.quantile(0.33)
p67 = vols.quantile(0.67)
merged['vol_regime'] = 'MID'
merged.loc[merged['vol_20bar'] < p33, 'vol_regime'] = 'LOW'
merged.loc[merged['vol_20bar'] > p67, 'vol_regime'] = 'HIGH'

mask = (merged['oi_roc_1h'] < -0.015) & (merged['ls_ratio'] > 1.5) & (merged['vol_regime'] == 'MID')
print(f"\nFull mask: {mask.sum()} events")
mask2 = (merged['oi_roc_1h'] < -0.015) & (merged['vol_regime'] == 'MID')
print(f"No-LS mask: {mask2.sum()} events")
mask3 = (merged['oi_roc_1h'] < -0.015)
print(f"OI<-0.015 only: {mask3.sum()} events")
mask4 = (merged['oi_roc_1h'] < -0.01)
print(f"OI<-0.01 only: {mask4.sum()} events")
