import pandas as pd, numpy as np
from scipy import stats

d = pd.read_csv('/root/.openclaw/workspace/jimi_audit/data/eth_15m_extended.csv')
d['timestamp'] = pd.to_datetime(d['Open time'])
d = d.sort_values('timestamp').reset_index(drop=True)

deriv = pd.read_csv('/root/.openclaw/workspace/jimi_audit/data/derivatives_history/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
m = pd.merge_asof(d, deriv[['timestamp','funding_rate']], on='timestamp', direction='backward', tolerance=pd.Timedelta('2h'))

fr = m['funding_rate']
fr_z = (fr - fr.rolling(96).mean()) / fr.rolling(96).std()
m['fr_z'] = fr_z
m['fwd_ret_16'] = m['Close'].shift(-16) / m['Close'] - 1

print("Funding rate z-score thresholds (SHORT direction):")
for thresh in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]:
    mask = (fr_z > thresh).shift(1).fillna(False)
    events = m[mask]
    rets = m.loc[events.index, 'fwd_ret_16'].dropna()
    if len(rets) < 5: continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    eff_wr = 1 - wr
    gate = "PASS" if p < 0.1 and abs(mean_r) > 0.001 else "FAIL"
    sym = "+" if gate == "PASS" else "-"
    print(f"  {sym} FR_z > {thresh:5.2f}: n={len(rets):5d}, mean={mean_r*100:+.4f}%, eff_WR={eff_wr:.1%}, p={p:.4f} [{gate}]")

# Test with backfilled data (longer history)
print("\nBackfilled funding data:")
deriv_b = pd.read_csv('/root/.openclaw/workspace/jimi_audit/data/derivatives_history/derivatives_backfilled.csv')
deriv_b['timestamp'] = pd.to_datetime(deriv_b['timestamp'])
fr_b = deriv_b['funding_rate'].dropna()
print(f"  Rows: {len(fr_b)}, range: {fr_b.min():.6f} to {fr_b.max():.6f}")

m2 = pd.merge_asof(d, deriv_b[['timestamp','funding_rate']], on='timestamp', direction='backward', tolerance=pd.Timedelta('2h'))
fr2 = m2['funding_rate']
fr_z2 = (fr2 - fr2.rolling(96).mean()) / fr2.rolling(96).std()
m2['fr_z'] = fr_z2
m2['fwd_ret_16'] = m2['Close'].shift(-16) / m2['Close'] - 1

print("\nBackfilled z-score thresholds (SHORT):")
for thresh in [1.5, 2.0, 2.5, 3.0]:
    mask = (fr_z2 > thresh).shift(1).fillna(False)
    events = m2[mask]
    rets = m2.loc[events.index, 'fwd_ret_16'].dropna()
    if len(rets) < 5: continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    eff_wr = 1 - (rets > 0).mean()
    gate = "PASS" if p < 0.1 and abs(mean_r) > 0.001 else "FAIL"
    sym = "+" if gate == "PASS" else "-"
    print(f"  {sym} FR_z > {thresh:5.2f}: n={len(rets):5d}, mean={mean_r*100:+.4f}%, eff_WR={eff_wr:.1%}, p={p:.4f} [{gate}]")
