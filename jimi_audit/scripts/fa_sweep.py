#!/usr/bin/env python3
"""Parameter sweep for funding_arb on extended data."""
import numpy as np, pandas as pd
from scipy import stats

df = pd.read_csv('/root/.openclaw/workspace/jimi_audit/data/eth_15m_merged_extended.csv')
print(f"Loaded: {len(df)} bars")
c = df['Close']; v = df['Volume']
tr = df['Taker buy base asset volume'] / v.replace(0, 1)
tr_ma = tr.rolling(100).mean()
tr_std = tr.rolling(100).std()
tr_zscore = (tr - tr_ma) / tr_std.replace(0, 1)
vol_ratio = v / v.rolling(20).mean().replace(0, 1)
price_round_dist = (c % 50) / 50

results = []
for z in [1.5, 1.4, 1.3, 1.25, 1.2, 1.1]:
    for rd in [0.02, 0.03, 0.05]:
        near_round = price_round_dist < rd
        long_ev = (tr_zscore < -z) & (vol_ratio > 1.0) & near_round
        short_ev = (tr_zscore > z) & (vol_ratio > 1.0) & near_round
        events = long_ev | short_ev
        idx = np.where(events)[0]
        if len(idx) < 10: continue
        close = c.values; n = len(close)
        fr = [(close[i+24]-close[i])/close[i] for i in idx if i+24 < n]
        if len(fr) < 10: continue
        fr_arr = np.array(fr); mean_r = np.mean(fr_arr)
        ne_mask = np.ones(n, dtype=bool); ne_mask[idx] = False
        ne_idx = np.where(ne_mask)[0]
        ne = np.array([(close[i+24]-close[i])/close[i] for i in ne_idx if i+24 < n])
        t, p = stats.ttest_ind(fr_arr, ne, equal_var=False)
        dir_ok = mean_r > 0
        eff_ok = abs(mean_r) > 0.001
        passed = dir_ok and p < 0.1 and eff_ok
        results.append({'z': z, 'rd': rd, 'events': len(idx), 'mean_pct': round(mean_r*100,4), 'p': round(float(p),4), 'dir': 'OK' if dir_ok else 'BAD', 'pass': 'PASS' if passed else ''})

print()
print(f"{'z':>5} {'rd':>5} {'events':>7} {'mean%':>10} {'p':>8} {'dir':>5} {'gate':>5}")
for r in sorted(results, key=lambda x: x['p']):
    print(f"{r['z']:>5.2f} {r['rd']:>5.2f} {r['events']:>7} {r['mean_pct']:>+10.4f} {r['p']:>8.4f} {r['dir']:>5} {r['pass']:>5}")
