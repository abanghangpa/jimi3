#!/usr/bin/env python3
"""Parameter sweep for judas_sweep on extended data."""
import numpy as np, pandas as pd
from scipy import stats

df = pd.read_csv('/root/.openclaw/workspace/jimi_audit/data/eth_15m_merged_extended.csv')
print(f"Loaded: {len(df)} bars")
c = df['Close']; h = df['High']; l = df['Low']; v = df['Volume']
vol_ratio = v / v.rolling(20).mean().replace(0, 1)

daily_high = h.rolling(96).max().shift(1)
daily_low = l.rolling(96).min().shift(1)
session_high = h.rolling(32).max().shift(1)
session_low = l.rolling(32).min().shift(1)

n = len(df)
results = []

for vol_thresh in [1.0, 1.1, 1.2, 1.3, 1.5]:
    for wick_mult in [1.0, 1.1, 1.2, 1.5]:
        sweep_high = np.zeros(n, dtype=bool)
        sweep_low = np.zeros(n, dtype=bool)
        
        for i in range(100, n):
            price_now = c.iloc[i]
            high_now = h.iloc[i]
            low_now = l.iloc[i]
            vol_now = vol_ratio.iloc[i] if not pd.isna(vol_ratio.iloc[i]) else 1.0
            if vol_now < vol_thresh: continue
            
            for level_val in [daily_high.iloc[i], session_high.iloc[i]]:
                if pd.isna(level_val) or level_val <= 0: continue
                if high_now > level_val * 1.001 and price_now < level_val:
                    if (high_now - price_now) > (price_now - low_now) * wick_mult:
                        sweep_high[i] = True
                        break
            
            for level_val in [daily_low.iloc[i], session_low.iloc[i]]:
                if pd.isna(level_val) or level_val <= 0: continue
                if low_now < level_val * 0.999 and price_now > level_val:
                    if (price_now - low_now) > (high_now - price_now) * wick_mult:
                        sweep_low[i] = True
                        break
        
        events = sweep_high | sweep_low
        idx = np.where(events)[0]
        if len(idx) < 10: continue
        
        close = c.values
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
        results.append({
            'vol': vol_thresh, 'wick': wick_mult, 'events': len(idx),
            'mean_pct': round(mean_r*100,4), 'p': round(float(p),4),
            'dir': 'OK' if dir_ok else 'BAD', 'pass': 'PASS' if passed else ''
        })

print()
print(f"{'vol':>5} {'wick':>5} {'events':>7} {'mean%':>10} {'p':>8} {'dir':>5} {'gate':>5}")
for r in sorted(results, key=lambda x: x['p']):
    print(f"{r['vol']:>5.1f} {r['wick']:>5.1f} {r['events']:>7} {r['mean_pct']:>+10.4f} {r['p']:>8.4f} {r['dir']:>5} {r['pass']:>5}")
