"""S01 Remaining: Agent 7 (sensitivity) + Agent 8 (Monte Carlo)"""
import pandas as pd, numpy as np, json, os
from scipy import stats
warnings_filter = __import__('warnings').filterwarnings
warnings_filter('ignore')

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'

ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
for c in ['Close','High','Low','Volume']: ohlcv[c] = ohlcv[c].astype(float)

highs = ohlcv['High'].values.astype(float)
lows = ohlcv['Low'].values.astype(float)
closes = ohlcv['Close'].values.astype(float)

# Detect events (same as before)
fb_events = []
seen = set()
for idx in range(56, len(ohlcv)):
    for lb in range(1, min(8, idx)):
        bar_idx = idx - lb
        if bar_idx < 48: continue
        bh, bl, bc = highs[bar_idx], lows[bar_idx], closes[bar_idx]
        sh = float(np.max(highs[bar_idx-48:bar_idx]))
        sl = float(np.min(lows[bar_idx-48:bar_idx]))
        
        if bh > sh * 1.001 and bc < sh:
            k = (bar_idx, 'S')
            if k not in seen:
                held = 0
                for j in range(bar_idx, idx+1):
                    if highs[j] > sh: held += 1
                    else: break
                if closes[idx] < sh and held >= 1:
                    seen.add(k)
                    fb_events.append({'idx': idx, 'd': 'SHORT', 'q': min(held/5, 1.0), 'bh': held, 'lv': sh})
                    break
        if bl < sl * 0.999 and bc > sl:
            k = (bar_idx, 'L')
            if k not in seen:
                held = 0
                for j in range(bar_idx, idx+1):
                    if lows[j] < sl: held += 1
                    else: break
                if closes[idx] > sl and held >= 1:
                    seen.add(k)
                    fb_events.append({'idx': idx, 'd': 'LONG', 'q': min(held/5, 1.0), 'bh': held, 'lv': sl})
                    break

# Merge for returns
merged = ohlcv[['timestamp','Close']].copy()
merged['fwd_ret_16'] = merged['Close'].shift(-16) / merged['Close'] - 1

print(f"Events: {len(fb_events)}")

# AGENT 7: SENSITIVITY
print("\n" + "="*70)
print("AGENT 7: SENSITIVITY")
print("="*70)

a7 = {}
for d in ['LONG', 'SHORT']:
    for q in [0, 0.2, 0.3, 0.5]:
        for bh in [1, 2, 3, 4]:
            evts = [e for e in fb_events if e['d'] == d and e['q'] >= q and e['bh'] >= bh]
            if len(evts) < 10: continue
            indices = [e['idx'] for e in evts]
            rets = merged.iloc[indices]['fwd_ret_16'].dropna()
            if len(rets) < 5: continue
            mean_r = rets.mean()
            eff = -mean_r if d == 'SHORT' else mean_r
            t, p = stats.ttest_1samp(rets, 0)
            wr = (rets < 0).mean() if d == 'SHORT' else (rets > 0).mean()
            gate = "PASS" if p < 0.1 and eff > 0.001 else "FAIL"
            if gate == "PASS":
                key = f"{d}_q{q}_bh{bh}"
                a7[key] = {'n': len(rets), 'eff': float(eff), 'p': float(p), 'wr': float(wr)}
                print(f"  + {d} q>={q} bh>={bh}: n={len(rets)}, eff={eff*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# AGENT 8: MONTE CARLO
print("\n" + "="*70)
print("AGENT 8: MONTE CARLO")
print("="*70)

indices = [e['idx'] for e in fb_events]
actual = merged.iloc[indices]['fwd_ret_16'].dropna()
am, aw, n = actual.mean(), (actual > 0).mean(), len(actual)
print(f"  n={n}, mean={am*100:+.4f}%, WR={aw:.1%}")

np.random.seed(42)
all_r = merged['fwd_ret_16'].dropna()
rm = np.array([all_r.sample(n).mean() for _ in range(10000)])
pm = (rm >= am).mean()

bm = np.array([actual.sample(n, replace=True).mean() for _ in range(10000)])
ci_lo, ci_hi = np.percentile(bm, 2.5), np.percentile(bm, 97.5)

print(f"  MC p: {pm:.4f}")
print(f"  CI: [{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%]")
print(f"  SIGNIFICANT: {'YES' if pm < 0.05 else 'NO'}")

# Save
with open('/root/.openclaw/workspace/jimi_audit/reports/s01_remaining.json', 'w') as f:
    json.dump({'agent_7': a7, 'agent_8': {'n': n, 'mean': float(am), 'wr': float(aw), 'mc_p': float(pm),
              'ci': [float(ci_lo), float(ci_hi)], 'sig': pm < 0.05}}, f, indent=2)
print("\nDone")
