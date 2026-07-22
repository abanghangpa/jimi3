#!/usr/bin/env python3
"""
Failed Breakout v8b — Short-Horizon Check
Same 23,305 sweep events, but testing 1-bar (15m) and 4-bar (1h) forward returns.
If the trapped-trader flush is a 15min-2h phenomenon, it would show here.
"""
import csv, json, sys, os
from datetime import datetime, timezone
import numpy as np

BASE = '/root/.openclaw/workspace/jimi_audit'
DATA_FILE = f'{BASE}/eth_15m_6m.csv'

print('Loading...', flush=True)
bars = []
with open(DATA_FILE) as f:
    for row in csv.DictReader(f):
        bars.append({
            'ts': datetime.strptime(row['Open time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc),
            'o': float(row['Open']), 'h': float(row['High']),
            'l': float(row['Low']), 'c': float(row['Close']),
            'v': float(row['Volume']),
            'tb': float(row.get('Taker buy base asset volume', 0)),
        })

N = len(bars)
closes = np.array([b['c'] for b in bars])
highs = np.array([b['h'] for b in bars])
lows = np.array([b['l'] for b in bars])
volumes = np.array([b['v'] for b in bars])
taker_buys = np.array([b['tb'] for b in bars])

atr = np.zeros(N)
for i in range(1, N):
    tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
    atr[i] = tr if i < 14 else (atr[i-1]*13 + tr)/14

avg_vol = np.zeros(N)
for i in range(20, N):
    avg_vol[i] = np.mean(volumes[i-20:i])

cvd_delta = 2 * taker_buys - volumes
cvd_cum = np.cumsum(cvd_delta)

print(f'Loaded {N} bars', flush=True)

# Swings
swing_highs = []
swing_lows = []
for i in range(3, N-3):
    if highs[i] > max(highs[i-3:i]) and highs[i] > max(highs[i+1:i+4]):
        swing_highs.append((i, highs[i]))
    if lows[i] < min(lows[i-3:i]) and lows[i] < min(lows[i+1:i+4]):
        swing_lows.append((i, lows[i]))

# CVD divergence
cvd_div_h = np.zeros(N, dtype=bool)
cvd_div_l = np.zeros(N, dtype=bool)
for i in range(20, N):
    if highs[i] > np.max(highs[i-20:i]):
        cm = np.max(cvd_cum[i-20:i])
        if cvd_cum[i] < cm:
            cvd_div_h[i] = True
    if lows[i] < np.min(lows[i-20:i]):
        cm = np.min(cvd_cum[i-20:i])
        if cvd_cum[i] > cm:
            cvd_div_l[i] = True

# Equal highs/lows (for location filter)
EQ_TOL = 0.002
def find_eq(swings):
    clusters = []; used = set()
    for i, (idx, price) in enumerate(swings):
        if i in used: continue
        cluster = [(idx, price)]
        for j in range(i+1, len(swings)):
            if j in used: continue
            if abs(price - swings[j][1]) / price < EQ_TOL:
                cluster.append((j, swings[j][1])); used.add(j)
        if len(cluster) >= 2:
            clusters.append({'price': np.mean([p for _,p in cluster]), 'count': len(cluster)})
        used.add(i)
    return clusters

eq_highs = find_eq(swing_highs)
eq_lows = find_eq(swing_lows)

# Session/Day H/L
sess_h = np.full(N, np.nan); sess_l = np.full(N, np.nan)
day_h = np.full(N, np.nan); day_l = np.full(N, np.nan)
for i in range(96, N):
    sh = max(0, i - 32)
    sess_h[i] = np.max(highs[sh:i+1]); sess_l[i] = np.min(lows[sh:i+1])
    day_h[i] = np.max(highs[i-96:i+1]); day_l[i] = np.min(lows[i-96:i+1])

def at_liquidity_pool(price, bar, direction):
    """Check if a price level is near a meaningful liquidity pool."""
    if direction == 'high':
        for eq in eq_highs:
            if abs(eq['price'] - price) / price < EQ_TOL:
                return True, f"eq_high({eq['count']})"
        if not np.isnan(sess_h[bar]) and atr[bar] > 0 and abs(sess_h[bar]-price)/atr[bar] < 0.3:
            return True, "session_high"
        if not np.isnan(day_h[bar]) and atr[bar] > 0 and abs(day_h[bar]-price)/atr[bar] < 0.3:
            return True, "day_high"
    else:
        for eq in eq_lows:
            if abs(eq['price'] - price) / price < EQ_TOL:
                return True, f"eq_low({eq['count']})"
        if not np.isnan(sess_l[bar]) and atr[bar] > 0 and abs(sess_l[bar]-price)/atr[bar] < 0.3:
            return True, "session_low"
        if not np.isnan(day_l[bar]) and atr[bar] > 0 and abs(day_l[bar]-price)/atr[bar] < 0.3:
            return True, "day_low"
    return False, None

# Detect sweeps
print('Detecting sweeps...', flush=True)
LOOKBACK = 200
sweeps = []

for i in range(96, N - 16):
    if atr[i] == 0 or avg_vol[i] == 0:
        continue

    # SHORT
    rsh = [(idx, p) for idx, p in swing_highs if i-LOOKBACK < idx < i-3]
    rsh.sort(key=lambda x: abs(x[1] - closes[i]))
    for li, lp in rsh[:3]:
        if highs[i] <= lp: continue
        sd = (highs[i] - lp) / atr[i] if atr[i] > 0 else 0
        if sd < 0.05: continue
        if closes[i] >= lp: continue

        is_pool, pool_type = at_liquidity_pool(lp, i, 'high')

        fwd_1b = (closes[min(i+1, N-1)] - closes[i]) / closes[i] * -100
        fwd_4b = (closes[min(i+4, N-1)] - closes[i]) / closes[i] * -100

        sweeps.append({
            'dir': 'SHORT', 'sweep_depth': round(sd, 3),
            'cvd_div': bool(cvd_div_h[i]),
            'vol_ratio': round(volumes[i]/avg_vol[i], 2),
            'is_pool': is_pool, 'pool_type': pool_type,
            'fwd_1b': round(fwd_1b, 4), 'fwd_4b': round(fwd_4b, 4),
            'ts': str(bars[i]['ts']),
        })
        break

    # LONG
    rsl = [(idx, p) for idx, p in swing_lows if i-LOOKBACK < idx < i-3]
    rsl.sort(key=lambda x: abs(x[1] - closes[i]))
    for li, lp in rsl[:3]:
        if lows[i] >= lp: continue
        sd = (lp - lows[i]) / atr[i] if atr[i] > 0 else 0
        if sd < 0.05: continue
        if closes[i] <= lp: continue

        is_pool, pool_type = at_liquidity_pool(lp, i, 'low')

        fwd_1b = (closes[min(i+1, N-1)] - closes[i]) / closes[i] * 100
        fwd_4b = (closes[min(i+4, N-1)] - closes[i]) / closes[i] * 100

        sweeps.append({
            'dir': 'LONG', 'sweep_depth': round(sd, 3),
            'cvd_div': bool(cvd_div_l[i]),
            'vol_ratio': round(volumes[i]/avg_vol[i], 2),
            'is_pool': is_pool, 'pool_type': pool_type,
            'fwd_1b': round(fwd_1b, 4), 'fwd_4b': round(fwd_4b, 4),
            'ts': str(bars[i]['ts']),
        })
        break

print(f'Total sweeps: {len(sweeps)}', flush=True)

# === Analysis ===
def analyze(events, label):
    if not events:
        print(f'\n{label}: no events')
        return
    fb1 = [s['fwd_1b'] for s in events]
    fb4 = [s['fwd_4b'] for s in events]
    print(f'\n{"="*70}')
    print(f'{label} ({len(events)} events)')
    print(f'{"="*70}')
    print(f'  15m: mean={np.mean(fb1):+.4f}% median={np.median(fb1):+.4f}% win={sum(1 for x in fb1 if x>0)/len(fb1)*100:.1f}%')
    print(f'  1h:  mean={np.mean(fb4):+.4f}% median={np.median(fb4):+.4f}% win={sum(1 for x in fb4 if x>0)/len(fb4)*100:.1f}%')

# All sweeps
analyze(sweeps, 'ALL SWEEPS')

# Split by CVD
with_div = [s for s in sweeps if s['cvd_div']]
without_div = [s for s in sweeps if not s['cvd_div']]
analyze(with_div, 'WITH CVD Divergence')
analyze(without_div, 'WITHOUT CVD Divergence')

# Split by location (liquidity pool vs random)
at_pool = [s for s in sweeps if s['is_pool']]
not_pool = [s for s in sweeps if not s['is_pool']]
analyze(at_pool, 'AT LIQUIDITY POOL (eq highs/lows, session/day H/L)')
analyze(not_pool, 'AT RANDOM SWING POINT')

# Cross: CVD + location
cvd_at_pool = [s for s in sweeps if s['cvd_div'] and s['is_pool']]
cvd_not_pool = [s for s in sweeps if s['cvd_div'] and not s['is_pool']]
no_cvd_at_pool = [s for s in sweeps if not s['cvd_div'] and s['is_pool']]
no_cvd_not_pool = [s for s in sweeps if not s['cvd_div'] and not s['is_pool']]
analyze(cvd_at_pool, 'CVD + AT POOL')
analyze(cvd_not_pool, 'CVD + NOT AT POOL')
analyze(no_cvd_at_pool, 'NO CVD + AT POOL')
analyze(no_cvd_not_pool, 'NO CVD + NOT AT POOL')

# Statistical tests
from scipy import stats

print(f'\n{"="*70}')
print('STATISTICAL TESTS')
print(f'{"="*70}')

# CVD vs no-CVD at short horizons
for horizon, label in [('fwd_1b', '15m'), ('fwd_4b', '1h')]:
    a = [s[horizon] for s in with_div]
    b = [s[horizon] for s in without_div]
    t, p = stats.ttest_ind(a, b)
    print(f'  CVD vs no-CVD ({label}): t={t:.2f} p={p:.4f} {"***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else "n.s."}')

# Pool vs not-pool
for horizon, label in [('fwd_1b', '15m'), ('fwd_4b', '1h')]:
    a = [s[horizon] for s in at_pool]
    b = [s[horizon] for s in not_pool]
    t, p = stats.ttest_ind(a, b)
    print(f'  Pool vs random ({label}): t={t:.2f} p={p:.4f} {"***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else "n.s."}')

# Volume ratio analysis
print(f'\n{"="*70}')
print('VOLUME RATIO BUCKETS')
print(f'{"="*70}')
for lo, hi, label in [(0.5, 1.0, 'low_vol'), (1.0, 2.0, 'normal_vol'), (2.0, 5.0, 'high_vol'), (5.0, 999, 'extreme_vol')]:
    bucket = [s for s in sweeps if lo <= s['vol_ratio'] < hi]
    if not bucket: continue
    fb1 = [s['fwd_1b'] for s in bucket]
    fb4 = [s['fwd_4b'] for s in bucket]
    cvd_pct = sum(1 for s in bucket if s['cvd_div']) / len(bucket) * 100
    print(f'  {label} ({lo}-{hi}x, n={len(bucket)}): 15m mean={np.mean(fb1):+.4f}% 1h mean={np.mean(fb4):+.4f}% cvd%={cvd_pct:.1f}%')

print('\nDone')
