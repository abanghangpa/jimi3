#!/usr/bin/env python3
"""
Failed Breakout v8 — CVD Divergence Isolation Test
===================================================
No scoring. No TP/SL optimization. No EMA filter.
Just: log every sweep event, split on CVD divergence present/not,
look at unconditional forward returns at fixed horizons.

This tells us if "CVD divergence at sweep" is a real mechanism
before building anything else on top of it.
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

print(f'Loaded {N} bars ({bars[0]["ts"].strftime("%Y-%m-%d")} to {bars[-1]["ts"].strftime("%Y-%m-%d")})', flush=True)

# === Swing points (5-bar fractal) ===
print('Computing swings...', flush=True)
swing_highs = []
swing_lows = []
for i in range(3, N-3):
    if highs[i] > max(highs[i-3:i]) and highs[i] > max(highs[i+1:i+4]):
        swing_highs.append((i, highs[i]))
    if lows[i] < min(lows[i-3:i]) and lows[i] < min(lows[i+1:i+4]):
        swing_lows.append((i, lows[i]))
print(f'  {len(swing_highs)}H {len(swing_lows)}L')

# === Pre-compute CVD divergence per bar ===
print('Pre-computing CVD divergence...', flush=True)
cvd_div_h = np.zeros(N, dtype=bool)
cvd_div_l = np.zeros(N, dtype=bool)
cvd_str_h = np.zeros(N)
cvd_str_l = np.zeros(N)

for i in range(20, N):
    if highs[i] > np.max(highs[i-20:i]):
        cm = np.max(cvd_cum[i-20:i])
        if cvd_cum[i] < cm:
            cvd_div_h[i] = True
            cvd_str_h[i] = (cm - cvd_cum[i]) / (abs(cm) + 1e-8)
    if lows[i] < np.min(lows[i-20:i]):
        cm = np.min(cvd_cum[i-20:i])
        if cvd_cum[i] > cm:
            cvd_div_l[i] = True
            cvd_str_l[i] = (cvd_cum[i] - cm) / (abs(cm) + 1e-8)

# === Detect ALL sweep events (no filtering) ===
print('Detecting all sweeps...', flush=True)
LOOKBACK = 200
COOLDOWN = 4  # shorter cooldown to get more events

sweeps = []  # all sweep events

for i in range(96, N - 96):  # leave room for 24h forward return
    if atr[i] == 0 or avg_vol[i] == 0:
        continue

    # SHORT sweeps: high above recent swing high, close back below
    rsh = [(idx, p) for idx, p in swing_highs if i-LOOKBACK < idx < i-3]
    rsh.sort(key=lambda x: abs(x[1] - closes[i]))
    for li, lp in rsh[:3]:  # top 3 nearest
        if highs[i] <= lp:
            continue
        sweep_depth = (highs[i] - lp) / atr[i] if atr[i] > 0 else 0
        if sweep_depth < 0.05:  # very loose threshold — just need *some* penetration
            continue
        if closes[i] >= lp:  # must close back below
            continue

        # This is a sweep event. Log it unconditionally.
        # Forward returns at fixed horizons
        fwd_4h = (closes[min(i+4, N-1)] - closes[i]) / closes[i] * -100  # SHORT: profit when price drops
        fwd_16h = (closes[min(i+16, N-1)] - closes[i]) / closes[i] * -100
        fwd_24h = (closes[min(i+24, N-1)] - closes[i]) / closes[i] * -100
        # Max adverse / favorable in next 16 bars
        future_highs = highs[i+1:min(i+17, N)]
        future_lows = lows[i+1:min(i+17, N)]
        if len(future_highs) > 0:
            max_adverse = (np.max(future_highs) - closes[i]) / closes[i] * 100  # SHORT: adverse = price goes up
            max_favorable = (closes[i] - np.min(future_lows)) / closes[i] * 100  # SHORT: favorable = price goes down
        else:
            max_adverse = 0; max_favorable = 0

        sweeps.append({
            'bar': i, 'ts': str(bars[i]['ts']),
            'dir': 'SHORT', 'level': lp,
            'sweep_depth': round(sweep_depth, 3),
            'cvd_div': bool(cvd_div_h[i]),
            'cvd_str': round(float(cvd_str_h[i]), 3),
            'vol_ratio': round(volumes[i]/avg_vol[i], 2) if avg_vol[i] > 0 else 0,
            'fwd_4h': round(fwd_4h, 4),
            'fwd_16h': round(fwd_16h, 4),
            'fwd_24h': round(fwd_24h, 4),
            'max_adverse': round(max_adverse, 4),
            'max_favorable': round(max_favorable, 4),
        })
        break  # only one sweep per bar per direction

    # LONG sweeps: low below recent swing low, close back above
    rsl = [(idx, p) for idx, p in swing_lows if i-LOOKBACK < idx < i-3]
    rsl.sort(key=lambda x: abs(x[1] - closes[i]))
    for li, lp in rsl[:3]:
        if lows[i] >= lp:
            continue
        sweep_depth = (lp - lows[i]) / atr[i] if atr[i] > 0 else 0
        if sweep_depth < 0.05:
            continue
        if closes[i] <= lp:
            continue

        fwd_4h = (closes[min(i+4, N-1)] - closes[i]) / closes[i] * 100  # LONG: profit when price rises
        fwd_16h = (closes[min(i+16, N-1)] - closes[i]) / closes[i] * 100
        fwd_24h = (closes[min(i+24, N-1)] - closes[i]) / closes[i] * 100
        future_highs = highs[i+1:min(i+17, N)]
        future_lows = lows[i+1:min(i+17, N)]
        if len(future_highs) > 0:
            max_adverse = (closes[i] - np.min(future_lows)) / closes[i] * 100
            max_favorable = (np.max(future_highs) - closes[i]) / closes[i] * 100
        else:
            max_adverse = 0; max_favorable = 0

        sweeps.append({
            'bar': i, 'ts': str(bars[i]['ts']),
            'dir': 'LONG', 'level': lp,
            'sweep_depth': round(sweep_depth, 3),
            'cvd_div': bool(cvd_div_l[i]),
            'cvd_str': round(float(cvd_str_l[i]), 3),
            'vol_ratio': round(volumes[i]/avg_vol[i], 2) if avg_vol[i] > 0 else 0,
            'fwd_4h': round(fwd_4h, 4),
            'fwd_16h': round(fwd_16h, 4),
            'fwd_24h': round(fwd_24h, 4),
            'max_adverse': round(max_adverse, 4),
            'max_favorable': round(max_favorable, 4),
        })
        break

# === Analysis ===
print(f'\nTotal sweeps detected: {len(sweeps)}', flush=True)

with_div = [s for s in sweeps if s['cvd_div']]
without_div = [s for s in sweeps if not s['cvd_div']]

print(f'With CVD divergence: {len(with_div)} ({len(with_div)/len(sweeps)*100:.1f}%)')
print(f'Without CVD divergence: {len(without_div)} ({len(without_div)/len(sweeps)*100:.1f}%)')

def bucket_stats(events, label):
    if not events:
        print(f'\n{label}: no events')
        return
    fwd_4h = [s['fwd_4h'] for s in events]
    fwd_16h = [s['fwd_16h'] for s in events]
    fwd_24h = [s['fwd_24h'] for s in events]
    ma = [s['max_adverse'] for s in events]
    mf = [s['max_favorable'] for s in events]
    depths = [s['sweep_depth'] for s in events]
    vols = [s['vol_ratio'] for s in events]

    print(f'\n{"="*80}')
    print(f'{label} ({len(events)} events)')
    print(f'{"="*80}')
    print(f'  Sweep depth:  min={min(depths):.3f} avg={np.mean(depths):.3f} max={max(depths):.3f} ATR')
    print(f'  Volume ratio: avg={np.mean(vols):.2f}')
    print(f'')
    print(f'  Forward returns (directional — positive = profitable):')
    print(f'    4h:   mean={np.mean(fwd_4h):+.3f}%  median={np.median(fwd_4h):+.3f}%  win_rate={sum(1 for x in fwd_4h if x > 0)/len(fwd_4h)*100:.1f}%')
    print(f'    16h:  mean={np.mean(fwd_16h):+.3f}%  median={np.median(fwd_16h):+.3f}%  win_rate={sum(1 for x in fwd_16h if x > 0)/len(fwd_16h)*100:.1f}%')
    print(f'    24h:  mean={np.mean(fwd_24h):+.3f}%  median={np.median(fwd_24h):+.3f}%  win_rate={sum(1 for x in fwd_24h if x > 0)/len(fwd_24h)*100:.1f}%')
    print(f'')
    print(f'  Risk metrics (16h window):')
    print(f'    Max adverse:   mean={np.mean(ma):.3f}%  p75={np.percentile(ma,75):.3f}%  p95={np.percentile(ma,95):.3f}%')
    print(f'    Max favorable: mean={np.mean(mf):.3f}%  p75={np.percentile(mf,75):.3f}%  p95={np.percentile(mf,95):.3f}%')
    print(f'    Reward/Risk:   {np.mean(mf)/np.mean(ma):.2f}x' if np.mean(ma) > 0 else '    Reward/Risk: inf')

    # Direction breakdown
    longs = [s for s in events if s['dir'] == 'LONG']
    shorts = [s for s in events if s['dir'] == 'SHORT']
    if longs:
        lfwd = [s['fwd_16h'] for s in longs]
        print(f'\n  LONG ({len(longs)}): 16h mean={np.mean(lfwd):+.3f}% win_rate={sum(1 for x in lfwd if x > 0)/len(lfwd)*100:.1f}%')
    if shorts:
        sfwd = [s['fwd_16h'] for s in shorts]
        print(f'  SHORT ({len(shorts)}): 16h mean={np.mean(sfwd):+.3f}% win_rate={sum(1 for x in sfwd if x > 0)/len(sfwd)*100:.1f}%')

    # Monthly breakdown
    monthly = {}
    for s in events:
        m = s['ts'][:7]
        if m not in monthly:
            monthly[m] = []
        monthly[m].append(s['fwd_16h'])
    print(f'\n  Monthly 16h returns:')
    for m in sorted(monthly.keys()):
        vals = monthly[m]
        print(f'    {m}: n={len(vals):3d} mean={np.mean(vals):+.3f}% win_rate={sum(1 for x in vals if x > 0)/len(vals)*100:.0f}%')

bucket_stats(with_div, 'WITH CVD Divergence')
bucket_stats(without_div, 'WITHOUT CVD Divergence')

# === Statistical test ===
if with_div and without_div:
    print(f'\n{"="*80}')
    print('STATISTICAL COMPARISON')
    print(f'{"="*80}')

    from scipy import stats

    for horizon, label in [('fwd_4h', '4h'), ('fwd_16h', '16h'), ('fwd_24h', '24h')]:
        a = [s[horizon] for s in with_div]
        b = [s[horizon] for s in without_div]
        t_stat, p_val = stats.ttest_ind(a, b)
        print(f'  {label}: CVD mean={np.mean(a):+.3f}% vs no-CVD mean={np.mean(b):+.3f}% | t={t_stat:.2f} p={p_val:.4f} {"***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else "n.s."}')

    # Effect of CVD strength
    if len(with_div) > 20:
        print(f'\n  CVD strength vs forward return (WITH divergence only):')
        for horizon, label in [('fwd_4h', '4h'), ('fwd_16h', '16h'), ('fwd_24h', '24h')]:
            strengths = [s['cvd_str'] for s in with_div]
            returns = [s[horizon] for s in with_div]
            corr = np.corrcoef(strengths, returns)[0, 1]
            print(f'    {label}: correlation={corr:.3f}')

        # Split by median CVD strength
        median_str = np.median([s['cvd_str'] for s in with_div])
        strong = [s for s in with_div if s['cvd_str'] >= median_str]
        weak = [s for s in with_div if s['cvd_str'] < median_str]
        print(f'\n  Strong CVD (str>={median_str:.3f}, n={len(strong)}):')
        for horizon, label in [('fwd_16h', '16h'), ('fwd_24h', '24h')]:
            vals = [s[horizon] for s in strong]
            print(f'    {label}: mean={np.mean(vals):+.3f}% win_rate={sum(1 for x in vals if x > 0)/len(vals)*100:.1f}%')
        print(f'  Weak CVD (str<{median_str:.3f}, n={len(weak)}):')
        for horizon, label in [('fwd_16h', '16h'), ('fwd_24h', '24h')]:
            vals = [s[horizon] for s in weak]
            print(f'    {label}: mean={np.mean(vals):+.3f}% win_rate={sum(1 for x in vals if x > 0)/len(vals)*100:.1f}%')

# === Depth bucket analysis ===
print(f'\n{"="*80}')
print('DEPTH BUCKETS (all sweeps)')
print(f'{"="*80}')
depth_buckets = [(0.05, 0.2, 'shallow'), (0.2, 0.5, 'medium'), (0.5, 1.0, 'deep'), (1.0, 999, 'extreme')]
for lo, hi, label in depth_buckets:
    bucket = [s for s in sweeps if lo <= s['sweep_depth'] < hi]
    if not bucket:
        continue
    fwd = [s['fwd_16h'] for s in bucket]
    cvd_pct = sum(1 for s in bucket if s['cvd_div']) / len(bucket) * 100
    print(f'  {label} ({lo}-{hi} ATR, n={len(bucket)}): 16h mean={np.mean(fwd):+.3f}% win={sum(1 for x in fwd if x > 0)/len(fwd)*100:.1f}% cvd%={cvd_pct:.1f}%')

# Save
def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o

# Don't save all sweeps (too large), just summary
out = {
    'total_sweeps': len(sweeps),
    'with_cvd_div': len(with_div),
    'without_cvd_div': len(without_div),
    'with_cvd_16h_mean': round(np.mean([s['fwd_16h'] for s in with_div]), 4) if with_div else None,
    'without_cvd_16h_mean': round(np.mean([s['fwd_16h'] for s in without_div]), 4) if without_div else None,
    'with_cvd_16h_wr': round(sum(1 for s in with_div if s['fwd_16h'] > 0) / len(with_div) * 100, 1) if with_div else None,
    'without_cvd_16h_wr': round(sum(1 for s in without_div if s['fwd_16h'] > 0) / len(without_div) * 100, 1) if without_div else None,
}
with open(f'{BASE}/reports/fb_v8_isolation.json', 'w') as f:
    json.dump(out, f, indent=2, default=conv)

print('\nDone')
