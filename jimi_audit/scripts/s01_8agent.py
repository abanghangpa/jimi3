"""
8-Agent Forensic Protocol: S01 Failed Breakout v2
==================================================
Hypothesis: Failed breakouts (false break → reversal) have edge, but only in
ranging/choppy regimes. In trending markets, breakouts are more likely to succeed.

Research basis:
- Brunnermeier & Pedersen (2009): Stop-loss cascades create liquidity spirals
- Wyckoff: Springs (false breaks below support) and upthrusts (false breaks above resistance)
- Cespa & Foucault (2022): Price discovery and liquidity provision in limit order markets

S01 v2 detects:
1. Price breaks above/below a swing level
2. Breakout holds for 1+ bars
3. Price reverses back inside the range
4. Filters: session, EMA200, volume regime

Agents:
1. Forensics     — False breakout event identification & data coverage
2. Non-Indicator — Raw breakout failure → reversal edge
3. Indicator     — S01 v2 trigger replication
4. Regime        — Edge by regime (the key question)
5. Gate          — Signal frequency & structural TP/SL analysis
6. Co-occurrence — Confluence with other signals
7. Sensitivity   — Threshold sweep (quality, bars_held, vol)
8. Monte Carlo   — Statistical significance
"""

import pandas as pd
import numpy as np
from scipy import stats
import json, os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'
OUTPUT_FILE = '/root/.openclaw/workspace/jimi_audit/reports/s01_failed_breakout_forensic.json'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print("Loading data...")
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
for c in ['Close','High','Low','Volume']: ohlcv[c] = ohlcv[c].astype(float)
print(f"OHLCV: {len(ohlcv)} bars, {ohlcv['timestamp'].min()} to {ohlcv['timestamp'].max()}")

deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

merged = pd.merge_asof(
    ohlcv[['timestamp','Open','High','Low','Close','Volume']],
    deriv[['timestamp','oi','ls_ratio','funding_rate']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)

# Features
merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()
merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')
merged['atr'] = (merged['High'] - merged['Low']).rolling(14).mean()

for h in [1, 4, 16, 24]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

# Regime
vols = merged['vol_20bar'].dropna()
p33, p67 = vols.quantile(0.33), vols.quantile(0.67)
merged['vol_regime'] = 'MID'
merged.loc[merged['vol_20bar'] < p33, 'vol_regime'] = 'LOW'
merged.loc[merged['vol_20bar'] > p67, 'vol_regime'] = 'HIGH'

def get_era(ts):
    if ts < pd.Timestamp('2024-07-01'): return '2024_H1'
    elif ts < pd.Timestamp('2025-01-01'): return '2024_H2'
    elif ts < pd.Timestamp('2025-07-01'): return '2025_H1'
    elif ts < pd.Timestamp('2026-01-01'): return '2025_H2'
    else: return '2026'
merged['era'] = merged['timestamp'].apply(get_era)
merged['is_july_2026'] = (merged['timestamp'] >= '2026-07-01') & (merged['timestamp'] < '2026-08-01')

# Session hour
merged['hour'] = merged['timestamp'].dt.hour
GOOD_HOURS = {9, 10, 11, 12, 14, 15, 16, 18}

round_trip_cost = 0.0010
results = {}


# ═══════════════════════════════════════════════════════════════
# AGENT 1: FORENSICS — False breakout event identification
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 1: FORENSICS — False breakout events")
print("="*70)

a1 = {}

# Detect failed breakouts using the same logic as S01
def detect_failed_breakouts(df, start_idx=48):
    """Detect all failed breakouts in the dataset."""
    events = []
    closes = df['Close'].values.astype(float)
    highs = df['High'].values.astype(float)
    lows = df['Low'].values.astype(float)
    
    for idx in range(start_idx, len(df)):
        lookback = min(48, idx)
        swing_high = float(np.max(highs[idx-lookback:idx]))
        swing_low = float(np.min(lows[idx-lookback:idx]))
        
        # Check failed breakout above
        for lb in range(1, min(8, idx)):
            bar_idx = idx - lb
            if highs[bar_idx] > swing_high * 1.001:  # broke above
                if closes[bar_idx] < swing_high:  # closed below (failed)
                    # Confirm: current bar still below
                    if closes[idx] < swing_high:
                        bars_held = 0
                        for j in range(bar_idx, idx + 1):
                            if highs[j] > swing_high:
                                bars_held += 1
                            else:
                                break
                        events.append({
                            'idx': idx, 'direction': 'SHORT',
                            'level': swing_high, 'level_type': 'swing_high',
                            'bars_held': bars_held,
                            'quality': min(bars_held / 5, 1.0),
                            'breakout_bar': bar_idx,
                        })
                        break
        
        # Check failed breakout below
        for lb in range(1, min(8, idx)):
            bar_idx = idx - lb
            if lows[bar_idx] < swing_low * 0.999:  # broke below
                if closes[bar_idx] > swing_low:  # closed above (failed)
                    if closes[idx] > swing_low:
                        bars_held = 0
                        for j in range(bar_idx, idx + 1):
                            if lows[j] < swing_low:
                                bars_held += 1
                            else:
                                break
                        events.append({
                            'idx': idx, 'direction': 'LONG',
                            'level': swing_low, 'level_type': 'swing_low',
                            'bars_held': bars_held,
                            'quality': min(bars_held / 5, 1.0),
                            'breakout_bar': bar_idx,
                        })
                        break
    
    return events

print("Detecting failed breakouts (this may take a moment)...")
fb_events = detect_failed_breakouts(merged)
print(f"  Total false breakout events: {len(fb_events)}")

# Quality distribution
qualities = [e['quality'] for e in fb_events]
bars_held = [e['bars_held'] for e in fb_events]
a1['total_events'] = len(fb_events)
a1['quality_mean'] = float(np.mean(qualities))
a1['quality_median'] = float(np.median(qualities))
a1['bars_held_mean'] = float(np.mean(bars_held))
a1['bars_held_median'] = float(np.median(bars_held))

# Direction split
long_events = [e for e in fb_events if e['direction'] == 'LONG']
short_events = [e for e in fb_events if e['direction'] == 'SHORT']
a1['long_events'] = len(long_events)
a1['short_events'] = len(short_events)

# Quality buckets
for q_min, q_max, label in [(0, 0.2, 'low'), (0.2, 0.5, 'medium'), (0.5, 1.0, 'high')]:
    n = sum(1 for q in qualities if q_min <= q < q_max)
    a1[f'quality_{label}'] = n
    print(f"  Quality {label} ({q_min}-{q_max}): {n}")

# July 2026
july_fb = [e for e in fb_events if merged.iloc[e['idx']]['is_july_2026']]
a1['july_2026'] = len(july_fb)
print(f"  July 2026: {len(july_fb)}")
print(f"  Direction: LONG={len(long_events)}, SHORT={len(short_events)}")

results['agent_1'] = a1


# ═══════════════════════════════════════════════════════════════
# AGENT 2: NON-INDICATOR — Raw breakout failure edge
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 2: NON-INDICATOR — Raw false breakout edge")
print("="*70)

a2 = {}

# Test raw edge by direction
for direction in ['LONG', 'SHORT']:
    dir_events = [e for e in fb_events if e['direction'] == direction]
    if len(dir_events) < 5:
        print(f"  {direction}: n={len(dir_events)} (too few)")
        continue
    
    indices = [e['idx'] for e in dir_events]
    for h, label in [(1, '15m'), (4, '1h'), (16, '4h')]:
        rets = merged.iloc[indices][f'fwd_ret_{h}'].dropna()
        if len(rets) < 3:
            continue
        mean_r = rets.mean()
        # For SHORT, negative return = win
        if direction == 'SHORT':
            effective_mean = -mean_r
        else:
            effective_mean = mean_r
        
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets > 0).mean() if direction == 'LONG' else (rets < 0).mean()
        gate = "PASS" if p < 0.1 and effective_mean > round_trip_cost else "FAIL"
        key = f"{direction}_{label}"
        a2[key] = {'n': len(rets), 'mean': float(mean_r), 'effective_mean': float(effective_mean), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} {direction:5s} {label}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, eff={effective_mean*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# Test by quality bucket
print("\n  --- Quality breakdown ---")
for q_min, q_max, label in [(0, 0.2, 'low'), (0.2, 0.5, 'medium'), (0.5, 1.0, 'high')]:
    q_events = [e for e in fb_events if q_min <= e['quality'] < q_max]
    if len(q_events) < 5:
        continue
    indices = [e['idx'] for e in q_events]
    rets = merged.iloc[indices]['fwd_ret_16'].dropna()
    if len(rets) < 3:
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
    a2[f'quality_{label}_4h'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
    print(f"  {'+' if gate=='PASS' else '-'} quality {label:6s} 4h: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# Test by bars_held
print("\n  --- Bars held breakdown ---")
for bh_min, bh_max, label in [(1, 2, '1bar'), (2, 4, '2-3bars'), (4, 10, '4+bars')]:
    bh_events = [e for e in fb_events if bh_min <= e['bars_held'] < bh_max]
    if len(bh_events) < 5:
        continue
    indices = [e['idx'] for e in bh_events]
    rets = merged.iloc[indices]['fwd_ret_16'].dropna()
    if len(rets) < 3:
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
    a2[f'bars_{label}_4h'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
    print(f"  {'+' if gate=='PASS' else '-'} bars {label:10s} 4h: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_2'] = a2


# ═══════════════════════════════════════════════════════════════
# AGENT 3: INDICATOR — S01 v2 trigger replication
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 3: INDICATOR — S01 v2 trigger replication")
print("="*70)

a3 = {}

# Apply S01's filters: session, EMA200, volume, quality
s01_events = []
for e in fb_events:
    idx = e['idx']
    row = merged.iloc[idx]
    
    # Session filter
    hour = row['hour']
    if hour not in GOOD_HOURS:
        continue
    
    # Quality filter
    if e['bars_held'] < 1:
        continue
    
    # EMA200 filter (3% band)
    ema200 = row['ema200']
    price = row['Close']
    if ema200 > 0:
        dist = (price - ema200) / ema200
        if e['direction'] == 'LONG' and dist < -0.03:
            continue
        if e['direction'] == 'SHORT' and dist > 0.03:
            continue
    
    # Volume filter
    vol_ratio = row['vol_ratio'] if pd.notna(row['vol_ratio']) else 1.0
    # In ranging, prefer low volume
    if vol_ratio > 2.0:
        continue
    
    s01_events.append(e)

a3['s01_signals'] = len(s01_events)
print(f"  S01 v2 signals (after all filters): {len(s01_events)}")

if len(s01_events) >= 5:
    indices = [e['idx'] for e in s01_events]
    for h, label in [(1, '15m'), (4, '1h'), (16, '4h')]:
        rets = merged.iloc[indices][f'fwd_ret_{h}'].dropna()
        if len(rets) < 3:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        a3[f's01_{label}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} S01 {label}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

    july_s01 = [e for e in s01_events if merged.iloc[e['idx']]['is_july_2026']]
    a3['s01_july'] = len(july_s01)
    print(f"  S01 July 2026: {len(july_s01)}")

# Compare: with vs without session filter
no_session = [e for e in fb_events if e['bars_held'] >= 1]
if len(no_session) >= 5:
    indices = [e['idx'] for e in no_session]
    rets = merged.iloc[indices]['fwd_ret_16'].dropna()
    a3['no_session_filter'] = {'n': len(rets), 'mean': float(rets.mean()), 'p': float(stats.ttest_1samp(rets, 0)[1])}
    print(f"  Without session filter: n={len(rets)}, mean={rets.mean()*100:+.4f}%")

results['agent_3'] = a3


# ═══════════════════════════════════════════════════════════════
# AGENT 4: REGIME — Edge by regime (THE KEY QUESTION)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 4: REGIME — Edge by regime")
print("="*70)

a4 = {}

all_fb = fb_events if len(s01_events) < 10 else s01_events
indices_all = [e['idx'] for e in all_fb]

for col, name in [('vol_regime', 'Vol Regime'), ('trend', 'Trend'), ('era', 'Era')]:
    print(f"\n  --- {name} ---")
    for reg in sorted(merged[col].dropna().unique()):
        reg_indices = [i for i in indices_all if merged.iloc[i][col] == reg]
        if len(reg_indices) < 3:
            print(f"    {reg}: n={len(reg_indices)} (too few)")
            continue
        rets = merged.iloc[reg_indices]['fwd_ret_16'].dropna()
        if len(rets) < 3:
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        a4[f'{col}_{reg}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"    {'+' if gate=='PASS' else '-'} {reg:12s}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# EMA200 distance analysis
print("\n  --- EMA200 Distance ---")
for dist_min, dist_max, label in [(-0.05, -0.02, 'below_2-5pct'), (-0.02, 0.02, 'near'), (0.02, 0.05, 'above_2-5pct')]:
    dist_indices = [i for i in indices_all if dist_min <= (merged.iloc[i]['Close'] - merged.iloc[i]['ema200']) / merged.iloc[i]['ema200'] < dist_max]
    if len(dist_indices) < 3:
        continue
    rets = merged.iloc[dist_indices]['fwd_ret_16'].dropna()
    if len(rets) < 3:
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
    a4[f'ema200_{label}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'gate': gate}
    print(f"  {'+' if gate=='PASS' else '-'} EMA200 {label:15s}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}")

results['agent_4'] = a4


# ═══════════════════════════════════════════════════════════════
# AGENT 5: GATE — Signal frequency & structural TP/SL
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 5: GATE — Signal frequency & structural levels")
print("="*70)

months = max(1, (merged['timestamp'].max() - merged['timestamp'].min()).days / 30)
a5 = {
    'total_raw': len(fb_events),
    'total_filtered': len(s01_events),
    'per_month_raw': len(fb_events) / months,
    'per_month_filtered': len(s01_events) / months,
}
print(f"  Raw events: {len(fb_events)} ({len(fb_events)/months:.1f}/month)")
print(f"  Filtered (S01): {len(s01_events)} ({len(s01_events)/months:.1f}/month)")

# Structural TP/SL: where would TP be based on swing levels?
# For SHORT: TP at recent swing low, SL above breakout level
# For LONG: TP at recent swing high, SL below breakout level
print("\n  --- Structural TP/SL analysis ---")
tp_hits = {'structural': 0, 'fixed_2pct': 0, 'total': 0}
for e in s01_events[:100]:  # sample
    idx = e['idx']
    if idx + 16 >= len(merged):
        continue
    
    entry = merged.iloc[idx]['Close']
    closes = merged.iloc[idx+1:idx+17]['Close'].values
    
    if e['direction'] == 'SHORT':
        # Structural TP: level below
        struct_tp = e['level'] * 0.99  # 1% below breakout level
        fixed_tp = entry * 0.98  # 2% below entry
        tp_hits['total'] += 1
        if any(c <= struct_tp for c in closes):
            tp_hits['structural'] += 1
        if any(c <= fixed_tp for c in closes):
            tp_hits['fixed_2pct'] += 1
    else:
        struct_tp = e['level'] * 1.01
        fixed_tp = entry * 1.02
        tp_hits['total'] += 1
        if any(c >= struct_tp for c in closes):
            tp_hits['structural'] += 1
        if any(c >= fixed_tp for c in closes):
            tp_hits['fixed_2pct'] += 1

if tp_hits['total'] > 0:
    a5['structural_tp_rate'] = tp_hits['structural'] / tp_hits['total']
    a5['fixed_tp_rate'] = tp_hits['fixed_2pct'] / tp_hits['total']
    print(f"  Structural TP hit rate: {tp_hits['structural']}/{tp_hits['total']} ({tp_hits['structural']/tp_hits['total']:.1%})")
    print(f"  Fixed 2% TP hit rate: {tp_hits['fixed_2pct']}/{tp_hits['total']} ({tp_hits['fixed_2pct']/tp_hits['total']:.1%})")

results['agent_5'] = a5


# ═══════════════════════════════════════════════════════════════
# AGENT 6: CO-OCCURRENCE — Confluence with other signals
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 6: CO-OCCURRENCE")
print("="*70)

a6 = {}
if len(s01_events) > 0:
    indices = [e['idx'] for e in s01_events]
    
    # Volume during false breakouts
    vr = merged.iloc[indices]['vol_ratio'].dropna()
    if len(vr) > 0:
        a6['vol_ratio_mean'] = float(vr.mean())
        print(f"  Vol ratio: {vr.mean():.2f}x")
    
    # LS ratio during false breakouts
    ls = merged.iloc[indices]['ls_ratio'].dropna()
    if len(ls) > 0:
        a6['ls_ratio_mean'] = float(ls.mean())
        print(f"  LS ratio: {ls.mean():.3f}")
    
    # Direction distribution
    long_n = sum(1 for e in s01_events if e['direction'] == 'LONG')
    short_n = sum(1 for e in s01_events if e['direction'] == 'SHORT')
    a6['long_pct'] = long_n / len(s01_events)
    print(f"  Direction: {long_n}L / {short_n}S ({long_n/len(s01_events):.0%} LONG)")

results['agent_6'] = a6


# ═══════════════════════════════════════════════════════════════
# AGENT 7: SENSITIVITY — Threshold sweep
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 7: SENSITIVITY — Quality & bars_held sweep")
print("="*70)

a7 = {}

# Sweep quality threshold
print("\n  --- Quality threshold ---")
for q_min in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    q_events = [e for e in fb_events if e['quality'] >= q_min and e['bars_held'] >= 1]
    if len(q_events) < 5:
        continue
    indices = [e['idx'] for e in q_events]
    rets = merged.iloc[indices]['fwd_ret_16'].dropna()
    if len(rets) < 5:
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
    if gate == "PASS":
        a7[f'quality>={q_min}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr)}
        print(f"  + quality>={q_min}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# Sweep bars_held threshold
print("\n  --- Bars held threshold ---")
for bh_min in [1, 2, 3, 4, 5]:
    bh_events = [e for e in fb_events if e['bars_held'] >= bh_min]
    if len(bh_events) < 5:
        continue
    indices = [e['idx'] for e in bh_events]
    rets = merged.iloc[indices]['fwd_ret_16'].dropna()
    if len(rets) < 5:
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
    if gate == "PASS":
        a7[f'bars_held>={bh_min}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr)}
        print(f"  + bars_held>={bh_min}: n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# Combined: quality + session + direction
print("\n  --- Best combo ---")
for direction in ['LONG', 'SHORT']:
    for q_min in [0.2, 0.3, 0.5]:
        combo = [e for e in fb_events if e['direction'] == direction and e['quality'] >= q_min and e['bars_held'] >= 2]
        if len(combo) < 5:
            continue
        indices = [e['idx'] for e in combo]
        rets = merged.iloc[indices]['fwd_ret_16'].dropna()
        if len(rets) < 3:
            continue
        mean_r = rets.mean()
        if direction == 'SHORT':
            effective_mean = -mean_r
        else:
            effective_mean = mean_r
        t, p = stats.ttest_1samp(rets, 0)
        wr = (rets > 0).mean() if direction == 'LONG' else (rets < 0).mean()
        gate = "PASS" if p < 0.1 and effective_mean > round_trip_cost else "FAIL"
        if gate == "PASS":
            key = f'{direction}_q>={q_min}_bh>=2'
            a7[key] = {'n': len(rets), 'mean': float(mean_r), 'effective_mean': float(effective_mean), 'p': float(p), 'wr': float(wr)}
            print(f"  + {direction} q>={q_min} bh>=2: n={len(rets)}, mean={mean_r*100:+.4f}%, eff={effective_mean*100:+.4f}%, WR={wr:.1%}")

results['agent_7'] = a7


# ═══════════════════════════════════════════════════════════════
# AGENT 8: MONTE CARLO — Statistical significance
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 8: MONTE CARLO")
print("="*70)

a8 = {}

test_events = s01_events if len(s01_events) >= 10 else fb_events
if len(test_events) >= 5:
    indices = [e['idx'] for e in test_events]
    actual = merged.iloc[indices]['fwd_ret_16'].dropna()
    am, aw, n = actual.mean(), (actual > 0).mean(), len(actual)
    
    print(f"  n={n}, mean={am*100:+.4f}%, WR={aw:.1%}")
    
    np.random.seed(42)
    all_r = merged['fwd_ret_16'].dropna()
    rm = np.array([all_r.sample(n).mean() for _ in range(10000)])
    pm = (rm >= am).mean()
    
    bm = np.array([actual.sample(n, replace=True).mean() for _ in range(10000)])
    ci_lo, ci_hi = np.percentile(bm, 2.5), np.percentile(bm, 97.5)
    
    a8 = {'n': n, 'mean': float(am), 'wr': float(aw), 'mc_p': float(pm),
          'ci': [float(ci_lo), float(ci_hi)], 'sig': pm < 0.05}
    print(f"  MC p: {pm:.4f}")
    print(f"  CI: [{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%]")
    print(f"  SIGNIFICANT: {'YES' if pm < 0.05 else 'NO'}")
else:
    a8 = {'n': len(test_events), 'sig': False}
    print(f"  Too few ({len(test_events)})")

results['agent_8'] = a8


# ═══════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("VERDICT")
print("="*70)

v = {
    'strategy': 'S01 Failed Breakout v2',
    'raw_events': len(fb_events),
    'filtered_events': len(s01_events),
    'mc_sig': a8.get('sig', False),
}

# Find best regime
best_regime = None
best_mean = 0
for key, val in a4.items():
    if val.get('gate') == 'PASS' and val.get('mean', 0) > best_mean:
        best_mean = val['mean']
        best_regime = key

v['best_regime'] = best_regime
v['best_regime_mean'] = float(best_mean) if best_mean else None

if a8.get('sig'):
    v['gate'] = 'PASS'
    v['rec'] = 'Deploy with regime filter, validate with 30+ trades'
elif best_regime and best_mean > 0.003:
    v['gate'] = 'CONDITIONAL'
    v['rec'] = f'Deploy only in {best_regime} regime'
elif len(s01_events) < 10:
    v['gate'] = 'LOW_SAMPLE'
    v['rec'] = 'Too few signals after filters — relax session or quality filter'
else:
    v['gate'] = 'FAIL'
    v['rec'] = 'No edge found'

print(f"  Raw events: {v['raw_events']}")
print(f"  Filtered: {v['filtered_events']}")
print(f"  Best regime: {v['best_regime']} ({v.get('best_regime_mean', 0)*100:+.4f}%)")
print(f"  Gate: {v['gate']}")
print(f"  Rec: {v['rec']}")
results['verdict'] = v

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUTPUT_FILE}")
