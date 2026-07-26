"""
S01 v3 Full Backtest — M14 + M21 + Derivatives on 2.5 years of data.
============================================================
Runs M14 sweep detection and M21 Wyckoff phase on the FULL OHLCV dataset,
not just 13 days of scan files.

This gives us:
- M14 sweeps across all regimes and eras
- M21 Wyckoff phases (Accumulation/Markup/Distribution/Markdown)
- Combined signal: sweep + Wyckoff context + derivatives
"""

import pandas as pd, numpy as np, json, os
from scipy import stats
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'
OUTPUT = '/root/.openclaw/workspace/jimi_audit/reports/s01_v3_full_backtest.json'
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("Loading data...")
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
for c in ['Close','High','Low','Open','Volume']: ohlcv[c] = ohlcv[c].astype(float)
print(f"OHLCV: {len(ohlcv)} bars")

deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

merged = pd.merge_asof(
    ohlcv[['timestamp','Open','High','Low','Close','Volume']],
    deriv[['timestamp','oi','ls_ratio','funding_rate']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)

merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()
merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')
merged['atr'] = (merged['High'] - merged['Low']).rolling(14).mean()
merged['hour'] = merged['timestamp'].dt.hour
GOOD_HOURS = {9, 10, 11, 12, 14, 15, 16, 18}

for h in [1, 4, 16, 24]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

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

round_trip_cost = 0.0010

# ═══════════════════════════════════════════════════════════════
# M14: SWEEP DETECTION (from m14_sweep.py logic)
# ═══════════════════════════════════════════════════════════════
print("\nRunning M14 sweep detection on full dataset...")

highs = merged['High'].values.astype(float)
lows = merged['Low'].values.astype(float)
closes = merged['Close'].values.astype(float)
opens = merged['Open'].values.astype(float)
volumes = merged['Volume'].values.astype(float)

M14_SWEEP_LOOKBACK = 20
M14_SWEEP_DEPTH_MIN = 0.001  # 0.1%
M14_SWEEP_DEPTH_MAX = 0.020  # 2%
M14_RECLAIM_BARS = 3
M14_RECLAIM_WICK_RATIO = 0.40
M14_VOL_CONFIRM_MULT = 1.2

def find_swing_levels(highs, lows, idx, lookback=48):
    """Find swing highs and lows (simple pivot detection)."""
    swing_highs = []
    swing_lows = []
    start = max(0, idx - lookback)
    
    for i in range(start + 2, idx - 1):
        # Swing high: higher than 2 bars on each side
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1]:
            swing_highs.append((highs[i], i))
        # Swing low: lower than 2 bars on each side
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1]:
            swing_lows.append((lows[i], i))
    
    return swing_highs, swing_lows

def detect_m14_sweep(idx, direction, swing_highs, swing_lows):
    """Detect sweep of a swing level (M14 logic)."""
    levels = swing_lows if direction == 'LONG' else swing_highs
    
    for level_price, level_idx in levels:
        start = max(0, idx - M14_SWEEP_LOOKBACK)
        for bar_i in range(start, idx + 1):
            bar_range = highs[bar_i] - lows[bar_i]
            if bar_range <= 0:
                continue
            
            if direction == 'LONG':
                sweep_depth = (level_price - lows[bar_i]) / level_price
                if M14_SWEEP_DEPTH_MIN <= sweep_depth <= M14_SWEEP_DEPTH_MAX:
                    reclaimed = closes[bar_i] > level_price
                    lower_wick = min(opens[bar_i], closes[bar_i]) - lows[bar_i]
                    wick_ratio = lower_wick / bar_range
                    return True, {
                        'sweep_bar': bar_i, 'bars_ago': idx - bar_i,
                        'level': level_price, 'depth_pct': sweep_depth * 100,
                        'reclaimed': reclaimed, 'wick_ratio': wick_ratio,
                    }
            
            elif direction == 'SHORT':
                sweep_depth = (highs[bar_i] - level_price) / level_price
                if M14_SWEEP_DEPTH_MIN <= sweep_depth <= M14_SWEEP_DEPTH_MAX:
                    reclaimed = closes[bar_i] < level_price
                    upper_wick = highs[bar_i] - max(opens[bar_i], closes[bar_i])
                    wick_ratio = upper_wick / bar_range
                    return True, {
                        'sweep_bar': bar_i, 'bars_ago': idx - bar_i,
                        'level': level_price, 'depth_pct': sweep_depth * 100,
                        'reclaimed': reclaimed, 'wick_ratio': wick_ratio,
                    }
    
    return False, {}

def check_reclaim(idx, direction, sweep_details):
    """Check if price reclaimed the level after sweep (M14 reclaim logic)."""
    sweep_bar = sweep_details['sweep_bar']
    level = sweep_details['level']
    bars_since = idx - sweep_bar
    
    if bars_since > M14_RECLAIM_BARS + 5:
        return 'NONE', {'reason': 'stale'}
    
    bar_range = highs[idx] - lows[idx]
    if bar_range <= 0:
        return 'NONE', {'reason': 'zero_range'}
    
    vol_avg = np.mean(volumes[max(0, idx-20):idx]) if idx >= 20 else volumes[idx]
    vol_ok = volumes[idx] > vol_avg * M14_VOL_CONFIRM_MULT
    
    if direction == 'LONG':
        if closes[idx] <= level:
            return 'NONE', {'reason': 'not_reclaimed'}
        lower_wick = min(opens[idx], closes[idx]) - lows[idx]
        wick_ratio = lower_wick / bar_range
        
        if wick_ratio >= M14_RECLAIM_WICK_RATIO and closes[idx] > opens[idx]:
            if vol_ok:
                return 'STRONG', {'wick_ratio': wick_ratio, 'vol': True}
            else:
                return 'WEAK', {'wick_ratio': wick_ratio, 'vol': False}
        if closes[idx] > opens[idx] and closes[idx] > level:
            return 'WEAK', {'wick_ratio': wick_ratio, 'vol': vol_ok, 'note': 'green_above'}
    
    elif direction == 'SHORT':
        if closes[idx] >= level:
            return 'NONE', {'reason': 'not_reclaimed'}
        upper_wick = highs[idx] - max(opens[idx], closes[idx])
        wick_ratio = upper_wick / bar_range
        
        if wick_ratio >= M14_RECLAIM_WICK_RATIO and closes[idx] < opens[idx]:
            if vol_ok:
                return 'STRONG', {'wick_ratio': wick_ratio, 'vol': True}
            else:
                return 'WEAK', {'wick_ratio': wick_ratio, 'vol': False}
        if closes[idx] < opens[idx] and closes[idx] < level:
            return 'WEAK', {'wick_ratio': wick_ratio, 'vol': vol_ok, 'note': 'red_below'}
    
    return 'NONE', {'reason': 'no_reclaim'}

# ═══════════════════════════════════════════════════════════════
# M21: WYCKOFF PHASE (from m21_wyckoff.py logic)
# ═══════════════════════════════════════════════════════════════
print("Running M21 Wyckoff detection...")

def detect_wyckoff_for_bar(idx):
    """Detect Wyckoff phase at each bar using 4H structure."""
    if idx < 96:
        return {'phase': 'RANGE', 'confidence': 0.3, 'zone': 'UNKNOWN'}
    
    # Use ~48 4H bars (768 15m bars) for structure
    lookback = min(768, idx)
    h4_highs = highs[idx-lookback:idx+1]
    h4_lows = lows[idx-lookback:idx+1]
    h4_closes = closes[idx-lookback:idx+1]
    
    # Recent vs prior half
    half = len(h4_closes) // 2
    recent_10_hi = h4_highs[-min(10, half):].max()
    prior_10_hi = h4_highs[-min(20, half):-min(10, half)].max() if len(h4_highs) > 10 else recent_10_hi
    recent_10_lo = h4_lows[-min(10, half):].min()
    prior_10_lo = h4_lows[-min(20, half):-min(10, half)].min() if len(h4_lows) > 10 else recent_10_lo
    
    hh = recent_10_hi > prior_10_hi
    hl = recent_10_lo > prior_10_lo
    lh = recent_10_hi < prior_10_hi
    ll = recent_10_lo < prior_10_lo
    
    # Range detection
    range_hi = float(h4_highs.max())
    range_lo = float(h4_lows.min())
    eq = (range_hi + range_lo) / 2
    current = float(h4_closes[-1])
    position = (current - range_lo) / (range_hi - range_lo) if range_hi > range_lo else 0.5
    
    # Phase
    phase = 'RANGE'
    confidence = 0.3
    
    if hh and hl:
        if position > 0.7:
            phase = 'DISTRIBUTION'
            confidence = 0.6
        elif position > 0.5:
            phase = 'MARKUP'
            confidence = 0.7
        else:
            phase = 'ACCUMULATION'
            confidence = 0.5
    elif lh and ll:
        if position < 0.3:
            phase = 'ACCUMULATION'
            confidence = 0.6
        elif position < 0.5:
            phase = 'MARKDOWN'
            confidence = 0.7
        else:
            phase = 'DISTRIBUTION'
            confidence = 0.5
    
    # Zone
    if position > 0.55:
        zone = 'PREMIUM'
    elif position < 0.45:
        zone = 'DISCOUNT'
    else:
        zone = 'EQUILIBRIUM'
    
    # Spring/upthrust detection
    spring = False
    upthrust = False
    if phase == 'ACCUMULATION' and position < 0.25:
        spring = True
    if phase == 'DISTRIBUTION' and position > 0.75:
        upthrust = True
    
    return {
        'phase': phase, 'confidence': confidence, 'zone': zone,
        'spring': spring, 'upthrust': upthrust,
        'position': position, 'hh': hh, 'hl': hl, 'lh': lh, 'll': ll,
    }

# ═══════════════════════════════════════════════════════════════
# MAIN LOOP: Run M14 + M21 on every bar
# ═══════════════════════════════════════════════════════════════
print("Running combined M14 + M21 on full dataset (this takes a while)...")

sweep_events = []
seen = set()

# Process every 4th bar for speed (still 22k+ samples)
step = 4
for idx in range(96, len(merged), step):
    if idx % 10000 == 0:
        print(f"  Processing bar {idx}/{len(merged)}...")
    
    # Find swing levels
    swing_highs, swing_lows = find_swing_levels(highs, lows, idx, lookback=48)
    
    # Get Wyckoff phase
    wyckoff = detect_wyckoff_for_bar(idx)
    
    # Check both directions
    for direction in ['LONG', 'SHORT']:
        found, details = detect_m14_sweep(idx, direction, swing_highs, swing_lows)
        if not found:
            continue
        
        # Deduplicate
        key = (details['sweep_bar'], direction)
        if key in seen:
            continue
        seen.add(key)
        
        # Check reclaim
        reclaim_type, reclaim_details = check_reclaim(idx, direction, details)
        
        # M14 score
        if reclaim_type == 'STRONG':
            m14_score = 0.85
            m14_signal = 'STRONG_RECLAIM'
        elif reclaim_type == 'WEAK':
            m14_score = 0.55
            m14_signal = 'WEAK_RECLAIM'
        else:
            m14_score = 0.30
            m14_signal = 'NO_RECLAIM'
        
        # Positioning
        ls = merged.iloc[idx]['ls_ratio'] if pd.notna(merged.iloc[idx]['ls_ratio']) else 1.0
        fr = merged.iloc[idx]['funding_rate'] if pd.notna(merged.iloc[idx]['funding_rate']) else 0
        
        sweep_events.append({
            'idx': idx, 'direction': direction,
            'sweep_depth': details['depth_pct'],
            'sweep_bar': details['sweep_bar'],
            'bars_ago': details['bars_ago'],
            'wick_ratio': details.get('wick_ratio', 0),
            'reclaimed': details.get('reclaimed', False),
            'm14_score': m14_score,
            'm14_signal': m14_signal,
            'reclaim_type': reclaim_type,
            'wyckoff_phase': wyckoff['phase'],
            'wyckoff_zone': wyckoff['zone'],
            'wyckoff_confidence': wyckoff['confidence'],
            'wyckoff_spring': wyckoff['spring'],
            'wyckoff_upthrust': wyckoff['upthrust'],
            'wyckoff_position': wyckoff['position'],
            'ls_ratio': ls,
            'funding_rate': fr,
            'vol_ratio': merged.iloc[idx]['vol_ratio'] if pd.notna(merged.iloc[idx]['vol_ratio']) else 1.0,
        })

print(f"  Total M14 sweep events: {len(sweep_events)}")

results = {}

# ═══════════ AGENT 1: FORENSICS ═══════════
print("\n" + "="*70)
print("AGENT 1: FORENSICS — M14 + M21 full dataset")
print("="*70)

a1 = {
    'total': len(sweep_events),
    'long': sum(1 for e in sweep_events if e['direction'] == 'LONG'),
    'short': sum(1 for e in sweep_events if e['direction'] == 'SHORT'),
    'strong_reclaim': sum(1 for e in sweep_events if e['reclaim_type'] == 'STRONG'),
    'weak_reclaim': sum(1 for e in sweep_events if e['reclaim_type'] == 'WEAK'),
    'no_reclaim': sum(1 for e in sweep_events if e['reclaim_type'] == 'NONE'),
}
print(f"  Total: {a1['total']}, LONG: {a1['long']}, SHORT: {a1['short']}")
print(f"  STRONG reclaim: {a1['strong_reclaim']}, WEAK: {a1['weak_reclaim']}, NONE: {a1['no_reclaim']}")

# Wyckoff phase distribution
phases = {}
for e in sweep_events:
    p = e['wyckoff_phase']
    phases[p] = phases.get(p, 0) + 1
print(f"  Wyckoff phases: {phases}")

# Wyckoff zone distribution
zones = {}
for e in sweep_events:
    z = e['wyckoff_zone']
    zones[z] = zones.get(z, 0) + 1
print(f"  Wyckoff zones: {zones}")

# Springs/upthrusts
a1['springs'] = sum(1 for e in sweep_events if e['wyckoff_spring'])
a1['upthrusts'] = sum(1 for e in sweep_events if e['wyckoff_upthrust'])
print(f"  Springs: {a1['springs']}, Upthrusts: {a1['upthrusts']}")

results['agent_1'] = a1

# ═══════════ AGENT 2: RAW EDGE BY M14 SIGNAL ═══════════
print("\n" + "="*70)
print("AGENT 2: EDGE BY M14 SIGNAL TYPE")
print("="*70)

a2 = {}
for signal in ['STRONG_RECLAIM', 'WEAK_RECLAIM', 'NO_RECLAIM']:
    for direction in ['LONG', 'SHORT']:
        evts = [e for e in sweep_events if e['m14_signal'] == signal and e['direction'] == direction]
        if len(evts) < 5: continue
        indices = [e['idx'] for e in evts]
        for h, label in [(4, '1h'), (16, '4h')]:
            rets = merged.iloc[indices][f'fwd_ret_{h}'].dropna()
            if len(rets) < 3: continue
            mean_r = rets.mean()
            eff = -mean_r if direction == 'SHORT' else mean_r
            t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
            wr = (rets < 0).mean() if direction == 'SHORT' else (rets > 0).mean()
            gate = "PASS" if p < 0.1 and eff > round_trip_cost else "FAIL"
            key = f"{signal}_{direction}_{label}"
            a2[key] = {'n': len(rets), 'eff': float(eff), 'p': float(p), 'wr': float(wr), 'gate': gate}
            print(f"  {'+' if gate=='PASS' else '-'} {signal:15s} {direction:5s} {label}: n={len(rets):4d}, eff={eff*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_2'] = a2

# ═══════════ AGENT 3: EDGE BY WYCKOFF PHASE ═══════════
print("\n" + "="*70)
print("AGENT 3: EDGE BY WYCKOFF PHASE")
print("="*70)

a3 = {}
for phase in ['ACCUMULATION', 'MARKUP', 'DISTRIBUTION', 'MARKDOWN', 'RANGE']:
    for direction in ['LONG', 'SHORT']:
        evts = [e for e in sweep_events if e['wyckoff_phase'] == phase and e['direction'] == direction]
        if len(evts) < 5: continue
        indices = [e['idx'] for e in evts]
        for h, label in [(4, '1h'), (16, '4h')]:
            rets = merged.iloc[indices][f'fwd_ret_{h}'].dropna()
            if len(rets) < 3: continue
            mean_r = rets.mean()
            eff = -mean_r if direction == 'SHORT' else mean_r
            t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
            wr = (rets < 0).mean() if direction == 'SHORT' else (rets > 0).mean()
            gate = "PASS" if p < 0.1 and eff > round_trip_cost else "FAIL"
            key = f"{phase}_{direction}_{label}"
            a3[key] = {'n': len(rets), 'eff': float(eff), 'p': float(p), 'wr': float(wr), 'gate': gate}
            print(f"  {'+' if gate=='PASS' else '-'} {phase:13s} {direction:5s} {label}: n={len(rets):4d}, eff={eff*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# Spring/upthrust specifically
for direction, label in [('LONG', 'spring'), ('SHORT', 'upthrust')]:
    evts = [e for e in sweep_events if (e['wyckoff_spring'] if direction == 'LONG' else e['wyckoff_upthrust'])]
    if len(evts) < 3: continue
    indices = [e['idx'] for e in evts]
    for h, hlabel in [(4, '1h'), (16, '4h')]:
        rets = merged.iloc[indices][f'fwd_ret_{h}'].dropna()
        if len(rets) < 3: continue
        mean_r = rets.mean()
        eff = -mean_r if direction == 'SHORT' else mean_r
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets < 0).mean() if direction == 'SHORT' else (rets > 0).mean()
        gate = "PASS" if p < 0.1 and eff > round_trip_cost else "FAIL"
        a3[f'{label}_{hlabel}'] = {'n': len(rets), 'eff': float(eff), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} {label:10s} {hlabel}: n={len(rets)}, eff={eff*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_3'] = a3

# ═══════════ AGENT 4: COMBINED SIGNAL (M14 + M21 + positioning) ═══════════
print("\n" + "="*70)
print("AGENT 4: COMBINED SIGNAL — M14 + M21 + positioning")
print("="*70)

a4 = {}

# Best combo: STRONG_RECLAIM + ACCUMULATION + LONG
# Or: STRONG_RECLAIM + DISTRIBUTION + SHORT
combos = [
    ('STRONG+ACCUM+LONG', {'m14': 'STRONG_RECLAIM', 'phase': 'ACCUMULATION', 'dir': 'LONG'}),
    ('STRONG+DISTRIB+SHORT', {'m14': 'STRONG_RECLAIM', 'phase': 'DISTRIBUTION', 'dir': 'SHORT'}),
    ('STRONG+MARKUP+LONG', {'m14': 'STRONG_RECLAIM', 'phase': 'MARKUP', 'dir': 'LONG'}),
    ('STRONG+MARKDOWN+SHORT', {'m14': 'STRONG_RECLAIM', 'phase': 'MARKDOWN', 'dir': 'SHORT'}),
    ('WEAK+ACCUM+LONG', {'m14': 'WEAK_RECLAIM', 'phase': 'ACCUMULATION', 'dir': 'LONG'}),
    ('WEAK+DISTRIB+SHORT', {'m14': 'WEAK_RECLAIM', 'phase': 'DISTRIBUTION', 'dir': 'SHORT'}),
    ('STRONG+DISCOUNT+LONG', {'m14': 'STRONG_RECLAIM', 'zone': 'DISCOUNT', 'dir': 'LONG'}),
    ('STRONG+PREMIUM+SHORT', {'m14': 'STRONG_RECLAIM', 'zone': 'PREMIUM', 'dir': 'SHORT'}),
    ('SPRING+LONG', {'spring': True, 'dir': 'LONG'}),
    ('UPthrust+SHORT', {'upthrust': True, 'dir': 'SHORT'}),
]

for name, filters in combos:
    evts = sweep_events
    for k, v in filters.items():
        if k == 'dir':
            evts = [e for e in evts if e['direction'] == v]
        elif k == 'm14':
            evts = [e for e in evts if e['m14_signal'] == v]
        elif k == 'phase':
            evts = [e for e in evts if e['wyckoff_phase'] == v]
        elif k == 'zone':
            evts = [e for e in evts if e['wyckoff_zone'] == v]
        elif k == 'spring':
            evts = [e for e in evts if e['wyckoff_spring']]
        elif k == 'upthrust':
            evts = [e for e in evts if e['wyckoff_upthrust']]
    
    if len(evts) < 3: 
        print(f"  {name}: n={len(evts)} (too few)")
        continue
    
    indices = [e['idx'] for e in evts]
    for h, label in [(4, '1h'), (16, '4h')]:
        rets = merged.iloc[indices][f'fwd_ret_{h}'].dropna()
        if len(rets) < 3: continue
        mean_r = rets.mean()
        direction = filters['dir']
        eff = -mean_r if direction == 'SHORT' else mean_r
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets < 0).mean() if direction == 'SHORT' else (rets > 0).mean()
        gate = "PASS" if p < 0.1 and eff > round_trip_cost else "FAIL"
        key = f"{name}_{label}"
        a4[key] = {'n': len(rets), 'eff': float(eff), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} {name:25s} {label}: n={len(rets):4d}, eff={eff*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_4'] = a4

# ═══════════ AGENT 5: MONTE CARLO ═══════════
print("\n" + "="*70)
print("AGENT 5: MONTE CARLO — Best combined signal")
print("="*70)

# Find best combo
best_key = None; best_eff = 0
for k, v in a4.items():
    if v.get('gate') == 'PASS' and v.get('eff', 0) > best_eff:
        best_eff = v['eff']
        best_key = k

if best_key:
    print(f"  Best combo: {best_key}")
    # Get the events for this combo
    # (re-derive from filters)
    best_cfg = None
    for name, filters in combos:
        if name in best_key:
            best_cfg = filters
            break
    
    if best_cfg:
        evts = sweep_events
        for k, v in best_cfg.items():
            if k == 'dir': evts = [e for e in evts if e['direction'] == v]
            elif k == 'm14': evts = [e for e in evts if e['m14_signal'] == v]
            elif k == 'phase': evts = [e for e in evts if e['wyckoff_phase'] == v]
            elif k == 'zone': evts = [e for e in evts if e['wyckoff_zone'] == v]
            elif k == 'spring': evts = [e for e in evts if e['wyckoff_spring']]
            elif k == 'upthrust': evts = [e for e in evts if e['wyckoff_upthrust']]
        
        indices = [e['idx'] for e in evts]
        actual = merged.iloc[indices]['fwd_ret_16'].dropna()
        am, aw, n = actual.mean(), (actual > 0).mean(), len(actual)
        
        np.random.seed(42)
        all_r = merged['fwd_ret_16'].dropna()
        rm = np.array([all_r.sample(n).mean() for _ in range(10000)])
        pm = (rm >= am).mean()
        
        bm = np.array([actual.sample(n, replace=True).mean() for _ in range(10000)])
        ci_lo, ci_hi = np.percentile(bm, 2.5), np.percentile(bm, 97.5)
        
        mc = {'n': n, 'mean': float(am), 'wr': float(aw), 'mc_p': float(pm),
              'ci': [float(ci_lo), float(ci_hi)], 'sig': bool(pm < 0.05)}
        print(f"  n={n}, mean={am*100:+.4f}%, WR={aw:.1%}")
        print(f"  MC p: {pm:.4f}, CI: [{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%]")
        print(f"  SIGNIFICANT: {'YES' if pm < 0.05 else 'NO'}")
        results['agent_5_mc'] = mc
else:
    print("  No passing combos found")
    results['agent_5_mc'] = {'sig': False}

# ═══════════ VERDICT ═══════════
print("\n" + "="*70)
print("VERDICT")
print("="*70)

verdict = {
    'total_sweep_events': len(sweep_events),
    'm14_signals': {
        'strong': a1['strong_reclaim'],
        'weak': a1['weak_reclaim'],
        'none': a1['no_reclaim'],
    },
    'wyckoff': phases,
    'best_combo': best_key,
    'mc_sig': results.get('agent_5_mc', {}).get('sig', False),
}

if results.get('agent_5_mc', {}).get('sig'):
    verdict['gate'] = 'PASS'
    verdict['rec'] = 'Deploy with 0.5x size'
elif best_key and best_eff > 0.003:
    verdict['gate'] = 'MARGINAL'
    verdict['rec'] = f'Deploy {best_key} with 0.3x size, validate with 30+ trades'
else:
    verdict['gate'] = 'FAIL'
    verdict['rec'] = 'No edge found'

print(f"  Total sweep events: {len(sweep_events)}")
print(f"  Best combo: {best_key} (eff={best_eff*100:+.4f}%)")
print(f"  MC sig: {verdict['mc_sig']}")
print(f"  Gate: {verdict['gate']}")
print(f"  Rec: {verdict['rec']}")
results['verdict'] = verdict

with open(OUTPUT, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUTPUT}")
