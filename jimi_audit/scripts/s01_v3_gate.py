"""
8-Agent Gate: S01 v3 Liquidity Trap Detection
==============================================
Tests: M14 sweep + M21 Wyckoff + M5 structural + derivatives + taker
"""
import pandas as pd, numpy as np, json, os
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'
OUTPUT = '/root/.openclaw/workspace/jimi_audit/reports/s01_v3_gate.json'
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("Loading data...")
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
for c in ['Close','High','Low','Volume']: ohlcv[c] = ohlcv[c].astype(float)

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

# ── Detect liquidity sweeps from price action ──
# A sweep = price moves beyond a swing level then reverses back
highs = merged['High'].values.astype(float)
lows = merged['Low'].values.astype(float)
closes = merged['Close'].values.astype(float)

print("Detecting liquidity sweeps...")

sweep_events = []
seen = set()

for idx in range(56, len(merged)):
    # Look for sweep in last 4 bars
    for lb in range(1, min(5, idx)):
        bar_idx = idx - lb
        if bar_idx < 48: continue
        
        # Swing levels from bars before the sweep bar
        swing_high = float(np.max(highs[bar_idx-48:bar_idx]))
        swing_low = float(np.min(lows[bar_idx-48:bar_idx]))
        
        # Sweep above → failed → SHORT reversal
        if highs[bar_idx] > swing_high * 1.003:  # 0.3% penetration (real trap)
            if closes[bar_idx] < swing_high:  # closed below (failed)
                if closes[idx] < swing_high:  # still below
                    key = (bar_idx, 'S')
                    if key not in seen:
                        # Check volume: low volume sweep = more likely false
                        vol_ratio = merged.iloc[bar_idx]['vol_ratio'] if pd.notna(merged.iloc[bar_idx]['vol_ratio']) else 1.0
                        seen.add(key)
                        sweep_events.append({
                            'idx': idx, 'direction': 'SHORT', 'sweep_dir': 'LONG',
                            'level': swing_high, 'penetration': (highs[bar_idx] - swing_high) / swing_high,
                            'vol_ratio': vol_ratio,
                            'bars_since': idx - bar_idx,
                        })
                        break
        
        # Sweep below → failed → LONG reversal
        if lows[bar_idx] < swing_low * 0.997:  # 0.3% penetration
            if closes[bar_idx] > swing_low:  # closed above (failed)
                if closes[idx] > swing_low:  # still above
                    key = (bar_idx, 'L')
                    if key not in seen:
                        vol_ratio = merged.iloc[bar_idx]['vol_ratio'] if pd.notna(merged.iloc[bar_idx]['vol_ratio']) else 1.0
                        seen.add(key)
                        sweep_events.append({
                            'idx': idx, 'direction': 'LONG', 'sweep_dir': 'SHORT',
                            'level': swing_low, 'penetration': (swing_low - lows[bar_idx]) / swing_low,
                            'vol_ratio': vol_ratio,
                            'bars_since': idx - bar_idx,
                        })
                        break

print(f"  Total sweep events: {len(sweep_events)}")

# Load scan data for M14/M21/M5 context
scan_dir = os.path.join(DATA_DIR, 'scans')
scan_files = sorted([f for f in os.listdir(scan_dir) if f.startswith('scan_') and f.endswith('.json')]) if os.path.exists(scan_dir) else []
print(f"  Scan files available: {len(scan_files)}")

# Load a sample scan to check what M14/M21/M5 contain
if scan_files:
    with open(os.path.join(scan_dir, scan_files[-1])) as f:
        sample = json.load(f)
    m14 = sample.get('m14', {})
    m21 = sample.get('m21', {})
    m5 = sample.get('m5', {})
    print(f"  M14 keys: {list(m14.keys())[:10]}")
    print(f"  M21 keys: {list(m21.keys())[:10]}")
    print(f"  M5 keys: {list(m5.keys())[:10]}")

results = {}
round_trip_cost = 0.0010

# ═══════════ AGENT 1: FORENSICS ═══════════
print("\n" + "="*70)
print("AGENT 1: FORENSICS")
print("="*70)

a1 = {
    'total_sweeps': len(sweep_events),
    'long_sweeps': sum(1 for e in sweep_events if e['direction'] == 'LONG'),
    'short_sweeps': sum(1 for e in sweep_events if e['direction'] == 'SHORT'),
    'july_2026': sum(1 for e in sweep_events if merged.iloc[e['idx']]['is_july_2026']),
}
print(f"  Total: {a1['total_sweeps']}, LONG: {a1['long_sweeps']}, SHORT: {a1['short_sweeps']}, July: {a1['july_2026']}")

# Penetration distribution
pens = [e['penetration'] for e in sweep_events]
a1['pen_mean'] = float(np.mean(pens)) if pens else 0
a1['pen_median'] = float(np.median(pens)) if pens else 0
print(f"  Penetration: mean={a1['pen_mean']:.4f}, median={a1['pen_median']:.4f}")

# Volume during sweeps
vols_sweep = [e['vol_ratio'] for e in sweep_events]
a1['vol_mean'] = float(np.mean(vols_sweep)) if vols_sweep else 0
print(f"  Vol ratio during sweeps: {a1['vol_mean']:.2f}x")

results['agent_1'] = a1

# ═══════════ AGENT 2: NON-INDICATOR ═══════════
print("\n" + "="*70)
print("AGENT 2: NON-INDICATOR — Raw sweep edge")
print("="*70)

a2 = {}
for direction in ['LONG', 'SHORT']:
    de = [e for e in sweep_events if e['direction'] == direction]
    if len(de) < 5: continue
    indices = [e['idx'] for e in de]
    for h, label in [(4, '1h'), (16, '4h')]:
        rets = merged.iloc[indices][f'fwd_ret_{h}'].dropna()
        if len(rets) < 3: continue
        mean_r = rets.mean()
        eff = -mean_r if direction == 'SHORT' else mean_r
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets < 0).mean() if direction == 'SHORT' else (rets > 0).mean()
        gate = "PASS" if p < 0.1 and eff > round_trip_cost else "FAIL"
        a2[f'{direction}_{label}'] = {'n': len(rets), 'eff': float(eff), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} {direction:5s} {label}: n={len(rets)}, eff={eff*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# By volume (low vol sweep = more likely false)
print("\n  --- Volume filter ---")
for vol_max, label in [(0.8, 'low_vol'), (1.0, 'below_avg'), (1.5, 'normal')]:
    ve = [e for e in sweep_events if e['vol_ratio'] <= vol_max]
    if len(ve) < 5: continue
    indices = [e['idx'] for e in ve]
    rets = merged.iloc[indices]['fwd_ret_16'].dropna()
    if len(rets) < 3: continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
    a2[f'vol_{label}_4h'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
    print(f"  {'+' if gate=='PASS' else '-'} vol<={vol_max} ({label}) 4h: n={len(rets)}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# By penetration (deeper = more stops triggered)
print("\n  --- Penetration filter ---")
for pen_min, label in [(0.003, '0.3pct'), (0.005, '0.5pct'), (0.01, '1pct')]:
    pe = [e for e in sweep_events if e['penetration'] >= pen_min]
    if len(pe) < 5: continue
    indices = [e['idx'] for e in pe]
    rets = merged.iloc[indices]['fwd_ret_16'].dropna()
    if len(rets) < 3: continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
    a2[f'pen_{label}_4h'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
    print(f"  {'+' if gate=='PASS' else '-'} pen>={pen_min} ({label}) 4h: n={len(rets)}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_2'] = a2

# ═══════════ AGENT 3: INDICATOR — v3 trigger ═══════════
print("\n" + "="*70)
print("AGENT 3: INDICATOR — v3 trigger (sweep + session + vol)")
print("="*70)

a3 = {}

# v3 filters: session + low volume sweep + not chased
v3_events = []
for e in sweep_events:
    idx = e['idx']
    row = merged.iloc[idx]
    
    # Session
    if row['hour'] not in GOOD_HOURS: continue
    # Low volume sweep (false breakouts have low volume)
    if e['vol_ratio'] > 1.5: continue
    # Not chased (price hasn't moved too far from level)
    if e['bars_since'] > 4: continue
    
    v3_events.append(e)

a3['v3_signals'] = len(v3_events)
print(f"  v3 signals: {len(v3_events)}")

if len(v3_events) >= 5:
    indices = [e['idx'] for e in v3_events]
    for h, label in [(4, '1h'), (16, '4h')]:
        rets = merged.iloc[indices][f'fwd_ret_{h}'].dropna()
        if len(rets) < 3: continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        a3[f'v3_{label}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"  {'+' if gate=='PASS' else '-'} v3 {label}: n={len(rets)}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_3'] = a3

# ═══════════ AGENT 4: REGIME ═══════════
print("\n" + "="*70)
print("AGENT 4: REGIME")
print("="*70)

a4 = {}
test_events = v3_events if len(v3_events) >= 10 else sweep_events
indices_all = [e['idx'] for e in test_events]

for col, name in [('vol_regime', 'Vol'), ('trend', 'Trend'), ('era', 'Era')]:
    print(f"\n  --- {name} ---")
    for reg in sorted(merged[col].dropna().unique()):
        ri = [i for i in indices_all if merged.iloc[i][col] == reg]
        if len(ri) < 3: continue
        rets = merged.iloc[ri]['fwd_ret_16'].dropna()
        if len(rets) < 3: continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        a4[f'{col}_{reg}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
        print(f"    {'+' if gate=='PASS' else '-'} {reg:12s}: n={len(rets)}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_4'] = a4

# ═══════════ AGENT 5: GATE ═══════════
print("\n" + "="*70)
print("AGENT 5: GATE")
print("="*70)

months = max(1, (merged['timestamp'].max() - merged['timestamp'].min()).days / 30)
a5 = {'raw': len(sweep_events), 'filtered': len(v3_events),
      'raw_per_month': len(sweep_events)/months, 'filtered_per_month': len(v3_events)/months}
print(f"  Raw: {a5['raw']} ({a5['raw_per_month']:.1f}/mo), Filtered: {a5['filtered']} ({a5['filtered_per_month']:.1f}/mo)")
results['agent_5'] = a5

# ═══════════ AGENT 6: CO-OCCURRENCE ═══════════
print("\n" + "="*70)
print("AGENT 6: CO-OCCURRENCE")
print("="*70)

a6 = {}
if len(v3_events) > 0:
    indices = [e['idx'] for e in v3_events]
    ls = merged.iloc[indices]['ls_ratio'].dropna()
    fr = merged.iloc[indices]['funding_rate'].dropna()
    if len(ls) > 0: a6['ls'] = float(ls.mean()); print(f"  LS: {ls.mean():.3f}")
    if len(fr) > 0: a6['fr'] = float(fr.mean()); print(f"  FR: {fr.mean():.6f}")
    
    # Direction distribution
    long_pct = sum(1 for e in v3_events if e['direction'] == 'LONG') / len(v3_events)
    a6['long_pct'] = long_pct; print(f"  Long%: {long_pct:.0%}")
results['agent_6'] = a6

# ═══════════ AGENT 7: SENSITIVITY ═══════════
print("\n" + "="*70)
print("AGENT 7: SENSITIVITY")
print("="*70)

a7 = {}
for vol_max in [0.8, 1.0, 1.2, 1.5, 2.0]:
    for pen_min in [0.003, 0.005, 0.01]:
        evts = [e for e in sweep_events if e['vol_ratio'] <= vol_max and e['penetration'] >= pen_min]
        if len(evts) < 10: continue
        indices = [e['idx'] for e in evts]
        rets = merged.iloc[indices]['fwd_ret_16'].dropna()
        if len(retts) < 5: continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
        if gate == "PASS":
            key = f"v{vol_max}_p{pen_min}"
            a7[key] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr)}
            print(f"  + vol<={vol_max} pen>={pen_min}: n={len(rets)}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_7'] = a7

# ═══════════ AGENT 8: MONTE CARLO ═══════════
print("\n" + "="*70)
print("AGENT 8: MONTE CARLO")
print("="*70)

a8 = {}
test = v3_events if len(v3_events) >= 10 else sweep_events
if len(test) >= 5:
    indices = [e['idx'] for e in test]
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
          'ci': [float(ci_lo), float(ci_hi)], 'sig': bool(pm < 0.05)}
    print(f"  MC p: {pm:.4f}, CI: [{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%]")
    print(f"  SIGNIFICANT: {'YES' if pm < 0.05 else 'NO'}")
results['agent_8'] = a8

# ═══════════ VERDICT ═══════════
print("\n" + "="*70)
print("VERDICT")
print("="*70)

best = None; best_m = 0
for k, v in a4.items():
    if v.get('gate') == 'PASS' and v.get('mean', 0) > best_m:
        best_m = v['mean']; best = k

verdict = {
    'strategy': 'S01 v3 Liquidity Trap', 'raw': len(sweep_events), 'filtered': len(v3_events),
    'mc_sig': a8.get('sig', False), 'best_regime': best,
}

if a8.get('sig'):
    verdict['gate'] = 'PASS'; verdict['rec'] = 'Deploy with 0.5x size'
elif best and best_m > 0.003:
    verdict['gate'] = 'CONDITIONAL'; verdict['rec'] = f'Deploy only in {best}'
elif len(v3_events) < 10:
    verdict['gate'] = 'LOW_SAMPLE'; verdict['rec'] = 'Need more data or relax filters'
else:
    verdict['gate'] = 'FAIL'; verdict['rec'] = 'No edge'

print(f"  Raw: {verdict['raw']}, Filtered: {verdict['filtered']}")
print(f"  Best: {verdict['best_regime']} ({best_m*100:+.4f}%)" if best else "  Best: none")
print(f"  Gate: {verdict['gate']}, Rec: {verdict['rec']}")
results['verdict'] = verdict

with open(OUTPUT, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUTPUT}")
