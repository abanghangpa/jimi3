"""
8-Agent Forensic Protocol: S11 Cross-Asset Divergence
======================================================
Tests: ETH/BTC divergence from BTC → relative value signal

Research basis:
- Lead-lag: BTC leads ETH (information flows from larger to smaller market)
- Mean reversion: ETH/BTC ratio reverts after extreme divergences
- BTC dominance cycles: capital rotation between BTC and altcoins
- Cross-market pricing anomalies (SSRN 6861841, 2026)

S11 uses: M10 (BTC trend + ETH/BTC relative strength), M7 (macro regime), exchange_activity
"""

import pandas as pd, numpy as np, json, os
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'
OUTPUT = '/root/.openclaw/workspace/jimi_audit/reports/s11_cross_asset_forensic.json'
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("="*70)
print("8-AGENT FORENSIC: S11 Cross-Asset Divergence")
print("="*70)

# Load data
print("\nLoading data...")
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
for c in ['Close','High','Low','Volume']: ohlcv[c] = ohlcv[c].astype(float)

# Try to load BTC data
btc_file = f'{DATA_DIR}/btc_15m_extended.csv'
if os.path.exists(btc_file):
    btc = pd.read_csv(btc_file)
    btc['timestamp'] = pd.to_datetime(btc['Open time'])
    btc = btc.sort_values('timestamp').reset_index(drop=True)
    for c in ['Close','High','Low','Volume']: btc[c] = btc[c].astype(float)
    print(f"BTC data: {len(btc)} bars")
else:
    btc = None
    print("No BTC data file — will use ETH-only proxies")

deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

merged = pd.merge_asof(
    ohlcv[['timestamp','Close','High','Low','Volume']],
    deriv[['timestamp','oi','ls_ratio','funding_rate']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)

# Merge BTC if available
if btc is not None:
    btc_subset = btc[['timestamp','Close']].rename(columns={'Close':'btc_close'})
    merged = pd.merge_asof(merged, btc_subset, on='timestamp', direction='backward', tolerance=pd.Timedelta('30min'))
    merged['eth_btc_ratio'] = merged['Close'] / merged['btc_close']
    merged['eth_btc_ratio_ma20'] = merged['eth_btc_ratio'].rolling(20).mean()
    merged['eth_btc_deviation'] = (merged['eth_btc_ratio'] - merged['eth_btc_ratio_ma20']) / merged['eth_btc_ratio_ma20']
else:
    # Use EMA cross as proxy for BTC trend
    merged['btc_close'] = None

# ETH features
merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()
merged['ema21'] = merged['Close'].ewm(span=21).mean()
merged['ema55'] = merged['Close'].ewm(span=55).mean()
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['atr'] = (merged['High'] - merged['Low']).rolling(14).mean()

# M10 proxy: BTC trend (EMA21 vs EMA55) + ETH/BTC relative strength
if btc is not None:
    merged['btc_ema21'] = merged['btc_close'].ewm(span=21).mean()
    merged['btc_ema55'] = merged['btc_close'].ewm(span=55).mean()
    merged['btc_trend'] = np.where(merged['btc_ema21'] > merged['btc_ema55'], 1, -1)
    # ETH/BTC relative strength (7-day ROC)
    merged['eth_btc_roc'] = merged['eth_btc_ratio'].pct_change(672)  # 7 days of 15m bars
else:
    merged['btc_trend'] = np.where(merged['ema21'] > merged['ema55'], 1, -1)
    merged['eth_btc_roc'] = 0

# M7 proxy: macro regime from vol + trend
merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')

# Forward returns
for h in [1, 4, 16, 24]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

# Regime
vols_valid = merged['vol_20bar'].dropna()
p33, p67 = vols_valid.quantile(0.33), vols_valid.quantile(0.67)
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

round_trip_cost = 0.0010
results = {}
print(f"Merged: {len(merged)} bars")

# ═══════════════════════════════════════════════════════════════
# AGENT 1: FORENSICS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 1: FORENSICS — Cross-asset data coverage")
print("="*70)

a1 = {
    'total_bars': len(merged),
    'btc_data': btc is not None,
    'eth_btc_ratio_coverage': int(merged['eth_btc_ratio'].notna().sum()) if 'eth_btc_ratio' in merged.columns else 0,
}

if btc is not None:
    # ETH/BTC ratio statistics
    ratio = merged['eth_btc_ratio'].dropna()
    a1['eth_btc_ratio_mean'] = float(ratio.mean())
    a1['eth_btc_ratio_std'] = float(ratio.std())
    
    # Deviation from MA20
    dev = merged['eth_btc_deviation'].dropna()
    a1['deviation_mean'] = float(dev.mean())
    a1['deviation_std'] = float(dev.std())
    
    # Extreme deviations
    for thresh in [0.02, 0.03, 0.05, 0.08]:
        n_up = (dev > thresh).sum()
        n_down = (dev < -thresh).sum()
        a1[f'dev_gt_{thresh}'] = int(n_up)
        a1[f'dev_lt_neg{thresh}'] = int(n_down)
        print(f"  ETH/BTC deviation > {thresh}: {n_up} events, < -{thresh}: {n_down} events")

print(f"  BTC data available: {btc is not None}")
print(f"  ETH/BTC ratio coverage: {a1['eth_btc_ratio_coverage']}")
results['agent_1'] = a1

# ═══════════════════════════════════════════════════════════════
# AGENT 2: NON-INDICATOR — Raw cross-asset signal
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 2: NON-INDICATOR — Raw cross-asset divergence edge")
print("="*70)

a2 = {}

if btc is not None:
    # Test: ETH/BTC deviation from MA → forward returns
    for direction in ['LONG', 'SHORT']:
        for dev_thresh in [0.01, 0.02, 0.03, 0.05]:
            if direction == 'LONG':
                mask = merged['eth_btc_deviation'] < -dev_thresh  # ETH underperforming BTC
            else:
                mask = merged['eth_btc_deviation'] > dev_thresh  # ETH outperforming BTC
            
            shifted = mask.shift(1).fillna(False)
            events = merged[shifted]
            
            for h, label in [(4, '1h'), (16, '4h')]:
                rets = events[f'fwd_ret_{h}'].dropna()
                if len(rets) < 5:
                    continue
                mean_r = rets.mean()
                eff = -mean_r if direction == 'SHORT' else mean_r
                t, p = stats.ttest_1samp(rets, 0)
                wr = (rets < 0).mean() if direction == 'SHORT' else (rets > 0).mean()
                gate = "PASS" if p < 0.1 and eff > round_trip_cost else "FAIL"
                key = f"{direction}_dev{dev_thresh}_{label}"
                a2[key] = {'n': len(rets), 'eff': float(eff), 'p': float(p), 'wr': float(wr), 'gate': gate}
                print(f"  {'+' if gate=='PASS' else '-'} {direction} dev>{dev_thresh} {label}: n={len(rets)}, eff={eff*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

    # Test: BTC trend + ETH/BTC deviation combined
    print("\n  --- BTC trend + deviation ---")
    for btc_dir in ['UP', 'DOWN']:
        for dev_thresh in [0.02, 0.03]:
            if btc_dir == 'UP':
                btc_mask = merged['btc_trend'] == 1
            else:
                btc_mask = merged['btc_trend'] == -1
            
            eth_mask = merged['eth_btc_deviation'].abs() > dev_thresh
            combined = btc_mask & eth_mask
            shifted = combined.shift(1).fillna(False)
            events = merged[shifted]
            
            rets = events['fwd_ret_16'].dropna()
            if len(rets) < 5:
                continue
            mean_r = rets.mean()
            t, p = stats.ttest_1samp(rets, 0)
            wr = (rets > 0).mean()
            gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
            key = f"btc{btc_dir}_dev{dev_thresh}_4h"
            a2[key] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
            print(f"  {'+' if gate=='PASS' else '-'} BTC {btc_dir} + dev>{dev_thresh} 4h: n={len(rets)}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_2'] = a2

# ═══════════════════════════════════════════════════════════════
# AGENT 3: INDICATOR — S11 trigger replication
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 3: INDICATOR — S11 trigger (M10+M7+EX alignment)")
print("="*70)

a3 = {}

# Replicate S11's logic: alignment score from M10, M7, exchange_activity
# M10 proxy: BTC trend + ETH/BTC relative strength
# M7 proxy: macro regime (vol + trend)
# Exchange proxy: vol_ratio

s11_signals = []
for idx in range(200, len(merged)):
    row = merged.iloc[idx]
    
    # M10 score: BTC trend direction
    btc_trend = row.get('btc_trend', 0)
    eth_btc_roc = row.get('eth_btc_roc', 0) if pd.notna(row.get('eth_btc_roc', 0)) else 0
    
    # BTC trending up + ETH underperforming = LONG opportunity
    # BTC trending down + ETH outperforming = SHORT opportunity
    m10_long = 0.5 + (btc_trend * 0.2) + (eth_btc_roc * -0.3)  # negative ROC = ETH underperforming = good for LONG
    m10_short = 1 - m10_long
    
    # M7 score: macro regime
    vol_regime = row.get('vol_regime', 'MID')
    trend = row.get('trend', 'BEAR')
    m7_long = 0.5
    if trend == 'BULL':
        m7_long += 0.15
    elif trend == 'BEAR':
        m7_long -= 0.15
    if vol_regime == 'HIGH':
        m7_long -= 0.10  # high vol = risk off
    elif vol_regime == 'LOW':
        m7_long += 0.05
    m7_short = 1 - m7_long
    
    # Exchange score: volume activity
    vol_ratio = row.get('vol_ratio', 1.0) if pd.notna(row.get('vol_ratio', 1.0)) else 1.0
    ex_score = min(vol_ratio / 2.0, 1.0)  # normalize
    
    # Alignment
    long_alignment = (m10_long + m7_long + ex_score) / 3
    short_alignment = (m10_short + m7_short + (1 - ex_score)) / 3
    
    if long_alignment >= short_alignment and long_alignment >= 0.55:
        direction = 'LONG'
        alignment = long_alignment
    elif short_alignment > long_alignment and short_alignment >= 0.55:
        direction = 'SHORT'
        alignment = short_alignment
    else:
        continue
    
    s11_signals.append({
        'idx': idx, 'direction': direction,
        'alignment': alignment,
        'm10_long': m10_long, 'm7_long': m7_long,
        'vol_ratio': vol_ratio,
        'trend': trend, 'vol_regime': vol_regime,
    })

a3['s11_signals'] = len(s11_signals)
print(f"  S11 signals: {len(s11_signals)}")

if len(s11_signals) >= 5:
    for direction in ['LONG', 'SHORT']:
        de = [s for s in s11_signals if s['direction'] == direction]
        if len(de) < 5:
            continue
        indices = [s['idx'] for s in de]
        for h, label in [(4, '1h'), (16, '4h')]:
            rets = merged.iloc[indices][f'fwd_ret_{h}'].dropna()
            if len(rets) < 3:
                continue
            mean_r = rets.mean()
            eff = -mean_r if direction == 'SHORT' else mean_r
            t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
            wr = (rets < 0).mean() if direction == 'SHORT' else (rets > 0).mean()
            gate = "PASS" if p < 0.1 and eff > round_trip_cost else "FAIL"
            a3[f's11_{direction}_{label}'] = {'n': len(rets), 'eff': float(eff), 'p': float(p), 'wr': float(wr), 'gate': gate}
            print(f"  {'+' if gate=='PASS' else '-'} S11 {direction} {label}: n={len(rets)}, eff={eff*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_3'] = a3

# ═══════════════════════════════════════════════════════════════
# AGENT 4: REGIME
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 4: REGIME — Edge by regime")
print("="*70)

a4 = {}
test_events = s11_signals if len(s11_signals) >= 10 else []
if test_events:
    for col, name in [('vol_regime', 'Vol'), ('trend', 'Trend'), ('era', 'Era')]:
        print(f"\n  --- {name} ---")
        for reg in sorted(merged[col].dropna().unique()):
            ri = [s['idx'] for s in test_events if merged.iloc[s['idx']][col] == reg]
            if len(ri) < 3:
                continue
            rets = merged.iloc[ri]['fwd_ret_16'].dropna()
            if len(rets) < 3:
                continue
            mean_r = rets.mean()
            t, p = stats.ttest_1samp(rets, 0) if len(rets) > 1 else (0, 1)
            wr = (rets > 0).mean()
            gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
            a4[f'{col}_{reg}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
            print(f"    {'+' if gate=='PASS' else '-'} {reg:12s}: n={len(rets)}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_4'] = a4

# ═══════════════════════════════════════════════════════════════
# AGENT 5: GATE — Frequency
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 5: GATE — Signal frequency")
print("="*70)

months = max(1, (merged['timestamp'].max() - merged['timestamp'].min()).days / 30)
a5 = {'total': len(s11_signals), 'per_month': len(s11_signals)/months}
long_n = sum(1 for s in s11_signals if s['direction'] == 'LONG')
short_n = sum(1 for s in s11_signals if s['direction'] == 'SHORT')
a5['long'] = long_n
a5['short'] = short_n
print(f"  Total: {len(s11_signals)} ({len(s11_signals)/months:.1f}/mo)")
print(f"  LONG: {long_n}, SHORT: {short_n}")
results['agent_5'] = a5

# ═══════════════════════════════════════════════════════════════
# AGENT 6: CO-OCCURRENCE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 6: CO-OCCURRENCE")
print("="*70)

a6 = {}
if s11_signals:
    indices = [s['idx'] for s in s11_signals]
    ls = merged.iloc[indices]['ls_ratio'].dropna()
    fr = merged.iloc[indices]['funding_rate'].dropna()
    if len(ls) > 0: a6['ls'] = float(ls.mean()); print(f"  LS: {ls.mean():.3f}")
    if len(fr) > 0: a6['fr'] = float(fr.mean()); print(f"  FR: {fr.mean():.6f}")
    
    alignments = [s['alignment'] for s in s11_signals]
    a6['alignment_mean'] = float(np.mean(alignments))
    print(f"  Alignment mean: {np.mean(alignments):.3f}")
results['agent_6'] = a6

# ═══════════════════════════════════════════════════════════════
# AGENT 7: SENSITIVITY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 7: SENSITIVITY — Alignment threshold sweep")
print("="*70)

a7 = {}
for align_min in [0.55, 0.60, 0.65, 0.70]:
    filtered = [s for s in s11_signals if s['alignment'] >= align_min]
    if len(filtered) < 10:
        continue
    indices = [s['idx'] for s in filtered]
    rets = merged.iloc[indices]['fwd_ret_16'].dropna()
    if len(rets) < 5:
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
    if gate == "PASS":
        a7[f'align>={align_min}'] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr)}
        print(f"  + align>={align_min}: n={len(rets)}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

results['agent_7'] = a7

# ═══════════════════════════════════════════════════════════════
# AGENT 8: MONTE CARLO
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 8: MONTE CARLO")
print("="*70)

a8 = {}
if s11_signals:
    indices = [s['idx'] for s in s11_signals]
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

# ═══════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("VERDICT")
print("="*70)

best = None; best_m = 0
for k, v in a4.items():
    if v.get('gate') == 'PASS' and v.get('mean', 0) > best_m:
        best_m = v['mean']; best = k

verdict = {
    'strategy': 'S11 Cross-Asset Divergence',
    'total_signals': len(s11_signals),
    'mc_sig': a8.get('sig', False),
    'best_regime': best,
}

if a8.get('sig'):
    verdict['gate'] = 'PASS'
    verdict['rec'] = 'Deploy with 0.5x size'
elif best and best_m > 0.003:
    verdict['gate'] = 'CONDITIONAL'
    verdict['rec'] = f'Deploy only in {best}'
elif len(s11_signals) < 10:
    verdict['gate'] = 'LOW_SAMPLE'
    verdict['rec'] = 'Need more data'
else:
    verdict['gate'] = 'FAIL'
    verdict['rec'] = 'No edge'

print(f"  Signals: {len(s11_signals)}")
print(f"  Best regime: {best} ({best_m*100:+.4f}%)" if best else "  Best: none")
print(f"  MC sig: {verdict['mc_sig']}")
print(f"  Gate: {verdict['gate']}")
print(f"  Rec: {verdict['rec']}")
results['verdict'] = verdict

with open(OUTPUT, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUTPUT}")
