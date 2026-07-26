"""
Optimization Framework v2 — momentum_v3 BULL+SHORT
Fixed: precompute signals once, filter for walk-forward
"""
import pandas as pd
import numpy as np
from scipy import stats
import json, os, itertools

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'

ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)

deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

merged = pd.merge_asof(ohlcv, deriv[['timestamp','oi','ls_ratio','funding_rate']],
                       on='timestamp', direction='backward', tolerance=pd.Timedelta('30min'))

merged['oi_roc'] = merged['oi'].pct_change(4, fill_method=None)
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')

for h in [4, 8, 16, 24, 48]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

closes = merged['Close'].values
volumes = merged['Volume'].values
n = len(merged)

# Precompute ALL signal components once (not per-param)
print("Precomputing signal components...")
signal_components = []
for idx in range(80, n):
    if merged.iloc[idx]['trend'] != 'BULL':
        continue
    mom_5 = (closes[idx] - closes[idx-5]) / closes[idx-5]
    mom_10 = (closes[idx] - closes[idx-10]) / closes[idx-10]
    accel = mom_5 - mom_10 / 2

    # DECEL required: UP but decelerating (SHORT signal)
    if not (mom_5 > 0 and accel < 0):
        continue

    vol_recent = np.mean(volumes[idx-5:idx])
    vol_prior = np.mean(volumes[idx-15:idx-5])
    vol_change = (vol_recent - vol_prior) / vol_prior if vol_prior > 0 else 0

    moves = []
    for j in range(max(0, idx-80), idx-5):
        if j+5 < n:
            m = abs(closes[j+5] - closes[j]) / closes[j]
            moves.append(m)
    current_move = abs(closes[idx] - closes[idx-5]) / closes[idx-5]
    percentile = sum(1 for m in moves if m < current_move) / len(moves) * 100 if moves else 0

    oi_roc = merged.iloc[idx].get('oi_roc', 0) or 0

    week = merged.iloc[idx]['timestamp'].isocalendar().week
    year = merged.iloc[idx]['timestamp'].isocalendar().year
    yw = year * 100 + week

    signal_components.append({
        'idx': idx, 'timestamp': merged.iloc[idx]['timestamp'], 'price': closes[idx],
        'vol_change': vol_change, 'percentile': percentile, 'oi_roc': oi_roc,
        'yw': yw, 'fwd_ret_16': merged.iloc[idx]['fwd_ret_16'] if idx + 16 < n else np.nan,
    })

print(f"Precomputed {len(signal_components)} DECEL candidates in BULL")

# Parameter grid
param_grid = {
    'vol_thresh': [-0.10, -0.15, -0.20, -0.25, -0.30],
    'extreme_pctl': [85, 90, 95],
    'dedup_bars': [4, 8, 16],
    'conv_base': [0.50, 0.55, 0.60, 0.65],
    'min_additional': [1, 2],
}

combos = list(itertools.product(*param_grid.values()))
print(f"Testing {len(combos)} combinations")

def filter_signals(vol_thresh, extreme_pctl, dedup_bars, conv_base, min_additional):
    """Filter precomputed signals by params."""
    signals = []
    last_idx = -999
    for sc in signal_components:
        vol_div = sc['vol_change'] < vol_thresh
        extreme = sc['percentile'] > extreme_pctl
        oi_div = sc['oi_roc'] < -0.02
        additional = sum([vol_div, extreme, oi_div])
        if additional < min_additional:
            continue
        if sc['idx'] - last_idx < dedup_bars:
            continue
        last_idx = sc['idx']
        base = conv_base
        if vol_div: base += 0.15
        if extreme: base += 0.10
        if oi_div: base += 0.10
        signals.append({**sc, 'conviction': min(base, 0.90), 'signals_count': 1 + additional,
                        'vol_div': vol_div, 'extreme': extreme, 'oi_div': oi_div})
    return signals

def evaluate(signals):
    """Quick evaluation."""
    if not signals:
        return None
    sdf = pd.DataFrame(signals)
    rets = sdf['fwd_ret_16'].dropna()
    if len(rets) < 5:
        return None
    adj = -rets  # SHORT
    wr = (adj > 0).mean()
    mean_r = adj.mean()
    pf = adj[adj > 0].sum() / abs(adj[adj < 0].sum()) if adj[adj < 0].sum() != 0 else float('inf')
    t, p = stats.ttest_1samp(adj, 0)
    p1 = p/2 if t > 0 else 1-p/2
    # Bootstrap (500 iterations for speed)
    boots = [np.random.choice(adj.values, size=len(adj), replace=True).mean() for _ in range(500)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return {'n': len(rets), 'wr': wr, 'mean': mean_r, 'pf': pf, 'p': p1, 'ci_lo': ci_lo, 'ci_hi': ci_hi}

def walk_forward(signals, train_weeks=4, test_weeks=1):
    """Walk-forward using pre-filtered signals."""
    if not signals:
        return []
    sdf = pd.DataFrame(signals)
    weeks = sorted(sdf['yw'].unique())
    results = []
    for i in range(0, len(weeks) - train_weeks - test_weeks + 1):
        test_w = weeks[i+train_weeks:i+train_weeks+test_weeks]
        test = sdf[sdf['yw'].isin(test_w)]
        rets = test['fwd_ret_16'].dropna()
        if len(rets) >= 2:
            adj = -rets
            results.append({'week': test_w[0], 'n': len(rets), 'wr': (adj > 0).mean(), 'mean': adj.mean()})
    return results

# ═══════════════════════════════════════════════════════
# GRID SEARCH
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("SEARCH AGENT: Grid Search for momentum_v3 BULL+SHORT")
print("="*70)

all_results = []

for i, combo in enumerate(combos):
    vol_thresh, extreme_pctl, dedup_bars, conv_base, min_additional = combo

    signals = filter_signals(vol_thresh, extreme_pctl, dedup_bars, conv_base, min_additional)
    metrics = evaluate(signals)
    if not metrics or metrics['n'] < 10:
        continue

    wf = walk_forward(signals)
    if not wf:
        continue

    wf_wr = np.mean([r['wr'] for r in wf])
    wf_mean = np.mean([r['mean'] for r in wf])
    overfit = max(0, metrics['wr'] - wf_wr - 0.05) * 2
    score = wf_mean * np.sqrt(min(metrics['n'], 100)) * (1 - overfit)

    all_results.append({
        'vol_thresh': vol_thresh, 'extreme_pctl': extreme_pctl, 'dedup_bars': dedup_bars,
        'conv_base': conv_base, 'min_additional': min_additional,
        'n': metrics['n'], 'wr': metrics['wr'], 'mean': metrics['mean'], 'pf': metrics['pf'],
        'p': metrics['p'], 'ci_lo': metrics['ci_lo'], 'ci_hi': metrics['ci_hi'],
        'wf_wr': wf_wr, 'wf_mean': wf_mean, 'overfit': overfit, 'score': score,
        'n_wf': len(wf),
    })

all_results.sort(key=lambda x: x['score'], reverse=True)
print(f"\nTested {len(all_results)} combinations")

print(f"\n=== TOP 10 ===\n")
for i, r in enumerate(all_results[:10]):
    tag = "PASS" if r['ci_lo'] > 0 else ("PROV" if r['mean'] > 0 else "FAIL")
    print(f"[{tag}] #{i+1}: vol={r['vol_thresh']:.2f} pctl={r['extreme_pctl']} dedup={r['dedup_bars']} conv={r['conv_base']:.2f} min_add={r['min_additional']}")
    print(f"  n={r['n']} WR={r['wr']*100:.1f}% mean={r['mean']*100:+.3f}% PF={r['pf']:.2f} p={r['p']:.4f}")
    print(f"  CI=[{r['ci_lo']*100:+.3f}%, {r['ci_hi']*100:+.3f}%]")
    print(f"  WF: WR={r['wf_wr']*100:.1f}% mean={r['wf_mean']*100:+.3f}% (n={r['n_wf']} periods)")
    print()

# ═══════════════════════════════════════════════════════
# DEEP VALIDATION TOP3
# ═══════════════════════════════════════════════════════
print("="*70)
print("VALIDATOR: Deep validation top3")
print("="*70)

for i, r in enumerate(all_results[:3]):
    print(f"\n--- #{i+1}: vol={r['vol_thresh']:.2f} pctl={r['extreme_pctl']} dedup={r['dedup_bars']} conv={r['conv_base']:.2f} ---")
    signals = filter_signals(r['vol_thresh'], r['extreme_pctl'], r['dedup_bars'], r['conv_base'], r['min_additional'])
    sdf = pd.DataFrame(signals)

    # Multiple horizons
    for h in [4, 8, 16, 24, 48]:
        col = f'fwd_ret_{h}'
        sdf[col] = sdf.apply(lambda r: merged.iloc[r['idx']][col] if r['idx'] + h < len(merged) else np.nan, axis=1)
        rets = sdf[col].dropna()
        if len(rets) > 0:
            adj = -rets
            wr = (adj > 0).mean()
            t, p = stats.ttest_1samp(adj, 0)
            p1 = p/2 if t > 0 else 1-p/2
            print(f"  {h}h: WR={wr*100:.1f}% mean={adj.mean()*100:+.3f}% p={p1:.4f} n={len(rets)}")

    # Monte Carlo
    rets = sdf['fwd_ret_16'].dropna()
    if len(rets) > 0:
        adj = -rets
        sims = [np.random.choice(adj.values, size=30, replace=True).sum() for _ in range(5000)]
        sims = np.array(sims)
        p5, p50, p95 = np.percentile(sims, [5, 50, 95])
        print(f"  MC: P5={p5*100:+.2f}% P50={p50*100:+.2f}% P95={p95*100:+.2f}% Prob(loss)={(sims<0).mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)
top = all_results[0]
if top['ci_lo'] > 0 and top['wr'] > 0.52 and top['p'] < 0.10:
    verdict = "PASS"
    print(f"PASS — Edge found")
elif top['mean'] > 0 and top['wr'] > 0.50:
    verdict = "PROVISIONAL"
    print(f"PROVISIONAL — Positive but not significant")
else:
    verdict = "FAIL"
    print(f"FAIL — No edge in BULL+SHORT")

print(f"  Best: vol={top['vol_thresh']:.2f} pctl={top['extreme_pctl']} dedup={top['dedup_bars']} conv={top['conv_base']:.2f}")
print(f"  n={top['n']} WR={top['wr']*100:.1f}% mean={top['mean']*100:+.3f}% PF={top['pf']:.2f}")
print(f"  CI=[{top['ci_lo']*100:+.3f}%, {top['ci_hi']*100:+.3f}%] p={top['p']:.4f}")
print(f"  WF: WR={top['wf_wr']*100:.1f}% mean={top['wf_mean']*100:+.3f}%")

# Save
out_path = '/root/.openclaw/workspace/jimi_audit/reports/momentum_v3_optimization.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump({
        'timestamp': str(pd.Timestamp.now()),
        'target': 'momentum_v3 BULL+SHORT',
        'verdict': verdict,
        'total_combos': len(combos),
        'tested': len(all_results),
        'top10': all_results[:10],
        'best': all_results[0] if all_results else None,
    }, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
