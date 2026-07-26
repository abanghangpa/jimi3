"""
Optimization Framework: S01 v3.1 — Walk-Forward + Deflated Sharpe + Threshold Sweep
=====================================================================================
Runs the optimization framework agents on S01's signal:
- Validator Agent: Walk-forward + deflated Sharpe + Monte Carlo
- Search Agent: Threshold optimization
- Selector Agent: Should this strategy be enabled?
"""

import pandas as pd, numpy as np, json, os
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'
OUTPUT = '/root/.openclaw/workspace/jimi_audit/reports/s01_optimization.json'
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("="*70)
print("OPTIMIZATION FRAMEWORK: S01 v3.1")
print("="*70)

# Load data (same as 5-agent)
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
for c in ['Close','High','Low','Open','Volume']: ohlcv[c] = ohlcv[c].astype(float)

deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

merged = pd.merge_asof(
    ohlcv[['timestamp','Open','High','Low','Close','Volume']],
    deriv[['timestamp','oi','ls_ratio','funding_rate']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)

merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['atr'] = (merged['High'] - merged['Low']).rolling(14).mean()
merged['hour'] = merged['timestamp'].dt.hour
GOOD_HOURS = {9, 10, 11, 12, 14, 15, 16, 18}

for h in [1, 4, 16, 24]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

highs = merged['High'].values.astype(float)
lows = merged['Low'].values.astype(float)
closes = merged['Close'].values.astype(float)
opens = merged['Open'].values.astype(float)
volumes = merged['Volume'].values.astype(float)

# ═══════════════════════════════════════════════════════════════
# DETECT SIGNALS (same as 5-agent)
# ═══════════════════════════════════════════════════════════════
print("Detecting signals...")

def find_swing_levels(highs, lows, idx, lookback=48):
    sh, sl = [], []
    start = max(0, idx - lookback)
    for i in range(start + 2, idx - 1):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1]:
            sh.append((highs[i], i))
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1]:
            sl.append((lows[i], i))
    return sh, sl

def detect_wyckoff(idx):
    if idx < 96:
        return None
    lookback = min(768, idx)
    h4h = highs[idx-lookback:idx+1]
    h4l = lows[idx-lookback:idx+1]
    h4c = closes[idx-lookback:idx+1]
    half = len(h4c) // 2
    recent_hi = h4h[-min(10, half):].max()
    prior_hi = h4h[-min(20, half):-min(10, half)].max() if len(h4h) > 10 else recent_hi
    recent_lo = h4l[-min(10, half):].min()
    prior_lo = h4l[-min(20, half):-min(10, half)].min() if len(h4l) > 10 else recent_lo
    hh, hl = recent_hi > prior_hi, recent_lo > prior_lo
    lh, ll = recent_hi < prior_hi, recent_lo < prior_lo
    range_hi, range_lo = float(h4h.max()), float(h4l.min())
    current = float(h4c[-1])
    position = (current - range_lo) / (range_hi - range_lo) if range_hi > range_lo else 0.5
    phase = 'RANGE'
    if hh and hl:
        phase = 'DISTRIBUTION' if position > 0.7 else 'MARKUP' if position > 0.5 else 'ACCUMULATION'
    elif lh and ll:
        phase = 'ACCUMULATION' if position < 0.3 else 'MARKDOWN' if position < 0.5 else 'DISTRIBUTION'
    return {'phase': phase, 'position': position}

def classify_regime(idx):
    if idx < 200:
        return 'UNKNOWN'
    cw = closes[idx-200:idx+1]
    ema50 = pd.Series(cw).ewm(span=50).mean().iloc[-1]
    ema200 = pd.Series(cw).ewm(span=200).mean().iloc[-1]
    current = cw[-1]
    if current > ema50 > ema200:
        return 'BULL'
    elif current < ema50 < ema200:
        return 'BEAR'
    return 'RANGING'

signals = []
seen = set()
step = 4

for idx in range(96, len(merged), step):
    hour = merged.iloc[idx]['hour']
    if hour not in GOOD_HOURS:
        continue
    swing_highs, swing_lows = find_swing_levels(highs, lows, idx)
    wyckoff = detect_wyckoff(idx)
    if not wyckoff or wyckoff['phase'] != 'ACCUMULATION':
        continue
    
    for level_price, level_idx in swing_lows:
        for lb in range(1, min(6, idx)):
            bar_idx = idx - lb
            if bar_idx < level_idx or bar_idx < 48:
                continue
            bar_range = highs[bar_idx] - lows[bar_idx]
            if bar_range <= 0:
                continue
            sweep_depth = (level_price - lows[bar_idx]) / level_price
            if not (0.001 <= sweep_depth <= 0.020):
                continue
            lower_wick = min(opens[bar_idx], closes[bar_idx]) - lows[bar_idx]
            wick_ratio = lower_wick / bar_range
            vol_avg = np.mean(volumes[max(0, bar_idx-20):bar_idx]) if bar_idx >= 20 else volumes[bar_idx]
            vol_ok = volumes[bar_idx] > vol_avg * 1.2
            if wick_ratio >= 0.40 and closes[bar_idx] > opens[bar_idx]:
                reclaim_type = 'STRONG' if vol_ok else 'WEAK'
            elif closes[bar_idx] > opens[bar_idx] and closes[bar_idx] > level_price:
                reclaim_type = 'WEAK'
            else:
                reclaim_type = 'NONE'
            if reclaim_type != 'WEAK':
                continue
            key = (bar_idx, 'LONG')
            if key in seen:
                continue
            seen.add(key)
            signals.append({
                'idx': idx, 'direction': 'LONG',
                'sweep_depth': sweep_depth * 100,
                'wick_ratio': wick_ratio,
                'wyckoff_position': wyckoff['position'],
                'ls_ratio': merged.iloc[idx]['ls_ratio'] if pd.notna(merged.iloc[idx]['ls_ratio']) else 1.0,
                'funding_rate': merged.iloc[idx]['funding_rate'] if pd.notna(merged.iloc[idx]['funding_rate']) else 0,
                'vol_ratio': merged.iloc[idx]['vol_ratio'] if pd.notna(merged.iloc[idx]['vol_ratio']) else 1.0,
                'atr': merged.iloc[idx]['atr'] if pd.notna(merged.iloc[idx]['atr']) else 0,
                'price': closes[idx],
                'regime': classify_regime(idx),
            })
            break

# Filter to BULL only (from 5-agent gate)
signals = [s for s in signals if s['regime'] == 'BULL']
print(f"  BULL signals: {len(signals)}")

results = {}

# ═══════════════════════════════════════════════════════════════
# VALIDATOR AGENT: Walk-Forward + Deflated Sharpe
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("VALIDATOR AGENT: Walk-Forward Analysis")
print("="*70)

# Walk-forward: train on 70%, test on 30%, slide forward
n_signals = len(signals)
train_pct = 0.7
test_pct = 0.3
step_size = max(10, int(n_signals * 0.1))  # slide by 10% of signals

wf_results = []
for start in range(0, n_signals - step_size, step_size):
    train_end = start + int(step_size * train_pct)
    test_end = min(start + step_size, n_signals)
    
    if train_end >= test_end or test_end > n_signals:
        continue
    
    train_signals = signals[start:train_end]
    test_signals = signals[train_end:test_end]
    
    if len(test_signals) < 3:
        continue
    
    # Test set performance
    test_indices = [s['idx'] for s in test_signals]
    test_rets = merged.iloc[test_indices]['fwd_ret_16'].dropna()
    
    if len(test_rets) < 3:
        continue
    
    wr = (test_rets > 0).mean()
    mean_r = test_rets.mean()
    
    wf_results.append({
        'window': len(wf_results),
        'train_n': len(train_signals),
        'test_n': len(test_signals),
        'test_wr': float(wr),
        'test_mean': float(mean_r),
    })

if wf_results:
    avg_wr = np.mean([r['test_wr'] for r in wf_results])
    avg_mean = np.mean([r['test_mean'] for r in wf_results])
    min_wr = min(r['test_wr'] for r in wf_results)
    max_wr = max(r['test_wr'] for r in wf_results)
    
    print(f"  Walk-forward windows: {len(wf_results)}")
    print(f"  Avg test WR: {avg_wr:.1%} (range: {min_wr:.1%} - {max_wr:.1%})")
    print(f"  Avg test mean: {avg_mean*100:+.4f}%")
    
    results['walk_forward'] = {
        'windows': len(wf_results),
        'avg_wr': float(avg_wr),
        'min_wr': float(min_wr),
        'max_wr': float(max_wr),
        'avg_mean': float(avg_mean),
        'details': wf_results,
    }
else:
    print("  Not enough data for walk-forward")
    results['walk_forward'] = {'windows': 0}

# Deflated Sharpe Ratio
print("\n" + "="*70)
print("VALIDATOR AGENT: Deflated Sharpe Ratio")
print("="*70)

# Number of trials tested (strategies + params we've tried)
n_trials = 20  # conservative estimate: S01 v2, v3, v3.1 + various params
n_trades = len(signals)
all_indices = [s['idx'] for s in signals]
all_rets = merged.iloc[all_indices]['fwd_ret_16'].dropna()

if len(all_rets) > 5:
    sharpe = all_rets.mean() / all_rets.std() * np.sqrt(len(all_rets)) if all_rets.std() > 0 else 0
    skew = float(stats.skew(all_rets))
    kurt = float(stats.kurtosis(all_rets))
    
    # Expected max Sharpe under null (multiple testing)
    from scipy.stats import norm
    e_max_sharpe = norm.ppf(1 - 1/n_trials)  # Bonferroni-like
    std_sharpe = 1 / np.sqrt(n_trades)
    
    dsr = (sharpe - e_max_sharpe) / std_sharpe if std_sharpe > 0 else 0
    
    print(f"  Observed Sharpe: {sharpe:.3f}")
    print(f"  Trials tested: {n_trials}")
    print(f"  Expected max Sharpe (null): {e_max_sharpe:.3f}")
    print(f"  Deflated Sharpe: {dsr:.3f}")
    print(f"  Significant (DSR > 1.96): {'YES' if dsr > 1.96 else 'NO'}")
    
    results['deflated_sharpe'] = {
        'observed_sharpe': float(sharpe),
        'n_trials': n_trials,
        'n_trades': n_trades,
        'skew': skew,
        'kurtosis': kurt,
        'expected_max_sharpe': float(e_max_sharpe),
        'deflated_sharpe': float(dsr),
        'significant': dsr > 1.96,
    }

# Monte Carlo Permutation Test
print("\n" + "="*70)
print("VALIDATOR AGENT: Monte Carlo Permutation")
print("="*70)

np.random.seed(42)
actual_mean = all_rets.mean()
actual_wr = (all_rets > 0).mean()
n = len(all_rets)

# Shuffle labels 10000 times
all_fwd = merged['fwd_ret_16'].dropna()
random_means = np.array([all_fwd.sample(n).mean() for _ in range(10000)])
mc_p = (random_means >= actual_mean).mean()

# Bootstrap CI
boot_means = np.array([all_rets.sample(n, replace=True).mean() for _ in range(10000)])
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

print(f"  n={n}, mean={actual_mean*100:+.4f}%, WR={actual_wr:.1%}")
print(f"  MC p-value: {mc_p:.4f}")
print(f"  Bootstrap CI: [{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%]")
print(f"  Significant (p < 0.05): {'YES' if mc_p < 0.05 else 'NO'}")

results['monte_carlo'] = {
    'n': n, 'mean': float(actual_mean), 'wr': float(actual_wr),
    'mc_p': float(mc_p), 'ci': [float(ci_lo), float(ci_hi)],
    'significant': bool(mc_p < 0.05),
}

# ═══════════════════════════════════════════════════════════════
# SEARCH AGENT: Threshold Sweep
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SEARCH AGENT: Threshold Optimization")
print("="*70)

# Sweep key parameters
best_config = None
best_score = -999

for depth_min in [0.001, 0.003, 0.005]:
    for depth_max in [0.010, 0.015, 0.020]:
        for wick_min in [0.30, 0.40, 0.50]:
            for wyckoff_pos_max in [0.20, 0.30, 0.40]:
                # Filter signals
                filtered = [s for s in signals 
                           if depth_min <= s['sweep_depth']/100 <= depth_max
                           and s['wick_ratio'] >= wick_min
                           and s['wyckoff_position'] <= wyckoff_pos_max]
                
                if len(filtered) < 10:
                    continue
                
                indices = [s['idx'] for s in filtered]
                rets = merged.iloc[indices]['fwd_ret_16'].dropna()
                if len(rets) < 10:
                    continue
                
                wr = (rets > 0).mean()
                mean_r = rets.mean()
                
                # Score = WR * mean_return * sqrt(n) (penalize low sample)
                score = wr * mean_r * np.sqrt(len(rets))
                
                if score > best_score:
                    best_score = score
                    best_config = {
                        'depth_min': depth_min, 'depth_max': depth_max,
                        'wick_min': wick_min, 'wyckoff_pos_max': wyckoff_pos_max,
                        'n': len(rets), 'wr': float(wr), 'mean': float(mean_r),
                        'score': float(score),
                    }

if best_config:
    print(f"  Best config: depth=[{best_config['depth_min']}, {best_config['depth_max']}]")
    print(f"               wick>={best_config['wick_min']}, wyckoff_pos<={best_config['wyckoff_pos_max']}")
    print(f"  n={best_config['n']}, WR={best_config['wr']:.1%}, mean={best_config['mean']*100:+.4f}%")
    results['threshold_optimization'] = best_config
else:
    print("  No valid config found")
    results['threshold_optimization'] = None

# ═══════════════════════════════════════════════════════════════
# SELECTOR AGENT: Should this strategy be enabled?
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SELECTOR AGENT: Deployment Decision")
print("="*70)

criteria = {
    'mc_significant': results.get('monte_carlo', {}).get('significant', False),
    'dsr_significant': results.get('deflated_sharpe', {}).get('significant', False),
    'wf_avg_wr_above_50': results.get('walk_forward', {}).get('avg_wr', 0) > 0.50,
    'n_above_30': n >= 30,
    'bull_regime_confirmed': True,  # from 5-agent gate
}

passed = sum(criteria.values())
total = len(criteria)

print(f"  Criteria met: {passed}/{total}")
for k, v in criteria.items():
    icon = '✅' if v else '❌'
    print(f"  {icon} {k}")

if passed >= 4:
    decision = 'DEPLOY'
    size = '0.5x'
elif passed >= 3:
    decision = 'CONDITIONAL'
    size = '0.3x'
else:
    decision = 'KILL'
    size = '0x'

results['selector'] = {
    'criteria': criteria,
    'passed': passed,
    'total': total,
    'decision': decision,
    'recommended_size': size,
}

print(f"\n  DECISION: {decision} (size: {size})")

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
with open(OUTPUT, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*70}")
print("FINAL VERDICT")
print(f"{'='*70}")
print(f"  Walk-forward: {results.get('walk_forward', {}).get('windows', 0)} windows, avg WR={results.get('walk_forward', {}).get('avg_wr', 0):.1%}")
print(f"  Deflated Sharpe: {results.get('deflated_sharpe', {}).get('deflated_sharpe', 0):.3f} (sig: {results.get('deflated_sharpe', {}).get('significant', False)})")
print(f"  Monte Carlo p: {results.get('monte_carlo', {}).get('mc_p', 1):.4f} (sig: {results.get('monte_carlo', {}).get('significant', False)})")
print(f"  Best threshold config: {results.get('threshold_optimization', {}).get('n', 0)} signals, WR={results.get('threshold_optimization', {}).get('wr', 0):.1%}")
print(f"  Decision: {decision} ({size})")
print(f"\nSaved to {OUTPUT}")
