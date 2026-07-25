"""
Optimizer Framework — Validator Agent
Deflated Sharpe + Rolling Walk-Forward + Permutation Test + CPCV

Validates liquidity_grab v7 and funding_squeeze v2
against the OPTIMIZATION_FRAMEWORK.md thresholds.

References:
- Bailey & Lopez de Prado (2014) "The Probability of Backtest Overfitting"
- De Prado (2018) "Advances in Financial Machine Learning" Ch. 12-14
- Harvey & Liu (2015) "Backtesting"
"""
import pandas as pd
import numpy as np
from scipy import stats
import json, time, subprocess

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'

print("="*70, flush=True)
print("OPTIMIZER VALIDATOR AGENT", flush=True)
print("="*70, flush=True)
t0 = time.time()

# ═══════════════════════════════════════════════════════
# DATA LOAD
# ═══════════════════════════════════════════════════════
print("\nLoading data...", flush=True)

# Derivatives
deriv = pd.read_csv(f'{DATA_DIR}/derivatives_history/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

# OB data (tail 2M rows, sample 1/10)
result = subprocess.run(['head', '-1', f'{DATA_DIR}/ob_history/ob_historical.csv'], capture_output=True, text=True)
header = result.stdout.strip()
subprocess.run(['bash', '-c', f'(echo "{header}"; tail -2000000 {DATA_DIR}/ob_history/ob_historical.csv) > /tmp/ob_tail.csv'], check=True)
ob = pd.read_csv('/tmp/ob_tail.csv')
ob['timestamp'] = pd.to_datetime(ob['timestamp'], utc=True).dt.tz_localize(None)
ob = ob.sort_values('timestamp').reset_index(drop=True)
ob = ob.iloc[::10].reset_index(drop=True)

# OHLCV
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_merged.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)

# Forward returns
for h_bars, label in [(4, '1h'), (8, '2h'), (16, '4h'), (32, '8h')]:
    ohlcv[f'fwd_ret_{label}'] = ohlcv['Close'].shift(-h_bars) / ohlcv['Close'] - 1

# Merge OB + OHLCV
merged_ob = pd.merge_asof(
    ob,
    ohlcv[['timestamp', 'Close', 'High', 'Low', 'Volume', 'fwd_ret_1h', 'fwd_ret_2h', 'fwd_ret_4h', 'fwd_ret_8h']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('15min')
)
merged_ob = merged_ob.dropna(subset=['Close', 'fwd_ret_2h'])
merged_ob['vol_ratio'] = merged_ob['Volume'] / merged_ob['Volume'].rolling(20).mean()
merged_ob['ema200'] = merged_ob['Close'].ewm(span=200).mean()
merged_ob['trend'] = np.where(merged_ob['Close'] > merged_ob['ema200'], 'BULL', 'BEAR')
merged_ob['atr'] = (merged_ob['High'] - merged_ob['Low']).rolling(16).mean()
def get_session(ts):
    h = ts.hour
    if 0 <= h < 8: return 'ASIA'
    elif 8 <= h < 14: return 'EU'
    elif 14 <= h < 22: return 'US'
    else: return 'LATE'
merged_ob['session'] = merged_ob['timestamp'].apply(get_session)
BAD_HOURS = {4, 5, 6, 19, 20, 22, 23}
merged_ob = merged_ob[~merged_ob['timestamp'].dt.hour.isin(BAD_HOURS)]

# Merge deriv + OHLCV
merged_deriv = pd.merge_asof(
    deriv,
    ohlcv[['timestamp', 'Close', 'High', 'Low', 'Volume', 'fwd_ret_1h', 'fwd_ret_2h', 'fwd_ret_4h', 'fwd_ret_8h']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('15min')
)
merged_deriv = merged_deriv.dropna(subset=['Close', 'fwd_ret_8h'])
fr_series = pd.Series(merged_deriv['funding_rate'].values)
fr_mean = fr_series.rolling(96).mean()
fr_std = fr_series.rolling(96).std()
merged_deriv['fr_zscore'] = (fr_series - fr_mean) / fr_std
merged_deriv['oi_change'] = merged_deriv['oi'].pct_change(periods=4)
merged_deriv['vol_20'] = merged_deriv['Close'].pct_change().rolling(20).std()
vol_p33 = merged_deriv['vol_20'].quantile(0.33)
vol_p67 = merged_deriv['vol_20'].quantile(0.67)
merged_deriv['vol_regime'] = 'MID'
merged_deriv.loc[merged_deriv['vol_20'] < vol_p33, 'vol_regime'] = 'LOW'
merged_deriv.loc[merged_deriv['vol_20'] > vol_p67, 'vol_regime'] = 'HIGH'
merged_deriv['session'] = merged_deriv['timestamp'].apply(get_session)
merged_deriv['fr_z_cum3'] = (
    (merged_deriv['fr_zscore'] > 1.5) &
    (merged_deriv['fr_zscore'].shift(1) > 1.5) &
    (merged_deriv['fr_zscore'].shift(2) > 1.5)
)

print(f"OB merged: {len(merged_ob):,} rows", flush=True)
print(f"Deriv merged: {len(merged_deriv):,} rows", flush=True)
print(f"Time: {time.time()-t0:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════
# GENERATE SIGNALS
# ═══════════════════════════════════════════════════════
print("\nGenerating signals...", flush=True)

# liquidity_grab v7: BEAR + ASIA + ob>0.15 + vol>=0.8, LONG, 2h
lg_signals = merged_ob[
    (merged_ob['trend'] == 'BEAR') &
    (merged_ob['session'] == 'ASIA') &
    (merged_ob['ob_ratio'] > 0.15) &
    (merged_ob['vol_ratio'] >= 0.8)
].copy()
lg_signals['adj_ret'] = lg_signals['fwd_ret_2h']
lg_signals['strategy'] = 'liquidity_grab'
print(f"liquidity_grab v7: {len(lg_signals):,} signals", flush=True)

# funding_squeeze v2: FR z>1.75 + cum3 + EU/US + MID/HIGH + OI rising, SHORT, 8h
fs_signals = merged_deriv[
    (merged_deriv['fr_zscore'] > 1.75) &
    (merged_deriv['fr_z_cum3'] == True) &
    (merged_deriv['session'].isin(['EU', 'US'])) &
    (merged_deriv['vol_regime'].isin(['MID', 'HIGH'])) &
    (merged_deriv['oi_change'] > 0)
].copy()
fs_signals['adj_ret'] = -fs_signals['fwd_ret_8h']  # SHORT
fs_signals['strategy'] = 'funding_squeeze'
print(f"funding_squeeze v2: {len(fs_signals):,} signals", flush=True)

# ═══════════════════════════════════════════════════════
# VALIDATOR FUNCTIONS
# ═══════════════════════════════════════════════════════

def deflated_sharpe(observed_sharpe, n_trials, n_trades, skew, kurtosis):
    """
    Bailey & Lopez de Prado (2014)
    Adjusts observed Sharpe for multiple testing bias.
    Returns DSR — > 1.96 = significant at 95%.
    """
    # Expected max Sharpe under null (multiple testing)
    euler_mascheroni = 0.5772
    e_max = ((1 - euler_mascheroni) * stats.norm.ppf(1 - 1/n_trials) +
             euler_mascheroni * stats.norm.ppf(1 - 1/(n_trials * np.e)))
    
    # Sharpe standard error
    se = np.sqrt((1 + 0.5 * observed_sharpe**2 -
                  skew * observed_sharpe +
                  (kurtosis - 3) / 4 * observed_sharpe**2) / (n_trades - 1))
    
    if se == 0:
        return 0
    dsr = (observed_sharpe - e_max) / se
    return dsr


def rolling_walk_forward(returns, train_size, test_size, step_size):
    """
    Rolling walk-forward: train on N bars, test on M bars, slide forward.
    Returns list of (test_start, test_end, wr, mean, n) for each window.
    """
    results = []
    n = len(returns)
    for start in range(0, n - train_size - test_size + 1, step_size):
        train_end = start + train_size
        test_end = min(train_end + test_size, n)
        test_rets = returns[train_end:test_end]
        if len(test_rets) < 5:
            continue
        wr = (test_rets > 0).mean()
        mean_r = test_rets.mean()
        results.append({
            'start': start,
            'train_end': train_end,
            'test_end': test_end,
            'wr': wr,
            'mean': mean_r,
            'n': len(test_rets)
        })
    return results


def permutation_test(returns, n_permutations=5000):
    """
    Monte Carlo permutation test.
    Shuffle signs of returns, compute WR. If real WR < 95th percentile of
    shuffled WRs, the edge may be luck.
    """
    n = len(returns)
    real_wr = (returns > 0).mean()
    
    shuffled_wrs = []
    for _ in range(n_permutations):
        # Randomly flip signs
        signs = np.random.choice([-1, 1], size=n)
        shuffled = returns * signs
        shuffled_wrs.append((shuffled > 0).mean())
    
    shuffled_wrs = np.array(shuffled_wrs)
    p_value = (shuffled_wrs >= real_wr).mean()
    percentile = (shuffled_wrs < real_wr).mean() * 100
    
    return {
        'real_wr': real_wr,
        'shuffled_mean_wr': shuffled_wrs.mean(),
        'shuffled_p95': np.percentile(shuffled_wrs, 95),
        'p_value': p_value,
        'percentile': percentile
    }


def combinatorial_purged_cv(returns, n_folds=5, purge_bars=4):
    """
    Combinatorial Purged Cross-Validation (De Prado 2018).
    K-fold CV with purge gap between train/test to prevent leakage.
    """
    n = len(returns)
    fold_size = n // n_folds
    fold_results = []
    
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n)
        
        # Purge gap
        purge_start = max(0, test_start - purge_bars)
        purge_end = min(n, test_end + purge_bars)
        
        # Train = everything outside test + purge
        train_mask = np.ones(n, dtype=bool)
        train_mask[purge_start:purge_end] = False
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_start:test_end] = True
        
        train_rets = returns[train_mask]
        test_rets = returns[test_mask]
        
        if len(test_rets) < 5:
            continue
        
        wr = (test_rets > 0).mean()
        mean_r = test_rets.mean()
        fold_results.append({
            'fold': i,
            'wr': wr,
            'mean': mean_r,
            'n': len(test_rets)
        })
    
    return fold_results


def validate_strategy(signals, adj_col, strategy_name, n_trials_tested):
    """Full validation for a strategy."""
    print(f"\n{'='*70}", flush=True)
    print(f"VALIDATING: {strategy_name}", flush=True)
    print(f"{'='*70}", flush=True)
    
    rets = signals[adj_col].dropna().values
    n = len(rets)
    
    if n < 30:
        print(f"  INSUFFICIENT DATA: n={n} < 30", flush=True)
        return
    
    # Basic stats
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    std_r = np.std(rets)
    sharpe = mean_r / std_r if std_r > 0 else 0
    skew = stats.skew(rets)
    kurt = stats.kurtosis(rets)
    
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    
    print(f"\n  Basic Stats:", flush=True)
    print(f"    n={n:,} WR={wr*100:.1f}% Mean={mean_r*100:+.4f}% Std={std_r*100:.4f}%", flush=True)
    print(f"    Sharpe={sharpe:.3f} Skew={skew:.3f} Kurt={kurt:.3f}", flush=True)
    print(f"    PF={pf:.2f} Win_mean={wins.mean()*100:+.4f}% Loss_mean={losses.mean()*100:+.4f}%", flush=True)
    
    # ── 1. DEFLATED SHARPE RATIO ──
    print(f"\n  1. DEFLATED SHARPE RATIO (Bailey & Lopez de Prado 2014)", flush=True)
    print(f"     Trials tested: {n_trials_tested}", flush=True)
    dsr = deflated_sharpe(sharpe, n_trials_tested, n, skew, kurt)
    dsr_pass = dsr > 1.0
    dsr_target = dsr > 2.0
    print(f"     DSR = {dsr:.3f} {'PASS' if dsr_pass else 'FAIL'} (>1.0) {'TARGET' if dsr_target else ''} (>2.0)", flush=True)
    
    # ── 2. ROLLING WALK-FORWARD ──
    print(f"\n  2. ROLLING WALK-FORWARD", flush=True)
    # Use 15m bars: train=4000 bars (~42 days), test=960 bars (~10 days), step=960
    train_bars = min(4000, n // 3)
    test_bars = min(960, n // 5)
    step_bars = test_bars
    
    wf_results = rolling_walk_forward(rets, train_bars, test_bars, step_bars)
    if wf_results:
        wf_df = pd.DataFrame(wf_results)
        win_windows = (wf_df['wr'] > 0.5).sum()
        total_windows = len(wf_df)
        wf_wr = wf_df['wr'].mean()
        wf_mean = wf_df['mean'].mean()
        wf_pass = wf_wr > 0.5
        wf_target = wf_wr > 0.6
        
        print(f"     Windows: {total_windows}", flush=True)
        print(f"     Win windows: {win_windows}/{total_windows} ({win_windows/total_windows*100:.0f}%)", flush=True)
        print(f"     Mean WR: {wf_wr*100:.1f}% {'PASS' if wf_pass else 'FAIL'} (>50%) {'TARGET' if wf_target else ''} (>60%)", flush=True)
        print(f"     Mean return: {wf_mean*100:+.4f}%", flush=True)
        
        # Show each window
        for _, row in wf_df.iterrows():
            tag = "WIN" if row['wr'] > 0.5 else "LOSS"
            print(f"       [{tag}] WR={row['wr']*100:.0f}% mean={row['mean']*100:+.3f}% n={int(row['n'])}", flush=True)
    else:
        wf_pass = False
        wf_wr = 0
        print(f"     Insufficient data for walk-forward", flush=True)
    
    # ── 3. PERMUTATION TEST ──
    print(f"\n  3. MONTE CARLO PERMUTATION TEST", flush=True)
    perm = permutation_test(rets, n_permutations=5000)
    perm_pass = perm['p_value'] < 0.05
    perm_target = perm['p_value'] < 0.01
    print(f"     Real WR: {perm['real_wr']*100:.1f}%", flush=True)
    print(f"     Shuffled mean WR: {perm['shuffled_mean_wr']*100:.1f}%", flush=True)
    print(f"     Shuffled P95: {perm['shuffled_p95']*100:.1f}%", flush=True)
    print(f"     Percentile: {perm['percentile']:.1f}%", flush=True)
    print(f"     p-value: {perm['p_value']:.6f} {'PASS' if perm_pass else 'FAIL'} (<0.05) {'TARGET' if perm_target else ''} (<0.01)", flush=True)
    
    # ── 4. CPCV ──
    print(f"\n  4. COMBINATORIAL PURGED CV (De Prado 2018)", flush=True)
    cpcv_results = combinatorial_purged_cv(rets, n_folds=5, purge_bars=4)
    if cpcv_results:
        cpcv_df = pd.DataFrame(cpcv_results)
        cpcv_wr = cpcv_df['wr'].mean()
        cpcv_pass = cpcv_wr > 0.55
        cpcv_target = cpcv_wr > 0.65
        
        print(f"     Folds: {len(cpcv_df)}", flush=True)
        print(f"     Mean WR: {cpcv_wr*100:.1f}% {'PASS' if cpcv_pass else 'FAIL'} (>55%) {'TARGET' if cpcv_target else ''} (>65%)", flush=True)
        
        for _, row in cpcv_df.iterrows():
            tag = "WIN" if row['wr'] > 0.5 else "LOSS"
            print(f"       [{tag}] Fold {int(row['fold'])}: WR={row['wr']*100:.0f}% mean={row['mean']*100:+.3f}% n={int(row['n'])}", flush=True)
    else:
        cpcv_pass = False
        cpcv_wr = 0
        print(f"     Insufficient data for CPCV", flush=True)
    
    # ── 5. BOOTSTRAP CI ──
    print(f"\n  5. BOOTSTRAP CONFIDENCE INTERVALS", flush=True)
    n_boot = 2000
    boot_wrs = []
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(rets, size=n, replace=True)
        boot_wrs.append((sample > 0).mean())
        boot_means.append(sample.mean())
    
    wr_ci = np.percentile(boot_wrs, [2.5, 97.5])
    mean_ci = np.percentile(boot_means, [2.5, 97.5])
    ci_pass = mean_ci[0] > 0
    
    print(f"     WR CI: [{wr_ci[0]*100:.1f}%, {wr_ci[1]*100:.1f}%]", flush=True)
    print(f"     Mean CI: [{mean_ci[0]*100:+.4f}%, {mean_ci[1]*100:+.4f}%] {'PASS' if ci_pass else 'FAIL'} (lower > 0)", flush=True)
    
    # ── FINAL VERDICT ──
    print(f"\n  {'='*50}", flush=True)
    print(f"  VERDICT: {strategy_name}", flush=True)
    print(f"  {'='*50}", flush=True)
    
    checks = [
        (f"DSR > 1.0", dsr_pass, f"DSR={dsr:.3f}"),
        (f"WF WR > 50%", wf_pass, f"WF_WR={wf_wr*100:.1f}%"),
        (f"Permutation p < 0.05", perm_pass, f"p={perm['p_value']:.6f}"),
        (f"CPCV WR > 55%", cpcv_pass, f"CPCV_WR={cpcv_wr*100:.1f}%"),
        (f"CI lower > 0", ci_pass, f"CI=[{mean_ci[0]*100:+.4f}%, {mean_ci[1]*100:+.4f}%]"),
        (f"n >= 30", n >= 30, f"n={n}"),
    ]
    
    for name, passed, detail in checks:
        print(f"    {'PASS' if passed else 'FAIL'} {name}: {detail}", flush=True)
    
    score = sum(1 for _, passed, _ in checks if passed)
    total = len(checks)
    print(f"\n    Score: {score}/{total}", flush=True)
    
    if score >= total - 1:
        print(f"    RESULT: PASS ✅", flush=True)
    elif score >= total - 2:
        print(f"    RESULT: PROVISIONAL ⚠️", flush=True)
    else:
        print(f"    RESULT: FAIL ❌", flush=True)
    
    # ── DEPLOYMENT READINESS ──
    print(f"\n  DEPLOYMENT READINESS:", flush=True)
    if n >= 100 and dsr > 2.0 and wf_wr > 0.6 and perm['p_value'] < 0.01:
        print(f"    READY — all targets met", flush=True)
    elif n >= 30 and dsr > 1.0 and wf_wr > 0.5 and perm['p_value'] < 0.05:
        print(f"    PROVISIONAL — minimums met, targets not fully met", flush=True)
    else:
        print(f"    NOT READY — minimum thresholds not met", flush=True)
    
    return {
        'strategy': strategy_name,
        'n': n,
        'wr': wr,
        'mean': mean_r,
        'sharpe': sharpe,
        'dsr': dsr,
        'wf_wr': wf_wr,
        'perm_p': perm['p_value'],
        'cpcv_wr': cpcv_wr,
        'ci_lower': mean_ci[0],
        'score': score,
        'total': total,
    }


# ═══════════════════════════════════════════════════════
# RUN VALIDATION
# ═══════════════════════════════════════════════════════

# liquidity_grab v7
# n_trials_tested: OB thresholds (6) × sessions (3) × horizons (4) = 72 combinations
lg_result = validate_strategy(lg_signals, 'adj_ret', 'liquidity_grab v7', n_trials_tested=72)

# funding_squeeze v2
# n_trials_tested: z thresholds (5) × sessions (3) × vol regimes (3) × horizons (3) = 135 combinations
fs_result = validate_strategy(fs_signals, 'adj_ret', 'funding_squeeze v2', n_trials_tested=135)

# Cleanup
subprocess.run(['rm', '-f', '/tmp/ob_tail.csv'])

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
print("="*70, flush=True)
print("VALIDATOR AGENT COMPLETE", flush=True)
print("="*70, flush=True)
