"""
8-Agent Protocol: Momentum v3.1 Exhaustion Filter
Forensic found v3 triggered 59 times in13 days (coin-flip WR).
v3.1 fixes: require DECEL, raise vol_div to -20%, extreme to >90th pctl,
fix OI matching, add dedup.

Agent 1: Forensics — data quality
Agent 2: Non-indicator — raw DECEL signal
Agent 3: Context filters (session, trend, vol)
Agent 4: Co-occurrence with Group A strategies
Agent 5: Walk-forward validation
Agent 6: Monte Carlo
Agent 7: Regime-conditional
Agent 8: Statistical significance gate
"""
import pandas as pd
import numpy as np
from scipy import stats
import json, os

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'

# Load OHLCV
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
print(f"OHLCV: {len(ohlcv)} bars, {ohlcv['timestamp'].iloc[0]} -> {ohlcv['timestamp'].iloc[-1]}")

# Load derivatives
deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

merged = pd.merge_asof(ohlcv, deriv[['timestamp','oi','ls_ratio','funding_rate']],
                       on='timestamp', direction='backward', tolerance=pd.Timedelta('30min'))

merged['oi_roc'] = merged['oi'].pct_change(4, fill_method=None)
merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')

for h in [4, 8, 16, 24, 48, 96]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

# ═══════════════════════════════════════════════════════
# COMPUTE MOMENTUM V3.1 SIGNALS
# ═══════════════════════════════════════════════════════
print("\nComputing momentum v3.1 signals...")

closes = merged['Close'].values
volumes = merged['Volume'].values
n = len(merged)

# Signal arrays
signals = []
last_trigger_idx = -999
last_trigger_dir = None

for idx in range(80, n):
    mom_5 = (closes[idx] - closes[idx-5]) / closes[idx-5]
    mom_10 = (closes[idx] - closes[idx-10]) / closes[idx-10]
    accel = mom_5 - mom_10 / 2

    # DECEL required
    decel = (mom_5 > 0 and accel < 0) or (mom_5 < 0 and accel > 0)
    if not decel:
        continue

    # Volume divergence (-20% threshold)
    vol_recent = np.mean(volumes[idx-5:idx])
    vol_prior = np.mean(volumes[idx-15:idx-5])
    vol_change = (vol_recent - vol_prior) / vol_prior if vol_prior > 0 else 0
    vol_div = (mom_5 > 0.005 and vol_change < -0.20) or (mom_5 < -0.005 and vol_change < -0.20)

    # Extreme move (>90th percentile)
    moves = []
    for j in range(max(0, idx-80), idx-5):
        if j+5 < n:
            m = abs(closes[j+5] - closes[j]) / closes[j]
            moves.append(m)
    current_move = abs(closes[idx] - closes[idx-5]) / closes[idx-5]
    percentile = sum(1 for m in moves if m < current_move) / len(moves) * 100 if moves else 0
    extreme = percentile > 90

    # OI divergence
    oi_roc = merged.iloc[idx].get('oi_roc', 0) or 0
    oi_div = (mom_5 > 0.005 and oi_roc < -0.02) or (mom_5 < -0.005 and oi_roc < -0.02)

    # Need DECEL + at least 1 more
    additional = sum([vol_div, extreme, oi_div])
    if additional < 1:
        continue

    # Dedup
    direction = 'SHORT' if mom_5 > 0 else 'LONG'
    if (idx - last_trigger_idx < 4 and direction == last_trigger_dir):
        continue

    last_trigger_idx = idx
    last_trigger_dir = direction

    # Conviction
    base = 0.55
    if vol_div: base += 0.15
    if extreme: base += 0.10
    if oi_div: base += 0.10
    conviction = min(base, 0.90)

    signals.append({
        'idx': idx,
        'timestamp': merged.iloc[idx]['timestamp'],
        'price': closes[idx],
        'direction': direction,
        'conviction': conviction,
        'mom_5': mom_5, 'accel': accel,
        'vol_change': vol_change, 'percentile': percentile,
        'oi_roc': oi_roc,
        'decel': decel, 'vol_div': vol_div,
        'extreme': extreme, 'oi_div': oi_div,
        'signals_count': 1 + additional,
        'trend': merged.iloc[idx]['trend'],
    })

print(f"v3.1 signals: {len(signals)} (was 59 in v3)")
print(f"Reduction: {59 - len(signals)} fewer signals ({(59-len(signals))/59*100:.0f}% reduction)")

if not signals:
    print("NO SIGNALS — strategy is dead after tightening. Exiting.")
    exit(0)

sdf = pd.DataFrame(signals)

# ═══════════════════════════════════════════════════════
# AGENT 1: FORENSICS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 1: FORENSICS")
print("="*70)

print(f"Total signals: {len(sdf)}")
print(f"Direction split: LONG={len(sdf[sdf['direction']=='LONG'])} SHORT={len(sdf[sdf['direction']=='SHORT'])}")
print(f"Conviction range: {sdf['conviction'].min():.2f} - {sdf['conviction'].max():.2f}")
print(f"Mean conviction: {sdf['conviction'].mean():.2f}")

# Signal composition
print(f"\nSignal composition:")
print(f"  DECEL (required): {sdf['decel'].sum()}/{len(sdf)} (100%)")
print(f"  Volume div: {sdf['vol_div'].sum()}/{len(sdf)} ({sdf['vol_div'].mean()*100:.1f}%)")
print(f"  Extreme: {sdf['extreme'].sum()}/{len(sdf)} ({sdf['extreme'].mean()*100:.1f}%)")
print(f"  OI div: {sdf['oi_div'].sum()}/{len(sdf)} ({sdf['oi_div'].mean()*100:.1f}%)")

# Forward returns
for h in [4, 8, 16, 24, 48]:
    col = f'fwd_ret_{h}'
    sdf[col] = sdf.apply(lambda r: merged.iloc[r['idx']][col] if r['idx'] + h < len(merged) else np.nan, axis=1)
    rets = sdf[col].dropna()
    if len(rets) > 0:
        dir_mult = sdf.loc[rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
        adj_rets = rets * dir_mult
        wr = (adj_rets > 0).mean()
        mean_r = adj_rets.mean()
        print(f"  {h}h: mean={mean_r*100:+.2f}% WR={wr*100:.1f}% n={len(rets)}")

# ═══════════════════════════════════════════════════════
# AGENT 2: NON-INDICATOR — Test DECEL alone
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 2: NON-INDICATOR — DECEL signal alone")
print("="*70)

decel_only = []
last_idx = -999
last_dir = None

for idx in range(80, n):
    mom_5 = (closes[idx] - closes[idx-5]) / closes[idx-5]
    mom_10 = (closes[idx] - closes[idx-10]) / closes[idx-10]
    accel = mom_5 - mom_10 / 2

    decel = (mom_5 > 0 and accel < 0) or (mom_5 < 0 and accel > 0)
    if not decel:
        continue

    direction = 'SHORT' if mom_5 > 0 else 'LONG'
    if idx - last_idx < 4 and direction == last_dir:
        continue
    last_idx = idx
    last_dir = direction

    decel_only.append({
        'idx': idx, 'direction': direction,
        'price': closes[idx],
    })

dedf = pd.DataFrame(decel_only)
print(f"DECEL-only signals: {len(dedf)}")

for h in [4, 8, 16, 24]:
    col = f'fwd_ret_{h}'
    dedf[col] = dedf.apply(lambda r: merged.iloc[r['idx']][col] if r['idx'] + h < len(merged) else np.nan, axis=1)
    rets = dedf[col].dropna()
    if len(rets) > 0:
        dir_mult = dedf.loc[rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
        adj_rets = rets * dir_mult
        wr = (adj_rets > 0).mean()
        mean_r = adj_rets.mean()
        print(f"  {h}h: mean={mean_r*100:+.2f}% WR={wr*100:.1f}% n={len(rets)}")

# ═══════════════════════════════════════════════════════
# AGENT 3: CONTEXT FILTERS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 3: CONTEXT FILTERS")
print("="*70)

# Session filter
def get_session(ts):
    h = ts.hour
    if 0 <= h < 8: return 'ASIA'
    elif 8 <= h < 14: return 'EU'
    elif 14 <= h < 22: return 'US'
    else: return 'LATE'

sdf['session'] = sdf['timestamp'].apply(get_session)

for session in ['ASIA', 'EU', 'US', 'LATE']:
    sub = sdf[sdf['session'] == session]
    if len(sub) < 3:
        continue
    col = 'fwd_ret_16'
    rets = sub[col].dropna()
    if len(rets) > 0:
        dir_mult = sub.loc[rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
        adj_rets = rets * dir_mult
        wr = (adj_rets > 0).mean()
        mean_r = adj_rets.mean()
        print(f"  {session}: mean={mean_r*100:+.2f}% WR={wr*100:.1f}% n={len(rets)}")

# Trend filter
for trend in ['BULL', 'BEAR']:
    sub = sdf[sdf['trend'] == trend]
    if len(sub) < 3:
        continue
    col = 'fwd_ret_16'
    rets = sub[col].dropna()
    if len(rets) > 0:
        dir_mult = sub.loc[rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
        adj_rets = rets * dir_mult
        wr = (adj_rets > 0).mean()
        mean_r = adj_rets.mean()
        print(f"  {trend}: mean={mean_r*100:+.2f}% WR={wr*100:.1f}% n={len(rets)}")

# Conviction filter
for conv_range in [(0.55, 0.65), (0.65, 0.75), (0.75, 1.0)]:
    sub = sdf[(sdf['conviction'] >= conv_range[0]) & (sdf['conviction'] < conv_range[1])]
    if len(sub) < 3:
        continue
    col = 'fwd_ret_16'
    rets = sub[col].dropna()
    if len(rets) > 0:
        dir_mult = sub.loc[rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
        adj_rets = rets * dir_mult
        wr = (adj_rets > 0).mean()
        mean_r = adj_rets.mean()
        print(f"  Conv {conv_range[0]:.2f}-{conv_range[1]:.2f}: mean={mean_r*100:+.2f}% WR={wr*100:.1f}% n={len(rets)}")

# ═══════════════════════════════════════════════════════
# AGENT 4: CO-OCCURRENCE WITH GROUP A
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 4: CO-OCCURRENCE (pairs with Group A event triggers)")
print("="*70)

# Load fired signals from strategy_signals.jsonl if available
sig_path = f'{DATA_DIR}/../live/data/strategy_signals.jsonl'
group_a_fired = []
if os.path.exists(sig_path):
    with open(sig_path) as f:
        for line in f:
            try:
                s = json.loads(line)
                if s.get('strategy') in ['trade_flow', 'orderbook_imbalance', 'funding_arb',
                                          'judas_sweep', 'liquidation_cascade', 'positioning_fade',
                                          'whale_watch', 'failed_breakout']:
                    group_a_fired.append(s)
            except:
                pass

print(f"Group A fired signals loaded: {len(group_a_fired)}")

# For each momentum_v3 signal, check if Group A fired within ±4 bars
co_occurrence = []
for _, sig in sdf.iterrows():
    sig_ts = sig['timestamp']
    for ga in group_a_fired:
        try:
            ga_ts = pd.to_datetime(ga.get('timestamp', ''))
            if abs((ga_ts - sig_ts).total_seconds()) <= 4 * 900:  # 4 bars
                co_occurrence.append({
                    'mv3_ts': sig_ts, 'mv3_dir': sig['direction'],
                    'ga_strat': ga.get('strategy'), 'ga_dir': ga.get('direction'),
                    'same_dir': sig['direction'] == ga.get('direction'),
                })
                break
        except:
            pass

print(f"Co-occurrences found: {len(co_occurrence)}")
if co_occurrence:
    same_dir = sum(1 for c in co_occurrence if c['same_dir'])
    print(f"  Same direction: {same_dir}/{len(co_occurrence)} ({same_dir/len(co_occurrence)*100:.1f}%)")

# ═══════════════════════════════════════════════════════
# AGENT 5: WALK-FORWARD
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 5: WALK-FORWARD (4-week train, 1-week test)")
print("="*70)

sdf_ts = sdf.set_index('timestamp')
weeks = sorted(sdf_ts.index.isocalendar().week.unique())

wf_results = []
for i in range(0, len(weeks) - 4):
    train_weeks = weeks[i:i+4]
    test_week = weeks[i+4] if i+4 < len(weeks) else None
    if not test_week:
        break

    train = sdf_ts[sdf_ts.index.isocalendar().week.isin(train_weeks)]
    test = sdf_ts[sdf_ts.index.isocalendar().week == test_week]

    if len(train) < 5 or len(test) < 1:
        continue

    col = 'fwd_ret_16'
    train_rets = train[col].dropna()
    test_rets = test[col].dropna()

    if len(train_rets) > 0 and len(test_rets) > 0:
        train_dir = train.loc[train_rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
        test_dir = test.loc[test_rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
        train_adj = train_rets * train_dir
        test_adj = test_rets * test_dir

        wf_results.append({
            'test_week': test_week,
            'train_wr': (train_adj > 0).mean(),
            'test_wr': (test_adj > 0).mean(),
            'train_mean': train_adj.mean(),
            'test_mean': test_adj.mean(),
            'n_test': len(test_rets),
        })

if wf_results:
    wf_df = pd.DataFrame(wf_results)
    print(f"Walk-forward periods: {len(wf_df)}")
    print(f"Train WR: {wf_df['train_wr'].mean()*100:.1f}% (mean)")
    print(f"Test WR: {wf_df['test_wr'].mean()*100:.1f}% (mean)")
    print(f"Test mean return: {wf_df['test_mean'].mean()*100:+.2f}%")
    for _, row in wf_df.iterrows():
        print(f"  Week {row['test_week']}: train_WR={row['train_wr']*100:.0f}% -> test_WR={row['test_wr']*100:.0f}% (n={row['n_test']})")

# ═══════════════════════════════════════════════════════
# AGENT 6: MONTE CARLO
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 6: MONTE CARLO (10k sims)")
print("="*70)

col = 'fwd_ret_16'
all_rets = sdf[col].dropna()
if len(all_rets) > 0:
    dir_mult = sdf.loc[all_rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
    adj_rets = (all_rets * dir_mult).values

    n_sims = 10000
    horizon = 30  # trades per sim
    sim_results = []
    for _ in range(n_sims):
        sampled = np.random.choice(adj_rets, size=horizon, replace=True)
        sim_results.append(sampled.sum())

    sim_results = np.array(sim_results)
    p5, p25, p50, p75, p95 = np.percentile(sim_results, [5, 25, 50, 75, 95])
    prob_loss = (sim_results < 0).mean()

    print(f"30-trade horizon:")
    print(f"  P5:  {p5*100:+.2f}%")
    print(f"  P25: {p25*100:+.2f}%")
    print(f"  P50: {p50*100:+.2f}%")
    print(f"  P75: {p75*100:+.2f}%")
    print(f"  P95: {p95*100:+.2f}%")
    print(f"  Prob(loss): {prob_loss*100:.1f}%")
    print(f"  Mean: {sim_results.mean()*100:+.2f}%")
    print(f"  Std: {sim_results.std()*100:.2f}%")

# ═══════════════════════════════════════════════════════
# AGENT 7: REGIME-CONDITIONAL
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 7: REGIME-CONDITIONAL")
print("="*70)

for trend in ['BULL', 'BEAR']:
    sub = sdf[sdf['trend'] == trend]
    if len(sub) < 3:
        continue

    for direction in ['LONG', 'SHORT']:
        sub_dir = sub[sub['direction'] == direction]
        if len(sub_dir) < 2:
            continue

        col = 'fwd_ret_16'
        rets = sub_dir[col].dropna()
        if len(rets) > 0:
            dir_mult = 1 if direction == 'LONG' else -1
            adj_rets = rets * dir_mult
            wr = (adj_rets > 0).mean()
            mean_r = adj_rets.mean()
            print(f"  {trend}+{direction}: mean={mean_r*100:+.2f}% WR={wr*100:.1f}% n={len(rets)}")

# ═══════════════════════════════════════════════════════
# AGENT 8: STATISTICAL SIGNIFICANCE
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 8: STATISTICAL SIGNIFICANCE")
print("="*70)

col = 'fwd_ret_16'
all_rets = sdf[col].dropna()
if len(all_rets) > 0:
    dir_mult = sdf.loc[all_rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
    adj_rets = all_rets * dir_mult

    # t-test: is mean return significantly different from 0?
    t_stat, p_value = stats.ttest_1samp(adj_rets, 0)
    print(f"t-test (mean != 0): t={t_stat:.3f}, p={p_value:.4f}")
    print(f"  Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")
    print(f"  Significant at 1%: {'YES' if p_value < 0.01 else 'NO'}")

    # One-sided: is mean return > 0?
    p_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    print(f"  One-sided (mean > 0): p={p_one_sided:.4f}")

    # Bootstrap CI
    n_bootstrap = 10000
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(adj_rets, size=len(adj_rets), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    print(f"  Bootstrap 95% CI: [{ci_low*100:+.2f}%, {ci_high*100:+.2f}%]")
    print(f"  CI excludes 0: {'YES' if ci_low > 0 or ci_high < 0 else 'NO'}")

    # Profit factor
    wins = adj_rets[adj_rets > 0]
    losses = adj_rets[adj_rets < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Win rate: {(adj_rets > 0).mean()*100:.1f}%")
    print(f"  Mean win: {wins.mean()*100:+.2f}%" if len(wins) > 0 else "  No wins")
    print(f"  Mean loss: {losses.mean()*100:+.2f}%" if len(losses) > 0 else "  No losses")

    # Final verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    n = len(adj_rets)
    wr = (adj_rets > 0).mean()
    mean_r = adj_rets.mean()

    pass_count = 0
    checks = []

    # Check 1: n >= 20
    c = n >= 20
    pass_count += c
    checks.append(f"n >= 20: {'PASS' if c else 'FAIL'} ({n})")

    # Check 2: WR >= 52%
    c = wr >= 0.52
    pass_count += c
    checks.append(f"WR >= 52%: {'PASS' if c else 'FAIL'} ({wr*100:.1f}%)")

    # Check 3: mean > 0
    c = mean_r > 0
    pass_count += c
    checks.append(f"Mean > 0: {'PASS' if c else 'FAIL'} ({mean_r*100:+.2f}%)")

    # Check 4: p < 0.10
    c = p_one_sided < 0.10
    pass_count += c
    checks.append(f"p < 0.10: {'PASS' if c else 'FAIL'} ({p_one_sided:.4f})")

    # Check 5: PF >= 1.2
    c = pf >= 1.2
    pass_count += c
    checks.append(f"PF >= 1.2: {'PASS' if c else 'FAIL'} ({pf:.2f})")

    # Check 6: CI excludes 0
    c = ci_low > 0
    pass_count += c
    checks.append(f"CI > 0: {'PASS' if c else 'FAIL'} ([{ci_low*100:+.2f}%, {ci_high*100:+.2f}%])")

    for chk in checks:
        print(f"  {chk}")

    print(f"\n  Score: {pass_count}/6")
    if pass_count >= 5:
        print("  RESULT: PASS — strategy has edge")
    elif pass_count >= 3:
        print("  RESULT: PROVISIONAL — needs more data")
    else:
        print("  RESULT: FAIL — no statistical edge")

print("\nDone.")
