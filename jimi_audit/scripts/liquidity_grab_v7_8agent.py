"""
8-Agent Protocol: Liquidity Grab v7
L2 orderbook mean-reversion at 2h horizon

Agent 1: Forensics — signal quality, direction, conviction
Agent 2: Non-indicator — raw ob_ratio predictive power
Agent 3: Context filters (session, regime, vol)
Agent 4: Co-occurrence — which strategies fire together
Agent 5: Walk-forward — weekly stability
Agent 6: Monte Carlo — risk simulation
Agent 7: Regime-conditional — BULL/BEAR/RANGING breakdown
Agent 8: Statistical significance — t-test, bootstrap CI, gate decision
"""
import pandas as pd
import numpy as np
from scipy import stats
import subprocess, time

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
OB_FILE = f'{DATA_DIR}/ob_history/ob_historical.csv'
TAIL_FILE = '/tmp/ob_tail.csv'

print("="*70, flush=True)
print("8-AGENT PROTOCOL: LIQUIDITY GRAB v7", flush=True)
print("="*70, flush=True)
t0 = time.time()

# ═══════════════════════════════════════════════════════
# DATA LOAD
# ═══════════════════════════════════════════════════════
print("\nLoading data...", flush=True)

result = subprocess.run(['head', '-1', OB_FILE], capture_output=True, text=True)
header = result.stdout.strip()
subprocess.run(['bash', '-c', f'(echo "{header}"; tail -2000000 {OB_FILE}) > {TAIL_FILE}'], check=True)

ob = pd.read_csv(TAIL_FILE)
ob['timestamp'] = pd.to_datetime(ob['timestamp'], utc=True).dt.tz_localize(None)
ob = ob.sort_values('timestamp').reset_index(drop=True)
ob = ob.iloc[::10].reset_index(drop=True)  # 1/10 sampling
print(f"OB: {len(ob):,} rows", flush=True)

ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_merged.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)

# 2h forward return (8 bars of 15m)
ohlcv['fwd_ret_2h'] = ohlcv['Close'].shift(-8) / ohlcv['Close'] - 1
ohlcv['fwd_ret_1h'] = ohlcv['Close'].shift(-4) / ohlcv['Close'] - 1
ohlcv['fwd_ret_4h'] = ohlcv['Close'].shift(-16) / ohlcv['Close'] - 1

merged = pd.merge_asof(
    ob,
    ohlcv[['timestamp', 'Close', 'High', 'Low', 'Volume', 'fwd_ret_1h', 'fwd_ret_2h', 'fwd_ret_4h']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('15min')
)
merged = merged.dropna(subset=['Close', 'fwd_ret_2h'])
print(f"Merged: {len(merged):,} rows", flush=True)

# ═══════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════
merged['ob_ratio_delta_3'] = merged['ob_ratio'] - merged['ob_ratio'].shift(3)
merged['ob_ratio_delta_6'] = merged['ob_ratio'] - merged['ob_ratio'].shift(6)
merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')
merged['atr'] = (merged['High'] - merged['Low']).rolling(16).mean()

# Session
def get_session(ts):
    h = ts.hour
    if 0 <= h < 8: return 'ASIA'
    elif 8 <= h < 14: return 'EU'
    elif 14 <= h < 22: return 'US'
    else: return 'LATE'
merged['session'] = merged['timestamp'].apply(get_session)

# Bad hours
BAD_HOURS = {4, 5, 6, 19, 20, 22, 23}
merged = merged[~merged['timestamp'].dt.hour.isin(BAD_HOURS)]

# ═══════════════════════════════════════════════════════
# SCENARIO SIGNALS
# ═══════════════════════════════════════════════════════
print("\nGenerating v7 signals...", flush=True)

# Scenario A: BEAR + ob_ratio > 0.15
scen_a = merged[(merged['trend'] == 'BEAR') & (merged['ob_ratio'] > 0.15) & (merged['vol_ratio'] >= 0.8)].copy()
scen_a['scenario'] = 'A'
scen_a['direction'] = 'LONG'
scen_a['adj_ret'] = scen_a['fwd_ret_2h']  # LONG = positive return

# Scenario B: OB spike reversal (delta negative but reversing)
merged['ob_spike'] = merged['ob_ratio_delta_3'].abs() > 0.5
merged['ob_reversal'] = (merged['ob_ratio_delta_3'] < -0.5) & (merged['ob_ratio_delta_3'] > merged['ob_ratio_delta_3'].shift(1))
scen_b = merged[merged['ob_reversal']].copy()
scen_b['scenario'] = 'B'
scen_b['direction'] = 'LONG'
scen_b['adj_ret'] = scen_b['fwd_ret_2h']

signals = pd.concat([scen_a, scen_b], ignore_index=True)
signals = signals.dropna(subset=['adj_ret'])
print(f"Total signals: {len(signals):,} (A={len(scen_a):,} B={len(scen_b):,})", flush=True)
print(f"Time: {time.time()-t0:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 1: FORENSICS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 1: FORENSICS", flush=True)
print("="*70, flush=True)

for scenario in ['A', 'B']:
    sub = signals[signals['scenario'] == scenario]
    if len(sub) < 10:
        print(f"  Scenario {scenario}: {len(sub)} signals (too few)", flush=True)
        continue
    
    rets = sub['adj_ret']
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    
    print(f"\n  Scenario {scenario}:", flush=True)
    print(f"    Signals: {len(sub):,}", flush=True)
    print(f"    WR: {wr*100:.1f}%", flush=True)
    print(f"    Mean return: {mean_r*100:+.4f}%", flush=True)
    print(f"    p-value: {p1:.6f}", flush=True)
    print(f"    OB ratio range: [{sub['ob_ratio'].min():.3f}, {sub['ob_ratio'].max():.3f}]", flush=True)
    print(f"    Mean OB ratio: {sub['ob_ratio'].mean():.3f}", flush=True)

# Combined
rets = signals['adj_ret']
wr = (rets > 0).mean()
print(f"\n  COMBINED:", flush=True)
print(f"    Signals: {len(signals):,}", flush=True)
print(f"    WR: {wr*100:.1f}%", flush=True)
print(f"    Mean: {rets.mean()*100:+.4f}%", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 2: NON-INDICATOR — raw ob_ratio predictive power
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 2: NON-INDICATOR — raw ob_ratio at 2h", flush=True)
print("="*70, flush=True)

for thresh in [0.10, 0.15, 0.20, 0.30, 0.40]:
    long_sig = merged[merged['ob_ratio'] > thresh]
    short_sig = merged[merged['ob_ratio'] < -thresh]
    
    if len(long_sig) > 50:
        rets = long_sig['fwd_ret_2h']
        wr = (rets > 0).mean()
        t, p = stats.ttest_1samp(rets, 0)
        p1 = p/2 if t > 0 else 1-p/2
        print(f"  LONG(ob>{thresh:.2f}): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.6f} n={len(long_sig):,}", flush=True)
    
    if len(short_sig) > 50:
        rets = -short_sig['fwd_ret_2h']
        wr = (rets > 0).mean()
        t, p = stats.ttest_1samp(rets, 0)
        p1 = p/2 if t > 0 else 1-p/2
        print(f"  SHORT(ob<{thresh:.2f}): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.6f} n={len(short_sig):,}", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 3: CONTEXT FILTERS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 3: CONTEXT FILTERS", flush=True)
print("="*70, flush=True)

# Session
print("\n  By session:", flush=True)
for session in ['ASIA', 'EU', 'US']:
    sub = signals[signals['session'] == session]
    if len(sub) < 10:
        continue
    rets = sub['adj_ret']
    wr = (rets > 0).mean()
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    print(f"    {session}: WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.6f} n={len(sub):,}", flush=True)

# Vol ratio buckets
print("\n  By vol_ratio:", flush=True)
for vr_label, vr_lo, vr_hi in [('low(0.8-1.0)', 0.8, 1.0), ('med(1.0-1.5)', 1.0, 1.5), ('high(1.5+)', 1.5, 10.0)]:
    sub = signals[(signals['vol_ratio'] >= vr_lo) & (signals['vol_ratio'] < vr_hi)]
    if len(sub) < 10:
        continue
    rets = sub['adj_ret']
    wr = (rets > 0).mean()
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    print(f"    {vr_label}: WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.6f} n={len(sub):,}", flush=True)

# OB ratio intensity
print("\n  By OB ratio intensity:", flush=True)
for ob_label, ob_lo, ob_hi in [('0.15-0.25', 0.15, 0.25), ('0.25-0.40', 0.25, 0.40), ('0.40+', 0.40, 2.0)]:
    sub = signals[(signals['ob_ratio'] >= ob_lo) & (signals['ob_ratio'] < ob_hi)]
    if len(sub) < 10:
        continue
    rets = sub['adj_ret']
    wr = (rets > 0).mean()
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    print(f"    {ob_label}: WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.6f} n={len(sub):,}", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 4: CO-OCCURRENCE (skipped — need strategy_signals.jsonl)
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 4: CO-OCCURRENCE", flush=True)
print("="*70, flush=True)
print("  Skipped — requires live strategy_signals.jsonl", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 5: WALK-FORWARD
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 5: WALK-FORWARD", flush=True)
print("="*70, flush=True)

signals['week'] = signals['timestamp'].dt.isocalendar().week.astype(int)
signals['year'] = signals['timestamp'].dt.isocalendar().year.astype(int)
signals['yw'] = signals['year'] * 100 + signals['week']

wf_results = []
for yw in sorted(signals['yw'].unique()):
    sub = signals[signals['yw'] == yw]
    rets = sub['adj_ret'].dropna()
    if len(rets) < 3:
        continue
    wr = (rets > 0).mean()
    wf_results.append({'yw': yw, 'n': len(rets), 'wr': wr, 'mean': rets.mean()})

wf_df = pd.DataFrame(wf_results)
if len(wf_df) > 0:
    win_weeks = (wf_df['wr'] > 0.5).sum()
    total_weeks = len(wf_df)
    print(f"  Walk-forward weeks: {total_weeks}", flush=True)
    print(f"  Winning weeks: {win_weeks}/{total_weeks} ({win_weeks/total_weeks*100:.0f}%)", flush=True)
    print(f"  Mean WR: {wf_df['wr'].mean()*100:.1f}%", flush=True)
    print(f"  Mean return: {wf_df['mean'].mean()*100:+.4f}%", flush=True)
    
    print(f"\n  Worst 5 weeks:", flush=True)
    for _, row in wf_df.nsmallest(5, 'wr').iterrows():
        tag = "WIN" if row['wr'] > 0.5 else "LOSS"
        print(f"    [{tag}] {int(row['yw'])}: WR={row['wr']*100:.0f}% mean={row['mean']*100:+.3f}% n={int(row['n'])}", flush=True)
    
    print(f"\n  Best 5 weeks:", flush=True)
    for _, row in wf_df.nlargest(5, 'wr').iterrows():
        tag = "WIN" if row['wr'] > 0.5 else "LOSS"
        print(f"    [{tag}] {int(row['yw'])}: WR={row['wr']*100:.0f}% mean={row['mean']*100:+.3f}% n={int(row['n'])}", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 6: MONTE CARLO
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 6: MONTE CARLO", flush=True)
print("="*70, flush=True)

rets = signals['adj_ret'].dropna().values
if len(rets) > 30:
    n_sims = 5000
    horizon = 30
    sims = []
    for _ in range(n_sims):
        sampled = np.random.choice(rets, size=horizon, replace=True)
        sims.append(sampled.sum())
    sims = np.array(sims)
    p5, p25, p50, p75, p95 = np.percentile(sims, [5, 25, 50, 75, 95])
    print(f"  30-trade horizon:", flush=True)
    print(f"    P5={p5*100:+.2f}% P25={p25*100:+.2f}% P50={p50*100:+.2f}% P75={p75*100:+.2f}% P95={p95*100:+.2f}%", flush=True)
    print(f"    Prob(loss)={(sims<0).mean()*100:.1f}%", flush=True)
    print(f"    Expected 30-trade PnL: {p50*100:+.2f}%", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 7: REGIME-CONDITIONAL
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 7: REGIME-CONDITIONAL", flush=True)
print("="*70, flush=True)

for scenario in ['A', 'B']:
    sub = signals[signals['scenario'] == scenario]
    if len(sub) < 10:
        continue
    
    for trend in ['BULL', 'BEAR']:
        sub2 = sub[sub['trend'] == trend]
        if len(sub2) < 5:
            continue
        rets = sub2['adj_ret'].dropna()
        wr = (rets > 0).mean()
        t, p = stats.ttest_1samp(rets, 0)
        p1 = p/2 if t > 0 else 1-p/2
        print(f"  Scen {scenario}+{trend}: WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.6f} n={len(rets):,}", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 8: STATISTICAL SIGNIFICANCE
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 8: STATISTICAL SIGNIFICANCE", flush=True)
print("="*70, flush=True)

for scenario in ['A', 'B']:
    sub = signals[signals['scenario'] == scenario]
    if len(sub) < 20:
        print(f"\n  Scenario {scenario}: TOO FEW ({len(sub)})", flush=True)
        continue
    
    rets = sub['adj_ret'].dropna().values
    n_trades = len(rets)
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    
    # Bootstrap CI
    boots = [np.random.choice(rets, size=n_trades, replace=True).mean() for _ in range(2000)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    
    # Profit factor
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    
    print(f"\n  Scenario {scenario}:", flush=True)
    print(f"    t-test: t={t:.3f} p(one-sided)={p1:.6f}", flush=True)
    print(f"    Bootstrap CI: [{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%]", flush=True)
    print(f"    WR={wr*100:.1f}% PF={pf:.2f}", flush=True)
    print(f"    Mean win: {wins.mean()*100:+.4f}% Mean loss: {losses.mean()*100:+.4f}%", flush=True)
    
    # Gate checks
    checks = []
    c = n_trades >= 30
    checks.append(f"n >= 30: {'PASS' if c else 'FAIL'} ({n_trades})")
    c = wr >= 0.52
    checks.append(f"WR >= 52%: {'PASS' if c else 'FAIL'} ({wr*100:.1f}%)")
    c = mean_r > 0
    checks.append(f"Mean > 0: {'PASS' if c else 'FAIL'} ({mean_r*100:+.4f}%)")
    c = p1 < 0.05
    checks.append(f"p < 0.05: {'PASS' if c else 'FAIL'} ({p1:.6f})")
    c = pf >= 1.2
    checks.append(f"PF >= 1.2: {'PASS' if c else 'FAIL'} ({pf:.2f})")
    c = ci_lo > 0
    checks.append(f"CI > 0: {'PASS' if c else 'FAIL'} ([{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%])")
    
    print(f"\n  VERDICT — Scenario {scenario}:", flush=True)
    for chk in checks:
        print(f"    {chk}", flush=True)
    score = sum(1 for chk in checks if 'PASS' in chk)
    print(f"    Score: {score}/6", flush=True)
    if score >= 5:
        print(f"    RESULT: PASS ✅", flush=True)
    elif score >= 3:
        print(f"    RESULT: PROVISIONAL ⚠️", flush=True)
    else:
        print(f"    RESULT: FAIL ❌", flush=True)

# Combined verdict
print("\n  COMBINED VERDICT:", flush=True)
all_rets = signals['adj_ret'].dropna().values
if len(all_rets) >= 20:
    n_trades = len(all_rets)
    wr = (all_rets > 0).mean()
    mean_r = all_rets.mean()
    t, p = stats.ttest_1samp(all_rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    boots = [np.random.choice(all_rets, size=n_trades, replace=True).mean() for _ in range(2000)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    wins = all_rets[all_rets > 0]
    losses = all_rets[all_rets < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    
    checks = []
    c = n_trades >= 30
    checks.append(f"n >= 30: {'PASS' if c else 'FAIL'} ({n_trades})")
    c = wr >= 0.52
    checks.append(f"WR >= 52%: {'PASS' if c else 'FAIL'} ({wr*100:.1f}%)")
    c = mean_r > 0
    checks.append(f"Mean > 0: {'PASS' if c else 'FAIL'} ({mean_r*100:+.4f}%)")
    c = p1 < 0.05
    checks.append(f"p < 0.05: {'PASS' if c else 'FAIL'} ({p1:.6f})")
    c = pf >= 1.2
    checks.append(f"PF >= 1.2: {'PASS' if c else 'FAIL'} ({pf:.2f})")
    c = ci_lo > 0
    checks.append(f"CI > 0: {'PASS' if c else 'FAIL'} ([{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%])")
    
    for chk in checks:
        print(f"    {chk}", flush=True)
    score = sum(1 for chk in checks if 'PASS' in chk)
    print(f"    Score: {score}/6", flush=True)
    if score >= 5:
        print(f"    RESULT: PASS ✅", flush=True)
    elif score >= 3:
        print(f"    RESULT: PROVISIONAL ⚠️", flush=True)
    else:
        print(f"    RESULT: FAIL ❌", flush=True)

# Cleanup
subprocess.run(['rm', '-f', TAIL_FILE])

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
print("="*70, flush=True)
print("8-AGENT PROTOCOL COMPLETE", flush=True)
print("="*70, flush=True)
