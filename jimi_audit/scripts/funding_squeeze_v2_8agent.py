"""
8-Agent Protocol: Funding Squeeze v2
FR z-score extreme + OI rising + EU/US only + 8h TP

Agent 1: Forensics — signal quality
Agent 2: Non-indicator — raw FR z-score predictive power
Agent 3: Context filters (session, vol regime, OI)
Agent 4: Co-occurrence (skipped — needs live data)
Agent 5: Walk-forward
Agent 6: Monte Carlo
Agent 7: Regime-conditional
Agent 8: Statistical significance
"""
import pandas as pd
import numpy as np
from scipy import stats
import time

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'

print("="*70, flush=True)
print("8-AGENT PROTOCOL: FUNDING SQUEEZE v2", flush=True)
print("="*70, flush=True)
t0 = time.time()

# ═══════════════════════════════════════════════════════
# DATA LOAD
# ═══════════════════════════════════════════════════════
print("\nLoading data...", flush=True)

deriv = pd.read_csv(f'{DATA_DIR}/derivatives_history/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)
print(f"Deriv: {len(deriv):,} rows", flush=True)

ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_merged.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)

# Forward returns at multiple horizons
for h_bars, label in [(4, '4h'), (8, '8h'), (16, '16h'), (32, '32h')]:
    ohlcv[f'fwd_ret_{label}'] = ohlcv['Close'].shift(-h_bars) / ohlcv['Close'] - 1

merged = pd.merge_asof(
    deriv,
    ohlcv[['timestamp', 'Close', 'High', 'Low', 'Volume', 'fwd_ret_4h', 'fwd_ret_8h', 'fwd_ret_16h', 'fwd_ret_32h']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('15min')
)
merged = merged.dropna(subset=['Close', 'fwd_ret_8h'])
print(f"Merged: {len(merged):,} rows", flush=True)

# ═══════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════
# FR z-score (96-bar lookback)
fr = merged['funding_rate'].values
fr_series = pd.Series(fr)
fr_mean = fr_series.rolling(96).mean()
fr_std = fr_series.rolling(96).std()
merged['fr_zscore'] = (fr_series - fr_mean) / fr_std

# OI change
merged['oi_change'] = merged['oi'].pct_change(periods=4)

# Vol regime
merged['vol_20'] = merged['Close'].pct_change().rolling(20).std()
vol_p33 = merged['vol_20'].quantile(0.33)
vol_p67 = merged['vol_20'].quantile(0.67)
merged['vol_regime'] = 'MID'
merged.loc[merged['vol_20'] < vol_p33, 'vol_regime'] = 'LOW'
merged.loc[merged['vol_20'] > vol_p67, 'vol_regime'] = 'HIGH'

# Trend
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')

# Session
def get_session(ts):
    h = ts.hour
    if 0 <= h < 8: return 'ASIA'
    elif 8 <= h < 14: return 'EU'
    elif 14 <= h < 22: return 'US'
    else: return 'LATE'
merged['session'] = merged['timestamp'].apply(get_session)

# Cumulative FR check: z > 1.5 for 3+ consecutive bars
merged['fr_z_cum3'] = (
    (merged['fr_zscore'] > 1.5) &
    (merged['fr_zscore'].shift(1) > 1.5) &
    (merged['fr_zscore'].shift(2) > 1.5)
)

# Drop NaN
valid = merged.dropna(subset=['fr_zscore', 'fwd_ret_8h']).copy()
print(f"Valid: {len(valid):,} rows", flush=True)
print(f"Time: {time.time()-t0:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════
# SIGNAL GENERATION — v2 filters
# ═══════════════════════════════════════════════════════
print("\nGenerating v2 signals...", flush=True)

signals = valid[
    (valid['fr_zscore'] > 1.75) &           # Instantaneous gate
    (valid['fr_z_cum3'] == True) &           # Cumulative check
    (valid['session'].isin(['EU', 'US'])) &  # No ASIA
    (valid['vol_regime'].isin(['MID', 'HIGH']))  # No LOW vol
].copy()

signals['direction'] = 'SHORT'
signals['adj_ret_8h'] = -signals['fwd_ret_8h']  # SHORT = flip sign
signals['adj_ret_4h'] = -signals['fwd_ret_4h']
signals['adj_ret_16h'] = -signals['fwd_ret_16h']

# Add OI filter
signals_oi = signals[signals['oi_change'] > 0].copy()  # OI rising

print(f"v2 signals (all): {len(signals):,}", flush=True)
print(f"v2 signals (OI rising): {len(signals_oi):,}", flush=True)
print(f"Time: {time.time()-t0:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 1: FORENSICS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 1: FORENSICS", flush=True)
print("="*70, flush=True)

for label, sub in [('v2 (no OI)', signals), ('v2+OI', signals_oi)]:
    if len(sub) < 10:
        print(f"\n  {label}: {len(sub)} signals (too few)", flush=True)
        continue
    
    for horizon in ['4h', '8h', '16h']:
        rets = sub[f'adj_ret_{horizon}'].dropna()
        wr = (rets > 0).mean()
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        p1 = p/2 if t > 0 else 1-p/2
        print(f"  {label} {horizon}: WR={wr*100:.1f}% mean={mean_r*100:+.4f}% p={p1:.6f} n={len(rets):,}", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 2: NON-INDICATOR — raw FR z-score
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 2: NON-INDICATOR — raw FR z-score at 8h", flush=True)
print("="*70, flush=True)

for thresh in [1.25, 1.50, 1.75, 2.0, 2.5]:
    sub = valid[valid['fr_zscore'] > thresh]
    if len(sub) < 20:
        continue
    rets = -sub['fwd_ret_8h']  # SHORT
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    print(f"  z>{thresh:.2f}: WR={wr*100:.1f}% mean={mean_r*100:+.4f}% p={p1:.6f} n={len(sub):,}", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 3: CONTEXT FILTERS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 3: CONTEXT FILTERS", flush=True)
print("="*70, flush=True)

# Session
print(f"\n  By session (FR z > 1.75 SHORT 8h):", flush=True)
for session in ['ASIA', 'EU', 'US']:
    sub = valid[(valid['fr_zscore'] > 1.75) & (valid['session'] == session)]
    if len(sub) < 10:
        continue
    rets = -sub['fwd_ret_8h']
    wr = (rets > 0).mean()
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    print(f"    {session}: WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.6f} n={len(sub):,}", flush=True)

# Vol regime
print(f"\n  By vol regime (FR z > 1.75 SHORT 8h):", flush=True)
for vol in ['LOW', 'MID', 'HIGH']:
    sub = valid[(valid['fr_zscore'] > 1.75) & (valid['vol_regime'] == vol)]
    if len(sub) < 10:
        continue
    rets = -sub['fwd_ret_8h']
    wr = (rets > 0).mean()
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    print(f"    {vol}: WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.6f} n={len(sub):,}", flush=True)

# OI change
print(f"\n  By OI change (FR z > 1.75 SHORT 8h):", flush=True)
for oi_label, oi_fn in [('OI rising (>0%)', lambda x: x > 0), ('OI falling (<0%)', lambda x: x < 0)]:
    sub = valid[(valid['fr_zscore'] > 1.75) & oi_fn(valid['oi_change'])]
    if len(sub) < 10:
        continue
    rets = -sub['fwd_ret_8h']
    wr = (rets > 0).mean()
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    print(f"    {oi_label}: WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.6f} n={len(sub):,}", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 4: CO-OCCURRENCE (skipped)
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

signals_oi['week'] = signals_oi['timestamp'].dt.isocalendar().week.astype(int)
signals_oi['year'] = signals_oi['timestamp'].dt.isocalendar().year.astype(int)
signals_oi['yw'] = signals_oi['year'] * 100 + signals_oi['week']

wf_results = []
for yw in sorted(signals_oi['yw'].unique()):
    sub = signals_oi[signals_oi['yw'] == yw]
    rets = sub['adj_ret_8h'].dropna()
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
    
    print(f"\n  All weeks:", flush=True)
    for _, row in wf_df.iterrows():
        tag = "WIN" if row['wr'] > 0.5 else "LOSS"
        print(f"    [{tag}] {int(row['yw'])}: WR={row['wr']*100:.0f}% mean={row['mean']*100:+.3f}% n={int(row['n'])}", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 6: MONTE CARLO
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 6: MONTE CARLO", flush=True)
print("="*70, flush=True)

rets = signals_oi['adj_ret_8h'].dropna().values
if len(rets) > 20:
    n_sims = 5000
    for horizon in [20, 30, 50]:
        sims = []
        for _ in range(n_sims):
            sampled = np.random.choice(rets, size=horizon, replace=True)
            sims.append(sampled.sum())
        sims = np.array(sims)
        p5, p25, p50, p75, p95 = np.percentile(sims, [5, 25, 50, 75, 95])
        prob_loss = (sims < 0).mean()
        print(f"\n  {horizon}-trade horizon:", flush=True)
        print(f"    P5={p5*100:+.2f}% P25={p25*100:+.2f}% P50={p50*100:+.2f}% P75={p75*100:+.2f}% P95={p95*100:+.2f}%", flush=True)
        print(f"    Prob(loss)={prob_loss*100:.1f}% Expected PnL: {p50*100:+.2f}%", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 7: REGIME-CONDITIONAL
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 7: REGIME-CONDITIONAL", flush=True)
print("="*70, flush=True)

for trend in ['BULL', 'BEAR']:
    sub = signals_oi[signals_oi['trend'] == trend]
    if len(sub) < 5:
        continue
    rets = sub['adj_ret_8h'].dropna()
    wr = (rets > 0).mean()
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    print(f"  {trend}: WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.6f} n={len(rets):,}", flush=True)

# ═══════════════════════════════════════════════════════
# AGENT 8: STATISTICAL SIGNIFICANCE
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("AGENT 8: STATISTICAL SIGNIFICANCE", flush=True)
print("="*70, flush=True)

rets = signals_oi['adj_ret_8h'].dropna().values
n_trades = len(rets)
wr = (rets > 0).mean()
mean_r = rets.mean()
t, p = stats.ttest_1samp(rets, 0)
p1 = p/2 if t > 0 else 1-p/2

boots = [np.random.choice(rets, size=n_trades, replace=True).mean() for _ in range(2000)]
ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
wins = rets[rets > 0]
losses = rets[rets < 0]
pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')

print(f"\n  t-test: t={t:.3f} p(one-sided)={p1:.6f}", flush=True)
print(f"  Bootstrap CI: [{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%]", flush=True)
print(f"  WR={wr*100:.1f}% PF={pf:.2f}", flush=True)
print(f"  Mean win: {wins.mean()*100:+.4f}% Mean loss: {losses.mean()*100:+.4f}%", flush=True)

# Gate checks
print(f"\n  GATE DECISION:", flush=True)
checks = []
c = n_trades >= 30
checks.append(f"n >= 30: {'PASS' if c else 'FAIL'} ({n_trades:,})")
c = wr >= 0.55
checks.append(f"WR >= 55%: {'PASS' if c else 'FAIL'} ({wr*100:.1f}%)")
c = mean_r > 0
checks.append(f"Mean > 0: {'PASS' if c else 'FAIL'} ({mean_r*100:+.4f}%)")
c = p1 < 0.05
checks.append(f"p < 0.05: {'PASS' if c else 'FAIL'} ({p1:.6f})")
c = pf >= 1.5
checks.append(f"PF >= 1.5: {'PASS' if c else 'FAIL'} ({pf:.2f})")
c = ci_lo > 0
checks.append(f"CI > 0: {'PASS' if c else 'FAIL'} ([{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%])")

# Walk-forward check
if len(wf_df) > 0:
    wf_win_rate = (wf_df['wr'] > 0.5).mean()
    c = wf_win_rate >= 0.5
    checks.append(f"WF win weeks >= 50%: {'PASS' if c else 'FAIL'} ({wf_win_rate*100:.0f}%)")

for chk in checks:
    print(f"    {chk}", flush=True)
score = sum(1 for chk in checks if 'PASS' in chk)
total_checks = len(checks)
print(f"    Score: {score}/{total_checks}", flush=True)
if score >= total_checks - 1:
    print(f"    RESULT: PASS ✅", flush=True)
elif score >= total_checks - 2:
    print(f"    RESULT: PROVISIONAL ⚠️", flush=True)
else:
    print(f"    RESULT: FAIL ❌", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
print("="*70, flush=True)
print("8-AGENT PROTOCOL COMPLETE", flush=True)
print("="*70, flush=True)
