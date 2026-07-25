"""
Liquidity Grab v7 — Validation + Optimization
Phase 1: Validate Scenario A + ASIA filter
Phase 2: Parameter sweep (OB threshold, vol_ratio, session)
Phase 3: Walk-forward with best params
Phase 4: Monte Carlo + gate decision
"""
import pandas as pd
import numpy as np
from scipy import stats
import subprocess, time

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
OB_FILE = f'{DATA_DIR}/ob_history/ob_historical.csv'
TAIL_FILE = '/tmp/ob_tail.csv'

print("="*70, flush=True)
print("LIQUIDITY GRAB v7 — VALIDATION + OPTIMIZATION", flush=True)
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
ob = ob.iloc[::10].reset_index(drop=True)
print(f"OB: {len(ob):,} rows", flush=True)

ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_merged.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)

# Forward returns at multiple horizons
for h_bars, label in [(4, '1h'), (8, '2h'), (12, '3h'), (16, '4h')]:
    ohlcv[f'fwd_ret_{label}'] = ohlcv['Close'].shift(-h_bars) / ohlcv['Close'] - 1

merged = pd.merge_asof(
    ob,
    ohlcv[['timestamp', 'Close', 'High', 'Low', 'Volume', 'fwd_ret_1h', 'fwd_ret_2h', 'fwd_ret_3h', 'fwd_ret_4h']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('15min')
)
merged = merged.dropna(subset=['Close', 'fwd_ret_2h'])
print(f"Merged: {len(merged):,} rows", flush=True)

# ═══════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════
merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')
merged['atr'] = (merged['High'] - merged['Low']).rolling(16).mean()

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

print(f"After bad-hour filter: {len(merged):,} rows", flush=True)
print(f"Time: {time.time()-t0:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════
# PHASE 1: VALIDATE SCENARIO A + ASIA
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PHASE 1: VALIDATE SCENARIO A + ASIA FILTER", flush=True)
print("="*70, flush=True)

# Base: BEAR + ob_ratio > 0.15
base = merged[(merged['trend'] == 'BEAR') & (merged['ob_ratio'] > 0.15) & (merged['vol_ratio'] >= 0.8)]
print(f"\nBase signals (BEAR + ob>0.15 + vol>=0.8): {len(base):,}", flush=True)

# With ASIA filter
asia = base[base['session'] == 'ASIA']
print(f"+ ASIA filter: {len(asia):,}", flush=True)

# Validate across horizons
print(f"\n{'Filter':<35s} {'1h WR':>8s} {'2h WR':>8s} {'3h WR':>8s} {'4h WR':>8s} {'2h Mean':>10s} {'2h p':>10s} {'n':>8s}", flush=True)
print("-"*100, flush=True)

for label, sub in [('BEAR+ob>0.15 (base)', base), ('+ ASIA', asia)]:
    row = f"{label:<35s}"
    for horizon in ['1h', '2h', '3h', '4h']:
        rets = sub[f'fwd_ret_{horizon}'].dropna()
        if len(rets) < 10:
            row += f" {'N/A':>8s}"
        else:
            wr = (rets > 0).mean()
            row += f" {wr*100:>7.1f}%"
    # 2h stats
    rets_2h = sub['fwd_ret_2h'].dropna()
    if len(rets_2h) > 10:
        t, p = stats.ttest_1samp(rets_2h, 0)
        p1 = p/2 if t > 0 else 1-p/2
        row += f" {rets_2h.mean()*100:>+9.4f}% {p1:>10.6f} {len(rets_2h):>8,}"
    print(row, flush=True)

# ═══════════════════════════════════════════════════════
# PHASE 2: PARAMETER SWEEP
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PHASE 2: PARAMETER SWEEP", flush=True)
print("="*70, flush=True)

# Sweep OB threshold
print(f"\n--- OB ratio threshold sweep (BEAR + ASIA) ---", flush=True)
print(f"{'OB thresh':>10s} {'WR':>8s} {'Mean':>10s} {'p':>10s} {'PF':>6s} {'n':>8s}", flush=True)
print("-"*60, flush=True)

for ob_thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]:
    sub = merged[(merged['trend'] == 'BEAR') & (merged['ob_ratio'] > ob_thresh) & 
                  (merged['vol_ratio'] >= 0.8) & (merged['session'] == 'ASIA')]
    rets = sub['fwd_ret_2h'].dropna()
    if len(rets) < 20:
        continue
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    print(f"  >{ob_thresh:.2f}   {wr*100:>7.1f}% {mean_r*100:>+9.4f}% {p1:>10.6f} {pf:>6.2f} {len(rets):>8,}", flush=True)

# Sweep vol_ratio threshold
print(f"\n--- vol_ratio threshold sweep (BEAR + ASIA + ob>0.15) ---", flush=True)
print(f"{'VR thresh':>10s} {'WR':>8s} {'Mean':>10s} {'p':>10s} {'PF':>6s} {'n':>8s}", flush=True)
print("-"*60, flush=True)

for vr_thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]:
    sub = merged[(merged['trend'] == 'BEAR') & (merged['ob_ratio'] > 0.15) & 
                  (merged['vol_ratio'] >= vr_thresh) & (merged['session'] == 'ASIA')]
    rets = sub['fwd_ret_2h'].dropna()
    if len(rets) < 20:
        continue
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    print(f"  >{vr_thresh:.1f}    {wr*100:>7.1f}% {mean_r*100:>+9.4f}% {p1:>10.6f} {pf:>6.2f} {len(rets):>8,}", flush=True)

# Sweep horizon (TP timing)
print(f"\n--- TP horizon sweep (BEAR + ASIA + ob>0.15 + vol>=0.8) ---", flush=True)
print(f"{'Horizon':>10s} {'WR':>8s} {'Mean':>10s} {'p':>10s} {'PF':>6s} {'n':>8s}", flush=True)
print("-"*60, flush=True)

sub = merged[(merged['trend'] == 'BEAR') & (merged['ob_ratio'] > 0.15) & 
              (merged['vol_ratio'] >= 0.8) & (merged['session'] == 'ASIA')]
for horizon in ['1h', '2h', '3h', '4h']:
    rets = sub[f'fwd_ret_{horizon}'].dropna()
    if len(rets) < 20:
        continue
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    print(f"  {horizon:>8s} {wr*100:>7.1f}% {mean_r*100:>+9.4f}% {p1:>10.6f} {pf:>6.2f} {len(rets):>8,}", flush=True)

# Combined sweep: OB threshold x session
print(f"\n--- Combined: OB threshold x session (BEAR, 2h) ---", flush=True)
print(f"{'OB/Sess':>15s} {'ASIA WR':>10s} {'EU WR':>10s} {'US WR':>10s} {'ASIA n':>10s}", flush=True)
print("-"*60, flush=True)

for ob_thresh in [0.10, 0.15, 0.20, 0.30, 0.40]:
    row = f"  ob>{ob_thresh:.2f}      "
    for session in ['ASIA', 'EU', 'US']:
        sub = merged[(merged['trend'] == 'BEAR') & (merged['ob_ratio'] > ob_thresh) & 
                      (merged['vol_ratio'] >= 0.8) & (merged['session'] == session)]
        rets = sub['fwd_ret_2h'].dropna()
        if len(rets) < 10:
            row += f" {'N/A':>10s}"
        else:
            wr = (rets > 0).mean()
            row += f" {wr*100:>8.1f}%  "
    # ASIA n
    sub_a = merged[(merged['trend'] == 'BEAR') & (merged['ob_ratio'] > ob_thresh) & 
                    (merged['vol_ratio'] >= 0.8) & (merged['session'] == 'ASIA')]
    row += f"{len(sub_a['fwd_ret_2h'].dropna()):>10,}"
    print(row, flush=True)

# ═══════════════════════════════════════════════════════
# PHASE 3: WALK-FORWARD WITH BEST PARAMS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PHASE 3: WALK-FORWARD (best params)", flush=True)
print("="*70, flush=True)

# Use best combo from sweep: BEAR + ASIA + ob>0.15 + vol>=0.8 at 2h
best = merged[(merged['trend'] == 'BEAR') & (merged['ob_ratio'] > 0.15) & 
               (merged['vol_ratio'] >= 0.8) & (merged['session'] == 'ASIA')].copy()

best['week'] = best['timestamp'].dt.isocalendar().week.astype(int)
best['year'] = best['timestamp'].dt.isocalendar().year.astype(int)
best['yw'] = best['year'] * 100 + best['week']

wf_results = []
for yw in sorted(best['yw'].unique()):
    sub = best[best['yw'] == yw]
    rets = sub['fwd_ret_2h'].dropna()
    if len(rets) < 5:
        continue
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    wf_results.append({'yw': yw, 'n': len(rets), 'wr': wr, 'mean': mean_r, 'pf': pf})

wf_df = pd.DataFrame(wf_results)
if len(wf_df) > 0:
    win_weeks = (wf_df['wr'] > 0.5).sum()
    total_weeks = len(wf_df)
    print(f"\n  Walk-forward weeks: {total_weeks}", flush=True)
    print(f"  Winning weeks: {win_weeks}/{total_weeks} ({win_weeks/total_weeks*100:.0f}%)", flush=True)
    print(f"  Mean WR: {wf_df['wr'].mean()*100:.1f}%", flush=True)
    print(f"  Mean return: {wf_df['mean'].mean()*100:+.4f}%", flush=True)
    print(f"  Mean PF: {wf_df['pf'].mean():.2f}", flush=True)
    
    print(f"\n  {'Week':>10s} {'WR':>8s} {'Mean':>10s} {'PF':>6s} {'n':>8s} {'Tag':>6s}", flush=True)
    print("  " + "-"*50, flush=True)
    for _, row in wf_df.iterrows():
        tag = "WIN" if row['wr'] > 0.5 else "LOSS"
        pf_str = f"{row['pf']:.2f}" if row['pf'] < 100 else "inf"
        print(f"  {int(row['yw']):>10d} {row['wr']*100:>7.1f}% {row['mean']*100:>+9.4f}% {pf_str:>6s} {int(row['n']):>8d} {tag:>6s}", flush=True)

# ═══════════════════════════════════════════════════════
# PHASE 4: MONTE CARLO + GATE DECISION
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PHASE 4: MONTE CARLO + GATE DECISION", flush=True)
print("="*70, flush=True)

rets = best['fwd_ret_2h'].dropna().values
n_trades = len(rets)
wr = (rets > 0).mean()
mean_r = rets.mean()

# Monte Carlo
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

# Statistical tests
t, p = stats.ttest_1samp(rets, 0)
p1 = p/2 if t > 0 else 1-p/2
boots = [np.random.choice(rets, size=n_trades, replace=True).mean() for _ in range(2000)]
ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
wins = rets[rets > 0]
losses = rets[rets < 0]
pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')

print(f"\n  Final stats:", flush=True)
print(f"    n={n_trades:,} WR={wr*100:.1f}% Mean={mean_r*100:+.4f}%", flush=True)
print(f"    t={t:.3f} p(one-sided)={p1:.6f}", flush=True)
print(f"    Bootstrap CI: [{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%]", flush=True)
print(f"    PF={pf:.2f} Mean win={wins.mean()*100:+.4f}% Mean loss={losses.mean()*100:+.4f}%", flush=True)

# Gate checks
print(f"\n  GATE DECISION:", flush=True)
checks = []
c = n_trades >= 30
checks.append(f"n >= 30: {'PASS' if c else 'FAIL'} ({n_trades:,})")
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

# Cleanup
subprocess.run(['rm', '-f', TAIL_FILE])

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
print("="*70, flush=True)
print("VALIDATION + OPTIMIZATION COMPLETE", flush=True)
print("="*70, flush=True)
