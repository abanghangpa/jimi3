"""
L2 Short Horizon Hypothesis Test
Does orderbook data predict 1-4h returns?
"""
import pandas as pd
import numpy as np
from scipy import stats
import subprocess, time

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
OB_FILE = f'{DATA_DIR}/ob_history/ob_historical.csv'
TAIL_FILE = '/tmp/ob_tail.csv'

print("L2 Short Horizon Test", flush=True)
t0 = time.time()

# ═══════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════
print("="*70, flush=True)
print("LOADING DATA", flush=True)
print("="*70, flush=True)

result = subprocess.run(['head', '-1', OB_FILE], capture_output=True, text=True)
header = result.stdout.strip()
subprocess.run(['bash', '-c', f'(echo "{header}"; tail -2000000 {OB_FILE}) > {TAIL_FILE}'], check=True)

ob = pd.read_csv(TAIL_FILE)
ob['timestamp'] = pd.to_datetime(ob['timestamp'], utc=True).dt.tz_localize(None)
ob = ob.sort_values('timestamp').reset_index(drop=True)
ob = ob.iloc[::10].reset_index(drop=True)  # sample 1/10
print(f"OB: {len(ob):,} rows, {ob['timestamp'].min()} to {ob['timestamp'].max()}", flush=True)

# Use merged CSV (current to Jul 25)
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_merged.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)

# Short-horizon forward returns: 15m, 30m, 1h, 2h, 4h (in 15m bars: 1, 2, 4, 8, 16)
for h in [1, 2, 4, 8, 16]:
    ohlcv[f'fwd_ret_{h}'] = ohlcv['Close'].shift(-h) / ohlcv['Close'] - 1

# Merge
merged = pd.merge_asof(
    ob,
    ohlcv[['timestamp', 'Close', 'fwd_ret_1', 'fwd_ret_2', 'fwd_ret_4', 'fwd_ret_8', 'fwd_ret_16', 'Volume']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('15min')
)
merged = merged.dropna(subset=['Close', 'fwd_ret_1'])
print(f"Merged: {len(merged):,} rows", flush=True)
print(f"Time: {time.time()-t0:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("FEATURE ENGINEERING", flush=True)
print("="*70, flush=True)

merged['bid_ask_imbalance'] = (merged['bid_total'] - merged['ask_total']) / (merged['bid_total'] + merged['ask_total'])
merged['top5_imbalance'] = (merged['top5_bid_vol'] - merged['top5_ask_vol']) / (merged['top5_bid_vol'] + merged['top5_ask_vol'])
merged['spread_bps'] = merged['spread_pct'] * 10000

for w in [3, 6, 12]:
    merged[f'ob_ratio_ma_{w}'] = merged['ob_ratio'].rolling(w).mean()
    merged[f'ob_ratio_std_{w}'] = merged['ob_ratio'].rolling(w).std()

merged['ob_ratio_delta_3'] = merged['ob_ratio'] - merged['ob_ratio'].shift(3)
merged['ob_ratio_delta_6'] = merged['ob_ratio'] - merged['ob_ratio'].shift(6)
merged['top5_delta_3'] = merged['top5_ratio'] - merged['top5_ratio'].shift(3)

merged['max_bid_pct'] = merged['max_bid_vol'] / (merged['bid_total'] + 0.01)
merged['max_ask_pct'] = merged['max_ask_vol'] / (merged['ask_total'] + 0.01)
merged['large_order_skew'] = merged['max_bid_pct'] - merged['max_ask_pct']

# Volume ratio
vr = merged['Volume'] / merged['Volume'].rolling(20).mean()
merged['vol_ratio'] = vr

# EMA200 for trend
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')

feature_cols = ['ob_ratio', 'top5_ratio', 'bid_ask_imbalance', 'top5_imbalance',
                'ob_ratio_ma_3', 'ob_ratio_ma_6', 'ob_ratio_ma_12',
                'ob_ratio_std_3', 'ob_ratio_std_6', 'ob_ratio_std_12',
                'ob_ratio_delta_3', 'ob_ratio_delta_6', 'top5_delta_3',
                'large_order_skew']

# ═══════════════════════════════════════════════════════
# CORRELATIONS BY HORIZON
# ═══════════════════════════════════════════════════════
clean = merged.dropna(subset=['fwd_ret_1']).copy()

print("\n" + "="*70, flush=True)
print("CORRELATIONS BY HORIZON (15m to 4h)", flush=True)
print("="*70, flush=True)

horizon_labels = {1: '15m', 2: '30m', 4: '1h', 8: '2h', 16: '4h'}

results = []
for f in feature_cols:
    if f not in clean.columns:
        continue
    sub = clean.dropna(subset=[f])
    if len(sub) < 200:
        continue
    x = sub[f].values
    for h in [1, 2, 4, 8, 16]:
        y = sub[f'fwd_ret_{h}'].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 200:
            continue
        r, p = stats.pearsonr(x[mask], y[mask])
        results.append({'feature': f, 'horizon': h, 'label': horizon_labels[h], 'r': r, 'p': p, 'n': int(mask.sum())})

corr_df = pd.DataFrame(results)

# Show for each feature, all horizons
print(f"\n{'Feature':<30s} {'15m':>12s} {'30m':>12s} {'1h':>12s} {'2h':>12s} {'4h':>12s}", flush=True)
print("-"*90, flush=True)
for f in feature_cols:
    if f not in corr_df['feature'].values:
        continue
    row = f"{f:<30s}"
    for h in [1, 2, 4, 8, 16]:
        sub = corr_df[(corr_df['feature'] == f) & (corr_df['horizon'] == h)]
        if len(sub) == 0:
            row += f" {'N/A':>12s}"
        else:
            r = sub.iloc[0]['r']
            p = sub.iloc[0]['p']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f" {r:+.5f}{sig:>3s}"
    print(row, flush=True)

# ═══════════════════════════════════════════════════════
# EXTREME SIGNALS — SHORT HORIZON
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("EXTREME OB SIGNALS — SHORT HORIZON", flush=True)
print("="*70, flush=True)

for h in [1, 2, 4, 8]:
    label = horizon_labels[h]
    print(f"\n--- {label} forward returns ---", flush=True)
    
    for threshold in [0.20, 0.30, 0.40]:
        long_sig = clean[clean['ob_ratio'] > threshold]
        short_sig = clean[clean['ob_ratio'] < -threshold]
        
        if len(long_sig) > 20:
            rets = long_sig[f'fwd_ret_{h}'].dropna()
            wr = (rets > 0).mean()
            t, p = stats.ttest_1samp(rets, 0)
            p1 = p/2 if t > 0 else 1-p/2
            print(f"  LONG(ob>{threshold}): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(rets):,}", flush=True)
        
        if len(short_sig) > 20:
            rets = -short_sig[f'fwd_ret_{h}'].dropna()
            wr = (rets > 0).mean()
            t, p = stats.ttest_1samp(rets, 0)
            p1 = p/2 if t > 0 else 1-p/2
            print(f"  SHORT(ob<{threshold}): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(rets):,}", flush=True)

# ═══════════════════════════════════════════════════════
# TOP5 RATIO EXTREMES — SHORT HORIZON
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("TOP5 RATIO EXTREMES — SHORT HORIZON", flush=True)
print("="*70, flush=True)

for h in [1, 2, 4, 8]:
    label = horizon_labels[h]
    print(f"\n--- {label} forward returns ---", flush=True)
    
    for threshold in [0.30, 0.40, 0.50]:
        long_sig = clean[clean['top5_ratio'] > threshold]
        short_sig = clean[clean['top5_ratio'] < -threshold]
        
        if len(long_sig) > 20:
            rets = long_sig[f'fwd_ret_{h}'].dropna()
            wr = (rets > 0).mean()
            t, p = stats.ttest_1samp(rets, 0)
            p1 = p/2 if t > 0 else 1-p/2
            print(f"  LONG(top5>{threshold}): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(rets):,}", flush=True)
        
        if len(short_sig) > 20:
            rets = -short_sig[f'fwd_ret_{h}'].dropna()
            wr = (rets > 0).mean()
            t, p = stats.ttest_1samp(rets, 0)
            p1 = p/2 if t > 0 else 1-p/2
            print(f"  SHORT(top5<{threshold}): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(rets):,}", flush=True)

# ═══════════════════════════════════════════════════════
# OB DELTA (MOMENTUM) — SHORT HORIZON
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("OB DELTA (MOMENTUM) — SHORT HORIZON", flush=True)
print("="*70, flush=True)

for h in [1, 2, 4, 8]:
    label = horizon_labels[h]
    print(f"\n--- {label} forward returns ---", flush=True)
    
    for delta_col in ['ob_ratio_delta_3', 'ob_ratio_delta_6', 'top5_delta_3']:
        sub = clean.dropna(subset=[delta_col])
        if len(sub) < 100:
            continue
        
        # Test extreme deltas
        for q_label, q_fn in [
            ('top10%', lambda x: x > x.quantile(0.9)),
            ('bot10%', lambda x: x < x.quantile(0.1)),
        ]:
            mask = q_fn(sub[delta_col])
            sig = sub[mask]
            if len(sig) < 20:
                continue
            
            if 'top' in q_label:
                rets = sig[f'fwd_ret_{h}'].dropna()
            else:
                rets = -sig[f'fwd_ret_{h}'].dropna()
            
            wr = (rets > 0).mean()
            t, p = stats.ttest_1samp(rets, 0)
            p1 = p/2 if t > 0 else 1-p/2
            print(f"  {delta_col} {q_label}: WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(rets):,}", flush=True)

# ═══════════════════════════════════════════════════════
# REGIME x OB x HORIZON
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("REGIME x OB x HORIZON", flush=True)
print("="*70, flush=True)

for h in [1, 2, 4, 8]:
    label = horizon_labels[h]
    print(f"\n--- {label} forward returns ---", flush=True)
    
    for trend in ['BULL', 'BEAR']:
        for ob_thresh in [0.15, 0.25]:
            long_sig = clean[(clean['trend'] == trend) & (clean['ob_ratio'] > ob_thresh)]
            short_sig = clean[(clean['trend'] == trend) & (clean['ob_ratio'] < -ob_thresh)]
            
            if len(long_sig) > 10:
                rets = long_sig[f'fwd_ret_{h}'].dropna()
                wr = (rets > 0).mean()
                t, p = stats.ttest_1samp(rets, 0)
                p1 = p/2 if t > 0 else 1-p/2
                print(f"  {trend}+LONG(ob>{ob_thresh}): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(rets):,}", flush=True)
            
            if len(short_sig) > 10:
                rets = -short_sig[f'fwd_ret_{h}'].dropna()
                wr = (rets > 0).mean()
                t, p = stats.ttest_1samp(rets, 0)
                p1 = p/2 if t > 0 else 1-p/2
                print(f"  {trend}+SHORT(ob<{ob_thresh}): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(rets):,}", flush=True)

# ═══════════════════════════════════════════════════════
# SPOOF DETECTION — rapid OB ratio change + reversal
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("SPOOF DETECTION — rapid OB shift then reversal", flush=True)
print("="*70, flush=True)

# Detect: ob_ratio jumps >0.3 in 3 bars, then reverses in next 3
clean['ob_jump'] = clean['ob_ratio_delta_3'].abs()
clean['ob_reversal'] = clean['ob_ratio_delta_3'] * clean['ob_ratio_delta_3'].shift(3)

# Big jump that reverses (spike then mean-revert)
spike_threshold = 0.5
spikes = clean[clean['ob_jump'] > spike_threshold]
print(f"\nOB spikes (|delta_3| > {spike_threshold}): {len(spikes):,}", flush=True)

for h in [1, 2, 4, 8]:
    label = horizon_labels[h]
    # After a positive spike (bid wall appears), does price reverse?
    pos_spikes = spikes[spikes['ob_ratio_delta_3'] > spike_threshold]
    neg_spikes = spikes[spikes['ob_ratio_delta_3'] < -spike_threshold]
    
    if len(pos_spikes) > 20:
        rets = pos_spikes[f'fwd_ret_{h}'].dropna()
        wr = (rets < 0).mean()  # expect reversal DOWN after bid wall appears
        t, p = stats.ttest_1samp(-rets, 0)
        p1 = p/2 if t > 0 else 1-p/2
        print(f"  {label} +spike→DOWN: WR={wr*100:.1f}% mean={(-rets).mean()*100:+.4f}% p={p1:.4f} n={len(rets):,}", flush=True)
    
    if len(neg_spikes) > 20:
        rets = neg_spikes[f'fwd_ret_{h}'].dropna()
        wr = (rets > 0).mean()  # expect reversal UP after ask wall appears
        t, p = stats.ttest_1samp(rets, 0)
        p1 = p/2 if t > 0 else 1-p/2
        print(f"  {label} -spike→UP: WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(rets):,}", flush=True)

# Cleanup
subprocess.run(['rm', '-f', TAIL_FILE])

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
print("="*70, flush=True)
print("SHORT HORIZON TEST COMPLETE", flush=True)
print("="*70, flush=True)
