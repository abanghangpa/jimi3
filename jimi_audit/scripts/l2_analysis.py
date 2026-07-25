"""
L2 Orderbook Analysis — v3 (use tail to extract, then analyze)
"""
import pandas as pd
import numpy as np
from scipy import stats
import subprocess, sys, time

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
OB_FILE = f'{DATA_DIR}/ob_history/ob_historical.csv'
TAIL_FILE = '/tmp/ob_tail.csv'

print("Starting L2 analysis v3...", flush=True)
t0 = time.time()

# ═══════════════════════════════════════════════════════
# PART 1: Extract last 2M rows via tail, then read
# ═══════════════════════════════════════════════════════
print("="*70, flush=True)
print("PART 1: EXTRACT + LOAD ORDERBOOK DATA", flush=True)
print("="*70, flush=True)

# Get header first
result = subprocess.run(['head', '-1', OB_FILE], capture_output=True, text=True)
header = result.stdout.strip()

# Extract last 2M rows (2,000,001 lines including header)
subprocess.run(['bash', '-c', f'(echo "{header}"; tail -2000000 {OB_FILE}) > {TAIL_FILE}'], check=True)
print(f"Extracted tail to {TAIL_FILE}", flush=True)

ob = pd.read_csv(TAIL_FILE)
print(f"Loaded: {len(ob):,} rows", flush=True)

ob['timestamp'] = pd.to_datetime(ob['timestamp'], utc=True).dt.tz_localize(None)
ob = ob.sort_values('timestamp').reset_index(drop=True)
print(f"Date range: {ob['timestamp'].min()} to {ob['timestamp'].max()}", flush=True)
print(f"Time: {time.time()-t0:.1f}s", flush=True)

# Sample every 10th row
ob = ob.iloc[::10].reset_index(drop=True)
print(f"After 1/10 sampling: {len(ob):,} rows", flush=True)

# ═══════════════════════════════════════════════════════
# PART 2: Merge with OHLCV
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PART 2: MERGING WITH OHLCV", flush=True)
print("="*70, flush=True)

ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_merged.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)

for h in [4, 8, 16, 24]:
    ohlcv[f'fwd_ret_{h}'] = ohlcv['Close'].shift(-h) / ohlcv['Close'] - 1

merged = pd.merge_asof(
    ob,
    ohlcv[['timestamp', 'Close', 'High', 'Low', 'fwd_ret_4', 'fwd_ret_8', 'fwd_ret_16', 'fwd_ret_24', 'Volume']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('30min')
)
merged = merged.dropna(subset=['Close', 'fwd_ret_16'])
print(f"Merged rows: {len(merged):,}", flush=True)
print(f"Time: {time.time()-t0:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════
# PART 3: Feature Engineering
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PART 3: OB FEATURE ENGINEERING", flush=True)
print("="*70, flush=True)

merged['bid_ask_imbalance'] = (merged['bid_total'] - merged['ask_total']) / (merged['bid_total'] + merged['ask_total'])
merged['top5_imbalance'] = (merged['top5_bid_vol'] - merged['top5_ask_vol']) / (merged['top5_bid_vol'] + merged['top5_ask_vol'])
merged['spread_bps'] = merged['spread_pct'] * 10000

for w in [3, 6, 12]:
    merged[f'ob_ratio_ma_{w}'] = merged['ob_ratio'].rolling(w).mean()
    merged[f'ob_ratio_std_{w}'] = merged['ob_ratio'].rolling(w).std()
    merged[f'top5_ratio_ma_{w}'] = merged['top5_ratio'].rolling(w).mean()

merged['ob_ratio_delta_3'] = merged['ob_ratio'] - merged['ob_ratio'].shift(3)
merged['ob_ratio_delta_6'] = merged['ob_ratio'] - merged['ob_ratio'].shift(6)
merged['top5_delta_3'] = merged['top5_ratio'] - merged['top5_ratio'].shift(3)

merged['max_bid_pct'] = merged['max_bid_vol'] / (merged['bid_total'] + 0.01)
merged['max_ask_pct'] = merged['max_ask_vol'] / (merged['ask_total'] + 0.01)
merged['large_order_skew'] = merged['max_bid_pct'] - merged['max_ask_pct']
merged['level_ratio'] = merged['bid_levels'] / (merged['ask_levels'] + 1)

# Volume ratio
vr = merged['Volume'] / merged['Volume'].rolling(20).mean()
merged['vol_ratio'] = vr

feature_cols = ['ob_ratio', 'top5_ratio', 'bid_ask_imbalance', 'top5_imbalance',
                'spread_bps', 'ob_ratio_ma_3', 'ob_ratio_ma_6', 'ob_ratio_ma_12',
                'ob_ratio_std_3', 'ob_ratio_std_6', 'ob_ratio_std_12',
                'top5_ratio_ma_3', 'top5_ratio_ma_6', 'top5_ratio_ma_12',
                'ob_ratio_delta_3', 'ob_ratio_delta_6', 'top5_delta_3',
                'max_bid_pct', 'max_ask_pct', 'large_order_skew', 'level_ratio']
print(f"Features: {len(feature_cols)}", flush=True)

# ═══════════════════════════════════════════════════════
# PART 4: CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PART 4: FEATURE ↔ FORWARD RETURN CORRELATIONS", flush=True)
print("="*70, flush=True)

clean = merged.dropna(subset=['fwd_ret_16']).copy()
print(f"Clean rows: {len(clean):,}", flush=True)

results = []
for f in feature_cols:
    if f not in clean.columns:
        continue
    sub = clean.dropna(subset=[f])
    if len(sub) < 100:
        continue
    x = sub[f].values
    for horizon in [4, 8, 16, 24]:
        y = sub[f'fwd_ret_{horizon}'].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 100:
            continue
        r, p = stats.pearsonr(x[mask], y[mask])
        results.append({'feature': f, 'horizon': horizon, 'r': r, 'p': p, 'n': int(mask.sum())})

corr_df = pd.DataFrame(results)
sig = corr_df[corr_df['p'] < 0.01].sort_values('r', key=abs, ascending=False)
print(f"\nSignificant correlations (p < 0.01): {len(sig)}", flush=True)
for _, row in sig.head(25).iterrows():
    print(f"  {row['feature']:30s} {row['horizon']}h: r={row['r']:+.6f} p={row['p']:.2e} n={row['n']:,}", flush=True)

# ═══════════════════════════════════════════════════════
# PART 5: QUANTILE ANALYSIS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PART 5: QUANTILE ANALYSIS — OB extremes vs returns", flush=True)
print("="*70, flush=True)

key_features = ['ob_ratio', 'top5_ratio', 'bid_ask_imbalance', 'ob_ratio_delta_6', 'large_order_skew']
for f in key_features:
    sub = clean.dropna(subset=[f]).copy()
    if len(sub) < 100:
        continue
    try:
        sub[f'{f}_q'] = pd.qcut(sub[f], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
    except:
        continue
    print(f"\n--- {f} quintiles vs 16h return ---", flush=True)
    for q in ['Q1','Q2','Q3','Q4','Q5']:
        qsub = sub[sub[f'{f}_q'] == q]['fwd_ret_16'].dropna()
        if len(qsub) < 10:
            continue
        wr = (qsub > 0).mean()
        t, p = stats.ttest_1samp(qsub, 0)
        p1 = p/2 if t > 0 else 1-p/2
        lo = sub[sub[f'{f}_q']==q][f].min()
        hi = sub[sub[f'{f}_q']==q][f].max()
        print(f"  {q}: WR={wr*100:.1f}% mean={qsub.mean()*100:+.4f}% p={p1:.4f} n={len(qsub):,} [{lo:.3f},{hi:.3f}]", flush=True)

# ═══════════════════════════════════════════════════════
# PART 6: EXTREME OB SIGNALS — directional
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PART 6: EXTREME OB SIGNALS — directional trades", flush=True)
print("="*70, flush=True)

print("\n--- ob_ratio extremes ---", flush=True)
for threshold in [0.10, 0.15, 0.20, 0.30, 0.40]:
    long_sig = clean[clean['ob_ratio'] > threshold]
    short_sig = clean[clean['ob_ratio'] < -threshold]
    
    if len(long_sig) > 20:
        rets = long_sig['fwd_ret_16']
        wr = (rets > 0).mean()
        t, p = stats.ttest_1samp(rets, 0)
        p1 = p/2 if t > 0 else 1-p/2
        print(f"  LONG (ob>{threshold:.2f}): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(long_sig):,}", flush=True)
    
    if len(short_sig) > 20:
        rets = short_sig['fwd_ret_16']
        wr = (rets < 0).mean()
        adj = -rets
        t, p = stats.ttest_1samp(adj, 0)
        p1 = p/2 if t > 0 else 1-p/2
        print(f"  SHORT(ob<{threshold:.2f}): WR={wr*100:.1f}% mean={adj.mean()*100:+.4f}% p={p1:.4f} n={len(short_sig):,}", flush=True)

print("\n--- top5_ratio extremes ---", flush=True)
for threshold in [0.15, 0.20, 0.30, 0.40, 0.50]:
    long_sig = clean[clean['top5_ratio'] > threshold]
    short_sig = clean[clean['top5_ratio'] < -threshold]
    
    if len(long_sig) > 20:
        rets = long_sig['fwd_ret_16']
        wr = (rets > 0).mean()
        t, p = stats.ttest_1samp(rets, 0)
        p1 = p/2 if t > 0 else 1-p/2
        print(f"  LONG (top5>{threshold:.2f}): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(long_sig):,}", flush=True)
    
    if len(short_sig) > 20:
        rets = short_sig['fwd_ret_16']
        wr = (rets < 0).mean()
        adj = -rets
        t, p = stats.ttest_1samp(adj, 0)
        p1 = p/2 if t > 0 else 1-p/2
        print(f"  SHORT(top5<{threshold:.2f}): WR={wr*100:.1f}% mean={adj.mean()*100:+.4f}% p={p1:.4f} n={len(short_sig):,}", flush=True)

# ═══════════════════════════════════════════════════════
# PART 7: OB PERSISTENCE
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PART 7: OB PERSISTENCE — sustained imbalance", flush=True)
print("="*70, flush=True)

def calc_persistence(series, threshold=0.05):
    signs = (series > threshold).astype(int) - (series < -threshold).astype(int)
    pers = []
    count = 0
    last_sign = 0
    for s in signs:
        if s == last_sign and s != 0:
            count += 1
        else:
            count = 1 if s != 0 else 0
            last_sign = s
        pers.append(count)
    return pers

clean['persistence'] = calc_persistence(clean['ob_ratio'].values)

print(f"\nPersistence distribution (ob_ratio > 0.05):", flush=True)
for p_val in sorted(clean['persistence'].unique())[:20]:
    sub = clean[clean['persistence'] == p_val]['fwd_ret_16'].dropna()
    if len(sub) < 10:
        continue
    wr = (sub > 0).mean()
    t, p = stats.ttest_1samp(sub, 0)
    p1 = p/2 if t > 0 else 1-p/2
    print(f"  pers={p_val:2d}: WR={wr*100:.1f}% mean={sub.mean()*100:+.4f}% p={p1:.4f} n={len(sub):,}", flush=True)

# ═══════════════════════════════════════════════════════
# PART 8: LARGE ORDER SKEW
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PART 8: LARGE ORDER SKEW", flush=True)
print("="*70, flush=True)

los = clean['large_order_skew'].dropna()
if len(los) > 50:
    try:
        quintiles = pd.qcut(los, 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
        clean['los_q'] = quintiles
        for q in ['Q1','Q2','Q3','Q4','Q5']:
            sub = clean[clean['los_q'] == q]['fwd_ret_16'].dropna()
            if len(sub) < 10:
                continue
            wr = (sub > 0).mean()
            t, p = stats.ttest_1samp(sub, 0)
            p1 = p/2 if t > 0 else 1-p/2
            lo = clean[clean['los_q']==q]['large_order_skew'].min()
            hi = clean[clean['los_q']==q]['large_order_skew'].max()
            print(f"  {q}: WR={wr*100:.1f}% mean={sub.mean()*100:+.4f}% p={p1:.4f} n={len(sub):,} [{lo:.3f},{hi:.3f}]", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)

# ═══════════════════════════════════════════════════════
# PART 9: REGIME x OB INTERACTION
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PART 9: REGIME x OB INTERACTION", flush=True)
print("="*70, flush=True)

# EMA200 for trend
merged2 = pd.merge_asof(
    clean[['timestamp']].drop_duplicates().sort_values('timestamp'),
    ohlcv[['timestamp', 'Close']].drop_duplicates().sort_values('timestamp'),
    on='timestamp', direction='backward', tolerance=pd.Timedelta('30min')
)
merged2['ema200'] = merged2['Close'].ewm(span=200).mean()
merged2['trend'] = np.where(merged2['Close'] > merged2['ema200'], 'BULL', 'BEAR')

clean = pd.merge_asof(
    clean.sort_values('timestamp'),
    merged2[['timestamp', 'trend']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('30min')
)

for trend in ['BULL', 'BEAR']:
    for ob_thresh in [0.10, 0.15, 0.20]:
        long_sig = clean[(clean['trend'] == trend) & (clean['ob_ratio'] > ob_thresh)]
        short_sig = clean[(clean['trend'] == trend) & (clean['ob_ratio'] < -ob_thresh)]
        
        if len(long_sig) > 10:
            rets = long_sig['fwd_ret_16']
            wr = (rets > 0).mean()
            t, p = stats.ttest_1samp(rets, 0)
            p1 = p/2 if t > 0 else 1-p/2
            print(f"  {trend}+LONG(ob>{ob_thresh}): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(long_sig):,}", flush=True)
        
        if len(short_sig) > 10:
            rets = -short_sig['fwd_ret_16']
            wr = (rets > 0).mean()
            t, p = stats.ttest_1samp(rets, 0)
            p1 = p/2 if t > 0 else 1-p/2
            print(f"  {trend}+SHORT(ob<{ob_thresh}): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(short_sig):,}", flush=True)

# ═══════════════════════════════════════════════════════
# PART 10: COMBINED SIGNALS — OB + volume + spread
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PART 10: COMBINED SIGNALS — OB + volume + spread", flush=True)
print("="*70, flush=True)

for ob_thresh in [0.15, 0.20, 0.30]:
    for spread_thresh in [0.5, 1.0]:
        long_sig = clean[(clean['ob_ratio'] > ob_thresh) & (clean['spread_bps'] < spread_thresh) & (clean['vol_ratio'] > 1.2)]
        short_sig = clean[(clean['ob_ratio'] < -ob_thresh) & (clean['spread_bps'] < spread_thresh) & (clean['vol_ratio'] > 1.2)]
        
        if len(long_sig) > 10:
            rets = long_sig['fwd_ret_16']
            wr = (rets > 0).mean()
            t, p = stats.ttest_1samp(rets, 0)
            p1 = p/2 if t > 0 else 1-p/2
            print(f"  LONG(ob>{ob_thresh},sp<{spread_thresh},vol>1.2): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(long_sig):,}", flush=True)
        
        if len(short_sig) > 10:
            rets = -short_sig['fwd_ret_16']
            wr = (rets > 0).mean()
            t, p = stats.ttest_1samp(rets, 0)
            p1 = p/2 if t > 0 else 1-p/2
            print(f"  SHORT(ob<{ob_thresh},sp<{spread_thresh},vol>1.2): WR={wr*100:.1f}% mean={rets.mean()*100:+.4f}% p={p1:.4f} n={len(short_sig):,}", flush=True)

# Cleanup
subprocess.run(['rm', '-f', TAIL_FILE])

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
print("="*70, flush=True)
print("ANALYSIS COMPLETE", flush=True)
print("="*70, flush=True)
