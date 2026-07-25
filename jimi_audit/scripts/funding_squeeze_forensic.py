"""
Forensic Analysis: Funding Squeeze (S25)
Understanding why it's losing money despite p=0.003 significance
"""
import pandas as pd
import numpy as np
from scipy import stats
import json, time

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'

print("="*70, flush=True)
print("FORENSIC ANALYSIS: FUNDING SQUEEZE (S25)", flush=True)
print("="*70, flush=True)
t0 = time.time()

# ═══════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════
print("\nLoading data...", flush=True)

# Strategy signals
signals = []
with open(f'{DATA_DIR}/strategy_signals.jsonl') as f:
    for line in f:
        try:
            sig = json.loads(line)
            if sig.get('strategy') == 'funding_squeeze':
                signals.append(sig)
        except:
            pass

print(f"funding_squeeze signals: {len(signals)}", flush=True)

if signals:
    df = pd.DataFrame(signals)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}", flush=True)
    print(f"Fired signals: {len(df[df.get('fired', True)])}", flush=True)
    
    # Show first/last few
    print(f"\nFirst 3 signals:", flush=True)
    for _, s in df.head(3).iterrows():
        print(f"  {s.get('timestamp','-')} dir={s.get('direction','-')} fired={s.get('fired',True)} conv={s.get('conviction','-')} price={s.get('price',0):.2f}", flush=True)

# ═══════════════════════════════════════════════════════
# DERIVATIVES DATA
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("DERIVATIVES DATA ANALYSIS", flush=True)
print("="*70, flush=True)

import subprocess
result = subprocess.run(['head', '-1', f'{DATA_DIR}/derivatives_history/derivatives_collected.csv'], capture_output=True, text=True)
print(f"Deriv columns: {result.stdout.strip()[:200]}", flush=True)

# Count rows
result = subprocess.run(['wc', '-l', f'{DATA_DIR}/derivatives_history/derivatives_collected.csv'], capture_output=True, text=True)
print(f"Deriv rows: {result.stdout.strip()}", flush=True)

# Sample
result = subprocess.run(['tail', '-3', f'{DATA_DIR}/derivatives_history/derivatives_collected.csv'], capture_output=True, text=True)
print(f"Latest deriv:\n{result.stdout.strip()}", flush=True)

# ═══════════════════════════════════════════════════════
# LOAD DERIVATIVES + OHLCV
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("MERGING DERIVATIVES WITH OHLCV", flush=True)
print("="*70, flush=True)

deriv = pd.read_csv(f'{DATA_DIR}/derivatives_history/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)
print(f"Deriv: {len(deriv):,} rows", flush=True)

ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_merged.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)

# Forward returns
for h in [4, 8, 16]:
    ohlcv[f'fwd_ret_{h}'] = ohlcv['Close'].shift(-h) / ohlcv['Close'] - 1

# Merge
merged = pd.merge_asof(
    deriv,
    ohlcv[['timestamp', 'Close', 'High', 'Low', 'Volume', 'fwd_ret_4', 'fwd_ret_8', 'fwd_ret_16']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('15min')
)
merged = merged.dropna(subset=['Close', 'fwd_ret_4'])
print(f"Merged: {len(merged):,} rows", flush=True)

# ═══════════════════════════════════════════════════════
# FR Z-SCORE ANALYSIS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("FUNDING RATE Z-SCORE ANALYSIS", flush=True)
print("="*70, flush=True)

# Compute FR z-score
if 'funding_rate' in merged.columns:
    fr = merged['funding_rate'].values
    # Rolling z-score (96-bar lookback)
    lookback = 96
    fr_series = pd.Series(fr)
    fr_mean = fr_series.rolling(lookback).mean()
    fr_std = fr_series.rolling(lookback).std()
    merged['fr_zscore'] = (fr_series - fr_mean) / fr_std
    
    # Drop NaN
    valid = merged.dropna(subset=['fr_zscore', 'fwd_ret_4'])
    print(f"Valid rows with FR z-score: {len(valid):,}", flush=True)
    
    # FR z-score distribution
    print(f"\nFR z-score distribution:", flush=True)
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(valid['fr_zscore'].dropna(), pct)
        print(f"  P{pct}: {val:.3f}", flush=True)
    
    # ═══════════════════════════════════════════════════════
    # EXTREME FR Z-SCORE → FORWARD RETURNS
    # ═══════════════════════════════════════════════════════
    print(f"\n--- FR z-score extremes vs forward returns ---", flush=True)
    print(f"{'Threshold':>12s} {'Direction':>10s} {'4h WR':>8s} {'4h Mean':>10s} {'8h WR':>8s} {'8h Mean':>10s} {'p(4h)':>10s} {'n':>8s}", flush=True)
    print("-"*80, flush=True)
    
    for thresh in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
        # SHORT: FR z > thresh (overleveraged longs → price drops)
        short_sig = valid[valid['fr_zscore'] > thresh]
        if len(short_sig) > 20:
            ret4 = short_sig['fwd_ret_4']
            ret8 = short_sig['fwd_ret_8']
            wr4 = (ret4 < 0).mean()  # SHORT wins when price drops
            wr8 = (ret8 < 0).mean()
            t, p = stats.ttest_1samp(-ret4, 0)
            p1 = p/2 if t > 0 else 1-p/2
            print(f"  z>{thresh:.2f} SHORT  {wr4*100:>7.1f}% {(-ret4).mean()*100:>+9.4f}% {wr8*100:>7.1f}% {(-ret8).mean()*100:>+9.4f}% {p1:>10.6f} {len(short_sig):>8,}", flush=True)
        
        # LONG: FR z < -thresh (overleveraged shorts → price rises)
        long_sig = valid[valid['fr_zscore'] < -thresh]
        if len(long_sig) > 20:
            ret4 = long_sig['fwd_ret_4']
            ret8 = long_sig['fwd_ret_8']
            wr4 = (ret4 > 0).mean()
            wr8 = (ret8 > 0).mean()
            t, p = stats.ttest_1samp(ret4, 0)
            p1 = p/2 if t > 0 else 1-p/2
            print(f"  z<{thresh:.2f} LONG   {wr4*100:>7.1f}% {ret4.mean()*100:>+9.4f}% {wr8*100:>7.1f}% {ret8.mean()*100:>+9.4f}% {p1:>10.6f} {len(long_sig):>8,}", flush=True)
    
    # ═══════════════════════════════════════════════════════
    # VOL REGIME BREAKDOWN
    # ═══════════════════════════════════════════════════════
    print(f"\n--- Vol regime breakdown (FR z > 1.25 SHORT) ---", flush=True)
    
    # Compute vol regime
    merged['vol_20'] = merged['Close'].pct_change().rolling(20).std()
    vol_p33 = merged['vol_20'].quantile(0.33)
    vol_p67 = merged['vol_20'].quantile(0.67)
    merged['vol_regime'] = 'MID'
    merged.loc[merged['vol_20'] < vol_p33, 'vol_regime'] = 'LOW'
    merged.loc[merged['vol_20'] > vol_p67, 'vol_regime'] = 'HIGH'
    
    fr_short = merged[(merged['fr_zscore'] > 1.25)]
    print(f"{'Vol Regime':>12s} {'4h WR':>8s} {'4h Mean':>10s} {'8h WR':>8s} {'8h Mean':>10s} {'p(4h)':>10s} {'n':>8s}", flush=True)
    print("-"*70, flush=True)
    
    for vol in ['LOW', 'MID', 'HIGH']:
        sub = fr_short[fr_short['vol_regime'] == vol]
        if len(sub) < 10:
            continue
        ret4 = sub['fwd_ret_4']
        ret8 = sub['fwd_ret_8']
        wr4 = (ret4 < 0).mean()
        wr8 = (ret8 < 0).mean()
        t, p = stats.ttest_1samp(-ret4, 0)
        p1 = p/2 if t > 0 else 1-p/2
        print(f"  {vol:>12s} {wr4*100:>7.1f}% {(-ret4).mean()*100:>+9.4f}% {wr8*100:>7.1f}% {(-ret8).mean()*100:>+9.4f}% {p1:>10.6f} {len(sub):>8,}", flush=True)
    
    # ═══════════════════════════════════════════════════════
    # REGIME BREAKDOWN
    # ═══════════════════════════════════════════════════════
    print(f"\n--- Regime breakdown (FR z > 1.25 SHORT) ---", flush=True)
    
    merged['ema200'] = merged['Close'].ewm(span=200).mean()
    merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')
    
    fr_short = merged[(merged['fr_zscore'] > 1.25)]
    print(f"{'Regime':>12s} {'4h WR':>8s} {'4h Mean':>10s} {'8h WR':>8s} {'8h Mean':>10s} {'p(4h)':>10s} {'n':>8s}", flush=True)
    print("-"*70, flush=True)
    
    for trend in ['BULL', 'BEAR']:
        sub = fr_short[fr_short['trend'] == trend]
        if len(sub) < 10:
            continue
        ret4 = sub['fwd_ret_4']
        ret8 = sub['fwd_ret_8']
        wr4 = (ret4 < 0).mean()
        wr8 = (ret8 < 0).mean()
        t, p = stats.ttest_1samp(-ret4, 0)
        p1 = p/2 if t > 0 else 1-p/2
        print(f"  {trend:>12s} {wr4*100:>7.1f}% {(-ret4).mean()*100:>+9.4f}% {wr8*100:>7.1f}% {(-ret8).mean()*100:>+9.4f}% {p1:>10.6f} {len(sub):>8,}", flush=True)
    
    # ═══════════════════════════════════════════════════════
    # SESSION BREAKDOWN
    # ═══════════════════════════════════════════════════════
    print(f"\n--- Session breakdown (FR z > 1.25 SHORT) ---", flush=True)
    
    def get_session(ts):
        h = ts.hour
        if 0 <= h < 8: return 'ASIA'
        elif 8 <= h < 14: return 'EU'
        elif 14 <= h < 22: return 'US'
        else: return 'LATE'
    
    fr_short = fr_short.copy()
    fr_short['session'] = fr_short['timestamp'].apply(get_session)
    
    print(f"{'Session':>12s} {'4h WR':>8s} {'4h Mean':>10s} {'8h WR':>8s} {'8h Mean':>10s} {'n':>8s}", flush=True)
    print("-"*60, flush=True)
    
    for session in ['ASIA', 'EU', 'US']:
        sub = fr_short[fr_short['session'] == session]
        if len(sub) < 10:
            continue
        ret4 = sub['fwd_ret_4']
        ret8 = sub['fwd_ret_8']
        wr4 = (ret4 < 0).mean()
        wr8 = (ret8 < 0).mean()
        print(f"  {session:>12s} {wr4*100:>7.1f}% {(-ret4).mean()*100:>+9.4f}% {wr8*100:>7.1f}% {(-ret8).mean()*100:>+9.4f}% {len(sub):>8,}", flush=True)
    
    # ═══════════════════════════════════════════════════════
    # OI INTERACTION
    # ═══════════════════════════════════════════════════════
    print(f"\n--- OI interaction (FR z > 1.25 SHORT) ---", flush=True)
    
    if 'oi' in merged.columns:
        merged['oi_change'] = merged['oi'].pct_change()
        fr_short = merged[(merged['fr_zscore'] > 1.25)].copy()
        
        for oi_label, oi_fn in [
            ('OI rising', lambda x: x > 0.01),
            ('OI falling', lambda x: x < -0.01),
            ('OI flat', lambda x: abs(x) <= 0.01),
        ]:
            sub = fr_short[oi_fn(fr_short['oi_change'])]
            if len(sub) < 10:
                continue
            ret4 = sub['fwd_ret_4']
            wr4 = (ret4 < 0).mean()
            print(f"  {oi_label}: WR={wr4*100:.1f}% mean={(-ret4).mean()*100:+.4f}% n={len(sub):,}", flush=True)
    
    # ═══════════════════════════════════════════════════════
    # DIAGNOSTIC: WHY IS THE STRATEGY LOSING?
    # ═══════════════════════════════════════════════════════
    print(f"\n" + "="*70, flush=True)
    print("DIAGNOSTIC: WHY IS THE STRATEGY LOSING?", flush=True)
    print("="*70, flush=True)
    
    # The gate says -0.153% mean at 4h, but WR=53.8%
    # This means: direction is correct (>50% WR) but mean is negative
    # This can happen if losses are bigger than wins
    
    fr_short = merged[(merged['fr_zscore'] > 1.25)]
    ret4 = fr_short['fwd_ret_4']
    
    wins = ret4[ret4 < 0]  # SHORT wins when price drops
    losses = ret4[ret4 > 0]  # SHORT loses when price rises
    
    print(f"\nWin/Loss asymmetry:", flush=True)
    print(f"  Win rate: {(ret4 < 0).mean()*100:.1f}% (price dropped)", flush=True)
    print(f"  Mean win: {(-wins).mean()*100:+.4f}% (how much price dropped)", flush=True)
    print(f"  Mean loss: {losses.mean()*100:+.4f}% (how much price rose)", flush=True)
    print(f"  Win/Loss ratio: {(-wins).mean() / losses.mean():.2f}", flush=True)
    print(f"  Mean return: {(-ret4).mean()*100:+.4f}% (strategy expectation)", flush=True)
    
    # The issue: WR > 50% but mean < 0 → losses are larger than wins
    if (ret4 < 0).mean() > 0.5 and (-ret4).mean() < 0:
        print(f"\n  ⚠️ DIAGNOSIS: Win rate > 50% but mean return is NEGATIVE", flush=True)
        print(f"  This means: small frequent wins, large infrequent losses", flush=True)
        print(f"  The strategy is picking up pennies in front of a steamroller", flush=True)
    
    # Check if OI rising amplifies losses
    if 'oi' in merged.columns:
        fr_short_oi = fr_short.copy()
        fr_short_oi['oi_change'] = merged.loc[fr_short.index, 'oi_change']
        
        oi_rising = fr_short_oi[fr_short_oi['oi_change'] > 0.01]
        oi_falling = fr_short_oi[fr_short_oi['oi_change'] < -0.01]
        
        if len(oi_rising) > 10 and len(oi_falling) > 10:
            print(f"\n  OI rising + FR z>1.25: mean={(-oi_rising['fwd_ret_4']).mean()*100:+.4f}% n={len(oi_rising)}", flush=True)
            print(f"  OI falling + FR z>1.25: mean={(-oi_falling['fwd_ret_4']).mean()*100:+.4f}% n={len(oi_falling)}", flush=True)
            
            if (-oi_rising['fwd_ret_4']).mean() > (-oi_falling['fwd_ret_4']).mean():
                print(f"  → OI RISING amplifies the SHORT edge (more overleveraged longs)", flush=True)
            else:
                print(f"  → OI FALLING amplifies the SHORT edge (longs closing = squeeze)", flush=True)

print(f"\nTime: {time.time()-t0:.1f}s", flush=True)
print("="*70, flush=True)
print("FORENSIC COMPLETE", flush=True)
print("="*70, flush=True)
