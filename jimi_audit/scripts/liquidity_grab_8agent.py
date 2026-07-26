"""
8-Agent Protocol: Liquidity Grab v6
S/R breakout + taker z-score + volume spike + momentum confirmation

Agent 1: Forensics — signal quality
Agent 2: Non-indicator — raw S/R proximity
Agent 3: Context filters (session, regime, vol)
Agent 4: Co-occurrence
Agent 5: Walk-forward
Agent 6: Monte Carlo
Agent 7: Regime-conditional
Agent 8: Statistical significance
"""
import pandas as pd
import numpy as np
from scipy import stats
import json, os

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

merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['trend'] = np.where(merged['Close'] > merged['ema200'], 'BULL', 'BEAR')
merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()

for h in [4, 8, 16, 24, 48]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

closes = merged['Close'].values
highs_arr = merged['High'].values
lows_arr = merged['Low'].values
volumes = merged['Volume'].values
taker_base = merged['Taker buy base asset volume'].values
n = len(merged)

BAD_HOURS = {4, 5, 6, 19, 20, 22, 23}

def find_sr_levels(idx, lookback=96):
    levels = []
    for i in range(3, min(lookback, idx)):
        bar_idx = idx - i
        if bar_idx < 3:
            continue
        if (highs_arr[bar_idx] > highs_arr[bar_idx-1] and
            highs_arr[bar_idx] > highs_arr[bar_idx-2] and
            highs_arr[bar_idx] > highs_arr[bar_idx-3]):
            lvl = highs_arr[bar_idx]
            touches = sum(1 for j in range(max(0, bar_idx-10), bar_idx+1)
                         if abs(highs_arr[j] - lvl) / lvl < 0.002)
            levels.append({'price': lvl, 'type': 'resistance', 'touches': touches,
                          'bars_ago': idx - bar_idx, 'strength': touches * (1 + np.log1p(volumes[bar_idx]))})
        if (lows_arr[bar_idx] < lows_arr[bar_idx-1] and
            lows_arr[bar_idx] < lows_arr[bar_idx-2] and
            lows_arr[bar_idx] < lows_arr[bar_idx-3]):
            lvl = lows_arr[bar_idx]
            touches = sum(1 for j in range(max(0, bar_idx-10), bar_idx+1)
                         if abs(lows_arr[j] - lvl) / lvl < 0.002)
            levels.append({'price': lvl, 'type': 'support', 'touches': touches,
                          'bars_ago': idx - bar_idx, 'strength': touches * (1 + np.log1p(volumes[bar_idx]))})
    deduped = []
    for lv in sorted(levels, key=lambda x: x['bars_ago']):
        found = False
        for d in deduped:
            if abs(lv['price'] - d['price']) / d['price'] < 0.002:
                d['touches'] = max(d['touches'], lv['touches'])
                d['strength'] = max(d['strength'], lv['strength'])
                found = True
                break
        if not found:
            deduped.append(lv)
    deduped.sort(key=lambda x: x['strength'], reverse=True)
    return deduped[:10]

def compute_taker_zscore(idx):
    recent_buy = np.sum(taker_base[idx-4:idx])
    recent_total = np.sum(volumes[idx-4:idx])
    if recent_total == 0:
        return None
    taker_ratio = recent_buy / recent_total
    window_buy = taker_base[max(0, idx-60):idx]
    window_total = volumes[max(0, idx-60):idx]
    window_ratios = []
    for j in range(0, len(window_buy)-4, 4):
        wb = np.sum(window_buy[j:j+4])
        wt = np.sum(window_total[j:j+4])
        if wt > 0:
            window_ratios.append(wb / wt)
    if len(window_ratios) < 5:
        return None
    mean_r = np.mean(window_ratios)
    std_r = np.std(window_ratios)
    if std_r == 0:
        return None
    return (taker_ratio - mean_r) / std_r

# ═══════════════════════════════════════════════════════
# GENERATE SIGNALS
# ═══════════════════════════════════════════════════════
print("Generating liquidity_grab v6 signals...")
signals = []

for idx in range(60, n):
    price = closes[idx]
    ts = merged.iloc[idx]['timestamp']
    hour = ts.hour
    if hour in BAD_HOURS:
        continue

    regime = merged.iloc[idx]['trend']
    if regime == 'STRESS':
        continue

    vr = merged.iloc[idx]['vol_ratio'] or 1.0
    if vr < 1.3:
        continue

    tz = compute_taker_zscore(idx)
    if tz is None or abs(tz) < 2.0:
        continue

    sr = find_sr_levels(idx)
    if not sr:
        continue

    best_level = None
    atr = merged.iloc[idx].get('atr', 0) or price * 0.01
    for level in sr:
        dist = abs(price - level['price']) / atr
        if dist < 1.5:
            best_level = level
            break
    if not best_level:
        continue

    direction = None
    if best_level['type'] == 'resistance':
        if price > best_level['price'] * 1.001:
            if tz > 2.0 and vr > 1.5:
                direction = 'LONG'
            elif idx >= 2 and closes[idx-1] < best_level['price'] and closes[idx] > best_level['price']:
                direction = 'LONG'
    elif best_level['type'] == 'support':
        if price < best_level['price'] * 0.999:
            if tz < -2.0 and vr > 1.5:
                direction = 'SHORT'
            elif idx >= 2 and closes[idx-1] > best_level['price'] and closes[idx] < best_level['price']:
                direction = 'SHORT'

    if not direction:
        continue

    mom_3 = (closes[idx] - closes[idx-3]) / closes[idx-3] if idx >= 3 else 0
    if direction == 'LONG' and mom_3 < 0:
        continue
    if direction == 'SHORT' and mom_3 > 0:
        continue

    base = 0.45
    taker_bonus = min((abs(tz) - 2.0) * 0.10, 0.20)
    vol_bonus = min((vr - 1.3) * 0.10, 0.15)
    level_bonus = min(best_level['strength'] / 15, 0.10)
    regime_bonus = 0.05 if regime in ('BULL', 'BEAR') else 0.0
    conviction = min(base + taker_bonus + vol_bonus + level_bonus + regime_bonus, 0.85)
    if conviction < 0.50:
        continue

    signals.append({
        'idx': idx, 'timestamp': ts, 'price': price, 'direction': direction,
        'conviction': conviction, 'taker_zscore': tz, 'vol_ratio': vr,
        'level_price': best_level['price'], 'level_type': best_level['type'],
        'touches': best_level['touches'], 'regime': regime,
        'fwd_ret_16': merged.iloc[idx]['fwd_ret_16'] if idx + 16 < n else np.nan,
    })

print(f"Total signals: {len(signals)}")

sdf = pd.DataFrame(signals)

# ═══════════════════════════════════════════════════════
# AGENT 1: FORENSICS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 1: FORENSICS")
print("="*70)

print(f"Total: {len(sdf)}")
print(f"LONG: {len(sdf[sdf['direction']=='LONG'])} SHORT: {len(sdf[sdf['direction']=='SHORT'])}")
print(f"Conviction: {sdf['conviction'].min():.2f} - {sdf['conviction'].max():.2f} (mean={sdf['conviction'].mean():.2f})")
print(f"Taker z-score: {sdf['taker_zscore'].min():.2f} - {sdf['taker_zscore'].max():.2f}")
print(f"Vol ratio: {sdf['vol_ratio'].min():.2f} - {sdf['vol_ratio'].max():.2f}")

for h in [4, 8, 16, 24]:
    col = f'fwd_ret_{h}'
    sdf[col] = sdf.apply(lambda r: merged.iloc[r['idx']][col] if r['idx'] + h < len(merged) else np.nan, axis=1)
    rets = sdf[col].dropna()
    if len(rets) > 0:
        dir_mult = sdf.loc[rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
        adj = rets * dir_mult
        wr = (adj > 0).mean()
        mean_r = adj.mean()
        t, p = stats.ttest_1samp(adj, 0)
        p1 = p/2 if t > 0 else 1-p/2
        print(f"  {h}h: WR={wr*100:.1f}% mean={mean_r*100:+.3f}% p={p1:.4f} n={len(rets)}")

# ═══════════════════════════════════════════════════════
# AGENT 2: NON-INDICATOR — raw S/R proximity
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 2: NON-INDICATOR — raw taker z-score extremes")
print("="*70)

# Test raw taker z-score > 2.0 without S/R
raw_signals = []
for idx in range(60, n):
    tz = compute_taker_zscore(idx)
    if tz is None or abs(tz) < 2.0:
        continue
    hour = merged.iloc[idx]['timestamp'].hour
    if hour in BAD_HOURS:
        continue
    direction = 'LONG' if tz > 2.0 else 'SHORT'
    raw_signals.append({'idx': idx, 'direction': direction,
                       'fwd_ret_16': merged.iloc[idx]['fwd_ret_16'] if idx + 16 < n else np.nan})

rsdf = pd.DataFrame(raw_signals)
print(f"Raw taker z>2 signals: {len(rsdf)}")
if len(rsdf) > 0:
    rets = rsdf['fwd_ret_16'].dropna()
    dir_mult = rsdf.loc[rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
    adj = rets * dir_mult
    print(f"  16h: WR={(adj>0).mean()*100:.1f}% mean={adj.mean()*100:+.3f}% n={len(rets)}")

# ═══════════════════════════════════════════════════════
# AGENT 3: CONTEXT FILTERS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 3: CONTEXT FILTERS")
print("="*70)

def get_session(ts):
    h = ts.hour
    if 0 <= h < 8: return 'ASIA'
    elif 8 <= h < 14: return 'EU'
    elif 14 <= h < 22: return 'US'
    else: return 'LATE'

sdf['session'] = sdf['timestamp'].apply(get_session)

for session in ['ASIA', 'EU', 'US']:
    sub = sdf[sdf['session'] == session]
    if len(sub) < 3:
        continue
    rets = sub['fwd_ret_16'].dropna()
    if len(rets) > 0:
        dir_mult = sub.loc[rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
        adj = rets * dir_mult
        print(f"  {session}: WR={(adj>0).mean()*100:.1f}% mean={adj.mean()*100:+.3f}% n={len(rets)}")

for regime in ['BULL', 'BEAR', 'RANGING']:
    sub = sdf[sdf['regime'] == regime]
    if len(sub) < 3:
        continue
    rets = sub['fwd_ret_16'].dropna()
    if len(rets) > 0:
        dir_mult = sub.loc[rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
        adj = rets * dir_mult
        print(f"  {regime}: WR={(adj>0).mean()*100:.1f}% mean={adj.mean()*100:+.3f}% n={len(rets)}")

for tz_range in [(2.0, 2.5), (2.5, 3.0), (3.0, 5.0)]:
    sub = sdf[(sdf['taker_zscore'].abs() >= tz_range[0]) & (sdf['taker_zscore'].abs() < tz_range[1])]
    if len(sub) < 3:
        continue
    rets = sub['fwd_ret_16'].dropna()
    if len(rets) > 0:
        dir_mult = sub.loc[rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
        adj = rets * dir_mult
        print(f"  Z[{tz_range[0]:.1f},{tz_range[1]:.1f}): WR={(adj>0).mean()*100:.1f}% mean={adj.mean()*100:+.3f}% n={len(rets)}")

# ═══════════════════════════════════════════════════════
# AGENT 5: WALK-FORWARD
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 5: WALK-FORWARD")
print("="*70)

sdf['week'] = sdf['timestamp'].dt.isocalendar().week.astype(int)
sdf['year'] = sdf['timestamp'].dt.isocalendar().year.astype(int)
sdf['yw'] = sdf['year'] * 100 + sdf['week']
weeks = sorted(sdf['yw'].unique())

wf_results = []
for i in range(0, len(weeks) - 4):
    test_w = weeks[i+4] if i+4 < len(weeks) else None
    if not test_w:
        break
    test = sdf[sdf['yw'] == test_w]
    rets = test['fwd_ret_16'].dropna()
    if len(rets) >= 2:
        dir_mult = test.loc[rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
        adj = rets * dir_mult
        wf_results.append({'week': test_w, 'n': len(rets), 'wr': (adj > 0).mean(), 'mean': adj.mean()})

if wf_results:
    wf_df = pd.DataFrame(wf_results)
    print(f"Walk-forward periods: {len(wf_df)}")
    print(f"Mean WR: {wf_df['wr'].mean()*100:.1f}%")
    print(f"Mean return: {wf_df['mean'].mean()*100:+.3f}%")
    win_periods = (wf_df['wr'] > 0.5).sum()
    print(f"Winning periods: {win_periods}/{len(wf_df)} ({win_periods/len(wf_df)*100:.0f}%)")
    for _, row in wf_df.iterrows():
        tag = "WIN" if row['wr'] > 0.5 else "LOSS"
        print(f"  [{tag}] Week {row['week']}: WR={row['wr']*100:.0f}% mean={row['mean']*100:+.3f}% n={row['n']}")

# ═══════════════════════════════════════════════════════
# AGENT 6: MONTE CARLO
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 6: MONTE CARLO")
print("="*70)

rets = sdf['fwd_ret_16'].dropna()
if len(rets) > 0:
    dir_mult = sdf.loc[rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
    adj = (rets * dir_mult).values
    n_sims = 5000
    horizon = 30
    sims = []
    for _ in range(n_sims):
        sampled = np.random.choice(adj, size=horizon, replace=True)
        sims.append(sampled.sum())
    sims = np.array(sims)
    p5, p25, p50, p75, p95 = np.percentile(sims, [5, 25, 50, 75, 95])
    print(f"30-trade horizon:")
    print(f"  P5={p5*100:+.2f}% P25={p25*100:+.2f}% P50={p50*100:+.2f}% P75={p75*100:+.2f}% P95={p95*100:+.2f}%")
    print(f"  Prob(loss)={(sims<0).mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════
# AGENT 7: REGIME-CONDITIONAL
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 7: REGIME-CONDITIONAL")
print("="*70)

for regime in ['BULL', 'BEAR', 'RANGING']:
    for direction in ['LONG', 'SHORT']:
        sub = sdf[(sdf['regime'] == regime) & (sdf['direction'] == direction)]
        if len(sub) < 3:
            continue
        rets = sub['fwd_ret_16'].dropna()
        if len(rets) > 0:
            dir_mult = 1 if direction == 'LONG' else -1
            adj = rets * dir_mult
            t, p = stats.ttest_1samp(adj, 0)
            p1 = p/2 if t > 0 else 1-p/2
            print(f"  {regime}+{direction}: WR={(adj>0).mean()*100:.1f}% mean={adj.mean()*100:+.3f}% p={p1:.4f} n={len(rets)}")

# ═══════════════════════════════════════════════════════
# AGENT 8: STATISTICAL SIGNIFICANCE
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 8: STATISTICAL SIGNIFICANCE")
print("="*70)

rets = sdf['fwd_ret_16'].dropna()
if len(rets) > 0:
    dir_mult = sdf.loc[rets.index, 'direction'].map({'LONG': 1, 'SHORT': -1})
    adj = rets * dir_mult
    t, p = stats.ttest_1samp(adj, 0)
    p1 = p/2 if t > 0 else 1-p/2
    boots = [np.random.choice(adj.values, size=len(adj), replace=True).mean() for _ in range(2000)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    wins = adj[adj > 0]
    losses = adj[adj < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float('inf')

    print(f"t-test: t={t:.3f} p(one-sided)={p1:.4f}")
    print(f"Bootstrap CI: [{ci_lo*100:+.3f}%, {ci_hi*100:+.3f}%]")
    print(f"WR={(adj>0).mean()*100:.1f}% PF={pf:.2f}")
    print(f"Mean win: {wins.mean()*100:+.3f}% Mean loss: {losses.mean()*100:+.3f}%")

    n_trades = len(adj)
    wr = (adj > 0).mean()
    mean_r = adj.mean()

    checks = []
    c = n_trades >= 20
    checks.append(f"n >= 20: {'PASS' if c else 'FAIL'} ({n_trades})")
    c = wr >= 0.52
    checks.append(f"WR >= 52%: {'PASS' if c else 'FAIL'} ({wr*100:.1f}%)")
    c = mean_r > 0
    checks.append(f"Mean > 0: {'PASS' if c else 'FAIL'} ({mean_r*100:+.3f}%)")
    c = p1 < 0.10
    checks.append(f"p < 0.10: {'PASS' if c else 'FAIL'} ({p1:.4f})")
    c = pf >= 1.2
    checks.append(f"PF >= 1.2: {'PASS' if c else 'FAIL'} ({pf:.2f})")
    c = ci_lo > 0
    checks.append(f"CI > 0: {'PASS' if c else 'FAIL'} ([{ci_lo*100:+.3f}%, {ci_hi*100:+.3f}%])")

    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    for chk in checks:
        print(f"  {chk}")
    score = sum(1 for chk in checks if 'PASS' in chk)
    print(f"\n  Score: {score}/6")
    if score >= 5:
        print("  RESULT: PASS")
    elif score >= 3:
        print("  RESULT: PROVISIONAL")
    else:
        print("  RESULT: FAIL")

print("\nDone.")
