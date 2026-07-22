"""FINAL isolation gate — collected data only, no backfilled contamination."""
import pandas as pd
import numpy as np
from scipy import stats

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'

ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)

# Use ONLY collected derivatives (real data, no backfilled contamination)
deriv = pd.read_csv(f'{DATA_DIR}/derivatives_history/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed')
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

merged = pd.merge_asof(
    ohlcv, deriv[['timestamp', 'oi', 'ls_ratio', 'funding_rate', 'futures_taker_ratio']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)

merged['oi_roc_1h'] = merged['oi'].pct_change(4, fill_method=None)
merged['price_change_4bar'] = merged['Close'].pct_change(4)
merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()

# Vol regime
vols = merged['vol_20bar'].dropna()
p33, p67 = vols.quantile(0.33), vols.quantile(0.67)
merged['vol_regime'] = 'MID'
merged.loc[merged['vol_20bar'] < p33, 'vol_regime'] = 'LOW'
merged.loc[merged['vol_20bar'] > p67, 'vol_regime'] = 'HIGH'

# Forward returns
for h in [1, 4, 16, 24]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

print(f"OI coverage: {merged['oi'].notna().sum()}/{len(merged)}")
print(f"LS coverage: {merged['ls_ratio'].notna().sum()}/{len(merged)}")
print(f"OI ROC valid: {merged['oi_roc_1h'].notna().sum()}")

# ═══════════════════════════════════════════════════════
# PHASE 1 REPLICATION with collected-only data
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 1 GATE — Collected data only")
print("="*60)

# Source A: OI shock
oi_shock = (
    (merged['oi_roc_1h'].abs() > 0.01) &
    (merged['oi_roc_1h'] < -0.01) &
    (merged['price_change_4bar'].abs() > 0.005)
)
# Source B: OI + L/S extreme (real LS values only)
oi_extreme = (
    (merged['oi_roc_1h'] < -0.01) &
    ((merged['ls_ratio'] > 1.8) | (merged['ls_ratio'] < 0.6))
)
# Source C: Large OI drop
oi_large = (merged['oi_roc_1h'] < -0.015)

cascade_mask = oi_shock | oi_extreme | oi_large
cascade_shifted = cascade_mask.shift(1).fillna(False)

print(f"Source A (OI shock): {oi_shock.sum()}")
print(f"Source B (OI+LS extreme): {oi_extreme.sum()}")
print(f"Source C (OI large drop): {oi_large.sum()}")
print(f"Combined events: {cascade_shifted.sum()}")

events = merged[cascade_shifted]
for h, label in [(1, '15m'), (4, '1h'), (16, '4h'), (24, '6h')]:
    rets = merged.loc[events.index, f'fwd_ret_{h}'].dropna()
    if len(rets) < 5:
        print(f"  {label}: Too few ({len(rets)})")
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    print(f"  {label}: mean={mean_r*100:+.4f}%, p={p:.4f}, n={len(rets)}, WR={wr:.1%}, dir={'OK' if mean_r > 0 else 'BACKWARDS'}")

# ═══════════════════════════════════════════════════════
# SINGLE-SOURCE TESTS (collected only)
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("SINGLE-SOURCE TESTS — Collected only")
print("="*60)

configs = [
    ('OI ROC < -0.005', merged['oi_roc_1h'] < -0.005),
    ('OI ROC < -0.01', merged['oi_roc_1h'] < -0.01),
    ('OI ROC < -0.015', merged['oi_roc_1h'] < -0.015),
    ('OI ROC < -0.01 + LS>1.5', (merged['oi_roc_1h'] < -0.01) & (merged['ls_ratio'] > 1.5)),
    ('OI ROC < -0.01 + LS>1.8', (merged['oi_roc_1h'] < -0.01) & (merged['ls_ratio'] > 1.8)),
    ('OI ROC < -0.015 + LS>1.5', (merged['oi_roc_1h'] < -0.015) & (merged['ls_ratio'] > 1.5)),
    ('OI ROC < -0.015 + MID vol', (merged['oi_roc_1h'] < -0.015) & (merged['vol_regime'] == 'MID')),
    ('OI ROC < -0.01 + MID vol', (merged['oi_roc_1h'] < -0.01) & (merged['vol_regime'] == 'MID')),
    ('OI ROC < -0.015 + LS>1.5 + MID', (merged['oi_roc_1h'] < -0.015) & (merged['ls_ratio'] > 1.5) & (merged['vol_regime'] == 'MID')),
    ('OI ROC < -0.01 + LS>1.5 + MID', (merged['oi_roc_1h'] < -0.01) & (merged['ls_ratio'] > 1.5) & (merged['vol_regime'] == 'MID')),
    ('LS ratio > 2.0 (no OI filter)', merged['ls_ratio'] > 2.0),
    ('LS ratio > 2.5', merged['ls_ratio'] > 2.5),
]

for name, mask in configs:
    shifted = mask.shift(1).fillna(False)
    events = merged[shifted]
    rets = merged.loc[events.index, 'fwd_ret_16'].dropna()
    if len(rets) < 5:
        print(f"  {name}: n={len(rets)} (too few)")
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    gate = "PASS" if p < 0.1 and mean_r > 0.001 else "FAIL"
    print(f"  {'✅' if gate=='PASS' else '❌'} {name:40s} n={len(rets):4d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# ═══════════════════════════════════════════════════════
# TP/SL SIMULATION (collected only)
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("TP/SL SIMULATION — TP=2%, SL=1%, Hold=4h, SHORT only")
print("="*60)

tp_pct = 0.02
sl_pct = 0.01
hold = 16

for name, mask in [
    ('OI<-0.015 + LS>1.5', (merged['oi_roc_1h'] < -0.015) & (merged['ls_ratio'] > 1.5)),
    ('OI<-0.015 + LS>1.5 + MID', (merged['oi_roc_1h'] < -0.015) & (merged['ls_ratio'] > 1.5) & (merged['vol_regime'] == 'MID')),
    ('OI<-0.01 + LS>1.5', (merged['oi_roc_1h'] < -0.01) & (merged['ls_ratio'] > 1.5)),
    ('OI<-0.01 + LS>1.5 + MID', (merged['oi_roc_1h'] < -0.01) & (merged['ls_ratio'] > 1.5) & (merged['vol_regime'] == 'MID')),
]:
    shifted = mask.shift(1).fillna(False)
    evts = merged[shifted]
    trades = []
    for idx in evts.index:
        entry = merged.loc[idx, 'Close']
        won = None
        bars = 0
        for h in range(1, hold + 1):
            if idx + h >= len(merged):
                break
            high = merged.loc[idx + h, 'High']
            low = merged.loc[idx + h, 'Low']
            if high >= entry * (1 + sl_pct):
                won = False; bars = h; break
            if low <= entry * (1 - tp_pct):
                won = True; bars = h; break
        if won is None and idx + hold < len(merged):
            close = merged.loc[idx + hold, 'Close']
            won = close < entry; bars = hold
        if won is not None:
            trades.append({'won': won, 'bars': bars})

    if not trades:
        print(f"  {name}: No trades")
        continue
    w = sum(1 for t in trades if t['won'])
    l = len(trades) - w
    wr = w / len(trades)
    pf = (w * tp_pct) / (l * sl_pct) if l > 0 else float('inf')
    avg_bars = np.mean([t['bars'] for t in trades])
    pnl = w * tp_pct * 1000 - l * sl_pct * 1000
    print(f"  {name:35s} trades={len(trades):3d}, WR={wr:.1%} ({w}W/{l}L), PF={pf:.2f}, avg_hold={avg_bars:.0f}bar, PnL=${pnl:+.0f}")
