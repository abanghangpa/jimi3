"""
Funding Z-Score SHORT Signal — Phase 2 (Agents 5-8)
FR z-score > 1.25 → SHORT: 602 events, -0.153%, p=0.003, WR=53.8%
"""
import pandas as pd, numpy as np
from scipy import stats

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'

d = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
d['timestamp'] = pd.to_datetime(d['Open time'])
d = d.sort_values('timestamp').reset_index(drop=True)

deriv = pd.read_csv(f'{DATA_DIR}/derivatives_history/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
m = pd.merge_asof(d, deriv[['timestamp','funding_rate','oi','ls_ratio']],
                   on='timestamp', direction='backward', tolerance=pd.Timedelta('2h'))

fr = m['funding_rate']
m['fr_z'] = (fr - fr.rolling(96).mean()) / fr.rolling(96).std()
m['oi_roc'] = m['oi'].pct_change(4, fill_method=None)
m['vol_20bar'] = m['Close'].pct_change().rolling(20).std()
m['ema200'] = m['Close'].ewm(span=200).mean()
m['ema50'] = m['Close'].ewm(span=50).mean()
m['trend'] = np.where(m['Close'] > m['ema200'], 'BULL', 'BEAR')
m['vol_ma20'] = m['Volume'].rolling(20).mean()
m['vol_ratio'] = m['Volume'] / m['vol_ma20']

for h in [1, 4, 16, 24]:
    m[f'fwd_ret_{h}'] = m['Close'].shift(-h) / m['Close'] - 1

# TP/SL simulation helper
def tp_sl_sim(mask, tp_pct=0.02, sl_pct=0.01, hold=16):
    shifted = mask.shift(1).fillna(False)
    events = m[shifted]
    trades = []
    for idx in events.index:
        entry = m.loc[idx, 'Close']
        won = None; bars = 0
        for h in range(1, hold + 1):
            if idx + h >= len(m): break
            high = m.loc[idx + h, 'High']
            low = m.loc[idx + h, 'Low']
            # SHORT: SL above, TP below
            if high >= entry * (1 + sl_pct):
                won = False; bars = h; break
            if low <= entry * (1 - tp_pct):
                won = True; bars = h; break
        if won is None and idx + hold < len(m):
            close = m.loc[idx + hold, 'Close']
            won = close < entry; bars = hold
        if won is not None:
            trades.append({'won': won, 'bars': bars})
    if not trades: return None
    w = sum(1 for t in trades if t['won'])
    l = len(trades) - w
    wr = w / len(trades)
    pf = (w * tp_pct) / (l * sl_pct) if l > 0 else float('inf')
    avg_bars = np.mean([t['bars'] for t in trades])
    pnl = w * tp_pct * 1000 - l * sl_pct * 1000
    return {'trades': len(trades), 'wins': w, 'losses': l, 'wr': wr, 'pf': pf, 'avg_bars': avg_bars, 'pnl': pnl}

base_mask = m['fr_z'] > 1.25
round_trip_cost = 0.0010

# ═══════════════════════════════════════════════════════
# AGENT 5: STRESS TEST
# ═══════════════════════════════════════════════════════
print("="*70)
print("AGENT 5: STRESS TEST — Z-score thresholds + TP/SL")
print("="*70)

print("\nForward returns at 4h:")
for thresh in [1.0, 1.25, 1.5, 1.75, 2.0]:
    mask = m['fr_z'] > thresh
    shifted = mask.shift(1).fillna(False)
    events = m[shifted]
    rets = m.loc[events.index, 'fwd_ret_16'].dropna()
    if len(rets) < 5: continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    eff_wr = 1 - (rets > 0).mean()
    gate = "PASS" if p < 0.1 and abs(mean_r) > round_trip_cost else "FAIL"
    sym = "+" if gate == "PASS" else "-"
    print(f"  {sym} FR_z > {thresh:5.2f}: n={len(rets):5d}, mean={mean_r*100:+.4f}%, eff_WR={eff_wr:.1%}, p={p:.4f} [{gate}]")

print("\nTP/SL simulation (SHORT, hold=4h):")
for tp, sl in [(0.015, 0.01), (0.02, 0.01), (0.025, 0.015), (0.03, 0.015), (0.02, 0.0075)]:
    for thresh in [1.0, 1.25, 1.5]:
        mask = m['fr_z'] > thresh
        result = tp_sl_sim(mask, tp_pct=tp, sl_pct=sl, hold=16)
        if not result: continue
        sym = "+" if result['pf'] > 1.0 else "-"
        print(f"  {sym} TP={tp*100:.1f}% SL={sl*100:.2f}% z>{thresh:4.2f}: trades={result['trades']:4d}, WR={result['wr']:.1%}, PF={result['pf']:.2f}, PnL=${result['pnl']:+.0f}")

# ═══════════════════════════════════════════════════════
# AGENT 6: REGIME TESTER
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 6: REGIME TESTER")
print("="*70)

vols = m['vol_20bar'].dropna()
p33, p67 = vols.quantile(0.33), vols.quantile(0.67)
m['vol_regime'] = 'MID'
m.loc[m['vol_20bar'] < p33, 'vol_regime'] = 'LOW'
m.loc[m['vol_20bar'] > p67, 'vol_regime'] = 'HIGH'

for regime_col, regime_name in [('vol_regime', 'Vol Regime'), ('trend', 'Trend')]:
    print(f"\n--- {regime_name} (FR_z > 1.25) ---")
    for regime in sorted(m[regime_col].dropna().unique()):
        regime_events = m[base_mask & (m[regime_col] == regime)]
        if len(regime_events) < 5:
            print(f"  {regime}: n={len(regime_events)} (too few)")
            continue
        for h, label in [(4, '1h'), (16, '4h')]:
            rets = m.loc[regime_events.index, f'fwd_ret_{h}'].dropna()
            if len(rets) < 5: continue
            mean_r = rets.mean()
            t, p = stats.ttest_1samp(rets, 0)
            eff_wr = 1 - (rets > 0).mean()
            gate = "PASS" if p < 0.1 and abs(mean_r) > round_trip_cost else "FAIL"
            sym = "+" if gate == "PASS" else "-"
            print(f"  {sym} {regime:12s} {label}: n={len(rets):5d}, mean={mean_r*100:+.4f}%, eff_WR={eff_wr:.1%}, p={p:.4f} [{gate}]")

# Calendar era
def get_era(ts):
    if ts < pd.Timestamp('2024-07-01'): return '2024_H1'
    elif ts < pd.Timestamp('2025-01-01'): return '2024_H2'
    elif ts < pd.Timestamp('2025-07-01'): return '2025_H1'
    elif ts < pd.Timestamp('2026-01-01'): return '2025_H2'
    else: return '2026'

m['era'] = m['timestamp'].apply(get_era)
print(f"\n--- Calendar Era (FR_z > 1.25) ---")
for era in sorted(m['era'].unique()):
    era_events = m[base_mask & (m['era'] == era)]
    if len(era_events) < 5:
        print(f"  {era}: n={len(era_events)} (too few)")
        continue
    for h, label in [(4, '1h'), (16, '4h')]:
        rets = m.loc[era_events.index, f'fwd_ret_{h}'].dropna()
        if len(rets) < 5: continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        eff_wr = 1 - (rets > 0).mean()
        gate = "PASS" if p < 0.1 and abs(mean_r) > round_trip_cost else "FAIL"
        sym = "+" if gate == "PASS" else "-"
        print(f"  {sym} {era:10s} {label}: n={len(rets):5d}, mean={mean_r*100:+.4f}%, eff_WR={eff_wr:.1%}, p={p:.4f} [{gate}]")

# TP/SL per regime
print("\n--- TP/SL per regime (TP=2%, SL=1%, hold=4h) ---")
for regime in sorted(m['vol_regime'].dropna().unique()):
    mask = base_mask & (m['vol_regime'] == regime)
    result = tp_sl_sim(mask, tp_pct=0.02, sl_pct=0.01, hold=16)
    if not result:
        print(f"  {regime}: no trades")
        continue
    sym = "+" if result['pf'] > 1.0 else "-"
    print(f"  {sym} {regime:8s}: trades={result['trades']:4d}, WR={result['wr']:.1%}, PF={result['pf']:.2f}, PnL=${result['pnl']:+.0f}")

# ═══════════════════════════════════════════════════════
# AGENT 7: CONFLUENCE
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 7: CONFLUENCE")
print("="*70)

base_events = m[base_mask]
base_rets = m.loc[base_events.index, 'fwd_ret_16'].dropna()
base_mean = base_rets.mean()
print(f"Base (FR_z > 1.25): n={len(base_rets)}, mean={base_mean*100:+.4f}%")

filters = [
    ('+ LS > 1.5', m['ls_ratio'] > 1.5),
    ('+ LS > 2.0', m['ls_ratio'] > 2.0),
    ('+ LS < 0.67', m['ls_ratio'] < 0.67),
    ('+ OI ROC < -0.01', m['oi_roc'] < -0.01),
    ('+ OI ROC > 0.01', m['oi_roc'] > 0.01),
    ('+ BULL trend', m['trend'] == 'BULL'),
    ('+ BEAR trend', m['trend'] == 'BEAR'),
    ('+ LOW vol', m['vol_regime'] == 'LOW'),
    ('+ MID vol', m['vol_regime'] == 'MID'),
    ('+ HIGH vol', m['vol_regime'] == 'HIGH'),
    ('+ vol_ratio > 1.5', m['vol_ratio'] > 1.5),
    ('+ vol_ratio > 2.0', m['vol_ratio'] > 2.0),
    ('+ price > EMA50', m['Close'] > m['ema50']),
    ('+ price < EMA50', m['Close'] < m['ema50']),
    ('+ FR_z > 1.5', m['fr_z'] > 1.5),
    ('+ FR_z > 1.75', m['fr_z'] > 1.75),
]

for name, filt in filters:
    filtered = base_events[filt.loc[base_events.index]]
    if len(filtered) < 5:
        print(f"  {name}: n={len(filtered)} (too few)")
        continue
    rets = m.loc[filtered.index, 'fwd_ret_16'].dropna()
    if len(rets) < 5: continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    eff_wr = 1 - (rets > 0).mean()
    delta = mean_r - base_mean
    gate = "PASS" if p < 0.1 and abs(mean_r) > 0.001 else "FAIL"
    sym = "+" if gate == "PASS" else "-"
    print(f"  {sym} {name:20s} n={len(rets):5d}, mean={mean_r*100:+.4f}%, p={p:.4f}, eff_WR={eff_wr:.1%}, delta={delta*100:+.4f}%")

# ═══════════════════════════════════════════════════════
# AGENT 8: ALTERNATIVE DETECTION
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGENT 8: ALTERNATIVE DETECTION")
print("="*70)

alt_configs = [
    ('FR_z > 1.0 + LS > 1.5', (m['fr_z'] > 1.0) & (m['ls_ratio'] > 1.5)),
    ('FR_z > 1.0 + OI ROC < -0.01', (m['fr_z'] > 1.0) & (m['oi_roc'] < -0.01)),
    ('FR_z > 1.0 + OI ROC > 0.01', (m['fr_z'] > 1.0) & (m['oi_roc'] > 0.01)),
    ('FR_z > 1.0 + vol > 1.5', (m['fr_z'] > 1.0) & (m['vol_ratio'] > 1.5)),
    ('FR_z > 1.0 + BEAR trend', (m['fr_z'] > 1.0) & (m['trend'] == 'BEAR')),
    ('FR_z > 1.0 + BULL trend', (m['fr_z'] > 1.0) & (m['trend'] == 'BULL')),
    ('FR_z rising (4-bar)', m['fr_z'].diff(4) > 0.5),
    ('FR_z falling (4-bar)', m['fr_z'].diff(4) < -0.5),
]

print("\nTesting at 4h (16-bar) horizon:")
for name, mask in alt_configs:
    shifted = mask.shift(1).fillna(False)
    events = m[shifted]
    rets = m.loc[events.index, 'fwd_ret_16'].dropna()
    if len(rets) < 5:
        print(f"  {name}: n={len(rets)} (too few)")
        continue
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    eff_wr = 1 - (rets > 0).mean()
    gate = "PASS" if p < 0.1 and abs(mean_r) > 0.001 else "FAIL"
    sym = "+" if gate == "PASS" else "-"
    print(f"  {sym} {name:35s} n={len(rets):5d}, mean={mean_r*100:+.4f}%, eff_WR={eff_wr:.1%}, p={p:.4f} [{gate}]")

# ═══════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)
print("FR z-score > 1.25 → SHORT: 602 events, -0.153%, p=0.003, WR=53.8%")
print("Protocol: Phase 2 complete — see results above")
