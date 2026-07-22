"""
Hold-out validation: Train on first70%, test on last30%.
Strategies: liquidation_cascade, judas_sweep, funding_squeeze.
"""
import pandas as pd, numpy as np
from scipy import stats

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'

# Load OHLCV
d = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
d['timestamp'] = pd.to_datetime(d['Open time'])
d = d.sort_values('timestamp').reset_index(drop=True)

# Load derivatives
deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

m = pd.merge_asof(d, deriv[['timestamp','oi','ls_ratio','funding_rate']],
                   on='timestamp', direction='backward', tolerance=pd.Timedelta('2h'))

# Features
m['oi_roc'] = m['oi'].pct_change(4, fill_method=None)
m['vol_20bar'] = m['Close'].pct_change().rolling(20).std()
m['ema200'] = m['Close'].ewm(span=200).mean()
m['vol_ma20'] = m['Volume'].rolling(20).mean()
m['vol_ratio'] = m['Volume'] / m['vol_ma20']
m['trend'] = np.where(m['Close'] > m['ema200'], 'BULL', 'BEAR')
m['fr'] = m['funding_rate']
m['fr_z'] = (m['fr'] - m['fr'].rolling(96).mean()) / m['fr'].rolling(96).std()

# Forward returns
for h in [1, 4, 16, 24]:
    m[f'fwd_ret_{h}'] = m['Close'].shift(-h) / m['Close'] - 1

# Train/test split (70/30 by time)
n = len(m)
split_idx = int(n * 0.7)
train = m.iloc[:split_idx].copy()
test = m.iloc[split_idx:].copy()
print(f"Data: {n} bars")
print(f"Train: {len(train)} bars ({train['timestamp'].min()} to {train['timestamp'].max()})")
print(f"Test:  {len(test)} bars ({test['timestamp'].min()} to {test['timestamp'].max()})")

# Regime
for df in [train, test]:
    vols = df['vol_20bar'].dropna()
    if len(vols) > 30:
        p33, p67 = vols.quantile(0.33), vols.quantile(0.67)
        df['vol_regime'] = 'MID'
        df.loc[df['vol_20bar'] < p33, 'vol_regime'] = 'LOW'
        df.loc[df['vol_20bar'] > p67, 'vol_regime'] = 'HIGH'
    else:
        df['vol_regime'] = 'UNKNOWN'

round_trip_cost = 0.0010

def test_strategy(df, mask, name, label):
    """Test a strategy on a dataset. Returns results dict."""
    shifted = mask.shift(1).fillna(False)
    events = df[shifted]
    rets = df.loc[events.index, 'fwd_ret_16'].dropna()
    if len(rets) < 5:
        return {'name': name, 'label': label, 'n': len(rets), 'mean': None, 'p': None, 'wr': None, 'pass': False, 'note': 'too few events'}
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    gate = p < 0.1 and abs(mean_r) > round_trip_cost
    return {'name': name, 'label': label, 'n': len(rets), 'mean': mean_r, 'p': p, 'wr': wr, 'pass': gate}

# ═══════════════════════════════════════════════════════
# STRATEGY 1: liquidation_cascade
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 1: liquidation_cascade")
print("="*70)

# Detection: OI ROC < -0.01 + LS > 1.5
cascade_mask = (m['oi_roc'] < -0.01) & (m['ls_ratio'] > 1.5)
cascade_train_mask = (train['oi_roc'] < -0.01) & (train['ls_ratio'] > 1.5)
cascade_test_mask = (test['oi_roc'] < -0.01) & (test['ls_ratio'] > 1.5)

r_train = test_strategy(train, cascade_train_mask, 'cascade', 'TRAIN')
r_test = test_strategy(test, cascade_test_mask, 'cascade', 'TEST')

print(f"\n  TRAIN: n={r_train['n']}, mean={r_train['mean']*100:+.4f}%, p={r_train['p']:.4f}, WR={r_train['wr']:.1%}, pass={r_train['pass']}")
print(f"  TEST:  n={r_test['n']}, mean={r_test['mean']*100 if r_test['mean'] is not None else 0:+.4f}%, p={r_test['p']:.4f if r_test['p'] is not None else 'N/A'}, WR={r_test['wr']:.1% if r_test['wr'] is not None else 'N/A'}, pass={r_test['pass']}")

if r_train['mean'] is not None and r_test['mean'] is not None:
    drift = abs(r_train['mean'] - r_test['mean'])
    print(f"  Drift: {drift*100:.4f}% {'(STABLE)' if drift < 0.005 else '(UNSTABLE)'}")

# Also test with OI ROC < -0.015 (high conviction)
cascade_hc_train = (train['oi_roc'] < -0.015) & (train['ls_ratio'] > 1.5)
cascade_hc_test = (test['oi_roc'] < -0.015) & (test['ls_ratio'] > 1.5)

r_train_hc = test_strategy(train, cascade_hc_train, 'cascade_hc', 'TRAIN')
r_test_hc = test_strategy(test, cascade_hc_test, 'cascade_hc', 'TEST')

print(f"\n  HIGH CONVICTION (OI<-0.015 + LS>1.5):")
print(f"    TRAIN: n={r_train_hc['n']}, mean={r_train_hc['mean']*100 if r_train_hc['mean'] is not None else 0:+.4f}%, p={r_train_hc['p']:.4f if r_train_hc['p'] is not None else 'N/A'}, pass={r_train_hc['pass']}")
print(f"    TEST:  n={r_test_hc['n']}, mean={r_test_hc['mean']*100 if r_test_hc['mean'] is not None else 0:+.4f}%, p={r_test_hc['p']:.4f if r_test_hc['p'] is not None else 'N/A'}, pass={r_test_hc['pass']}")

# TP/SL simulation
def tp_sl_sim(df, mask, tp_pct=0.02, sl_pct=0.01, hold=16):
    shifted = mask.shift(1).fillna(False)
    events = df[shifted]
    trades = []
    for idx in events.index:
        entry = df.loc[idx, 'Close']
        won = None; bars = 0
        for h in range(1, hold + 1):
            if idx + h >= len(df): break
            high = df.loc[idx + h, 'High']
            low = df.loc[idx + h, 'Low']
            if high >= entry * (1 + sl_pct):
                won = False; bars = h; break
            if low <= entry * (1 - tp_pct):
                won = True; bars = h; break
        if won is None and idx + hold < len(df):
            close = df.loc[idx + hold, 'Close']
            won = close < entry; bars = hold
        if won is not None:
            trades.append({'won': won, 'bars': bars})
    if not trades: return None
    w = sum(1 for t in trades if t['won'])
    l = len(trades) - w
    wr = w / len(trades)
    pf = (w * tp_pct) / (l * sl_pct) if l > 0 else float('inf')
    pnl = w * tp_pct * 1000 - l * sl_pct * 1000
    return {'trades': len(trades), 'wr': wr, 'pf': pf, 'pnl': pnl}

print(f"\n  TP/SL (TP=2%, SL=1%, hold=4h):")
for label, df, mask in [('TRAIN', train, cascade_train_mask), ('TEST', test, cascade_test_mask)]:
    r = tp_sl_sim(df, mask)
    if r:
        print(f"    {label}: trades={r['trades']}, WR={r['wr']:.1%}, PF={r['pf']:.2f}, PnL=${r['pnl']:+.0f}")
    else:
        print(f"    {label}: no trades")

# ═══════════════════════════════════════════════════════
# STRATEGY 2: judas_sweep
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 2: judas_sweep (SHORT only)")
print("="*70)

# Detection: sweep daily/session high, close back below, wick > 1.5x body, vol > 1.0x
def detect_judas_sweep(df):
    highs = df['High'].values.astype(float)
    lows = df['Low'].values.astype(float)
    closes = df['Close'].values.astype(float)
    volumes = df['Volume'].values.astype(float)
    vol_ma = pd.Series(volumes).rolling(20).mean()
    vol_ok = pd.Series(volumes) >= vol_ma

    daily_highs = pd.Series(highs).rolling(96).max().shift(1)
    session_highs = pd.Series(highs).rolling(32).max().shift(1)

    cur_high = pd.Series(highs)
    cur_close = pd.Series(closes)
    prev_close = pd.Series(closes).shift(1)
    wick_up = cur_high - cur_close
    body = (cur_close - prev_close).abs().replace(0, 0.001)

    sweep_daily = (cur_high > daily_highs * 1.001) & (cur_close < daily_highs) & (wick_up > body * 1.5) & vol_ok
    sweep_session = (cur_high > session_highs * 1.001) & (cur_close < session_highs) & (wick_up > body * 1.5) & vol_ok
    return (sweep_daily | sweep_session)

judas_train_mask = detect_judas_sweep(train)
judas_test_mask = detect_judas_sweep(test)

r_train = test_strategy(train, judas_train_mask, 'judas_sweep', 'TRAIN')
r_test = test_strategy(test, judas_test_mask, 'judas_sweep', 'TEST')

print(f"\n  TRAIN: n={r_train['n']}, mean={r_train['mean']*100:+.4f}%, p={r_train['p']:.4f}, WR={r_train['wr']:.1%}, pass={r_train['pass']}")
print(f"  TEST:  n={r_test['n']}, mean={r_test['mean']*100 if r_test['mean'] is not None else 0:+.4f}%, p={r_test['p']:.4f if r_test['p'] is not None else 'N/A'}, WR={r_test['wr']:.1% if r_test['wr'] is not None else 'N/A'}, pass={r_test['pass']}")

if r_train['mean'] is not None and r_test['mean'] is not None:
    drift = abs(r_train['mean'] - r_test['mean'])
    print(f"  Drift: {drift*100:.4f}% {'(STABLE)' if drift < 0.005 else '(UNSTABLE)'}")

print(f"\n  TP/SL (TP=2.5%, SL=1.5%, hold=4h):")
for label, df, mask in [('TRAIN', train, judas_train_mask), ('TEST', test, judas_test_mask)]:
    r = tp_sl_sim(df, mask, tp_pct=0.025, sl_pct=0.015)
    if r:
        print(f"    {label}: trades={r['trades']}, WR={r['wr']:.1%}, PF={r['pf']:.2f}, PnL=${r['pnl']:+.0f}")
    else:
        print(f"    {label}: no trades")

# ═══════════════════════════════════════════════════════
# STRATEGY 3: funding_squeeze
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 3: funding_squeeze (SHORT)")
print("="*70)

# Detection: FR z-score > 1.25
fr_train_mask = train['fr_z'] > 1.25
fr_test_mask = test['fr_z'] > 1.25

r_train = test_strategy(train, fr_train_mask, 'funding_squeeze', 'TRAIN')
r_test = test_strategy(test, fr_test_mask, 'funding_squeeze', 'TEST')

print(f"\n  TRAIN: n={r_train['n']}, mean={r_train['mean']*100 if r_train['mean'] is not None else 0:+.4f}%, p={r_train['p']:.4f if r_train['p'] is not None else 'N/A'}, WR={r_train['wr']:.1% if r_train['wr'] is not None else 'N/A'}, pass={r_train['pass']}")
print(f"  TEST:  n={r_test['n']}, mean={r_test['mean']*100 if r_test['mean'] is not None else 0:+.4f}%, p={r_test['p']:.4f if r_test['p'] is not None else 'N/A'}, WR={r_test['wr']:.1% if r_test['wr'] is not None else 'N/A'}, pass={r_test['pass']}")

if r_train['mean'] is not None and r_test['mean'] is not None:
    drift = abs(r_train['mean'] - r_test['mean'])
    print(f"  Drift: {drift*100:.4f}% {'(STABLE)' if drift < 0.005 else '(UNSTABLE)'}")

# Also test with higher thresholds
for thresh in [1.25, 1.5, 1.75]:
    train_m = train['fr_z'] > thresh
    test_m = test['fr_z'] > thresh
    r_tr = test_strategy(train, train_m, f'funding_z>{thresh}', 'TRAIN')
    r_te = test_strategy(test, test_m, f'funding_z>{thresh}', 'TEST')
    if r_tr['mean'] is not None and r_te['mean'] is not None:
        drift = abs(r_tr['mean'] - r_te['mean'])
        print(f"\n  z>{thresh}: TRAIN n={r_tr['n']} mean={r_tr['mean']*100:+.4f}% p={r_tr['p']:.4f} | TEST n={r_te['n']} mean={r_te['mean']*100:+.4f}% p={r_te['p']:.4f} | drift={drift*100:.4f}% {'STABLE' if drift < 0.005 else 'UNSTABLE'}")
    elif r_tr['mean'] is not None:
        print(f"\n  z>{thresh}: TRAIN n={r_tr['n']} mean={r_tr['mean']*100:+.4f}% p={r_tr['p']:.4f} | TEST: too few events")

print(f"\n  TP/SL (TP=2%, SL=1%, hold=4h):")
for label, df, mask in [('TRAIN', train, fr_train_mask), ('TEST', test, fr_test_mask)]:
    r = tp_sl_sim(df, mask, tp_pct=0.02, sl_pct=0.01)
    if r:
        print(f"    {label}: trades={r['trades']}, WR={r['wr']:.1%}, PF={r['pf']:.2f}, PnL=${r['pnl']:+.0f}")
    else:
        print(f"    {label}: no trades")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("PF drift < 0.5 = stable, >= 0.5 = unstable")
print("Both train AND test must pass for deployment confidence")
