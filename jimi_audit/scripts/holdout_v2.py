
import pandas as pd, numpy as np
from scipy import stats

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'

d = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
d['timestamp'] = pd.to_datetime(d['Open time'])
d = d.sort_values('timestamp').reset_index(drop=True)

deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

m = pd.merge_asof(d, deriv[['timestamp','oi','ls_ratio','funding_rate']],
                   on='timestamp', direction='backward', tolerance=pd.Timedelta('2h'))

# Only use rows that have derivatives data
m = m[m['oi'].notna()].reset_index(drop=True)
print(f"Rows with derivatives: {len(m)}")
print(f"Date range: {m['timestamp'].min()} to {m['timestamp'].max()}")

# Features
m['oi_roc'] = m['oi'].pct_change(4, fill_method=None)
m['vol_20bar'] = m['Close'].pct_change().rolling(20).std()
m['ema200'] = m['Close'].ewm(span=200).mean()
m['vol_ma20'] = m['Volume'].rolling(20).mean()
m['vol_ratio'] = m['Volume'] / m['vol_ma20']
m['trend'] = np.where(m['Close'] > m['ema200'], 'BULL', 'BEAR')
m['fr'] = m['funding_rate']
m['fr_z'] = (m['fr'] - m['fr'].rolling(96).mean()) / m['fr'].rolling(96).std()

for h in [1, 4, 16, 24]:
    m[f'fwd_ret_{h}'] = m['Close'].shift(-h) / m['Close'] - 1

# Train/test split (70/30 by time)
n = len(m)
split_idx = int(n * 0.7)
train = m.iloc[:split_idx].copy()
test = m.iloc[split_idx:].copy()
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

round_trip_cost = 0.0010

def fmt(r):
    if r['mean'] is None:
        return f"n={r['n']} (too few)"
    return f"n={r['n']}, mean={r['mean']*100:+.4f}%, p={r['p']:.4f}, WR={r['wr']:.1%}, pass={r['pass']}"

def test_strat(df, mask, name, label):
    shifted = mask.shift(1).fillna(False)
    events = df[shifted]
    rets = df.loc[events.index, 'fwd_ret_16'].dropna()
    if len(rets) < 5:
        return {'name': name, 'label': label, 'n': len(rets), 'mean': None, 'p': None, 'wr': None, 'pass': False}
    mean_r = rets.mean()
    t, p = stats.ttest_1samp(rets, 0)
    wr = (rets > 0).mean()
    gate = p < 0.1 and abs(mean_r) > round_trip_cost
    return {'name': name, 'label': label, 'n': len(rets), 'mean': mean_r, 'p': p, 'wr': wr, 'pass': gate}

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

def sim_fmt(r):
    if r is None: return "no trades"
    return f"trades={r['trades']}, WR={r['wr']:.1%}, PF={r['pf']:.2f}, PnL=${r['pnl']:+.0f}"

# ═══ CASCADE ═══
print("\n" + "="*70)
print("1. liquidation_cascade (OI<-0.01 + LS>1.5 SHORT)")
print("="*70)

cm_train = (train['oi_roc'] < -0.01) & (train['ls_ratio'] > 1.5)
cm_test = (test['oi_roc'] < -0.01) & (test['ls_ratio'] > 1.5)
r_tr = test_strat(train, cm_train, 'cascade', 'TRAIN')
r_te = test_strat(test, cm_test, 'cascade', 'TEST')
print(f"  TRAIN: {fmt(r_tr)}")
print(f"  TEST:  {fmt(r_te)}")
if r_tr['mean'] is not None and r_te['mean'] is not None:
    drift = abs(r_tr['mean'] - r_te['mean'])
    print(f"  Drift: {drift*100:.4f}% {'STABLE' if drift < 0.005 else 'UNSTABLE'}")
print(f"  TP/SL TRAIN: {sim_fmt(tp_sl_sim(train, cm_train))}")
print(f"  TP/SL TEST:  {sim_fmt(tp_sl_sim(test, cm_test))}")

# High conviction
cm_hc_train = (train['oi_roc'] < -0.015) & (train['ls_ratio'] > 1.5)
cm_hc_test = (test['oi_roc'] < -0.015) & (test['ls_ratio'] > 1.5)
r_tr_hc = test_strat(train, cm_hc_train, 'cascade_hc', 'TRAIN')
r_te_hc = test_strat(test, cm_hc_test, 'cascade_hc', 'TEST')
print(f"\n  HIGH CONVICTION (OI<-0.015 + LS>1.5):")
print(f"    TRAIN: {fmt(r_tr_hc)}")
print(f"    TEST:  {fmt(r_te_hc)}")
if r_tr_hc['mean'] is not None and r_te_hc['mean'] is not None:
    drift = abs(r_tr_hc['mean'] - r_te_hc['mean'])
    print(f"    Drift: {drift*100:.4f}% {'STABLE' if drift < 0.005 else 'UNSTABLE'}")

# ═══ JUDAS SWEEP ═══
print("\n" + "="*70)
print("2. judas_sweep SHORT (sweep high, wick>1.5x, vol>1.0)")
print("="*70)

def detect_judas(df):
    highs = df['High'].values.astype(float)
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
    sweep_d = (cur_high > daily_highs * 1.001) & (cur_close < daily_highs) & (wick_up > body * 1.5) & vol_ok
    sweep_s = (cur_high > session_highs * 1.001) & (cur_close < session_highs) & (wick_up > body * 1.5) & vol_ok
    return sweep_d | sweep_s

jm_train = detect_judas(train)
jm_test = detect_judas(test)
r_tr = test_strat(train, jm_train, 'judas', 'TRAIN')
r_te = test_strat(test, jm_test, 'judas', 'TEST')
print(f"  TRAIN: {fmt(r_tr)}")
print(f"  TEST:  {fmt(r_te)}")
if r_tr['mean'] is not None and r_te['mean'] is not None:
    drift = abs(r_tr['mean'] - r_te['mean'])
    print(f"  Drift: {drift*100:.4f}% {'STABLE' if drift < 0.005 else 'UNSTABLE'}")
print(f"  TP/SL TRAIN: {sim_fmt(tp_sl_sim(train, jm_train, tp_pct=0.025, sl_pct=0.015))}")
print(f"  TP/SL TEST:  {sim_fmt(tp_sl_sim(test, jm_test, tp_pct=0.025, sl_pct=0.015))}")

# ═══ FUNDING SQUEEZE ═══
print("\n" + "="*70)
print("3. funding_squeeze SHORT (FR z-score)")
print("="*70)

for thresh in [1.0, 1.25, 1.5, 1.75]:
    fm_train = train['fr_z'] > thresh
    fm_test = test['fr_z'] > thresh
    r_tr = test_strat(train, fm_train, f'frz>{thresh}', 'TRAIN')
    r_te = test_strat(test, fm_test, f'frz>{thresh}', 'TEST')
    if r_tr['mean'] is None and r_te['mean'] is None:
        print(f"  z>{thresh}: both too few events")
        continue
    tr_s = fmt(r_tr)
    te_s = fmt(r_te)
    drift_s = ""
    if r_tr['mean'] is not None and r_te['mean'] is not None:
        drift = abs(r_tr['mean'] - r_te['mean'])
        drift_s = f" drift={drift*100:.4f}% {'STABLE' if drift < 0.005 else 'UNSTABLE'}"
    print(f"  z>{thresh}: TRAIN {tr_s} | TEST {te_s}{drift_s}")

# TP/SL for z>1.25
fm_train = train['fr_z'] > 1.25
fm_test = test['fr_z'] > 1.25
print(f"\n  TP/SL z>1.25 (TP=2%, SL=1%, hold=4h):")
print(f"    TRAIN: {sim_fmt(tp_sl_sim(train, fm_train))}")
print(f"    TEST:  {sim_fmt(tp_sl_sim(test, fm_test))}")

# ═══ SUMMARY ═══
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("PF drift < 0.5 = stable, >= 0.5 = unstable")
print("Both train AND test must pass for deployment confidence")
