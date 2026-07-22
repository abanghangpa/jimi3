"""Validation backtest for liquidation_cascade v4 config."""
import pandas as pd
import numpy as np
from scipy import stats

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'

# Load data
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)

deriv = pd.read_csv(f'{DATA_DIR}/derivatives_history/derivatives_backfilled.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'])
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

# Merge
merged = pd.merge_asof(ohlcv, deriv[['timestamp', 'oi', 'ls_ratio']], on='timestamp', direction='backward', tolerance=pd.Timedelta('2h'))
merged['oi_roc_1h'] = merged['oi'].pct_change(4, fill_method=None)
merged['vol_20bar'] = merged['Close'].pct_change().rolling(20).std()

# Regime
vols = merged['vol_20bar'].dropna()
p33 = vols.quantile(0.33)
p67 = vols.quantile(0.67)
merged['vol_regime'] = 'MID'
merged.loc[merged['vol_20bar'] < p33, 'vol_regime'] = 'LOW'
merged.loc[merged['vol_20bar'] > p67, 'vol_regime'] = 'HIGH'

# Forward returns
merged['fwd_ret_16'] = merged['Close'].shift(-16) / merged['Close'] - 1

# Detection masks
mask_full = (
    (merged['oi_roc_1h'] < -0.015) &
    (merged['ls_ratio'] > 1.5) &
    (merged['vol_regime'] == 'MID')
).shift(1).fillna(False)

mask_no_ls = (
    (merged['oi_roc_1h'] < -0.015) &
    (merged['vol_regime'] == 'MID')
).shift(1).fillna(False)

# Isolation gate results
print("=" * 60)
print("ISOLATION GATE — Forward Returns (4h / 16-bar)")
print("=" * 60)

for label, mask in [("OI<-0.015 + LS>1.5 + MID vol", mask_full),
                     ("OI<-0.015 + MID vol (no LS)", mask_no_ls)]:
    events = merged[mask]
    rets = merged.loc[events.index, 'fwd_ret_16'].dropna()
    if len(rets) < 5:
        print(f"\n{label}: Too few events ({len(rets)})")
        continue
    mean_r = rets.mean()
    wins = (rets > 0).sum()
    wr = wins / len(rets)
    t, p = stats.ttest_1samp(rets, 0)
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    print(f"\n{label}:")
    print(f"  Events: {len(rets)}")
    print(f"  Mean return: {mean_r*100:+.4f}%")
    print(f"  Win rate: {wr:.1%}")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  p-value: {p:.4f}")
    print(f"  Gate: {'PASS' if p < 0.1 and mean_r > 0.001 else 'FAIL'}")

# TP/SL simulation
print(f"\n{'=' * 60}")
print("TP/SL SIMULATION — TP=2.0%, SL=1.0%, Hold=4h (16 bars)")
print("=" * 60)

tp_pct = 0.02
sl_pct = 0.01
hold_bars = 16

for label, mask in [("Full (OI<-0.015 + LS>1.5 + MID)", mask_full),
                     ("OI<-0.015 + MID (no LS)", mask_no_ls)]:
    events = merged[mask]
    trades = []
    for idx in events.index:
        entry = merged.loc[idx, 'Close']
        won = None
        bars_held = 0
        for h in range(1, hold_bars + 1):
            if idx + h >= len(merged):
                break
            high = merged.loc[idx + h, 'High']
            low = merged.loc[idx + h, 'Low']
            # SHORT: SL is above entry, TP is below
            if high >= entry * (1 + sl_pct):
                won = False
                bars_held = h
                break
            if low <= entry * (1 - tp_pct):
                won = True
                bars_held = h
                break
        if won is None:
            if idx + hold_bars < len(merged):
                close = merged.loc[idx + hold_bars, 'Close']
                won = close < entry
                bars_held = hold_bars
            else:
                continue
        trades.append({'won': won, 'bars': bars_held, 'entry': entry})

    if not trades:
        print(f"\n{label}: No trades")
        continue

    wins = sum(1 for t in trades if t['won'])
    losses = len(trades) - wins
    wr = wins / len(trades)
    avg_bars = np.mean([t['bars'] for t in trades])
    pnl_per_win = tp_pct * 1000
    pnl_per_loss = sl_pct * 1000
    total_pnl = wins * pnl_per_win - losses * pnl_per_loss
    pf = (wins * tp_pct) / (losses * sl_pct) if losses > 0 else float('inf') if wins > 0 else 0

    print(f"\n{label}:")
    print(f"  Trades: {len(trades)}")
    print(f"  Win rate: {wr:.1%} ({wins}W / {losses}L)")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Avg hold: {avg_bars:.1f} bars ({avg_bars*15:.0f} min)")
    print(f"  Total PnL ($1000/trade): ${total_pnl:+.0f}")
    print(f"  Avg PnL/trade: ${total_pnl/len(trades):+.2f}")
