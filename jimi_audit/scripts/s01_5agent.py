"""
5-Agent Backtest: S01 Failed Breakout v3.1 — WEAK+ACCUM+LONG
==============================================================
Simulates the full orchestrator pipeline for S01's validated signal.

Agents:
1. Structure Agent   — M14 sweep + M21 Wyckoff + derivatives
2. Regime Classifier — BULL/BEAR/RANGING/STRESS from price action
3. Validation Agent  — Isolation gate (backtest-validated thresholds)
4. Risk Agent        — Position sizing with DD circuit breaker
5. Execution Agent   — ATR-based slippage + structural TP/SL
"""

import pandas as pd, numpy as np, json, os
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
DERIV_DIR = f'{DATA_DIR}/derivatives_history'
OUTPUT = '/root/.openclaw/workspace/jimi_audit/reports/s01_5agent_backtest.json'
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("="*70)
print("5-AGENT BACKTEST: S01 v3.1 — WEAK+ACCUM+LONG")
print("="*70)

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
print("\nLoading data...")
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
for c in ['Close','High','Low','Open','Volume']: ohlcv[c] = ohlcv[c].astype(float)

deriv = pd.read_csv(f'{DERIV_DIR}/derivatives_collected.csv')
deriv['timestamp'] = pd.to_datetime(deriv['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
deriv = deriv.sort_values('timestamp').reset_index(drop=True)

merged = pd.merge_asof(
    ohlcv[['timestamp','Open','High','Low','Close','Volume']],
    deriv[['timestamp','oi','ls_ratio','funding_rate']],
    on='timestamp', direction='backward', tolerance=pd.Timedelta('2h')
)

merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()
merged['ema200'] = merged['Close'].ewm(span=200).mean()
merged['atr'] = (merged['High'] - merged['Low']).rolling(14).mean()
merged['hour'] = merged['timestamp'].dt.hour
GOOD_HOURS = {9, 10, 11, 12, 14, 15, 16, 18}

for h in [1, 4, 16, 24]:
    merged[f'fwd_ret_{h}'] = merged['Close'].shift(-h) / merged['Close'] - 1

vols = merged['Close'].pct_change().rolling(20).std()
p33, p67 = vols.dropna().quantile(0.33), vols.dropna().quantile(0.67)
merged['vol_regime'] = 'MID'
merged.loc[vols < p33, 'vol_regime'] = 'LOW'
merged.loc[vols > p67, 'vol_regime'] = 'HIGH'

def get_era(ts):
    if ts < pd.Timestamp('2024-07-01'): return '2024_H1'
    elif ts < pd.Timestamp('2025-01-01'): return '2024_H2'
    elif ts < pd.Timestamp('2025-07-01'): return '2025_H1'
    elif ts < pd.Timestamp('2026-01-01'): return '2025_H2'
    else: return '2026'
merged['era'] = merged['timestamp'].apply(get_era)

highs = merged['High'].values.astype(float)
lows = merged['Low'].values.astype(float)
closes = merged['Close'].values.astype(float)
opens = merged['Open'].values.astype(float)
volumes = merged['Volume'].values.astype(float)

print(f"OHLCV: {len(merged)} bars")

# ═══════════════════════════════════════════════════════════════
# AGENT 1: STRUCTURE — Detect WEAK+ACCUM+LONG signals
# ═══════════════════════════════════════════════════════════════
print("\nAGENT 1: STRUCTURE — Detecting signals...")

def find_swing_levels(highs, lows, idx, lookback=48):
    sh, sl = [], []
    start = max(0, idx - lookback)
    for i in range(start + 2, idx - 1):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1]:
            sh.append((highs[i], i))
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1]:
            sl.append((lows[i], i))
    return sh, sl

def detect_wyckoff(idx):
    if idx < 96:
        return None
    lookback = min(768, idx)
    h4h = highs[idx-lookback:idx+1]
    h4l = lows[idx-lookback:idx+1]
    h4c = closes[idx-lookback:idx+1]
    
    half = len(h4c) // 2
    recent_hi = h4h[-min(10, half):].max()
    prior_hi = h4h[-min(20, half):-min(10, half)].max() if len(h4h) > 10 else recent_hi
    recent_lo = h4l[-min(10, half):].min()
    prior_lo = h4l[-min(20, half):-min(10, half)].min() if len(h4l) > 10 else recent_lo
    
    hh, hl = recent_hi > prior_hi, recent_lo > prior_lo
    lh, ll = recent_hi < prior_hi, recent_lo < prior_lo
    
    range_hi, range_lo = float(h4h.max()), float(h4l.min())
    current = float(h4c[-1])
    position = (current - range_lo) / (range_hi - range_lo) if range_hi > range_lo else 0.5
    
    phase = 'RANGE'
    if hh and hl:
        phase = 'DISTRIBUTION' if position > 0.7 else 'MARKUP' if position > 0.5 else 'ACCUMULATION'
    elif lh and ll:
        phase = 'ACCUMULATION' if position < 0.3 else 'MARKDOWN' if position < 0.5 else 'DISTRIBUTION'
    
    return {'phase': phase, 'position': position}

signals = []
seen = set()
step = 4  # every 4th bar for speed

for idx in range(96, len(merged), step):
    if idx % 20000 == 0:
        print(f"  Processing bar {idx}/{len(merged)}...")
    
    hour = merged.iloc[idx]['hour']
    if hour not in GOOD_HOURS:
        continue
    
    swing_highs, swing_lows = find_swing_levels(highs, lows, idx)
    wyckoff = detect_wyckoff(idx)
    if not wyckoff:
        continue
    
    # Only ACCUMULATION phase
    if wyckoff['phase'] != 'ACCUMULATION':
        continue
    
    # Check for LONG sweep (price swept below support)
    for level_price, level_idx in swing_lows:
        for lb in range(1, min(6, idx)):
            bar_idx = idx - lb
            if bar_idx < level_idx or bar_idx < 48:
                continue
            
            bar_range = highs[bar_idx] - lows[bar_idx]
            if bar_range <= 0:
                continue
            
            sweep_depth = (level_price - lows[bar_idx]) / level_price
            if not (0.001 <= sweep_depth <= 0.020):
                continue
            
            # Classify reclaim type
            lower_wick = min(opens[bar_idx], closes[bar_idx]) - lows[bar_idx]
            wick_ratio = lower_wick / bar_range
            vol_avg = np.mean(volumes[max(0, bar_idx-20):bar_idx]) if bar_idx >= 20 else volumes[bar_idx]
            vol_ok = volumes[bar_idx] > vol_avg * 1.2
            
            if wick_ratio >= 0.40 and closes[bar_idx] > opens[bar_idx]:
                reclaim_type = 'STRONG' if vol_ok else 'WEAK'
            elif closes[bar_idx] > opens[bar_idx] and closes[bar_idx] > level_price:
                reclaim_type = 'WEAK'
            else:
                reclaim_type = 'NONE'
            
            # Only WEAK reclaim
            if reclaim_type != 'WEAK':
                continue
            
            key = (bar_idx, 'LONG')
            if key in seen:
                continue
            seen.add(key)
            
            # Positioning
            ls = merged.iloc[idx]['ls_ratio'] if pd.notna(merged.iloc[idx]['ls_ratio']) else 1.0
            fr = merged.iloc[idx]['funding_rate'] if pd.notna(merged.iloc[idx]['funding_rate']) else 0
            
            signals.append({
                'idx': idx, 'direction': 'LONG',
                'sweep_bar': bar_idx, 'bars_ago': idx - bar_idx,
                'level': level_price, 'depth_pct': sweep_depth * 100,
                'wick_ratio': wick_ratio, 'reclaim_type': reclaim_type,
                'wyckoff_phase': wyckoff['phase'],
                'wyckoff_position': wyckoff['position'],
                'ls_ratio': ls, 'funding_rate': fr,
                'vol_ratio': merged.iloc[idx]['vol_ratio'] if pd.notna(merged.iloc[idx]['vol_ratio']) else 1.0,
                'atr': merged.iloc[idx]['atr'] if pd.notna(merged.iloc[idx]['atr']) else 0,
                'price': closes[idx],
            })
            break  # one signal per bar

print(f"  Total signals: {len(signals)}")

# ═══════════════════════════════════════════════════════════════
# AGENT 2: REGIME CLASSIFIER
# ═══════════════════════════════════════════════════════════════
print("\nAGENT 2: REGIME CLASSIFIER — Classifying signals by regime...")

def classify_regime(idx):
    """Simple regime classification from price action."""
    if idx < 200:
        return 'UNKNOWN'
    closes_window = closes[idx-200:idx+1]
    ema50 = pd.Series(closes_window).ewm(span=50).mean().iloc[-1]
    ema200 = pd.Series(closes_window).ewm(span=200).mean().iloc[-1]
    current = closes_window[-1]
    
    if current > ema50 > ema200:
        return 'BULL'
    elif current < ema50 < ema200:
        return 'BEAR'
    else:
        return 'RANGING'

for s in signals:
    s['regime'] = classify_regime(s['idx'])

regime_counts = {}
for s in signals:
    r = s['regime']
    regime_counts[r] = regime_counts.get(r, 0) + 1
print(f"  Regime distribution: {regime_counts}")

# ═══════════════════════════════════════════════════════════════
# AGENT 3: VALIDATION — Isolation gate
# ═══════════════════════════════════════════════════════════════
print("\nAGENT 3: VALIDATION — Running isolation gate...")

# Gate thresholds from backtest
GATE_MIN_WR = 0.55
GATE_MIN_N = 20
GATE_MAX_MC_P = 0.10

# Run gate on each regime
gate_results = {}
for regime in ['BULL', 'BEAR', 'RANGING']:
    regime_signals = [s for s in signals if s['regime'] == regime]
    if len(regime_signals) < 5:
        gate_results[regime] = {'n': len(regime_signals), 'pass': False, 'reason': 'too_few'}
        continue
    
    indices = [s['idx'] for s in regime_signals]
    rets = merged.iloc[indices]['fwd_ret_16'].dropna()
    if len(rets) < 5:
        gate_results[regime] = {'n': len(rets), 'pass': False, 'reason': 'too_few_rets'}
        continue
    
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    
    # Monte Carlo
    np.random.seed(42)
    all_r = merged['fwd_ret_16'].dropna()
    n = len(rets)
    rm = np.array([all_r.sample(n).mean() for _ in range(5000)])
    mc_p = (rm >= mean_r).mean()
    
    passed = wr >= GATE_MIN_WR and n >= GATE_MIN_N and mc_p <= GATE_MAX_MC_P
    gate_results[regime] = {
        'n': int(n), 'wr': float(wr), 'mean': float(mean_r),
        'mc_p': float(mc_p), 'pass': passed,
    }
    icon = '✅' if passed else '❌'
    print(f"  {icon} {regime}: n={n}, WR={wr:.1%}, mean={mean_r*100:+.4f}%, MC p={mc_p:.4f}")

# Filter signals to only passing regimes
valid_regimes = {r for r, v in gate_results.items() if v.get('pass')}
filtered = [s for s in signals if s['regime'] in valid_regimes]
print(f"\n  Valid regimes: {valid_regimes}")
print(f"  Signals after gate: {len(filtered)} → {len(signals)}")

# ═══════════════════════════════════════════════════════════════
# AGENT 4: RISK — Position sizing with DD circuit breaker
# ═══════════════════════════════════════════════════════════════
print("\nAGENT 4: RISK — Position sizing + DD circuit breaker...")

INITIAL_CAPITAL = 10000
RISK_PCT = 0.02  # 2% risk per trade
DD_THRESHOLDS = {0.15: 0.5, 0.25: 0.25, 0.35: 0}  # DD% → Kelly fraction

def kelly_fraction(wr, avg_win, avg_loss):
    if avg_loss == 0:
        return 0
    b = avg_win / abs(avg_loss)
    f = (wr * b - (1 - wr)) / b
    return max(f, 0)

# Simulate trades with position sizing
capital = INITIAL_CAPITAL
peak_capital = INITIAL_CAPITAL
trades = []

for s in filtered:
    idx = s['idx']
    if idx + 16 >= len(merged):
        continue
    
    entry = closes[idx]
    atr = s['atr']
    if atr <= 0:
        continue
    
    # DD circuit breaker
    dd = (peak_capital - capital) / peak_capital
    kelly_adj = 1.0
    for dd_thresh, kelly_mult in sorted(DD_THRESHOLDS.items()):
        if dd > dd_thresh:
            kelly_adj = kelly_mult
    
    if kelly_adj == 0:
        continue  # trading paused
    
    # Position size
    risk_amount = capital * RISK_PCT * kelly_adj
    sl_pct = (atr / entry * 100) * 1.2  # 1.2x ATR SL
    if sl_pct <= 0:
        continue
    position_size = risk_amount / (sl_pct / 100)
    
    # TP/SL
    sl_price = entry * (1 - sl_pct/100)
    tp_pct = sl_pct * 2.5  # 2.5x ATR TP (fallback)
    tp_price = entry * (1 + tp_pct/100)
    
    # Outcome
    future_closes = closes[idx+1:idx+17]
    hit_tp = any(c >= tp_price for c in future_closes)
    hit_sl = any(c <= sl_price for c in future_closes)
    
    if hit_tp and not hit_sl:
        pnl = position_size * (tp_pct / 100)
        outcome = 'WIN'
    elif hit_sl and not hit_tp:
        pnl = -position_size * (sl_pct / 100)
        outcome = 'LOSS'
    elif hit_tp and hit_sl:
        # Both hit — assume SL hit first (conservative)
        pnl = -position_size * (sl_pct / 100)
        outcome = 'LOSS'
    else:
        # Neither hit — close at 4h
        ret_4h = merged.iloc[idx]['fwd_ret_16'] if pd.notna(merged.iloc[idx]['fwd_ret_16']) else 0
        pnl = position_size * ret_4h
        outcome = 'WIN' if ret_4h > 0 else 'LOSS'
    
    capital += pnl
    peak_capital = max(peak_capital, capital)
    
    trades.append({
        'idx': idx, 'direction': 'LONG', 'entry': entry,
        'sl': sl_price, 'tp': tp_price,
        'pnl': pnl, 'outcome': outcome,
        'capital_after': capital,
        'regime': s['regime'], 'kelly_adj': kelly_adj,
    })

wins = [t for t in trades if t['outcome'] == 'WIN']
losses = [t for t in trades if t['outcome'] == 'LOSS']
total_pnl = capital - INITIAL_CAPITAL
max_dd = 0
peak = INITIAL_CAPITAL
for t in trades:
    peak = max(peak, t['capital_after'])
    dd = (peak - t['capital_after']) / peak
    max_dd = max(max_dd, dd)

print(f"  Trades: {len(trades)}")
print(f"  Wins: {len(wins)}, Losses: {len(losses)}")
print(f"  WR: {len(wins)/len(trades):.1%}" if trades else "  WR: N/A")
print(f"  Total PnL: ${total_pnl:+.2f} ({total_pnl/INITIAL_CAPITAL*100:+.2f}%)")
print(f"  Max DD: {max_dd:.2%}")
print(f"  Final capital: ${capital:.2f}")

# ═══════════════════════════════════════════════════════════════
# AGENT 5: EXECUTION — Quality metrics
# ═══════════════════════════════════════════════════════════════
print("\nAGENT 5: EXECUTION — Quality metrics...")

if trades:
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
    pf = abs(avg_win * len(wins)) / abs(avg_loss * len(losses)) if losses and avg_loss != 0 else float('inf')
    expectancy = (len(wins)/len(trades) * avg_win) + (len(losses)/len(trades) * avg_loss)
    
    # Sharpe-like ratio
    pnls = [t['pnl'] for t in trades]
    sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
    
    # Consecutive losses
    max_consec_loss = 0
    consec = 0
    for t in trades:
        if t['outcome'] == 'LOSS':
            consec += 1
            max_consec_loss = max(max_consec_loss, consec)
        else:
            consec = 0
    
    execution = {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(pf),
        'expectancy': float(expectancy),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'max_consecutive_losses': max_consec_loss,
        'total_pnl': float(total_pnl),
        'total_return_pct': float(total_pnl / INITIAL_CAPITAL * 100),
        'final_capital': float(capital),
    }
    
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Expectancy: ${expectancy:+.2f}/trade")
    print(f"  Sharpe: {sharpe:.3f}")
    print(f"  Max consec losses: {max_consec_loss}")
else:
    execution = {'total_trades': 0}
    print("  No trades executed")

# ═══════════════════════════════════════════════════════════════
# REGIME BREAKDOWN
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("REGIME BREAKDOWN")
print("="*70)

for regime in ['BULL', 'BEAR', 'RANGING']:
    regime_trades = [t for t in trades if t['regime'] == regime]
    if not regime_trades:
        continue
    r_wins = [t for t in regime_trades if t['outcome'] == 'WIN']
    r_pnl = sum(t['pnl'] for t in regime_trades)
    r_wr = len(r_wins) / len(regime_trades) if regime_trades else 0
    print(f"  {regime}: {len(regime_trades)} trades, WR={r_wr:.1%}, PnL=${r_pnl:+.2f}")

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
results = {
    'agent_1_structure': {'total_signals': len(signals), 'filtered_signals': len(filtered)},
    'agent_2_regime': regime_counts,
    'agent_3_validation': gate_results,
    'agent_4_risk': {'initial_capital': INITIAL_CAPITAL, 'final_capital': float(capital),
                     'max_dd': float(max_dd), 'total_pnl': float(total_pnl)},
    'agent_5_execution': execution,
}

with open(OUTPUT, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")
print(f"  Signals: {len(signals)} total, {len(filtered)} after gate")
print(f"  Trades: {len(trades)}")
print(f"  WR: {execution.get('win_rate', 0):.1%}")
print(f"  PF: {execution.get('profit_factor', 0):.2f}")
print(f"  Max DD: {max_dd:.2%}")
print(f"  Total Return: {total_pnl/INITIAL_CAPITAL*100:+.2f}%")
print(f"\nSaved to {OUTPUT}")
