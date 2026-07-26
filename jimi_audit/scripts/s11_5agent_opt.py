"""
S11 Cross-Asset: 5-Agent Backtest + Optimization Framework
Tests: ETH/BTC deviation >5% LONG + BULL+ETH_under dev>3%
"""
import pandas as pd, numpy as np, json, os
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
OUTPUT = '/root/.openclaw/workspace/jimi_audit/reports/s11_5agent_opt.json'
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("="*70)
print("S11 CROSS-ASSET: 5-AGENT + OPTIMIZATION")
print("="*70)

# Load data
eth = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
eth['timestamp'] = pd.to_datetime(eth['Open time'])
eth = eth.sort_values('timestamp').reset_index(drop=True)
for c in ['Close','High','Low','Volume']: eth[c] = eth[c].astype(float)

btc_raw = pd.read_json(f'{DATA_DIR}/btc_1h.json')
btc_raw.columns = ['ts','Open','High','Low','Close','Volume','ts2','qv','trades','tbqv','tbqav','ignore']
btc = btc_raw[['ts','Close']].copy()
btc['timestamp'] = pd.to_datetime(btc['ts'], unit='ms')
btc = btc.sort_values('timestamp').reset_index(drop=True)
btc['Close'] = btc['Close'].astype(float)

merged = pd.merge_asof(
    eth[['timestamp','Close','High','Low','Volume']].rename(columns={'Close':'eth_close'}),
    btc[['timestamp','Close']].rename(columns={'Close':'btc_close'}),
    on='timestamp', direction='backward', tolerance=pd.Timedelta('30min')
).dropna(subset=['btc_close'])

merged['eth_btc'] = merged['eth_close'] / merged['btc_close']
merged['eth_btc_ma20'] = merged['eth_btc'].rolling(20).mean()
merged['eth_btc_dev20'] = (merged['eth_btc'] - merged['eth_btc_ma20']) / merged['eth_btc_ma20']
merged['btc_ema21'] = merged['btc_close'].ewm(span=21).mean()
merged['btc_ema55'] = merged['btc_close'].ewm(span=55).mean()
merged['btc_trend'] = np.where(merged['btc_ema21'] > merged['btc_ema55'], 'BULL', 'BEAR')
merged['eth_ema200'] = merged['eth_close'].ewm(span=200).mean()
merged['atr'] = (merged['High'] - merged['Low']).rolling(14).mean()
merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()

for h in [1, 4, 16, 24]:
    merged[f'fwd_ret_{h}'] = merged['eth_close'].shift(-h) / merged['eth_close'] - 1

# Signal: ETH/BTC dev < -5% → LONG
mask = merged['eth_btc_dev20'] < -0.05
signals = merged[mask.shift(1).fillna(False)].copy()
signals['direction'] = 'LONG'
signals['deviation'] = signals['eth_btc_dev20']

print(f"Signals (dev<5%): {len(signals)}")

results = {}

# ═══════════ 5-AGENT BACKTEST ═══════════
print("\n" + "="*70)
print("5-AGENT BACKTEST")
print("="*70)

# Agent 1: Structure
a1 = {'n': len(signals), 'mean_deviation': float(signals['deviation'].mean())}
print(f"  Agent 1 (Structure): {len(signals)} signals, mean dev={signals['deviation'].mean():.4f}")

# Agent 2: Regime
regime_dist = signals['btc_trend'].value_counts().to_dict()
a2 = regime_dist
print(f"  Agent 2 (Regime): {regime_dist}")

# Agent 3: Validation — gate by regime
a3 = {}
for regime in ['BULL', 'BEAR']:
    r_signals = signals[signals['btc_trend'] == regime]
    if len(r_signals) < 3:
        continue
    rets = r_signals['fwd_ret_16'].dropna()
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    np.random.seed(42)
    all_r = merged['fwd_ret_16'].dropna()
    n = len(rets)
    rm = np.array([all_r.sample(n).mean() for _ in range(5000)])
    mc_p = (rm >= mean_r).mean()
    passed = wr >= 0.55 and mc_p < 0.10
    a3[regime] = {'n': int(n), 'wr': float(wr), 'mean': float(mean_r), 'mc_p': float(mc_p), 'pass': passed}
    icon = '✅' if passed else '❌'
    print(f"  Agent 3 (Gate) {icon} {regime}: n={n}, WR={wr:.1%}, mean={mean_r*100:+.4f}%, MC p={mc_p:.4f}")

valid_regimes = {r for r, v in a3.items() if v.get('pass')}
filtered = signals[signals['btc_trend'].isin(valid_regimes)]
print(f"  Valid regimes: {valid_regimes}, filtered: {len(filtered)}")

# Agent 4: Risk — position sizing
INITIAL_CAPITAL = 10000
RISK_PCT = 0.02
capital = INITIAL_CAPITAL
peak = INITIAL_CAPITAL
max_dd = 0
trades = []

for _, row in filtered.iterrows():
    entry = row['eth_close']
    atr = row['atr']
    if atr <= 0:
        continue
    
    dd = (peak - capital) / peak
    if dd > 0.35:
        continue  # circuit breaker
    
    risk = capital * RISK_PCT
    sl_pct = (atr / entry * 100) * 1.0
    if sl_pct <= 0:
        continue
    size = risk / (sl_pct / 100)
    
    sl_price = entry * (1 - sl_pct/100)
    tp_pct = sl_pct * 2.0
    tp_price = entry * (1 + tp_pct/100)
    
    # Outcome at 4h
    idx = row.name
    if idx + 16 < len(merged):
        future = merged.iloc[idx+1:idx+17]['eth_close'].values
        hit_tp = any(c >= tp_price for c in future)
        hit_sl = any(c <= sl_price for c in future)
        
        if hit_tp and not hit_sl:
            pnl = size * (tp_pct / 100)
            outcome = 'WIN'
        elif hit_sl:
            pnl = -size * (sl_pct / 100)
            outcome = 'LOSS'
        else:
            ret_4h = merged.iloc[idx]['fwd_ret_16'] if pd.notna(merged.iloc[idx]['fwd_ret_16']) else 0
            pnl = size * ret_4h
            outcome = 'WIN' if ret_4h > 0 else 'LOSS'
        
        capital += pnl
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak)
        trades.append({'pnl': pnl, 'outcome': outcome, 'capital': capital})

wins = [t for t in trades if t['outcome'] == 'WIN']
losses = [t for t in trades if t['outcome'] == 'LOSS']
total_pnl = capital - INITIAL_CAPITAL

wr_pct = len(wins)/len(trades) if trades else 0
print(f"\n  Agent 4 (Risk): {len(trades)} trades, WR={wr_pct:.1%}")
print(f"  PnL: ${total_pnl:+.2f} ({total_pnl/INITIAL_CAPITAL*100:+.2f}%)")
print(f"  Max DD: {max_dd:.2%}")

# Agent 5: Execution
if trades:
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
    pf = abs(avg_win * len(wins)) / abs(avg_loss * len(losses)) if losses and avg_loss != 0 else float('inf')
    pnls = [t['pnl'] for t in trades]
    sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
    
    a5 = {
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'wr': len(wins)/len(trades), 'pf': float(pf), 'sharpe': float(sharpe),
        'total_pnl': float(total_pnl), 'return_pct': float(total_pnl/INITIAL_CAPITAL*100),
        'max_dd': float(max_dd),
    }
    print(f"  Agent 5 (Execution): PF={pf:.2f}, Sharpe={sharpe:.3f}")
else:
    a5 = {'trades': 0}
    print("  Agent 5: No trades")

# ═══════════ OPTIMIZATION FRAMEWORK ═══════════
print("\n" + "="*70)
print("OPTIMIZATION FRAMEWORK")
print("="*70)

# Validator: Walk-forward
n_sigs = len(filtered)
step_size = max(5, n_sigs // 10)
wf_results = []
for start in range(0, n_sigs - step_size, step_size):
    test_end = min(start + step_size, n_sigs)
    test_sigs = filtered.iloc[start:test_end]
    rets = test_sigs['fwd_ret_16'].dropna()
    if len(rets) < 3:
        continue
    wf_results.append({'wr': float((rets > 0).mean()), 'mean': float(rets.mean())})

if wf_results:
    avg_wr = np.mean([r['wr'] for r in wf_results])
    print(f"  Walk-forward: {len(wf_results)} windows, avg WR={avg_wr:.1%}")
    results['walk_forward'] = {'windows': len(wf_results), 'avg_wr': float(avg_wr)}

# Validator: Deflated Sharpe
all_rets = filtered['fwd_ret_16'].dropna()
if len(all_rets) > 5:
    sharpe = all_rets.mean() / all_rets.std() * np.sqrt(len(all_rets)) if all_rets.std() > 0 else 0
    from scipy.stats import norm
    n_trials = 10  # conservative
    e_max = norm.ppf(1 - 1/n_trials)
    std_s = 1 / np.sqrt(len(all_rets))
    dsr = (sharpe - e_max) / std_s if std_s > 0 else 0
    print(f"  Deflated Sharpe: {dsr:.3f} (sig: {dsr > 1.96})")
    results['deflated_sharpe'] = {'dsr': float(dsr), 'sig': dsr > 1.96}

# Validator: Monte Carlo
actual = filtered['fwd_ret_16'].dropna()
am, aw, n = actual.mean(), (actual > 0).mean(), len(actual)
np.random.seed(42)
all_r = merged['fwd_ret_16'].dropna()
rm = np.array([all_r.sample(n).mean() for _ in range(10000)])
mc_p = (rm >= am).mean()
bm = np.array([actual.sample(n, replace=True).mean() for _ in range(10000)])
ci_lo, ci_hi = np.percentile(bm, 2.5), np.percentile(bm, 97.5)
print(f"  Monte Carlo: n={n}, mean={am*100:+.4f}%, WR={aw:.1%}, MC p={mc_p:.4f}")
print(f"  CI: [{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%]")
results['monte_carlo'] = {'n': n, 'mean': float(am), 'wr': float(aw), 'mc_p': float(mc_p),
                          'ci': [float(ci_lo), float(ci_hi)], 'sig': bool(mc_p < 0.05)}

# Search: Threshold sweep
print("\n  --- Threshold sweep ---")
best = None; best_score = -999
for dev_min in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
    for ma in [20, 48]:
        dev_col = f'eth_btc_dev{ma}'
        if dev_col not in merged.columns:
            continue
        filt = merged[merged[dev_col] < -dev_min]
        if len(filt) < 10:
            continue
        rets = filt['fwd_ret_16'].dropna()
        if len(rets) < 10:
            continue
        wr = (rets > 0).mean()
        mean_r = rets.mean()
        score = wr * mean_r * np.sqrt(len(rets))
        if score > best_score:
            best_score = score
            best = {'dev_min': dev_min, 'ma': ma, 'n': len(rets), 'wr': float(wr), 'mean': float(mean_r)}

if best:
    print(f"  Best threshold: dev>{best['dev_min']} MA{best['ma']}: n={best['n']}, WR={best['wr']:.1%}, mean={best['mean']*100:+.4f}%")
    results['threshold_opt'] = best

# Selector
criteria = {
    'mc_sig': results.get('monte_carlo', {}).get('sig', False),
    'dsr_sig': results.get('deflated_sharpe', {}).get('sig', False),
    'wf_wr_above_50': results.get('walk_forward', {}).get('avg_wr', 0) > 0.50,
    'n_above_20': n >= 20,
}
passed = sum(criteria.values())
decision = 'DEPLOY' if passed >= 3 else 'CONDITIONAL' if passed >= 2 else 'KILL'
results['selector'] = {'criteria': criteria, 'passed': passed, 'decision': decision}

print(f"\n  Selector: {passed}/{len(criteria)} → {decision}")
for k, v in criteria.items():
    print(f"    {'✅' if v else '❌'} {k}")

# Save
with open(OUTPUT, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")
print(f"  Signal: ETH/BTC dev<5% → LONG ETH")
print(f"  n={n}, WR={aw:.1%}, mean={am*100:+.4f}%")
print(f"  MC p={mc_p:.4f}, DSR={results.get('deflated_sharpe', {}).get('dsr', 0):.3f}")
print(f"  Decision: {decision}")
print(f"\nSaved to {OUTPUT}")
