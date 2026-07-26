"""
S11 Cross-Asset: ETH/BTC divergence test with real BTC data.
Tests: does ETH/BTC deviation from mean predict ETH returns?
"""
import pandas as pd, numpy as np, json, os
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
OUTPUT = '/root/.openclaw/workspace/jimi_audit/reports/s11_btc_test.json'
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("Loading ETH 15m...")
eth = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
eth['timestamp'] = pd.to_datetime(eth['Open time'])
eth = eth.sort_values('timestamp').reset_index(drop=True)
for c in ['Close','High','Low','Volume']: eth[c] = eth[c].astype(float)

print("Loading BTC 1h...")
btc_raw = pd.read_json(f'{DATA_DIR}/btc_1h.json')
btc_raw.columns = ['ts','Open','High','Low','Close','Volume','ts2','qv','trades','tbqv','tbqav','ignore']
btc = btc_raw[['ts','Close']].copy()
btc['timestamp'] = pd.to_datetime(btc['ts'], unit='ms')
btc = btc.sort_values('timestamp').reset_index(drop=True)
btc['Close'] = btc['Close'].astype(float)

print(f"ETH: {len(eth)} bars, BTC: {len(btc)} bars")

# Merge ETH with BTC (nearest within30min)
merged = pd.merge_asof(
    eth[['timestamp','Close','High','Low','Volume']].rename(columns={'Close':'eth_close'}),
    btc[['timestamp','Close']].rename(columns={'Close':'btc_close'}),
    on='timestamp', direction='backward', tolerance=pd.Timedelta('30min')
)

merged = merged.dropna(subset=['btc_close'])
print(f"Merged: {len(merged)} bars")

# ETH/BTC ratio
merged['eth_btc'] = merged['eth_close'] / merged['btc_close']
merged['eth_btc_ma20'] = merged['eth_btc'].rolling(20).mean()
merged['eth_btc_ma48'] = merged['eth_btc'].rolling(48).mean()
merged['eth_btc_dev20'] = (merged['eth_btc'] - merged['eth_btc_ma20']) / merged['eth_btc_ma20']
merged['eth_btc_dev48'] = (merged['eth_btc'] - merged['eth_btc_ma48']) / merged['eth_btc_ma48']

# BTC trend
merged['btc_ema21'] = merged['btc_close'].ewm(span=21).mean()
merged['btc_ema55'] = merged['btc_close'].ewm(span=55).mean()
merged['btc_trend'] = np.where(merged['btc_ema21'] > merged['btc_ema55'], 'BULL', 'BEAR')

# ETH features
merged['eth_ema200'] = merged['eth_close'].ewm(span=200).mean()
merged['eth_trend'] = np.where(merged['eth_close'] > merged['eth_ema200'], 'BULL', 'BEAR')
merged['vol_ratio'] = merged['Volume'] / merged['Volume'].rolling(20).mean()
merged['atr'] = (merged['High'] - merged['Low']).rolling(14).mean()

# Forward returns
for h in [1, 4, 16, 24]:
    merged[f'fwd_ret_{h}'] = merged['eth_close'].shift(-h) / merged['eth_close'] - 1

round_trip_cost = 0.0010
results = {}

# ═══════════ RAW SIGNAL TEST ═══════════
print("\n" + "="*70)
print("RAW SIGNAL: ETH/BTC deviation → ETH forward returns")
print("="*70)

for dev_col, dev_name in [('eth_btc_dev20', 'MA20'), ('eth_btc_dev48', 'MA48')]:
    for direction in ['LONG', 'SHORT']:
        for thresh in [0.01, 0.02, 0.03, 0.05]:
            if direction == 'LONG':
                mask = merged[dev_col] < -thresh  # ETH underperforming
            else:
                mask = merged[dev_col] > thresh  # ETH outperforming
            
            shifted = mask.shift(1).fillna(False)
            events = merged[shifted]
            
            for h, label in [(4, '1h'), (16, '4h')]:
                rets = events[f'fwd_ret_{h}'].dropna()
                if len(rets) < 5:
                    continue
                mean_r = rets.mean()
                eff = -mean_r if direction == 'SHORT' else mean_r
                t, p = stats.ttest_1samp(rets, 0)
                wr = (rets < 0).mean() if direction == 'SHORT' else (rets > 0).mean()
                gate = "PASS" if p < 0.1 and eff > round_trip_cost else "FAIL"
                key = f"{dev_name}_{direction}_{thresh}_{label}"
                results[key] = {'n': len(rets), 'eff': float(eff), 'p': float(p), 'wr': float(wr), 'gate': gate}
                print(f"  {'+' if gate=='PASS' else '-'} {dev_name:4s} {direction:5s} dev>{thresh} {label}: n={len(rets):5d}, eff={eff*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# ═══════════ REGIME-SPECIFIC ═══════════
print("\n" + "="*70)
print("REGIME-SPECIFIC: BTC trend + ETH/BTC deviation")
print("="*70)

for btc_dir in ['BULL', 'BEAR']:
    for dev_thresh in [0.02, 0.03, 0.05]:
        btc_mask = merged['btc_trend'] == btc_dir
        eth_under = merged['eth_btc_dev48'] < -dev_thresh
        eth_over = merged['eth_btc_dev48'] > dev_thresh
        
        for combo_name, combo_mask in [
            (f'{btc_dir}+ETH_under', btc_mask & eth_under),
            (f'{btc_dir}+ETH_over', btc_mask & eth_over),
        ]:
            shifted = combo_mask.shift(1).fillna(False)
            events = merged[shifted]
            rets = events['fwd_ret_16'].dropna()
            if len(rets) < 5:
                continue
            mean_r = rets.mean()
            t, p = stats.ttest_1samp(rets, 0)
            wr = (rets > 0).mean()
            gate = "PASS" if p < 0.1 and mean_r > round_trip_cost else "FAIL"
            key = f"{combo_name}_dev{dev_thresh}_4h"
            results[key] = {'n': len(rets), 'mean': float(mean_r), 'p': float(p), 'wr': float(wr), 'gate': gate}
            print(f"  {'+' if gate=='PASS' else '-'} {combo_name:25s} dev>{dev_thresh} 4h: n={len(rets):5d}, mean={mean_r*100:+.4f}%, p={p:.4f}, WR={wr:.1%}")

# ═══════════ BEST COMBO MONTE CARLO ═══════════
print("\n" + "="*70)
print("MONTE CARLO — Best combo")
print("="*70)

# Find best passing combo
best_key = None; best_eff = 0
for k, v in results.items():
    if v.get('gate') == 'PASS' and v.get('eff', v.get('mean', 0)) > best_eff:
        best_eff = v.get('eff', v.get('mean', 0))
        best_key = k

if best_key:
    print(f"  Best: {best_key}")
    # Reconstruct the mask
    parts = best_key.split('_')
    # Parse the key to find the matching events
    # Find the events
    best_events = None
    for btc_dir in ['BULL', 'BEAR']:
        for dev_thresh in [0.02, 0.03, 0.05]:
            for combo_name, combo_mask in [
                (f'{btc_dir}+ETH_under', (merged['btc_trend'] == btc_dir) & (merged['eth_btc_dev48'] < -dev_thresh)),
                (f'{btc_dir}+ETH_over', (merged['btc_trend'] == btc_dir) & (merged['eth_btc_dev48'] > dev_thresh)),
            ]:
                key = f"{combo_name}_dev{dev_thresh}_4h"
                if key == best_key:
                    best_events = merged[combo_mask.shift(1).fillna(False)]
                    break
    
    if best_events is not None and len(best_events) >= 5:
        actual = best_events['fwd_ret_16'].dropna()
        am, aw, n = actual.mean(), (actual > 0).mean(), len(actual)
        print(f"  n={n}, mean={am*100:+.4f}%, WR={aw:.1%}")
        
        np.random.seed(42)
        all_r = merged['fwd_ret_16'].dropna()
        rm = np.array([all_r.sample(n).mean() for _ in range(10000)])
        pm = (rm >= am).mean()
        
        bm = np.array([actual.sample(n, replace=True).mean() for _ in range(10000)])
        ci_lo, ci_hi = np.percentile(bm, 2.5), np.percentile(bm, 97.5)
        
        mc = {'n': n, 'mean': float(am), 'wr': float(aw), 'mc_p': float(pm),
              'ci': [float(ci_lo), float(ci_hi)], 'sig': bool(pm < 0.05)}
        print(f"  MC p: {pm:.4f}, CI: [{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%]")
        print(f"  SIGNIFICANT: {'YES' if pm < 0.05 else 'NO'}")
        results['monte_carlo'] = mc
    else:
        print("  Could not reconstruct events")
else:
    print("  No passing combo found")

# ═══════════ VERDICT ═══════════
print("\n" + "="*70)
print("VERDICT")
print("="*70)

passing = {k: v for k, v in results.items() if v.get('gate') == 'PASS'}
print(f"  Passing signals: {len(passing)}")
for k, v in sorted(passing.items(), key=lambda x: x[1].get('eff', x[1].get('mean', 0)), reverse=True)[:5]:
    eff = v.get('eff', v.get('mean', 0))
    print(f"  + {k}: n={v['n']}, eff={eff*100:+.4f}%, p={v['p']:.4f}")

mc = results.get('monte_carlo', {})
verdict = {
    'passing_signals': len(passing),
    'best_signal': best_key,
    'mc_sig': mc.get('sig', False),
    'mc_p': mc.get('mc_p'),
}

if mc.get('sig'):
    verdict['gate'] = 'PASS'
    verdict['rec'] = 'Deploy with ETH/BTC deviation filter'
elif passing:
    verdict['gate'] = 'MARGINAL'
    verdict['rec'] = 'Some signals pass but MC not significant'
else:
    verdict['gate'] = 'FAIL'
    verdict['rec'] = 'No edge found'

print(f"  Gate: {verdict['gate']}")
print(f"  Rec: {verdict['rec']}")
results['verdict'] = verdict

with open(OUTPUT, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUTPUT}")
