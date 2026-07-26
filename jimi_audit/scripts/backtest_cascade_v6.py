"""Backtest + Monte Carlo for Liquidation Cascade v6 (optimized)."""
import json, os, sys, csv
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict

DATA_DIR = "/root/.openclaw/workspace/jimi_audit/data"
OHLCV_FILE = os.path.join(DATA_DIR, "history/ETHUSDT_15m.csv")

TP_PCT = 1.5
SL_PCT = 1.0
HOLD_BARS = 16
COMMISSION = 0.002  # 0.2% round trip
MC_ITERATIONS = 1000


def load_ohlcv():
    timestamps, closes, highs, lows, volumes = [], [], [], [], []
    with open(OHLCV_FILE) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            try:
                timestamps.append(int(row[0]))
                closes.append(float(row[4]))
                highs.append(float(row[2]))
                lows.append(float(row[3]))
                volumes.append(float(row[5]))
            except:
                continue
    return {
        'ts': np.array(timestamps),
        'close': np.array(closes),
        'high': np.array(highs),
        'low': np.array(lows),
        'volume': np.array(volumes),
    }


def detect_signals(ohlcv):
    closes = ohlcv['close']
    volumes = ohlcv['volume']
    n = len(closes)
    
    # Precompute rolling stats (O(n) total)
    log_ret = np.diff(np.log(closes))
    log_ret = np.insert(log_ret, 0, 0)
    
    # Rolling vol (20-bar std)
    vol_20 = np.zeros(n)
    for i in range(20, n):
        vol_20[i] = np.std(log_ret[i-20:i])
    
    # Rolling vol percentiles (expanding window, sampled every 10 bars for speed)
    vol_p33 = np.zeros(n)
    vol_p67 = np.zeros(n)
    for i in range(50, n):
        if i % 10 == 0 or i == 50:
            hist = vol_20[30:i]
            if len(hist) > 10:
                vol_p33[i] = np.percentile(hist, 33)
                vol_p67[i] = np.percentile(hist, 67)
            else:
                vol_p33[i] = vol_p33[i-1] if i > 50 else 0
                vol_p67[i] = vol_p67[i-1] if i > 50 else 1
        else:
            vol_p33[i] = vol_p33[i-1]
            vol_p67[i] = vol_p67[i-1]
    
    # Rolling volume MA
    vol_ma20 = np.zeros(n)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volumes[i-20:i])
    
    signals = []
    
    for i in range(100, n - HOLD_BARS):
        vol_ratio = volumes[i] / vol_ma20[i] if vol_ma20[i] > 0 else 1.0
        mom_5 = (closes[i] - closes[i-5]) / closes[i-5]
        mom_3 = (closes[i] - closes[i-3]) / closes[i-3]
        
        # Vol regime
        if vol_20[i] < vol_p33[i]:
            vol_regime = 'LOW'
        elif vol_20[i] < vol_p67[i]:
            vol_regime = 'MID'
        else:
            vol_regime = 'HIGH'
        
        # Path convexity
        if i >= 6:
            vel1 = (closes[i-3] - closes[i-6]) / closes[i-6]
            vel2 = (closes[i] - closes[i-3]) / closes[i-3]
            accel = vel2 - vel1
        else:
            accel = 0
            vel2 = 0
        
        # SHORT cascade
        short_score = 0
        if vol_ratio > 2.0 and mom_5 < -0.005:
            short_score += 2
        if mom_3 < -0.003:
            short_score += 1
        if accel < -0.002 and vel2 < 0:
            short_score += 1
        if vol_regime == 'MID':
            short_score += 1
        
        # LONG cascade
        long_score = 0
        if vol_ratio > 2.0 and mom_5 > 0.005:
            long_score += 2
        if mom_3 > 0.003:
            long_score += 1
        if accel > 0.002 and vel2 > 0:
            long_score += 1
        if vol_regime == 'MID':
            long_score += 1
        
        direction = None
        score = 0
        if short_score >= 3 and short_score > long_score:
            direction = 'SHORT'
            score = short_score
        elif long_score >= 3 and long_score > short_score:
            direction = 'LONG'
            score = long_score
        
        if not direction:
            continue
        
        # Session filter
        ts_ms = ohlcv['ts'][i]
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        hour = dt.hour
        if hour < 7 or hour > 21:
            continue
        
        cascade_score = min(score / 5.0, 1.0)
        if cascade_score < 0.45:
            continue
        
        session_quality = 1.0 if 12 <= hour <= 16 else 0.7
        conviction = min(0.45 + cascade_score * 0.35 + (score/5.0) * 0.15 + session_quality * 0.05, 0.92)
        
        tp_pct = TP_PCT
        sl_pct = SL_PCT
        if vol_regime == 'HIGH':
            tp_pct, sl_pct = 1.2, 0.8
        elif vol_regime == 'LOW':
            tp_pct, sl_pct = 1.0, 0.8
        
        signals.append({
            'idx': i, 'direction': direction, 'entry': closes[i],
            'tp_pct': tp_pct, 'sl_pct': sl_pct,
            'score': score, 'cascade_score': cascade_score,
            'conviction': conviction, 'vol_regime': vol_regime,
            'dt': dt.strftime('%Y-%m-%d %H:%M'),
        })
    
    return signals


def simulate(ohlcv, signals):
    closes = ohlcv['close']
    highs = ohlcv['high']
    lows = ohlcv['low']
    n = len(closes)
    results = []
    
    for sig in signals:
        idx = sig['idx']
        entry = sig['entry']
        d = sig['direction']
        tp = entry * (1 + sig['tp_pct']/100) if d == 'LONG' else entry * (1 - sig['tp_pct']/100)
        sl = entry * (1 - sig['sl_pct']/100) if d == 'LONG' else entry * (1 + sig['sl_pct']/100)
        
        outcome, exit_p, hold = 'TIMEOUT', entry, 0
        for j in range(1, min(HOLD_BARS+1, n-idx)):
            h, l = highs[idx+j], lows[idx+j]
            hold = j
            if d == 'LONG':
                if h >= tp: outcome, exit_p = 'WIN', tp; break
                if l <= sl: outcome, exit_p = 'LOSS', sl; break
            else:
                if l <= tp: outcome, exit_p = 'WIN', tp; break
                if h >= sl: outcome, exit_p = 'LOSS', sl; break
        
        if outcome == 'TIMEOUT':
            exit_p = closes[min(idx+HOLD_BARS, n-1)]
        
        pnl = ((exit_p - entry) / entry * 100) if d == 'LONG' else ((entry - exit_p) / entry * 100)
        pnl -= COMMISSION * 100
        
        results.append({**sig, 'outcome': outcome, 'exit': exit_p, 'pnl_pct': pnl, 'hold': hold})
    
    return results


def monte_carlo(trades):
    pnls = np.array([t['pnl_pct'] for t in trades])
    n = len(pnls)
    rng = np.random.default_rng(42)
    
    totals, wrs, pfs, mds, sharps = [], [], [], [], []
    for _ in range(MC_ITERATIONS):
        s = rng.choice(pnls, size=n, replace=True)
        totals.append(np.sum(s))
        wrs.append(np.sum(s > 0) / n)
        gp = np.sum(s[s > 0])
        gl = abs(np.sum(s[s < 0]))
        pfs.append(gp / gl if gl > 0 else 99)
        cum = np.cumsum(s)
        mds.append(np.min(cum - np.maximum.accumulate(cum)))
        std = np.std(s)
        sharps.append((np.mean(s) / std) * np.sqrt(30*12) if std > 0 else 0)
    
    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    
    return {
        'n': n, 'iterations': MC_ITERATIONS,
        'total_pnl': {'mean': float(np.mean(totals)), 'ci': ci(totals)},
        'win_rate': {'mean': float(np.mean(wrs)), 'ci': ci(wrs)},
        'pf': {'mean': float(np.mean(pfs)), 'median': float(np.median(pfs)), 'ci': ci(pfs)},
        'max_dd': {'mean': float(np.mean(mds)), 'worst': float(np.min(mds))},
        'sharpe': {'mean': float(np.mean(sharps)), 'ci': ci(sharps)},
        'expectancy': float(np.mean(pnls)),
        'avg_win': float(np.mean(pnls[pnls > 0])) if np.any(pnls > 0) else 0,
        'avg_loss': float(np.mean(pnls[pnls < 0])) if np.any(pnls < 0) else 0,
    }


def main():
    print("=" * 60)
    print("  CASCADE v6 BACKTEST + MONTE CARLO")
    print("=" * 60)
    
    print("\n[1/4] Loading OHLCV...")
    ohlcv = load_ohlcv()
    n = len(ohlcv['close'])
    t0 = datetime.fromtimestamp(ohlcv['ts'][0]/1000, tz=timezone.utc)
    t1 = datetime.fromtimestamp(ohlcv['ts'][-1]/1000, tz=timezone.utc)
    print(f"  {n} bars | {t0:%Y-%m-%d} to {t1:%Y-%m-%d}")
    
    print("\n[2/4] Detecting signals...")
    signals = detect_signals(ohlcv)
    shorts = [s for s in signals if s['direction'] == 'SHORT']
    longs = [s for s in signals if s['direction'] == 'LONG']
    print(f"  {len(signals)} signals ({len(shorts)}S / {len(longs)}L)")
    
    if not signals:
        print("  No signals. Exiting.")
        return
    
    print("\n[3/4] Simulating trades...")
    trades = simulate(ohlcv, signals)
    wins = [t for t in trades if t['outcome'] == 'WIN']
    losses = [t for t in trades if t['outcome'] == 'LOSS']
    timeouts = [t for t in trades if t['outcome'] == 'TIMEOUT']
    
    total_pnl = sum(t['pnl_pct'] for t in trades)
    wr = len(wins) / len(trades)
    gp = sum(t['pnl_pct'] for t in wins)
    gl = abs(sum(t['pnl_pct'] for t in losses))
    pf = gp / gl if gl > 0 else 99
    
    print(f"\n  RESULTS")
    print(f"  {'─'*45}")
    print(f"  Trades:    {len(trades)} ({len(wins)}W / {len(losses)}L / {len(timeouts)}T)")
    print(f"  Win Rate:  {wr:.1%}")
    print(f"  PF:        {pf:.2f}")
    print(f"  Total PnL: {total_pnl:+.2f}%")
    print(f"  Per trade: {total_pnl/len(trades):+.3f}%")
    if wins: print(f"  Avg WIN:   +{np.mean([t['pnl_pct'] for t in wins]):.3f}%")
    if losses: print(f"  Avg LOSS:  {np.mean([t['pnl_pct'] for t in losses]):.3f}%")
    
    # By regime
    print(f"\n  BY REGIME")
    print(f"  {'─'*45}")
    for regime in ['MID', 'HIGH', 'LOW']:
        subset = [t for t in trades if t['vol_regime'] == regime]
        if not subset: continue
        w = len([t for t in subset if t['outcome'] == 'WIN'])
        p = sum(t['pnl_pct'] for t in subset)
        g = abs(sum(t['pnl_pct'] for t in subset if t['pnl_pct'] < 0))
        pr = sum(t['pnl_pct'] for t in subset if t['pnl_pct'] > 0)
        rpf = pr / g if g > 0 else 99
        print(f"  {regime:6s} | {len(subset):3d} trades | WR={w/len(subset):.0%} | PF={rpf:.2f} | PnL={p:+.2f}%")
    
    # By direction
    print(f"\n  BY DIRECTION")
    print(f"  {'─'*45}")
    for d in ['SHORT', 'LONG']:
        subset = [t for t in trades if t['direction'] == d]
        if not subset: continue
        w = len([t for t in subset if t['outcome'] == 'WIN'])
        p = sum(t['pnl_pct'] for t in subset)
        g = abs(sum(t['pnl_pct'] for t in subset if t['pnl_pct'] < 0))
        pr = sum(t['pnl_pct'] for t in subset if t['pnl_pct'] > 0)
        rpf = pr / g if g > 0 else 99
        print(f"  {d:6s} | {len(subset):3d} trades | WR={w/len(subset):.0%} | PF={rpf:.2f} | PnL={p:+.2f}%")
    
    print(f"\n[4/4] Monte Carlo ({MC_ITERATIONS} iterations)...")
    mc = monte_carlo(trades)
    
    print(f"\n  MONTE CARLO")
    print(f"  {'─'*45}")
    print(f"  Total PnL:     {mc['total_pnl']['mean']:+.2f}% [{mc['total_pnl']['ci'][0]:+.2f}, {mc['total_pnl']['ci'][1]:+.2f}]")
    print(f"  Win Rate:      {mc['win_rate']['mean']:.1%} [{mc['win_rate']['ci'][0]:.1%}, {mc['win_rate']['ci'][1]:.1%}]")
    print(f"  PF:            {mc['pf']['mean']:.2f} (med={mc['pf']['median']:.2f}) [{mc['pf']['ci'][0]:.2f}, {mc['pf']['ci'][1]:.2f}]")
    print(f"  Max Drawdown:  {mc['max_dd']['mean']:.2f}% (worst: {mc['max_dd']['worst']:.2f}%)")
    print(f"  Sharpe (ann):  {mc['sharpe']['mean']:.2f} [{mc['sharpe']['ci'][0]:.2f}, {mc['sharpe']['ci'][1]:.2f}]")
    print(f"  Expectancy:    {mc['expectancy']:+.3f}%/trade")
    print(f"  Avg WIN:       +{mc['avg_win']:.3f}%")
    print(f"  Avg LOSS:      {mc['avg_loss']:.3f}%")
    
    # Save
    output = {
        'strategy': 'liquidation_cascade_v6',
        'date': datetime.now(timezone.utc).isoformat(),
        'data': f"{t0:%Y-%m-%d} to {t1:%Y-%m-%d} ({n} bars)",
        'signals': {'total': len(signals), 'short': len(shorts), 'long': len(longs)},
        'backtest': {'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
                     'timeouts': len(timeouts), 'wr': wr, 'pf': pf, 'total_pnl': total_pnl,
                     'expectancy': total_pnl/len(trades)},
        'monte_carlo': mc,
    }
    with open('/tmp/cascade_v6_backtest.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\n  Saved to /tmp/cascade_v6_backtest.json")

if __name__ == '__main__':
    main()
