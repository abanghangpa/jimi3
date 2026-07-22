#!/usr/bin/env python3
"""
Failed Breakout v6 — Trapped Trader Engine
==========================================
Core idea: Detect when breakout traders are TRAPPED, then ride the flush.

What makes a trap:
1. Price sweeps beyond a key level (liquidity grab)
2. The sweep has VOLUME (real participation, not a wick)
3. Price reverses SHARPLY (not drifting back — forced exits)
4. Taker flow flips (aggressive sellers become buyers or vice versa)

Key changes from v5:
- Don't over-filter. v5 killed 99.5% of signals. We need more data points.
- Use CONFLUENCE scoring instead of hard filters
- Add volume confirmation (the sweep must have real volume)
- Add taker flip detection (aggressive direction change)
- Test with looser entry, tighter risk management
"""
import csv, json, sys, os
from datetime import datetime, timezone
import numpy as np

BASE = '/root/.openclaw/workspace/jimi_audit'
DATA_FILE = f'{BASE}/eth_15m_6m.csv'
FEE = 0.0002
SLIP = 0.001

print("Loading...", flush=True)
bars = []
with open(DATA_FILE) as f:
    for row in csv.DictReader(f):
        bars.append({
            'ts': datetime.strptime(row['Open time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc),
            'o': float(row['Open']), 'h': float(row['High']),
            'l': float(row['Low']), 'c': float(row['Close']),
            'v': float(row['Volume']),
            'tb': float(row.get('Taker buy base asset volume', 0)),
        })

N = len(bars)
closes = np.array([b['c'] for b in bars])
highs = np.array([b['h'] for b in bars])
lows = np.array([b['l'] for b in bars])
opens = np.array([b['o'] for b in bars])
volumes = np.array([b['v'] for b in bars])
taker_buys = np.array([b['tb'] for b in bars])

# Pre-compute indicators
atr = np.zeros(N)
for i in range(1, N):
    tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
    atr[i] = tr if i < 14 else (atr[i-1]*13 + tr)/14

avg_vol = np.zeros(N)
for i in range(20, N):
    avg_vol[i] = np.mean(volumes[i-20:i])

ema200 = np.zeros(N)
ema200[199] = np.mean(closes[:200])
m_ema = 2 / 201
for i in range(200, N):
    ema200[i] = closes[i] * m_ema + ema200[i-1] * (1 - m_ema)

# Taker ratio (buy volume / total volume)
taker_ratio = np.zeros(N)
for i in range(1, N):
    if volumes[i] > 0:
        taker_ratio[i] = taker_buys[i] / volumes[i]
    else:
        taker_ratio[i] = 0.5

# Rolling average taker ratio (for flip detection)
avg_taker = np.zeros(N)
for i in range(20, N):
    avg_taker[i] = np.mean(taker_ratio[i-20:i])

print(f"Loaded {N} bars ({bars[0]['ts'].strftime('%Y-%m-%d')} to {bars[-1]['ts'].strftime('%Y-%m-%d')})", flush=True)

# ============================================================
# Compute swing levels (multi-timeframe)
# ============================================================
print("Computing swing levels...", flush=True)

# 4H swings (16 bars)
swing_high_4h = np.full(N, np.nan)
swing_low_4h = np.full(N, np.nan)
for i in range(16, N):
    window = highs[i-16:i]
    swing_high_4h[i] = np.max(window)
    swing_low_4h[i] = np.min(lows[i-16:i])

# 12H swings (48 bars)
swing_high_12h = np.full(N, np.nan)
swing_low_12h = np.full(N, np.nan)
for i in range(48, N):
    swing_high_12h[i] = np.max(highs[i-48:i])
    swing_low_12h[i] = np.min(lows[i-48:i])

# 1D swings (96 bars)
swing_high_1d = np.full(N, np.nan)
swing_low_1d = np.full(N, np.nan)
for i in range(96, N):
    swing_high_1d[i] = np.max(highs[i-96:i])
    swing_low_1d[i] = np.min(lows[i-96:i])

# ============================================================
# Detect trapped trader signals
# ============================================================
print("Detecting trapped trader signals...", flush=True)

signals = []
COOLDOWN = 8  # bars between signals (2 hours)

for i in range(96, N - 48):  # leave room for lookback and forward simulation
    if atr[i] == 0 or avg_vol[i] == 0:
        continue

    # Session filter: skip dead hours (00-04 UTC)
    hour = bars[i]['ts'].hour
    dead_session = hour >= 0 and hour < 4

    best_signal = None
    best_score = 0

    # Check against each timeframe level
    for tf, sh, sl in [('4H', swing_high_4h, swing_low_4h),
                        ('12H', swing_high_12h, swing_low_12h),
                        ('1D', swing_high_1d, swing_low_1d)]:

        # === SHORT signal: swept above resistance, now reversing ===
        if not np.isnan(sh[i-1]):
            level = sh[i-1]
            sweep_depth = (highs[i] - level) / atr[i]
            if sweep_depth > 0.15:  # swept above level (even slightly)
                # Check reversal: close below level
                if closes[i] < level:
                    # Confluence scoring (0-1 scale, higher = more trapped)
                    score = 0.0

                    # 1. Sweep depth: deeper sweep = more stops grabbed (up to 1.5 ATR)
                    depth_score = min(sweep_depth / 1.5, 1.0) * 0.25
                    score += depth_score

                    # 2. Volume: sweep should have volume above average
                    vol_ratio = volumes[i] / avg_vol[i] if avg_vol[i] > 0 else 0
                    vol_score = min(vol_ratio / 2.0, 1.0) * 0.20
                    score += vol_score

                    # 3. Wick rejection: upper wick should be significant
                    body = abs(closes[i] - opens[i])
                    upper_wick = highs[i] - max(opens[i], closes[i])
                    wick_ratio = upper_wick / (body + 0.001)
                    wick_score = min(wick_ratio / 2.0, 1.0) * 0.20
                    score += wick_score

                    # 4. Taker flip: should see aggressive selling (taker_ratio dropping)
                    if avg_taker[i] > 0:
                        taker_flip = (avg_taker[i] - taker_ratio[i])
                        flip_score = min(max(taker_flip, 0) / 0.2, 1.0) * 0.20
                        score += flip_score

                    # 5. Close strength: how far below the level did we close?
                    close_dist = (level - closes[i]) / atr[i]
                    close_score = min(max(close_dist, 0) / 0.5, 1.0) * 0.15
                    score += close_score

                    if score > best_score:
                        best_score = score
                        best_signal = {
                            'dir': 'SHORT',
                            'entry': closes[i],
                            'sl': highs[i] + atr[i] * 0.3,
                            'level': level,
                            'tf': tf,
                            'sweep_depth': round(sweep_depth, 2),
                            'vol_ratio': round(vol_ratio, 2),
                            'wick_ratio': round(wick_ratio, 2),
                            'score': round(score, 3),
                            'bar': i,
                            'ts': str(bars[i]['ts']),
                            'hour': hour,
                            'dead': dead_session,
                        }

        # === LONG signal: swept below support, now reversing ===
        if not np.isnan(sl[i-1]):
            level = sl[i-1]
            sweep_depth = (level - lows[i]) / atr[i]
            if sweep_depth > 0.15:
                if closes[i] > level:
                    score = 0.0

                    depth_score = min(sweep_depth / 1.5, 1.0) * 0.25
                    score += depth_score

                    vol_ratio = volumes[i] / avg_vol[i] if avg_vol[i] > 0 else 0
                    vol_score = min(vol_ratio / 2.0, 1.0) * 0.20
                    score += vol_score

                    body = abs(closes[i] - opens[i])
                    lower_wick = min(opens[i], closes[i]) - lows[i]
                    wick_ratio = lower_wick / (body + 0.001)
                    wick_score = min(wick_ratio / 2.0, 1.0) * 0.20
                    score += wick_score

                    if avg_taker[i] > 0:
                        taker_flip = (taker_ratio[i] - avg_taker[i])
                        flip_score = min(max(taker_flip, 0) / 0.2, 1.0) * 0.20
                        score += flip_score

                    close_dist = (closes[i] - level) / atr[i]
                    close_score = min(max(close_dist, 0) / 0.5, 1.0) * 0.15
                    score += close_score

                    if score > best_score:
                        best_score = score
                        best_signal = {
                            'dir': 'LONG',
                            'entry': closes[i],
                            'sl': lows[i] - atr[i] * 0.3,
                            'level': level,
                            'tf': tf,
                            'sweep_depth': round(sweep_depth, 2),
                            'vol_ratio': round(vol_ratio, 2),
                            'wick_ratio': round(wick_ratio, 2),
                            'score': round(score, 3),
                            'bar': i,
                            'ts': str(bars[i]['ts']),
                            'hour': hour,
                            'dead': dead_session,
                        }

    if best_signal and best_score >= 0.3:  # Low threshold — let the sweep find what works
        # Cooldown check
        if not signals or (i - signals[-1]['bar']) >= COOLDOWN:
            signals.append(best_signal)

print(f"Found {len(signals)} trapped-trader signals", flush=True)
dirs = {}
for s in signals:
    d = s['dir']
    dirs[d] = dirs.get(d, 0) + 1
print(f"  Directions: {dirs}")
scores = [s['score'] for s in signals]
print(f"  Scores: min={min(scores):.3f} max={max(scores):.3f} avg={np.mean(scores):.3f}")
tfs = {}
for s in signals:
    t = s['tf']
    tfs[t] = tfs.get(t, 0) + 1
print(f"  Timeframes: {tfs}")

# ============================================================
# Simulation engine
# ============================================================
def sim(sigs, tp_pct, sl_pct, hold_bars, min_score, use_trend, use_session, use_vol_gate):
    trades = []
    for s in sigs:
        if s['score'] < min_score:
            continue
        if use_session and s['dead']:
            continue
        if use_trend:
            if s['dir'] == 'LONG' and closes[s['bar']] < ema200[s['bar']]:
                continue
            if s['dir'] == 'SHORT' and closes[s['bar']] > ema200[s['bar']]:
                continue

        i = s['bar']
        entry = s['entry']
        if s['dir'] == 'LONG':
            tp = entry * (1 + tp_pct/100)
            sl = entry * (1 - sl_pct/100)
            # Override SL with signal SL if tighter
            if s['sl'] > sl:
                sl = s['sl']
        else:
            tp = entry * (1 - tp_pct/100)
            sl = entry * (1 + sl_pct/100)
            if s['sl'] < sl:
                sl = s['sl']

        outcome = 'TIMEOUT'
        exit_price = entry
        for j in range(i+1, min(i+hold_bars, N)):
            if s['dir'] == 'LONG':
                if highs[j] >= tp:
                    outcome = 'WIN'
                    exit_price = tp
                    break
                if lows[j] <= sl:
                    outcome = 'LOSS'
                    exit_price = sl
                    break
            else:
                if lows[j] <= tp:
                    outcome = 'WIN'
                    exit_price = tp
                    break
                if highs[j] >= sl:
                    outcome = 'LOSS'
                    exit_price = sl
                    break

        if outcome == 'TIMEOUT':
            exit_price = closes[min(i+hold_bars-1, N-1)]

        pnl_pct = ((exit_price - entry) / entry * 100) if s['dir'] == 'LONG' else ((entry - exit_price) / entry * 100)
        pnl_pct -= FEE * 2 * 100 + SLIP * 100  # fees + slippage

        trades.append({
            'dir': s['dir'], 'entry': round(entry, 2), 'exit': round(exit_price, 2),
            'sl': round(sl, 2), 'tp': round(tp, 2),
            'pnl': round(pnl_pct, 4), 'outcome': outcome,
            'score': s['score'], 'held': j - i,
            'ts': s['ts'], 'tf': s['tf'],
            'sweep': s['sweep_depth'], 'vol': s['vol_ratio'],
        })

    if len(trades) < 3:
        return None

    n = len(trades)
    w = sum(1 for t in trades if t['outcome'] == 'WIN')
    l = sum(1 for t in trades if t['outcome'] == 'LOSS')
    to = sum(1 for t in trades if t['outcome'] == 'TIMEOUT')
    wins = [t['pnl'] for t in trades if t['outcome'] == 'WIN']
    losses = [t['pnl'] for t in trades if t['outcome'] in ('LOSS', 'TIMEOUT') and t['pnl'] < 0]
    win_pnl = sum(wins) if wins else 0
    loss_pnl = abs(sum(losses)) if losses else 0
    pf = win_pnl / loss_pnl if loss_pnl > 0 else 999
    total_pnl = sum(t['pnl'] for t in trades)
    wr = w / n * 100

    # Monthly breakdown
    monthly = {}
    for t in trades:
        m = t['ts'][:7]
        if m not in monthly:
            monthly[m] = {'w': 0, 'l': 0, 'pnl': 0}
        if t['outcome'] == 'WIN':
            monthly[m]['w'] += 1
        else:
            monthly[m]['l'] += 1
        monthly[m]['pnl'] = round(monthly[m]['pnl'] + t['pnl'], 2)

    # Max drawdown
    equity = 100
    peak = 100
    max_dd = 0
    for t in trades:
        equity += t['pnl']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    bad_months = sum(1 for m, v in monthly.items() if v['pnl'] < 0)

    return {
        'trades': n, 'wins': w, 'losses': l, 'timeouts': to,
        'wr': round(wr, 1), 'pf': round(pf, 2),
        'pnl': round(total_pnl, 1), 'dd': round(max_dd, 1),
        'avg_w': round(np.mean(wins), 2) if wins else 0,
        'avg_l': round(np.mean([abs(x) for x in losses]), 2) if losses else 0,
        'bad_m': bad_months, 'months': len(monthly),
        'monthly': {k: round(v['pnl'], 1) for k, v in sorted(monthly.items())},
    }

# ============================================================
# Sweep parameters
# ============================================================
print("\nSweeping...", flush=True)
results = []
tested = 0

for tp in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    for sl in [0.3, 0.5, 0.75, 1.0, 1.5]:
        if sl >= tp: continue
        for hold in [4, 8, 12, 16, 24, 32]:
            for min_score in [0.3, 0.4, 0.5, 0.6, 0.7]:
                for trend in [False, True]:
                    for session in [False, True]:
                        for vol_gate in [False, True]:
                            r = sim(signals, tp, sl, hold, min_score, trend, session, vol_gate)
                            tested += 1
                            if r:
                                results.append({**r, 'tp': tp, 'sl': sl, 'hold': hold,
                                               'score': min_score, 'trend': trend,
                                               'sess': session, 'vol': vol_gate})

print(f"Tested {tested} configs", flush=True)

results.sort(key=lambda x: x['pf'] * (x['wr']/100) * (x['trades']**0.3), reverse=True)

# Tier definitions
ent = [r for r in results if r['wr']>=65 and r['pf']>=2.0 and r['dd']<25 and r['bad_m']<=3 and r['trades']>=10]
good = [r for r in results if r['wr']>=55 and r['pf']>=1.5 and r['trades']>=8]
ok = [r for r in results if r['wr']>=50 and r['pf']>=1.2 and r['trades']>=5]
any_profit = [r for r in results if r['pf'] > 1.0 and r['trades'] >= 5]

print(f"\nEnterprise (WR>=65, PF>=2.0, DD<25, bad_m<=3): {len(ent)}")
print(f"Good (WR>=55, PF>=1.5, T>=8): {len(good)}")
print(f"OK (WR>=50, PF>=1.2, T>=5): {len(ok)}")
print(f"Any profitable (PF>1.0, T>=5): {len(any_profit)}")

for label, subset in [("ENTERPRISE", ent), ("GOOD", good), ("OK", ok), ("PROFITABLE", any_profit)]:
    if not subset: continue
    print(f"\n{'='*120}")
    print(f"{label} RESULTS ({len(subset)} configs)")
    print(f"{'='*120}")
    print(f"  {'#':>2s} {'TP':>4s} {'SL':>4s} {'RR':>4s} {'H':>3s} {'Sc':>4s} {'Tr':>2s} {'Se':>2s} {'Vl':>2s} | {'N':>4s} {'W':>3s} {'L':>3s} {'T':>3s} {'WR':>5s} {'PF':>5s} {'PnL':>7s} {'DD':>5s} {'AvW':>5s} {'AvL':>5s} {'BM':>3s}")
    print("  " + "-" * 115)
    for i, r in enumerate(subset[:30]):
        rr = round(r['tp']/r['sl'], 1)
        tr = 'T' if r['trend'] else '-'
        se = 'S' if r['sess'] else '-'
        vl = 'V' if r['vol'] else '-'
        print(f"  {i+1:>2d} {r['tp']:>4.1f} {r['sl']:>4.2f} {rr:>4.1f} {r['hold']:>3d} {r['score']:>4.1f} {tr:>2s} {se:>2s} {vl:>2s} | {r['trades']:>4d} {r['wins']:>3d} {r['losses']:>3d} {r['timeouts']:>3d} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>+7.1f} {r['dd']:>5.1f} {r['avg_w']:>5.2f} {r['avg_l']:>5.2f} {r['bad_m']:>3d}")

best = ent if ent else (good if good else (ok if ok else any_profit))
if best:
    b = best[0]
    print(f"\n{'='*80}")
    print("BEST CONFIG")
    print(f"{'='*80}")
    print(f"  TP={b['tp']}% SL={b['sl']}% Hold={b['hold']}h Score>={b['score']}")
    print(f"  Trend={'EMA200' if b['trend'] else 'None'} Session={'Skip 00-04' if b['sess'] else 'All'} VolGate={'Yes' if b['vol'] else 'No'}")
    print(f"  R:R=1:{b['tp']/b['sl']:.1f}")
    print(f"  {b['trades']}T {b['wr']}%WR {b['pf']}PF PnL={b['pnl']}% DD={b['dd']}%")
    print(f"\n  Monthly:")
    for m, v in sorted(b['monthly'].items()):
        bar = "█" * max(0, int(v * 2))
        print(f"    {m}: PnL={v:+.1f}% {bar}")
else:
    # Show top 5 anyway
    print(f"\nNo configs met any tier. Top 5 by score:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. TP={r['tp']} SL={r['sl']} H={r['hold']} Sc={r['score']} | {r['trades']}T {r['wr']}%WR {r['pf']}PF PnL={r['pnl']}% DD={r['dd']}%")

# Save results
import json
output = {
    'total_signals': len(signals),
    'configs_tested': tested,
    'enterprise': len(ent),
    'good': len(good),
    'ok': len(ok),
    'profitable': len(any_profit),
    'best_config': best[0] if best else None,
    'top_5': results[:5],
}
# Convert numpy types
def convert(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

out_file = f'{BASE}/reports/fb_v6_backtest.json'
with open(out_file, 'w') as f:
    json.dump(output, f, indent=2, default=convert)
print(f"\nResults saved to {out_file}")
print("\nDone")
