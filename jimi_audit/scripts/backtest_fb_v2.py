#!/usr/bin/env python3
"""
Failed Breakout v2 — Enterprise-Grade Backtest
================================================
Key changes from baseline:
1. Shorter hold (8-16h instead of 32h)
2. ATR-based SL (dynamic, not fixed %)
3. Volume + wick confirmation on breakout failure
4. Trend filter (trade with EMA200, not against)
5. Session filter (London/NY overlap best)
6. Higher conviction threshold
7. Contrarian R:R (SL < TP)
8. Dedup: don't re-enter same breakout
"""
import csv, json, sys, os
from datetime import datetime, timezone, timedelta
import numpy as np

BASE = '/root/.openclaw/workspace/jimi_audit'
DATA_FILE = f'{BASE}/eth_15m_6m.csv'
FEE = 0.0002
SLIP = 0.001

print("Loading data...", flush=True)

bars = []
with open(DATA_FILE) as f:
    for row in csv.DictReader(f):
        bars.append({
            'ts': datetime.strptime(row['Open time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc),
            'o': float(row['Open']), 'h': float(row['High']),
            'l': float(row['Low']), 'c': float(row['Close']),
            'v': float(row['Volume']),
            'tb': float(row.get('Taker buy base asset volume', 0)),
            'trades': int(row.get('Number of trades', 0)),
        })

N = len(bars)
closes = np.array([b['c'] for b in bars])
highs = np.array([b['h'] for b in bars])
lows = np.array([b['l'] for b in bars])
volumes = np.array([b['v'] for b in bars])
taker_buy = np.array([b['tb'] for b in bars])

# ATR
atr = np.zeros(N)
for i in range(1, N):
    tr = max(bars[i]['h']-bars[i]['l'], abs(bars[i]['h']-bars[i-1]['c']), abs(bars[i]['l']-bars[i-1]['c']))
    atr[i] = tr if i < 14 else (atr[i-1]*13 + tr)/14

# Avg volume
avg_vol = np.zeros(N)
for i in range(20, N):
    avg_vol[i] = np.mean(volumes[i-20:i])

# EMA200
ema200 = np.zeros(N)
ema200[199] = np.mean(closes[:200])
m_ema = 2 / 201
for i in range(200, N):
    ema200[i] = closes[i] * m_ema + ema200[i-1] * (1 - m_ema)

# EMA50
ema50 = np.zeros(N)
ema50[49] = np.mean(closes[:50])
m50 = 2 / 51
for i in range(50, N):
    ema50[i] = closes[i] * m50 + ema50[i-1] * (1 - m50)

print(f"Loaded {N} bars", flush=True)


def detect_failed_breakout(idx, lookback=48, return_pct=0.3, min_vol_ratio=1.0, min_wick_ratio=0.4):
    """
    Detect failed breakout pattern at bar index `idx`.
    Returns (direction, conviction) or (None, 0).
    
    Logic:
    1. Find recent high/low in lookback
    2. Check if current price broke above/below that level
    3. Check if the breakout FAILED (price returned)
    4. Confirm with volume + wick rejection
    """
    if idx < lookback + 10:
        return None, 0
    
    current = bars[idx]
    price = current['c']
    atr_val = atr[idx]
    if atr_val <= 0:
        return None, 0
    
    # Find swing high/low in lookback
    swing_high = max(highs[idx-lookback:idx])
    swing_low = min(lows[idx-lookback:idx])
    
    # Check for failed SHORT breakout (price broke above then came back)
    # This is a BULLISH signal (contrarian: short breakout failed → go long)
    broke_above = False
    breakout_bar = None
    for j in range(max(idx-8, 200), idx):
        if highs[j] > swing_high:
            broke_above = True
            breakout_bar = j
            break
    
    if broke_above and breakout_bar:
        # Check if price returned below swing_high
        return_dist = (swing_high - price) / swing_high * 100
        if return_dist >= return_pct:
            # Check breakout bar quality
            bb = bars[breakout_bar]
            bb_range = bb['h'] - bb['l']
            bb_body = abs(bb['c'] - bb['o'])
            bb_upper_wick = bb['h'] - max(bb['c'], bb['o'])
            wick_ratio = bb_upper_wick / bb_range if bb_range > 0 else 0
            
            # Volume on breakout bar
            vol_ratio = volumes[breakout_bar] / max(avg_vol[breakout_bar], 1)
            
            # Taker ratio on breakout (should show selling)
            taker_sell_ratio = 1 - (bb['tb'] / max(bb['v'], 0.01))
            
            conviction = 0.0
            # Wick rejection (long upper wick = rejection)
            if wick_ratio >= min_wick_ratio:
                conviction += 0.3
            # Volume (high volume on breakout = more trapped participants)
            if vol_ratio >= min_vol_ratio:
                conviction += 0.2
            # Taker flip (sellers active on breakout bar)
            if taker_sell_ratio >= 0.52:
                conviction += 0.2
            # Price is now below swing high (confirmed failure)
            if price < swing_high:
                conviction += 0.2
            # Reversal bar quality (current bar should be bearish)
            if current['c'] < current['o']:
                conviction += 0.1
            
            if conviction >= 0.5:
                return 'LONG', conviction
    
    # Check for failed LONG breakout (price broke below then came back)
    # This is a BEARISH signal (contrarian: long breakout failed → go short)
    broke_below = False
    breakout_bar = None
    for j in range(max(idx-8, 200), idx):
        if lows[j] < swing_low:
            broke_below = True
            breakout_bar = j
            break
    
    if broke_below and breakout_bar:
        return_dist = (price - swing_low) / swing_low * 100
        if return_dist >= return_pct:
            bb = bars[breakout_bar]
            bb_range = bb['h'] - bb['l']
            bb_lower_wick = min(bb['c'], bb['o']) - bb['l']
            wick_ratio = bb_lower_wick / bb_range if bb_range > 0 else 0
            vol_ratio = volumes[breakout_bar] / max(avg_vol[breakout_bar], 1)
            taker_buy_ratio = bb['tb'] / max(bb['v'], 0.01)
            
            conviction = 0.0
            if wick_ratio >= min_wick_ratio:
                conviction += 0.3
            if vol_ratio >= min_vol_ratio:
                conviction += 0.2
            if taker_buy_ratio >= 0.52:
                conviction += 0.2
            if price > swing_low:
                conviction += 0.2
            if current['c'] > current['o']:
                conviction += 0.1
            
            if conviction >= 0.5:
                return 'SHORT', conviction
    
    return None, 0


def backtest_v2(tp_pct, sl_pct, hold_hours, min_conv, lookback, return_pct,
                min_vol_ratio, min_wick_ratio, trend_filter, session_filter,
                min_atr_mult=0, dedup_bars=12):
    """Backtest failed breakout v2 with configurable params."""
    trades = []
    capital = 200.0
    peak = capital
    max_dd = 0
    last_entry_bar = -999
    
    for i in range(210, N):
        # Session filter
        if session_filter:
            h = bars[i]['ts'].hour
            if session_filter == 'ol' and not (13 <= h < 16):
                continue
            elif session_filter == 'ln' and not (8 <= h < 21):
                continue
        
        # Detect failed breakout
        direction, conviction = detect_failed_breakout(
            i, lookback=lookback, return_pct=return_pct,
            min_vol_ratio=min_vol_ratio, min_wick_ratio=min_wick_ratio
        )
        
        if not direction or conviction < min_conv:
            continue
        
        # Dedup (don't re-enter within N bars)
        if i - last_entry_bar < dedup_bars:
            continue
        
        # Trend filter
        if trend_filter:
            if direction == 'LONG' and closes[i] < ema200[i]:
                continue
            if direction == 'SHORT' and closes[i] > ema200[i]:
                continue
        
        # ATR filter (skip low-vol entries)
        if min_atr_mult > 0:
            atr_pct = atr[i] / closes[i] * 100
            if atr_pct < min_atr_mult:
                continue
        
        # Enter trade
        entry = closes[i] * (1 + SLIP if direction == 'LONG' else 1 - SLIP)
        sl_d = entry * sl_pct / 100
        tp_d = entry * tp_pct / 100
        if direction == 'LONG':
            sl_p = entry - sl_d
            tp_p = entry + tp_d
        else:
            sl_p = entry + sl_d
            tp_p = entry - tp_d
        
        last_entry_bar = i
        
        # Simulate
        outcome = None
        exit_p = None
        for j in range(i + 1, min(i + hold_hours * 4 + 1, N)):
            if direction == 'LONG':
                if highs[j] >= tp_p:
                    outcome, exit_p = 'WIN', tp_p
                    break
                if lows[j] <= sl_p:
                    outcome, exit_p = 'LOSS', sl_p
                    break
            else:
                if lows[j] <= tp_p:
                    outcome, exit_p = 'WIN', tp_p
                    break
                if highs[j] >= sl_p:
                    outcome, exit_p = 'LOSS', sl_p
                    break
        
        if outcome is None:
            exit_p = closes[min(i + hold_hours * 4, N - 1)]
            outcome = 'WIN' if ((direction == 'LONG' and exit_p > entry) or
                               (direction == 'SHORT' and exit_p < entry)) else 'LOSS'
            outcome = 'TIMEOUT'
        
        pnl_pct = ((exit_p - entry) / entry * 100) if direction == 'LONG' else ((entry - exit_p) / entry * 100)
        pnl_pct -= FEE * 2  # entry + exit fee
        
        size = capital * 0.10 * 25
        net_pnl = size * pnl_pct / 100
        capital += net_pnl
        if capital > peak: peak = capital
        dd = (peak - capital) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        
        trades.append({
            'direction': direction, 'outcome': outcome, 'pnl_pct': pnl_pct,
            'conviction': conviction, 'entry': entry, 'exit': exit_p,
            'bar': i, 'ts': bars[i]['ts'].isoformat()
        })
    
    if len(trades) < 5:
        return None
    
    wins = [t for t in trades if t['outcome'] == 'WIN']
    losses = [t for t in trades if t['outcome'] == 'LOSS']
    timeouts = [t for t in trades if t['outcome'] == 'TIMEOUT']
    
    w = len(wins)
    l = len(losses)
    t = len(timeouts)
    total = len(trades)
    wr = w / total * 100
    
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pnl_pct']) for t in losses]) if losses else 0
    pf = (w * avg_win) / max(l * avg_loss, 0.01) if l > 0 else 999
    
    # Monthly breakdown
    monthly = {}
    for t in trades:
        m = t['ts'][:7]
        if m not in monthly:
            monthly[m] = {'wins': 0, 'losses': 0, 'pnl': 0}
        if t['outcome'] == 'WIN':
            monthly[m]['wins'] += 1
        else:
            monthly[m]['losses'] += 1
        monthly[m]['pnl'] += t['pnl_pct']
    
    bad_months = sum(1 for v in monthly.values() if v['pnl'] < 0)
    
    return {
        'trades': total, 'wins': w, 'losses': l, 'timeouts': t,
        'wr': round(wr, 1), 'pf': round(pf, 2),
        'pnl': round((capital - 200) / 200 * 100, 2),
        'dd': round(max_dd, 1), 'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2), 'capital': round(capital, 2),
        'bad_months': bad_months, 'monthly': monthly
    }


# ============================================================
# PARAMETER SWEEP
# ============================================================
print("\n" + "=" * 100, flush=True)
print("FAILED BREAKOUT v2 — PARAMETER SWEEP", flush=True)
print("=" * 100, flush=True)

results = []
tested = 0

# Conservative sweep
for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for sl in [0.3, 0.5, 0.75, 1.0]:
        if sl >= tp: continue  # SL < TP
        for hold in [4, 8, 12, 16]:
            for min_conv in [0.5, 0.6, 0.7, 0.8]:
                for lookback in [24, 48]:
                    for return_pct in [0.2, 0.3, 0.5]:
                        for vol_ratio in [0.8, 1.0, 1.3]:
                            for wick_ratio in [0.3, 0.4, 0.5]:
                                for trend in [None, True]:
                                    for session in [None, 'ol', 'ln']:
                                        r = backtest_v2(
                                            tp_pct=tp, sl_pct=sl, hold_hours=hold,
                                            min_conv=min_conv, lookback=lookback,
                                            return_pct=return_pct, min_vol_ratio=vol_ratio,
                                            min_wick_ratio=wick_ratio, trend_filter=trend,
                                            session_filter=session
                                        )
                                        tested += 1
                                        if r and r['trades'] >= 10 and r['wr'] >= 50 and r['pf'] >= 1.3:
                                            results.append({
                                                **r, 'tp': tp, 'sl': sl, 'hold': hold,
                                                'min_conv': min_conv, 'lookback': lookback,
                                                'return_pct': return_pct, 'vol_ratio': vol_ratio,
                                                'wick_ratio': wick_ratio, 'trend': trend,
                                                'session': session or 'all'
                                            })
                                        if tested % 10000 == 0:
                                            print(f"  {tested}...", flush=True)

results.sort(key=lambda x: x['pf'] * (x['wr'] / 100), reverse=True)

# Filter for enterprise grade
enterprise = [r for r in results if r['wr'] >= 65 and r['pf'] >= 2.0 and r['dd'] < 25 and r['bad_months'] <= 3]
good = [r for r in results if r['wr'] >= 55 and r['pf'] >= 1.5 and r['trades'] >= 15]

print(f"\nTested {tested} configs", flush=True)
print(f"Enterprise-grade (WR>=65%, PF>=2.0, DD<25%, bad_m<=3): {len(enterprise)}", flush=True)
print(f"Good (WR>=55%, PF>=1.5, T>=15): {len(good)}", flush=True)

if enterprise:
    print(f"\n{'='*120}", flush=True)
    print(f"{'ENTERPRISE-GRADE RESULTS':^120s}", flush=True)
    print(f"{'='*120}", flush=True)
    print(f"  {'#':>2s} {'TP':>4s} {'SL':>4s} {'RR':>4s} {'Hold':>4s} {'Conv':>4s} {'LB':>3s} {'Ret':>4s} {'Vol':>4s} {'Wck':>4s} {'Trnd':>4s} {'Sess':>3s} | {'Tr':>4s} {'W':>3s} {'L':>3s} {'T':>3s} {'WR':>5s} {'PF':>5s} {'PnL%':>7s} {'DD%':>5s} {'AvgW':>5s} {'AvgL':>5s} {'BadM':>4s}", flush=True)
    print("  " + "-" * 115, flush=True)
    for i, r in enumerate(enterprise[:30]):
        rr = round(r['tp'] / r['sl'], 1)
        tr = 'T' if r['trend'] else 'F'
        print(f"  {i+1:>2d} {r['tp']:>4.1f} {r['sl']:>4.2f} {rr:>4.1f} {r['hold']:>4d} {r['min_conv']:>4.1f} {r['lookback']:>3d} {r['return_pct']:>4.1f} {r['vol_ratio']:>4.1f} {r['wick_ratio']:>4.1f} {tr:>4s} {r['session']:>3s} | {r['trades']:>4d} {r['wins']:>3d} {r['losses']:>3d} {r['timeouts']:>3d} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>+7.1f} {r['dd']:>5.1f} {r['avg_win']:>5.2f} {r['avg_loss']:>5.2f} {r['bad_months']:>4d}", flush=True)

    # Best config detail
    b = enterprise[0]
    print(f"\n{'='*80}", flush=True)
    print(f"BEST CONFIG", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  TP={b['tp']}% SL={b['sl']}% Hold={b['hold']}h Conv>={b['min_conv']}", flush=True)
    print(f"  Lookback={b['lookback']} bars, Return>={b['return_pct']}%", flush=True)
    print(f"  Min Vol Ratio={b['vol_ratio']}x, Min Wick Ratio={b['wick_ratio']}", flush=True)
    print(f"  Trend Filter={'EMA200' if b['trend'] else 'None'}, Session={b['session']}", flush=True)
    print(f"  R:R = 1:{b['tp']/b['sl']:.1f}", flush=True)
    print(f"  {b['trades']}T {b['wr']}%WR {b['pf']}PF PnL={b['pnl']}% DD={b['dd']}%", flush=True)
    print(f"  {b['bad_months']} bad months out of {len(b['monthly'])}", flush=True)
    
    # Monthly detail
    print(f"\n  Monthly breakdown:", flush=True)
    for m, v in sorted(b['monthly'].items()):
        total = v['wins'] + v['losses']
        wr = v['wins'] / total * 100 if total > 0 else 0
        print(f"    {m}: {v['wins']}W/{v['losses']}L ({wr:.0f}%WR) PnL={v['pnl']:+.1f}%", flush=True)

elif good:
    print(f"\nNo enterprise-grade. Showing best 'good' results:", flush=True)
    print(f"  {'#':>2s} {'TP':>4s} {'SL':>4s} {'RR':>4s} {'Hold':>4s} {'Conv':>4s} {'Trnd':>4s} {'Sess':>3s} | {'Tr':>4s} {'WR':>5s} {'PF':>5s} {'PnL%':>7s} {'DD%':>5s} {'BadM':>4s}", flush=True)
    print("  " + "-" * 70, flush=True)
    for i, r in enumerate(good[:20]):
        rr = round(r['tp'] / r['sl'], 1)
        tr = 'T' if r['trend'] else 'F'
        print(f"  {i+1:>2d} {r['tp']:>4.1f} {r['sl']:>4.2f} {rr:>4.1f} {r['hold']:>4d} {r['min_conv']:>4.1f} {tr:>4s} {r['session']:>3s} | {r['trades']:>4d} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>+7.1f} {r['dd']:>5.1f} {r['bad_months']:>4d}", flush=True)
else:
    print("\nNo viable configs found. Strategy may need fundamental rework.", flush=True)
    # Show top 10 by PF regardless
    all_results = []
    for tp in [1.0, 1.5, 2.0]:
        for sl in [0.3, 0.5]:
            if sl >= tp: continue
            r = backtest_v2(tp, sl, 8, 0.6, 48, 0.3, 1.0, 0.4, None, 'ol')
            if r and r['trades'] >= 5:
                all_results.append({**r, 'tp': tp, 'sl': sl})
    all_results.sort(key=lambda x: x['pf'], reverse=True)
    print(f"\nTop configs by PF:", flush=True)
    for r in all_results[:10]:
        print(f"  TP={r['tp']}% SL={r['sl']}% | {r['trades']}T {r['wr']}%WR {r['pf']}PF PnL={r['pnl']}%", flush=True)

print("\nDone", flush=True)
