#!/usr/bin/env python3
"""
VOL ROTATION — 4H BACKTEST
Hypothesis: Volume expanding from compression + directional bias predicts continuation.
On 15m: correct direction but +0.024% < 0.10% costs. Testing on 4h where moves are bigger.
"""
import json, sys, time
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

t0 = time.time()

# Load 1h data
with open('/root/.openclaw/workspace/jimi_audit/data/eth_full_1h.json') as f:
    raw = json.load(f)

N = len(raw)
o1h = [float(c[1]) for c in raw]
h1h = [float(c[2]) for c in raw]
l1h = [float(c[3]) for c in raw]
c1h = [float(c[4]) for c in raw]
v1h = [float(c[5]) for c in raw]
ts1h = [c[0] for c in raw]

print(f"1h candles: {N}")

# Build 4h candles
c4h_o, c4h_h, c4h_l, c4h_c, c4h_v, c4h_ts = [], [], [], [], [], []
i = 0
while i < N:
    end = min(i + 4, N)
    c4h_o.append(o1h[i])
    c4h_h.append(max(h1h[i:end]))
    c4h_l.append(min(l1h[i:end]))
    c4h_c.append(c1h[end - 1])
    c4h_v.append(sum(v1h[i:end]))
    c4h_ts.append(ts1h[i])
    i = end

N4 = len(c4h_o)
print(f"4h candles: {N4}")

# ATR on 4h
atr_period = 14
atr = [0.0] * N4
for i in range(1, N4):
    tr = max(c4h_h[i] - c4h_l[i],
             abs(c4h_h[i] - c4h_c[i - 1]),
             abs(c4h_l[i] - c4h_c[i - 1]))
    if i < atr_period:
        atr[i] = tr
    else:
        atr[i] = (atr[i - 1] * (atr_period - 1) + tr) / atr_period

# ATR percentile (rolling 100-bar window)
atr_pct = [0.5] * N4
for i in range(100, N4):
    window = atr[i - 100:i + 1]
    atr_pct[i] = sum(1 for x in window if x <= atr[i]) / len(window)

# Volume ratio (current vol vs 20-bar average)
vol_ratio = [1.0] * N4
for i in range(20, N4):
    avg = np.mean(c4h_v[i - 20:i])
    vol_ratio[i] = c4h_v[i] / avg if avg > 0 else 1.0

# EMA 200 on 4h
ema200 = [0.0] * N4
ema200[0] = c4h_c[0]
k = 2 / 201
for i in range(1, N4):
    ema200[i] = c4h_c[i] * k + ema200[i - 1] * (1 - k)

# EMA 50 on 4h
ema50 = [0.0] * N4
ema50[0] = c4h_c[0]
k50 = 2 / 51
for i in range(1, N4):
    ema50[i] = c4h_c[i] * k50 + ema50[i - 1] * (1 - k50)

# RSI 14 on 4h
rsi = [50.0] * N4
for i in range(15, N4):
    gains, losses = [], []
    for j in range(i - 13, i + 1):
        d = c4h_c[j] - c4h_c[j - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_g = np.mean(gains)
    avg_l = np.mean(losses)
    if avg_l > 0:
        rs = avg_g / avg_l
        rsi[i] = 100 - 100 / (1 + rs)
    else:
        rsi[i] = 100

# Vol compression detection: ATR percentile < 0.25 = compressed
# Vol expansion: ATR percentile crosses above 0.5 from below 0.25
# Direction: EMA trend + RSI + price action

def detect_signal(i):
    """Detect vol rotation signal at bar i."""
    if i < 200:
        return None
    
    # Need: recent compression followed by expansion
    # Compression: ATR pct was < 0.3 in last 10 bars
    recent_compressed = any(atr_pct[i - j] < 0.30 for j in range(1, 11))
    # Expansion: current ATR pct > 0.5 (above median)
    expanding = atr_pct[i] > 0.50
    # Volume confirmation: current vol > 1.2x average
    vol_confirm = vol_ratio[i] > 1.2
    
    if not (recent_compressed and expanding and vol_confirm):
        return None
    
    # Direction from EMA trend + RSI
    direction = None
    price = c4h_c[i]
    
    # Primary: EMA trend
    ema_up = price > ema200[i] and ema50[i] > ema200[i]
    ema_down = price < ema200[i] and ema50[i] < ema200[i]
    
    # Secondary: RSI momentum
    rsi_bull = 40 < rsi[i] < 70  # not overbought, has room
    rsi_bear = 30 < rsi[i] < 60  # not oversold, has room
    
    # Tertiary: recent price action (last 3 bars direction)
    recent_move = (c4h_c[i] - c4h_c[i - 3]) / c4h_c[i - 3]
    
    if ema_up and rsi_bull and recent_move > 0:
        direction = 'LONG'
    elif ema_down and rsi_bear and recent_move < 0:
        direction = 'SHORT'
    else:
        return None
    
    # Conviction from strength of signals
    vol_strength = min((vol_ratio[i] - 1.0) / 2.0, 0.3)  # 0-0.3
    atr_strength = min(atr_pct[i] / 2.0, 0.25)  # 0-0.25
    trend_strength = min(abs(recent_move) * 10, 0.25)  # 0-0.25
    conviction = min(0.3 + vol_strength + atr_strength + trend_strength, 0.85)
    
    return {
        'direction': direction,
        'conviction': conviction,
        'price': price,
        'atr': atr[i],
        'vol_ratio': vol_ratio[i],
        'atr_pct': atr_pct[i],
        'rsi': rsi[i],
    }


def backtest(tp_pct, sl_pct, hold_bars, risk_pct=0.02, leverage=25, fee=0.001, init_cap=200):
    """Run backtest with given parameters."""
    cap = float(init_cap)
    peak = cap
    max_dd = 0.0
    wins = 0
    losses = 0
    total = 0
    gross_p = 0.0
    gross_l = 0.0
    trades = []
    monthly = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    
    i = 200  # skip warmup
    while i < N4 - 1:
        sig = detect_signal(i)
        if sig is None:
            i += 1
            continue
        
        direction = sig['direction']
        entry_price = c4h_o[i + 1]  # enter at next bar open
        atr_val = sig['atr']
        
        if direction == 'LONG':
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)
        
        # Position sizing
        risk_per_trade = cap * risk_pct
        sl_dist = abs(entry_price - sl_price)
        if sl_dist == 0:
            i += 1
            continue
        pos_size = min(risk_per_trade / sl_dist, cap * leverage / entry_price)
        if pos_size <= 0:
            i += 1
            continue
        
        # Simulate trade
        closed = False
        for j in range(i + 1, min(i + 1 + hold_bars, N4)):
            hit = False
            exit_price = 0.0
            
            if direction == 'LONG':
                if c4h_h[j] >= tp_price:
                    hit = True
                    exit_price = tp_price
                elif c4h_l[j] <= sl_price:
                    hit = True
                    exit_price = sl_price
            else:
                if c4h_l[j] <= tp_price:
                    hit = True
                    exit_price = tp_price
                elif c4h_h[j] >= sl_price:
                    hit = True
                    exit_price = sl_price
            
            if hit:
                pnl = (exit_price - entry_price) * pos_size if direction == 'LONG' else (entry_price - exit_price) * pos_size
                pnl -= entry_price * pos_size * fee * 2  # round-trip fees
                cap += pnl
                total += 1
                
                mk = datetime.fromtimestamp(c4h_ts[i] / 1000, tz=timezone.utc).strftime('%Y-%m')
                monthly[mk]['trades'] += 1
                monthly[mk]['pnl'] += pnl
                
                if pnl > 0:
                    wins += 1
                    gross_p += pnl
                    monthly[mk]['wins'] += 1
                else:
                    losses += 1
                    gross_l += abs(pnl)
                
                if cap > peak:
                    peak = cap
                dd = (peak - cap) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                
                trades.append({
                    'bar': i, 'dir': direction, 'entry': round(entry_price, 2),
                    'exit': round(exit_price, 2), 'pnl': round(pnl, 2),
                    'conviction': round(sig['conviction'], 3),
                    'vol_ratio': round(sig['vol_ratio'], 2),
                    'atr_pct': round(sig['atr_pct'], 3),
                })
                
                closed = True
                i = j + 1
                break
        
        if not closed:
            # Timeout: close at bar end
            j = min(i + hold_bars, N4 - 1)
            exit_price = c4h_c[j]
            pnl = (exit_price - entry_price) * pos_size if direction == 'LONG' else (entry_price - exit_price) * pos_size
            pnl -= entry_price * pos_size * fee * 2
            cap += pnl
            total += 1
            
            mk = datetime.fromtimestamp(c4h_ts[i] / 1000, tz=timezone.utc).strftime('%Y-%m')
            monthly[mk]['trades'] += 1
            monthly[mk]['pnl'] += pnl
            
            if pnl > 0:
                wins += 1
                gross_p += pnl
                monthly[mk]['wins'] += 1
            else:
                losses += 1
                gross_l += abs(pnl)
            
            if cap > peak:
                peak = cap
            dd = (peak - cap) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
            
            i = j + 1
    
    wr = wins / total * 100 if total > 0 else 0
    pf = gross_p / gross_l if gross_l > 0 else float('inf')
    avg_win = gross_p / wins if wins > 0 else 0
    avg_loss = gross_l / losses if losses > 0 else 0
    expectancy = (wr / 100 * avg_win - (1 - wr / 100) * avg_loss) if total > 0 else 0
    
    return {
        'cap': round(cap, 2), 'pk': round(peak, 2), 'dd': round(max_dd, 1),
        'trades': total, 'wins': wins, 'wr': round(wr, 1),
        'pf': round(pf, 2), 'expectancy': round(expectancy, 2),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'monthly': dict(monthly),
        'sample_trades': trades[:10],
    }


# === PARAMETER GRID ===
print("\n=== VOL ROTATION 4H BACKTEST ===")
print(f"Data: {N4} 4h candles ({datetime.fromtimestamp(c4h_ts[0]/1000, tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(c4h_ts[-1]/1000, tz=timezone.utc).strftime('%Y-%m-%d')})")
print(f"Round-trip cost: 0.10%\n")

configs = [
    # (tp%, sl%, hold_bars, label)
    (1.0, 0.75, 6, "conservative"),
    (1.5, 1.0, 8, "standard"),
    (2.0, 1.5, 12, "wide"),
    (2.5, 1.5, 16, "very_wide"),
    (1.0, 0.5, 4, "tight_fast"),
    (1.5, 0.75, 6, "medium"),
]

for tp, sl, hold, label in configs:
    r = backtest(tp_pct=tp/100, sl_pct=sl/100, hold_bars=hold)
    profitable = r['pf'] > 1.0 and r['wr'] > 50
    sign = "✅" if profitable else "❌"
    print(f"{sign} {label:15s} TP={tp}% SL={sl}% hold={hold}4h | "
          f"trades={r['trades']:4d} WR={r['wr']:5.1f}% PF={r['pf']:5.2f} "
          f"cap=${r['cap']:8.2f} dd={r['dd']:5.1f}% exp={r['expectancy']:7.2f}")
    
    if r['trades'] > 0 and profitable:
        # Show monthly breakdown
        for mk in sorted(r['monthly'].keys()):
            m = r['monthly'][mk]
            if m['trades'] > 0:
                mwr = m['wins'] / m['trades'] * 100
                print(f"    {mk}: {m['trades']} trades, {m['wins']} wins, WR={mwr:.0f}%, PnL=${m['pnl']:.2f}")

# Isolation gate check on best config
print("\n=== ISOLATION GATE CHECK ===")
best = backtest(tp_pct=1.5/100, sl_pct=1.0/100, hold_bars=8)
if best['trades'] >= 50:
    # Simple z-test: is WR significantly > 50%?
    from math import sqrt
    n = best['trades']
    p_hat = best['wins'] / n
    se = sqrt(0.5 * 0.5 / n)
    z = (p_hat - 0.5) / se if se > 0 else 0
    p_val = 1.0
    # Approximate p-value
    if z > 0:
        import math
        p_val = 0.5 * (1 - math.erf(z / sqrt(2)))
    
    mean_return = (best['cap'] - 200) / 200 / best['trades'] * 100
    print(f"Events: {best['trades']}")
    print(f"WR: {best['wr']}%")
    print(f"Mean return per trade: {mean_return:.4f}%")
    print(f"Round-trip cost: 0.10%")
    print(f"Edge over costs: {mean_return - 0.10:.4f}%")
    print(f"z-stat: {z:.3f}")
    print(f"p-value (approx): {p_val:.4f}")
    print(f"Gate: {'PASS' if p_val < 0.1 and mean_return > 0.10 else 'FAIL'}")
else:
    print(f"Only {best['trades']} events — need 50+ for gate")

print(f"\nTotal time: {time.time()-t0:.1f}s")
