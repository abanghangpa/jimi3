#!/usr/bin/env python3
"""
SQUEEZE V6 OPTIMIZER — Find optimal TP/SL/hold parameters.

Tests thousands of combinations across 89,000 candles:
- TP multipliers: 0.5 to 4.0 ATR
- SL multipliers: 0.5 to 3.0 ATR
- Hold periods: 2 to 16 bars
- Trailing stops: on/off
- Time-based exits: on/off
- Conviction sizing: on/off

Target: Maximize PF while maintaining min 100 trades for significance.
"""

import csv, json, os, sys, itertools
from datetime import datetime, timezone
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data():
    csv_path = os.path.join(BASE, "data", "history", "ETHUSDT_15m.csv")
    closes, highs, lows, volumes = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            closes.append(float(row["close"]))
            highs.append(float(row["high"]))
            lows.append(float(row["low"]))
            volumes.append(float(row["volume"]))
    return np.array(closes), np.array(highs), np.array(lows), np.array(volumes)


# ═══════════════════════════════════════════════════════════════
# INDICATORS (pre-compute once)
# ═══════════════════════════════════════════════════════════════

def rolling_mean(arr, p):
    r = np.full(len(arr), np.nan)
    cs = np.cumsum(arr)
    r[p-1:] = (cs[p-1:] - np.concatenate([[0], cs[:-p]])) / p
    return r

def rolling_std(arr, p):
    r = np.full(len(arr), np.nan)
    for i in range(p-1, len(arr)):
        r[i] = np.std(arr[i-p+1:i+1])
    return r

def ema(arr, p):
    r = np.full(len(arr), np.nan)
    if len(arr) < p: return r
    r[p-1] = np.mean(arr[:p])
    m = 2 / (p + 1)
    for i in range(p, len(arr)):
        r[i] = arr[i] * m + r[i-1] * (1 - m)
    return r

def calc_atr(highs, lows, closes, p=14):
    trs = np.maximum(highs[1:]-lows[1:], np.maximum(np.abs(highs[1:]-closes[:-1]), np.abs(lows[1:]-closes[:-1])))
    r = np.full(len(highs), np.nan)
    for i in range(p, len(highs)):
        r[i] = np.mean(trs[i-p:i])
    return r

def precompute(closes, highs, lows, volumes):
    """Pre-compute all indicators once."""
    bb_mid = rolling_mean(closes, 20)
    bb_std = rolling_std(closes, 20)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = np.where(bb_mid > 0, (bb_upper - bb_lower) / bb_mid, 0)
    
    kc_mid = ema(closes, 20)
    atr14 = calc_atr(highs, lows, closes, 14)
    kc_upper = kc_mid + 1.5 * atr14
    kc_lower = kc_mid - 1.5 * atr14
    
    vol_ma = rolling_mean(volumes, 20)
    ema_50 = ema(closes, 50)
    ema_20 = ema(closes, 20)
    
    in_squeeze = (kc_upper < bb_upper) & (kc_lower > bb_lower)
    
    return {
        "bb_mid": bb_mid, "bb_upper": bb_upper, "bb_lower": bb_lower,
        "bb_width": bb_width, "atr": atr14, "vol_ma": vol_ma,
        "ema_50": ema_50, "ema_20": ema_20, "in_squeeze": in_squeeze,
    }


# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATION (same as V6)
# ═══════════════════════════════════════════════════════════════

def generate_signals(closes, highs, lows, volumes, ind, n):
    """Generate all V6 fade signals (once, reuse across all param combos)."""
    signals = []
    in_sqz_prev = False
    breakout_bar = None; breakout_dir = None; breakout_extreme = None
    
    for i in range(60, n):
        price = closes[i]
        
        if np.isnan(ind["bb_width"][i]): continue
        
        in_sqz = bool(ind["in_squeeze"][i]) if not np.isnan(ind["in_squeeze"][i]) else False
        
        if in_sqz_prev and not in_sqz:
            if price > ind["bb_upper"][i]:
                breakout_bar = i; breakout_dir = "LONG"; breakout_extreme = highs[i]
            elif price < ind["bb_lower"][i]:
                breakout_bar = i; breakout_dir = "SHORT"; breakout_extreme = lows[i]
            else:
                breakout_bar = None; breakout_dir = None
        
        in_sqz_prev = in_sqz
        
        if breakout_bar is None or in_sqz: continue
        if i - breakout_bar > 12: breakout_bar = None; continue
        
        a = ind["atr"][i]
        if np.isnan(a) or a == 0: continue
        
        # Trend filter
        if not np.isnan(ind["ema_50"][i]):
            if breakout_dir == "LONG" and price > ind["ema_50"][i]:
                breakout_bar = None; continue  # don't fade with trend
            if breakout_dir == "SHORT" and price < ind["ema_50"][i]:
                breakout_bar = None; continue
        
        # Volume filter
        if not np.isnan(ind["vol_ma"][i]) and ind["vol_ma"][i] > 0:
            vol_ratio = volumes[i] / ind["vol_ma"][i]
            if vol_ratio > 1.5:
                breakout_bar = None; continue
        else:
            vol_ratio = 1.0
        
        # Generate fade signal
        if breakout_dir == "LONG":
            direction = "SHORT"
            entry = price
            ref_price = breakout_extreme
        else:
            direction = "LONG"
            entry = price
            ref_price = breakout_extreme
        
        signals.append({
            "bar": i, "direction": direction, "entry": entry,
            "ref": ref_price, "atr": a,
            "bb_width": ind["bb_width"][i],
            "vol_ratio": vol_ratio,
        })
        
        breakout_bar = None  # consumed
    
    return signals


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE (parameterized)
# ═══════════════════════════════════════════════════════════════

def backtest_signals(signals, closes, highs, lows, params):
    """
    Run backtest on pre-generated signals with given parameters.
    
    params: {
        "tp_mult": float,       # TP in ATR multiples
        "sl_mult": float,       # SL in ATR multiples
        "hold": int,            # max hold bars
        "trailing": bool,       # trailing stop
        "trailing_atr": float,  # trailing stop distance in ATR
        "time_exit": int,       # exit after N bars if not hit TP/SL (0=off)
        "min_rr": float,        # minimum risk/reward ratio
        "cooldown": int,        # bars between signals
        "vol_sizing": bool,     # size by conviction (inverse vol)
    }
    """
    tp_mult = params.get("tp_mult", 1.5)
    sl_mult = params.get("sl_mult", 1.0)
    hold = params.get("hold", 8)
    trailing = params.get("trailing", False)
    trailing_atr = params.get("trailing_atr", 1.0)
    time_exit = params.get("time_exit", 0)
    min_rr = params.get("min_rr", 1.0)
    cooldown = params.get("cooldown", 2)
    vol_sizing = params.get("vol_sizing", False)
    
    trades = []
    positions = []
    last_sig = -999
    fee_rate = 0.0004
    
    for i in range(60, len(closes)):
        # Exits
        new = []
        for p in positions:
            p["b"] += 1
            ex = False
            o = "T"
            ep = closes[i]
            
            # Trailing stop
            if trailing and p["b"] > 0:
                if p["d"] == "LONG":
                    new_trail = highs[i] - p["atr"] * trailing_atr
                    p["sl"] = max(p["sl"], new_trail)
                else:
                    new_trail = lows[i] + p["atr"] * trailing_atr
                    p["sl"] = min(p["sl"], new_trail)
            
            # Check SL/TP
            if p["d"] == "LONG":
                if lows[i] <= p["sl"]: ex = True; ep = p["sl"]; o = "L"
                elif highs[i] >= p["tp"]: ex = True; ep = p["tp"]; o = "W"
            else:
                if highs[i] >= p["sl"]: ex = True; ep = p["sl"]; o = "L"
                elif lows[i] <= p["tp"]: ex = True; ep = p["tp"]; o = "W"
            
            # Time exit
            if not ex and time_exit > 0 and p["b"] >= time_exit:
                ex = True; ep = closes[i]; o = "T"
            
            # Max hold
            if not ex and p["b"] >= hold:
                ex = True; ep = closes[i]; o = "T"
            
            if ex:
                if p["d"] == "LONG":
                    pnl = (ep - p["e"]) / p["e"] * 100
                else:
                    pnl = (p["e"] - ep) / p["e"] * 100
                pnl -= fee_rate * 100
                
                # Vol-based sizing
                size_mult = 1.0
                if vol_sizing and p.get("vol_ratio", 1.0) < 0.8:
                    size_mult = 1.5  # bigger size on low-vol breakouts
                
                pnl_dollar = pnl * size_mult
                trades.append({"outcome": o, "pnl": pnl, "pnl_dollar": pnl_dollar})
            else:
                new.append(p)
        positions = new
        
        if positions: continue
        
        # Find next signal
        sig = None
        for s in signals:
            if s["bar"] <= last_sig: continue
            if s["bar"] > i: break
            if i - s["bar"] < cooldown: continue
            sig = s
            break
        
        if not sig or sig["bar"] != i: continue
        
        a = sig["atr"]
        entry = sig["entry"]
        
        # TP/SL
        if sig["direction"] == "LONG":
            sl = entry - a * sl_mult
            tp = entry + a * tp_mult
        else:
            sl = entry + a * sl_mult
            tp = entry - a * tp_mult
        
        # RR check
        risk = abs(entry - sl)
        reward = abs(entry - tp)
        if risk == 0 or reward / risk < min_rr: continue
        
        positions.append({
            "d": sig["direction"], "e": entry, "sl": sl, "tp": tp,
            "b": 0, "hb": hold, "atr": a,
            "vol_ratio": sig.get("vol_ratio", 1.0),
        })
        last_sig = i
    
    # Close remaining
    for p in positions:
        ep = closes[-1]
        if p["d"] == "LONG": pnl = (ep - p["e"]) / p["e"] * 100
        else: pnl = (p["e"] - ep) / p["e"] * 100
        pnl -= fee_rate * 100
        trades.append({"outcome": "T", "pnl": pnl, "pnl_dollar": pnl})
    
    if not trades:
        return {"trades": 0, "wr": 0, "pf": 0, "pnl": 0, "avg_win": 0, "avg_loss": 0, "max_dd": 0}
    
    wins = [t for t in trades if t["outcome"] == "W"]
    losses = [t for t in trades if t["outcome"] == "L"]
    timeouts = [t for t in trades if t["outcome"] == "T"]
    
    total = len(trades)
    wr = len(wins) / total * 100 if total else 0
    
    gp = sum(t["pnl_dollar"] for t in trades if t["pnl_dollar"] > 0)
    gl = abs(sum(t["pnl_dollar"] for t in trades if t["pnl_dollar"] < 0))
    pf = gp / gl if gl > 0 else float("inf")
    
    pnl = sum(t["pnl_dollar"] for t in trades)
    
    # Max drawdown
    equity = [0]
    for t in trades:
        equity.append(equity[-1] + t["pnl_dollar"])
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = np.max(peak - equity) if len(equity) > 0 else 0
    
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    
    return {
        "trades": total, "wins": len(wins), "losses": len(losses), "timeouts": len(timeouts),
        "wr": round(wr, 1), "pf": round(pf, 2), "pnl": round(pnl, 2),
        "avg_win": round(avg_win, 4), "avg_loss": round(avg_loss, 4),
        "max_dd": round(dd, 2),
    }


# ═══════════════════════════════════════════════════════════════
# GRID SEARCH
# ═══════════════════════════════════════════════════════════════

def grid_search(signals, closes, highs, lows):
    """Test all parameter combinations and rank by PF."""
    
    # Parameter grid
    tp_mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
    sl_mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    holds = [4, 6, 8, 12, 16]
    trailing_options = [False, True]
    time_exits = [0, 4, 6]
    
    total_combos = len(tp_mults) * len(sl_mults) * len(holds) * len(trailing_options) * len(time_exits)
    print(f"📊 Testing {total_combos} parameter combinations...")
    
    results = []
    tested = 0
    
    for tp in tp_mults:
        for sl in sl_mults:
            if tp / sl < 0.8: continue  # skip bad RR
            for hold in holds:
                for trailing in trailing_options:
                    for time_exit in time_exits:
                        if time_exit > 0 and time_exit >= hold:
                            continue
                        
                        params = {
                            "tp_mult": tp, "sl_mult": sl, "hold": hold,
                            "trailing": trailing, "trailing_atr": 1.0,
                            "time_exit": time_exit, "min_rr": 1.0,
                            "cooldown": 2, "vol_sizing": False,
                        }
                        
                        r = backtest_signals(signals, closes, highs, lows, params)
                        tested += 1
                        
                        if r["trades"] >= 50:  # minimum significance
                            results.append({
                                "tp": tp, "sl": sl, "hold": hold,
                                "trailing": trailing, "time_exit": time_exit,
                                **r,
                            })
                        
                        if tested % 500 == 0:
                            print(f"   Tested {tested}/{total_combos}...")
    
    # Sort by PF (desc), then by trades (desc)
    results.sort(key=lambda x: (-x["pf"], -x["trades"]))
    
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SQUEEZE V6 OPTIMIZER — Grid Search")
    print("=" * 70)
    
    print("\n📊 Loading data...")
    closes, highs, lows, volumes = load_data()
    n = len(closes)
    print(f"   Candles: {n}")
    
    print("📊 Pre-computing indicators...")
    ind = precompute(closes, highs, lows, volumes)
    
    print("📊 Generating signals...")
    signals = generate_signals(closes, highs, lows, volumes, ind, n)
    print(f"   Signals: {len(signals)}")
    
    # Run grid search
    results = grid_search(signals, closes, highs, lows)
    
    # Print top 30 results
    print(f"\n{'='*90}")
    print(f"TOP 30 PARAMETER COMBINATIONS (by PF)")
    print(f"{'='*90}")
    print(f"{'#':<4} {'TP':<6} {'SL':<6} {'Hold':<6} {'Trail':<7} {'TExit':<6} {'Trades':<8} {'W':<5} {'L':<5} {'WR':<7} {'PF':<6} {'PnL%':<8} {'DD%':<6}")
    print("-" * 90)
    
    for i, r in enumerate(results[:30]):
        trail = "Yes" if r["trailing"] else "No"
        texit = str(r["time_exit"]) if r["time_exit"] > 0 else "Off"
        print(f"  {i+1:<3} {r['tp']:<6} {r['sl']:<6} {r['hold']:<6} {trail:<7} {texit:<6} "
              f"{r['trades']:<8} {r['wins']:<5} {r['losses']:<5} {r['wr']:<7} {r['pf']:<6} {r['pnl']:<8} {r['max_dd']:<6}")
    
    # Best overall
    if results:
        best = results[0]
        print(f"\n🏆 BEST: TP={best['tp']} SL={best['sl']} Hold={best['hold']} "
              f"Trail={'Yes' if best['trailing'] else 'No'} TExit={best['time_exit']}")
        print(f"   Trades: {best['trades']} WR: {best['wr']}% PF: {best['pf']} "
              f"PnL: {best['pnl']}% MaxDD: {best['max_dd']}%")
    
    # Best with PF > 2.0 and WR > 15%
    high_pf = [r for r in results if r["pf"] >= 2.0 and r["wr"] >= 15 and r["trades"] >= 100]
    if high_pf:
        print(f"\n📈 HIGH PF + MIN WR (PF>=2.0, WR>=15%, trades>=100):")
        for r in high_pf[:10]:
            trail = "Y" if r["trailing"] else "N"
            print(f"   TP={r['tp']} SL={r['sl']} Hold={r['hold']} T={trail} "
                  f"Trades={r['trades']} WR={r['wr']}% PF={r['pf']} PnL={r['pnl']}%")
    
    # Best with max trades (most robust)
    robust = sorted([r for r in results if r["pf"] >= 1.5 and r["trades"] >= 200],
                    key=lambda x: -x["trades"])
    if robust:
        print(f"\n🛡️ MOST ROBUST (PF>=1.5, trades>=200, sorted by trade count):")
        for r in robust[:10]:
            trail = "Y" if r["trailing"] else "N"
            print(f"   TP={r['tp']} SL={r['sl']} Hold={r['hold']} T={trail} "
                  f"Trades={r['trades']} WR={r['wr']}% PF={r['pf']} PnL={r['pnl']}%")
    
    # Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candles": n, "signals": len(signals),
        "total_tested": len(results),
        "top_30": results[:30],
        "best": results[0] if results else None,
        "high_pf": [r for r in results if r["pf"] >= 2.0 and r["wr"] >= 15 and r["trades"] >= 100][:10],
        "robust": sorted([r for r in results if r["pf"] >= 1.5 and r["trades"] >= 200], key=lambda x: -x["trades"])[:10],
    }
    out_path = os.path.join(BASE, "data", "5agent_backtest", "squeeze_v6_optimizer.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Saved: {out_path}")


if __name__ == "__main__":
    main()
