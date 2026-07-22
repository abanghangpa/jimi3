#!/usr/bin/env python3
"""Squeeze V6 Optimizer — Fast version with indexed signals."""
import csv, json, os
from datetime import datetime, timezone
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def main():
    print("=" * 70)
    print("SQUEEZE V6 OPTIMIZER (FAST)")
    print("=" * 70)
    
    closes, highs, lows, volumes = load_data()
    n = len(closes)
    print(f"Candles: {n}")
    
    # Pre-compute
    bb_mid = rolling_mean(closes, 20)
    bb_std = rolling_std(closes, 20)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = np.where(bb_mid > 0, (bb_upper - bb_lower) / bb_mid, 0)
    atr14 = calc_atr(highs, lows, closes, 14)
    vol_ma = rolling_mean(volumes, 20)
    ema_50 = ema(closes, 50)
    
    # Generate signals (indexed by bar)
    sig_by_bar = {}
    for i in range(60, n):
        price = closes[i]
        if np.isnan(bb_width[i]) or bb_width[i] > 0.02:
            continue
        a = atr14[i]
        if np.isnan(a) or a == 0:
            continue
        
        if price > bb_upper[i]:
            direction = "S"  # SHORT
        elif price < bb_lower[i]:
            direction = "L"  # LONG
        else:
            continue
        
        vr = volumes[i] / vol_ma[i] if not np.isnan(vol_ma[i]) and vol_ma[i] > 0 else 1.0
        e50 = ema_50[i] if not np.isnan(ema_50[i]) else price
        
        sig_by_bar[i] = {"d": direction, "e": price, "a": a, "vr": vr, "e50": e50}
    
    total_sigs = len(sig_by_bar)
    print(f"Signals: {total_sigs}")
    
    # Grid search
    tp_mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
    sl_mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    holds = [4, 6, 8, 12, 16]
    trailing_opts = [False, True]
    time_exits = [0, 4, 6]
    trend_opts = [False, True]
    vol_opts = [False, True]
    
    results = []
    tested = 0
    
    for tp in tp_mults:
        for sl in sl_mults:
            if tp / sl < 0.8: continue
            for hold in holds:
                for trailing in trailing_opts:
                    for te in time_exits:
                        if te > 0 and te >= hold: continue
                        for trend in trend_opts:
                            for vol in vol_opts:
                                # Run backtest
                                trades = []
                                positions = []
                                last_sig = -999
                                fee = 0.0004
                                
                                for i in range(60, n):
                                    # Exits
                                    new = []
                                    for p in positions:
                                        p["b"] += 1
                                        ex = False; o = "T"; ep = closes[i]
                                        
                                        if trailing and p["b"] > 0:
                                            if p["d"] == "L":
                                                p["sl"] = max(p["sl"], highs[i] - p["a"])
                                            else:
                                                p["sl"] = min(p["sl"], lows[i] + p["a"])
                                        
                                        if p["d"] == "L":
                                            if lows[i] <= p["sl"]: ex = True; ep = p["sl"]; o = "L"
                                            elif highs[i] >= p["tp"]: ex = True; ep = p["tp"]; o = "W"
                                        else:
                                            if highs[i] >= p["sl"]: ex = True; ep = p["sl"]; o = "L"
                                            elif lows[i] <= p["tp"]: ex = True; ep = p["tp"]; o = "W"
                                        
                                        if not ex and te > 0 and p["b"] >= te:
                                            ex = True; ep = closes[i]
                                        if not ex and p["b"] >= hold:
                                            ex = True; ep = closes[i]
                                        
                                        if ex:
                                            pnl = ((ep - p["e"]) / p["e"] * 100) if p["d"] == "L" else ((p["e"] - ep) / p["e"] * 100)
                                            pnl -= fee * 100
                                            trades.append(pnl)
                                        else:
                                            new.append(p)
                                    positions = new
                                    if positions: continue
                                    
                                    sig = sig_by_bar.get(i)
                                    if not sig: continue
                                    if i - last_sig < 2: continue
                                    
                                    # Filters
                                    if trend:
                                        if sig["d"] == "S" and sig["e"] > sig["e50"]: continue
                                        if sig["d"] == "L" and sig["e"] < sig["e50"]: continue
                                    if vol and sig["vr"] > 1.5: continue
                                    
                                    a = sig["a"]
                                    entry = sig["e"]
                                    if sig["d"] == "L":
                                        s = entry - a * sl; t = entry + a * tp
                                    else:
                                        s = entry + a * sl; t = entry - a * tp
                                    
                                    risk = abs(entry - s)
                                    reward = abs(entry - t)
                                    if risk == 0 or reward / risk < 1.0: continue
                                    
                                    positions.append({"d": sig["d"], "e": entry, "sl": s, "tp": t, "b": 0, "a": a})
                                    last_sig = i
                                
                                # Close remaining
                                for p in positions:
                                    ep = closes[-1]
                                    pnl = ((ep - p["e"]) / p["e"] * 100) if p["d"] == "L" else ((p["e"] - ep) / p["e"] * 100)
                                    pnl -= fee * 100
                                    trades.append(pnl)
                                
                                tested += 1
                                
                                if len(trades) < 50: continue
                                
                                total = len(trades)
                                wins = sum(1 for t in trades if t > 0)
                                losses = sum(1 for t in trades if t < 0)
                                wr = wins / total * 100
                                gp = sum(t for t in trades if t > 0)
                                gl = abs(sum(t for t in trades if t < 0))
                                pf = gp / gl if gl > 0 else float("inf")
                                pnl = sum(trades)
                                
                                eq = [0]
                                for t in trades: eq.append(eq[-1] + t)
                                eq = np.array(eq)
                                pk = np.maximum.accumulate(eq)
                                dd = float(np.max(pk - eq))
                                
                                results.append({
                                    "tp": tp, "sl": sl, "hold": hold,
                                    "trail": trailing, "texit": te,
                                    "trend": trend, "vol": vol,
                                    "trades": total, "wins": wins, "losses": losses,
                                    "wr": round(wr, 1), "pf": round(pf, 2),
                                    "pnl": round(pnl, 2), "max_dd": round(dd, 2),
                                })
    
    results.sort(key=lambda x: (-x["pf"], -x["trades"]))
    
    print(f"\nTested {tested} combinations, {len(results)} with 50+ trades")
    
    print(f"\n{'#':<4} {'TP':<5} {'SL':<5} {'H':<4} {'Tr':<4} {'TE':<4} {'Trnd':<5} {'Vol':<4} {'Trades':<7} {'W':<5} {'L':<5} {'WR':<6} {'PF':<6} {'PnL%':<8} {'DD%':<6}")
    print("-" * 85)
    for i, r in enumerate(results[:30]):
        tr = "Y" if r["trail"] else "N"
        te = str(r["texit"]) if r["texit"] > 0 else "-"
        tn = "Y" if r["trend"] else "N"
        v = "Y" if r["vol"] else "N"
        print(f"  {i+1:<3} {r['tp']:<5} {r['sl']:<5} {r['hold']:<4} {tr:<4} {te:<4} {tn:<5} {v:<4} "
              f"{r['trades']:<7} {r['wins']:<5} {r['losses']:<5} {r['wr']:<6} {r['pf']:<6} {r['pnl']:<8} {r['max_dd']:<6}")
    
    if results:
        best = results[0]
        print(f"\nBEST: TP={best['tp']} SL={best['sl']} Hold={best['hold']} "
              f"Trail={'Y' if best['trail'] else 'N'} TExit={best['texit']} "
              f"Trend={'Y' if best['trend'] else 'N'} Vol={'Y' if best['vol'] else 'N'}")
        print(f"  Trades={best['trades']} WR={best['wr']}% PF={best['pf']} PnL={best['pnl']}% DD={best['max_dd']}%")
    
    out = {"candles": n, "signals": total_sigs, "tested": tested,
           "results_50plus": len(results), "top_30": results[:30],
           "best": results[0] if results else None}
    out_path = os.path.join(BASE, "data", "5agent_backtest", "squeeze_v6_optimizer.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
