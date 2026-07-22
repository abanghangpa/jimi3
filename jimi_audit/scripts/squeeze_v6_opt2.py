#!/usr/bin/env python3
"""Squeeze V6 Optimizer — Fixed signal generation to match V6-A raw invert."""
import csv, json, os, sys
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

def precompute(closes, highs, lows, volumes):
    bb_mid = rolling_mean(closes, 20)
    bb_std = rolling_std(closes, 20)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = np.where(bb_mid > 0, (bb_upper - bb_lower) / bb_mid, 0)
    atr14 = calc_atr(highs, lows, closes, 14)
    vol_ma = rolling_mean(volumes, 20)
    ema_50 = ema(closes, 50)
    return {"bb_mid": bb_mid, "bb_upper": bb_upper, "bb_lower": bb_lower,
            "bb_width": bb_width, "atr": atr14, "vol_ma": vol_ma, "ema_50": ema_50}

def generate_signals(closes, highs, lows, volumes, ind, n):
    """
    Generate V6-A raw invert signals.
    This matches the backtest that produced 4805 trades / PF 2.52.
    Every BB squeeze breakout = fade signal (invert).
    """
    signals = []
    for i in range(60, n):
        price = closes[i]
        if np.isnan(ind["bb_width"][i]) or ind["bb_width"][i] > 0.02:
            continue
        a = ind["atr"][i]
        if np.isnan(a) or a == 0:
            continue
        
        # Check if price is breaking out of BB
        if price > ind["bb_upper"][i]:
            direction = "SHORT"  # fade the breakout
            entry = price
        elif price < ind["bb_lower"][i]:
            direction = "LONG"  # fade the breakout
            entry = price
        else:
            continue
        
        signals.append({
            "bar": i, "direction": direction, "entry": entry,
            "atr": a, "bb_width": ind["bb_width"][i],
            "vol_ratio": volumes[i] / ind["vol_ma"][i] if not np.isnan(ind["vol_ma"][i]) and ind["vol_ma"][i] > 0 else 1.0,
            "ema_50": ind["ema_50"][i] if not np.isnan(ind["ema_50"][i]) else price,
        })
    return signals

def backtest(signals, closes, highs, lows, params):
    tp_mult = params.get("tp_mult", 1.5)
    sl_mult = params.get("sl_mult", 1.0)
    hold = params.get("hold", 8)
    trailing = params.get("trailing", False)
    trailing_atr = params.get("trailing_atr", 1.0)
    time_exit = params.get("time_exit", 0)
    cooldown = params.get("cooldown", 2)
    trend_filter = params.get("trend_filter", False)
    vol_filter = params.get("vol_filter", False)
    
    trades = []
    positions = []
    last_sig = -999
    fee = 0.0004
    
    for i in range(60, len(closes)):
        new = []
        for p in positions:
            p["b"] += 1
            ex = False; o = "T"; ep = closes[i]
            
            if trailing and p["b"] > 0:
                if p["d"] == "LONG":
                    p["sl"] = max(p["sl"], highs[i] - p["a"] * trailing_atr)
                else:
                    p["sl"] = min(p["sl"], lows[i] + p["a"] * trailing_atr)
            
            if p["d"] == "LONG":
                if lows[i] <= p["sl"]: ex = True; ep = p["sl"]; o = "L"
                elif highs[i] >= p["tp"]: ex = True; ep = p["tp"]; o = "W"
            else:
                if highs[i] >= p["sl"]: ex = True; ep = p["sl"]; o = "L"
                elif lows[i] <= p["tp"]: ex = True; ep = p["tp"]; o = "W"
            
            if not ex and time_exit > 0 and p["b"] >= time_exit:
                ex = True; ep = closes[i]
            if not ex and p["b"] >= hold:
                ex = True; ep = closes[i]
            
            if ex:
                pnl = ((ep - p["e"]) / p["e"] * 100) if p["d"] == "LONG" else ((p["e"] - ep) / p["e"] * 100)
                pnl -= fee * 100
                trades.append({"o": o, "pnl": pnl})
            else:
                new.append(p)
        positions = new
        if positions: continue
        
        # Find signal at this bar
        sig = None
        for s in signals:
            if s["bar"] < i: continue
            if s["bar"] > i: break
            sig = s
            break
        
        if not sig: continue
        if i - last_sig < cooldown: continue
        
        # Apply filters
        if trend_filter:
            if sig["direction"] == "SHORT" and sig["entry"] > sig["ema_50"]:
                continue  # don't fade LONG breakout in uptrend
            if sig["direction"] == "LONG" and sig["entry"] < sig["ema_50"]:
                continue  # don't fade SHORT breakout in downtrend
        
        if vol_filter and sig["vol_ratio"] > 1.5:
            continue  # skip high-volume breakouts
        
        a = sig["atr"]
        entry = sig["entry"]
        
        if sig["direction"] == "LONG":
            sl = entry - a * sl_mult
            tp = entry + a * tp_mult
        else:
            sl = entry + a * sl_mult
            tp = entry - a * tp_mult
        
        risk = abs(entry - sl)
        reward = abs(entry - tp)
        if risk == 0 or reward / risk < 1.0: continue
        
        positions.append({"d": sig["direction"], "e": entry, "sl": sl, "tp": tp, "b": 0, "a": a})
        last_sig = i
    
    for p in positions:
        ep = closes[-1]
        pnl = ((ep - p["e"]) / p["e"] * 100) if p["d"] == "LONG" else ((p["e"] - ep) / p["e"] * 100)
        pnl -= fee * 100
        trades.append({"o": "T", "pnl": pnl})
    
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "wr": 0, "pf": 0, "pnl": 0, "avg_win": 0, "avg_loss": 0, "max_dd": 0}
    
    wins = [t for t in trades if t["o"] == "W"]
    losses = [t for t in trades if t["o"] == "L"]
    total = len(trades)
    wr = len(wins) / total * 100
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gp / gl if gl > 0 else float("inf")
    pnl = sum(t["pnl"] for t in trades)
    eq = [0]
    for t in trades: eq.append(eq[-1] + t["pnl"])
    eq = np.array(eq)
    pk = np.maximum.accumulate(eq)
    dd = float(np.max(pk - eq))
    aw = float(np.mean([t["pnl"] for t in wins])) if wins else 0
    al = float(np.mean([t["pnl"] for t in losses])) if losses else 0
    
    return {"trades": total, "wins": len(wins), "losses": len(losses),
            "wr": round(wr, 1), "pf": round(pf, 2), "pnl": round(pnl, 2),
            "avg_win": round(aw, 4), "avg_loss": round(al, 4), "max_dd": round(dd, 2)}


def main():
    print("=" * 70)
    print("SQUEEZE V6 OPTIMIZER")
    print("=" * 70)
    
    closes, highs, lows, volumes = load_data()
    n = len(closes)
    print(f"Candles: {n}")
    
    ind = precompute(closes, highs, lows, volumes)
    signals = generate_signals(closes, highs, lows, volumes, ind, n)
    print(f"Signals: {len(signals)}")
    
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
                                params = {"tp_mult": tp, "sl_mult": sl, "hold": hold,
                                          "trailing": trailing, "time_exit": te,
                                          "cooldown": 2, "trend_filter": trend, "vol_filter": vol}
                                r = backtest(signals, closes, highs, lows, params)
                                tested += 1
                                if r["trades"] >= 50:
                                    results.append({"tp": tp, "sl": sl, "hold": hold,
                                                    "trail": trailing, "texit": te,
                                                    "trend": trend, "vol": vol, **r})
    
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
    
    # Save
    out = {"candles": n, "signals": len(signals), "tested": tested,
           "results_50plus": len(results), "top_30": results[:30],
           "best": results[0] if results else None}
    out_path = os.path.join(BASE, "data", "5agent_backtest", "squeeze_v6_optimizer.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
