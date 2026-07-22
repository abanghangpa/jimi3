#!/usr/bin/env python3
"""Squeeze V6: Invert V3 + quality filters. Trade the 71% failure rate."""
import csv, json, os, numpy as np
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE, "data", "history", "ETHUSDT_15m.csv")

closes, highs, lows, volumes = [], [], [], []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        closes.append(float(row["close"]))
        highs.append(float(row["high"]))
        lows.append(float(row["low"]))
        volumes.append(float(row["volume"]))

closes = np.array(closes); highs = np.array(highs)
lows = np.array(lows); volumes = np.array(volumes)
n = len(closes)
print(f"Loaded {n} candles")

# Pre-compute
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

def calc_atr(h, l, c, p=14):
    trs = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    r = np.full(len(h), np.nan)
    for i in range(p, len(h)):
        r[i] = np.mean(trs[i-p:i])
    return r

print("Computing indicators...")
bb_mid = rolling_mean(closes, 20)
bb_std = rolling_std(closes, 20)
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
bb_width = np.where(bb_mid > 0, (bb_upper - bb_lower) / bb_mid, 0)
atr14 = calc_atr(highs, lows, closes, 14)
vol_ma = rolling_mean(volumes, 20)
ema_20 = ema(closes, 20)
ema_50 = ema(closes, 50)


def run_v6(filters, label):
    """Run inverted squeeze with specified filters."""
    trades = []
    positions = []
    last_sig = -999
    
    for i in range(60, n):
        # Exits
        new = []
        for p in positions:
            p["b"] += 1
            ex = False
            if p["d"] == "L":
                if lows[i] <= p["sl"]: ex = True; ep = p["sl"]; o = "L"
                elif highs[i] >= p["tp"]: ex = True; ep = p["tp"]; o = "W"
            else:
                if highs[i] >= p["sl"]: ex = True; ep = p["sl"]; o = "L"
                elif lows[i] <= p["tp"]: ex = True; ep = p["tp"]; o = "W"
            if not ex and p["b"] >= p["hb"]: ex = True; ep = closes[i]; o = "T"
            if ex:
                pnl = ((ep - p["e"]) / p["e"] * 100) if p["d"] == "L" else ((p["e"] - ep) / p["e"] * 100)
                pnl -= 0.04
                trades.append({"outcome": o, "pnl": pnl})
            else:
                new.append(p)
        positions = new
        if positions: continue
        if i - last_sig < filters.get("cooldown", 2): continue
        
        if np.isnan(bb_width[i]) or bb_width[i] > 0.02: continue
        a = atr14[i]
        if np.isnan(a) or a == 0: continue
        
        price = closes[i]
        
        # V3 signal (breakout direction)
        v3_dir = None
        if price > bb_upper[i]: v3_dir = "LONG"
        elif price < bb_lower[i]: v3_dir = "SHORT"
        if not v3_dir: continue
        
        # INVERT: fade the breakout
        if v3_dir == "LONG":
            direction = "SHORT"
            entry = price
            sl = price + a * filters.get("sl_mult", 1.0)
            tp = price - a * filters.get("tp_mult", 1.5)
        else:
            direction = "LONG"
            entry = price
            sl = price - a * filters.get("sl_mult", 1.0)
            tp = price + a * filters.get("tp_mult", 1.5)
        
        # FILTERS
        passed = True
        
        # Trend filter: only fade counter-trend breakouts
        if filters.get("trend_filter"):
            if not np.isnan(ema_20[i]) and not np.isnan(ema_50[i]):
                if v3_dir == "LONG" and closes[i] > ema_50[i]:
                    passed = False  # LONG breakout in uptrend = don't fade
                elif v3_dir == "SHORT" and closes[i] < ema_50[i]:
                    passed = False  # SHORT breakout in downtrend = don't fade
        
        # Volume filter: fade low-volume breakouts (likely false)
        if filters.get("vol_filter"):
            if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
                vol_ratio = volumes[i] / vol_ma[i]
                if vol_ratio > 1.5:
                    passed = False  # High volume = real breakout, don't fade
        
        # Momentum filter: fade when momentum is weak
        if filters.get("mom_filter"):
            if i >= 4:
                mom = (closes[i] - closes[i-4]) / closes[i-4]
                if abs(mom) > 0.02:
                    passed = False  # Strong momentum = real breakout
        
        if not passed: continue
        
        # RR check
        risk = abs(entry - sl)
        reward = abs(entry - tp)
        if risk == 0 or reward / risk < filters.get("min_rr", 1.0):
            continue
        
        positions.append({"d": direction, "e": entry, "sl": sl, "tp": tp,
                          "b": 0, "hb": filters.get("hold", 8)})
        last_sig = i
    
    # Stats
    if not trades:
        return {"label": label, "trades": 0, "wins": 0, "losses": 0, "wr": 0, "pf": 0, "pnl": 0}
    
    wins = sum(1 for t in trades if t["outcome"] == "W")
    losses = sum(1 for t in trades if t["outcome"] == "L")
    total = len(trades)
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gp/gl if gl > 0 else float("inf")
    pnl = sum(t["pnl"] for t in trades)
    
    return {"label": label, "trades": total, "wins": wins, "losses": losses,
            "wr": round(wins/total*100, 1), "pf": round(pf, 2), "pnl": round(pnl, 2)}


# Run different filter combinations
configs = [
    ({"cooldown": 2, "sl_mult": 1.0, "tp_mult": 1.5, "hold": 8},
     "V6-A: Raw invert"),
    
    ({"cooldown": 2, "sl_mult": 1.0, "tp_mult": 1.5, "hold": 8, "trend_filter": True},
     "V6-B: +Trend filter"),
    
    ({"cooldown": 2, "sl_mult": 1.0, "tp_mult": 1.5, "hold": 8, "trend_filter": True, "vol_filter": True},
     "V6-C: +Trend+Vol"),
    
    ({"cooldown": 2, "sl_mult": 1.0, "tp_mult": 1.5, "hold": 8, "trend_filter": True, "vol_filter": True, "mom_filter": True},
     "V6-D: +Trend+Vol+Mom"),
    
    ({"cooldown": 4, "sl_mult": 0.8, "tp_mult": 2.0, "hold": 6, "trend_filter": True, "vol_filter": True, "min_rr": 1.5},
     "V6-E: Tight SL, wide TP, trend+vol"),
    
    ({"cooldown": 4, "sl_mult": 1.5, "tp_mult": 2.5, "hold": 12, "trend_filter": True, "vol_filter": True, "min_rr": 1.5},
     "V6-F: Wide SL, wide TP, trend+vol"),
]

print()
print("=" * 70)
print("SQUEEZE V6: INVERT V3 + QUALITY FILTERS (89,000 CANDLES)")
print("=" * 70)
print(f"\n{'Config':<45} {'Trades':<8} {'W':<5} {'L':<5} {'WR':<8} {'PF':<6} {'PnL%':<8}")
print("-" * 70)

for cfg, label in configs:
    r = run_v6(cfg, label)
    print(f"  {r['label']:<43} {r['trades']:<8} {r['wins']:<5} {r['losses']:<5} {r['wr']:<8} {r['pf']:<6} {r['pnl']:<8}")
