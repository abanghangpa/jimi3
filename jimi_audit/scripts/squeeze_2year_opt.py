#!/usr/bin/env python3
"""Optimized 2.5-year squeeze backtest using pre-computed indicators."""
import csv, json, os, numpy as np
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE, "data", "history", "ETHUSDT_15m.csv")

# Load data
closes, highs, lows, volumes, timestamps = [], [], [], [], []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        closes.append(float(row["close"]))
        highs.append(float(row["high"]))
        lows.append(float(row["low"]))
        volumes.append(float(row["volume"]))
        timestamps.append(int(row["ts"]))

closes = np.array(closes)
highs = np.array(highs)
lows = np.array(lows)
volumes = np.array(volumes)
n = len(closes)
print(f"Loaded {n} candles")
print(f"Period: {datetime.fromtimestamp(timestamps[0]/1000, tz=timezone.utc).strftime('%Y-%m-%d')} → {datetime.fromtimestamp(timestamps[-1]/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")

# Pre-compute indicators
def rolling_mean(arr, period):
    result = np.full(len(arr), np.nan)
    cumsum = np.cumsum(arr)
    result[period-1:] = (cumsum[period-1:] - np.concatenate([[0], cumsum[:-period]])) / period
    return result

def rolling_std(arr, period):
    result = np.full(len(arr), np.nan)
    for i in range(period-1, len(arr)):
        result[i] = np.std(arr[i-period+1:i+1])
    return result

def ema(arr, period):
    result = np.full(len(arr), np.nan)
    if len(arr) < period: return result
    result[period-1] = np.mean(arr[:period])
    mult = 2 / (period + 1)
    for i in range(period, len(arr)):
        result[i] = arr[i] * mult + result[i-1] * (1 - mult)
    return result

def calc_atr(highs, lows, closes, period=14):
    trs = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    result = np.full(len(highs), np.nan)
    for i in range(period, len(highs)):
        result[i] = np.mean(trs[i-period:i])
    return result

print("Computing indicators...")
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
in_squeeze = (kc_upper < bb_upper) & (kc_lower > bb_lower)

print(f"Squeeze bars: {np.sum(in_squeeze & ~np.isnan(in_squeeze))} / {n}")

# V3: Current breakout
v3_trades = []
v3_pos = []
for i in range(60, n):
    new = []
    for p in v3_pos:
        p["b"] += 1
        ex = False
        if p["d"] == "L":
            if lows[i] <= p["sl"]: ex = True; ep = p["sl"]; o = "L"
            elif highs[i] >= p["tp"]: ex = True; ep = p["tp"]; o = "W"
        else:
            if highs[i] >= p["sl"]: ex = True; ep = p["sl"]; o = "L"
            elif lows[i] <= p["tp"]: ex = True; ep = p["tp"]; o = "W"
        if not ex and p["b"] >= 8: ex = True; ep = closes[i]; o = "T"
        if ex:
            pnl = ((ep - p["e"]) / p["e"] * 100) if p["d"] == "L" else ((p["e"] - ep) / p["e"] * 100)
            pnl -= 0.04
            v3_trades.append({"outcome": o, "pnl": pnl})
        else:
            new.append(p)
    v3_pos = new
    if v3_pos: continue
    if np.isnan(bb_width[i]) or bb_width[i] > 0.02: continue
    a = atr14[i]
    if np.isnan(a) or a == 0: continue
    p = closes[i]
    if p > bb_upper[i]: v3_pos.append({"d": "L", "e": p, "sl": p-a, "tp": p+a*2, "b": 0})
    elif p < bb_lower[i]: v3_pos.append({"d": "S", "e": p, "sl": p+a, "tp": p-a*2, "b": 0})

# V5B: Fade the failure
v5_trades = []
v5_pos = []
breakout_bar = None; breakout_dir = None; breakout_extreme = None
in_sqz_prev = False; last_sig = -999

for i in range(60, n):
    new = []
    for p in v5_pos:
        p["b"] += 1
        ex = False
        if p["d"] == "L":
            if lows[i] <= p["sl"]: ex = True; ep = p["sl"]; o = "L"
            elif highs[i] >= p["tp"]: ex = True; ep = p["tp"]; o = "W"
        else:
            if highs[i] >= p["sl"]: ex = True; ep = p["sl"]; o = "L"
            elif lows[i] <= p["tp"]: ex = True; ep = p["tp"]; o = "W"
        if not ex and p["b"] >= 8: ex = True; ep = closes[i]; o = "T"
        if ex:
            pnl = ((ep - p["e"]) / p["e"] * 100) if p["d"] == "L" else ((p["e"] - ep) / p["e"] * 100)
            pnl -= 0.04
            v5_trades.append({"outcome": o, "pnl": pnl, "mode": p.get("m", "?")})
        else:
            new.append(p)
    v5_pos = new
    if v5_pos: continue
    if np.isnan(bb_width[i]): continue
    in_sqz = bool(in_squeeze[i]) if not np.isnan(in_squeeze[i]) else False
    if in_sqz_prev and not in_sqz:
        price = closes[i]
        if price > bb_upper[i]: breakout_bar = i; breakout_dir = "L"; breakout_extreme = highs[i]
        elif price < bb_lower[i]: breakout_bar = i; breakout_dir = "S"; breakout_extreme = lows[i]
        else: breakout_bar = None; breakout_dir = None
    in_sqz_prev = in_sqz
    if breakout_bar is None or in_sqz: continue
    if i - breakout_bar > 12: breakout_bar = None; continue
    if i - last_sig < 4: continue
    a = atr14[i]
    if np.isnan(a) or a == 0: continue
    price = closes[i]
    if breakout_dir == "L":
        if price < bb_upper[i] and closes[i-1] >= bb_upper[i]:
            entry = price; sl = breakout_extreme + a*0.3; tp = bb_mid[i]
            risk = abs(entry - sl); reward = abs(entry - tp)
            if risk > 0 and reward/risk >= 1.0:
                v5_pos.append({"d": "S", "e": entry, "sl": sl, "tp": tp, "b": 0, "m": "FADE"})
                last_sig = i; breakout_bar = None
    elif breakout_dir == "S":
        if price > bb_lower[i] and closes[i-1] <= bb_lower[i]:
            entry = price; sl = breakout_extreme - a*0.3; tp = bb_mid[i]
            risk = abs(entry - sl); reward = abs(entry - tp)
            if risk > 0 and reward/risk >= 1.0:
                v5_pos.append({"d": "L", "e": entry, "sl": sl, "tp": tp, "b": 0, "m": "FADE"})
                last_sig = i; breakout_bar = None

# Results
def stats(trades, label):
    if not trades: return f"{label}: 0 trades"
    wins = sum(1 for t in trades if t["outcome"] == "W")
    losses = sum(1 for t in trades if t["outcome"] == "L")
    total = len(trades)
    wr = wins/total*100 if total else 0
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gp/gl if gl > 0 else float("inf")
    pnl = sum(t["pnl"] for t in trades)
    return f"{label}: {total}T {wins}W/{losses}L WR={wr:.1f}% PF={pf:.2f} PnL={pnl:+.2f}%"

print()
print("=" * 60)
print("RESULTS: 2.5-YEAR BACKTEST (89,000 CANDLES)")
print("=" * 60)
print(stats(v3_trades, "V3 (breakout)"))
print(stats(v5_trades, "V5B (fade)"))

if v5_trades:
    modes = {}
    for t in v5_trades:
        m = t.get("mode", "?")
        if m not in modes: modes[m] = []
        modes[m].append(t)
    for m, ts in modes.items():
        print(stats(ts, f"  {m}"))

# Save
output = {
    "candles": n,
    "v3": {"trades": len(v3_trades), "wins": sum(1 for t in v3_trades if t["outcome"]=="W"), "losses": sum(1 for t in v3_trades if t["outcome"]=="L"), "pnl_pct": round(sum(t["pnl"] for t in v3_trades), 2)},
    "v5b": {"trades": len(v5_trades), "wins": sum(1 for t in v5_trades if t["outcome"]=="W"), "losses": sum(1 for t in v5_trades if t["outcome"]=="L"), "pnl_pct": round(sum(t["pnl"] for t in v5_trades), 2)},
}
out_path = os.path.join(BASE, "data", "5agent_backtest", "squeeze_2year_backtest.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
