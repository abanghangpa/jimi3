
import csv, json, os, sys, math
from datetime import datetime, timezone
import numpy as np

CSV_PATH = "/root/.openclaw/workspace/jimi_audit/data/history/ETHUSDT_15m.csv"

def load_data():
    closes, highs, lows, volumes, taker_vols, timestamps = [], [], [], [], [], []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(int(row["ts"]))
            closes.append(float(row["close"]))
            highs.append(float(row["high"]))
            lows.append(float(row["low"]))
            volumes.append(float(row["volume"]))
            taker_vols.append(float(row.get("taker_buy_vol", 0)))
    return np.array(timestamps), np.array(closes), np.array(highs), np.array(lows), np.array(volumes), np.array(taker_vols)

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

def get_session(ts_ms):
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    hour = dt.hour
    if 0 <= hour < 8: return "ASIA"
    elif 8 <= hour < 14: return "EU"
    elif 14 <= hour < 21: return "US"
    else: return "LATE_US"

ts, closes, highs, lows, volumes, taker_vols = load_data()
n = len(closes)

bb_mid = rolling_mean(closes, 20)
bb_std = rolling_std(closes, 20)
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
bb_width = np.where(bb_mid > 0, (bb_upper - bb_lower) / bb_mid, 0)
atr14 = calc_atr(highs, lows, closes, 14)
vol_ma20 = rolling_mean(volumes, 20)
vol_ratio = np.where(vol_ma20 > 0, volumes / vol_ma20, 1.0)
ema_20 = ema(closes, 20)
ema_50 = ema(closes, 50)

# KC for squeeze detection
kc_mid = ema(closes, 20)
kc_upper = kc_mid + 1.5 * atr14
kc_lower = kc_mid - 1.5 * atr14
in_squeeze = (kc_upper < bb_upper) & (kc_lower > bb_lower)

# ── TEST 1: MOMENTUM (follow breakout) ──
# Test 2: Also try "fade with trend" (fade breakout that goes AGAINST trend)
# Test 3: "Fade counter-trend only" (fade breakout that goes WITH the trend = fade back)

print("=" * 80)
print("  COMPREHENSIVE SQUEEZE BREAKOUT ANALYSIS")
print("  Testing: Momentum vs Fade vs Various Filters")
print("=" * 80)

# Generate all squeeze breakout events (regardless of direction)
all_events = []
in_sqz_prev = False
breakout_bar = None
breakout_dir = None

for i in range(60, n):
    price = closes[i]
    if np.isnan(bb_width[i]): continue
    in_sqz = bool(in_squeeze[i]) if not np.isnan(in_squeeze[i]) else False

    if in_sqz_prev and not in_sqz:
        if price > bb_upper[i]:
            breakout_bar = i
            breakout_dir = "LONG"
        elif price < bb_lower[i]:
            breakout_bar = i
            breakout_dir = "SHORT"
        else:
            breakout_bar = None
            breakout_dir = None
    in_sqz_prev = in_sqz

    if breakout_bar is None or in_sqz: continue
    if i - breakout_bar > 12:
        breakout_bar = None
        continue

    a = atr14[i]
    if np.isnan(a) or a == 0: continue

    vr = vol_ratio[i]
    if np.isnan(vr): vr = 1.0

    ema50_val = ema_50[i]
    ema20_val = ema_20[i]

    # Classify breakout
    trend_aligned = False
    if not np.isnan(ema50_val):
        if breakout_dir == "LONG" and price > ema50_val:
            trend_aligned = True
        elif breakout_dir == "SHORT" and price < ema50_val:
            trend_aligned = True

    all_events.append({
        "bar": i,
        "breakout_dir": breakout_dir,
        "price": price,
        "atr": a,
        "bb_width": bb_width[i],
        "vol_ratio": vr,
        "trend_aligned": trend_aligned,
        "session": get_session(ts[i]),
    })
    breakout_bar = None

print(f"\nTotal squeeze breakout events: {len(all_events)}")
print(f"  Trend-aligned: {len([e for e in all_events if e['trend_aligned']])}")
print(f"  Counter-trend: {len([e for e in all_events if not e['trend_aligned']])}")

# ── Run backtest for each strategy variant ──
def run_variant(events, closes, highs, lows, variant_name, direction_mode, filters=None):
    """
    direction_mode: 'momentum' = follow breakout, 'fade' = opposite
    filters: dict of additional filters
    """
    filters = filters or {}
    trades = []
    last_bar = -999
    fee = 0.0004

    tp_mult = filters.get("tp_mult", 1.5)
    sl_mult = filters.get("sl_mult", 1.0)
    hold = filters.get("hold", 16)
    min_vol = filters.get("min_vol", 0)
    max_vol = filters.get("max_vol", 999)
    require_trend = filters.get("require_trend", None)  # True = with trend, False = against, None = any
    cooldown = filters.get("cooldown", 2)
    max_bb_width = filters.get("max_bb_width", 999)
    min_bb_width = filters.get("min_bb_width", 0)

    for ev in events:
        if ev["bar"] - last_bar < cooldown:
            continue
        if ev["vol_ratio"] < min_vol or ev["vol_ratio"] > max_vol:
            continue
        if ev["bb_width"] > max_bb_width or ev["bb_width"] < min_bb_width:
            continue
        if require_trend is not None:
            if require_trend and not ev["trend_aligned"]:
                continue
            if not require_trend and ev["trend_aligned"]:
                continue

        entry = ev["price"]
        a = ev["atr"]
        breakout_dir = ev["breakout_dir"]

        if direction_mode == "momentum":
            direction = breakout_dir
        else:
            direction = "SHORT" if breakout_dir == "LONG" else "LONG"

        if direction == "LONG":
            sl = entry - a * sl_mult
            tp = entry + a * tp_mult
        else:
            sl = entry + a * sl_mult
            tp = entry - a * tp_mult

        # Simulate
        exited = False
        exit_price = entry
        outcome = "TIMEOUT"
        for j in range(ev["bar"] + 1, min(ev["bar"] + hold + 1, n)):
            if direction == "LONG":
                if lows[j] <= sl:
                    exited = True; exit_price = sl; outcome = "LOSS"; break
                if highs[j] >= tp:
                    exited = True; exit_price = tp; outcome = "WIN"; break
            else:
                if highs[j] >= sl:
                    exited = True; exit_price = sl; outcome = "LOSS"; break
                if lows[j] <= tp:
                    exited = True; exit_price = tp; outcome = "WIN"; break
        if not exited:
            exit_price = closes[min(ev["bar"] + hold, n - 1)]

        if direction == "LONG":
            pnl = (exit_price - entry) / entry * 100
        else:
            pnl = (entry - exit_price) / entry * 100
        pnl -= fee * 2 * 100

        trades.append({"pnl": pnl, "outcome": outcome, "session": ev["session"], "direction": direction})
        last_bar = ev["bar"]

    if not trades:
        return {"variant": variant_name, "trades": 0}

    wins = len([t for t in trades if t["outcome"] == "WIN"])
    losses = len([t for t in trades if t["outcome"] == "LOSS"])
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gp / gl if gl > 0 else float("inf")

    return {
        "variant": variant_name,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "wr": round(wins / len(trades) * 100, 1),
        "pf": round(pf, 3),
        "pnl": round(sum(t["pnl"] for t in trades), 2),
        "avg_pnl": round(np.mean([t["pnl"] for t in trades]), 4),
    }

# ── Test Matrix ──
variants = []

# MOMENTUM variants
variants.append(run_variant(all_events, closes, highs, lows, "MOM_default", "momentum", {"tp_mult": 1.5, "sl_mult": 1.0, "hold": 16}))
variants.append(run_variant(all_events, closes, highs, lows, "MOM_tp2_sl1", "momentum", {"tp_mult": 2.0, "sl_mult": 1.0, "hold": 16}))
variants.append(run_variant(all_events, closes, highs, lows, "MOM_tp3_sl1", "momentum", {"tp_mult": 3.0, "sl_mult": 1.0, "hold": 16}))
variants.append(run_variant(all_events, closes, highs, lows, "MOM_with_trend", "momentum", {"tp_mult": 2.0, "sl_mult": 1.0, "hold": 16, "require_trend": True}))
variants.append(run_variant(all_events, closes, highs, lows, "MOM_counter_trend", "momentum", {"tp_mult": 2.0, "sl_mult": 1.0, "hold": 16, "require_trend": False}))
variants.append(run_variant(all_events, closes, highs, lows, "MOM_low_vol", "momentum", {"tp_mult": 2.0, "sl_mult": 1.0, "hold": 16, "max_vol": 1.2}))
variants.append(run_variant(all_events, closes, highs, lows, "MOM_tight_sqz", "momentum", {"tp_mult": 2.5, "sl_mult": 0.75, "hold": 16, "max_bb_width": 0.015}))

# FADE variants (original approach)
variants.append(run_variant(all_events, closes, highs, lows, "FADE_default", "fade", {"tp_mult": 1.5, "sl_mult": 1.0, "hold": 16}))
variants.append(run_variant(all_events, closes, highs, lows, "FADE_tp2_sl1", "fade", {"tp_mult": 2.0, "sl_mult": 1.0, "hold": 16}))
variants.append(run_variant(all_events, closes, highs, lows, "FADE_with_trend", "fade", {"tp_mult": 2.0, "sl_mult": 1.0, "hold": 16, "require_trend": True}))
variants.append(run_variant(all_events, closes, highs, lows, "FADE_counter_trend", "fade", {"tp_mult": 2.0, "sl_mult": 1.0, "hold": 16, "require_trend": False}))
variants.append(run_variant(all_events, closes, highs, lows, "FADE_tight_sqz", "fade", {"tp_mult": 2.5, "sl_mult": 0.75, "hold": 16, "max_bb_width": 0.015}))
variants.append(run_variant(all_events, closes, highs, lows, "FADE_low_vol", "fade", {"tp_mult": 2.0, "sl_mult": 1.0, "hold": 16, "max_vol": 1.2}))

# Best hold period tests
for hold in [4, 8, 16, 32]:
    variants.append(run_variant(all_events, closes, highs, lows, f"MOM_h{hold}", "momentum", {"tp_mult": 2.0, "sl_mult": 1.0, "hold": hold}))
    variants.append(run_variant(all_events, closes, highs, lows, f"FADE_h{hold}", "fade", {"tp_mult": 2.0, "sl_mult": 1.0, "hold": hold}))

# Sort by PF
variants.sort(key=lambda x: x.get("pf", 0), reverse=True)

print(f"\n{'='*80}")
print(f"  ALL VARIANTS (sorted by PF)")
print(f"{'='*80}")
print(f"  {'Variant':<25} {'Trades':>6} {'WR%':>6} {'PF':>6} {'PnL%':>8}")
print(f"  {'─'*25} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
for v in variants:
    if v.get("trades", 0) > 0:
        print(f"  {v['variant']:<25} {v['trades']:>6} {v['wr']:>6.1f} {v['pf']:>6.3f} {v['pnl']:>8.2f}")

# ── Honest summary ──
print(f"\n{'='*80}")
print("  SUMMARY")
print(f"{'='*80}")
best = variants[0] if variants else None
if best and best.get("pf", 0) > 1.0:
    print(f"  Best variant: {best['variant']}")
    print(f"    Trades={best['trades']} WR={best['wr']}% PF={best['pf']} PnL={best['pnl']}%")
    if best["wr"] >= 75 and best["pf"] >= 2.0:
        print("  ✅ MEETS TARGET")
    else:
        print(f"  ⚠️ Does not meet 75% WR / 2.0 PF target")
else:
    print("  ❌ NO PROFITABLE VARIANT FOUND")
    print("  The squeeze breakout pattern has NO statistical edge on ETH 15m data.")
    print("  Both momentum and fade approaches produce PF < 1.0 across all tested configs.")
    print("  RECOMMENDATION: Remove squeeze_breakout from active strategy roster.")
    print("  The m18 squeeze module can remain as a detection tool for other strategies.")
