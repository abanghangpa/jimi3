import json, os, sys, csv
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

sys.path.insert(0, "/root/.openclaw/workspace/jimi_audit/src")
sys.path.insert(0, "/root/.openclaw/workspace/jimi_audit/scripts")

csv_path = "/root/.openclaw/workspace/jimi_audit/data/eth_15m_extended.csv"
closes = []
volumes = []
timestamps = []

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            closes.append(float(row["Close"]))
            volumes.append(float(row["Volume"]))
            timestamps.append(row.get("Open time", row.get("timestamp", "")))
        except:
            continue

closes = np.array(closes, dtype=float)
volumes = np.array(volumes, dtype=float)

print(f"Loaded {len(closes)} bars")
print(f"Range: {timestamps[0]} -> {timestamps[-1]}")

# Load derivatives
deriv_path = "/root/.openclaw/workspace/jimi_audit/data/derivatives_history/derivatives_collected.csv"
deriv_by_ts = {}
if os.path.exists(deriv_path):
    with open(deriv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp", "")
            if ts:
                try:
                    deriv_by_ts[ts] = {
                        "oi": float(row.get("oi", 0) or 0),
                        "fr": float(row.get("funding_rate", 0) or 0),
                    }
                except:
                    pass

print(f"Derivatives: {len(deriv_by_ts)} rows")

# Find July 2026
july_start = None
july_end = None
for i, ts in enumerate(timestamps):
    if "2026-07" in ts:
        if july_start is None:
            july_start = i
        july_end = i

print(f"\nJuly 2026: bars {july_start}-{july_end}")
print(f"Range: {timestamps[july_start]} -> {timestamps[july_end]}")

# Build OI lookup by hour
oi_by_hour = {}
for dts, data in sorted(deriv_by_ts.items()):
    hour_key = dts[:13]
    oi_by_hour[hour_key] = data["oi"]

# Run momentum_v3
triggers = []
for idx in range(max(july_start, 80), july_end + 1):
    price = closes[idx]

    mom_5 = (closes[idx] - closes[idx-5]) / closes[idx-5]
    mom_10 = (closes[idx] - closes[idx-10]) / closes[idx-10]
    accel = mom_5 - mom_10 / 2

    decel_signal = (mom_5 > 0 and accel < 0) or (mom_5 < 0 and accel > 0)

    vol_recent = np.mean(volumes[idx-5:idx])
    vol_prior = np.mean(volumes[idx-15:idx-5])
    vol_change = (vol_recent - vol_prior) / vol_prior if vol_prior > 0 else 0
    vol_divergence = (mom_5 > 0.005 and vol_change < -0.1) or (mom_5 < -0.005 and vol_change < -0.1)

    moves = []
    for j in range(max(0, idx-80), idx-5):
        if j+5 < len(closes):
            m = abs(closes[j+5] - closes[j]) / closes[j]
            moves.append(m)
    current_move = abs(closes[idx] - closes[idx-5]) / closes[idx-5]
    percentile = sum(1 for m in moves if m < current_move) / len(moves) * 100 if moves else 0
    extreme_move = percentile > 85

    ts_str = timestamps[idx]
    hour_key = ts_str[:13]
    oi_roc = 0
    if hour_key in oi_by_hour:
        try:
            h = int(hour_key[11:13])
            prev_h = f"{h-1:02d}"
            prev_key = hour_key[:11] + prev_h
            if prev_key in oi_by_hour and oi_by_hour[prev_key] > 0:
                oi_roc = (oi_by_hour[hour_key] - oi_by_hour[prev_key]) / oi_by_hour[prev_key]
        except:
            pass

    oi_divergence = (mom_5 > 0.005 and oi_roc < -0.02) or (mom_5 < -0.005 and oi_roc < -0.02)

    signals_count = sum([decel_signal, vol_divergence, extreme_move, oi_divergence])
    if signals_count >= 2:
        direction = "SHORT" if mom_5 > 0 else "LONG"
        base = 0.40
        if decel_signal: base += 0.15
        if vol_divergence: base += 0.15
        if extreme_move: base += 0.10
        if oi_divergence: base += 0.10
        conviction = min(base, 0.85)

        dir_mult = 1 if direction == "LONG" else -1
        entry = price
        p4h = closes[min(idx+16, len(closes)-1)]
        p8h = closes[min(idx+32, len(closes)-1)]
        p24h = closes[min(idx+96, len(closes)-1)]
        r4h = (p4h - entry) / entry * dir_mult * 100
        r8h = (p8h - entry) / entry * dir_mult * 100
        r24h = (p24h - entry) / entry * dir_mult * 100

        triggers.append({
            "idx": idx, "timestamp": timestamps[idx], "price": price,
            "direction": direction, "conviction": conviction,
            "mom_5": mom_5, "accel": accel,
            "vol_change": vol_change, "percentile": percentile,
            "oi_roc": oi_roc,
            "decel": decel_signal, "vol_div": vol_divergence,
            "extreme": extreme_move, "oi_div": oi_divergence,
            "signals_count": signals_count,
            "r4h": r4h, "r8h": r8h, "r24h": r24h,
        })

print(f"\n=== MOMENTUM V3 TRIGGERS IN JULY 2026 ===")
print(f"Total triggers: {len(triggers)}")
print()

by_date = defaultdict(list)
for t in triggers:
    day = t["timestamp"][:10]
    by_date[day].append(t)

for day in sorted(by_date.keys()):
    ts_list = by_date[day]
    print(f"=== {day} ({len(ts_list)} triggers) ===")
    for t in ts_list:
        sigs = []
        if t["decel"]: sigs.append("DECEL")
        if t["vol_div"]: sigs.append("VOL_DIV")
        if t["extreme"]: sigs.append("EXTREME({:.0f})".format(t["percentile"]))
        if t["oi_div"]: sigs.append("OI_DIV")
        sig_str = " + ".join(sigs)
        print("  {} | ${:.2f} | {} conv={:.2f} | {} | 4h={:+.2f}% 8h={:+.2f}% 24h={:+.2f}%".format(
            t["timestamp"][11:16], t["price"], t["direction"], t["conviction"],
            sig_str, t["r4h"], t["r8h"], t["r24h"]))
    print()

r4hs = [t["r4h"] for t in triggers]
r8hs = [t["r8h"] for t in triggers]
r24hs = [t["r24h"] for t in triggers]

if r4hs:
    print("=== SUMMARY ===")
    print("4h: mean={:.2f}% WR={:.1f}%".format(np.mean(r4hs), sum(1 for r in r4hs if r > 0)/len(r4hs)*100))
    print("8h: mean={:.2f}% WR={:.1f}%".format(np.mean(r8hs), sum(1 for r in r8hs if r > 0)/len(r8hs)*100))
    print("24h: mean={:.2f}% WR={:.1f}%".format(np.mean(r24hs), sum(1 for r in r24hs if r > 0)/len(r24hs)*100))

print("\n=== WHY DIDN'T IT FIRE? ===")
print("1. Strategy is DISABLED in executor config (enabled: False)")
print("2. It's a Group B (state filter) - pairs with event triggers, not standalone")
print("3. No co-occurrence check with Group A strategies in executor")

decel_count = sum(1 for t in triggers if t["decel"])
vol_count = sum(1 for t in triggers if t["vol_div"])
ext_count = sum(1 for t in triggers if t["extreme"])
oi_count = sum(1 for t in triggers if t["oi_div"])
print(f"\n=== SIGNAL FREQUENCY ===")
print(f"Deceleration: {decel_count}/{len(triggers)} ({decel_count/len(triggers)*100:.1f}%)")
print(f"Volume div:   {vol_count}/{len(triggers)} ({vol_count/len(triggers)*100:.1f}%)")
print(f"Extreme move: {ext_count}/{len(triggers)} ({ext_count/len(triggers)*100:.1f}%)")
print(f"OI divergence: {oi_count}/{len(triggers)} ({oi_count/len(triggers)*100:.1f}%)")

high_conv = [t for t in triggers if t["conviction"] >= 0.55]
print(f"\n=== HIGH CONVICTION (>=0.55) ===")
print(f"Count: {len(high_conv)}")
if high_conv:
    r4h_hc = [t["r4h"] for t in high_conv]
    r8h_hc = [t["r8h"] for t in high_conv]
    print("4h: mean={:.2f}% WR={:.1f}%".format(np.mean(r4h_hc), sum(1 for r in r4h_hc if r > 0)/len(r4h_hc)*100))
    print("8h: mean={:.2f}% WR={:.1f}%".format(np.mean(r8h_hc), sum(1 for r in r8h_hc if r > 0)/len(r8h_hc)*100))
    for t in high_conv:
        sigs = []
        if t["decel"]: sigs.append("DECEL")
        if t["vol_div"]: sigs.append("VOL_DIV")
        if t["extreme"]: sigs.append("EXTREME")
        if t["oi_div"]: sigs.append("OI_DIV")
        print("  {} | ${:.2f} | {} conv={:.2f} | 4h={:+.2f}% | {}".format(
            t["timestamp"][:16], t["price"], t["direction"], t["conviction"], t["r4h"], " + ".join(sigs)))
