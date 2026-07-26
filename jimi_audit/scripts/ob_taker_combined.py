"""
OB State + Taker Flow + S/R — Combined Signal

Jeon (2026) hierarchy:
1. L2 state (ob_ratio, top5_ratio, persistence) — PRIMARY
2. Order flow (taker z-score) — OVERLAY
3. S/R proximity — STRUCTURAL

Test: Does OB state + taker confirmation beat either alone?
"""
import pandas as pd
import numpy as np
from scipy import stats
import json, os

DATA_DIR = "/root/.openclaw/workspace/jimi_audit/data"
DERIV_DIR = f"{DATA_DIR}/derivatives_history"

# Load OHLCV
ohlcv = pd.read_csv(f"{DATA_DIR}/eth_15m_extended.csv")
ohlcv["timestamp"] = pd.to_datetime(ohlcv["Open time"])
ohlcv = ohlcv.sort_values("timestamp").reset_index(drop=True)

# Load derivatives
deriv = pd.read_csv(f"{DERIV_DIR}/derivatives_collected.csv")
deriv["timestamp"] = pd.to_datetime(deriv["timestamp"], format="mixed", utc=True).dt.tz_localize(None)
deriv = deriv.sort_values("timestamp").reset_index(drop=True)

merged = pd.merge_asof(ohlcv, deriv[["timestamp","oi","ls_ratio","funding_rate"]],
                       on="timestamp", direction="backward", tolerance=pd.Timedelta("30min"))
merged["vol_ratio"] = merged["Volume"] / merged["Volume"].rolling(20).mean()
merged["ema200"] = merged["Close"].ewm(span=200).mean()
merged["trend"] = np.where(merged["Close"] > merged["ema200"], "BULL", "BEAR")
for h in [4, 8, 16, 24]:
    merged["fwd_ret_" + str(h)] = merged["Close"].shift(-h) / merged["Close"] - 1

closes = merged["Close"].values
volumes = merged["Volume"].values
taker_base = merged["Taker buy base asset volume"].values
n = len(merged)

# Load OB historical data
print("Loading OB historical data...")
ob_path = f"{DATA_DIR}/ob_history/ob_historical.csv"

# Read OB data from start (Jul 1-13 overlaps with OHLCV) - use head to get early data
import subprocess
temp = "/tmp/ob_recent.csv"
subprocess.run('head -1 ' + ob_path + ' > ' + temp + ' && head -n 2000000 ' + ob_path + ' | tail -n +2 >> ' + temp, shell=True, check=True)
ob = pd.read_csv(temp, usecols=["timestamp","ob_ratio","top5_ratio","bid_total","ask_total"], on_bad_lines="skip", low_memory=False)
ob["timestamp"] = pd.to_datetime(ob["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
ob = ob.dropna(subset=["timestamp"])
ob = ob.sort_values("timestamp").reset_index(drop=True)
for col in ["ob_ratio","top5_ratio","bid_total","ask_total"]:
    ob[col] = pd.to_numeric(ob[col], errors="coerce")
print(f"OB data: {len(ob)} rows, {ob['timestamp'].min()} -> {ob['timestamp'].max()}")

# Merge OB with OHLCV (nearest match within 15min)
merged_ob = pd.merge_asof(merged, ob[["timestamp","ob_ratio","top5_ratio","bid_total","ask_total"]],
                          on="timestamp", direction="nearest", tolerance=pd.Timedelta("15min"))
print(f"Merged: {len(merged_ob)} rows, OB coverage: {merged_ob['ob_ratio'].notna().sum()}")

# Precompute taker z-scores
print("Precomputing taker z-scores...")
taker_zscores = np.full(n, np.nan)
for idx in range(60, n):
    recent_buy = np.sum(taker_base[idx-4:idx])
    recent_total = np.sum(volumes[idx-4:idx])
    if recent_total == 0:
        continue
    taker_ratio = recent_buy / recent_total
    window_buy = taker_base[max(0, idx-60):idx]
    window_total = volumes[max(0, idx-60):idx]
    window_ratios = []
    for j in range(0, len(window_buy)-4, 4):
        wb = np.sum(window_buy[j:j+4])
        wt = np.sum(window_total[j:j+4])
        if wt > 0:
            window_ratios.append(wb / wt)
    if len(window_ratios) >= 5:
        mean_r = np.mean(window_ratios)
        std_r = np.std(window_ratios)
        if std_r > 0:
            taker_zscores[idx] = (taker_ratio - mean_r) / std_r
print("Done")

# ═══════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("SIGNAL GENERATION: OB + Taker + S/R")
print("="*70)

# Test multiple OB thresholds
ob_thresholds = [0.3, 0.5, 0.7, 1.0]
taker_thresholds = [1.5, 2.0, 2.5]

results = []

for ob_thresh in ob_thresholds:
    for taker_thresh in taker_thresholds:
        signals = []
        last_idx = -999

        for idx in range(200, n):
            # OB state: extreme imbalance
            ob_r = merged_ob.iloc[idx].get("ob_ratio", np.nan)
            if pd.isna(ob_r) or abs(ob_r) < ob_thresh:
                continue

            # OB direction: positive = bid heavy (LONG), negative = ask heavy (SHORT)
            ob_direction = "LONG" if ob_r > 0 else "SHORT"

            # Taker confirmation
            tz = taker_zscores[idx]
            if np.isnan(tz):
                continue
            taker_confirms = (ob_direction == "LONG" and tz > taker_thresh) or \
                             (ob_direction == "SHORT" and tz < -taker_thresh)
            if not taker_confirms:
                continue

            # Dedup (8 bars)
            if idx - last_idx < 8:
                continue
            last_idx = idx

            ret = merged_ob.iloc[idx]["fwd_ret_16"]
            if pd.isna(ret):
                continue

            dir_mult = 1 if ob_direction == "LONG" else -1
            adj_ret = ret * dir_mult
            signals.append({"idx": idx, "direction": ob_direction, "adj_ret": adj_ret, "ob_ratio": ob_r, "taker_z": tz})

        if len(signals) < 5:
            continue

        rets = np.array([s["adj_ret"] for s in signals])
        wr = (rets > 0).mean()
        mean_r = rets.mean()
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
        t, p = stats.ttest_1samp(rets, 0)
        p1 = p/2 if t > 0 else 1-p/2
        boots = [np.random.choice(rets, size=len(rets), replace=True).mean() for _ in range(1000)]
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

        results.append({
            "ob_thresh": ob_thresh, "taker_thresh": taker_thresh,
            "n": len(signals), "wr": wr, "mean": mean_r, "pf": pf,
            "p": p1, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "signals": signals,
        })

print(f"\nTested {len(results)} OB+taker combinations")
print(f"\n{'OB':>5} {'Taker':>6} {'n':>6} {'WR':>8} {'Mean':>10} {'PF':>8} {'p':>8} {'CI':>20}")
print("-"*75)
for r in sorted(results, key=lambda x: x["mean"], reverse=True):
    tag = "PASS" if r["ci_lo"] > 0 else ("PROV" if r["mean"] > 0 else "FAIL")
    ci_str = "[" + ("+" if r["ci_lo"] >= 0 else "") + str(round(r["ci_lo"]*100,3)) + "%, +" + str(round(r["ci_hi"]*100,3)) + "%]"
    print(f"[{tag}] {r['ob_thresh']:>5.1f} {r['taker_thresh']:>6.1f} {r['n']:>6} {r['wr']*100:>7.1f}% {r['mean']*100:>+9.3f}% {r['pf']:>7.2f} {r['p']:>7.4f} {ci_str:>20}")

# ═══════════════════════════════════════════════════════
# OB STATE ALONE (no taker)
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("OB STATE ALONE (no taker confirmation)")
print("="*70)

for ob_thresh in ob_thresholds:
    signals = []
    last_idx = -999
    for idx in range(200, n):
        ob_r = merged_ob.iloc[idx].get("ob_ratio", np.nan)
        if pd.isna(ob_r) or abs(ob_r) < ob_thresh:
            continue
        ob_direction = "LONG" if ob_r > 0 else "SHORT"
        if idx - last_idx < 8:
            continue
        last_idx = idx
        ret = merged_ob.iloc[idx]["fwd_ret_16"]
        if pd.isna(ret):
            continue
        dir_mult = 1 if ob_direction == "LONG" else -1
        signals.append(dir_mult * ret)

    if len(signals) < 5:
        continue
    rets = np.array(signals)
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    print(f"  OB>{ob_thresh}: n={len(rets)} WR={wr*100:.1f}% mean={mean_r*100:+.3f}% PF={pf:.2f} p={p1:.4f}")

# ═══════════════════════════════════════════════════════
# TAKER ALONE (no OB)
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("TAKER ALONE (no OB state)")
print("="*70)

for taker_thresh in taker_thresholds:
    signals = []
    last_idx = -999
    for idx in range(200, n):
        tz = taker_zscores[idx]
        if np.isnan(tz) or abs(tz) < taker_thresh:
            continue
        direction = "LONG" if tz > 0 else "SHORT"
        if idx - last_idx < 8:
            continue
        last_idx = idx
        ret = merged_ob.iloc[idx]["fwd_ret_16"]
        if pd.isna(ret):
            continue
        dir_mult = 1 if direction == "LONG" else -1
        signals.append(dir_mult * ret)

    if len(signals) < 5:
        continue
    rets = np.array(signals)
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2
    print(f"  Z>{taker_thresh}: n={len(rets)} WR={wr*100:.1f}% mean={mean_r*100:+.3f}% PF={pf:.2f} p={p1:.4f}")

# ═══════════════════════════════════════════════════════
# BEST COMBO: Deep validation
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("BEST COMBO: Deep validation")
print("="*70)

if results:
    best = sorted(results, key=lambda x: x["mean"], reverse=True)[0]
    print(f"OB>{best['ob_thresh']} Taker>{best['taker_thresh']}")
    print(f"n={best['n']} WR={best['wr']*100:.1f}% mean={best['mean']*100:+.3f}% PF={best['pf']:.2f}")
    print(f"CI=[{best['ci_lo']*100:+.3f}%, {best['ci_hi']*100:+.3f}%] p={best['p']:.4f}")

    # Walk-forward
    sdf = pd.DataFrame(best["signals"])
    sdf["week"] = pd.to_datetime(sdf["idx"].apply(lambda i: merged_ob.iloc[i]["timestamp"])).dt.isocalendar().week.astype(int)
    sdf["year"] = pd.to_datetime(sdf["idx"].apply(lambda i: merged_ob.iloc[i]["timestamp"])).dt.isocalendar().year.astype(int)
    sdf["yw"] = sdf["year"] * 100 + sdf["week"]
    weeks = sorted(sdf["yw"].unique())
    wf = []
    for i in range(0, len(weeks)-4):
        test_w = weeks[i+4] if i+4 < len(weeks) else None
        if not test_w:
            break
        test = sdf[sdf["yw"] == test_w]
        if len(test) >= 2:
            rets = test["adj_ret"].values
            wf.append({"week": test_w, "n": len(rets), "wr": (rets > 0).mean(), "mean": rets.mean()})
    if wf:
        wf_df = pd.DataFrame(wf)
        win_periods = (wf_df["wr"] > 0.5).sum()
        print(f"\nWalk-forward: {len(wf_df)} periods, {win_periods}/{len(wf_df)} winning ({win_periods/len(wf_df)*100:.0f}%)")
        print(f"WF mean WR: {wf_df['wr'].mean()*100:.1f}%")
        print(f"WF mean return: {wf_df['mean'].mean()*100:+.3f}%")

    # Monte Carlo
    rets = best["signals"]
    adj_rets = np.array([s["adj_ret"] for s in rets])
    sims = [np.random.choice(adj_rets, size=min(30, len(adj_rets)), replace=True).sum() for _ in range(5000)]
    sims = np.array(sims)
    p5, p50, p95 = np.percentile(sims, [5, 50, 95])
    print(f"\nMC: P5={p5*100:+.2f}% P50={p50*100:+.2f}% P95={p95*100:+.2f}% Prob(loss)={(sims<0).mean()*100:.1f}%")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    if best["ci_lo"] > 0 and best["wr"] > 0.52 and best["p"] < 0.10:
        print("PASS - OB+taker combined has edge")
    elif best["mean"] > 0 and best["wr"] > 0.50:
        print("PROVISIONAL - Positive but needs more data")
    else:
        print("FAIL - No edge from OB+taker combination")

print("\nDone.")
