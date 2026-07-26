"""
Triple Combo v2: OB + Taker + S/R (full overlap period)
OB data: May 12 - Jul 19, 2026 (~5M rows)
OHLCV: until Jul 13, 2026
Overlap: ~60 days
"""
import pandas as pd
import numpy as np
from scipy import stats
import json, os, subprocess

DATA_DIR = "/root/.openclaw/workspace/jimi_audit/data"
DERIV_DIR = f"{DATA_DIR}/derivatives_history"

ohlcv = pd.read_csv(f"{DATA_DIR}/eth_15m_extended.csv")
ohlcv["timestamp"] = pd.to_datetime(ohlcv["Open time"])
ohlcv = ohlcv.sort_values("timestamp").reset_index(drop=True)

deriv = pd.read_csv(f"{DERIV_DIR}/derivatives_collected.csv")
deriv["timestamp"] = pd.to_datetime(deriv["timestamp"], format="mixed", utc=True).dt.tz_localize(None)
deriv = deriv.sort_values("timestamp").reset_index(drop=True)

merged = pd.merge_asof(ohlcv, deriv[["timestamp","oi","ls_ratio","funding_rate"]],
                       on="timestamp", direction="backward", tolerance=pd.Timedelta("30min"))
merged["vol_ratio"] = merged["Volume"] / merged["Volume"].rolling(20).mean()
for h in [4, 8, 16, 24]:
    merged["fwd_ret_" + str(h)] = merged["Close"].shift(-h) / merged["Close"] - 1

closes = merged["Close"].values
highs_arr = merged["High"].values
lows_arr = merged["Low"].values
volumes = merged["Volume"].values
taker_base = merged["Taker buy base asset volume"].values
n = len(merged)

# Load OB data - use last 5M rows
ob_path = f"{DATA_DIR}/ob_history/ob_historical.csv"
temp = "/tmp/ob_v2.csv"
subprocess.run("head -1 " + ob_path + " > " + temp + " && tail -n 5000000 " + ob_path + " >> " + temp, shell=True, check=True)
ob = pd.read_csv(temp, usecols=["timestamp","ob_ratio","top5_ratio"], on_bad_lines="skip", low_memory=False)
ob["timestamp"] = pd.to_datetime(ob["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
ob = ob.dropna(subset=["timestamp"])
ob = ob.sort_values("timestamp").reset_index(drop=True)
for col in ["ob_ratio","top5_ratio"]:
    ob[col] = pd.to_numeric(ob[col], errors="coerce")
print(f"OB data: {len(ob)} rows, {ob['timestamp'].min()} -> {ob['timestamp'].max()}")

merged_ob = pd.merge_asof(merged, ob[["timestamp","ob_ratio","top5_ratio"]],
                          on="timestamp", direction="nearest", tolerance=pd.Timedelta("15min"))
ob_coverage = merged_ob["ob_ratio"].notna().sum()
print(f"OB coverage: {ob_coverage} / {n} ({ob_coverage/n*100:.1f}%)")

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

# Precompute S/R proximity
print("Precomputing S/R proximity...")
sr_near = np.full(n, False)
for idx in range(96, n):
    price = closes[idx]
    trs = np.abs(np.diff(highs_arr[idx-14:idx+1]))
    atr = np.mean(trs) if len(trs) > 0 else price * 0.01
    if atr == 0:
        atr = price * 0.01
    for i in range(3, min(96, idx)):
        bar_idx = idx - i
        if bar_idx < 3:
            continue
        if (highs_arr[bar_idx] > highs_arr[bar_idx-1] and
            highs_arr[bar_idx] > highs_arr[bar_idx-2] and
            highs_arr[bar_idx] > highs_arr[bar_idx-3]):
            if abs(price - highs_arr[bar_idx]) / atr < 1.5:
                sr_near[idx] = True
                break
        if (lows_arr[bar_idx] < lows_arr[bar_idx-1] and
            lows_arr[bar_idx] < lows_arr[bar_idx-2] and
            lows_arr[bar_idx] < lows_arr[bar_idx-3]):
            if abs(price - lows_arr[bar_idx]) / atr < 1.5:
                sr_near[idx] = True
                break
print("Done")

# ═══════════════════════════════════════════════════════
# TEST ALL COMBINATIONS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("TRIPLE COMBINATION: OB + Taker + S/R (full overlap)")
print("="*70)

configs = [
    {"name": "OB(0.3)", "ob": 0.3, "taker": None, "sr": False},
    {"name": "OB(0.3)+SR", "ob": 0.3, "taker": None, "sr": True},
    {"name": "OB(0.3)+T1.5", "ob": 0.3, "taker": 1.5, "sr": False},
    {"name": "OB(0.3)+T1.5+SR", "ob": 0.3, "taker": 1.5, "sr": True},
    {"name": "OB(0.5)", "ob": 0.5, "taker": None, "sr": False},
    {"name": "OB(0.5)+SR", "ob": 0.5, "taker": None, "sr": True},
    {"name": "OB(0.5)+T1.5", "ob": 0.5, "taker": 1.5, "sr": False},
    {"name": "OB(0.5)+T1.5+SR", "ob": 0.5, "taker": 1.5, "sr": True},
    {"name": "OB(0.7)", "ob": 0.7, "taker": None, "sr": False},
    {"name": "OB(0.7)+SR", "ob": 0.7, "taker": None, "sr": True},
    {"name": "OB(0.7)+T1.5", "ob": 0.7, "taker": 1.5, "sr": False},
    {"name": "OB(0.7)+T1.5+SR", "ob": 0.7, "taker": 1.5, "sr": True},
    {"name": "OB(1.0)", "ob": 1.0, "taker": None, "sr": False},
    {"name": "OB(1.0)+SR", "ob": 1.0, "taker": None, "sr": True},
    {"name": "OB(1.0)+T1.5", "ob": 1.0, "taker": 1.5, "sr": False},
    {"name": "OB(1.0)+T1.5+SR", "ob": 1.0, "taker": 1.5, "sr": True},
    {"name": "SR only", "ob": None, "taker": None, "sr": True},
    {"name": "T1.5 only", "ob": None, "taker": 1.5, "sr": False},
]

all_results = []
for cfg in configs:
    signals = []
    last_idx = -999
    for idx in range(200, n):
        direction = None
        if cfg["ob"] is not None:
            ob_r = merged_ob.iloc[idx].get("ob_ratio", np.nan)
            if pd.isna(ob_r) or abs(ob_r) < cfg["ob"]:
                continue
            direction = "LONG" if ob_r > 0 else "SHORT"
        if cfg["taker"] is not None:
            tz = taker_zscores[idx]
            if np.isnan(tz) or abs(tz) < cfg["taker"]:
                continue
            if direction is None:
                direction = "LONG" if tz > 0 else "SHORT"
            else:
                confirms = (direction == "LONG" and tz > cfg["taker"]) or (direction == "SHORT" and tz < -cfg["taker"])
                if not confirms:
                    continue
        if cfg["sr"] and not sr_near[idx]:
            continue
        if direction is None:
            continue
        if idx - last_idx < 8:
            continue
        last_idx = idx
        ret = merged_ob.iloc[idx]["fwd_ret_16"]
        if pd.isna(ret):
            continue
        dir_mult = 1 if direction == "LONG" else -1
        signals.append({"adj_ret": dir_mult * ret, "idx": idx, "direction": direction})

    if len(signals) < 5:
        all_results.append({"name": cfg["name"], "n": len(signals), "wr": 0, "mean": 0, "pf": 0, "p": 1.0})
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

    all_results.append({"name": cfg["name"], "n": len(signals), "wr": wr, "mean": mean_r, "pf": pf, "p": p1, "ci_lo": ci_lo, "ci_hi": ci_hi, "rets": rets, "signals": signals})

print(f"\n{'Config':<22} {'n':>6} {'WR':>8} {'Mean':>10} {'PF':>8} {'p':>8} {'CI':>20}")
print("-"*85)
for r in sorted(all_results, key=lambda x: x.get("mean", 0), reverse=True):
    tag = "PASS" if r.get("ci_lo", 0) > 0 and r.get("wr", 0) > 0.52 else ("PROV" if r.get("mean", 0) > 0 else "FAIL")
    ci = f"[{r.get('ci_lo',0)*100:+.2f}%, {r.get('ci_hi',0)*100:+.2f}%]" if r.get("ci_lo") else "n/a"
    print(f"[{tag}] {r['name']:<22} {r['n']:>6} {r['wr']*100:>7.1f}% {r['mean']*100:>+9.3f}% {r['pf']:>7.2f} {r['p']:>7.4f} {ci:>20}")

# ═══════════════════════════════════════════════════════
# BEST CONFIG: Walk-forward + MC
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("BEST CONFIG: Walk-forward validation")
print("="*70)

best = sorted([r for r in all_results if r.get("ci_lo", 0) > 0 and r["n"] >= 10], key=lambda x: x["mean"], reverse=True)
if best:
    b = best[0]
    print(f"Config: {b['name']}")
    print(f"n={b['n']} WR={b['wr']*100:.1f}% mean={b['mean']*100:+.3f}% PF={b['pf']:.2f} p={b['p']:.4f}")
    print(f"CI=[{b['ci_lo']*100:+.3f}%, {b['ci_hi']*100:+.3f}%]")

    # Walk-forward
    sdf = pd.DataFrame(b["signals"])
    sdf["ts"] = sdf["idx"].apply(lambda i: merged_ob.iloc[i]["timestamp"])
    sdf["week"] = pd.to_datetime(sdf["ts"]).dt.isocalendar().week.astype(int)
    sdf["year"] = pd.to_datetime(sdf["ts"]).dt.isocalendar().year.astype(int)
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

    # MC
    rets = b["rets"]
    sims = [np.random.choice(rets, size=min(30, len(rets)), replace=True).sum() for _ in range(5000)]
    sims = np.array(sims)
    p5, p50, p95 = np.percentile(sims, [5, 50, 95])
    print(f"\nMC: P5={p5*100:+.2f}% P50={p50*100:+.2f}% P95={p95*100:+.2f}% Prob(loss)={(sims<0).mean()*100:.1f}%")
else:
    print("No config with CI>0 found")

# ═══════════════════════════════════════════════════════
# LAYER COMPARISON
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("LAYER COMPARISON: Does adding each layer help?")
print("="*70)

def get_r(name):
    for r in all_results:
        if r["name"] == name:
            return r
    return None

for ob_t in [0.3, 0.5, 0.7, 1.0]:
    base = get_r(f"OB({ob_t})")
    sr = get_r(f"OB({ob_t})+SR")
    taker = get_r(f"OB({ob_t})+T1.5")
    triple = get_r(f"OB({ob_t})+T1.5+SR")
    if not base or base["n"] < 3:
        continue
    print(f"\nOB>{ob_t}:")
    for label, r in [("  alone", base), ("  +S/R", sr), ("  +Taker", taker), ("  +Taker+S/R", triple)]:
        if r and r["n"] >= 3:
            print(f"    {label:<15} n={r['n']:>4} WR={r['wr']*100:.1f}% mean={r['mean']*100:+.3f}% PF={r['pf']:.2f}")

print("\nDone.")
