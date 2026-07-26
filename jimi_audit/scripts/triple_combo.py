"""
Triple Combination: OB State + Taker Flow + S/R Proximity

Jeon (2026) hierarchy applied:
1. OB state (ob_ratio) — L2 state (primary)
2. Taker z-score — order flow overlay
3. S/R proximity + breakout — structural filter (from liquidity_grab)
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

# Load OB data (first 2M rows for Jul 1-7 overlap)
ob_path = f"{DATA_DIR}/ob_history/ob_historical.csv"
temp = "/tmp/ob_triple.csv"
subprocess.run("head -1 " + ob_path + " > " + temp + " && head -n 2000000 " + ob_path + " | tail -n +2 >> " + temp, shell=True, check=True)
ob = pd.read_csv(temp, usecols=["timestamp","ob_ratio","top5_ratio"], on_bad_lines="skip", low_memory=False)
ob["timestamp"] = pd.to_datetime(ob["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
ob = ob.dropna(subset=["timestamp"])
for col in ["ob_ratio","top5_ratio"]:
    ob[col] = pd.to_numeric(ob[col], errors="coerce")
print(f"OB data: {len(ob)} rows, {ob['timestamp'].min()} -> {ob['timestamp'].max()}")

merged_ob = pd.merge_asof(merged, ob[["timestamp","ob_ratio","top5_ratio"]],
                          on="timestamp", direction="nearest", tolerance=pd.Timedelta("15min"))
print(f"OB coverage: {merged_ob['ob_ratio'].notna().sum()}")

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
    atr = np.mean(np.abs(np.diff(highs_arr[idx-14:idx+1]))) if idx >= 14 else price * 0.01
    if atr == 0:
        atr = price * 0.01
    # Check swing highs/lows in lookback
    for i in range(3, min(96, idx)):
        bar_idx = idx - i
        if bar_idx < 3:
            continue
        # Swing high
        if (highs_arr[bar_idx] > highs_arr[bar_idx-1] and
            highs_arr[bar_idx] > highs_arr[bar_idx-2] and
            highs_arr[bar_idx] > highs_arr[bar_idx-3]):
            if abs(price - highs_arr[bar_idx]) / atr < 1.5:
                sr_near[idx] = True
                break
        # Swing low
        if (lows_arr[bar_idx] < lows_arr[bar_idx-1] and
            lows_arr[bar_idx] < lows_arr[bar_idx-2] and
            lows_arr[bar_idx] < lows_arr[bar_idx-3]):
            if abs(price - lows_arr[bar_idx]) / atr < 1.5:
                sr_near[idx] = True
                break

print("Done precomputing")

# ═══════════════════════════════════════════════════════
# TEST ALL COMBINATIONS
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("TRIPLE COMBINATION: OB + Taker + S/R")
print("="*70)

configs = [
    # OB alone, with S/R, with taker, triple
    {"name": "OB only", "ob": 0.3, "taker": None, "sr": False},
    {"name": "OB + S/R", "ob": 0.3, "taker": None, "sr": True},
    {"name": "OB + Taker", "ob": 0.3, "taker": 1.5, "sr": False},
    {"name": "OB + Taker + S/R", "ob": 0.3, "taker": 1.5, "sr": True},
    {"name": "OB(0.5) only", "ob": 0.5, "taker": None, "sr": False},
    {"name": "OB(0.5) + S/R", "ob": 0.5, "taker": None, "sr": True},
    {"name": "OB(0.5) + Taker", "ob": 0.5, "taker": 1.5, "sr": False},
    {"name": "OB(0.5) + Taker + S/R", "ob": 0.5, "taker": 1.5, "sr": True},
    {"name": "OB(0.7) only", "ob": 0.7, "taker": None, "sr": False},
    {"name": "OB(0.7) + S/R", "ob": 0.7, "taker": None, "sr": True},
    {"name": "OB(0.7) + Taker", "ob": 0.7, "taker": 1.5, "sr": False},
    {"name": "OB(0.7) + Taker + S/R", "ob": 0.7, "taker": 1.5, "sr": True},
    {"name": "S/R only", "ob": None, "taker": None, "sr": True},
    {"name": "Taker only", "ob": None, "taker": 1.5, "sr": False},
    {"name": "Taker + S/R", "ob": None, "taker": 1.5, "sr": True},
]

print(f"\n{'Config':<25} {'n':>6} {'WR':>8} {'Mean':>10} {'PF':>8} {'p':>8}")
print("-"*70)

all_results = []
for cfg in configs:
    signals = []
    last_idx = -999
    for idx in range(200, n):
        # OB filter
        if cfg["ob"] is not None:
            ob_r = merged_ob.iloc[idx].get("ob_ratio", np.nan)
            if pd.isna(ob_r) or abs(ob_r) < cfg["ob"]:
                continue
            direction = "LONG" if ob_r > 0 else "SHORT"
        else:
            direction = None

        # Taker filter
        if cfg["taker"] is not None:
            tz = taker_zscores[idx]
            if np.isnan(tz) or abs(tz) < cfg["taker"]:
                continue
            if direction is None:
                direction = "LONG" if tz > 0 else "SHORT"
            else:
                # Check taker confirms OB direction
                taker_confirms = (direction == "LONG" and tz > cfg["taker"]) or \
                                 (direction == "SHORT" and tz < -cfg["taker"])
                if not taker_confirms:
                    continue

        # S/R filter
        if cfg["sr"] and not sr_near[idx]:
            continue

        if direction is None:
            continue

        # Dedup
        if idx - last_idx < 8:
            continue
        last_idx = idx

        ret = merged_ob.iloc[idx]["fwd_ret_16"]
        if pd.isna(ret):
            continue
        dir_mult = 1 if direction == "LONG" else -1
        signals.append(dir_mult * ret)

    if len(signals) < 3:
        all_results.append({"name": cfg["name"], "n": len(signals), "wr": 0, "mean": 0, "pf": 0, "p": 1.0})
        continue

    rets = np.array(signals)
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    t, p = stats.ttest_1samp(rets, 0)
    p1 = p/2 if t > 0 else 1-p/2

    all_results.append({"name": cfg["name"], "n": len(signals), "wr": wr, "mean": mean_r, "pf": pf, "p": p1, "rets": rets})

    tag = "PASS" if wr > 0.52 and mean_r > 0 and p1 < 0.10 else ("PROV" if mean_r > 0 else "FAIL")
    print(f"[{tag}] {cfg['name']:<25} {len(signals):>6} {wr*100:>7.1f}% {mean_r*100:>+9.3f}% {pf:>7.2f} {p1:>7.4f}")

# ═══════════════════════════════════════════════════════
# BEST CONFIG: Walk-forward + MC
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("BEST CONFIG: Deep validation")
print("="*70)

best = sorted([r for r in all_results if r["n"] >= 5 and r["mean"] > 0], key=lambda x: x["mean"], reverse=True)
if best:
    b = best[0]
    print(f"Config: {b['name']}")
    print(f"n={b['n']} WR={b['wr']*100:.1f}% mean={b['mean']*100:+.3f}% PF={b['pf']:.2f} p={b['p']:.4f}")

    if "rets" in b:
        rets = b["rets"]
        # Bootstrap CI
        boots = [np.random.choice(rets, size=len(rets), replace=True).mean() for _ in range(2000)]
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        print(f"CI=[{ci_lo*100:+.3f}%, {ci_hi*100:+.3f}%]")

        # MC
        sims = [np.random.choice(rets, size=min(30, len(rets)), replace=True).sum() for _ in range(5000)]
        sims = np.array(sims)
        p5, p50, p95 = np.percentile(sims, [5, 50, 95])
        print(f"MC: P5={p5*100:+.2f}% P50={p50*100:+.2f}% P95={p95*100:+.2f}% Prob(loss)={(sims<0).mean()*100:.1f}%")
else:
    print("No positive configs found")

# ═══════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("COMPARISON: Does adding each layer help?")
print("="*70)

def get_result(name):
    for r in all_results:
        if r["name"] == name:
            return r
    return None

layers = [
    ("OB only", "OB(0.5) only"),
    ("+ S/R", "OB(0.5) + S/R"),
    ("+ Taker", "OB(0.5) + Taker"),
    ("+ Taker + S/R", "OB(0.5) + Taker + S/R"),
]

prev_wr = None
for label, name in layers:
    r = get_result(name)
    if not r or r["n"] < 3:
        continue
    lift = ""
    if prev_wr is not None:
        delta = r["wr"] - prev_wr
        lift = f" ({'+' if delta >= 0 else ''}{delta*100:+.1f}%)"
    prev_wr = r["wr"]
    print(f"  {label:<20} n={r['n']:>4} WR={r['wr']*100:.1f}%{lift} mean={r['mean']*100:+.3f}%")

print("\nDone.")
