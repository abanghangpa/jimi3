#!/usr/bin/env python3
"""
Real-Data Isolation Gate — 4 Borderline Strategies
Tests regime_switch, kill_zone, structural_break, vol_rotation on real ETH 15m data.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


# === INDICATORS ===
def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()
def calc_rsi(s, p=14):
    d = s.diff(); g = d.where(d>0,0).rolling(p).mean(); l = (-d.where(d<0,0)).rolling(p).mean()
    return 100 - (100/(1+g/l.replace(0,1)))
def calc_atr(h,l,c,p=14):
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()
def calc_vol_ratio(v, p=20): return v / v.rolling(p).mean().replace(0,1)


# === DETECTION FUNCTIONS (same as synthetic gate v2) ===

def detect_vol_rotation(df):
    v = df["Volume"]; c = df["Close"]
    vol_ratio = calc_vol_ratio(v, 20)
    vol_ratio_ma = vol_ratio.rolling(10).mean()
    expanding = (vol_ratio > 1.5) & (vol_ratio_ma.shift(5) < 0.8)
    ema20 = calc_ema(c, 20)
    direction = pd.Series("NONE", index=df.index)
    direction[expanding & (c > ema20)] = "LONG"
    direction[expanding & (c < ema20)] = "SHORT"
    events = expanding & (direction != "NONE")
    return events, direction

def detect_kill_zone(df):
    c = df["Close"]
    # Parse hour from timestamp
    ts = pd.to_datetime(df["Open time"])
    hour = ts.dt.hour
    in_kill_zone = ((hour >= 7) & (hour <= 10)) | ((hour >= 13) & (hour <= 16))
    ema20 = calc_ema(c, 20)
    direction = pd.Series("NONE", index=df.index)
    direction[in_kill_zone & (c > ema20)] = "LONG"
    direction[in_kill_zone & (c < ema20)] = "SHORT"
    events = in_kill_zone & (direction != "NONE")
    return events, direction

def detect_structural_break(df):
    c = df["Close"]; h = df["High"]; l = df["Low"]
    resistance = h.rolling(50).max().shift(1)
    support = l.rolling(50).min().shift(1)
    vol_ratio = calc_vol_ratio(df["Volume"], 20)
    break_up = (c > resistance) & (vol_ratio > 1.3)
    break_down = (c < support) & (vol_ratio > 1.3)
    break_up_fresh = break_up & ~break_up.shift(1).fillna(False)
    break_down_fresh = break_down & ~break_down.shift(1).fillna(False)
    events = break_up_fresh | break_down_fresh
    direction = pd.Series("NONE", index=df.index)
    direction[break_up_fresh] = "LONG"
    direction[break_down_fresh] = "SHORT"
    return events, direction

def detect_regime_switch(df):
    c = df["Close"]; v = df["Volume"]
    atr = calc_atr(df["High"], df["Low"], df["Close"], 14)
    atr_pctl = atr.rolling(100, min_periods=20).apply(
        lambda x: (x.iloc[-1]-x.min())/(x.max()-x.min()) if x.max()>x.min() else 0.5, raw=False)
    vol_expanding = (atr_pctl > 0.8) & (atr_pctl.shift(10) < 0.5)
    ema20 = calc_ema(c, 20)
    direction = pd.Series("NONE", index=df.index)
    direction[vol_expanding & (c > ema20)] = "LONG"
    direction[vol_expanding & (c < ema20)] = "SHORT"
    events = vol_expanding & (direction != "NONE")
    return events, direction


# === ISOLATION GATE ===
def run_gate(df, events, direction, name, cost=0.001):
    close = df["Close"].values; n = len(close)
    idx = np.where(events)[0]
    if len(idx) < 5:
        return {"events": len(idx), "gate_passed": False, "reason": f"Too few ({len(idx)})"}

    horizons = [1, 4, 16, 24]
    results = {}
    for h in horizons:
        fr = []
        for i in idx:
            if i+h < n: fr.append((close[i+h]-close[i])/close[i])
        if len(fr) < 3:
            results[f"{h}bar"] = {"mean_pct": None, "n": len(fr), "p": None}
            continue
        fr = np.array(fr); mean_r = np.mean(fr)
        ne_mask = np.ones(n, dtype=bool); ne_mask[idx] = False
        ne_idx = np.where(ne_mask)[0]
        ne = np.array([(close[i+h]-close[i])/close[i] for i in ne_idx if i+h < n])
        if len(fr)>1 and len(ne)>1:
            t, p = stats.ttest_ind(fr, ne, equal_var=False)
        else: t, p = 0.0, 1.0
        results[f"{h}bar"] = {"mean_pct": round(mean_r*100,4), "n": len(fr), "p": round(float(p),4)}

    best_h, best_p, best_m = None, 1.0, 0.0
    for hs, r in results.items():
        if r.get("mean_pct") is None: continue
        if r["p"] < best_p: best_p=r["p"]; best_h=hs; best_m=r["mean_pct"]

    dir_ok = best_m > 0; p_ok = best_p < 0.1; eff_ok = abs(best_m) > cost*100
    passed = dir_ok and p_ok and eff_ok
    reasons = []
    if not dir_ok: reasons.append(f"dir backwards ({best_m:+.4f}%)")
    if not p_ok: reasons.append(f"p={best_p:.4f}")
    if not eff_ok: reasons.append(f"effect {abs(best_m):.4f}% < costs")

    return {"events": len(idx), "horizons": results, "best_h": best_h,
            "best_p": best_p, "best_mean_pct": best_m, "dir_correct": dir_ok,
            "gate_passed": passed, "reason": "; ".join(reasons) if reasons else "PASS"}


# === MAIN ===
def main():
    CSV = "/root/.openclaw/workspace/jimi_audit/data/eth_15m_merged.csv"
    df = pd.read_csv(CSV)
    print(f"Loaded: {len(df)} bars ({df['Open time'].iloc[0]} -> {df['Open time'].iloc[-1]})")

    # Also compute 1h data for mtf-like checks
    df["Open time_dt"] = pd.to_datetime(df["Open time"])

    strategies = {
        "regime_switch": detect_regime_switch,
        "kill_zone": detect_kill_zone,
        "structural_break": detect_structural_break,
        "vol_rotation": detect_vol_rotation,
    }

    all_results = {}
    for name, fn in strategies.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        events, direction = fn(df)
        result = run_gate(df, events, direction, name)
        all_results[name] = result

        # Per-horizon breakdown
        for h_str, r in result.get("horizons", {}).items():
            if r.get("mean_pct") is not None:
                print(f"  {h_str}: n={r['n']:>5} | mean={r['mean_pct']:>+8.4f}% | p={r['p']:.4f}")

        status = "PASS" if result["gate_passed"] else "FAIL"
        print(f"\n  RESULT: {status} | events={result['events']} | best={result.get('best_h','?')} | mean={result.get('best_mean_pct',0):+.4f}% | p={result.get('best_p',1):.4f}")
        print(f"  {result['reason']}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  REAL DATA GATE SUMMARY")
    print(f"{'='*60}")
    for name, r in all_results.items():
        status = "PASS" if r["gate_passed"] else "FAIL"
        print(f"  [{status:4s}] {name:25s} | n={r['events']:>5} | mean={r.get('best_mean_pct',0):>+.4f}% | p={r.get('best_p',1):.4f} | {r['reason'][:40]}")

    # Save
    out_dir = "/root/.openclaw/workspace/jimi_audit/reports/real_data_gates"
    os.makedirs(out_dir, exist_ok=True)
    for name, r in all_results.items():
        with open(f"{out_dir}/{name}_real_gate.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir}/")

if __name__ == "__main__":
    main()
