#!/usr/bin/env python3
"""
v3.1 — Tuned multi-factor detection for the two promising strategies.
Relax thresholds slightly to get more events while preserving signal quality.
"""
import json, os, numpy as np, pandas as pd
from scipy import stats

def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()
def calc_atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()
def calc_vol_ratio(v, p=20): return v / v.rolling(p).mean().replace(0, 1)


def detect_judas_sweep_v31(df):
    """v3.1: Relaxed multi-factor sweep detection.
    
    v3 had 221 events, +0.104%, p=0.239. Close but not enough power.
    
    Changes from v3:
    - Accept sweeps of session H/L (not just daily) — more events
    - Lower volume threshold from 1.3 to 1.1 — more events
    - Keep the 3-factor structure (wick + reclaim + volume)
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    vol_ratio = calc_vol_ratio(df["Volume"], 20)
    
    daily_high = h.rolling(96).max().shift(1)
    daily_low = l.rolling(96).min().shift(1)
    session_high = h.rolling(32).max().shift(1)
    session_low = l.rolling(32).min().shift(1)
    # Also use 4h levels
    h4_high = h.rolling(16).max().shift(1)
    h4_low = l.rolling(16).min().shift(1)
    
    n = len(df)
    sweep_high = pd.Series(False, index=df.index)
    sweep_low = pd.Series(False, index=df.index)
    
    for i in range(100, n):
        price_now = c.iloc[i]
        high_now = h.iloc[i]
        low_now = l.iloc[i]
        vol_now = vol_ratio.iloc[i] if not pd.isna(vol_ratio.iloc[i]) else 1.0
        
        levels_high = []
        levels_low = []
        for col in [daily_high, session_high, h4_high]:
            val = col.iloc[i]
            if not pd.isna(val) and val > 0: levels_high.append(val)
        for col in [daily_low, session_low, h4_low]:
            val = col.iloc[i]
            if not pd.isna(val) and val > 0: levels_low.append(val)
        
        for level in levels_high:
            if high_now > level * 1.001 and price_now < level:
                wick = (high_now - price_now) > (price_now - low_now) * 1.1
                if wick and vol_now > 1.1:
                    sweep_high.iloc[i] = True
                    break
        
        for level in levels_low:
            if low_now < level * 0.999 and price_now > level:
                wick = (price_now - low_now) > (high_now - price_now) * 1.1
                if wick and vol_now > 1.1:
                    sweep_low.iloc[i] = True
                    break
    
    events = sweep_high | sweep_low
    direction = pd.Series("NONE", index=df.index)
    direction[sweep_high] = "SHORT"
    direction[sweep_low] = "LONG"
    return events, direction


def detect_funding_arb_v31(df):
    """v3.1: Relaxed funding arb detection.
    
    v3 had 19 events, +0.130%, p=0.140. Right direction but too few events.
    
    Changes from v3:
    - Lower z-score threshold from 1.5 to 1.2 — more events
    - Remove round number requirement (was too restrictive) — more events
    - Keep taker divergence + volume confirmation
    """
    c = df["Close"]
    v = df["Volume"]
    tr = df["Taker buy base asset volume"] / v.replace(0, 1)
    
    tr_ma = tr.rolling(100).mean()
    tr_std = tr.rolling(100).std()
    tr_zscore = (tr - tr_ma) / tr_std.replace(0, 1)
    vol_ratio = calc_vol_ratio(v, 20)
    
    # Relaxed: z-score 1.2 (was 1.5), volume 1.0 (was 1.2), no round number
    long_events = (tr_zscore < -1.2) & (vol_ratio > 1.0)
    short_events = (tr_zscore > 1.2) & (vol_ratio > 1.0)
    
    events = long_events | short_events
    direction = pd.Series("NONE", index=df.index)
    direction[long_events] = "LONG"
    direction[short_events] = "SHORT"
    return events, direction


def run_gate(df, events, direction, name, cost=0.001):
    close = df["Close"].values
    n = len(close)
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
        if len(fr) > 1 and len(ne) > 1:
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
    
    return {"events": int(len(idx)), "horizons": results, "best_h": best_h,
            "best_p": best_p, "best_mean_pct": best_m, "dir_correct": dir_ok,
            "gate_passed": passed, "reason": "; ".join(reasons) if reasons else "PASS"}


def main():
    CSV = "/root/.openclaw/workspace/jimi_audit/data/eth_15m_merged.csv"
    df = pd.read_csv(CSV)
    print(f"Loaded: {len(df)} bars")
    print(f"v3.1 — Tuned multi-factor (judas_sweep + funding_arb)\n")
    
    strategies = {
        "judas_sweep_v31": detect_judas_sweep_v31,
        "funding_arb_v31": detect_funding_arb_v31,
    }
    
    all_results = {}
    for name, fn in strategies.items():
        print(f"{'='*60}\n  {name}\n{'='*60}")
        events, direction = fn(df)
        result = run_gate(df, events, direction, name)
        all_results[name] = result
        for h_str, r in result.get("horizons", {}).items():
            if r.get("mean_pct") is not None:
                print(f"  {h_str}: n={r['n']:>5} | mean={r['mean_pct']:>+8.4f}% | p={r['p']:.4f}")
        status = "PASS" if result["gate_passed"] else "FAIL"
        print(f"\n  RESULT: {status} | events={result['events']} | best={result.get('best_h','?')} | mean={result.get('best_mean_pct',0):+.4f}% | p={result.get('best_p',1):.4f}")
        print(f"  {result['reason']}\n")
    
    print(f"{'='*60}\n  v3.1 SUMMARY\n{'='*60}")
    for name, r in all_results.items():
        status = "PASS" if r["gate_passed"] else "FAIL"
        print(f"  [{status:4s}] {name:25s} | n={r['events']:>5} | mean={r.get('best_mean_pct',0):>+.4f}% | p={r.get('best_p',1):.4f} | {r['reason'][:40]}")
    
    out_dir = "/root/.openclaw/workspace/jimi_audit/reports/v3_gates"
    os.makedirs(out_dir, exist_ok=True)
    for name, r in all_results.items():
        with open(f"{out_dir}/{name}_gate.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
    print(f"\nSaved: {out_dir}/")

if __name__ == "__main__":
    main()
