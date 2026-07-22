#!/usr/bin/env python3
"""
Fixable Strategies v3 — Multi-Factor Detection
Each strategy combines 2-3 confirmation signals instead of single-variable detection.

Key insight: the CONCEPTS are valid market mechanics. The failure was in
detecting them with single variables. Real market events are multi-factor.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats


# === INDICATORS ===
def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()

def calc_atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def calc_vol_ratio(v, p=20): return v / v.rolling(p).mean().replace(0, 1)

def calc_rsi(s, p=14):
    d = s.diff(); g = d.where(d>0,0).rolling(p).mean(); l = (-d.where(d<0,0)).rolling(p).mean()
    return 100 - (100/(1+g/l.replace(0,1)))


# === V3 DETECTION: MULTI-FACTOR ===

def detect_funding_arb_v3(df):
    """v3: Funding rate differential across exchanges.
    
    The CONCEPT: When funding rates diverge across exchanges, arbitrageurs
    converge them. We can't see cross-exchange FR directly, but we can
    detect it through taker flow divergence.
    
    Multi-factor:
    1. Taker ratio divergence from recent mean (informed flow)
    2. Volume spike (arb activity)
    3. Price near a round number (where arb desks are active)
    
    Why v1/v2 failed: Used FR change on single exchange. The arb is in the
    DIFFERENTIAL, not the level.
    """
    c = df["Close"]
    v = df["Volume"]
    tr = df["Taker buy base asset volume"] / v.replace(0, 1)
    
    # Factor 1: Taker ratio divergence (proxy for cross-exchange FR differential)
    tr_ma = tr.rolling(100).mean()
    tr_std = tr.rolling(100).std()
    tr_zscore = (tr - tr_ma) / tr_std.replace(0, 1)
    
    # Factor 2: Volume spike (arb desks active)
    vol_ratio = calc_vol_ratio(v, 20)
    
    # Factor 3: Price near round number ($50 levels)
    price_round_dist = (c % 50) / 50
    near_round = price_round_dist < 0.02  # within 2% of $50 level
    
    # Multi-factor: taker extreme + volume + near round number
    long_events = (tr_zscore < -1.5) & (vol_ratio > 1.2) & near_round
    short_events = (tr_zscore > 1.5) & (vol_ratio > 1.2) & near_round
    
    events = long_events | short_events
    direction = pd.Series("NONE", index=df.index)
    direction[long_events] = "LONG"
    direction[short_events] = "SHORT"
    return events, direction


def detect_squeeze_breakout_v3(df):
    """v3: Multi-factor compression -> expansion.
    
    The CONCEPT: Volatility compression + volume contraction + OI buildup
    = energy stored. Release = directional move.
    
    Multi-factor:
    1. ATR compression (< 30th percentile for 15+ bars)
    2. Volume contraction (< 0.8x average for 10+ bars) — dry powder accumulating
    3. Expansion: ATR crosses 40th percentile + volume returns
    
    Why v1/v2 failed: Only used ATR (or BB/KC). Missing the volume contraction
    confirmation. Compression without volume contraction is just low vol, not
    a squeeze.
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]
    
    atr = calc_atr(h, l, c, 14)
    atr_pctl = atr.rolling(200, min_periods=50).apply(
        lambda x: (x.iloc[-1]-x.min())/(x.max()-x.min()) if x.max()>x.min() else 0.5, raw=False)
    
    vol_ratio = calc_vol_ratio(v, 20)
    
    # Factor 1: ATR compression
    atr_compressed = atr_pctl < 0.30
    atr_compressed_count = atr_compressed.rolling(15).sum()
    was_atr_compressed = atr_compressed_count >= 12
    
    # Factor 2: Volume contraction
    vol_contracted = vol_ratio < 0.8
    vol_contracted_count = vol_contracted.rolling(10).sum()
    was_vol_contracted = vol_contracted_count >= 7
    
    # Both compressed simultaneously
    both_compressed = was_atr_compressed & was_vol_contracted
    
    # Factor 3: Expansion (ATR rising + volume returning)
    atr_expanding = (atr_pctl > 0.40) & (atr_pctl.shift(1) <= 0.40)
    vol_returning = vol_ratio > 1.0
    
    # Signal: was compressed, now expanding with volume
    breakout = both_compressed & atr_expanding & vol_returning
    
    # Direction from price momentum
    ema20 = calc_ema(c, 20)
    direction = pd.Series("NONE", index=df.index)
    direction[breakout & (c > ema20)] = "LONG"
    direction[breakout & (c < ema20)] = "SHORT"
    
    events = breakout & (direction != "NONE")
    return events, direction


def detect_judas_sweep_v3(df):
    """v3: Multi-factor liquidity sweep detection.
    
    The CONCEPT: Price sweeps a key level, traps traders, then reverses.
    The trapped traders' stops fuel the reversal.
    
    Multi-factor:
    1. Price wick through structural level (daily/session H/L or round number)
    2. Price RECLAIMS the level (closes back inside) — this is the trap
    3. Volume spike on the reclaim (stops being hit)
    
    Why v1/v2 failed: v1 used rolling fractals (noise). v2 used structural
    levels but missed the volume confirmation. The volume spike on reclaim
    is what confirms the trap — without it, it's just a normal wick.
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]
    vol_ratio = calc_vol_ratio(v, 20)
    
    # Pre-compute structural levels
    daily_high = h.rolling(96).max().shift(1)
    daily_low = l.rolling(96).min().shift(1)
    session_high = h.rolling(32).max().shift(1)
    session_low = l.rolling(32).min().shift(1)
    
    n = len(df)
    sweep_high = pd.Series(False, index=df.index)
    sweep_low = pd.Series(False, index=df.index)
    
    for i in range(100, n):
        price_now = c.iloc[i]
        high_now = h.iloc[i]
        low_now = l.iloc[i]
        prev_close = c.iloc[i-1]
        vol_now = vol_ratio.iloc[i] if not pd.isna(vol_ratio.iloc[i]) else 1.0
        
        dh = daily_high.iloc[i] if not pd.isna(daily_high.iloc[i]) else 0
        dl = daily_low.iloc[i] if not pd.isna(daily_low.iloc[i]) else 0
        sh = session_high.iloc[i] if not pd.isna(session_high.iloc[i]) else 0
        sl = session_low.iloc[i] if not pd.isna(session_low.iloc[i]) else 0
        
        # Sweep high: wick above level, close below, volume spike
        for level in [dh, sh]:
            if level > 0 and high_now > level * 1.002 and price_now < level:
                # Factor 1: Rejection wick (wick > body)
                wick_reject = (high_now - price_now) > (price_now - low_now) * 1.2
                # Factor 2: Close below level (reclaim)
                reclaimed = price_now < level
                # Factor 3: Volume spike
                vol_spike = vol_now > 1.3
                
                if wick_reject and reclaimed and vol_spike:
                    sweep_high.iloc[i] = True
                    break
        
        # Sweep low: wick below level, close above, volume spike
        for level in [dl, sl]:
            if level > 0 and low_now < level * 0.998 and price_now > level:
                wick_reject = (price_now - low_now) > (high_now - price_now) * 1.2
                reclaimed = price_now > level
                vol_spike = vol_now > 1.3
                
                if wick_reject and reclaimed and vol_spike:
                    sweep_low.iloc[i] = True
                    break
    
    events = sweep_high | sweep_low
    direction = pd.Series("NONE", index=df.index)
    direction[sweep_high] = "SHORT"
    direction[sweep_low] = "LONG"
    return events, direction


def detect_structural_break_v3(df):
    """v3: Multi-factor structural break.
    
    The CONCEPT: Breaking a key level with conviction = continuation.
    
    Multi-factor:
    1. Break of DAILY level (not 4h — daily levels are institutional)
    2. Volume confirmation (vol_ratio > 1.5 — not just > 1.2)
    3. NO immediate reclaim (close stays above/below for 4+ bars)
    
    Why v1/v2 failed: Used 50-bar and 4h levels (too many false breaks).
    Also missed the "no reclaim" check — many breaks immediately reverse.
    The 4-bar confirmation filter eliminates false breaks.
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    vol_ratio = calc_vol_ratio(df["Volume"], 20)
    
    # Daily levels (96 bars of 15m)
    daily_high = h.rolling(96).max().shift(1)
    daily_low = l.rolling(96).min().shift(1)
    
    # Factor 1: Break daily level
    break_up = (c > daily_high) & (c.shift(1) <= daily_high)
    break_down = (c < daily_low) & (c.shift(1) >= daily_low)
    
    # Factor 2: Volume confirmation (strong)
    vol_confirm = vol_ratio > 1.5
    
    # Factor 3: No immediate reclaim (close stays above/below for 4 bars)
    # We check 4 bars later — if still above/below, it's a real break
    no_reclaim_up = pd.Series(False, index=df.index)
    no_reclaim_down = pd.Series(False, index=df.index)
    for i in range(len(df) - 4):
        if break_up.iloc[i] and vol_confirm.iloc[i]:
            # Check if close stays above level for next 4 bars
            level = daily_high.iloc[i]
            if all(c.iloc[i+1:i+5] > level):
                no_reclaim_up.iloc[i] = True
        if break_down.iloc[i] and vol_confirm.iloc[i]:
            level = daily_low.iloc[i]
            if all(c.iloc[i+1:i+5] < level):
                no_reclaim_down.iloc[i] = True
    
    events = no_reclaim_up | no_reclaim_down
    direction = pd.Series("NONE", index=df.index)
    direction[no_reclaim_up] = "LONG"
    direction[no_reclaim_down] = "SHORT"
    return events, direction


def detect_regime_switch_v3(df):
    """v3: Multi-factor regime transition detection.
    
    The CONCEPT: Markets transition between regimes (trending/ranging/crisis).
    Early detection of transitions = directional edge.
    
    Multi-factor:
    1. Volatility regime change (ATR compression -> expansion)
    2. Volume regime change (low vol -> high vol)
    3. Trend direction shift (EMA crossover)
    
    All three must align for a regime switch signal.
    
    Why v1/v2 failed: Used only ATR percentile. Real regime transitions
    involve vol + volume + trend all shifting. Single-factor detection
    catches noise.
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]
    
    atr = calc_atr(h, l, c, 14)
    atr_pctl = atr.rolling(200, min_periods=50).apply(
        lambda x: (x.iloc[-1]-x.min())/(x.max()-x.min()) if x.max()>x.min() else 0.5, raw=False)
    
    vol_ratio = calc_vol_ratio(v, 20)
    ema20 = calc_ema(c, 20)
    ema50 = calc_ema(c, 50)
    
    # Factor 1: ATR expanding from compression
    atr_compressed = atr_pctl < 0.30
    atr_compressed_count = atr_compressed.rolling(15).sum()
    was_atr_compressed = atr_compressed_count >= 12
    atr_expanding = atr_pctl > 0.40
    
    # Factor 2: Volume returning from contraction
    vol_contracted = vol_ratio < 0.8
    vol_contracted_count = vol_contracted.rolling(10).sum()
    was_vol_contracted = vol_contracted_count >= 7
    vol_returning = vol_ratio > 1.2
    
    # Factor 3: Trend direction change (EMA crossover)
    ema_cross_up = (ema20 > ema50) & (ema20.shift(1) <= ema50.shift(1))
    ema_cross_down = (ema20 < ema50) & (ema20.shift(1) >= ema50.shift(1))
    
    # Multi-factor: all three aligned
    regime_switch_long = was_atr_compressed & was_vol_contracted & atr_expanding & vol_returning & ema_cross_up
    regime_switch_short = was_atr_compressed & was_vol_contracted & atr_expanding & vol_returning & ema_cross_down
    
    events = regime_switch_long | regime_switch_short
    direction = pd.Series("NONE", index=df.index)
    direction[regime_switch_long] = "LONG"
    direction[regime_switch_short] = "SHORT"
    return events, direction


# === ISOLATION GATE ===
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
    CSV = "/root/.openclaw/workspace/jimi_audit/data/eth_15m_merged_extended.csv"
    df = pd.read_csv(CSV)
    print(f"Loaded: {len(df)} bars ({df['Open time'].iloc[0]} -> {df['Open time'].iloc[-1]})")
    print(f"Testing 5 strategies v3 (multi-factor) on real ETH 15m data\n")
    
    strategies = {
        "funding_arb_v3": detect_funding_arb_v3,
        "squeeze_breakout_v3": detect_squeeze_breakout_v3,
        "judas_sweep_v3": detect_judas_sweep_v3,
        "structural_break_v3": detect_structural_break_v3,
        "regime_switch_v3": detect_regime_switch_v3,
    }
    
    all_results = {}
    for name, fn in strategies.items():
        print(f"{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        events, direction = fn(df)
        result = run_gate(df, events, direction, name)
        all_results[name] = result
        for h_str, r in result.get("horizons", {}).items():
            if r.get("mean_pct") is not None:
                print(f"  {h_str}: n={r['n']:>5} | mean={r['mean_pct']:>+8.4f}% | p={r['p']:.4f}")
        status = "PASS" if result["gate_passed"] else "FAIL"
        print(f"\n  RESULT: {status} | events={result['events']} | best={result.get('best_h','?')} | mean={result.get('best_mean_pct',0):+.4f}% | p={result.get('best_p',1):.4f}")
        print(f"  {result['reason']}\n")
    
    print(f"{'='*60}")
    print(f"  V3 MULTI-FACTOR SUMMARY")
    print(f"{'='*60}")
    for name, r in all_results.items():
        status = "PASS" if r["gate_passed"] else "FAIL"
        print(f"  [{status:4s}] {name:30s} | n={r['events']:>5} | mean={r.get('best_mean_pct',0):>+.4f}% | p={r.get('best_p',1):.4f}")
    
    out_dir = "/root/.openclaw/workspace/jimi_audit/reports/v3_gates"
    os.makedirs(out_dir, exist_ok=True)
    for name, r in all_results.items():
        with open(f"{out_dir}/{name}_gate.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir}/")

if __name__ == "__main__":
    main()
