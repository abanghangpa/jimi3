#!/usr/bin/env python3
"""
Fixable Strategies v2 — Improved Detection Logic
Tests 5 strategies with fixed detection on real ETH 15m data.

Changes from v1:
1. funding_arb: lower threshold, use FR change not level
2. squeeze_breakout: ATR compression instead of BB/KC
3. judas_sweep: structural levels (round numbers, daily H/L)
4. structural_break: higher timeframe levels (4h H/L)
5. regime_switch: vol compression breakout (start of expansion)
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


# === V2 DETECTION FUNCTIONS ===

def detect_funding_arb_v2(df):
    """v2: Use funding rate CHANGE (not level), lower threshold.
    
    Mechanism: When FR changes sharply (diverges from recent mean), market is
    repricing risk. The arb is: if FR spikes up, longs are paying shorts,
    incentivizing shorts to open -> price likely to dip temporarily.
    
    Key fix: Use 4-bar FR change instead of z-score of level.
    """
    if "funding_rate" not in df.columns:
        # Compute proxy from taker ratio
        tr = df["Taker buy base asset volume"] / df["Volume"].replace(0, 1)
        fr_proxy = tr.rolling(48).mean() - 0.5  # deviation from neutral
        fr = fr_proxy
    else:
        fr = df["funding_rate"]
    
    # 4-bar FR change (1 hour of 15m bars)
    fr_change = fr.diff(4)
    fr_change_ma = fr_change.rolling(20).mean()
    fr_change_std = fr_change.rolling(20).std()
    fr_zscore = (fr_change - fr_change_ma) / fr_change_std.replace(0, 1)
    
    vol_ratio = calc_vol_ratio(df["Volume"], 20)
    
    # v1 was z-score > 2.0 (too extreme). v2 uses 1.5.
    # Also: FR change positive = longs paying -> SHORT (mean reversion)
    # FR change negative = shorts paying -> LONG
    long_events = (fr_zscore < -1.5) & (vol_ratio > 1.0)
    short_events = (fr_zscore > 1.5) & (vol_ratio > 1.0)
    
    events = long_events | short_events
    direction = pd.Series("NONE", index=df.index)
    direction[long_events] = "LONG"
    direction[short_events] = "SHORT"
    return events, direction


def detect_squeeze_breakout_v2(df):
    """v2: ATR compression -> expansion instead of BB/KC squeeze.
    
    Mechanism: When volatility compresses (ATR < 30th percentile for 20+ bars),
    energy accumulates. Release (ATR expanding) predicts directional move.
    
    Key fix: ATR compression is more common than BB/KC squeeze.
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    
    atr = calc_atr(h, l, c, 14)
    
    # ATR percentile over rolling 200-bar window
    atr_pctl = atr.rolling(200, min_periods=50).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)
    
    # Compression: ATR < 30th percentile for 20+ bars
    compressed = atr_pctl < 0.30
    compressed_count = compressed.rolling(20).sum()
    was_compressed = compressed_count >= 15  # at least 15 of last 20 bars compressed
    
    # Expansion: ATR crosses above 40th percentile
    expanding = atr_pctl > 0.40
    
    # Signal: was compressed, now expanding
    breakout = was_compressed & expanding & ~expanding.shift(1).fillna(False)
    
    # Direction from price momentum
    ema20 = calc_ema(c, 20)
    direction = pd.Series("NONE", index=df.index)
    direction[breakout & (c > ema20)] = "LONG"
    direction[breakout & (c < ema20)] = "SHORT"
    
    events = breakout & (direction != "NONE")
    return events, direction


def detect_judas_sweep_v2(df):
    """v2: Structural levels instead of rolling fractals.
    
    Mechanism: Price sweeps a REAL level (round number, daily H/L, session H/L)
    then reverses. Real levels have real stops clustered there.
    
    Key fix: Use structural levels (round numbers + daily pivots) instead of
    10-bar rolling high/low which catches noise.
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    
    # Structural levels:
    # 1. Round numbers (every $50 for ETH ~$1800)
    price = c.iloc[-1]
    round_step = 50  # $50 levels
    round_levels = np.arange(
        (price // round_step - 20) * round_step,
        (price // round_step + 20) * round_step,
        round_step
    )
    
    # 2. Previous day high/low (rolling 96 bars = 24h of 15m)
    daily_high = h.rolling(96).max().shift(1)
    daily_low = l.rolling(96).min().shift(1)
    
    # 3. Session high/low (rolling 32 bars = 8h)
    session_high = h.rolling(32).max().shift(1)
    session_low = l.rolling(32).min().shift(1)
    
    # Combine all levels into proximity check
    # For each bar, check if price swept a level and reversed
    
    vol_ratio = calc_vol_ratio(df["Volume"], 20)
    
    # Sweep detection with structural levels
    sweep_high_events = pd.Series(False, index=df.index)
    sweep_low_events = pd.Series(False, index=df.index)
    
    for i in range(100, len(df)):
        price_now = c.iloc[i]
        high_now = h.iloc[i]
        low_now = l.iloc[i]
        
        # Check daily high sweep
        dh = daily_high.iloc[i] if not pd.isna(daily_high.iloc[i]) else 0
        dl = daily_low.iloc[i] if not pd.isna(daily_low.iloc[i]) else 0
        sh = session_high.iloc[i] if not pd.isna(session_high.iloc[i]) else 0
        sl = session_low.iloc[i] if not pd.isna(session_low.iloc[i]) else 0
        
        # Sweep high: wick above level, close below
        for level in [dh, sh]:
            if level > 0 and high_now > level * 1.001 and price_now < level:
                if (high_now - price_now) > (price_now - low_now) * 1.2:  # rejection wick
                    sweep_high_events.iloc[i] = True
                    break
        
        # Sweep low: wick below level, close above
        for level in [dl, sl]:
            if level > 0 and low_now < level * 0.999 and price_now > level:
                if (price_now - low_now) > (high_now - price_now) * 1.2:  # rejection wick
                    sweep_low_events.iloc[i] = True
                    break
        
        # Also check round number sweeps
        for rlevel in round_levels:
            if abs(high_now - rlevel) / rlevel < 0.002 and price_now < rlevel:
                if (high_now - price_now) > (price_now - low_now) * 1.2:
                    sweep_high_events.iloc[i] = True
                    break
            if abs(low_now - rlevel) / rlevel < 0.002 and price_now > rlevel:
                if (price_now - low_now) > (high_now - price_now) * 1.2:
                    sweep_low_events.iloc[i] = True
                    break
    
    events = sweep_high_events | sweep_low_events
    direction = pd.Series("NONE", index=df.index)
    direction[sweep_high_events] = "SHORT"
    direction[sweep_low_events] = "LONG"
    return events, direction


def detect_structural_break_v2(df):
    """v2: Higher timeframe levels + close confirmation.
    
    Mechanism: Breaking a 4h level with volume = genuine breakout.
    15m rolling levels are noise. 4h levels are institutional.
    
    Key fix: Use 16-bar rolling (4h equivalent) instead of 50-bar (12.5h).
    Also require close above/below, not just wick.
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    
    # 4h levels (16 bars of 15m)
    resistance_4h = h.rolling(16).max().shift(1)
    support_4h = l.rolling(16).min().shift(1)
    
    # Daily levels (96 bars)
    resistance_daily = h.rolling(96).max().shift(1)
    support_daily = l.rolling(96).min().shift(1)
    
    vol_ratio = calc_vol_ratio(df["Volume"], 20)
    
    # Break above 4h resistance with volume + close confirmation
    break_up_4h = (c > resistance_4h) & (c.shift(1) <= resistance_4h) & (vol_ratio > 1.2)
    # Break below 4h support with volume + close confirmation
    break_down_4h = (c < support_4h) & (c.shift(1) >= support_4h) & (vol_ratio > 1.2)
    
    # Break above daily resistance (higher conviction)
    break_up_daily = (c > resistance_daily) & (c.shift(1) <= resistance_daily) & (vol_ratio > 1.3)
    break_down_daily = (c < support_daily) & (c.shift(1) >= support_daily) & (vol_ratio > 1.3)
    
    # Combine: daily breaks are higher conviction
    break_up = break_up_4h | break_up_daily
    break_down = break_down_4h | break_down_daily
    
    events = break_up | break_down
    direction = pd.Series("NONE", index=df.index)
    direction[break_up] = "LONG"
    direction[break_down] = "SHORT"
    return events, direction


def detect_regime_switch_v2(df):
    """v2: Volatility compression breakout (catch START of expansion).
    
    Mechanism: When ATR is compressed (< 30th percentile) for 20+ bars,
    the first expansion bar predicts a directional move.
    
    Key fix: v1 used ATR > 80th percentile (caught END of expansion).
    v2 catches the START (ATR crossing from compressed to normal).
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    
    atr = calc_atr(h, l, c, 14)
    atr_pctl = atr.rolling(200, min_periods=50).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)
    
    # Compression: ATR < 30th percentile
    compressed = atr_pctl < 0.30
    compressed_count = compressed.rolling(20).sum()
    was_compressed = compressed_count >= 15
    
    # Expansion start: ATR crosses above 35th percentile (from compressed)
    expanding_start = (atr_pctl > 0.35) & (atr_pctl.shift(1) <= 0.35)
    
    # Signal: was in compression, now starting to expand
    regime_switch = was_compressed & expanding_start
    
    # Direction from recent price action
    returns_4 = c.pct_change(4)
    ema20 = calc_ema(c, 20)
    
    direction = pd.Series("NONE", index=df.index)
    direction[regime_switch & (c > ema20)] = "LONG"
    direction[regime_switch & (c < ema20)] = "SHORT"
    
    events = regime_switch & (direction != "NONE")
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
            if i + h < n:
                fr.append((close[i + h] - close[i]) / close[i])
        if len(fr) < 3:
            results[f"{h}bar"] = {"mean_pct": None, "n": len(fr), "p": None}
            continue
        fr = np.array(fr)
        mean_r = np.mean(fr)
        ne_mask = np.ones(n, dtype=bool)
        ne_mask[idx] = False
        ne_idx = np.where(ne_mask)[0]
        ne = np.array([(close[i + h] - close[i]) / close[i] for i in ne_idx if i + h < n])
        if len(fr) > 1 and len(ne) > 1:
            t, p = stats.ttest_ind(fr, ne, equal_var=False)
        else:
            t, p = 0.0, 1.0
        results[f"{h}bar"] = {"mean_pct": round(mean_r * 100, 4), "n": len(fr), "p": round(float(p), 4)}
    
    best_h, best_p, best_m = None, 1.0, 0.0
    for hs, r in results.items():
        if r.get("mean_pct") is None:
            continue
        if r["p"] < best_p:
            best_p = r["p"]
            best_h = hs
            best_m = r["mean_pct"]
    
    dir_ok = best_m > 0
    p_ok = best_p < 0.1
    eff_ok = abs(best_m) > cost * 100
    passed = dir_ok and p_ok and eff_ok
    
    reasons = []
    if not dir_ok:
        reasons.append(f"dir backwards ({best_m:+.4f}%)")
    if not p_ok:
        reasons.append(f"p={best_p:.4f}")
    if not eff_ok:
        reasons.append(f"effect {abs(best_m):.4f}% < costs")
    
    return {
        "events": int(len(idx)),
        "horizons": results,
        "best_h": best_h,
        "best_p": best_p,
        "best_mean_pct": best_m,
        "dir_correct": dir_ok,
        "gate_passed": passed,
        "reason": "; ".join(reasons) if reasons else "PASS"
    }


# === MAIN ===
def main():
    CSV = "/root/.openclaw/workspace/jimi_audit/data/eth_15m_merged.csv"
    df = pd.read_csv(CSV)
    print(f"Loaded: {len(df)} bars ({df['Open time'].iloc[0]} -> {df['Open time'].iloc[-1]})")
    print(f"Testing 5 fixable strategies v2 on real ETH 15m data\n")
    
    strategies = {
        "funding_arb_v2": detect_funding_arb_v2,
        "squeeze_breakout_v2": detect_squeeze_breakout_v2,
        "judas_sweep_v2": detect_judas_sweep_v2,
        "structural_break_v2": detect_structural_break_v2,
        "regime_switch_v2": detect_regime_switch_v2,
    }
    
    all_results = {}
    
    for name, fn in strategies.items():
        print(f"{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        
        events, direction = fn(df)
        result = run_gate(df, events, direction, name)
        all_results[name] = result
        
        for h_str, r in result.get("horizons", {}).items():
            if r.get("mean_pct") is not None:
                print(f"  {h_str}: n={r['n']:>5} | mean={r['mean_pct']:>+8.4f}% | p={r['p']:.4f}")
        
        status = "PASS" if result["gate_passed"] else "FAIL"
        print(f"\n  RESULT: {status} | events={result['events']} | best={result.get('best_h', '?')} | mean={result.get('best_mean_pct', 0):+.4f}% | p={result.get('best_p', 1):.4f}")
        print(f"  {result['reason']}")
        print()
    
    # Summary
    print(f"{'=' * 60}")
    print(f"  V2 STRATEGIES — REAL DATA SUMMARY")
    print(f"{'=' * 60}")
    for name, r in all_results.items():
        status = "PASS" if r["gate_passed"] else "FAIL"
        print(f"  [{status:4s}] {name:30s} | n={r['events']:>5} | mean={r.get('best_mean_pct', 0):>+.4f}% | p={r.get('best_p', 1):.4f} | {r['reason'][:35]}")
    
    # Save
    out_dir = "/root/.openclaw/workspace/jimi_audit/reports/v2_gates"
    os.makedirs(out_dir, exist_ok=True)
    for name, r in all_results.items():
        with open(f"{out_dir}/{name}_gate.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir}/")


if __name__ == "__main__":
    main()
