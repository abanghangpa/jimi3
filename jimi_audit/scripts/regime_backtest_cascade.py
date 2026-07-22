#!/usr/bin/env python3
"""
Regime-Specific Backtest for Liquidation Cascade (s20)
Tests the strategy across BULL, BEAR, RANGING, STRESS regimes.

v2: Fixed CSV parsing (no headers in data files).
"""

import json, os, sys, csv, math, statistics
from datetime import datetime, timezone, timedelta
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

OI_CSV = os.path.join(BASE, "data", "forced_movement", "oi_history.csv")
FUNDING_CSV = os.path.join(BASE, "data", "forced_movement", "funding_history.csv")

# ============================================================
# DATA LOADING (headerless CSVs)
# ============================================================
def load_oi_data():
    """Load OI history. Format: ts,oi,volume (no header)."""
    if not os.path.exists(OI_CSV):
        print(f"  ⚠️ OI data not found: {OI_CSV}")
        return []
    rows = []
    with open(OI_CSV) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    rows.append({
                        "ts": int(parts[0]),
                        "oi": float(parts[1]),
                        "vol": float(parts[2]),
                    })
                except (ValueError, IndexError):
                    continue
    return sorted(rows, key=lambda x: x["ts"])


def load_funding_data():
    """Load funding history. Format: exchange,ts,rate,collected_at (no header)."""
    if not os.path.exists(FUNDING_CSV):
        return {}
    by_ts = {}
    with open(FUNDING_CSV) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    ts = int(parts[1])
                    rate = float(parts[2])
                    by_ts[ts] = rate
                except (ValueError, IndexError):
                    continue
    return by_ts


def load_candles(symbol="ETHUSDT", interval="15m", limit=1500):
    """Load candles from Binance."""
    import requests
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        candles = []
        for c in data:
            candles.append({
                "ts": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })
        return candles
    except Exception as e:
        print(f"  ⚠️ Candle fetch failed: {e}")
        return []


# ============================================================
# REGIME CLASSIFIER
# ============================================================
def classify_regime(deriv_window):
    if len(deriv_window) < 3:
        return "RANGING", 0.5

    latest = deriv_window[-1]
    fr = latest.get("fr", 0)
    ls = latest.get("ls", 2.0)
    oi = latest.get("oi", 0)

    if len(deriv_window) >= 2:
        prev_oi = deriv_window[-2].get("oi", oi)
        oi_roc = ((oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
    else:
        oi_roc = 0

    avg_fr = sum(d.get("fr", 0) for d in deriv_window) / len(deriv_window)
    ls_values = [d.get("ls", 2.0) for d in deriv_window]
    ls_trend = ls_values[-1] - ls_values[0] if len(ls_values) > 1 else 0

    fr_values = [d.get("fr", 0) for d in deriv_window]
    fr_std = (sum((f - avg_fr)**2 for f in fr_values) / max(len(fr_values)-1, 1)) ** 0.5

    bull_score = 0.0
    bear_score = 0.0
    stress_score = 0.0

    if fr > 0.000030:
        bull_score += 1
    elif fr < -0.000010:
        bear_score += 1

    if ls > 2.2:
        bear_score += 0.8
    elif ls < 1.8:
        bull_score += 0.8

    if oi_roc < -3:
        stress_score += 2
    elif oi_roc > 5:
        bull_score += 0.5

    if avg_fr > 0.000015:
        bull_score += 0.5
    elif avg_fr < -0.000005:
        bear_score += 0.5

    if ls_trend > 0.1:
        bull_score += 0.2
    elif ls_trend < -0.1:
        bear_score += 0.2

    if fr_std > 0.00003:
        stress_score += 1

    if stress_score > 2:
        regime = "STRESS"
        confidence = min(0.9, 0.5 + stress_score * 0.1)
    elif bull_score > bear_score + 0.5:
        regime = "BULL"
        confidence = min(0.9, 0.5 + (bull_score - bear_score) * 0.15)
    elif bear_score > bull_score + 0.5:
        regime = "BEAR"
        confidence = min(0.9, 0.5 + (bear_score - bull_score) * 0.15)
    elif bear_score > bull_score and bear_score >= 1.0:
        regime = "MILDLY_BEARISH"
        confidence = min(0.8, 0.5 + (bear_score - bull_score) * 0.1)
    else:
        regime = "RANGING"
        confidence = 0.5

    return regime, confidence


# ============================================================
# CASCADE SIGNAL DETECTION
# ============================================================
def check_cascade_signal(candles, oi_window, funding_by_ts, idx):
    """
    Generate cascade signal using OI data + funding + price action.
    Sources: OI shock, OI estimate, funding divergence.
    """
    if idx < 10:
        return None

    price = candles[idx]["close"]

    # ATR (14 bars)
    atr_vals = []
    for j in range(max(1, idx-14), idx):
        h = candles[j]["high"]
        l = candles[j]["low"]
        pc = candles[j-1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        atr_vals.append(tr)
    atr = sum(atr_vals) / len(atr_vals) if atr_vals else 0
    if not atr or atr == 0:
        return None

    signals = []

    # === SOURCE 1: OI shock (OI ROC > 0.5% + price move) ===
    if len(oi_window) >= 3:
        cur_oi = oi_window[-1]["oi"]
        # Compare to 5 bars ago (~75min)
        ref_idx = max(0, len(oi_window) - 6)
        ref_oi = oi_window[ref_idx]["oi"]
        if ref_oi > 0:
            oi_roc = (cur_oi - ref_oi) / ref_oi
        else:
            oi_roc = 0

        # Price momentum
        price_change = (candles[idx]["close"] - candles[idx-5]["close"]) / candles[idx-5]["close"]

        # OI dropping + price dropping = long cascade
        if oi_roc < -0.005 and price_change < -0.003:
            signals.append({
                "direction": "SHORT",
                "strength": min(abs(oi_roc) * 15, 0.8),
                "source": "oi_shock"
            })

        # OI dropping + price rising = short cascade
        if oi_roc < -0.005 and price_change > 0.003:
            signals.append({
                "direction": "LONG",
                "strength": min(abs(oi_roc) * 15, 0.8),
                "source": "oi_shock"
            })

        # OI surging + extreme move = continuation
        if oi_roc > 0.01 and abs(price_change) > 0.005:
            direction = "LONG" if price_change > 0 else "SHORT"
            signals.append({
                "direction": direction,
                "strength": min(abs(oi_roc) * 10, 0.7),
                "source": "oi_surge"
            })

    # === SOURCE 2: OI estimate (OI ROC > 0.3% + volume spike) ===
    if not signals and len(oi_window) >= 3:
        cur_oi = oi_window[-1]["oi"]
        ref_idx = max(0, len(oi_window) - 6)
        ref_oi = oi_window[ref_idx]["oi"]
        if ref_oi > 0:
            oi_roc = (cur_oi - ref_oi) / ref_oi
        else:
            oi_roc = 0

        # Volume spike
        avg_vol = sum(c["volume"] for c in candles[max(0,idx-20):idx]) / min(20, idx)
        cur_vol = candles[idx]["volume"]
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1

        if abs(oi_roc) > 0.003 and vol_ratio > 1.5:
            # High volume + OI change = directional signal
            price_change = (candles[idx]["close"] - candles[idx-3]["close"]) / candles[idx-3]["close"]
            if oi_roc < -0.003:
                direction = "SHORT" if price_change < 0 else "LONG"
            else:
                direction = "LONG" if price_change > 0 else "SHORT"

            signals.append({
                "direction": direction,
                "strength": min(abs(oi_roc) * 10 + (vol_ratio - 1) * 0.2, 0.7),
                "source": "oi_estimate"
            })

    # === SOURCE 3: Funding divergence ===
    if not signals:
        # Find nearest funding rate
        candle_ts = candles[idx]["ts"]
        nearest_fr = None
        min_diff = float('inf')
        for fts, rate in funding_by_ts.items():
            diff = abs(fts - candle_ts)
            if diff < min_diff:
                min_diff = diff
                nearest_fr = rate

        if nearest_fr is not None and abs(nearest_fr) > 0.0001:
            # Extreme funding + price reversal potential
            price_change = (candles[idx]["close"] - candles[idx-10]["close"]) / candles[idx-10]["close"]

            if nearest_fr > 0.0001 and price_change < -0.005:
                # Longs paying + price dropping = long squeeze potential
                signals.append({
                    "direction": "SHORT",
                    "strength": min(abs(nearest_fr) * 5000, 0.6),
                    "source": "funding_div"
                })
            elif nearest_fr < -0.0001 and price_change > 0.005:
                # Shorts paying + price rising = short squeeze potential
                signals.append({
                    "direction": "LONG",
                    "strength": min(abs(nearest_fr) * 5000, 0.6),
                    "source": "funding_div"
                })

    if not signals:
        return None

    best = max(signals, key=lambda x: x["strength"])
    direction = best["direction"]
    strength = best["strength"]

    if direction == "LONG":
        sl = price - atr * 1.5
        tp1 = price + atr * 1.5
    else:
        sl = price + atr * 1.5
        tp1 = price - atr * 1.5

    return {
        "direction": direction,
        "entry": price,
        "sl": sl,
        "tp1": tp1,
        "sl_pct": abs(price - sl) / price * 100,
        "tp_pct": abs(tp1 - price) / price * 100,
        "strength": strength,
        "source": best["source"],
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(candles, oi_data, funding_by_ts, regime_filter=None):
    WINDOW = 20
    HOLD_BARS = 8
    trades = []

    for i in range(WINDOW + 10, len(candles) - HOLD_BARS):
        candle_ts = candles[i]["ts"]

        # Get OI window (within 2h of candle)
        oi_window = [d for d in oi_data if abs(d["ts"] - candle_ts) < 7200000]
        if len(oi_window) < 3:
            continue

        # Classify regime using OI + funding
        deriv_window = []
        for d in oi_window[-WINDOW:]:
            nearest_fr = 0
            min_diff = float('inf')
            for fts, rate in funding_by_ts.items():
                diff = abs(fts - d["ts"])
                if diff < min_diff:
                    min_diff = diff
                    nearest_fr = rate
            deriv_window.append({
                "oi": d["oi"],
                "fr": nearest_fr,
                "ls": 2.0,  # neutral (no LS data in this dataset)
            })

        regime, confidence = classify_regime(deriv_window)

        if regime_filter and regime != regime_filter:
            continue

        signal = check_cascade_signal(candles, oi_window, funding_by_ts, i)
        if not signal:
            continue

        entry = signal["entry"]
        sl = signal["sl"]
        tp1 = signal["tp1"]
        direction = signal["direction"]

        outcome = "TIMEOUT"
        exit_price = entry
        bars_held = 0

        for j in range(1, HOLD_BARS + 1):
            bar = candles[i + j]
            bars_held = j

            if direction == "LONG":
                if bar["low"] <= sl:
                    outcome = "LOSS"
                    exit_price = sl
                    break
                elif bar["high"] >= tp1:
                    outcome = "WIN"
                    exit_price = tp1
                    break
            else:
                if bar["high"] >= sl:
                    outcome = "LOSS"
                    exit_price = sl
                    break
                elif bar["low"] <= tp1:
                    outcome = "WIN"
                    exit_price = tp1
                    break

        if direction == "LONG":
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100
        pnl_pct -= 0.04  # fees

        trades.append({
            "entry_idx": i,
            "timestamp": candle_ts,
            "direction": direction,
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp1": round(tp1, 2),
            "exit": round(exit_price, 2),
            "outcome": outcome,
            "pnl_pct": round(pnl_pct, 4),
            "regime": regime,
            "regime_confidence": round(confidence, 3),
            "source": signal["source"],
            "bars_held": bars_held,
        })

    return trades


# ============================================================
# STATISTICAL TEST
# ============================================================
def calc_p_value(trades):
    if len(trades) < 5:
        return 1.0
    returns = [t["pnl_pct"] for t in trades]
    n = len(returns)
    mean = statistics.mean(returns)
    if n < 2:
        return 1.0
    std = statistics.stdev(returns)
    if std == 0:
        return 0.0 if mean > 0 else 1.0
    t_stat = mean / (std / math.sqrt(n))
    from math import erf, sqrt
    p = 0.5 * (1 - erf(abs(t_stat) / sqrt(2)))
    return p


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("LIQUIDATION CASCADE — REGIME-SPECIFIC BACKTEST v2")
    print("=" * 60)

    print("\n📊 Loading data...")
    candles = load_candles(limit=1500)
    print(f"  Candles: {len(candles)} (15m)")

    oi_data = load_oi_data()
    print(f"  OI data: {len(oi_data)} points")

    funding_by_ts = load_funding_data()
    print(f"  Funding: {len(funding_by_ts)} points")

    if not candles:
        print("❌ No candle data. Exiting.")
        return

    if len(oi_data) < 10:
        print("❌ Insufficient OI data. Exiting.")
        return

    # OI ROC stats for debugging
    if len(oi_data) >= 2:
        rocs = [(oi_data[i]["oi"] - oi_data[i-1]["oi"]) / oi_data[i-1]["oi"]
                for i in range(1, len(oi_data)) if oi_data[i-1]["oi"] > 0]
        big_rocs = [r for r in rocs if abs(r) > 0.005]
        print(f"  OI ROC > 0.5%: {len(big_rocs)} events")

    # Run backtests
    print("\n🔄 Running backtest (all regimes)...")
    all_trades = run_backtest(candles, oi_data, funding_by_ts)
    print(f"  Total signals: {len(all_trades)}")

    if all_trades:
        sources = {}
        for t in all_trades:
            s = t["source"]
            sources[s] = sources.get(s, 0) + 1
        print(f"  Signal sources: {sources}")

    if not all_trades:
        print("\n❌ No signals generated.")
        print("   Possible reasons:")
        print("   - OI ROC thresholds too strict")
        print("   - Not enough OI variation in data")
        print("   - Price/OI correlation too weak")
        return

    # Per-regime backtests
    regimes = ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH"]
    results = {}

    for regime in regimes:
        trades = run_backtest(candles, oi_data, funding_by_ts, regime_filter=regime)
        if not trades:
            results[regime] = {"events": 0, "passed": False, "reason": "No signals"}
            continue

        wins = sum(1 for t in trades if t["outcome"] == "WIN")
        losses = sum(1 for t in trades if t["outcome"] == "LOSS")
        timeouts = sum(1 for t in trades if t["outcome"] == "TIMEOUT")
        total = len(trades)
        returns = [t["pnl_pct"] for t in trades]
        mean_return = statistics.mean(returns)
        wr = wins / total if total > 0 else 0
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        p_value = calc_p_value(trades)
        passed = p_value < 0.05 and mean_return > 0

        results[regime] = {
            "events": total, "wins": wins, "losses": losses, "timeouts": timeouts,
            "wr": round(wr, 4), "pf": round(pf, 3),
            "mean_return_pct": round(mean_return, 4),
            "p_value": round(p_value, 4), "passed": passed,
        }

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS BY REGIME")
    print("=" * 60)

    gate_update = {}

    for regime, r in results.items():
        status = "✅ PASS" if r.get("passed") else "❌ FAIL"
        print(f"\n{'─' * 40}")
        print(f"  Regime: {regime}  {status}")
        if r["events"] == 0:
            print(f"  No signals in this regime")
            continue
        print(f"  Events:        {r['events']}")
        print(f"  W/L/T:         {r['wins']}/{r['losses']}/{r['timeouts']}")
        print(f"  Win Rate:      {r['wr']:.1%}")
        print(f"  Profit Factor: {r['pf']:.2f}")
        print(f"  Mean Return:   {r['mean_return_pct']:.4f}%")
        print(f"  p-value:       {r['p_value']:.4f}")

        if r.get("passed"):
            gate_update[regime.lower()] = {
                "passed": True,
                "p_value": r["p_value"],
                "effect_direction": "correct",
                "mean_return_pct": r["mean_return_pct"],
                "events": r["events"],
                "wr": r["wr"],
                "pf": r["pf"],
            }

    # Gate update
    print("\n" + "=" * 60)
    print("GATE UPDATE RECOMMENDATION")
    print("=" * 60)

    if gate_update:
        print("\n✅ Passing regimes:")
        for regime, data in gate_update.items():
            print(f"  {regime}: p={data['p_value']:.4f}, {data['events']} events, WR={data['wr']:.1%}")

        regime_breakdown = {}
        for regime in regimes:
            r = results.get(regime, {})
            if r.get("events", 0) == 0:
                continue
            regime_breakdown[regime.lower()] = {
                "passed": r.get("passed", False),
                "p_value": r.get("p_value", 1.0),
                "mean_return_pct": r.get("mean_return_pct", 0),
                "events": r.get("events", 0),
                "wr": r.get("wr", 0),
                "pf": r.get("pf", 0),
            }

        all_returns = [t["pnl_pct"] for t in all_trades]
        pooled_p = calc_p_value(all_trades)
        pooled_mean = statistics.mean(all_returns)
        all_wr = sum(1 for t in all_trades if t["outcome"] == "WIN") / len(all_trades)
        gross_profit = sum(r for r in all_returns if r > 0)
        gross_loss = abs(sum(r for r in all_returns if r < 0))
        pooled_pf = gross_profit / gross_loss if gross_loss > 0 else 0

        update_entry = {
            "passed": any(r.get("passed") for r in results.values()),
            "p_value": round(pooled_p, 4),
            "effect_direction": "correct",
            "mean_return_pct": round(pooled_mean, 4),
            "events": len(all_trades),
            "wr": round(all_wr, 4),
            "pf": round(pooled_pf, 3),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "notes": f"Regime-specific backtest v2. Passing: {list(gate_update.keys())}",
            "regime_breakdown": regime_breakdown,
        }

        print(f"\n📝 Gate file update:")
        print(json.dumps({"liquidation_cascade": update_entry}, indent=2))

        update_path = os.path.join(BASE, "config", "liquidation_cascade_regime_update.json")
        with open(update_path, "w") as f:
            json.dump({"liquidation_cascade": update_entry}, f, indent=2)
        print(f"\n💾 Saved to: {update_path}")
    else:
        print("\n❌ No regime passes p < 0.05.")
        best = max(results.items(), key=lambda x: x[1].get("p_value", 0) * -1 if x[1].get("events", 0) > 0 else -999)
        if best[1].get("events", 0) > 0:
            r = best[1]
            print(f"   Best: {best[0]} — p={r['p_value']:.4f}, {r['events']} events, WR={r['wr']:.1%}")

    # Save full results
    results_path = os.path.join(BASE, "data", "liquidation_cascade_regime_backtest.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candles": len(candles),
            "oi_points": len(oi_data),
            "funding_points": len(funding_by_ts),
            "results": results,
            "all_trades": len(all_trades),
            "per_source": {s: sum(1 for t in all_trades if t["source"] == s) for s in set(t["source"] for t in all_trades)},
        }, f, indent=2)
    print(f"\n📊 Full results: {results_path}")


if __name__ == "__main__":
    main()
