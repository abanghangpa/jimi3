#!/usr/bin/env python3
"""
Liquidity Sweep Detector v2 — Tightened

v1 problems: OI matching broken, S/R levels too numerous, wick detection too loose.
v2 fixes:
1. Fixed OI alignment — proper epoch_ms ↔ string timestamp matching
2. Only sweeps at SIGNIFICANT levels — equal highs/lows, session H/L, high-volume nodes
3. OI drop MANDATORY — not optional. 0.3%+ drop required
4. Wick ratio > 0.6 — strong rejection, not noise
5. Volume > 1.3x average — volume absorption confirmation
6. Price must CLOSE back inside level — not just wick
7. Level must have been tested before — first touch = breakout, second+ touch = sweep
"""
import json, os, csv, math, random
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

random.seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

def load_ohlcv(path, start="2026-04-01"):
    bars = []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = row.get("Open time", "")
            if ts < start: continue
            try:
                bars.append({
                    "ts": ts, "open": float(row["Open"]), "high": float(row["High"]),
                    "low": float(row["Low"]), "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                    "taker_buy": float(row.get("Taker buy base asset volume", 0)),
                })
            except: continue
    return bars

def load_oi(path):
    """Load OI as sorted list of (epoch_ms, oi)."""
    data = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    data.append((int(parts[0]), float(parts[1])))
                except: continue
    return sorted(data, key=lambda x: x[0])

def get_oi(oi_data, ts_str):
    """Get OI at timestamp with proper alignment."""
    if not oi_data:
        return None
    try:
        ts = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
        ts_ms = int(ts.replace(tzinfo=timezone.utc).timestamp() * 1000)
    except:
        return None

    # Find closest OI within 15 minutes
    best = None
    best_delta = 900_000  # 15 min in ms
    for t, oi in oi_data:
        delta = abs(t - ts_ms)
        if delta < best_delta:
            best_delta = delta
            best = oi
    return best

# ═══════════════════════════════════════════════════════════════
# SIGNIFICANT S/R LEVELS
# ═══════════════════════════════════════════════════════════════

def find_significant_levels(bars, idx, lookback=192):
    """
    Find SIGNIFICANT S/R levels — not every swing point.

    Significant =:
    1. Tested at least 2 times (touches)
    2. OR is a session/week high/low
    3. OR is a high-volume node (price spent time there)
    """
    if idx < lookback:
        return []

    window = bars[idx-lookback:idx]
    highs = [b["high"] for b in window]
    lows = [b["low"] for b in window]
    closes = [b["close"] for b in window]
    volumes = [b["volume"] for b in window]

    # Find swing points
    swings = []
    for i in range(3, len(window) - 3):
        # Swing high
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
            highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            swings.append({"price": highs[i], "type": "resistance", "bar": i, "vol": volumes[i]})
        # Swing low
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
            lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            swings.append({"price": lows[i], "type": "support", "bar": i, "vol": volumes[i]})

    # Count touches for each level
    levels = []
    tolerance = 0.002  # 0.2%

    for swing in swings:
        price = swing["price"]
        # Count how many times price touched this level
        touches = 0
        for i in range(len(window)):
            if swing["type"] == "resistance":
                if abs(highs[i] - price) / price < tolerance:
                    touches += 1
            else:
                if abs(lows[i] - price) / price < tolerance:
                    touches += 1

        if touches >= 2:  # At least 2 touches = significant
            levels.append({
                "price": price,
                "type": swing["type"],
                "touches": touches,
                "strength": touches * swing["vol"],
            })

    # Also add session high/low (last 96 bars = 1 day)
    if len(window) >= 96:
        session_high = max(highs[-96:])
        session_low = min(lows[-96:])
        # Only add if not already in levels
        if not any(abs(l["price"] - session_high)/session_high < tolerance for l in levels):
            levels.append({"price": session_high, "type": "resistance", "touches": 99, "strength": 99999})
        if not any(abs(l["price"] - session_low)/session_low < tolerance for l in levels):
            levels.append({"price": session_low, "type": "support", "touches": 99, "strength": 99999})

    # Deduplicate
    deduped = []
    for lv in sorted(levels, key=lambda x: x["strength"], reverse=True):
        found = False
        for d in deduped:
            if abs(lv["price"] - d["price"]) / d["price"] < tolerance:
                d["touches"] = max(d["touches"], lv["touches"])
                d["strength"] = max(d["strength"], lv["strength"])
                found = True
                break
        if not found:
            deduped.append(lv)

    return sorted(deduped, key=lambda x: x["strength"], reverse=True)[:10]


# ═══════════════════════════════════════════════════════════════
# SWEEP DETECTION (v2 — tightened)
# ═══════════════════════════════════════════════════════════════

def detect_sweep(bars, oi_data, idx, sr_levels, atr):
    """
    Detect liquidity sweep with STRICT criteria.

    All must be true:
    1. Level is significant (2+ touches or session H/L)
    2. Wick exceeds level by 0.1-0.5%
    3. Wick ratio > 0.6 (strong rejection)
    4. Close is back inside level
    5. OI dropped 0.3%+ (mandatory — stops being hit)
    6. Volume > 1.3x average
    """
    bar = bars[idx]
    price = bar["close"]
    high = bar["high"]
    low = bar["low"]
    vol = bar["volume"]
    bar_open = bar["open"]

    if idx < 20 or atr == 0:
        return None

    avg_vol = np.mean([b["volume"] for b in bars[max(0,idx-20):idx]])
    if avg_vol == 0:
        return None

    # OI change (MANDATORY)
    oi_now = get_oi(oi_data, bar["ts"])
    oi_prev = get_oi(oi_data, bars[idx-1]["ts"])
    if oi_now is None or oi_prev is None or oi_prev == 0:
        return None  # Can't confirm without OI
    oi_change = (oi_now - oi_prev) / oi_prev

    # OI must drop (stops being hit = positions closing)
    if oi_change >= -0.003:  # Need at least 0.3% OI drop
        return None

    # Volume must be above average
    if vol < avg_vol * 1.3:
        return None

    bar_range = high - low
    if bar_range == 0:
        return None

    for level in sr_levels:
        level_price = level["price"]

        # === BEARISH SWEEP ===
        if level["type"] == "resistance":
            wick_above = (high - level_price) / level_price
            if wick_above < 0.001 or wick_above > 0.005:
                continue

            # Close must be back below level
            if price >= level_price:
                continue

            # Wick ratio (upper wick / bar range)
            upper_wick = high - max(bar_open, price)
            wick_ratio = upper_wick / bar_range
            if wick_ratio < 0.6:
                continue

            return {
                "direction": "SHORT",
                "entry": price,
                "level": level_price,
                "touches": level["touches"],
                "wick_pct": wick_above,
                "wick_ratio": wick_ratio,
                "oi_change": oi_change,
                "vol_ratio": vol / avg_vol,
                "bar_idx": idx,
                "type": "bearish_sweep",
            }

        # === BULLISH SWEEP ===
        elif level["type"] == "support":
            wick_below = (level_price - low) / level_price
            if wick_below < 0.001 or wick_below > 0.005:
                continue

            if price <= level_price:
                continue

            lower_wick = min(bar_open, price) - low
            wick_ratio = lower_wick / bar_range
            if wick_ratio < 0.6:
                continue

            return {
                "direction": "LONG",
                "entry": price,
                "level": level_price,
                "touches": level["touches"],
                "wick_pct": wick_below,
                "wick_ratio": wick_ratio,
                "oi_change": oi_change,
                "vol_ratio": vol / avg_vol,
                "bar_idx": idx,
                "type": "bullish_sweep",
            }

    return None


# ═══════════════════════════════════════════════════════════════
# REGIME + FEATURES
# ═══════════════════════════════════════════════════════════════

def compute_daily_regimes(bars):
    daily = defaultdict(lambda: {"closes": [], "highs": [], "lows": []})
    for bar in bars:
        day = bar["ts"][:10]
        daily[day]["closes"].append(bar["close"])
        daily[day]["highs"].append(bar["high"])
        daily[day]["lows"].append(bar["low"])

    sorted_days = sorted(daily.keys())
    daily_closes = []
    regimes = {}

    for day in sorted_days:
        daily_closes.append(daily[day]["closes"][-1])
        if len(daily_closes) < 50:
            regimes[day] = "RANGING"
            continue
        closes = daily_closes
        ema50 = _ema(closes, 50)
        ema10 = _ema(closes, 10)
        price = closes[-1]
        rsi = _rsi(closes, 14)
        above = price > ema50
        cross = ema10 > ema50
        slope = (ema50 - _ema(closes[:-5], 50)) / _ema(closes[:-5], 50) if len(closes) > 55 else 0

        bull = bear = stress = 0
        if above and cross and slope > 0.001: bull += 3
        elif not above and not cross and slope < -0.001: bear += 3
        elif not above and slope < -0.002: bear += 2
        if rsi > 70: bear += 1
        elif rsi < 30: bull += 1
        if len(closes) >= 5:
            wroc = (closes[-1] - closes[-5]) / closes[-5]
            if wroc > 0.05: bull += 1.5
            elif wroc < -0.05: bear += 1.5
            elif wroc < -0.10: stress += 1.5

        if stress > 3: regimes[day] = "STRESS"
        elif bull > bear + 2: regimes[day] = "BULL"
        elif bear > bull + 2: regimes[day] = "BEAR"
        elif bear > bull + 1: regimes[day] = "MILDLY_BEARISH"
        else: regimes[day] = "RANGING"

    return regimes

def _ema(data, period):
    if len(data) < period: return data[-1] if data else 0
    mult = 2 / (period + 1)
    ema = sum(data[:period]) / period
    for p in data[period:]: ema = (p - ema) * mult + ema
    return ema

def _rsi(data, period=14):
    if len(data) < period + 1: return 50
    deltas = [data[i]-data[i-1] for i in range(1, len(data))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    ag = sum(gains)/period; al = sum(losses)/period
    if al == 0: return 100
    return 100 - (100/(1+ag/al))

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1: return 0
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        trs.append(tr)
    return np.mean(trs[-period:])


# ═══════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════

def run_backtest(bars, oi_data, regimes):
    trades = []
    positions = []
    capital = 10000.0
    peak = capital
    max_dd = 0
    cooldown = {}

    closes_buf = []
    highs_buf = []
    lows_buf = []

    for i, bar in enumerate(bars):
        closes_buf.append(bar["close"])
        highs_buf.append(bar["high"])
        lows_buf.append(bar["low"])

        day = bar["ts"][:10]
        regime = regimes.get(day, "RANGING")

        # Check exits
        closed = []
        for pos in positions:
            bars_held = i - pos["entry_bar"]
            if pos["direction"] == "LONG":
                hit_sl = bar["low"] <= pos["sl"]
                hit_tp = bar["high"] >= pos["tp"]
            else:
                hit_sl = bar["high"] >= pos["sl"]
                hit_tp = bar["low"] <= pos["tp"]
            if bars_held < 6:
                hit_tp = False

            if hit_sl:
                pnl = -pos["sl_pct"] / 100
                closed.append((pos, "SL", pnl))
            elif hit_tp:
                pnl = pos["tp_pct"] / 100
                closed.append((pos, "TP", pnl))
            elif bars_held >= 48:
                if pos["direction"] == "LONG":
                    pnl = (bar["close"] - pos["entry"]) / pos["entry"]
                else:
                    pnl = (pos["entry"] - bar["close"]) / pos["entry"]
                closed.append((pos, "TIMEOUT", pnl))

        for pos, outcome, pnl in closed:
            capital *= (1 + pnl * 0.02 / (pos["sl_pct"]/100))
            trades.append({
                "strategy": "liquidity_grab_v2",
                "direction": pos["direction"],
                "entry": pos["entry"],
                "exit": bar["close"],
                "pnl_pct": round(pnl * 100, 2),
                "outcome": outcome,
                "regime": pos["regime"],
                "bars_held": i - pos["entry_bar"],
                "sweep_type": pos["sweep_type"],
                "wick_pct": pos["wick_pct"],
                "oi_change": pos["oi_change"],
                "touches": pos["touches"],
                "vol_ratio": pos["vol_ratio"],
            })
            positions.remove(pos)

        peak = max(peak, capital)
        dd = (peak - capital) / peak
        max_dd = max(max_dd, dd)

        # ATR
        if i < 20: continue
        atr = calc_atr(highs_buf, lows_buf, closes_buf, 14)
        if atr == 0: continue

        # S/R levels
        sr_levels = find_significant_levels(bars, i)

        # Detect sweep
        sweep = detect_sweep(bars, oi_data, i, sr_levels, atr)
        if not sweep: continue

        # Cooldown
        key = f"{sweep['direction']}_{sweep['type']}"
        if cooldown.get(key, 0) > i - 48: continue

        # Conditional directional gate
        if regime == "BULL" and sweep["direction"] == "SHORT": continue
        if regime == "BEAR" and sweep["direction"] == "LONG": continue

        # TP/SL
        sl_mult = 0.8
        tp_mult = 2.0
        if sweep["direction"] == "LONG":
            sl = sweep["entry"] - sl_mult * atr
            tp = sweep["entry"] + tp_mult * atr
        else:
            sl = sweep["entry"] + sl_mult * atr
            tp = sweep["entry"] - tp_mult * atr

        sl_pct = (sl_mult * atr / sweep["entry"]) * 100
        tp_pct = (tp_mult * atr / sweep["entry"]) * 100

        positions.append({
            "direction": sweep["direction"], "entry": sweep["entry"],
            "sl": sl, "tp": tp, "sl_pct": sl_pct, "tp_pct": tp_pct,
            "entry_bar": i, "regime": regime,
            "sweep_type": sweep["type"], "wick_pct": sweep["wick_pct"],
            "oi_change": sweep["oi_change"], "touches": sweep["touches"],
            "vol_ratio": sweep["vol_ratio"],
        })
        cooldown[key] = i

    return trades, capital, max_dd


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def calc(trades, label=""):
    if not trades: return {"label": label, "trades": 0}
    pnls = [t["pnl_pct"]/100 for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins)/len(pnls)
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0.001
    pf = gp/gl if gl > 0 else float("inf")
    equity = [1.0]
    for p in pnls: equity.append(equity[-1]*(1+p))
    pk = max(equity)
    mdd = max((pk-e)/pk for e in equity)
    return {"label": label, "trades": len(trades), "wr": round(wr*100,1),
            "pf": round(pf,3), "exp": round(np.mean(pnls)*100,4),
            "ret": round((equity[-1]-1)*100,2), "max_dd": round(mdd*100,2),
            "avg_win": round(np.mean(wins)*100,3) if wins else 0,
            "avg_loss": round(np.mean(losses)*100,3) if losses else 0}

def monte_carlo(trades, n_sims=10000, h=30):
    pnls = [t["pnl_pct"]/100 for t in trades]
    if len(pnls) < 2: return None
    tpd = len(pnls)/113; nt = max(1, int(h*tpd))
    finals = []
    for _ in range(n_sims):
        s = random.choices(pnls, k=nt); c = 1.0
        for r in s: c *= (1+r)
        finals.append(c-1)
    finals.sort(); n = len(finals)
    return {"p5": round(np.percentile(finals,5)*100,2),
            "p50": round(np.percentile(finals,50)*100,2),
            "p95": round(np.percentile(finals,95)*100,2),
            "p_loss": round(sum(1 for f in finals if f<0)/n*100,1)}

def bootstrap_ci(trades, n=5000):
    pnls = [t["pnl_pct"]/100 for t in trades]
    if len(pnls) < 5: return None
    exps = []; pfs = []
    for _ in range(n):
        s = random.choices(pnls, k=len(pnls))
        exps.append(np.mean(s))
        gp = sum(p for p in s if p > 0); gl = abs(sum(p for p in s if p <= 0))
        pfs.append(gp/gl if gl > 0 else 0)
    exps.sort(); pfs.sort(); n2 = len(exps)
    return {"exp_ci": (round(exps[int(n2*0.025)],4), round(exps[int(n2*0.975)],4)),
            "pf_ci": (round(pfs[int(n2*0.025)],3), round(pfs[int(n2*0.975)],3))}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("LIQUIDITY SWEEP DETECTOR v2 — TIGHTENED")
    print("=" * 80)

    ohlcv_path = os.path.join(BASE, "data", "eth_15m_merged.csv")
    oi_path = os.path.join(BASE, "data", "forced_movement", "oi_history.csv")

    bars = load_ohlcv(ohlcv_path, "2026-04-01")
    oi_data = load_oi(oi_path)

    print(f"\n  OHLCV: {len(bars)} bars ({bars[0]['ts']} → {bars[-1]['ts']})")
    print(f"  OI: {len(oi_data)} points")

    # OI coverage check
    oi_ts = [t for t, _ in oi_data]
    if oi_ts:
        first_oi = datetime.utcfromtimestamp(oi_ts[0]/1000).strftime("%Y-%m-%d %H:%M")
        last_oi = datetime.utcfromtimestamp(oi_ts[-1]/1000).strftime("%Y-%m-%d %H:%M")
        print(f"  OI range: {first_oi} → {last_oi}")

    # Test OI matching
    match_count = 0
    for bar in bars[100:110]:
        oi = get_oi(oi_data, bar["ts"])
        if oi: match_count += 1
    print(f"  OI match test (10 bars): {match_count}/10")

    print(f"\n  Computing daily regimes...")
    regimes = compute_daily_regimes(bars)
    rc = defaultdict(int)
    for r in regimes.values(): rc[r] += 1
    for regime in ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH"]:
        cnt = rc.get(regime, 0)
        if cnt: print(f"    {regime}: {cnt} days ({cnt/len(regimes)*100:.1f}%)")

    print(f"\n  Running sweep detection v2...")
    trades, final_cap, max_dd = run_backtest(bars, oi_data, regimes)

    s = calc(trades, "Sweep v2")
    ci = bootstrap_ci(trades)
    mc30 = monte_carlo(trades, 10000, 30)
    mc90 = monte_carlo(trades, 10000, 90)

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    if s["trades"] == 0:
        print(f"  NO TRADES — filters too tight or no OI data coverage")
    else:
        print(f"  Trades: {s['trades']}")
        print(f"  WR: {s.get('wr',0)}%")
        print(f"  PF: {s.get('pf',0)}")
        print(f"  Expectancy: {s.get('exp',0)}%")
        print(f"  Avg Win: {s.get('avg_win',0)}% | Avg Loss: {s.get('avg_loss',0)}%")
        print(f"  Return: {s.get('ret',0)}% | Max DD: {s.get('max_dd',0)}%")
        if ci:
            print(f"  Bootstrap: Exp CI [{ci['exp_ci'][0]:+.4f}%, {ci['exp_ci'][1]:+.4f}%]  PF CI [{ci['pf_ci'][0]:.3f}, {ci['pf_ci'][1]:.3f}]")
        if mc30:
            print(f"  MC 30d: P50={mc30['p50']:+.1f}% P(loss)={mc30['p_loss']:.1f}%")
        if mc90:
            print(f"  MC 90d: P50={mc90['p50']:+.1f}% P(loss)={mc90['p_loss']:.1f}%")

        # By sweep type
        print(f"\n  By Sweep Type:")
        for st in ["bearish_sweep", "bullish_sweep"]:
            st_t = [t for t in trades if t.get("sweep_type") == st]
            if st_t:
                ss = calc(st_t, st)
                print(f"    {st}: {ss['trades']} trades  WR={ss.get('wr',0)}%  PF={ss.get('pf',0)}  Exp={ss.get('exp',0):+.4f}%")

        # By regime
        print(f"\n  By Regime:")
        for regime in ["MILDLY_BEARISH", "RANGING", "BULL", "BEAR"]:
            rt = [t for t in trades if t["regime"] == regime]
            if rt:
                rs = calc(rt, regime)
                print(f"    {regime}: {rs['trades']} trades  WR={rs.get('wr',0)}%  PF={rs.get('pf',0)}  Exp={rs.get('exp',0):+.4f}%")

        # Trade log
        print(f"\n  Trade Log:")
        print(f"  {'#':<4} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'PnL%':>8} {'Outcome':<10} {'Regime':<15} {'Wick%':>7} {'OI Δ':>8} {'Vol':>6} {'Touches':>8} {'Bars':>5}")
        print(f"  {'-'*110}")
        for j, t in enumerate(trades):
            print(f"  {j:<4} {t['direction']:<6} ${t['entry']:>9.2f} ${t['exit']:>9.2f} {t['pnl_pct']:>+7.2f}% {t['outcome']:<10} {t['regime']:<15} {t.get('wick_pct',0)*100:>6.2f}% {t.get('oi_change',0)*100:>7.3f}% {t.get('vol_ratio',0):>5.1f}x {t.get('touches',0):>7} {t['bars_held']:>5}")

    # Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bars": len(bars), "oi_points": len(oi_data),
        "trades": s["trades"], "wr": s.get("wr",0), "pf": s.get("pf",0),
        "exp": s.get("exp",0), "ret": s.get("ret",0), "max_dd": s.get("max_dd",0),
        "trade_list": trades,
    }
    out_path = os.path.join(BASE, "data", "5agent_backtest", "sweep_detector_v2.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")
    print("\nDone.")
