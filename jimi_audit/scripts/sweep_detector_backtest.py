#!/usr/bin/env python3
"""
Liquidity Sweep Detector v1 — Backtest

Mechanism:
1. Price exceeds S/R level by 0.1-0.5% (wick beyond)
2. OI drops sharply (stops being hit → positions closed)
3. Volume spikes (liquidation cascade)
4. Price closes back inside level (rejection)
5. Enter counter-sweep direction

Data: 15m OHLCV + OI history + volume profile
"""
import json, os, csv, math, random
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

random.seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_ohlcsv(path, start="2026-04-01"):
    bars = []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = row.get("Open time", "")
            if ts < start:
                continue
            try:
                bars.append({
                    "ts": ts, "open": float(row["Open"]), "high": float(row["High"]),
                    "low": float(row["Low"]), "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                    "taker_buy": float(row.get("Taker buy base asset volume", 0)),
                })
            except:
                continue
    return bars

def load_oi(path):
    """Load OI history (timestamp_ms, oi, oi_usd)."""
    oi_data = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    ts_ms = int(parts[0])
                    oi = float(parts[1])
                    oi_data.append({"ts_ms": ts_ms, "oi": oi})
                except:
                    continue
    return oi_data

def load_daily_regimes(path):
    """Load daily regime classifications from backtest results."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("regimes", {})
    except:
        return {}

# ═══════════════════════════════════════════════════════════════
# S/R DETECTION
# ═══════════════════════════════════════════════════════════════

def find_sr_levels(bars, idx, lookback=96):
    """Find S/R levels from swing highs/lows."""
    if idx < lookback:
        return []
    highs = [b["high"] for b in bars[idx-lookback:idx]]
    lows = [b["low"] for b in bars[idx-lookback:idx]]
    volumes = [b["volume"] for b in bars[idx-lookback:idx]]
    levels = []

    for i in range(3, lookback - 3):
        # Swing high
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
            highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            levels.append({
                "price": highs[i], "type": "resistance",
                "bar_idx": idx - lookback + i,
                "strength": volumes[i],
            })
        # Swing low
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
            lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            levels.append({
                "price": lows[i], "type": "support",
                "bar_idx": idx - lookback + i,
                "strength": volumes[i],
            })

    # Deduplicate (within 0.2%)
    deduped = []
    for lv in sorted(levels, key=lambda x: x["bar_idx"]):
        found = False
        for d in deduped:
            if abs(lv["price"] - d["price"]) / d["price"] < 0.002:
                d["strength"] = max(d["strength"], lv["strength"])
                found = True
                break
        if not found:
            deduped.append(lv)

    return sorted(deduped, key=lambda x: x["strength"], reverse=True)[:15]


# ═══════════════════════════════════════════════════════════════
# SWEEP DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_sweep(bars, oi_data, idx, sr_levels, engine):
    """
    Detect a liquidity sweep at bar index.

    Sweep conditions:
    1. Price wick exceeds S/R level by 0.1-0.5%
    2. Close is back inside (wick rejection)
    3. OI drops (stops being hit)
    4. Volume is above average
    """
    bar = bars[idx]
    price = bar["close"]
    high = bar["high"]
    low = bar["low"]
    vol = bar["volume"]

    # Need minimum data
    if idx < 20:
        return None

    # Average volume
    avg_vol = np.mean([b["volume"] for b in bars[max(0,idx-20):idx]])
    if avg_vol == 0:
        return None

    # OI change (compare current bar's OI to previous)
    oi_now = get_oi_at_time(oi_data, bar["ts"])
    oi_prev = get_oi_at_time(oi_data, bars[idx-1]["ts"]) if idx > 0 else oi_now
    oi_change = (oi_now - oi_prev) / oi_prev if oi_prev > 0 else 0

    # ATR for distance normalization
    atr = engine.get("atr", 0)
    if atr == 0:
        return None

    # Check each S/R level for sweep
    for level in sr_levels:
        level_price = level["price"]
        dist_pct = abs(price - level_price) / level_price

        # Skip if price is too far from level
        if dist_pct > 0.02:
            continue

        # === BEARISH SWEEP (price swept above resistance, then rejected) ===
        if level["type"] == "resistance":
            # Wick exceeded level
            wick_above = (high - level_price) / level_price
            if wick_above < 0.001 or wick_above > 0.005:  # 0.1% to 0.5%
                continue

            # Close is back below level (rejection)
            if price >= level_price:
                continue

            # Wick rejection ratio
            bar_range = high - low
            if bar_range == 0:
                continue
            upper_wick = high - max(bar["open"], bar["close"])
            wick_ratio = upper_wick / bar_range
            if wick_ratio < 0.4:  # At least 40% upper wick
                continue

            # OI dropped (stops hit) or volume spike
            oi_dropped = oi_change < -0.002  # 0.2% OI drop
            vol_spike = vol > avg_vol * 1.3

            if not (oi_dropped or vol_spike):
                continue

            # Sweep confirmed
            direction = "SHORT"  # Counter-sweep: sell the rejection
            return {
                "direction": direction,
                "entry": price,
                "level": level_price,
                "wick_pct": wick_above,
                "wick_ratio": wick_ratio,
                "oi_change": oi_change,
                "vol_ratio": vol / avg_vol,
                "bar_idx": idx,
                "type": "bearish_sweep",
            }

        # === BULLISH SWEEP (price swept below support, then rejected) ===
        elif level["type"] == "support":
            wick_below = (level_price - low) / level_price
            if wick_below < 0.001 or wick_below > 0.005:
                continue

            # Close is back above level
            if price <= level_price:
                continue

            bar_range = high - low
            if bar_range == 0:
                continue
            lower_wick = min(bar["open"], bar["close"]) - low
            wick_ratio = lower_wick / bar_range
            if wick_ratio < 0.4:
                continue

            oi_dropped = oi_change < -0.002
            vol_spike = vol > avg_vol * 1.3

            if not (oi_dropped or vol_spike):
                continue

            direction = "LONG"
            return {
                "direction": direction,
                "entry": price,
                "level": level_price,
                "wick_pct": wick_below,
                "wick_ratio": wick_ratio,
                "oi_change": oi_change,
                "vol_ratio": vol / avg_vol,
                "bar_idx": idx,
                "type": "bullish_sweep",
            }

    return None


def get_oi_at_time(oi_data, ts_str):
    """Get OI closest to a timestamp string."""
    if not oi_data:
        return 0
    try:
        ts = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
        ts_ms = int(ts.timestamp() * 1000)
        # Binary search for closest
        best = oi_data[0]
        best_delta = abs(best["ts_ms"] - ts_ms)
        for d in oi_data:
            delta = abs(d["ts_ms"] - ts_ms)
            if delta < best_delta:
                best_delta = delta
                best = d
        return best["oi"]
    except:
        return 0


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINE (for ATR, EMA, etc.)
# ═══════════════════════════════════════════════════════════════

class FeatureEngine:
    def __init__(self):
        self.closes = []
        self.highs = []
        self.lows = []
        self.volumes = []

    def update(self, bar):
        self.closes.append(bar["close"])
        self.highs.append(bar["high"])
        self.lows.append(bar["low"])
        self.volumes.append(bar["volume"])

    def get(self, idx):
        if idx < 20:
            return {"atr": 0}
        closes = self.closes[:idx+1]
        highs = self.highs[:idx+1]
        lows = self.lows[:idx+1]
        atr = self._atr(highs, lows, closes, 14)
        return {"atr": atr}

    def _atr(self, highs, lows, closes, period):
        if len(closes) < period + 1:
            return 0
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
            trs.append(tr)
        return np.mean(trs[-period:])


# ═══════════════════════════════════════════════════════════════
# REGIME CLASSIFIER (simplified V4)
# ═══════════════════════════════════════════════════════════════

def compute_daily_regimes(bars):
    daily = defaultdict(lambda: {"highs": [], "lows": [], "closes": []})
    for bar in bars:
        day = bar["ts"][:10]
        daily[day]["highs"].append(bar["high"])
        daily[day]["lows"].append(bar["low"])
        daily[day]["closes"].append(bar["close"])

    sorted_days = sorted(daily.keys())
    daily_closes = []
    regimes = {}

    for day in sorted_days:
        d = daily[day]
        daily_closes.append(d["closes"][-1])
        if len(daily_closes) < 50:
            regimes[day] = "RANGING"
            continue

        closes = daily_closes
        ema50 = _ema(closes, 50)
        ema10 = _ema(closes, 10)
        price = closes[-1]
        rsi = _rsi(closes, 14)

        above_ema50 = price > ema50
        ema_cross = ema10 > ema50
        slope = (ema50 - _ema(closes[:-5], 50)) / _ema(closes[:-5], 50) if len(closes) > 55 else 0

        bull = bear = stress = 0
        if above_ema50 and ema_cross and slope > 0.001: bull += 3
        elif not above_ema50 and not ema_cross and slope < -0.001: bear += 3
        elif not above_ema50 and slope < -0.002: bear += 2
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
    ag = sum(gains)/period
    al = sum(losses)/period
    if al == 0: return 100
    return 100 - (100/(1+ag/al))


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def run_backtest(bars, oi_data, regimes):
    engine = FeatureEngine()
    trades = []
    positions = []
    capital = 10000.0
    peak = capital
    max_dd = 0
    cooldown = {}

    for i, bar in enumerate(bars):
        engine.update(bar)
        feat = engine.get(i)
        day = bar["ts"][:10]
        regime = regimes.get(day, "RANGING")

        # Check open positions for exit
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
                hit_tp = False  # Min hold

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
                "strategy": "liquidity_grab",
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
            })
            positions.remove(pos)

        peak = max(peak, capital)
        dd = (peak - capital) / peak
        max_dd = max(max_dd, dd)

        # Find S/R levels
        sr_levels = find_sr_levels(bars, i)

        # Detect sweep
        sweep = detect_sweep(bars, oi_data, i, sr_levels, feat)

        if sweep:
            # Cooldown check
            key = f"{sweep['direction']}_{sweep['type']}"
            if cooldown.get(key, 0) > i - 48:
                continue

            # Conditional directional gate
            if regime == "BULL" and sweep["direction"] == "SHORT":
                continue
            if regime == "BEAR" and sweep["direction"] == "LONG":
                continue

            # TP/SL
            atr = feat["atr"]
            if atr == 0:
                continue

            sl_mult = 0.8  # Tight SL for sweep trades
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
                "direction": sweep["direction"],
                "entry": sweep["entry"],
                "sl": sl, "tp": tp,
                "sl_pct": sl_pct, "tp_pct": tp_pct,
                "entry_bar": i,
                "regime": regime,
                "sweep_type": sweep["type"],
                "wick_pct": sweep["wick_pct"],
                "oi_change": sweep["oi_change"],
            })
            cooldown[key] = i

    return trades, capital, max_dd


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def calc(trades, label=""):
    if not trades:
        return {"label": label, "trades": 0}
    pnls = [t["pnl_pct"]/100 for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins)/len(pnls)
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0.001
    pf = gp/gl if gl > 0 else float("inf")
    equity = [1.0]
    for p in pnls:
        equity.append(equity[-1]*(1+p))
    peak = max(equity)
    max_dd = max((peak-e)/peak for e in equity)
    ret = equity[-1]-1
    return {"label": label, "trades": len(trades), "wr": round(wr*100,1),
            "pf": round(pf,3), "exp": round(np.mean(pnls)*100,4),
            "ret": round(ret*100,2), "max_dd": round(max_dd*100,2),
            "avg_win": round(np.mean(wins)*100,3) if wins else 0,
            "avg_loss": round(np.mean(losses)*100,3) if losses else 0}

def monte_carlo(trades, n_sims=10000, h=30):
    pnls = [t["pnl_pct"]/100 for t in trades]
    if len(pnls) < 2: return None
    tpd = len(pnls)/113
    nt = max(1, int(h*tpd))
    finals = []
    for _ in range(n_sims):
        s = random.choices(pnls, k=nt)
        c = 1.0
        for r in s: c *= (1+r)
        finals.append(c-1)
    finals.sort()
    n = len(finals)
    return {"p5": round(np.percentile(finals,5)*100,2),
            "p50": round(np.percentile(finals,50)*100,2),
            "p95": round(np.percentile(finals,95)*100,2),
            "p_loss": round(sum(1 for f in finals if f<0)/n*100,1)}

def bootstrap_ci(trades, n=5000):
    pnls = [t["pnl_pct"]/100 for t in trades]
    if len(pnls) < 5: return None
    exps = []
    pfs = []
    for _ in range(n):
        s = random.choices(pnls, k=len(pnls))
        exps.append(np.mean(s))
        gp = sum(p for p in s if p > 0)
        gl = abs(sum(p for p in s if p <= 0))
        pfs.append(gp/gl if gl > 0 else 0)
    exps.sort(); pfs.sort()
    n2 = len(exps)
    return {"exp_ci": (round(exps[int(n2*0.025)],4), round(exps[int(n2*0.975)],4)),
            "pf_ci": (round(pfs[int(n2*0.025)],3), round(pfs[int(n2*0.975)],3))}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("LIQUIDITY SWEEP DETECTOR v1 — BACKTEST")
    print("=" * 80)

    # Load data
    ohlcv_path = os.path.join(BASE, "data", "eth_15m_merged.csv")
    oi_path = os.path.join(BASE, "data", "forced_movement", "oi_history.csv")

    print(f"\nLoading OHLCV: {ohlcv_path}")
    bars = load_ohlcsv(ohlcv_path, "2026-04-01")
    print(f"  {len(bars)} bars ({bars[0]['ts']} → {bars[-1]['ts']})")

    print(f"Loading OI: {oi_path}")
    oi_data = load_oi(oi_path)
    print(f"  {len(oi_data)} OI points")

    print(f"\nComputing daily regimes...")
    regimes = compute_daily_regimes(bars)
    rc = defaultdict(int)
    for r in regimes.values(): rc[r] += 1
    total_days = len(regimes)
    for regime in ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH"]:
        cnt = rc.get(regime, 0)
        if cnt: print(f"  {regime}: {cnt} days ({cnt/total_days*100:.1f}%)")

    # Run backtest
    print(f"\nRunning sweep detection backtest...")
    trades, final_cap, max_dd = run_backtest(bars, oi_data, regimes)

    # Results
    s = calc(trades, "Sweep v1")
    ci = bootstrap_ci(trades)
    mc30 = monte_carlo(trades, 10000, 30)
    mc90 = monte_carlo(trades, 10000, 90)

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"  Trades: {s['trades']}")
    print(f"  WR: {s.get('wr',0)}%")
    print(f"  PF: {s.get('pf',0)}")
    print(f"  Expectancy: {s.get('exp',0)}%")
    print(f"  Avg Win: {s.get('avg_win',0)}% | Avg Loss: {s.get('avg_loss',0)}%")
    print(f"  Return: {s.get('ret',0)}%")
    print(f"  Max DD: {s.get('max_dd',0)}%")
    if ci:
        print(f"  Bootstrap Exp CI: [{ci['exp_ci'][0]:+.4f}%, {ci['exp_ci'][1]:+.4f}%]")
        print(f"  Bootstrap PF CI: [{ci['pf_ci'][0]:.3f}, {ci['pf_ci'][1]:.3f}]")
    if mc30:
        print(f"  MC 30-day: P50={mc30['p50']:+.1f}% P(loss)={mc30['p_loss']:.1f}%")
    if mc90:
        print(f"  MC 90-day: P50={mc90['p50']:+.1f}% P(loss)={mc90['p_loss']:.1f}%")

    # By sweep type
    if trades:
        print(f"\n  By Sweep Type:")
        for st in ["bearish_sweep", "bullish_sweep"]:
            st_trades = [t for t in trades if t.get("sweep_type") == st]
            if st_trades:
                ss = calc(st_trades, st)
                print(f"    {st}: {ss['trades']} trades  WR={ss.get('wr',0)}%  PF={ss.get('pf',0)}  Exp={ss.get('exp',0):+.4f}%")

        # By regime
        print(f"\n  By Regime:")
        for regime in ["MILDLY_BEARISH", "RANGING", "BULL", "BEAR"]:
            rt = [t for t in trades if t["regime"] == regime]
            if rt:
                rs = calc(rt, regime)
                print(f"    {regime}: {rs['trades']} trades  WR={rs.get('wr',0)}%  PF={rs.get('pf',0)}  Exp={rs.get('exp',0):+.4f}%")

        # By direction
        print(f"\n  By Direction:")
        for d in ["LONG", "SHORT"]:
            dt = [t for t in trades if t["direction"] == d]
            if dt:
                ds = calc(dt, d)
                print(f"    {d}: {ds['trades']} trades  WR={ds.get('wr',0)}%  PF={ds.get('pf',0)}  Exp={ds.get('exp',0):+.4f}%")

        # Trade log
        print(f"\n  Trade Log (first 20):")
        print(f"  {'#':<4} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'PnL%':>8} {'Outcome':<10} {'Regime':<15} {'Type':<15} {'Wick%':>7} {'OI Δ':>8} {'Bars':>5}")
        print(f"  {'-'*100}")
        for j, t in enumerate(trades[:20]):
            print(f"  {j:<4} {t['direction']:<6} ${t['entry']:>9.2f} ${t['exit']:>9.2f} {t['pnl_pct']:>+7.2f}% {t['outcome']:<10} {t['regime']:<15} {t.get('sweep_type',''):<15} {t.get('wick_pct',0)*100:>6.2f}% {t.get('oi_change',0)*100:>7.3f}% {t['bars_held']:>5}")

    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bars_tested": len(bars),
        "oi_points": len(oi_data),
        "total_trades": s["trades"],
        "win_rate": s.get("wr", 0),
        "profit_factor": s.get("pf", 0),
        "expectancy": s.get("exp", 0),
        "return_pct": s.get("ret", 0),
        "max_dd_pct": s.get("max_dd", 0),
        "trades": trades,
    }
    out_path = os.path.join(BASE, "data", "5agent_backtest", "sweep_detector_backtest.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")
    print("\nDone.")
