#!/usr/bin/env python3
"""
Liquidity Sweep Detector v2.1 — Uses derivatives_collected.csv for OI.
Same tight filters as v2 but with full OI coverage.
"""
import json, os, csv, math, random
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

random.seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_ohlcv(path, start="2026-04-13 07:00:00"):
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
                })
            except: continue
    return bars

def load_oi_from_derivatives(path):
    """Load OI from derivatives_collected.csv (full range)."""
    data = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp", "")
            oi_str = row.get("oi", "").strip()
            if not ts or not oi_str:
                continue
            try:
                oi = float(oi_str)
                # Parse timestamp — handle both formats
                ts_clean = ts.replace("T", " ")[:19]
                dt = datetime.strptime(ts_clean, "%Y-%m-%d %H:%M:%S")
                data.append({"ts_str": ts_clean, "dt": dt, "oi": oi})
            except:
                continue
    return sorted(data, key=lambda x: x["dt"])

def get_oi(oi_data, bar_ts_str):
    """Get OI closest to bar timestamp (within 15 min)."""
    if not oi_data:
        return None
    try:
        bar_dt = datetime.strptime(bar_ts_str[:19], "%Y-%m-%d %H:%M:%S")
    except:
        return None

    best = None
    best_delta = 900  # 15 min in seconds
    for d in oi_data:
        delta = abs((d["dt"] - bar_dt).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best = d["oi"]
    return best


def find_significant_levels(bars, idx, lookback=192):
    """Find S/R with 2+ touches or session H/L."""
    if idx < lookback: return []
    window = bars[idx-lookback:idx]
    highs = [b["high"] for b in window]
    lows = [b["low"] for b in window]
    volumes = [b["volume"] for b in window]

    swings = []
    for i in range(3, len(window) - 3):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swings.append({"price": highs[i], "type": "resistance", "vol": volumes[i]})
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swings.append({"price": lows[i], "type": "support", "vol": volumes[i]})

    tol = 0.002
    levels = []
    for swing in swings:
        price = swing["price"]
        touches = 0
        for i in range(len(window)):
            if swing["type"] == "resistance":
                if abs(highs[i] - price) / price < tol: touches += 1
            else:
                if abs(lows[i] - price) / price < tol: touches += 1
        if touches >= 2:
            levels.append({"price": price, "type": swing["type"], "touches": touches, "strength": touches * swing["vol"]})

    # Session H/L
    if len(window) >= 96:
        sh = max(highs[-96:]); sl = min(lows[-96:])
        if not any(abs(l["price"] - sh)/sh < tol for l in levels):
            levels.append({"price": sh, "type": "resistance", "touches": 99, "strength": 99999})
        if not any(abs(l["price"] - sl)/sl < tol for l in levels):
            levels.append({"price": sl, "type": "support", "touches": 99, "strength": 99999})

    deduped = []
    for lv in sorted(levels, key=lambda x: x["strength"], reverse=True):
        found = False
        for d in deduped:
            if abs(lv["price"] - d["price"]) / d["price"] < tol:
                d["touches"] = max(d["touches"], lv["touches"])
                found = True
                break
        if not found:
            deduped.append(lv)
    return sorted(deduped, key=lambda x: x["strength"], reverse=True)[:10]


def detect_sweep(bars, oi_data, idx, sr_levels, atr):
    """Detect sweep with strict criteria."""
    bar = bars[idx]
    price = bar["close"]; high = bar["high"]; low = bar["low"]
    vol = bar["volume"]; bar_open = bar["open"]

    if idx < 20 or atr == 0: return None

    avg_vol = np.mean([b["volume"] for b in bars[max(0,idx-20):idx]])
    if avg_vol == 0: return None

    # OI change (MANDATORY)
    oi_now = get_oi(oi_data, bar["ts"])
    oi_prev = get_oi(oi_data, bars[idx-1]["ts"])
    if oi_now is None or oi_prev is None or oi_prev == 0:
        return None
    oi_change = (oi_now - oi_prev) / oi_prev

    if oi_change >= -0.003:  # Need 0.3%+ OI drop
        return None

    if vol < avg_vol * 1.3:  # Volume spike required
        return None

    bar_range = high - low
    if bar_range == 0: return None

    for level in sr_levels:
        lp = level["price"]

        if level["type"] == "resistance":
            wick = (high - lp) / lp
            if wick < 0.001 or wick > 0.005: continue
            if price >= lp: continue
            upper_wick = high - max(bar_open, price)
            if upper_wick / bar_range < 0.6: continue
            return {"direction": "SHORT", "entry": price, "level": lp,
                    "touches": level["touches"], "wick_pct": wick,
                    "wick_ratio": upper_wick / bar_range,
                    "oi_change": oi_change, "vol_ratio": vol / avg_vol,
                    "bar_idx": idx, "type": "bearish_sweep"}

        elif level["type"] == "support":
            wick = (lp - low) / lp
            if wick < 0.001 or wick > 0.005: continue
            if price <= lp: continue
            lower_wick = min(bar_open, price) - low
            if lower_wick / bar_range < 0.6: continue
            return {"direction": "LONG", "entry": price, "level": lp,
                    "touches": level["touches"], "wick_pct": wick,
                    "wick_ratio": lower_wick / bar_range,
                    "oi_change": oi_change, "vol_ratio": vol / avg_vol,
                    "bar_idx": idx, "type": "bullish_sweep"}

    return None


def compute_daily_regimes(bars):
    daily = defaultdict(lambda: {"closes": []})
    for bar in bars:
        daily[bar["ts"][:10]]["closes"].append(bar["close"])
    sorted_days = sorted(daily.keys())
    dc = []; regimes = {}
    for day in sorted_days:
        dc.append(daily[day]["closes"][-1])
        if len(dc) < 50: regimes[day] = "RANGING"; continue
        ema50 = _ema(dc, 50); ema10 = _ema(dc, 10); price = dc[-1]
        rsi = _rsi(dc, 14)
        above = price > ema50; cross = ema10 > ema50
        slope = (ema50 - _ema(dc[:-5], 50)) / _ema(dc[:-5], 50) if len(dc) > 55 else 0
        bull = bear = stress = 0
        if above and cross and slope > 0.001: bull += 3
        elif not above and not cross and slope < -0.001: bear += 3
        elif not above and slope < -0.002: bear += 2
        if rsi > 70: bear += 1
        elif rsi < 30: bull += 1
        if len(dc) >= 5:
            wroc = (dc[-1] - dc[-5]) / dc[-5]
            if wroc > 0.05: bull += 1.5
            elif wroc < -0.05: bear += 1.5
            elif wroc < -0.10: stress += 1.5
        if stress > 3: regimes[day] = "STRESS"
        elif bull > bear + 2: regimes[day] = "BULL"
        elif bear > bull + 2: regimes[day] = "BEAR"
        elif bear > bull + 1: regimes[day] = "MILDLY_BEARISH"
        else: regimes[day] = "RANGING"
    return regimes

def _ema(d, p):
    if len(d) < p: return d[-1] if d else 0
    m = 2/(p+1); e = sum(d[:p])/p
    for x in d[p:]: e = (x-e)*m+e
    return e

def _rsi(d, p=14):
    if len(d) < p+1: return 50
    dl = [d[i]-d[i-1] for i in range(1,len(d))]
    g = [x if x>0 else 0 for x in dl[-p:]]
    l = [-x if x<0 else 0 for x in dl[-p:]]
    ag = sum(g)/p; al = sum(l)/p
    if al == 0: return 100
    return 100-(100/(1+ag/al))

def calc_atr(h, l, c, p=14):
    if len(c) < p+1: return 0
    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1,len(c))]
    return np.mean(trs[-p:])


def run_backtest(bars, oi_data, regimes):
    trades = []; positions = []
    capital = 10000.0; peak = capital; max_dd = 0; cooldown = {}
    cbuf = []; hbuf = []; lbuf = []

    for i, bar in enumerate(bars):
        cbuf.append(bar["close"]); hbuf.append(bar["high"]); lbuf.append(bar["low"])
        regime = regimes.get(bar["ts"][:10], "RANGING")

        closed = []
        for pos in positions:
            bh = i - pos["entry_bar"]
            if pos["direction"] == "LONG":
                sl_hit = bar["low"] <= pos["sl"]; tp_hit = bar["high"] >= pos["tp"]
            else:
                sl_hit = bar["high"] >= pos["sl"]; tp_hit = bar["low"] <= pos["tp"]
            if bh < 6: tp_hit = False

            if sl_hit:
                pnl = -pos["sl_pct"]/100; closed.append((pos, "SL", pnl))
            elif tp_hit:
                pnl = pos["tp_pct"]/100; closed.append((pos, "TP", pnl))
            elif bh >= 48:
                pnl = (bar["close"]-pos["entry"])/pos["entry"] if pos["direction"]=="LONG" else (pos["entry"]-bar["close"])/pos["entry"]
                closed.append((pos, "TIMEOUT", pnl))

        for pos, outcome, pnl in closed:
            capital *= (1 + pnl * 0.02 / (pos["sl_pct"]/100))
            trades.append({
                "strategy": "liquidity_grab_v2", "direction": pos["direction"],
                "entry": pos["entry"], "exit": bar["close"],
                "pnl_pct": round(pnl*100,2), "outcome": outcome,
                "regime": pos["regime"], "bars_held": i-pos["entry_bar"],
                "sweep_type": pos["sweep_type"], "wick_pct": pos["wick_pct"],
                "oi_change": pos["oi_change"], "touches": pos["touches"],
                "vol_ratio": pos["vol_ratio"],
            })
            positions.remove(pos)

        peak = max(peak, capital); dd = (peak-capital)/peak; max_dd = max(max_dd, dd)
        if i < 20: continue
        atr = calc_atr(hbuf, lbuf, cbuf, 14)
        if atr == 0: continue

        sr = find_significant_levels(bars, i)
        sweep = detect_sweep(bars, oi_data, i, sr, atr)
        if not sweep: continue

        key = f"{sweep['direction']}_{sweep['type']}"
        if cooldown.get(key, 0) > i - 48: continue
        if regime == "BULL" and sweep["direction"] == "SHORT": continue
        if regime == "BEAR" and sweep["direction"] == "LONG": continue

        sl_m = 0.8; tp_m = 2.0
        if sweep["direction"] == "LONG":
            sl = sweep["entry"]-sl_m*atr; tp = sweep["entry"]+tp_m*atr
        else:
            sl = sweep["entry"]+sl_m*atr; tp = sweep["entry"]-tp_m*atr

        positions.append({
            "direction": sweep["direction"], "entry": sweep["entry"],
            "sl": sl, "tp": tp,
            "sl_pct": (sl_m*atr/sweep["entry"])*100,
            "tp_pct": (tp_m*atr/sweep["entry"])*100,
            "entry_bar": i, "regime": regime,
            "sweep_type": sweep["type"], "wick_pct": sweep["wick_pct"],
            "oi_change": sweep["oi_change"], "touches": sweep["touches"],
            "vol_ratio": sweep["vol_ratio"],
        })
        cooldown[key] = i

    return trades, capital, max_dd


def calc(trades, label=""):
    if not trades: return {"label": label, "trades": 0}
    pnls = [t["pnl_pct"]/100 for t in trades]
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    wr = len(wins)/len(pnls)
    gp = sum(wins) if wins else 0; gl = abs(sum(losses)) if losses else 0.001
    pf = gp/gl if gl > 0 else float("inf")
    eq = [1.0]
    for p in pnls: eq.append(eq[-1]*(1+p))
    pk = max(eq); mdd = max((pk-e)/pk for e in eq)
    return {"label": label, "trades": len(trades), "wr": round(wr*100,1),
            "pf": round(pf,3), "exp": round(np.mean(pnls)*100,4),
            "ret": round((eq[-1]-1)*100,2), "max_dd": round(mdd*100,2),
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


if __name__ == "__main__":
    print("=" * 80)
    print("SWEEP DETECTOR v2.1 — Full OI Coverage")
    print("=" * 80)

    bars = load_ohlcv(os.path.join(BASE, "data", "eth_15m_merged.csv"), "2026-04-01")
    oi_data = load_oi_from_derivatives(os.path.join(BASE, "data", "derivatives_history", "derivatives_collected.csv"))

    print(f"\n  OHLCV: {len(bars)} bars ({bars[0]['ts']} → {bars[-1]['ts']})")
    print(f"  OI: {len(oi_data)} points ({oi_data[0]['ts_str']} → {oi_data[-1]['ts_str']})")

    # Test OI matching
    match = 0
    for bar in bars[1000:1010]:
        if get_oi(oi_data, bar["ts"]) is not None: match += 1
    print(f"  OI match test (10 bars): {match}/10")

    regimes = compute_daily_regimes(bars)
    rc = defaultdict(int)
    for r in regimes.values(): rc[r] += 1
    for regime in ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH"]:
        cnt = rc.get(regime, 0)
        if cnt: print(f"  {regime}: {cnt} days ({cnt/len(regimes)*100:.1f}%)")

    print(f"\n  Running sweep detection...")
    trades, final_cap, max_dd = run_backtest(bars, oi_data, regimes)

    s = calc(trades, "Sweep v2.1")
    ci = bootstrap_ci(trades)
    mc30 = monte_carlo(trades, 10000, 30)
    mc90 = monte_carlo(trades, 10000, 90)

    print(f"\n{'='*80}")
    if s["trades"] == 0:
        print("NO TRADES — filters too tight or OI matching still broken")
    else:
        print(f"Trades: {s['trades']}  WR: {s.get('wr',0)}%  PF: {s.get('pf',0)}  Exp: {s.get('exp',0)}%  Ret: {s.get('ret',0)}%  MaxDD: {s.get('max_dd',0)}%")
        if ci: print(f"Bootstrap: Exp CI [{ci['exp_ci'][0]:+.4f}%, {ci['exp_ci'][1]:+.4f}%]  PF CI [{ci['pf_ci'][0]:.3f}, {ci['pf_ci'][1]:.3f}]")
        if mc30: print(f"MC 30d: P50={mc30['p50']:+.1f}% P(loss)={mc30['p_loss']:.1f}%")
        if mc90: print(f"MC 90d: P50={mc90['p50']:+.1f}% P(loss)={mc90['p_loss']:.1f}%")

        print(f"\nBy Type:")
        for st in ["bearish_sweep", "bullish_sweep"]:
            st_t = [t for t in trades if t.get("sweep_type") == st]
            if st_t:
                ss = calc(st_t, st)
                print(f"  {st}: {ss['trades']} trades  WR={ss.get('wr',0)}%  PF={ss.get('pf',0)}")

        print(f"\nBy Regime:")
        for regime in ["MILDLY_BEARISH", "RANGING", "BULL", "BEAR"]:
            rt = [t for t in trades if t["regime"] == regime]
            if rt:
                rs = calc(rt, regime)
                print(f"  {regime}: {rs['trades']} trades  WR={rs.get('wr',0)}%  PF={rs.get('pf',0)}")

        print(f"\nTrade Log:")
        print(f"  {'#':<4} {'Dir':<6} {'Entry':>10} {'PnL%':>8} {'Out':<8} {'Regime':<15} {'OI Δ':>8} {'Vol':>6} {'Touch':>6} {'Bars':>5}")
        print(f"  {'-'*85}")
        for j, t in enumerate(trades):
            print(f"  {j:<4} {t['direction']:<6} ${t['entry']:>9.2f} {t['pnl_pct']:>+7.2f}% {t['outcome']:<8} {t['regime']:<15} {t.get('oi_change',0)*100:>7.3f}% {t.get('vol_ratio',0):>5.1f}x {t.get('touches',0):>5} {t['bars_held']:>5}")

    out_path = os.path.join(BASE, "data", "5agent_backtest", "sweep_v21.json")
    with open(out_path, "w") as f:
        json.dump({"trades": trades, "total": s["trades"], "pf": s.get("pf",0), "wr": s.get("wr",0)}, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
