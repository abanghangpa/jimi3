#!/usr/bin/env python3
"""
Conditional Directional Model — The real discovery.
Experiments A, B, C + walk-forward + Monte Carlo + bootstrap.
"""
import json, os, math, random
from collections import defaultdict
import numpy as np

random.seed(42)
np.random.seed(42)

trades = json.load(open(os.path.expanduser("~/.openclaw/workspace/.openclaw/tmp/trades.json")))

def calc_metrics(trades, label=""):
    if not trades:
        return {"label": label, "trades": 0, "win_rate": 0, "expectancy": 0, "profit_factor": 0, "max_dd": 0, "total_return": 0, "sharpe": 0, "sortino": 0, "mar": 0, "return_dd_ratio": 0, "avg_win": 0, "avg_loss": 0}
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    pnls = [t["pnl_pct"] for t in trades]
    equity = [10000.0]
    for p in pnls:
        equity.append(equity[-1] * (1 + p / 100))
    peak = equity[0]
    max_dd = 0
    for e in equity:
        peak = max(peak, e)
        dd = (peak - e) / peak
        max_dd = max(max_dd, dd)
    wr = len(wins) / len(trades)
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    expectancy = wr * avg_win + (1 - wr) * avg_loss
    gp = sum(t["pnl_pct"] for t in wins) if wins else 0
    gl = abs(sum(t["pnl_pct"] for t in losses)) if losses else 0.001
    pf = gp / gl if gl > 0 else float("inf")
    std = np.std(pnls) if len(pnls) > 1 else 0.001
    sharpe = (np.mean(pnls) / std) * math.sqrt(365 * 6) if std > 0 else 0
    ds = [p for p in pnls if p < 0]
    ds_std = np.std(ds) if len(ds) > 1 else 0.001
    sortino = (np.mean(pnls) / ds_std) * math.sqrt(365 * 6) if ds_std > 0 else 0
    ret = equity[-1] / equity[0] - 1
    ann = ret * (365 / 113)
    mar = ann / max_dd if max_dd > 0 else 0
    rdd = ret / max_dd if max_dd > 0 else 0
    max_consec = 0
    consec = 0
    for t in trades:
        if t["pnl_pct"] <= 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0
    return {"label": label, "trades": len(trades), "win_rate": round(wr*100,1), "avg_win": round(avg_win,3), "avg_loss": round(avg_loss,3),
            "expectancy": round(expectancy,4), "profit_factor": round(pf,3), "total_return": round(ret*100,2), "max_dd": round(max_dd*100,2),
            "sharpe": round(sharpe,2), "sortino": round(sortino,2), "mar": round(mar,3), "return_dd_ratio": round(rdd,2), "max_consec_loss": max_consec}

def monte_carlo(trades, n_sims=10000, horizon_days=30):
    returns = [t["pnl_pct"] / 100 for t in trades]
    if len(returns) < 2: return None
    trades_per_day = len(returns) / 113
    n_trades = max(1, int(horizon_days * trades_per_day))
    finals = []
    max_dds = []
    for _ in range(n_sims):
        sampled = random.choices(returns, k=n_trades)
        cum = 1.0
        peak = cum
        mdd = 0
        for r in sampled:
            cum *= (1 + r)
            peak = max(peak, cum)
            dd = (peak - cum) / peak
            mdd = max(mdd, dd)
        finals.append(cum - 1)
        max_dds.append(mdd)
    finals.sort()
    n = len(finals)
    return {"horizon": horizon_days, "trades_per_sim": n_trades,
            "p5": round(np.percentile(finals, 5)*100,2), "p50": round(np.percentile(finals, 50)*100,2),
            "p95": round(np.percentile(finals, 95)*100,2), "prob_loss": round(sum(1 for f in finals if f < 0)/n*100,1),
            "max_dd_p50": round(np.median(max_dds)*100,2), "max_dd_p95": round(np.percentile(max_dds, 95)*100,2)}

def bootstrap_ci(trades, n_boot=5000):
    pnls = [t["pnl_pct"] for t in trades]
    if len(pnls) < 5: return None
    exps = []
    pfs = []
    for _ in range(n_boot):
        sampled = random.choices(pnls, k=len(pnls))
        exps.append(np.mean(sampled))
        gp = sum(p for p in sampled if p > 0)
        gl = abs(sum(p for p in sampled if p <= 0))
        pfs.append(gp / gl if gl > 0 else 0)
    exps.sort()
    pfs.sort()
    n = len(exps)
    return {"exp_ci": (round(exps[int(n*0.025)],4), round(exps[int(n*0.975)],4)),
            "pf_ci": (round(pfs[int(n*0.025)],3), round(pfs[int(n*0.975)],3))}

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT A: Hard Directional Gating
# ═══════════════════════════════════════════════════════════════

print("=" * 100)
print("EXPERIMENT A: HARD DIRECTIONAL GATING")
print("=" * 100)

def allow_trade(t, mode):
    """Directional gating logic."""
    r = t.get("regime", "RANGING")
    d = t.get("direction", "LONG")
    if mode == "conditional":
        # BULL → LONG only, BEAR → SHORT only, MB → both, RANGING → both
        if r == "BULL" and d != "LONG": return False
        if r == "BEAR" and d != "SHORT": return False
        return True
    elif mode == "conditional_strict":
        # Same as conditional but skip RANGING
        if r == "BULL" and d != "LONG": return False
        if r == "BEAR" and d != "SHORT": return False
        if r == "RANGING": return False
        return True
    elif mode == "high_exp_only":
        # Only the 4 high-expectancy quadrants
        return (r, d) in [("MILDLY_BEARISH", "LONG"), ("MILDLY_BEARISH", "SHORT"), ("BULL", "LONG"), ("BEAR", "SHORT")]
    elif mode == "high_exp_plus_ranging":
        # 4 quadrants + both RANGING
        if r == "RANGING": return True
        return (r, d) in [("MILDLY_BEARISH", "LONG"), ("MILDLY_BEARISH", "SHORT"), ("BULL", "LONG"), ("BEAR", "SHORT")]
    return True

experiments_a = {
    "BASELINE": lambda t: True,
    "A1: Conditional (BULL→L, BEAR→S, MB→both, RANG→both)": lambda t: allow_trade(t, "conditional"),
    "A2: Conditional strict (skip RANGING)": lambda t: allow_trade(t, "conditional_strict"),
    "A3: High-exp quadrants only": lambda t: allow_trade(t, "high_exp_only"),
    "A4: High-exp + RANGING": lambda t: allow_trade(t, "high_exp_plus_ranging"),
    "A5: SHORT + min 6 bars (reference)": lambda t: t["direction"] == "SHORT" and t.get("bars_held", 0) >= 6,
}

# Also add min 6-bar versions
experiments_a6 = {
    "A6: Conditional + min 6 bars": lambda t: allow_trade(t, "conditional") and t.get("bars_held", 0) >= 6,
    "A7: Conditional strict + min 6 bars": lambda t: allow_trade(t, "conditional_strict") and t.get("bars_held", 0) >= 6,
    "A8: High-exp + min 6 bars": lambda t: allow_trade(t, "high_exp_only") and t.get("bars_held", 0) >= 6,
    "A9: High-exp + RANGING + min 6 bars": lambda t: allow_trade(t, "high_exp_plus_ranging") and t.get("bars_held", 0) >= 6,
}

all_exps = {**experiments_a, **experiments_a6}

print(f"\n  {'Experiment':<50} {'n':>5} {'WR%':>6} {'PF':>7} {'Exp%':>8} {'MaxDD%':>8} {'Ret%':>8} {'R/D':>7} {'Sharpe':>7} {'MC90 P50':>9} {'MC90 P(loss)':>12}")
print("  " + "-" * 135)

for name, filt in all_exps.items():
    t_list = [t for t in trades if filt(t)]
    if len(t_list) < 5:
        print(f"  {name:<50} {len(t_list):>5}")
        continue
    s = calc_metrics(t_list, name)
    mc = monte_carlo(t_list, 10000, 90)
    mc_p50 = f"{mc['p50']:+.1f}%" if mc else "N/A"
    mc_loss = f"{mc['prob_loss']:.1f}%" if mc else "N/A"
    print(f"  {s['label']:<50} {s['trades']:>5} {s['win_rate']:>5.1f}% {s['profit_factor']:>6.2f} {s['expectancy']:>+7.4f}% {s['max_dd']:>7.2f}% {s['total_return']:>+7.2f}% {s['return_dd_ratio']:>6.2f} {s['sharpe']:>6.2f} {mc_p50:>9} {mc_loss:>12}")

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT B: Remove Weak Segments
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("EXPERIMENT B: REMOVE WEAK SEGMENTS")
print("=" * 100)

# Keep only: LONG+BULL, LONG+MB, SHORT+BEAR, SHORT+MB
strong_quadrants = [("MILDLY_BEARISH", "LONG"), ("MILDLY_BEARISH", "SHORT"), ("BULL", "LONG"), ("BEAR", "SHORT")]

exp_b = {
    "B1: Strong 4 quadrants": [t for t in trades if (t["regime"], t["direction"]) in strong_quadrants],
    "B2: Strong 4 + min 6 bars": [t for t in trades if (t["regime"], t["direction"]) in strong_quadrants and t.get("bars_held", 0) >= 6],
    "B3: Remove LONG+BEAR + SHORT+BULL": [t for t in trades if not (t["regime"] == "BEAR" and t["direction"] == "LONG") and not (t["regime"] == "BULL" and t["direction"] == "SHORT")],
    "B4: B3 + min 6 bars": [t for t in trades if not (t["regime"] == "BEAR" and t["direction"] == "LONG") and not (t["regime"] == "BULL" and t["direction"] == "SHORT") and t.get("bars_held", 0) >= 6],
}

print(f"\n  {'Experiment':<50} {'n':>5} {'WR%':>6} {'PF':>7} {'Exp%':>8} {'MaxDD%':>8} {'Ret%':>8} {'R/D':>7}")
print("  " + "-" * 105)
for name, t_list in exp_b.items():
    if len(t_list) < 5:
        print(f"  {name:<50} {len(t_list):>5}")
        continue
    s = calc_metrics(t_list, name)
    print(f"  {s['label']:<50} {s['trades']:>5} {s['win_rate']:>5.1f}% {s['profit_factor']:>6.2f} {s['expectancy']:>+7.4f}% {s['max_dd']:>7.2f}% {s['total_return']:>+7.2f}% {s['return_dd_ratio']:>6.2f}")

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT C: Confidence-Weighted Allocation
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("EXPERIMENT C: CONFIDENCE-WEIGHTED SIZING")
print("=" * 100)

def get_risk_multiplier(t):
    """Regime × direction risk multiplier."""
    r = t.get("regime", "RANGING")
    d = t.get("direction", "LONG")
    weights = {
        ("BULL", "LONG"): 1.5,
        ("BULL", "SHORT"): 0.0,  # block
        ("BEAR", "SHORT"): 1.5,
        ("BEAR", "LONG"): 0.0,  # block
        ("MILDLY_BEARISH", "LONG"): 1.2,
        ("MILDLY_BEARISH", "SHORT"): 1.2,
        ("RANGING", "LONG"): 0.5,
        ("RANGING", "SHORT"): 0.5,
    }
    return weights.get((r, d), 0.5)

def simulate_weighted(trades, label, min_bars=0):
    """Simulate with regime-weighted sizing."""
    filtered = [t for t in trades if t.get("bars_held", 0) >= min_bars and get_risk_multiplier(t) > 0]
    if not filtered:
        return None
    capital = 10000.0
    peak = capital
    max_dd = 0
    pnls = []
    for t in filtered:
        mult = get_risk_multiplier(t)
        pnl = t["pnl_pct"] * mult
        capital *= (1 + pnl / 100)
        peak = max(peak, capital)
        dd = (peak - capital) / peak
        max_dd = max(max_dd, dd)
        pnls.append(pnl)
    ret = (capital / 10000 - 1) * 100
    ann = ret * (365 / 113)
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gp / gl if gl > 0 else float("inf")
    wr = sum(1 for p in pnls if p > 0) / len(pnls)
    std = np.std(pnls) if len(pnls) > 1 else 0.001
    sharpe = (np.mean(pnls) / std) * math.sqrt(365 * 6) if std > 0 else 0
    return {"label": label, "trades": len(filtered), "win_rate": round(wr*100,1),
            "profit_factor": round(pf,3), "total_return": round(ret,2), "max_dd": round(max_dd*100,2),
            "sharpe": round(sharpe,2), "mar": round(ann/max_dd,3) if max_dd > 0 else 0,
            "return_dd_ratio": round(ret/(max_dd*100),2) if max_dd > 0 else 0}

exp_c = [
    simulate_weighted(trades, "C1: Weighted (all trades)"),
    simulate_weighted(trades, "C2: Weighted + min 6 bars", min_bars=6),
]

print(f"\n  {'Model':<50} {'n':>5} {'WR%':>6} {'PF':>7} {'Ret%':>8} {'MaxDD%':>8} {'R/D':>7} {'MAR':>7}")
print("  " + "-" * 95)
for m in exp_c:
    if m:
        print(f"  {m['label']:<50} {m['trades']:>5} {m['win_rate']:>5.1f}% {m['profit_factor']:>6.2f} {m['total_return']:>+7.2f}% {m['max_dd']:>7.2f}% {m['return_dd_ratio']:>6.2f} {m['mar']:>6.3f}")

# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD + MONTE CARLO + BOOTSTRAP for top configs
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("WALK-FORWARD + MONTE CARLO + BOOTSTRAP (top configurations)")
print("=" * 100)

top_configs = {
    "Conditional + min 6 bars": [t for t in trades if allow_trade(t, "conditional") and t.get("bars_held", 0) >= 6],
    "High-exp quadrants + min 6 bars": [t for t in trades if (t["regime"], t["direction"]) in strong_quadrants and t.get("bars_held", 0) >= 6],
    "B4: Remove killers + min 6 bars": [t for t in trades if not (t["regime"] == "BEAR" and t["direction"] == "LONG") and not (t["regime"] == "BULL" and t["direction"] == "SHORT") and t.get("bars_held", 0) >= 6],
    "SHORT + min 6 bars (reference)": [t for t in trades if t["direction"] == "SHORT" and t.get("bars_held", 0) >= 6],
}

for name, t_list in top_configs.items():
    if len(t_list) < 10:
        continue
    s = calc_metrics(t_list, name)
    ci = bootstrap_ci(t_list)
    
    # Walk-forward
    train = [t for t in t_list if t.get("entry_bar", 0) < 7000]
    test = [t for t in t_list if t.get("entry_bar", 0) >= 7000]
    tr = calc_metrics(train, "train") if len(train) >= 5 else None
    te = calc_metrics(test, "test") if len(test) >= 5 else None
    
    # Monte Carlo
    mc30 = monte_carlo(t_list, 10000, 30)
    mc90 = monte_carlo(t_list, 10000, 90)
    
    print(f"\n  {name}")
    print(f"    Trades: {s['trades']}  WR: {s['win_rate']}%  PF: {s['profit_factor']:.2f}  Exp: {s['expectancy']:+.4f}%  MaxDD: {s['max_dd']:.2f}%  R/D: {s['return_dd_ratio']:.2f}")
    if ci:
        print(f"    Bootstrap: Exp CI [{ci['exp_ci'][0]:+.4f}%, {ci['exp_ci'][1]:+.4f}%]  PF CI [{ci['pf_ci'][0]:.3f}, {ci['pf_ci'][1]:.3f}]")
    if tr and te:
        decay = (te['profit_factor'] - tr['profit_factor']) / tr['profit_factor'] * 100 if tr['profit_factor'] > 0 else 0
        print(f"    Walk-forward: Train PF={tr['profit_factor']:.2f} ({tr['trades']} trades) → Test PF={te['profit_factor']:.2f} ({te['trades']} trades)  Decay: {decay:+.1f}%")
    if mc30:
        print(f"    MC 30-day: P50={mc30['p50']:+.1f}%  P(loss)={mc30['prob_loss']:.1f}%  MaxDD P50={mc30['max_dd_p50']:.1f}%")
    if mc90:
        print(f"    MC 90-day: P50={mc90['p50']:+.1f}%  P(loss)={mc90['prob_loss']:.1f}%  MaxDD P95={mc90['max_dd_p95']:.1f}%")

# ═══════════════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("FINAL COMPARISON")
print("=" * 100)

final = {
    "BASELINE": trades,
    "SHORT + min 6 bars": [t for t in trades if t["direction"] == "SHORT" and t.get("bars_held", 0) >= 6],
    "Conditional + min 6 bars": [t for t in trades if allow_trade(t, "conditional") and t.get("bars_held", 0) >= 6],
    "High-exp + min 6 bars": [t for t in trades if (t["regime"], t["direction"]) in strong_quadrants and t.get("bars_held", 0) >= 6],
    "Remove killers + min 6 bars": [t for t in trades if not (t["regime"] == "BEAR" and t["direction"] == "LONG") and not (t["regime"] == "BULL" and t["direction"] == "SHORT") and t.get("bars_held", 0) >= 6],
}

print(f"\n  {'Config':<35} {'n':>5} {'WR%':>6} {'PF':>7} {'Exp%':>8} {'MaxDD%':>8} {'Ret%':>8} {'R/D':>7} {'Sharpe':>7} {'MC90 P50':>9} {'MC90 P(loss)':>12}")
print("  " + "-" * 130)
for name, t_list in final.items():
    s = calc_metrics(t_list, name)
    mc = monte_carlo(t_list, 10000, 90)
    mc_p50 = f"{mc['p50']:+.1f}%" if mc else "N/A"
    mc_loss = f"{mc['prob_loss']:.1f}%" if mc else "N/A"
    print(f"  {name:<35} {s['trades']:>5} {s['win_rate']:>5.1f}% {s['profit_factor']:>6.2f} {s['expectancy']:>+7.4f}% {s['max_dd']:>7.2f}% {s['total_return']:>+7.2f}% {s['return_dd_ratio']:>6.2f} {s['sharpe']:>6.2f} {mc_p50:>9} {mc_loss:>12}")

print("\nDone.")
