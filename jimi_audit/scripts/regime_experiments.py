#!/usr/bin/env python3
"""
Experiments 1-4: SHORT-only, MB+RANGING, min-hold, combined.
Run on event-driven backtest trades.
"""
import json, os, math, random
from collections import defaultdict
import numpy as np

random.seed(42)

trades = json.load(open(os.path.expanduser("~/.openclaw/workspace/.openclaw/tmp/trades.json")))

def calc_metrics(trades, label=""):
    if not trades:
        return {"label": label, "trades": 0, "win_rate": 0, "expectancy": 0, "profit_factor": 0, "max_dd": 0, "total_return": 0, "sharpe": 0, "sortino": 0, "mar": 0, "avg_win": 0, "avg_loss": 0, "max_consec_loss": 0}
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
    gross_profit = sum(t["pnl_pct"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 0.001
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    std = np.std(pnls) if len(pnls) > 1 else 0.001
    sharpe = (np.mean(pnls) / std) * math.sqrt(365 * 6) if std > 0 else 0
    downside = [p for p in pnls if p < 0]
    ds_std = np.std(downside) if len(downside) > 1 else 0.001
    sortino = (np.mean(pnls) / ds_std) * math.sqrt(365 * 6) if ds_std > 0 else 0
    total_ret = equity[-1] / equity[0] - 1
    annual_ret = total_ret * (365 / 113)
    mar = annual_ret / max_dd if max_dd > 0 else 0
    max_consec = 0
    consec = 0
    for t in trades:
        if t["pnl_pct"] <= 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0
    return {
        "label": label, "trades": len(trades), "win_rate": round(wr * 100, 1),
        "avg_win": round(avg_win, 3), "avg_loss": round(avg_loss, 3),
        "expectancy": round(expectancy, 4), "profit_factor": round(pf, 3),
        "total_return": round(total_ret * 100, 2), "max_dd": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2), "sortino": round(sortino, 2), "mar": round(mar, 3),
        "max_consec_loss": max_consec,
    }

def monte_carlo(trades, n_sims=10000, horizon_days=30):
    returns = [t["pnl_pct"] / 100 for t in trades]
    if len(returns) < 2:
        return None
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
    return {
        "horizon": horizon_days, "trades_per_sim": n_trades,
        "p5": round(np.percentile(finals, 5) * 100, 2),
        "p25": round(np.percentile(finals, 25) * 100, 2),
        "p50": round(np.percentile(finals, 50) * 100, 2),
        "p75": round(np.percentile(finals, 75) * 100, 2),
        "p95": round(np.percentile(finals, 95) * 100, 2),
        "mean": round(np.mean(finals) * 100, 2),
        "prob_loss": round(sum(1 for f in finals if f < 0) / n * 100, 1),
        "max_dd_p50": round(np.median(max_dds) * 100, 2),
        "max_dd_p95": round(np.percentile(max_dds, 95) * 100, 2),
    }

# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

experiments = {
    "BASELINE (all)": trades,
    "EXP 1: SHORT-only": [t for t in trades if t["direction"] == "SHORT"],
    "EXP 2: MB+RANGING only": [t for t in trades if t["regime"] in ("MILDLY_BEARISH", "RANGING")],
    "EXP 3: Min 6-bar hold": [t for t in trades if t.get("bars_held", 0) >= 6],
    "EXP 4a: SHORT + MB/RANGING": [t for t in trades if t["direction"] == "SHORT" and t["regime"] in ("MILDLY_BEARISH", "RANGING")],
    "EXP 4b: SHORT + min 6 bars": [t for t in trades if t["direction"] == "SHORT" and t.get("bars_held", 0) >= 6],
    "EXP 4c: MB/RANGING + min 6 bars": [t for t in trades if t["regime"] in ("MILDLY_BEARISH", "RANGING") and t.get("bars_held", 0) >= 6],
    "EXP 4d: SHORT + MB/RANGING + 6bars": [t for t in trades if t["direction"] == "SHORT" and t["regime"] in ("MILDLY_BEARISH", "RANGING") and t.get("bars_held", 0) >= 6],
    "EXP 5: LONG-only (control)": [t for t in trades if t["direction"] == "LONG"],
    "EXP 6: Disable BEAR": [t for t in trades if t["regime"] != "BEAR"],
    "EXP 7: SHORT + disable BEAR": [t for t in trades if t["direction"] == "SHORT" and t["regime"] != "BEAR"],
}

# Walk-forward: train Apr-May, test Jun-Jul
train_cutoff = "2026-06-01"
train = [t for t in trades if t.get("entry_bar", 0) < 7000]  # ~first 2 months
test = [t for t in trades if t.get("entry_bar", 0) >= 7000]

print("=" * 110)
print("EXPERIMENT RESULTS")
print("=" * 110)

print(f"\n{'Experiment':<35} {'Trades':>7} {'WR%':>6} {'Exp%':>8} {'PF':>7} {'MaxDD%':>8} {'Ret%':>8} {'Sharpe':>7} {'Sortino':>8} {'MAR':>7} {'MaxCL':>6}")
print("-" * 110)
for name, t_list in experiments.items():
    if not t_list:
        print(f"  {name:<35} {'0':>7}")
        continue
    s = calc_metrics(t_list, name)
    print(f"  {s['label']:<35} {s['trades']:>7} {s['win_rate']:>5.1f}% {s['expectancy']:>+7.4f}% {s['profit_factor']:>6.2f} {s['max_dd']:>7.2f}% {s['total_return']:>+7.2f}% {s['sharpe']:>6.2f} {s['sortino']:>7.2f} {s['mar']:>6.3f} {s['max_consec_loss']:>5}")

# Walk-forward
print(f"\n{'='*110}")
print("WALK-FORWARD VALIDATION")
print("=" * 110)
print(f"\n  Train (Apr-May): {len(train)} trades")
for name in ["BASELINE (all)", "EXP 1: SHORT-only", "EXP 2: MB+RANGING only", "EXP 4d: SHORT + MB/RANGING + 6bars"]:
    base_exp = experiments[name]
    train_t = [t for t in base_exp if t.get("entry_bar", 0) < 7000]
    test_t = [t for t in base_exp if t.get("entry_bar", 0) >= 7000]
    if train_t and test_t:
        tr = calc_metrics(train_t, f"{name} [train]")
        te = calc_metrics(test_t, f"{name} [test]")
        print(f"\n  {name}:")
        print(f"    Train: {tr['trades']} trades  WR={tr['win_rate']}%  PF={tr['profit_factor']:.2f}  Exp={tr['expectancy']:+.4f}%")
        print(f"    Test:  {te['trades']} trades  WR={te['win_rate']}%  PF={te['profit_factor']:.2f}  Exp={te['expectancy']:+.4f}%")
        pf_decay = (te['profit_factor'] - tr['profit_factor']) / tr['profit_factor'] * 100 if tr['profit_factor'] > 0 else 0
        print(f"    PF decay: {pf_decay:+.1f}% {'OK' if pf_decay > -30 else 'WARNING: overfitting risk'}")

# Monte Carlo for key experiments
print(f"\n{'='*110}")
print("MONTE CARLO (10,000 sims)")
print("=" * 110)

key_exps = ["BASELINE (all)", "EXP 1: SHORT-only", "EXP 2: MB+RANGING only", "EXP 4d: SHORT + MB/RANGING + 6bars"]
for name in key_exps:
    t_list = experiments[name]
    if len(t_list) < 5:
        continue
    print(f"\n  {name} ({len(t_list)} trades):")
    for h in [30, 90]:
        mc = monte_carlo(t_list, 10000, h)
        if mc:
            print(f"    {h}-day: P5={mc['p5']:+.1f}% P25={mc['p25']:+.1f}% P50={mc['p50']:+.1f}% P75={mc['p75']:+.1f}% P95={mc['p95']:+.1f}%  P(loss)={mc['prob_loss']:.1f}%  MaxDD P50={mc['max_dd_p50']:.1f}% P95={mc['max_dd_p95']:.1f}%")

# Consecutive loss analysis for best combo
print(f"\n{'='*110}")
print("RISK PROFILE: EXP 4d (SHORT + MB/RANGING + 6 bars)")
print("=" * 110)
best = experiments["EXP 4d: SHORT + MB/RANGING + 6bars"]
if best:
    # Consecutive losses
    consec_counts = {}
    consec = 0
    for t in best:
        if t["pnl_pct"] <= 0:
            consec += 1
        else:
            if consec > 0:
                consec_counts[consec] = consec_counts.get(consec, 0) + 1
            consec = 0
    if consec > 0:
        consec_counts[consec] = consec_counts.get(consec, 0) + 1
    print(f"\n  Consecutive Loss Distribution:")
    for length in sorted(consec_counts.keys()):
        print(f"    {length} consec losses: {consec_counts[length]} times")

    # By regime within combo
    print(f"\n  Performance by Regime (within combo):")
    for r in ["MILDLY_BEARISH", "RANGING", "BULL", "BEAR"]:
        rt = [t for t in best if t["regime"] == r]
        if rt:
            s = calc_metrics(rt, r)
            print(f"    {r:<18} trades={s['trades']:>3}  WR={s['win_rate']}%  PF={s['profit_factor']:.2f}  Exp={s['expectancy']:+.4f}%")

# Improvement summary
print(f"\n{'='*110}")
print("IMPROVEMENT SUMMARY")
print("=" * 110)

baseline = calc_metrics(trades, "BASELINE")
best_exp = experiments["EXP 4d: SHORT + MB/RANGING + 6bars"]
if best_exp:
    best_m = calc_metrics(best_exp, "BEST COMBO")
    print(f"\n  {'Metric':<25} {'Baseline':>12} {'Best Combo':>12} {'Change':>12}")
    print(f"  {'-'*65}")
    for key in ["trades", "win_rate", "expectancy", "profit_factor", "total_return", "max_dd", "sharpe", "sortino", "mar", "max_consec_loss"]:
        b = baseline.get(key, 0)
        c = best_m.get(key, 0)
        if key == "trades":
            delta = f"{c - b:+d}"
        elif isinstance(b, float):
            delta = f"{c - b:+.2f}"
        else:
            delta = f"{c - b:+d}"
        print(f"  {key:<25} {str(b):>12} {str(c):>12} {delta:>12}")

print("\nDone.")
