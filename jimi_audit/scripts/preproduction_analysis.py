#!/usr/bin/env python3
"""
Pre-production analysis: Monthly breakdown, streak analysis, position sizing models.
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
    return {"label": label, "trades": len(trades), "win_rate": round(wr*100,1), "avg_win": round(avg_win,3), "avg_loss": round(avg_loss,3),
            "expectancy": round(expectancy,4), "profit_factor": round(pf,3), "total_return": round(ret*100,2), "max_dd": round(max_dd*100,2),
            "sharpe": round(sharpe,2), "sortino": round(sortino,2), "mar": round(mar,3), "return_dd_ratio": round(rdd,2)}

# ═══════════════════════════════════════════════════════════════
# Focus on SHORT + min 6 bars (the core discovery)
# ═══════════════════════════════════════════════════════════════

core = [t for t in trades if t["direction"] == "SHORT" and t.get("bars_held", 0) >= 6]
core_all_short = [t for t in trades if t["direction"] == "SHORT"]

# Build month mapping from entry_bar
# Bars are 15min, starting 2026-04-01. ~96 bars/day, ~2920 bars/month.
# entry_bar 0 = Apr 1, entry_bar ~2920 = May 1, etc.
def get_month(t):
    bar = t.get("entry_bar", 0)
    if bar < 2920: return "Apr"
    elif bar < 5760: return "May"
    elif bar < 8600: return "Jun"
    else: return "Jul"

# ═══════════════════════════════════════════════════════════════
# A. MONTHLY BREAKDOWN
# ═══════════════════════════════════════════════════════════════

print("=" * 100)
print("A. MONTHLY BREAKDOWN (SHORT + min 6 bars)")
print("=" * 100)

months = ["Apr", "May", "Jun", "Jul"]
print(f"\n  {'Month':<8} {'Trades':>7} {'WR%':>6} {'PF':>7} {'Exp%':>8} {'Ret%':>8} {'MaxDD%':>8} {'Sharpe':>7}")
print("  " + "-" * 65)
for m in months:
    mt = [t for t in core if get_month(t) == m]
    if mt:
        s = calc_metrics(mt, m)
        print(f"  {s['label']:<8} {s['trades']:>7} {s['win_rate']:>5.1f}% {s['profit_factor']:>6.2f} {s['expectancy']:>+7.4f}% {s['total_return']:>+7.2f}% {s['max_dd']:>7.2f}% {s['sharpe']:>6.2f}")
    else:
        print(f"  {m:<8} {'0':>7}")

# Monthly trade count
print(f"\n  Monthly trade count (SHORT + min 6 bars):")
for m in months:
    mt = [t for t in core if get_month(t) == m]
    print(f"    {m}: {len(mt)} trades ({len(mt)/len(core)*100:.0f}% of total)" if core else f"    {m}: 0")

# ═══════════════════════════════════════════════════════════════
# B. STREAK ANALYSIS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("B. STREAK ANALYSIS (SHORT + min 6 bars)")
print("=" * 100)

pnls = [t["pnl_pct"] for t in core]

# Consecutive losses
max_loss_streak = 0
max_win_streak = 0
loss_streak = 0
win_streak = 0
loss_streaks = []
win_streaks = []
for p in pnls:
    if p <= 0:
        loss_streak += 1
        if win_streak > 0:
            win_streaks.append(win_streak)
        win_streak = 0
    else:
        win_streak += 1
        if loss_streak > 0:
            loss_streaks.append(loss_streak)
        loss_streak = 0
if loss_streak > 0: loss_streaks.append(loss_streak)
if win_streak > 0: win_streaks.append(win_streak)

print(f"\n  Longest losing streak: {max(loss_streaks) if loss_streaks else 0}")
print(f"  Longest winning streak: {max(win_streaks) if win_streaks else 0}")
print(f"  Avg consecutive losses: {np.mean(loss_streaks):.1f}" if loss_streaks else "  Avg consecutive losses: N/A")
print(f"  Avg consecutive wins: {np.mean(win_streaks):.1f}" if win_streaks else "  Avg consecutive wins: N/A")

# Consecutive loss distribution
print(f"\n  Consecutive Loss Distribution:")
loss_dist = defaultdict(int)
for s in loss_streaks:
    loss_dist[s] += 1
for length in sorted(loss_dist.keys()):
    print(f"    {length} consec losses: {loss_dist[length]} times")

# Worst 10-trade and 20-trade sequences
print(f"\n  Worst rolling sequences:")
if len(pnls) >= 10:
    worst_10 = min(sum(pnls[i:i+10]) for i in range(len(pnls)-9))
    worst_10_idx = min(range(len(pnls)-9), key=lambda i: sum(pnls[i:i+10]))
    print(f"    Worst 10-trade: {worst_10:+.2f}% (bars {worst_10_idx}-{worst_10_idx+9})")
if len(pnls) >= 20:
    worst_20 = min(sum(pnls[i:i+20]) for i in range(len(pnls)-19))
    worst_20_idx = min(range(len(pnls)-19), key=lambda i: sum(pnls[i:i+20]))
    print(f"    Worst 20-trade: {worst_20:+.2f}% (bars {worst_20_idx}-{worst_20_idx+19})")

# Best sequences
if len(pnls) >= 10:
    best_10 = max(sum(pnls[i:i+10]) for i in range(len(pnls)-9))
    print(f"    Best 10-trade: {best_10:+.2f}%")
if len(pnls) >= 20:
    best_20 = max(sum(pnls[i:i+20]) for i in range(len(pnls)-19))
    print(f"    Best 20-trade: {best_20:+.2f}%")

# ═══════════════════════════════════════════════════════════════
# C. POSITION SIZING MODELS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("C. POSITION SIZING MODELS (SHORT + min 6 bars)")
print("=" * 100)

def simulate_sizing(trades, sizing_fn, label):
    """Simulate equity curve with custom sizing."""
    capital = 10000.0
    peak = capital
    max_dd = 0
    equity = [capital]
    for t in trades:
        risk_pct = sizing_fn(t)
        pnl_pct = t["pnl_pct"]
        # PnL is proportional to risk
        pnl_dollar = capital * risk_pct * (pnl_pct / abs(t.get("sl_pct", 1.0)))
        capital += pnl_dollar
        peak = max(peak, capital)
        dd = (peak - capital) / peak
        max_dd = max(max_dd, dd)
        equity.append(capital)
    ret = (capital / 10000 - 1) * 100
    ann = ret * (365 / 113)
    mar = ann / (max_dd * 100) if max_dd > 0 else 0
    return {"label": label, "final_capital": round(capital, 2), "return": round(ret, 2),
            "max_dd": round(max_dd * 100, 2), "mar": round(mar, 3),
            "return_dd_ratio": round(ret / (max_dd * 100), 2) if max_dd > 0 else 0}

# Model A: Fixed 2% risk
model_a = simulate_sizing(core, lambda t: 0.02, "A: Fixed 2% risk")

# Model B: ATR-adjusted (scale by inverse ATR — lower ATR = larger size)
atr_vals = [t.get("bars_held", 10) for t in core]  # proxy: use bars held as vol proxy
median_bars = np.median(atr_vals) if atr_vals else 10
model_b = simulate_sizing(core, lambda t: 0.02 * (median_bars / max(t.get("bars_held", 10), 1)), "B: ATR-adjusted")

# Model C: Volatility targeting (target 15% annual vol)
# Simplified: scale risk inversely with recent trade volatility
recent_vol = [abs(t["pnl_pct"]) for t in core[:10]] if len(core) >= 10 else [0.5]
def vol_target(t):
    global recent_vol
    current_vol = np.std(recent_vol[-20:]) if len(recent_vol) >= 5 else 0.5
    target_vol = 0.15 / math.sqrt(365 * 6)  # per-trade target
    scale = target_vol / current_vol if current_vol > 0 else 1.0
    return min(max(0.02 * scale, 0.005), 0.05)  # cap between 0.5% and 5%
    recent_vol.append(abs(t["pnl_pct"]))
model_c = simulate_sizing(core, vol_target, "C: Vol targeting (15% ann)")

# Model D: Regime-weighted sizing
def regime_sizing(t):
    r = t.get("regime", "RANGING")
    return {"MILDLY_BEARISH": 0.03, "RANGING": 0.02, "BULL": 0.0, "BEAR": 0.01}.get(r, 0.02)
model_d = simulate_sizing(core, regime_sizing, "D: Regime-weighted")

# Model E: Regime-weighted v2 (only MB+RANGING, higher MB exposure)
core_mb_ranging = [t for t in core if t["regime"] in ("MILDLY_BEARISH", "RANGING")]
model_e = simulate_sizing(core_mb_ranging, lambda t: 0.03 if t["regime"] == "MILDLY_BEARISH" else 0.02, "E: MB+RANGING, MB@3%")

# Model F: Regime-weighted v2 with all SHORT+6 but skip BULL
core_no_bull = [t for t in core if t["regime"] != "BULL"]
model_f = simulate_sizing(core_no_bull, lambda t: 0.03 if t["regime"] == "MILDLY_BEARISH" else 0.02, "F: Skip BULL, MB@3%")

print(f"\n  {'Model':<35} {'Trades':>7} {'Return%':>9} {'MaxDD%':>8} {'R/D':>7} {'MAR':>7}")
print("  " + "-" * 75)
models = [
    (model_a, core),
    (model_b, core),
    (model_c, core),
    (model_d, core),
    (model_e, core_mb_ranging),
    (model_f, core_no_bull),
]
for m, t_list in models:
    print(f"  {m['label']:<35} {len(t_list):>7} {m['return']:>+8.2f}% {m['max_dd']:>7.2f}% {m['return_dd_ratio']:>6.2f} {m['mar']:>6.3f}")

# ═══════════════════════════════════════════════════════════════
# D. REGIME × DIRECTION MATRIX (complete)
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("D. REGIME × DIRECTION MATRIX (complete)")
print("=" * 100)

regimes = ["MILDLY_BEARISH", "RANGING", "BULL", "BEAR"]
directions = ["LONG", "SHORT"]

print(f"\n  {'Regime':<18} {'LONG PF':>10} {'LONG WR%':>10} {'LONG Exp%':>10} {'LONG n':>8} {'SHORT PF':>10} {'SHORT WR%':>10} {'SHORT Exp%':>10} {'SHORT n':>8}")
print("  " + "-" * 95)
for r in regimes:
    long_t = [t for t in trades if t["regime"] == r and t["direction"] == "LONG"]
    short_t = [t for t in trades if t["regime"] == r and t["direction"] == "SHORT"]
    l = calc_metrics(long_t, "L") if long_t else None
    s = calc_metrics(short_t, "S") if short_t else None
    l_pf = f"{l['profit_factor']:.2f}" if l else "N/A"
    l_wr = f"{l['win_rate']:.1f}%" if l else "N/A"
    l_exp = f"{l['expectancy']:+.4f}%" if l else "N/A"
    s_pf = f"{s['profit_factor']:.2f}" if s else "N/A"
    s_wr = f"{s['win_rate']:.1f}%" if s else "N/A"
    s_exp = f"{s['expectancy']:+.4f}%" if s else "N/A"
    print(f"  {r:<18} {l_pf:>10} {l_wr:>10} {l_exp:>10} {len(long_t):>8} {s_pf:>10} {s_wr:>10} {s_exp:>10} {len(short_t):>8}")

# ═══════════════════════════════════════════════════════════════
# E. EXPECTANCY CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("E. EXPECTANCY BOOTSTRAP CIs (5,000 resamples)")
print("=" * 100)

segments = [
    ("SHORT + min 6 bars (core)", core),
    ("SHORT + MB + min 6 bars", [t for t in core if t["regime"] == "MILDLY_BEARISH"]),
    ("SHORT + RANGING + min 6 bars", [t for t in core if t["regime"] == "RANGING"]),
    ("SHORT + BEAR + min 6 bars", [t for t in core if t["regime"] == "BEAR"]),
    ("LONG + BULL", [t for t in trades if t["direction"] == "LONG" and t["regime"] == "BULL"]),
    ("LONG + BEAR", [t for t in trades if t["direction"] == "LONG" and t["regime"] == "BEAR"]),
]

print(f"\n  {'Segment':<35} {'Trades':>7} {'Exp%':>8} {'CI Low':>10} {'CI High':>10} {'PF CI Low':>10} {'PF CI High':>10}")
print("  " + "-" * 95)
for name, t_list in segments:
    if len(t_list) < 5:
        continue
    pnls_s = [t["pnl_pct"] for t in t_list]
    exps = []
    pfs = []
    for _ in range(5000):
        sampled = random.choices(pnls_s, k=len(pnls_s))
        exps.append(np.mean(sampled))
        gp = sum(p for p in sampled if p > 0)
        gl = abs(sum(p for p in sampled if p <= 0))
        pfs.append(gp / gl if gl > 0 else 0)
    exps.sort()
    pfs.sort()
    n = len(exps)
    s = calc_metrics(t_list, name)
    print(f"  {name:<35} {s['trades']:>7} {s['expectancy']:>+7.4f}% {exps[int(n*0.025)]:>+9.4f}% {exps[int(n*0.975)]:>+9.4f}% {pfs[int(n*0.025)]:>9.3f} {pfs[int(n*0.975)]:>9.3f}")

# ═══════════════════════════════════════════════════════════════
# F. QUARTERLY TRADE COUNT CHECK
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print("F. TRADE FREQUENCY CHECK")
print("=" * 100)

print(f"\n  Core strategy (SHORT + min 6 bars): {len(core)} trades in 113 days")
print(f"  Average: {len(core)/113*7:.1f} trades/week, {len(core)/113*30:.1f} trades/month")
print(f"  Monthly breakdown:")
for m in months:
    mt = [t for t in core if get_month(t) == m]
    print(f"    {m}: {len(mt)} trades ({len(mt)/2920*96*30:.0f} expected/month)")

print(f"\n  Statistically: {len(core)} trades with WR {len([t for t in core if t['pnl_pct']>0])/len(core)*100:.1f}%")
se = math.sqrt((len([t for t in core if t['pnl_pct']>0])/len(core)) * (1 - len([t for t in core if t['pnl_pct']>0])/len(core)) / len(core))
print(f"  WR standard error: ±{se*100:.1f}%")
print(f"  95% CI: [{len([t for t in core if t['pnl_pct']>0])/len(core) - 1.96*se:.3f}, {len([t for t in core if t['pnl_pct']>0])/len(core) + 1.96*se:.3f}]")

print("\nDone.")
