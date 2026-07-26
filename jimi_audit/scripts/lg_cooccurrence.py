import pandas as pd
import numpy as np
from scipy import stats
import json, os

DATA_DIR = "/root/.openclaw/workspace/jimi_audit/data"
DERIV_DIR = f"{DATA_DIR}/derivatives_history"

ohlcv = pd.read_csv(f"{DATA_DIR}/eth_15m_extended.csv")
ohlcv["timestamp"] = pd.to_datetime(ohlcv["Open time"])
ohlcv = ohlcv.sort_values("timestamp").reset_index(drop=True)

deriv = pd.read_csv(f"{DERIV_DIR}/derivatives_collected.csv")
deriv["timestamp"] = pd.to_datetime(deriv["timestamp"], format="mixed", utc=True).dt.tz_localize(None)
deriv = deriv.sort_values("timestamp").reset_index(drop=True)

merged = pd.merge_asof(ohlcv, deriv[["timestamp","oi","ls_ratio","funding_rate"]],
                       on="timestamp", direction="backward", tolerance=pd.Timedelta("30min"))
merged["vol_ratio"] = merged["Volume"] / merged["Volume"].rolling(20).mean()
merged["ema200"] = merged["Close"].ewm(span=200).mean()
merged["trend"] = np.where(merged["Close"] > merged["ema200"], "BULL", "BEAR")
for h in [4, 8, 16, 24]:
    merged["fwd_ret_" + str(h)] = merged["Close"].shift(-h) / merged["Close"] - 1

closes = merged["Close"].values
volumes = merged["Volume"].values
taker_base = merged["Taker buy base asset volume"].values
n = len(merged)

print("Precomputing taker z-scores...")
taker_zscores = np.full(n, np.nan)
for idx in range(60, n):
    recent_buy = np.sum(taker_base[idx-4:idx])
    recent_total = np.sum(volumes[idx-4:idx])
    if recent_total == 0:
        continue
    taker_ratio = recent_buy / recent_total
    window_buy = taker_base[max(0, idx-60):idx]
    window_total = volumes[max(0, idx-60):idx]
    window_ratios = []
    for j in range(0, len(window_buy)-4, 4):
        wb = np.sum(window_buy[j:j+4])
        wt = np.sum(window_total[j:j+4])
        if wt > 0:
            window_ratios.append(wb / wt)
    if len(window_ratios) >= 5:
        mean_r = np.mean(window_ratios)
        std_r = np.std(window_ratios)
        if std_r > 0:
            taker_zscores[idx] = (taker_ratio - mean_r) / std_r
print("Computed z-scores for", int(np.sum(~np.isnan(taker_zscores))), "bars")

sig_path = DATA_DIR + "/../live/data/strategy_signals.jsonl"
group_a_signals = []
if os.path.exists(sig_path):
    with open(sig_path) as f:
        for line in f:
            try:
                s = json.loads(line)
                strat = s.get("strategy", "")
                if strat in ["trade_flow","orderbook_imbalance","funding_arb","judas_sweep","liquidation_cascade","failed_breakout","positioning_fade","whale_watch"]:
                    group_a_signals.append(s)
            except:
                pass
print("Group A signals from jsonl:", len(group_a_signals))

if len(group_a_signals) < 10:
    print("Not enough live signals. Generating synthetic Group A signals...")
    for idx in range(200, n):
        ema20 = np.mean(closes[idx-20:idx])
        ema50 = np.mean(closes[idx-50:idx])
        vr = merged.iloc[idx]["vol_ratio"] or 1.0
        if ema20 > ema50 and vr > 1.2:
            direction = "LONG"
        elif ema20 < ema50 and vr > 1.2:
            direction = "SHORT"
        else:
            continue
        group_a_signals.append({"strategy":"synthetic_trend","timestamp":str(merged.iloc[idx]["timestamp"]),"direction":direction,"conviction":0.6,"idx":idx})
    print("Generated", len(group_a_signals), "synthetic signals")

matched_signals = []
for sig in group_a_signals:
    ts_str = sig.get("timestamp", "")
    if not ts_str:
        continue
    try:
        ts = pd.to_datetime(ts_str)
        mask = merged["timestamp"] == ts
        if mask.any():
            idx = merged[mask].index[0]
            sig["idx"] = idx
            matched_signals.append(sig)
        else:
            time_diff = abs(merged["timestamp"] - ts)
            closest = time_diff.idxmin()
            if time_diff[closest] < pd.Timedelta("15min"):
                sig["idx"] = closest
                matched_signals.append(sig)
    except:
        continue
print("Matched signals:", len(matched_signals))

print("\n" + "="*70)
print("CO-OCCURRENCE: Group A + taker confirmation")
print("="*70)

all_rets = []
taker_confirm_rets = []
taker_oppose_rets = []
vol_confirm_rets = []
all_confirm_rets = []

for sig in matched_signals:
    idx = sig["idx"]
    direction = sig.get("direction", "")
    if not direction or idx + 16 >= n:
        continue
    ret = merged.iloc[idx]["fwd_ret_16"]
    if pd.isna(ret):
        continue
    dir_mult = 1 if direction == "LONG" else -1
    adj_ret = ret * dir_mult
    all_rets.append(adj_ret)
    tz = taker_zscores[idx] if idx < len(taker_zscores) else np.nan
    if not np.isnan(tz):
        taker_confirms = (direction == "LONG" and tz > 2.0) or (direction == "SHORT" and tz < -2.0)
        taker_opposes = (direction == "LONG" and tz < -2.0) or (direction == "SHORT" and tz > 2.0)
        if taker_confirms:
            taker_confirm_rets.append(adj_ret)
        if taker_opposes:
            taker_oppose_rets.append(adj_ret)
    vr = merged.iloc[idx]["vol_ratio"] or 1.0
    if vr > 1.3:
        vol_confirm_rets.append(adj_ret)
    if not np.isnan(tz) and (direction == "LONG" and tz > 2.0) or (direction == "SHORT" and tz < -2.0):
        if vr > 1.3:
            all_confirm_rets.append(adj_ret)

def print_stats(name, rets):
    rets = np.array(rets)
    if len(rets) < 3:
        return
    wr = (rets > 0).mean()
    mean_r = rets.mean()
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    print(name.ljust(30), "n=" + str(len(rets)).rjust(5), "WR=" + str(round(wr*100,1)).rjust(5) + "%", "mean=" + ("+" if mean_r >= 0 else "") + str(round(mean_r*100,3)).rjust(7) + "%", "PF=" + str(round(pf,2)))

print_stats("All Group A", all_rets)
print_stats("Taker z>2.0 confirms", taker_confirm_rets)
print_stats("Taker z>2.0 opposes", taker_oppose_rets)
print_stats("Vol ratio > 1.3", vol_confirm_rets)
print_stats("All 3 confirms", all_confirm_rets)

all_rets = np.array(all_rets)
confirm_rets = np.array(taker_confirm_rets)
oppose_rets = np.array(taker_oppose_rets)

if len(confirm_rets) >= 5 and len(all_rets) >= 5:
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON")
    print("="*70)
    t, p = stats.ttest_ind(confirm_rets, all_rets)
    print("Confirmed vs All: t=" + str(round(t,3)) + " p=" + str(round(p,4)))
    print("  Confirmed: n=" + str(len(confirm_rets)) + " WR=" + str(round((confirm_rets>0).mean()*100,1)) + "% mean=" + ("+" if confirm_rets.mean() >= 0 else "") + str(round(confirm_rets.mean()*100,3)) + "%")
    print("  All:       n=" + str(len(all_rets)) + " WR=" + str(round((all_rets>0).mean()*100,1)) + "% mean=" + ("+" if all_rets.mean() >= 0 else "") + str(round(all_rets.mean()*100,3)) + "%")
    diffs = []
    for _ in range(5000):
        s1 = np.random.choice(confirm_rets, size=len(confirm_rets), replace=True)
        s2 = np.random.choice(all_rets, size=len(all_rets), replace=True)
        diffs.append(s1.mean() - s2.mean())
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    print("  Bootstrap diff CI: [" + ("+" if ci_lo >= 0 else "") + str(round(ci_lo*100,3)) + "%, +" + str(round(ci_hi*100,3)) + "%]")
    print("  CI excludes 0:", "YES" if ci_lo > 0 else "NO")

if len(oppose_rets) >= 3:
    print("\nOpposing taker: n=" + str(len(oppose_rets)) + " WR=" + str(round((oppose_rets>0).mean()*100,1)) + "% mean=" + ("+" if oppose_rets.mean() >= 0 else "") + str(round(oppose_rets.mean()*100,3)) + "%")

print("\n" + "="*70)
print("VERDICT")
print("="*70)
if len(confirm_rets) >= 5:
    all_wr = (all_rets > 0).mean()
    conf_wr = (confirm_rets > 0).mean()
    wr_lift = conf_wr - all_wr
    conf_mean = confirm_rets.mean()
    all_mean = all_rets.mean()
    mean_lift = conf_mean - all_mean
    print("Win rate lift: " + ("+" if wr_lift >= 0 else "") + str(round(wr_lift*100,1)) + "% (" + str(round(all_wr*100,1)) + "% -> " + str(round(conf_wr*100,1)) + "%)")
    print("Mean return lift: " + ("+" if mean_lift >= 0 else "") + str(round(mean_lift*100,3)) + "% (" + ("+" if all_mean >= 0 else "") + str(round(all_mean*100,3)) + "% -> " + ("+" if conf_mean >= 0 else "") + str(round(conf_mean*100,3)) + "%)")
    if wr_lift > 0.03 and conf_wr > 0.52:
        print("RESULT: taker confirmation IMPROVES Group A -> Deploy as Group B")
    elif wr_lift > 0:
        print("RESULT: SMALL positive effect -> Marginal")
    else:
        print("RESULT: taker confirmation does NOT improve Group A -> Do not deploy")
else:
    print("Not enough confirmed signals (" + str(len(confirm_rets)) + "), need >= 5")
print("\nDone.")
