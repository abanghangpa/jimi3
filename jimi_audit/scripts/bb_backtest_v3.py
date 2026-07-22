"""
Backtest: BB Regime Reversion v3 (weighted confirmations, strict entry)
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd
import numpy as np
from strategies.bb_regime_reversion_v3 import BBRegimeReversionV3

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eth_15m_merged.csv")

def load_data():
    df = pd.read_csv(DATA_FILE)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            for c in df.columns:
                if col.lower() in c.lower():
                    df[col] = df[c]; break
    return df

def classify_regime(df, idx, lookback=20):
    if idx < lookback: return "RANGING", 0.5
    closes = df['Close'].iloc[idx-lookback:idx+1].values
    highs = df['High'].iloc[idx-lookback:idx+1].values
    lows = df['Low'].iloc[idx-lookback:idx+1].values
    ema20 = pd.Series(closes).ewm(span=20).mean().iloc[-1]
    ema50 = pd.Series(closes).ewm(span=min(50, lookback)).mean().iloc[-1]
    price = closes[-1]
    atr = np.mean(np.array(highs[-10:]) - np.array(lows[-10:]))
    avg_price = np.mean(closes[-10:])
    vol_pct = atr / avg_price * 100 if avg_price > 0 else 0
    if vol_pct > 3.5: return "STRESS", 0.6
    recent_high = max(highs[-5:]); prev_high = max(highs[-10:-5])
    recent_low = min(lows[-5:]); prev_low = min(lows[-10:-5])
    ema_bull = price > ema20 > ema50
    ema_bear = price < ema20 < ema50
    hh_hl = recent_high > prev_high and recent_low > prev_low
    lh_ll = recent_high < prev_high and recent_low < prev_low
    if ema_bull and hh_hl: return "BULL", 0.75
    elif ema_bear and lh_ll: return "BEAR", 0.75
    else: return "RANGING", 0.6

def run_backtest(df, start_idx=672):
    strategy = BBRegimeReversionV3()
    trades = []
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_upper'] = sma + std * 2.0
    df['BB_middle'] = sma
    df['BB_lower'] = sma - std * 2.0

    i = start_idx
    while i < len(df) - 50:
        regime, confidence = classify_regime(df, i)
        result = strategy.analyze(df, i, regime, confidence)
        if result["signal"] != "NEUTRAL":
            entry = result["entry"]; tp = result["tp"]; sl = result["sl"]
            direction = result["signal"]
            outcome = "TIMEOUT"; exit_price = entry; bars_held = 0
            for j in range(i+1, min(i+50, len(df))):
                bars_held += 1
                h = df['High'].iloc[j]; l = df['Low'].iloc[j]
                if direction == "LONG":
                    if h >= tp: outcome = "WIN"; exit_price = tp; break
                    elif l <= sl: outcome = "LOSS"; exit_price = sl; break
                else:
                    if l <= tp: outcome = "WIN"; exit_price = tp; break
                    elif h >= sl: outcome = "LOSS"; exit_price = sl; break
            pnl = ((exit_price-entry)/entry*100) if direction=="LONG" else ((entry-exit_price)/entry*100)
            trades.append({
                "bar": i, "regime": regime, "direction": direction,
                "entry": round(entry,2), "tp": round(tp,2), "sl": round(sl,2),
                "exit": round(exit_price,2), "outcome": outcome,
                "pnl_pct": round(pnl,4), "bars_held": bars_held,
                "reason": result["reason"], "conviction": result["conviction"],
                "confirmations": [c["type"] for c in result.get("confirmations",[])],
                "weighted_score": result.get("weighted_score",0), "rr": result.get("rr",0),
            })
        i += 1
    return trades

def analyze(trades):
    if not trades:
        print("No trades generated."); return
    wins = [t for t in trades if t["outcome"]=="WIN"]
    losses = [t for t in trades if t["outcome"]=="LOSS"]
    total = len(trades)
    wr = len(wins)/total*100 if total else 0
    mean_pnl = np.mean([t["pnl_pct"] for t in trades])
    mean_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    mean_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    total_win = sum(t["pnl_pct"] for t in wins)
    total_loss = abs(sum(t["pnl_pct"] for t in losses))
    pf = total_win/total_loss if total_loss > 0 else float('inf')

    # By confirmation count
    by_conf = {}
    for t in trades:
        n = len(t.get("confirmations",[]))
        if n not in by_conf: by_conf[n] = {"trades":0,"wins":0,"pnl":0}
        by_conf[n]["trades"] += 1
        if t["outcome"]=="WIN": by_conf[n]["wins"] += 1
        by_conf[n]["pnl"] += t["pnl_pct"]

    # By score range
    by_score = {}
    for t in trades:
        s = int(t.get("weighted_score",0))
        if s not in by_score: by_score[s] = {"trades":0,"wins":0,"pnl":0}
        by_score[s]["trades"] += 1
        if t["outcome"]=="WIN": by_score[s]["wins"] += 1
        by_score[s]["pnl"] += t["pnl_pct"]

    # Confirmation types
    conf_types = {}
    for t in trades:
        for c in t.get("confirmations",[]):
            if c not in conf_types: conf_types[c] = {"trades":0,"wins":0}
            conf_types[c]["trades"] += 1
            if t["outcome"]=="WIN": conf_types[c]["wins"] += 1

    print("="*60)
    print("BB REGIME REVERSION v3 — BACKTEST RESULTS")
    print("(weighted confirmations, strict entry)")
    print("="*60)
    print(f"Total trades:    {total}")
    print(f"Wins:            {len(wins)} ({wr:.1f}%)")
    print(f"Losses:          {len(losses)}")
    print(f"Mean PnL:        {mean_pnl:+.4f}%")
    print(f"Mean Win:        {mean_win:+.4f}%")
    print(f"Mean Loss:       {mean_loss:+.4f}%")
    print(f"Profit Factor:   {pf:.2f}")

    pnls = [t["pnl_pct"] for t in trades]
    if len(pnls) >= 10:
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(pnls, 0)
        print(f"p-value:         {p_value:.4f} {'*** SIGNIFICANT ***' if p_value < 0.05 else '(not significant)'}")

    print(f"\nBy Confirmation Count:")
    for n in sorted(by_conf.keys()):
        d = by_conf[n]
        r_wr = d["wins"]/d["trades"]*100 if d["trades"] else 0
        r_mean = d["pnl"]/d["trades"] if d["trades"] else 0
        print(f"  {n} conf: {d['trades']:>4} trades, {r_wr:>5.1f}% WR, mean={r_mean:+.4f}%, total={d['pnl']:+.2f}%")

    print(f"\nBy Weighted Score:")
    for s in sorted(by_score.keys()):
        d = by_score[s]
        if d["trades"] < 3: continue
        r_wr = d["wins"]/d["trades"]*100 if d["trades"] else 0
        r_mean = d["pnl"]/d["trades"] if d["trades"] else 0
        print(f"  score {s}: {d['trades']:>4} trades, {r_wr:>5.1f}% WR, mean={r_mean:+.4f}%")

    print(f"\nConfirmation Types:")
    for c in sorted(conf_types.keys(), key=lambda x: conf_types[x]["wins"]/max(conf_types[x]["trades"],1), reverse=True):
        d = conf_types[c]
        c_wr = d["wins"]/d["trades"]*100 if d["trades"] else 0
        print(f"  {c:<20} {d['trades']:>4} trades, {c_wr:>5.1f}% WR")

    equity = 200.0; peak = equity; max_dd = 0
    for t in trades:
        equity += equity * t["pnl_pct"] / 100
        if equity > peak: peak = equity
        dd = (peak-equity)/peak*100; max_dd = max(max_dd, dd)
    print(f"\nEquity ($200): ${equity:.2f} (max DD: {max_dd:.1f}%)")

    return {"total":total,"wr":round(wr,1),"mean_pnl":round(mean_pnl,4),
            "pf":round(pf,2),"p_value":round(p_value,4) if len(pnls)>=10 else None,
            "equity":round(equity,2),"max_dd":round(max_dd,1)}

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} bars")
    print("Running backtest...")
    trades = run_backtest(df)
    results = analyze(trades)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "bb_regime_v3_backtest.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({"summary":results,"sample":trades[:30]}, f, indent=2)
    print(f"\nSaved to {out}")
