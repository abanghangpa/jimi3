"""
BB Regime Reversion v3 — Extended backtest (88K bars, Jan 2024 - Jul 2026)
Uses the full eth_15m_merged_extended.csv dataset.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from strategies.bb_regime_reversion_v3 import BBRegimeReversionV3

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eth_15m_merged_extended.csv")


def load_data():
    df = pd.read_csv(DATA_FILE)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            for c in df.columns:
                if col.lower() in c.lower():
                    df[col] = df[c]; break
    return df


def classify_regime(df, idx, lookback=20):
    if idx < lookback:
        return "RANGING", 0.5
    closes = df['Close'].iloc[idx-lookback:idx+1].values.astype(float)
    highs = df['High'].iloc[idx-lookback:idx+1].values.astype(float)
    lows = df['Low'].iloc[idx-lookback:idx+1].values.astype(float)
    ema20 = pd.Series(closes).ewm(span=20).mean().iloc[-1]
    ema50 = pd.Series(closes).ewm(span=min(50, lookback)).mean().iloc[-1]
    price = closes[-1]
    atr = np.mean(np.array(highs[-10:]) - np.array(lows[-10:]))
    avg_price = np.mean(closes[-10:])
    vol_pct = atr / avg_price * 100 if avg_price > 0 else 0

    if vol_pct > 3.5:
        return "STRESS", 0.6
    recent_high = max(highs[-5:])
    prev_high = max(highs[-10:-5])
    recent_low = min(lows[-5:])
    prev_low = min(lows[-10:-5])
    ema_bull = price > ema20 > ema50
    ema_bear = price < ema20 < ema50
    hh_hl = recent_high > prev_high and recent_low > prev_low
    lh_ll = recent_high < prev_high and recent_low < prev_low

    if ema_bull and hh_hl:
        return "BULL", 0.75
    elif ema_bear and lh_ll:
        return "BEAR", 0.75
    else:
        return "RANGING", 0.6


def run_backtest():
    print(f"Loading data from {DATA_FILE}...")
    df = load_data()
    print(f"  {len(df)} bars loaded ({df['Open time'].iloc[0]} to {df['Open time'].iloc[-1]})")

    strategy = BBRegimeReversionV3()
    start_idx = 200  # Need warmup for BB + RSI
    trades = []
    signals_total = 0
    signals_by_regime = {}

    for idx in range(start_idx, len(df) - 50):  # Leave room for trade exit
        regime, conf = classify_regime(df, idx)
        result = strategy.analyze(df, idx, regime, conf)

        if result is None or not hasattr(result, 'direction') or result.direction is None:
            continue

        signals_total += 1
        signals_by_regime[regime] = signals_by_regime.get(regime, 0) + 1

        direction = result.direction
        entry = result.entry if hasattr(result, 'entry') and result.entry else df['Close'].iloc[idx]
        tp = result.tp if hasattr(result, 'tp') and result.tp else None
        sl = result.sl if hasattr(result, 'sl') and result.sl else None

        if tp is None or sl is None or entry is None:
            continue

        # Simulate trade — walk forward
        outcome = None
        exit_price = entry
        bars_held = 0
        hold_bars = 32  # 8 hours = 32 bars of 15m

        for j in range(1, hold_bars + 1):
            if idx + j >= len(df):
                break
            bar_high = float(df['High'].iloc[idx + j])
            bar_low = float(df['Low'].iloc[idx + j])
            bar_close = float(df['Close'].iloc[idx + j])
            bars_held = j

            if direction == "LONG":
                if bar_low <= sl:
                    outcome = "LOSS"
                    exit_price = sl
                    break
                if bar_high >= tp:
                    outcome = "WIN"
                    exit_price = tp
                    break
            else:  # SHORT
                if bar_high >= sl:
                    outcome = "LOSS"
                    exit_price = sl
                    break
                if bar_low <= tp:
                    outcome = "WIN"
                    exit_price = tp
                    break

        if outcome is None:
            # Timeout — close at market
            if idx + hold_bars < len(df):
                exit_price = float(df['Close'].iloc[idx + hold_bars])
                bars_held = hold_bars
                if direction == "LONG":
                    outcome = "WIN" if exit_price > entry else "LOSS"
                else:
                    outcome = "WIN" if exit_price < entry else "LOSS"
            else:
                continue

        if direction == "LONG":
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100

        trades.append({
            "bar": idx,
            "time": str(df['Open time'].iloc[idx]),
            "regime": regime,
            "direction": direction,
            "entry": round(entry, 2),
            "tp": round(tp, 2),
            "sl": round(sl, 2),
            "exit": round(exit_price, 2),
            "outcome": outcome,
            "pnl_pct": round(pnl_pct, 4),
            "bars_held": bars_held,
            "conviction": getattr(result, 'conviction', 0.5),
        })

    # === STATISTICS ===
    if not trades:
        print("No trades generated!")
        return

    pnls = np.array([t["pnl_pct"] for t in trades])
    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    losses = sum(1 for t in trades if t["outcome"] == "LOSS")
    n = len(trades)
    wr = wins / n if n > 0 else 0
    mean_ret = np.mean(pnls)
    std_ret = np.std(pnls, ddof=1) if n > 1 else 0

    # One-sided t-test
    if std_ret > 0 and n > 1:
        t_stat = mean_ret / (std_ret / np.sqrt(n))
        p_value = 1 - sp_stats.t.cdf(t_stat, df=n-1)
    else:
        p_value = 1.0

    gross_profit = pnls[pnls > 0].sum() if any(pnls > 0) else 0
    gross_loss = abs(pnls[pnls <= 0].sum()) if any(pnls <= 0) else 0.001
    pf = gross_profit / gross_loss

    # Equity curve
    equity = 200.0
    max_equity = equity
    max_dd = 0
    for t in trades:
        pnl_dollar = equity * t["pnl_pct"] / 100
        equity += pnl_dollar
        max_equity = max(max_equity, equity)
        dd = (max_equity - equity) / max_equity * 100
        max_dd = max(max_dd, dd)

    passed = p_value < 0.05 and mean_ret > 0

    # Regime breakdown
    regime_stats = {}
    for t in trades:
        r = t["regime"]
        if r not in regime_stats:
            regime_stats[r] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0}
        regime_stats[r]["trades"] += 1
        if t["outcome"] == "WIN":
            regime_stats[r]["wins"] += 1
        else:
            regime_stats[r]["losses"] += 1
        regime_stats[r]["pnl"] += t["pnl_pct"]

    print(f"\n{'='*60}")
    print(f"BB REGIME REVERSION v3 — EXTENDED BACKTEST")
    print(f"{'='*60}")
    print(f"Data: {len(df)} bars ({df['Open time'].iloc[0]} to {df['Open time'].iloc[-1]})")
    print(f"Total signals: {signals_total}")
    print(f"Signals by regime: {signals_by_regime}")
    print(f"\nTrades: {n} | Wins: {wins} | Losses: {losses}")
    print(f"WR: {wr:.1%} | Mean PnL: {mean_ret:+.4f}% | PF: {pf:.2f}")
    print(f"p-value: {p_value:.4f} | {'PASS' if passed else 'FAIL'}")
    print(f"Final equity: ${equity:.2f} | Max DD: {max_dd:.1f}%")
    print(f"\nRegime breakdown:")
    for r, s in sorted(regime_stats.items()):
        wr_r = s["wins"] / s["trades"] if s["trades"] > 0 else 0
        print(f"  {r:20s} trades={s['trades']:>4d} WR={wr_r:.1%} PnL={s['pnl']:+.2f}%")

    # Save results
    results = {
        "summary": {
            "total": n,
            "wins": wins,
            "losses": losses,
            "wr": round(wr, 4),
            "mean_pnl": round(mean_ret, 4),
            "pf": round(pf, 2),
            "p_value": round(p_value, 4),
            "equity": round(equity, 2),
            "max_dd": round(max_dd, 1),
            "data_bars": len(df),
            "date_range": f"{df['Open time'].iloc[0]} to {df['Open time'].iloc[-1]}",
        },
        "regime_breakdown": regime_stats,
        "signals_by_regime": signals_by_regime,
        "sample": trades[:30],
    }

    output = "/root/.openclaw/workspace/jimi_audit/reports/bb_regime_v3_extended_backtest.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output}")

    return results


if __name__ == "__main__":
    run_backtest()
