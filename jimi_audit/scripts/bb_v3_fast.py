"""
BB Regime Reversion v3 — FAST vectorized backtest (88K bars)
Precomputes all indicators, then checks for signals.
"""
import sys, os, json
import pandas as pd
import numpy as np
from scipy import stats as sp_stats

DATA_FILE = "/root/.openclaw/workspace/jimi_audit/data/eth_15m_merged_extended.csv"


def load_data():
    df = pd.read_csv(DATA_FILE)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            for c in df.columns:
                if col.lower() in c.lower():
                    df[col] = df[c]; break
    return df


def calc_bb(close, period=20, std_mult=2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std * std_mult
    lower = sma - std * std_mult
    return upper, sma, lower


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_confirmations(df, idx, direction, upper, lower, rsi):
    """Check for confirmation signals at given index."""
    confirmations = []
    score = 0.0

    close = df['Close'].iloc[idx]
    open_p = df['Open'].iloc[idx]
    high = df['High'].iloc[idx]
    low = df['Low'].iloc[idx]
    prev_close = df['Close'].iloc[idx-1]
    prev_open = df['Open'].iloc[idx-1]
    prev_high = df['High'].iloc[idx-1]
    prev_low = df['Low'].iloc[idx-1]
    volume = df['Volume'].iloc[idx]
    avg_vol = df['Volume'].iloc[max(0,idx-20):idx].mean()

    body = abs(close - open_p)
    upper_wick = high - max(close, open_p)
    lower_wick = min(close, open_p) - low

    if direction == "SHORT":
        # Bearish engulfing
        if close < open_p and prev_close > prev_open and open_p >= prev_close and close <= prev_open:
            confirmations.append("BEARISH_ENGULFING"); score += 3.0
        # RSI overbought
        if rsi[idx] > 70:
            confirmations.append("RSI_OVERBOUGHT"); score += 0.3
        # RSI divergence (simplified: RSI dropping while price making higher high)
        if idx >= 5:
            if high >= max(df['High'].iloc[idx-5:idx]) and rsi[idx] < rsi[idx-3]:
                confirmations.append("RSI_DIVERGENCE"); score += 2.5
        # Volume exhaustion
        if avg_vol > 0 and volume > avg_vol * 1.5 and close < open_p:
            confirmations.append("VOL_EXHAUSTION"); score += 2.0
        # Rejection wick
        if upper_wick > body * 2 and body > 0:
            confirmations.append("REJECTION_WICK"); score += 1.5
        # Doji at band
        if body < (high - low) * 0.1 and high - low > 0:
            confirmations.append("DOJI_AT_BAND"); score += 1.0
        # Multiple touches (simplified: price near upper band in recent bars)
        if idx >= 5:
            touches = sum(1 for j in range(idx-5, idx) if df['High'].iloc[j] >= upper.iloc[j] * 0.998)
            if touches >= 2:
                confirmations.append("MULTIPLE_TOUCHES"); score += 0.5
    else:  # LONG
        # Bullish engulfing
        if close > open_p and prev_close < prev_open and open_p <= prev_close and close >= prev_open:
            confirmations.append("BULLISH_ENGULFING"); score += 3.0
        # RSI oversold
        if rsi[idx] < 30:
            confirmations.append("RSI_OVERSOLD"); score += 0.3
        # RSI divergence
        if idx >= 5:
            if low <= min(df['Low'].iloc[idx-5:idx]) and rsi[idx] > rsi[idx-3]:
                confirmations.append("RSI_DIVERGENCE"); score += 2.5
        # Volume exhaustion
        if avg_vol > 0 and volume > avg_vol * 1.5 and close > open_p:
            confirmations.append("VOL_EXHAUSTION"); score += 2.0
        # Rejection wick
        if lower_wick > body * 2 and body > 0:
            confirmations.append("REJECTION_WICK"); score += 1.5
        # Doji at band
        if body < (high - low) * 0.1 and high - low > 0:
            confirmations.append("DOJI_AT_BAND"); score += 1.0
        # Multiple touches
        if idx >= 5:
            touches = sum(1 for j in range(idx-5, idx) if df['Low'].iloc[j] <= lower.iloc[j] * 1.002)
            if touches >= 2:
                confirmations.append("MULTIPLE_TOUCHES"); score += 0.5

    return confirmations, score


def run_backtest():
    print(f"Loading data...")
    df = load_data()
    print(f"  {len(df)} bars ({df['Open time'].iloc[0]} to {df['Open time'].iloc[-1]})")

    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)

    # Precompute indicators
    print("Computing indicators...")
    upper, middle, lower = calc_bb(close)
    rsi = calc_rsi(close)

    # Classify regimes (vectorized)
    print("Classifying regimes...")
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    atr = (high - low).rolling(10).mean()
    avg_price = close.rolling(10).mean()
    vol_pct = atr / avg_price * 100

    recent_high5 = high.rolling(5).max()
    prev_high5 = high.rolling(10).max().shift(5)
    recent_low5 = low.rolling(5).min()
    prev_low5 = low.rolling(10).min().shift(5)

    hh_hl = (recent_high5 > prev_high5) & (recent_low5 > prev_low5)
    lh_ll = (recent_high5 < prev_high5) & (recent_low5 < prev_low5)
    ema_bull = (close > ema20) & (ema20 > ema50)
    ema_bear = (close < ema20) & (ema20 < ema50)

    regime = np.where(vol_pct > 3.5, "STRESS",
             np.where(ema_bull & hh_hl, "BULL",
             np.where(ema_bear & lh_ll, "BEAR", "RANGING")))

    # Find BB touch signals
    print("Finding signals...")
    touched_upper = high >= upper
    touched_lower = low <= lower
    skip_regime = pd.Series(regime).isin(["STRESS", "MILDLY_BEARISH"]).values

    trades = []
    start_idx = 200
    TP_PCT = {"LONG": 0.012, "SHORT": 0.012}  # 1.2% TP (targeting middle)
    SL_PCT = {"LONG": 0.005, "SHORT": 0.005}  # 0.5% SL
    HOLD_BARS = 32  # 8 hours

    HIGH_QUALITY = {"BEARISH_ENGULFING", "BULLISH_ENGULFING", "RSI_DIVERGENCE", "VOL_EXHAUSTION"}
    MIN_CONFS = 3
    MIN_SCORE = 4.0

    for idx in range(start_idx, len(df) - HOLD_BARS - 1):
        if skip_regime[idx]:
            continue

        direction = None
        if touched_upper.iloc[idx]:
            direction = "SHORT"
        elif touched_lower.iloc[idx]:
            direction = "LONG"
        else:
            continue

        # Get confirmations
        confs, score = detect_confirmations(df, idx, direction, upper, lower, rsi.values)

        if len(confs) < MIN_CONFS:
            continue
        if score < MIN_SCORE:
            continue
        if not any(c in HIGH_QUALITY for c in confs):
            continue

        entry = float(close.iloc[idx])
        if direction == "LONG":
            tp = entry * (1 + TP_PCT["LONG"])
            sl = entry * (1 - SL_PCT["LONG"])
        else:
            tp = entry * (1 - TP_PCT["SHORT"])
            sl = entry * (1 + SL_PCT["SHORT"])

        # Simulate trade
        outcome = None
        exit_price = entry
        bars_held = 0

        for j in range(1, HOLD_BARS + 1):
            bar_idx = idx + j
            if bar_idx >= len(df):
                break
            bh = float(high.iloc[bar_idx])
            bl = float(low.iloc[bar_idx])
            bars_held = j

            if direction == "LONG":
                if bl <= sl:
                    outcome = "LOSS"; exit_price = sl; break
                if bh >= tp:
                    outcome = "WIN"; exit_price = tp; break
            else:
                if bh >= sl:
                    outcome = "LOSS"; exit_price = sl; break
                if bl <= tp:
                    outcome = "WIN"; exit_price = tp; break

        if outcome is None:
            exit_price = float(close.iloc[min(idx + HOLD_BARS, len(df)-1)])
            bars_held = HOLD_BARS
            outcome = "WIN" if (direction == "LONG" and exit_price > entry) or (direction == "SHORT" and exit_price < entry) else "LOSS"

        pnl_pct = (exit_price - entry) / entry * 100 if direction == "LONG" else (entry - exit_price) / entry * 100

        trades.append({
            "bar": idx,
            "time": str(df['Open time'].iloc[idx]),
            "regime": str(regime[idx]),
            "direction": direction,
            "entry": round(entry, 2),
            "tp": round(tp, 2),
            "sl": round(sl, 2),
            "exit": round(exit_price, 2),
            "outcome": outcome,
            "pnl_pct": round(pnl_pct, 4),
            "bars_held": bars_held,
            "confirmations": confs,
            "score": round(score, 1),
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

    if std_ret > 0 and n > 1:
        t_stat = mean_ret / (std_ret / np.sqrt(n))
        p_value = 1 - sp_stats.t.cdf(t_stat, df=n-1)
    else:
        p_value = 1.0

    gross_profit = pnls[pnls > 0].sum() if any(pnls > 0) else 0
    gross_loss = abs(pnls[pnls <= 0].sum()) if any(pnls <= 0) else 0.001
    pf = gross_profit / gross_loss

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
    print(f"BB REGIME REVERSION v3 — EXTENDED BACKTEST (FAST)")
    print(f"{'='*60}")
    print(f"Data: {len(df)} bars ({df['Open time'].iloc[0]} to {df['Open time'].iloc[-1]})")
    print(f"\nTrades: {n} | Wins: {wins} | Losses: {losses}")
    print(f"WR: {wr:.1%} | Mean PnL: {mean_ret:+.4f}% | PF: {pf:.2f}")
    print(f"p-value: {p_value:.4f} | {'PASS' if passed else 'FAIL'}")
    print(f"Final equity: ${equity:.2f} | Max DD: {max_dd:.1f}%")
    print(f"\nRegime breakdown:")
    for r, s in sorted(regime_stats.items()):
        wr_r = s["wins"] / s["trades"] if s["trades"] > 0 else 0
        print(f"  {r:20s} trades={s['trades']:>4d} WR={wr_r:.1%} PnL={s['pnl']:+.2f}%")

    results = {
        "summary": {
            "total": n, "wins": wins, "losses": losses,
            "wr": round(wr, 4), "mean_pnl": round(mean_ret, 4),
            "pf": round(pf, 2), "p_value": round(p_value, 4),
            "equity": round(equity, 2), "max_dd": round(max_dd, 1),
            "data_bars": len(df),
            "date_range": f"{df['Open time'].iloc[0]} to {df['Open time'].iloc[-1]}",
        },
        "regime_breakdown": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in regime_stats.items()},
        "sample": trades[:30],
    }

    output = "/root/.openclaw/workspace/jimi_audit/reports/bb_regime_v3_extended_backtest.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output}")

    return results


if __name__ == "__main__":
    run_backtest()
