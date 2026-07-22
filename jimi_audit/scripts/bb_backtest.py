"""
Backtest: BB Regime Reversion Strategy
Tests the regime-aware BB strategy on historical ETH 15m data.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd
import numpy as np
from strategies.bb_regime_reversion import BBRegimeReversion

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eth_15m_merged.csv")

def load_data():
    df = pd.read_csv(DATA_FILE)
    # Standardize column names
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            for c in df.columns:
                if col.lower() in c.lower():
                    df[col] = df[c]
                    break
    return df

def classify_regime_simple(df, idx, lookback=20):
    """Simple regime classification based on price structure."""
    if idx < lookback:
        return "RANGING", 0.5
    
    closes = df['Close'].iloc[idx-lookback:idx+1].values
    highs = df['High'].iloc[idx-lookback:idx+1].values
    lows = df['Low'].iloc[idx-lookback:idx+1].values
    
    # EMA 20
    ema20 = pd.Series(closes).ewm(span=20).mean().iloc[-1]
    # EMA 50 (approximate with available data)
    ema50 = pd.Series(closes).ewm(span=min(50, lookback)).mean().iloc[-1]
    
    # Higher highs and higher lows = BULL
    recent_high = max(highs[-5:])
    prev_high = max(highs[-10:-5])
    recent_low = min(lows[-5:])
    prev_low = min(lows[-10:-5])
    
    price = closes[-1]
    
    # Volatility check
    atr = np.mean(np.array(highs[-10:]) - np.array(lows[-10:]))
    avg_price = np.mean(closes[-10:])
    vol_pct = atr / avg_price * 100 if avg_price > 0 else 0
    
    if vol_pct > 3.0:
        return "STRESS", 0.6
    
    if price > ema20 > ema50 and recent_high > prev_high and recent_low > prev_low:
        return "BULL", 0.7
    elif price < ema20 < ema50 and recent_high < prev_high and recent_low < prev_low:
        return "BEAR", 0.7
    else:
        return "RANGING", 0.6

def run_backtest(df, start_idx=672, tp_mult=1.0, sl_mult=1.0):
    """Run backtest on historical data."""
    strategy = BBRegimeReversion()
    trades = []
    
    # Calculate BB
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_upper'] = sma + (std * 2.0)
    df['BB_middle'] = sma
    df['BB_lower'] = sma - (std * 2.0)
    
    i = start_idx
    while i < len(df) - 50:  # Leave room for forward check
        regime, confidence = classify_regime_simple(df, i)
        
        result = strategy.analyze(df, i, regime, confidence)
        
        if result["signal"] != "NEUTRAL":
            entry = result["entry"]
            tp = result["tp"]
            sl = result["sl"]
            direction = result["signal"]
            
            # Forward check: did price hit TP or SL?
            outcome = "TIMEOUT"
            exit_price = entry
            bars_held = 0
            
            for j in range(i + 1, min(i + 50, len(df))):
                bars_held += 1
                high = df['High'].iloc[j]
                low = df['Low'].iloc[j]
                
                if direction == "LONG":
                    if high >= tp:
                        outcome = "WIN"
                        exit_price = tp
                        break
                    elif low <= sl:
                        outcome = "LOSS"
                        exit_price = sl
                        break
                else:  # SHORT
                    if low <= tp:
                        outcome = "WIN"
                        exit_price = tp
                        break
                    elif high >= sl:
                        outcome = "LOSS"
                        exit_price = sl
                        break
            
            pnl_pct = ((exit_price - entry) / entry * 100) if direction == "LONG" else ((entry - exit_price) / entry * 100)
            
            trades.append({
                "bar": i,
                "time": str(df.iloc[i].get('Open time', df.index[i])),
                "regime": regime,
                "direction": direction,
                "entry": round(entry, 2),
                "tp": round(tp, 2),
                "sl": round(sl, 2),
                "exit": round(exit_price, 2),
                "outcome": outcome,
                "pnl_pct": round(pnl_pct, 4),
                "bars_held": bars_held,
                "reason": result["reason"],
                "conviction": result["conviction"],
            })
        
        i += 1
    
    return trades

def analyze_trades(trades):
    """Analyze trade results."""
    if not trades:
        print("No trades generated.")
        return
    
    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    timeouts = [t for t in trades if t["outcome"] == "TIMEOUT"]
    
    total = len(trades)
    wr = len(wins) / total * 100 if total > 0 else 0
    mean_pnl = np.mean([t["pnl_pct"] for t in trades])
    mean_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    mean_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    
    # Profit factor
    total_win = sum(t["pnl_pct"] for t in wins)
    total_loss = abs(sum(t["pnl_pct"] for t in losses))
    pf = total_win / total_loss if total_loss > 0 else float('inf')
    
    # By regime
    regimes = {}
    for t in trades:
        r = t["regime"]
        if r not in regimes:
            regimes[r] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0}
        regimes[r]["trades"] += 1
        if t["outcome"] == "WIN":
            regimes[r]["wins"] += 1
        elif t["outcome"] == "LOSS":
            regimes[r]["losses"] += 1
        regimes[r]["pnl"] += t["pnl_pct"]
    
    print("=" * 60)
    print("BB REGIME REVERSION — BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total trades:  {total}")
    print(f"Wins:          {len(wins)} ({wr:.1f}%)")
    print(f"Losses:        {len(losses)}")
    print(f"Timeouts:      {len(timeouts)}")
    print(f"Mean PnL:      {mean_pnl:+.4f}%")
    print(f"Mean Win:      {mean_win:+.4f}%")
    print(f"Mean Loss:     {mean_loss:+.4f}%")
    print(f"Profit Factor: {pf:.2f}")
    
    print(f"\nBy Regime:")
    for r, data in sorted(regimes.items()):
        r_wr = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0
        r_mean = data["pnl"] / data["trades"] if data["trades"] > 0 else 0
        print(f"  {r:<15} {data['trades']:>4} trades, {r_wr:>5.1f}% WR, mean={r_mean:+.4f}%, total={data['pnl']:+.2f}%")
    
    # Statistical significance
    from scipy import stats
    pnls = [t["pnl_pct"] for t in trades]
    if len(pnls) >= 10:
        t_stat, p_value = stats.ttest_1samp(pnls, 0)
        print(f"\nStatistical Test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.4f}")
        print(f"  Significant: {'YES (p < 0.05)' if p_value < 0.05 else 'NO (p >= 0.05)'}")
    
    # Win/Loss streaks
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    for t in trades:
        if t["outcome"] == "WIN":
            if streak >= 0:
                streak += 1
            else:
                max_loss_streak = max(max_loss_streak, abs(streak))
                streak = 1
        elif t["outcome"] == "LOSS":
            if streak <= 0:
                streak -= 1
            else:
                max_win_streak = max(max_win_streak, streak)
                streak = -1
    max_win_streak = max(max_win_streak, max(0, streak))
    max_loss_streak = max(max_loss_streak, max(0, abs(streak)))
    
    print(f"\nStreaks:")
    print(f"  Max win streak:  {max_win_streak}")
    print(f"  Max loss streak: {max_loss_streak}")
    
    # Drawdown simulation
    equity = 200.0
    peak = equity
    max_dd = 0
    for t in trades:
        pnl_dollar = equity * t["pnl_pct"] / 100
        equity += pnl_dollar
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)
    
    print(f"\nEquity Simulation ($200 start):")
    print(f"  Final equity: ${equity:.2f}")
    print(f"  Max drawdown: {max_dd:.1f}%")
    
    return {
        "total": total,
        "wins": len(wins),
        "losses": len(losses),
        "wr": round(wr, 2),
        "mean_pnl": round(mean_pnl, 4),
        "pf": round(pf, 2),
        "p_value": round(p_value, 4) if len(pnls) >= 10 else None,
        "max_dd": round(max_dd, 1),
        "regimes": regimes,
    }

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} bars")
    
    print("Running backtest...")
    trades = run_backtest(df, start_idx=672)
    
    results = analyze_trades(trades)
    
    # Save results
    output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "bb_regime_reversion_backtest.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({"summary": results, "trades": trades[:50]}, f, indent=2)  # Save first 50 trades
    print(f"\nResults saved to {output_file}")
