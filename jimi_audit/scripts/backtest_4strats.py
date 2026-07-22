"""
Backtest 4 stuck strategies against historical signals + price data.
Outputs results compatible with isolation_gate_results.json format.
"""
import json, os, sys
import numpy as np
from datetime import datetime, timezone, timedelta
from scipy import stats as sp_stats

DATA_DIR = "/root/.openclaw/workspace/jimi_audit/data"
SIGNALS_FILE = os.path.join(DATA_DIR, "strategy_signals.jsonl")
PRICE_FILE = os.path.join(DATA_DIR, "eth_15m_merged.csv")

# Strategy configs: TP%, SL%, hold_hours
STRATEGY_CFG = {
    "liquidation_cascade": {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 4},
    "liquidity_grab":      {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 12},
    "structural_break":    {"tp_pct": 0.5, "sl_pct": 0.5, "hold_hours": 8, "direction": "SHORT"},
    "forced_movement":     {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 4},
}

TARGET_STRATEGIES = set(STRATEGY_CFG.keys())


def load_prices():
    """Load 15m OHLCV data into dict keyed by timestamp string."""
    prices = {}
    with open(PRICE_FILE) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            ts_str = parts[0]  # "2026-01-01 00:15:00"
            try:
                prices[ts_str] = {
                    "open": float(parts[1]),
                    "high": float(parts[2]),
                    "low": float(parts[3]),
                    "close": float(parts[4]),
                    "volume": float(parts[5]),
                }
            except (ValueError, IndexError):
                continue
    return prices


def load_signals():
    """Load fired signals for target strategies."""
    signals = {s: [] for s in TARGET_STRATEGIES}
    with open(SIGNALS_FILE) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                strat = d.get("strategy", "")
                if strat in TARGET_STRATEGIES and d.get("fired") and d.get("direction"):
                    signals[strat].append(d)
            except (json.JSONDecodeError, ValueError):
                continue
    return signals


def get_price_at(prices, ts_str, offset_bars=0):
    """Get price data at ts_str + offset_bars*15min."""
    try:
        dt = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
        dt += timedelta(minutes=15 * offset_bars)
        key = dt.strftime("%Y-%m-%d %H:%M:%S")
        return prices.get(key)
    except Exception:
        return None


def simulate_trade(prices, signal, cfg):
    """
    Simulate a trade from signal timestamp.
    Returns pnl_pct or None if data missing.
    """
    ts = signal["timestamp"]
    entry = signal.get("entry") or signal.get("price")
    direction = signal.get("direction", "LONG")
    tp_pct = cfg["tp_pct"] / 100
    sl_pct = cfg["sl_pct"] / 100
    hold_bars = cfg["hold_hours"] * 4  # 15m bars

    if not entry or entry <= 0:
        return None

    # Check if we have price data at entry
    entry_bar = get_price_at(prices, ts)
    if not entry_bar:
        return None

    # Walk forward bar by bar
    for i in range(1, hold_bars + 1):
        bar = get_price_at(prices, ts, i)
        if not bar:
            continue

        if direction == "LONG":
            # Check SL first (conservative)
            if bar["low"] <= entry * (1 - sl_pct):
                return -sl_pct * 100
            # Check TP
            if bar["high"] >= entry * (1 + tp_pct):
                return tp_pct * 100
        else:  # SHORT
            if bar["high"] >= entry * (1 + sl_pct):
                return -sl_pct * 100
            if bar["low"] <= entry * (1 - tp_pct):
                return tp_pct * 100

    # Not hit TP or SL — close at market (last bar close)
    last_bar = get_price_at(prices, ts, hold_bars)
    if not last_bar:
        return None

    if direction == "LONG":
        return (last_bar["close"] - entry) / entry * 100
    else:
        return (entry - last_bar["close"]) / entry * 100


def run_backtest():
    print("Loading prices...")
    prices = load_prices()
    print(f"  {len(prices)} bars loaded")

    print("Loading signals...")
    signals = load_signals()

    results = {}
    for strat in sorted(TARGET_STRATEGIES):
        cfg = STRATEGY_CFG[strat]
        sigs = signals[strat]
        print(f"\n{'='*60}")
        print(f"Strategy: {strat} ({len(sigs)} signals)")
        print(f"  TP={cfg['tp_pct']}% SL={cfg['sl_pct']}% Hold={cfg['hold_hours']}h")

        if len(sigs) == 0:
            print("  NO SIGNALS — cannot backtest")
            results[strat] = {
                "passed": False,
                "p_value": 1.0,
                "effect_direction": "unknown",
                "mean_return_pct": 0.0,
                "events": 0,
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "notes": "No historical signals found in strategy_signals.jsonl"
            }
            continue

        # Simulate all trades
        pnls = []
        wins = 0
        losses = 0
        for sig in sigs:
            pnl = simulate_trade(prices, sig, cfg)
            if pnl is not None:
                pnls.append(pnl)
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

        if len(pnls) == 0:
            print("  No valid trades (missing price data)")
            results[strat] = {
                "passed": False,
                "p_value": 1.0,
                "effect_direction": "unknown",
                "mean_return_pct": 0.0,
                "events": 0,
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "notes": "Signals exist but no price data to backtest"
            }
            continue

        pnls = np.array(pnls)
        mean_ret = np.mean(pnls)
        std_ret = np.std(pnls, ddof=1)
        n = len(pnls)
        wr = wins / n if n > 0 else 0

        # One-sample t-test: H0 mean=0, H1 mean>0
        if std_ret > 0 and n > 1:
            t_stat = mean_ret / (std_ret / np.sqrt(n))
            p_value = 1 - sp_stats.t.cdf(t_stat, df=n-1)  # one-sided
        else:
            p_value = 1.0

        # Profit factor
        gross_profit = pnls[pnls > 0].sum() if any(pnls > 0) else 0
        gross_loss = abs(pnls[pnls <= 0].sum()) if any(pnls <= 0) else 0.001
        pf = gross_profit / gross_loss

        passed = p_value < 0.05 and mean_ret > 0
        effect_dir = "correct" if mean_ret > 0 else "backwards"

        print(f"  Trades: {n} | Wins: {wins} | Losses: {losses}")
        print(f"  WR: {wr:.1%} | Mean PnL: {mean_ret:+.4f}% | PF: {pf:.2f}")
        print(f"  p-value: {p_value:.4f} | {'PASS' if passed else 'FAIL'}")

        results[strat] = {
            "passed": passed,
            "p_value": round(p_value, 4),
            "effect_direction": effect_dir,
            "mean_return_pct": round(mean_ret, 4),
            "events": n,
            "wr": round(wr, 4),
            "pf": round(pf, 2),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "notes": f"v3 backtest: {n} events from strategy_signals.jsonl, {wr:.1%} WR, PF={pf:.2f}"
        }

    return results


if __name__ == "__main__":
    results = run_backtest()

    # Save results
    output_file = "/root/.openclaw/workspace/jimi_audit/config/backtest_4strats_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for strat, r in results.items():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {strat:30s} {status:5s} events={r['events']:>5d} mean={r['mean_return_pct']:+.4f}% p={r['p_value']:.4f}")
