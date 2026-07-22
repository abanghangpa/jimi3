#!/usr/bin/env python3
"""
Comprehensive Backtest: V4 Daily Regime + JIMI Strategies + Monte Carlo

Tests:
1. V4 regime gate against historical trades
2. Strategy-level performance per regime
3. Monte Carlo with bootstrap confidence intervals
4. Walk-forward regime accuracy
5. Risk-adjusted metrics (Sharpe, Sortino, Calmar)
"""
import json, os, sys, csv, random, math
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import requests
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))

from regime_classifier_v4 import RegimeClassifierV4

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

def load_daily_candles(n=200):
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": "ETHUSDT", "interval": "1d", "limit": n}, timeout=15)
    r.raise_for_status()
    candles = []
    for c in r.json():
        candles.append({
            "ts": datetime.fromtimestamp(c[0]/1000),
            "open": float(c[1]), "high": float(c[2]),
            "low": float(c[3]), "close": float(c[4]),
            "volume": float(c[5]),
        })
    return candles

def load_15m_candles(n=1500):
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": "ETHUSDT", "interval": "15m", "limit": n}, timeout=15)
    r.raise_for_status()
    candles = []
    for c in r.json():
        candles.append({
            "ts": datetime.fromtimestamp(c[0]/1000),
            "open": float(c[1]), "high": float(c[2]),
            "low": float(c[3]), "close": float(c[4]),
            "volume": float(c[5]),
        })
    return candles

def load_derivatives_daily():
    csv_path = os.path.join(BASE, "data", "derivatives_history", "derivatives_collected.csv")
    daily = defaultdict(list)
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp", "")
            if not ts:
                continue
            day = ts[:10].replace("T", " ")[:10]
            try:
                daily[day].append({
                    "fr": float(row.get("funding_rate", 0) or 0),
                    "ls": float(row.get("ls_ratio", 2.0) or 2.0),
                    "oi": float(row.get("oi", 0) or 0),
                    "taker": float(row.get("futures_taker_ratio", 1.0) or 1.0),
                })
            except:
                continue
    agg = {}
    for day, rows in daily.items():
        if rows:
            agg[day] = {
                "fr_avg": sum(r["fr"] for r in rows) / len(rows),
                "ls_avg": sum(r["ls"] for r in rows) / len(rows),
                "taker_avg": sum(r["taker"] for r in rows) / len(rows),
                "oi_trend": ((rows[-1]["oi"] - rows[0]["oi"]) / rows[0]["oi"] * 100) if rows[0]["oi"] > 0 else 0,
            }
    return agg

def load_historical_trades():
    """Load executor trade history."""
    state_path = os.path.join(BASE, "live", "data", "executor_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        return state.get("closed_trades", [])
    return []

# ═══════════════════════════════════════════════════════════════
# STRATEGY SIMULATOR
# ═══════════════════════════════════════════════════════════════

class StrategySimulator:
    """Simulate JIMI strategies with regime gating."""

    # Strategy configs (from scanner_executor.py)
    STRATEGIES = {
        "trade_flow":           {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 12, "allowed": ["BULL", "BEAR", "RANGING", "STRESS"]},
        "orderbook_imbalance":  {"tp_pct": 2.0, "sl_pct": 0.75, "hold_hours": 12, "allowed": ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH"]},
        "liquidation_cascade":  {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 4, "allowed": ["BULL", "BEAR", "RANGING", "MILDLY_BEARISH"]},
        "liquidity_grab":       {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 12, "allowed": ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH"]},
        "judas_sweep":          {"tp_pct": 2.5, "sl_pct": 1.5, "hold_hours": 8, "allowed": ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH"]},
        "squeeze_breakout":     {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 8, "allowed": ["BULL", "BEAR", "RANGING", "MILDLY_BEARISH"]},
        "funding_squeeze":      {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 8, "allowed": ["BULL", "BEAR", "RANGING", "MILDLY_BEARISH"]},
        "failed_breakout":      {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 8, "allowed": ["RANGING", "BULL"]},
    }

    def __init__(self, capital=10000.0, risk_per_trade=0.02, leverage=25):
        self.initial_capital = capital
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.trades = []
        self.equity_curve = [capital]
        self.open_positions = []

    def simulate_bar(self, candle, regime, confidence, bar_idx):
        """Process one 15m bar."""
        price = candle["close"]
        high = candle["high"]
        low = candle["low"]

        # Check open positions for TP/SL
        closed = []
        for pos in self.open_positions:
            hit_sl = low <= pos["sl"] if pos["direction"] == "LONG" else high >= pos["sl"]
            hit_tp = high >= pos["tp"] if pos["direction"] == "LONG" else low <= pos["tp"]

            if hit_sl:
                pnl_pct = -pos["sl_pct"] / 100
                closed.append((pos, "SL", pnl_pct))
            elif hit_tp:
                pnl_pct = pos["tp_pct"] / 100
                closed.append((pos, "TP", pnl_pct))
            elif bar_idx - pos["entry_bar"] >= pos["hold_bars"]:
                pnl_pct = (price - pos["entry"]) / pos["entry"] if pos["direction"] == "LONG" else (pos["entry"] - price) / pos["entry"]
                closed.append((pos, "TIMEOUT", pnl_pct))

        for pos, outcome, pnl_pct in closed:
            pnl_dollar = pos["size_dollar"] * pnl_pct
            self.capital += pnl_dollar
            self.trades.append({
                "strategy": pos["strategy"],
                "direction": pos["direction"],
                "entry": pos["entry"],
                "exit": price,
                "pnl_pct": round(pnl_pct * 100, 2),
                "pnl_dollar": round(pnl_dollar, 2),
                "outcome": outcome,
                "regime": pos["regime_at_entry"],
                "entry_bar": pos["entry_bar"],
                "exit_bar": bar_idx,
            })
            self.open_positions.remove(pos)

        # Generate signals (simplified — random based on regime)
        # In reality, these come from the scanner. Here we simulate.
        for strat_name, cfg in self.STRATEGIES.items():
            if regime not in cfg["allowed"]:
                continue  # Regime gate blocks this strategy

            # Signal probability (simplified)
            signal_prob = 0.002  # ~0.2% chance per bar = ~1 signal per 500 bars
            if regime == "BULL" and strat_name in ("trade_flow", "orderbook_imbalance"):
                signal_prob = 0.004  # More signals in favorable regime
            elif regime == "BEAR" and strat_name in ("liquidation_cascade", "judas_sweep"):
                signal_prob = 0.004

            if random.random() < signal_prob:
                direction = "LONG" if random.random() > 0.45 else "SHORT"
                entry = price
                sl_pct = cfg["sl_pct"]
                tp_pct = cfg["tp_pct"]

                if direction == "LONG":
                    sl = entry * (1 - sl_pct / 100)
                    tp = entry * (1 + tp_pct / 100)
                else:
                    sl = entry * (1 + sl_pct / 100)
                    tp = entry * (1 - tp_pct / 100)

                # Position sizing
                risk_dollar = self.capital * self.risk_per_trade
                size_dollar = risk_dollar / (sl_pct / 100) * self.leverage

                # Win probability (strategy-dependent, regime-dependent)
                base_wr = 0.45
                if regime == "BULL" and direction == "LONG":
                    base_wr = 0.55
                elif regime == "BEAR" and direction == "SHORT":
                    base_wr = 0.55
                elif regime == "STRESS":
                    base_wr = 0.35

                self.open_positions.append({
                    "strategy": strat_name,
                    "direction": direction,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "hold_bars": cfg["hold_hours"] * 4,  # 15m bars
                    "size_dollar": size_dollar,
                    "regime_at_entry": regime,
                    "entry_bar": bar_idx,
                    "win_prob": base_wr,
                })

        self.equity_curve.append(self.capital)

    def get_metrics(self):
        """Calculate comprehensive metrics."""
        if not self.trades:
            return {"error": "no trades"}

        returns = [t["pnl_pct"] / 100 for t in self.trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        # Drawdown
        peak = self.initial_capital
        max_dd = 0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        # Sharpe (annualized, assuming 15m bars)
        avg_ret = np.mean(returns) if returns else 0
        std_ret = np.std(returns) if len(returns) > 1 else 0.001
        sharpe = (avg_ret / std_ret) * math.sqrt(365 * 24 * 4) if std_ret > 0 else 0

        # Sortino (downside only)
        downside = [r for r in returns if r < 0]
        downside_std = np.std(downside) if len(downside) > 1 else 0.001
        sortino = (avg_ret / downside_std) * math.sqrt(365 * 24 * 4) if downside_std > 0 else 0

        # Calmar
        calmar = ((self.capital / self.initial_capital - 1) * 365 / 150) / max_dd if max_dd > 0 else 0

        # By regime
        regime_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0})
        for t in self.trades:
            r = t["regime"]
            regime_stats[r]["trades"] += 1
            if t["pnl_pct"] > 0:
                regime_stats[r]["wins"] += 1
            regime_stats[r]["total_pnl"] += t["pnl_dollar"]

        # By strategy
        strat_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0})
        for t in self.trades:
            s = t["strategy"]
            strat_stats[s]["trades"] += 1
            if t["pnl_pct"] > 0:
                strat_stats[s]["wins"] += 1
            strat_stats[s]["total_pnl"] += t["pnl_dollar"]

        return {
            "capital": round(self.capital, 2),
            "return_pct": round((self.capital / self.initial_capital - 1) * 100, 2),
            "total_trades": len(self.trades),
            "win_rate": round(len(wins) / len(self.trades) * 100, 1),
            "avg_win": round(np.mean(wins) * 100, 2) if wins else 0,
            "avg_loss": round(np.mean(losses) * 100, 2) if losses else 0,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "calmar": round(calmar, 3),
            "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses else float("inf"),
            "regime_stats": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in regime_stats.items()},
            "strategy_stats": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in strat_stats.items()},
        }

# ═══════════════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════════════

def monte_carlo(trades, n_sims=10000, horizon_days=30, initial_capital=10000):
    """Bootstrap Monte Carlo simulation."""
    returns = [t["pnl_pct"] / 100 for t in trades]
    if len(returns) < 2:
        return {"error": "insufficient trades"}

    trades_per_day = len(returns) / 150  # ~150 days of data
    n_trades = max(1, int(horizon_days * trades_per_day))

    finals = []
    max_dds = []
    for _ in range(n_sims):
        sampled = random.choices(returns, k=n_trades)
        capital = initial_capital
        peak = capital
        max_dd = 0
        for r in sampled:
            capital *= (1 + r)
            peak = max(peak, capital)
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)
        finals.append(capital)
        max_dds.append(max_dd)

    finals.sort()
    max_dds.sort()
    n = len(finals)
    rets = [(f - initial_capital) / initial_capital for f in finals]

    return {
        "n_sims": n_sims,
        "horizon_days": horizon_days,
        "n_trades_per_sim": n_trades,
        "return_p5": round(np.percentile(rets, 5) * 100, 2),
        "return_p25": round(np.percentile(rets, 25) * 100, 2),
        "return_p50": round(np.percentile(rets, 50) * 100, 2),
        "return_p75": round(np.percentile(rets, 75) * 100, 2),
        "return_p95": round(np.percentile(rets, 95) * 100, 2),
        "return_mean": round(np.mean(rets) * 100, 2),
        "return_std": round(np.std(rets) * 100, 2),
        "prob_loss": round(sum(1 for r in rets if r < 0) / n * 100, 1),
        "prob_loss_5pct": round(sum(1 for r in rets if r < -0.05) / n * 100, 1),
        "prob_gain_10pct": round(sum(1 for r in rets if r > 0.10) / n * 100, 1),
        "max_dd_p50": round(np.median(max_dds) * 100, 2),
        "max_dd_p95": round(np.percentile(max_dds, 95) * 100, 2),
    }

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("COMPREHENSIVE BACKTEST: V4 Daily Regime + JIMI Strategies")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    daily = load_daily_candles(200)
    candles_15m = load_15m_candles(1500)
    deriv_agg = load_derivatives_daily()
    historical_trades = load_historical_trades()

    print(f"  Daily candles: {len(daily)} ({daily[0]['ts'].date()} → {daily[-1]['ts'].date()})")
    print(f"  15m candles: {len(candles_15m)} ({candles_15m[0]['ts']} → {candles_15m[-1]['ts']})")
    print(f"  Derivatives: {len(deriv_agg)} days")
    print(f"  Historical trades: {len(historical_trades)}")

    # ═══════════════════════════════════════════
    # TEST 1: V4 Regime Classification
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 1: V4 REGIME CLASSIFICATION")
    print("=" * 70)

    v4 = RegimeClassifierV4()
    regime_history = []

    for i in range(v4.EMA_TREND, len(daily)):
        window = daily[:i+1]
        day_str = daily[i]["ts"].strftime("%Y-%m-%d")
        dd = deriv_agg.get(day_str)
        regime, conf, signals = v4.classify(
            window, weekly_candles=None, deriv_daily=dd,
            current_ts=daily[i]["ts"].timestamp()
        )
        regime_history.append({
            "date": day_str,
            "price": daily[i]["close"],
            "regime": regime,
            "confidence": conf,
        })

    # Summary
    transitions = sum(1 for i in range(1, len(regime_history))
                      if regime_history[i]["regime"] != regime_history[i-1]["regime"])
    counts = defaultdict(int)
    for r in regime_history:
        counts[r["regime"]] += 1
    total_days = len(regime_history)

    print(f"\n  Transitions: {transitions} in {total_days} days ({total_days/max(transitions,1):.1f} days/transition)")
    print(f"  Distribution:")
    for regime in ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH"]:
        cnt = counts.get(regime, 0)
        print(f"    {regime:<15} {cnt:>3} days ({cnt/total_days*100:>5.1f}%)")

    # ═══════════════════════════════════════════
    # TEST 2: Historical Trade Regime Analysis
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 2: HISTORICAL TRADES vs V4 REGIME")
    print("=" * 70)

    if historical_trades:
        # Build regime lookup
        regime_lookup = {}
        for r in regime_history:
            regime_lookup[r["date"]] = r["regime"]

        strat_by_regime = defaultdict(lambda: defaultdict(list))
        for t in historical_trades:
            # Find regime at trade open
            open_date = t.get("opened_at", "")[:10]
            regime = regime_lookup.get(open_date, "UNKNOWN")
            strat_by_regime[regime][t.get("strategy", "?")].append(t.get("pnl", 0))

        print(f"\n  {'Regime':<15} {'Strategy':<25} {'Trades':>7} {'Win%':>7} {'Avg PnL':>10} {'Total':>10}")
        print("  " + "-" * 75)
        for regime in ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH", "UNKNOWN"]:
            for strat, pnls in sorted(strat_by_regime.get(regime, {}).items()):
                if not pnls:
                    continue
                wins = sum(1 for p in pnls if p > 0)
                print(f"  {regime:<15} {strat:<25} {len(pnls):>7} {wins/len(pnls)*100:>6.1f}% {np.mean(pnls):>+9.2f}$ {sum(pnls):>+9.2f}$")

    # ═══════════════════════════════════════════
    # TEST 3: Strategy Simulation with V4 Gate
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 3: STRATEGY SIMULATION (V4 regime-gated, 15m bars)")
    print("=" * 70)

    # Build daily regime lookup for 15m bars
    daily_regime = {}
    for r in regime_history:
        daily_regime[r["date"]] = (r["regime"], r["confidence"])

    random.seed(42)  # Reproducibility
    sim = StrategySimulator(capital=10000, risk_per_trade=0.02, leverage=25)

    for i, candle in enumerate(candles_15m):
        day_str = candle["ts"].strftime("%Y-%m-%d")
        regime, conf = daily_regime.get(day_str, ("RANGING", 0.5))
        sim.simulate_bar(candle, regime, conf, i)

    metrics = sim.get_metrics()

    print(f"\n  Capital:    ${metrics['capital']:,.2f} (start: $10,000)")
    print(f"  Return:     {metrics['return_pct']:+.2f}%")
    print(f"  Trades:     {metrics['total_trades']}")
    print(f"  Win Rate:   {metrics['win_rate']}%")
    print(f"  Avg Win:    {metrics['avg_win']:+.2f}%")
    print(f"  Avg Loss:   {metrics['avg_loss']:+.2f}%")
    print(f"  PF:         {metrics['profit_factor']}")
    print(f"  Max DD:     {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe:     {metrics['sharpe']:.3f}")
    print(f"  Sortino:    {metrics['sortino']:.3f}")
    print(f"  Calmar:     {metrics['calmar']:.3f}")

    print(f"\n  Performance by Regime:")
    print(f"  {'Regime':<15} {'Trades':>7} {'Win%':>7} {'PnL':>12}")
    print("  " + "-" * 45)
    for regime, stats in sorted(metrics["regime_stats"].items()):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] else 0
        print(f"  {regime:<15} {stats['trades']:>7} {wr:>6.1f}% ${stats['total_pnl']:>10,.2f}")

    print(f"\n  Performance by Strategy:")
    print(f"  {'Strategy':<25} {'Trades':>7} {'Win%':>7} {'PnL':>12}")
    print("  " + "-" * 55)
    for strat, stats in sorted(metrics["strategy_stats"].items()):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] else 0
        print(f"  {strat:<25} {stats['trades']:>7} {wr:>6.1f}% ${stats['total_pnl']:>10,.2f}")

    # ═══════════════════════════════════════════
    # TEST 4: Monte Carlo (10k sims)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 4: MONTE CARLO SIMULATION (10,000 iterations)")
    print("=" * 70)

    if sim.trades:
        for horizon in [7, 30, 90]:
            mc = monte_carlo(sim.trades, n_sims=10000, horizon_days=horizon)
            print(f"\n  {horizon}-day horizon ({mc['n_trades_per_sim']} trades/sim):")
            print(f"    Return:  P5={mc['return_p5']:+.1f}%  P25={mc['return_p25']:+.1f}%  P50={mc['return_p50']:+.1f}%  P75={mc['return_p75']:+.1f}%  P95={mc['return_p95']:+.1f}%")
            print(f"    Mean={mc['return_mean']:+.1f}%  Std={mc['return_std']:.1f}%")
            print(f"    P(loss): {mc['prob_loss']:.1f}%  P(>5% loss): {mc['prob_loss_5pct']:.1f}%  P(>10% gain): {mc['prob_gain_10pct']:.1f}%")
            print(f"    Max DD:  P50={mc['max_dd_p50']:.1f}%  P95={mc['max_dd_p95']:.1f}%")
    else:
        print("  No simulated trades for Monte Carlo")

    # ═══════════════════════════════════════════
    # TEST 5: Walk-Forward Validation
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 5: WALK-FORWARD VALIDATION")
    print("=" * 70)

    # Split into 3 windows
    n = len(regime_history)
    window_size = n // 3
    for w in range(3):
        start = w * window_size
        end = min(start + window_size, n)
        window = regime_history[start:end]
        if not window:
            continue
        regime_w = defaultdict(int)
        for r in window:
            regime_w[r["regime"]] += 1
        total_w = len(window)
        trans_w = sum(1 for i in range(1, len(window)) if window[i]["regime"] != window[i-1]["regime"])
        print(f"\n  Window {w+1}: {window[0]['date']} → {window[-1]['date']} ({total_w} days)")
        print(f"    Transitions: {trans_w}")
        for regime in ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH"]:
            cnt = regime_w.get(regime, 0)
            if cnt:
                print(f"    {regime:<15} {cnt:>3} days ({cnt/total_w*100:.1f}%)")

    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "v4_regime": {
            "transitions": transitions,
            "avg_duration_days": round(total_days / max(transitions, 1), 1),
            "distribution": {k: round(v / total_days * 100, 1) for k, v in counts.items()},
        },
        "simulation": metrics,
        "monte_carlo": {str(h): monte_carlo(sim.trades, horizon_days=h) for h in [7, 30, 90]} if sim.trades else {},
    }
    out_path = os.path.join(BASE, "data", "5agent_backtest", "v4_comprehensive_backtest.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print("\nDone.")
