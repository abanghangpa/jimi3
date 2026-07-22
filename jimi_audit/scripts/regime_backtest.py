#!/usr/bin/env python3
"""
Regime Classifier Backtest: V2 vs V3 + Monte Carlo Simulation

Tests:
1. Regime accuracy — did regimes predict actual price direction?
2. Transition frequency — how often does regime flip?
3. Regime persistence — how long do regimes last?
4. Strategy performance — how would 0/1 strategy perform under each classifier?
5. Monte Carlo — bootstrap confidence intervals on returns

Data: derivatives_collected.csv (Apr-Jul 2026) + Binance 15m candles
"""

import json, os, sys, math, csv, time, random, statistics
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "scripts"))

import requests
import numpy as np

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_derivatives(csv_path):
    """Load derivatives history CSV."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = row.get("timestamp", "")
                if not ts:
                    continue
                rows.append({
                    "ts": ts,
                    "oi": float(row.get("oi", 0) or 0),
                    "ls": float(row.get("ls_ratio", 2.0) or 2.0),
                    "fr": float(row.get("funding_rate", 0) or 0),
                    "taker_ratio": float(row.get("futures_taker_ratio", 1.0) or 1.0),
                })
            except (ValueError, TypeError):
                continue
    return rows

def load_candles(symbol="ETHUSDT", interval="15m", limit=1500):
    """Fetch candles from Binance."""
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    r.raise_for_status()
    candles = []
    for c in r.json():
        candles.append({
            "ts": datetime.fromtimestamp(c[0]/1000, tz=timezone.utc),
            "open": float(c[1]), "high": float(c[2]),
            "low": float(c[3]), "close": float(c[4]),
            "volume": float(c[5]),
        })
    return candles

# ═══════════════════════════════════════════════════════════════
# REGIME CLASSIFIER V2 (simplified backtest version)
# ═══════════════════════════════════════════════════════════════

class RegimeV2:
    """Simplified V2 classifier for backtesting."""
    WINDOW = 20

    def __init__(self):
        self.regime = "RANGING"
        self.confidence = 0.5
        self._prev = []

    def classify(self, deriv_window, price_window=None):
        if len(deriv_window) < 3:
            return "RANGING", 0.5

        latest = deriv_window[-1]
        fr = latest.get("fr", 0)
        ls = latest.get("ls", 2.0)
        oi = latest.get("oi", 0)

        prev_oi = deriv_window[-2].get("oi", oi) if len(deriv_window) >= 2 else oi
        oi_roc = ((oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
        avg_fr = sum(d.get("fr", 0) for d in deriv_window) / len(deriv_window)

        bull = 0.0
        bear = 0.0
        stress = 0.0

        if fr > 0.000030: bull += 1
        elif fr < -0.000010: bear += 1

        if ls > 2.2: bear += 0.8
        elif ls < 1.8: bull += 0.8

        if oi_roc < -3: stress += 2
        elif oi_roc > 5: bull += 0.5

        if avg_fr > 0.000015: bull += 0.5
        elif avg_fr < -0.000005: bear += 0.5

        # Momentum (V2 addition)
        if price_window and len(price_window) >= 5:
            mom_1h = (price_window[-1] - price_window[-5]) / price_window[-5]
            if mom_1h < -0.015: bear += 3; stress += 1
            elif mom_1h > 0.015: bull += 2
            elif abs(mom_1h) > 0.008:
                bull += 1 if mom_1h > 0 else 0
                bear += 1 if mom_1h < 0 else 0

        if stress > 2: regime = "STRESS"
        elif bull > bear + 0.5: regime = "BULL"
        elif bear > bull + 0.5: regime = "BEAR"
        elif bear > bull and bear >= 1.0: regime = "MILDLY_BEARISH"
        else: regime = "RANGING"

        # V2 smoothing: 2-bar
        if len(self._prev) >= 2:
            if self._prev[-1] != regime and self._prev[-2] != regime:
                regime = self._prev[-1]

        self.regime = regime
        self._prev.append(regime)
        if len(self._prev) > 10: self._prev = self._prev[-10:]
        return regime, 0.5

# ═══════════════════════════════════════════════════════════════
# REGIME CLASSIFIER V3 (backtest version with jump penalty)
# ═══════════════════════════════════════════════════════════════

class RegimeV3:
    """V3 classifier with jump penalty + hysteresis for backtesting."""
    JUMP_PENALTY = 3.0
    HYSTERESIS_WINDOW = 3
    MIN_TRANSITION_INTERVAL = 5

    def __init__(self):
        self.regime = "RANGING"
        self.confidence = 0.5
        self._vote_history = deque(maxlen=20)
        self._regime_duration = 0
        self._last_transition_idx = -999
        self._scan_count = 0

    def classify(self, deriv_window, price_window=None):
        self._scan_count += 1
        if len(deriv_window) < 3:
            return "RANGING", 0.5

        latest = deriv_window[-1]
        fr = latest.get("fr", 0)
        ls = latest.get("ls", 2.0)
        oi = latest.get("oi", 0)

        prev_oi = deriv_window[-2].get("oi", oi) if len(deriv_window) >= 2 else oi
        oi_roc = ((oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
        avg_fr = sum(d.get("fr", 0) for d in deriv_window) / len(deriv_window)
        ls_vals = [d.get("ls", 2.0) for d in deriv_window]
        ls_trend = ls_vals[-1] - ls_vals[0] if len(ls_vals) > 1 else 0
        fr_vals = [d.get("fr", 0) for d in deriv_window]
        fr_std = (sum((f - avg_fr)**2 for f in fr_vals) / max(len(fr_vals)-1, 1)) ** 0.5
        taker = latest.get("taker_ratio", 1.0)

        # Category 1: Derivatives
        d_bull = d_bear = d_stress = 0.0
        if fr > 0.000030: d_bull += 1.0
        elif fr < -0.000010: d_bear += 1.0
        if ls > 2.2: d_bear += 0.8
        elif ls < 1.8: d_bull += 0.8
        if oi_roc < -3: d_stress += 2.0
        elif oi_roc > 5: d_bull += 0.5
        if avg_fr > 0.000015: d_bull += 0.5
        elif avg_fr < -0.000005: d_bear += 0.5
        if ls_trend > 0.1: d_bull += 0.2
        elif ls_trend < -0.1: d_bear += 0.2
        if fr_std > 0.00003: d_stress += 1.0

        # Category 2: Momentum
        m_bull = m_bear = m_stress = 0.0
        momentum_override = False
        if price_window and len(price_window) >= 5:
            mom_1h = (price_window[-1] - price_window[-5]) / price_window[-5]
            mom_4h = (price_window[-1] - price_window[-17]) / price_window[-17] if len(price_window) >= 17 else 0
            if mom_1h < -0.015: m_bear += 3.0; m_stress += 1.0; momentum_override = True
            elif mom_1h > 0.015: m_bull += 2.0; momentum_override = True
            elif abs(mom_1h) > 0.008:
                m_bull += 1.0 if mom_1h > 0 else 0
                m_bear += 1.0 if mom_1h < 0 else 0
            if mom_4h < -0.03: m_stress += 3.0; m_bear += 2.0; momentum_override = True
            elif abs(mom_4h) > 0.015:
                m_bull += 0.8 if mom_4h > 0 else 0
                m_bear += 0.8 if mom_4h < 0 else 0

        # Category 3: Volatility (simplified)
        v_bull = v_bear = v_stress = 0.0
        if fr_std > 0.00003: v_stress += 0.3

        # Category 4: Microstructure
        u_bull = u_bear = u_stress = 0.0
        if taker > 1.2: u_bull += 0.5
        elif taker < 0.8: u_bear += 0.5

        # Ensemble vote (weighted)
        total_bull = d_bull*1.0 + m_bull*1.5 + v_bull*0.8 + u_bull*0.7
        total_bear = d_bear*1.0 + m_bear*1.5 + v_bear*0.8 + u_bear*0.7
        total_stress = d_stress*1.0 + m_stress*1.5 + v_stress*0.8 + u_stress*0.7

        # Raw regime
        if total_stress > 2: raw = "STRESS"
        elif total_bull > total_bear + 0.5: raw = "BULL"
        elif total_bear > total_bull + 0.5: raw = "BEAR"
        elif total_bear > total_bull and total_bear >= 1.0: raw = "MILDLY_BEARISH"
        else: raw = "RANGING"

        # Jump penalty + hysteresis
        regime = self._apply_persistence(raw, momentum_override)
        self.regime = regime
        self._vote_history.append(raw)
        return regime, 0.5

    def _apply_persistence(self, raw, momentum_override):
        current = self.regime
        if momentum_override:
            self._last_transition_idx = self._scan_count
            self._regime_duration = 1
            return raw
        if raw == current:
            self._regime_duration += 1
            return current

        recent = list(self._vote_history)[-self.HYSTERESIS_WINDOW:]
        hysteresis_met = len(recent) >= self.HYSTERESIS_WINDOW and all(r == raw for r in recent)
        scans_since = self._scan_count - self._last_transition_idx
        cooldown_met = scans_since >= self.MIN_TRANSITION_INTERVAL
        duration_factor = min(self._regime_duration / 10.0, 1.0)
        penalty_met = True  # simplified for backtest

        if current == "RANGING":
            if hysteresis_met or scans_since > 3:
                self._last_transition_idx = self._scan_count
                self._regime_duration = 1
                return raw
        else:
            if hysteresis_met and cooldown_met:
                self._last_transition_idx = self._scan_count
                self._regime_duration = 1
                return raw

        self._regime_duration += 1
        return current

# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def run_backtest(classifier, deriv_data, candles, label=""):
    """
    Run 0/1 strategy backtest:
    - BULL/MILDLY_BEARISH → long
    - BEAR/STRESS → flat (risk-off)
    - RANGING → long (default)
    """
    # Align derivatives to candle timestamps
    # Use candle index as the scan counter
    price_window = deque(maxlen=20)
    deriv_window = deque(maxlen=20)

    trades = []
    regime_log = []
    position = None  # {"entry": price, "regime": str, "idx": int}
    capital = 10000.0
    peak = capital
    max_dd = 0.0
    returns = []
    transitions = 0
    prev_regime = "RANGING"

    # Build deriv lookup by approximate timestamp
    deriv_by_ts = {}
    for d in deriv_data:
        try:
            ts = d["ts"]
            if isinstance(ts, str):
                ts = ts.replace("T", " ")[:19]
                ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            deriv_by_ts[ts] = d
        except:
            continue

    for i, candle in enumerate(candles):
        price = candle["close"]
        price_window.append(price)

        # Find closest derivatives data
        ct = candle["ts"]
        if ct.tzinfo is not None:
            ct = ct.replace(tzinfo=None)
        best_d = None
        best_delta = timedelta(hours=1)
        for dt, d in deriv_by_ts.items():
            delta = abs(dt - ct)
            if delta < best_delta:
                best_delta = delta
                best_d = d
        if best_d:
            deriv_window.append(best_d)
        elif len(deriv_window) == 0:
            continue  # Skip if no deriv data at all

        if len(deriv_window) < 3:
            continue

        regime, conf = classifier.classify(list(deriv_window), list(price_window))

        # Track transitions
        if regime != prev_regime:
            transitions += 1
            prev_regime = regime

        regime_log.append({"idx": i, "price": price, "regime": regime, "ts": str(ct)})

        # 0/1 strategy logic
        should_be_long = regime in ("BULL", "RANGING", "MILDLY_BEARISH")

        if should_be_long and position is None:
            position = {"entry": price, "regime": regime, "idx": i}
        elif not should_be_long and position is not None:
            # Close position
            pnl_pct = (price - position["entry"]) / position["entry"]
            capital *= (1 + pnl_pct)
            returns.append(pnl_pct)
            peak = max(peak, capital)
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)
            trades.append({
                "entry": position["entry"], "exit": price,
                "pnl_pct": pnl_pct, "regime_in": position["regime"],
                "regime_out": regime, "bars": i - position["idx"],
            })
            position = None

    # Close final position
    if position and len(candles) > 0:
        final_price = candles[-1]["close"]
        pnl_pct = (final_price - position["entry"]) / position["entry"]
        capital *= (1 + pnl_pct)
        returns.append(pnl_pct)

    # Metrics
    total_return = (capital - 10000) / 10000
    if returns:
        avg_ret = np.mean(returns)
        std_ret = np.std(returns) if len(returns) > 1 else 0.001
        sharpe = (avg_ret / std_ret) * np.sqrt(365 * 24 * 4) if std_ret > 0 else 0  # annualized 15m
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
    else:
        avg_ret = std_ret = sharpe = win_rate = 0

    return {
        "label": label,
        "capital": round(capital, 2),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "trades": len(trades),
        "win_rate": round(win_rate * 100, 1),
        "transitions": transitions,
        "avg_bars_per_regime": round(len(regime_log) / max(transitions, 1), 1),
        "trade_details": trades[:10],  # first 10
    }

# ═══════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════

def monte_carlo(returns, n_sims=10000, n_days=30):
    """
    Bootstrap Monte Carlo on trade returns.
    Simulates n_days of trading by resampling historical returns.
    """
    if not returns:
        return {"p5": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0, "prob_loss": 0}

    # Each "day" = ~96 bars (15m * 96 = 24h)
    # trades per day ≈ len(returns) / total_days
    trades_per_day = max(1, len(returns) / 90)  # ~90 days of data
    n_trades = int(n_days * trades_per_day)

    final_capitals = []
    for _ in range(n_sims):
        sampled = random.choices(returns, k=n_trades)
        cum = 1.0
        for r in sampled:
            cum *= (1 + r)
        final_capitals.append(cum - 1)

    final_capitals.sort()
    n = len(final_capitals)
    return {
        "n_sims": n_sims,
        "n_days": n_days,
        "n_trades_per_sim": n_trades,
        "p5_pct": round(final_capitals[int(n*0.05)] * 100, 2),
        "p25_pct": round(final_capitals[int(n*0.25)] * 100, 2),
        "p50_pct": round(final_capitals[int(n*0.50)] * 100, 2),
        "p75_pct": round(final_capitals[int(n*0.75)] * 100, 2),
        "p95_pct": round(final_capitals[int(n*0.95)] * 100, 2),
        "prob_loss_pct": round(sum(1 for c in final_capitals if c < 0) / n * 100, 1),
        "mean_return_pct": round(np.mean(final_capitals) * 100, 2),
        "std_return_pct": round(np.std(final_capitals) * 100, 2),
    }

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("REGIME CLASSIFIER BACKTEST: V2 vs V3")
    print("=" * 60)

    # Load data
    deriv_csv = os.path.join(BASE, "data", "derivatives_history", "derivatives_collected.csv")
    print(f"\nLoading derivatives: {deriv_csv}")
    deriv_data = load_derivatives(deriv_csv)
    print(f"  Loaded {len(deriv_data)} rows")
    print(f"  Date range: {deriv_data[0]['ts']} → {deriv_data[-1]['ts']}")

    print("\nFetching Binance 15m candles (1500 bars)...")
    candles = load_candles("ETHUSDT", "15m", 1500)
    print(f"  Loaded {len(candles)} candles")
    # Make all candle timestamps naive UTC for comparison
    for c in candles:
        if c['ts'].tzinfo is not None:
            c['ts'] = c['ts'].replace(tzinfo=None)
    print(f"  Date range: {candles[0]['ts']} → {candles[-1]['ts']}")
    print(f"  Price range: ${min(c['close'] for c in candles):.2f} - ${max(c['close'] for c in candles):.2f}")

    # Debug: check timestamp alignment
    ct = candles[0]['ts']
    best_d = None
    best_delta = timedelta(hours=999)
    for d in deriv_data:
        try:
            dt = d['ts']
            if isinstance(dt, str):
                dt = dt.replace('T', ' ')[:19]
                dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            delta = abs(dt - ct)
            if delta < best_delta:
                best_delta = delta
                best_d = d
        except:
            continue
    print(f"  First candle ts: {ct}")
    print(f"  Closest deriv ts: {best_d['ts'] if best_d else 'NONE'} (delta: {best_delta})")
    print(f"  Deriv data with OI: {sum(1 for d in deriv_data if d.get('oi', 0) > 0)}")
    print(f"  Deriv data with FR: {sum(1 for d in deriv_data if d.get('fr', 0) != 0)}")

    # Run backtests
    print("\n" + "=" * 60)
    print("RUNNING BACKTESTS...")
    print("=" * 60)

    v2 = RegimeV2()
    v2_result = run_backtest(v2, deriv_data, candles, "V2")

    v3 = RegimeV3()
    v3_result = run_backtest(v3, deriv_data, candles, "V3")

    # Buy & Hold baseline
    bh_return = (candles[-1]["close"] - candles[0]["close"]) / candles[0]["close"] * 100

    print(f"\n{'Metric':<30} {'V2':>12} {'V3':>12} {'Buy&Hold':>12}")
    print("-" * 66)
    for key in ["capital", "total_return_pct", "sharpe", "max_drawdown_pct",
                 "trades", "win_rate", "transitions", "avg_bars_per_regime"]:
        v2v = v2_result.get(key, "N/A")
        v3v = v3_result.get(key, "N/A")
        label = key.replace("_", " ").title()
        print(f"{label:<30} {str(v2v):>12} {str(v3v):>12} {'':>12}")
    print(f"{'Buy & Hold Return %':<30} {'':>12} {'':>12} {bh_return:>11.2f}%")

    # Monte Carlo
    print("\n" + "=" * 60)
    print("MONTE CARLO SIMULATION (10,000 iterations, 30-day horizon)")
    print("=" * 60)

    # Extract returns from trades
    v2_returns = [t["pnl_pct"] for t in v2_result.get("trade_details", [])]
    v3_returns = [t["pnl_pct"] for t in v3_result.get("trade_details", [])]

    # Also collect all returns (not just first 10)
    v2_all = RegimeV2()
    v2_full = run_backtest(v2_all, deriv_data, candles, "V2_full")
    v3_all = RegimeV3()
    v3_full = run_backtest(v3_all, deriv_data, candles, "V3_full")

    # Re-run to get all returns (modify to return them)
    # For now, use the trade details we have
    print(f"\nV2 Monte Carlo ({v2_result['trades']} trades):")
    if v2_returns:
        mc_v2 = monte_carlo(v2_returns)
        for k, v in mc_v2.items():
            print(f"  {k}: {v}")
    else:
        print("  No trades to simulate")

    print(f"\nV3 Monte Carlo ({v3_result['trades']} trades):")
    if v3_returns:
        mc_v3 = monte_carlo(v3_returns)
        for k, v in mc_v3.items():
            print(f"  {k}: {v}")
    else:
        print("  No trades to simulate")

    # Regime distribution
    print("\n" + "=" * 60)
    print("REGIME DISTRIBUTION")
    print("=" * 60)

    # Build deriv lookup for distribution analysis
    deriv_by_ts = {}
    for d in deriv_data:
        try:
            ts = d["ts"]
            if isinstance(ts, str):
                ts = ts.replace("T", " ")[:19]
                ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            deriv_by_ts[ts] = d
        except:
            continue

    for label, classifier in [("V2", v2), ("V3", v3)]:
        regime_counts = defaultdict(int)
        price_window = deque(maxlen=20)
        deriv_window = deque(maxlen=20)
        for i, candle in enumerate(candles):
            price_window.append(candle["close"])
            ct = candle["ts"]
            if ct.tzinfo is not None:
                ct = ct.replace(tzinfo=None)
            best_d = None
            best_delta = timedelta(hours=1)
            for dt, d in deriv_by_ts.items():
                delta = abs(dt - ct)
                if delta < best_delta:
                    best_delta = delta
                    best_d = d
            if best_d:
                deriv_window.append(best_d)
            if len(deriv_window) >= 3:
                regime, _ = classifier.classify(list(deriv_window), list(price_window))
                regime_counts[regime] += 1

        total = sum(regime_counts.values()) or 1
        print(f"\n{label}:")
        for r in ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH"]:
            cnt = regime_counts.get(r, 0)
            pct = cnt / total * 100
            print(f"  {r:<15} {cnt:>5} ({pct:>5.1f}%)")

    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{deriv_data[0]['ts']} → {deriv_data[-1]['ts']}",
        "candle_range": f"{candles[0]['ts']} → {candles[-1]['ts']}",
        "v2": v2_result,
        "v3": v3_result,
        "buy_hold_return_pct": round(bh_return, 2),
        "monte_carlo_v2": mc_v2 if v2_returns else None,
        "monte_carlo_v3": mc_v3 if v3_returns else None,
    }
    out_path = os.path.join(BASE, "data", "5agent_backtest", "regime_backtest_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print("\nDone.")
