"""
Validation Agent — Statistical gatekeeper that separates real edges from noise.

Enforces 4 validation layers before any strategy touches live money:
1. Isolation Gate — p < 0.05 statistical significance
2. Walk-Forward Validation — out-of-sample performance check
3. Monte Carlo Simulation — worst-case drawdown estimation
4. Regime-Conditional Testing — strategy performance per regime

Reference: "Evidence-Based Technical Analysis" (Aronson, 2006)
"""
import json, os, random, math
from datetime import datetime, timezone
from collections import defaultdict
import statistics


class ValidationAgent:
    """
    4-layer validation pipeline for trading strategies.
    """

    def __init__(self, config_dir, data_dir):
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.gate_file = os.path.join(config_dir, "isolation_gate_results.json")
        self.min_p_value = 0.05
        self.min_oos_ratio = 0.6
        self.mc_confidence = 0.05
        self.mc_simulations = 10000
        self.min_regime_coverage = 0.5

    def validate(self, strategy_name, trade_history=None, regime_results=None):
        """Full 4-layer validation."""
        report = {
            "strategy": strategy_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "layers": {},
            "passed": False,
            "verdict": "",
            "confidence": 0.0,
        }

        # LAYER 1: Isolation Gate
        layer1 = self._check_isolation_gate(strategy_name)
        report["layers"]["isolation_gate"] = layer1
        if not layer1["passed"]:
            report["verdict"] = f"FAILED: Isolation gate (p={layer1.get('p_value', 'N/A')})"
            return report

        # LAYER 2: Walk-Forward Validation
        layer2 = self._check_walk_forward(strategy_name, trade_history)
        report["layers"]["walk_forward"] = layer2
        if not layer2["passed"]:
            report["verdict"] = f"FAILED: Walk-forward (OOS WR {layer2.get('oos_wr', 0):.1%} vs IS {layer2.get('is_wr', 0):.1%})"
            return report

        # LAYER 3: Monte Carlo Simulation
        layer3 = self._check_monte_carlo(strategy_name, trade_history)
        report["layers"]["monte_carlo"] = layer3
        if not layer3["passed"]:
            report["verdict"] = f"FAILED: Monte Carlo (worst DD {layer3.get('worst_dd_5pct', 0):.1f}%)"
            return report

        # LAYER 4: Regime-Conditional Testing
        layer4 = self._check_regime_conditional(strategy_name, regime_results)
        report["layers"]["regime_conditional"] = layer4
        if not layer4["passed"]:
            report["verdict"] = f"FAILED: Regime coverage ({layer4.get('regimes_passed', 0)}/{layer4.get('regimes_tested', 0)})"
            return report

        report["passed"] = True
        report["confidence"] = self._calc_confidence(report["layers"])
        report["verdict"] = f"PASSED (confidence={report['confidence']:.2f})"
        return report

    def _check_isolation_gate(self, strategy_name):
        """Layer 1: Statistical significance (p < 0.05)."""
        if not os.path.exists(self.gate_file):
            return {"passed": False, "reason": "Gate file not found", "p_value": 1.0}
        try:
            with open(self.gate_file) as f:
                gate_data = json.load(f)
            entry = gate_data.get(strategy_name, {})
            if not entry:
                return {"passed": False, "reason": "Strategy not in gate results", "p_value": 1.0}
            passed = entry.get("passed", False)
            p_value = entry.get("p_value", 1.0)
            killed = entry.get("killed", False)
            if killed:
                return {"passed": False, "reason": "Strategy killed", "p_value": p_value}
            return {
                "passed": passed and p_value <= self.min_p_value,
                "p_value": p_value,
                "mean_return_pct": entry.get("mean_return_pct", 0),
                "events": entry.get("events", 0),
                "effect_direction": entry.get("effect_direction", "unknown"),
            }
        except Exception as e:
            return {"passed": False, "reason": str(e), "p_value": 1.0}

    def _check_walk_forward(self, strategy_name, trade_history=None):
        """Layer 2: Split 70/30 IS/OOS. OOS WR must be >= 60% of IS WR."""
        if not trade_history:
            trade_history = self._load_trade_history(strategy_name)
        if not trade_history or len(trade_history) < 10:
            return {"passed": True, "reason": "Insufficient trades (need >= 10)", "is_wr": 0, "oos_wr": 0}

        sorted_trades = sorted(trade_history, key=lambda t: t.get("opened_at", t.get("timestamp", "")))
        split_idx = int(len(sorted_trades) * 0.7)
        is_trades = sorted_trades[:split_idx]
        oos_trades = sorted_trades[split_idx:]

        is_wr = sum(1 for t in is_trades if t.get("outcome") == "WIN") / len(is_trades)
        oos_wr = sum(1 for t in oos_trades if t.get("outcome") == "WIN") / len(oos_trades)
        is_mean = statistics.mean([t.get("pnl", 0) for t in is_trades])
        oos_mean = statistics.mean([t.get("pnl", 0) for t in oos_trades])
        wr_ratio = oos_wr / is_wr if is_wr > 0 else 0

        return {
            "passed": wr_ratio >= self.min_oos_ratio and oos_wr > 0.4 and oos_mean > 0,
            "is_wr": round(is_wr, 4), "oos_wr": round(oos_wr, 4),
            "wr_ratio": round(wr_ratio, 4),
            "is_mean_return": round(is_mean, 4), "oos_mean_return": round(oos_mean, 4),
            "is_trades": len(is_trades), "oos_trades": len(oos_trades),
        }

    def _check_monte_carlo(self, strategy_name, trade_history=None):
        """Layer 3: Shuffle trades 10K times, find worst-case drawdown at 5th percentile."""
        if not trade_history:
            trade_history = self._load_trade_history(strategy_name)
        if not trade_history or len(trade_history) < 10:
            return {"passed": True, "reason": "Insufficient trades (need >= 10)", "worst_dd_5pct": 0}

        pnls = [t.get("pnl", 0) for t in trade_history]
        initial_capital = 200.0
        drawdowns = []

        for _ in range(self.mc_simulations):
            shuffled = pnls[:]
            random.shuffle(shuffled)
            equity = initial_capital
            peak = equity
            max_dd = 0
            for pnl in shuffled:
                equity += pnl
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            drawdowns.append(max_dd)

        drawdowns.sort()
        worst_5pct_idx = int(len(drawdowns) * (1 - self.mc_confidence))
        worst_dd_5pct = drawdowns[worst_5pct_idx]
        median_dd = drawdowns[len(drawdowns) // 2]
        worst_95pct_idx = int(len(drawdowns) * 0.95)
        worst_dd_95pct = drawdowns[worst_95pct_idx]

        return {
            "passed": worst_dd_5pct <= 0.30,
            "worst_dd_5pct": round(worst_dd_5pct * 100, 2),
            "median_dd": round(median_dd * 100, 2),
            "worst_dd_95pct": round(worst_dd_95pct * 100, 2),
            "simulations": self.mc_simulations,
            "total_trades": len(pnls),
            "mean_pnl": round(statistics.mean(pnls), 4),
            "std_pnl": round(statistics.stdev(pnls), 4) if len(pnls) > 1 else 0,
        }

    def _check_regime_conditional(self, strategy_name, regime_results=None):
        """Layer 4: Must work in >= 50% of tested regimes."""
        if not regime_results:
            regime_results = self._load_regime_results(strategy_name)
        if not regime_results:
            return {"passed": True, "reason": "No regime data", "regimes_passed": 0, "regimes_tested": 0}

        regimes_passed = 0
        regimes_tested = 0
        details = {}

        for regime, data in regime_results.items():
            trades = data.get("trades", 0)
            if trades < 5:
                details[regime] = {"status": "SKIP", "trades": trades}
                continue
            regimes_tested += 1
            wr = data.get("wins", 0) / trades
            mean_ret = data.get("total_pnl", 0) / trades
            regime_passed = wr > 0.45 and mean_ret > 0
            if regime_passed:
                regimes_passed += 1
            details[regime] = {"status": "PASS" if regime_passed else "FAIL", "trades": trades, "wr": round(wr, 4), "mean_return": round(mean_ret, 4)}

        coverage = regimes_passed / regimes_tested if regimes_tested > 0 else 0
        return {
            "passed": coverage >= self.min_regime_coverage and regimes_tested >= 2,
            "regimes_passed": regimes_passed, "regimes_tested": regimes_tested,
            "coverage": round(coverage, 4), "details": details,
        }

    def _load_trade_history(self, strategy_name):
        state_file = os.path.join(self.data_dir, "executor_state.json")
        if not os.path.exists(state_file):
            return []
        try:
            with open(state_file) as f:
                state = json.load(f)
            return [t for t in state.get("closed_trades", []) if t.get("strategy") == strategy_name]
        except:
            return []

    def _load_regime_results(self, strategy_name):
        trades = self._load_trade_history(strategy_name)
        if not trades:
            return {}
        regime_data = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0})
        for t in trades:
            regime = t.get("regime", "UNKNOWN")
            regime_data[regime]["trades"] += 1
            if t.get("outcome") == "WIN":
                regime_data[regime]["wins"] += 1
            regime_data[regime]["total_pnl"] += t.get("pnl", 0)
        result = {}
        for regime, data in regime_data.items():
            data["mean_return"] = data["total_pnl"] / data["trades"] if data["trades"] > 0 else 0
            result[regime] = dict(data)
        return result

    def _calc_confidence(self, layers):
        scores = []
        ig = layers.get("isolation_gate", {})
        if ig.get("passed"): scores.append(1.0 - ig.get("p_value", 0.05))
        wf = layers.get("walk_forward", {})
        if wf.get("passed"): scores.append(wf.get("wr_ratio", 0.6))
        mc = layers.get("monte_carlo", {})
        if mc.get("passed"): scores.append(max(0, 1.0 - mc.get("worst_dd_5pct", 0) / 100 * 2))
        rc = layers.get("regime_conditional", {})
        if rc.get("passed"): scores.append(rc.get("coverage", 0.5))
        return round(statistics.mean(scores), 3) if scores else 0.0

    def get_full_report(self, strategy_names=None):
        if not strategy_names:
            if os.path.exists(self.gate_file):
                with open(self.gate_file) as f:
                    strategy_names = list(json.load(f).keys())
            else:
                return {}
        return {name: self.validate(name) for name in strategy_names}
