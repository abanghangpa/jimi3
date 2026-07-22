"""
Orchestrator — Wires all agents together and makes final trading decisions.

Flow:
1. Structure Agent -> assesses market microstructure
2. Regime Classifier -> determines market regime (already exists)
3. Validation Agent -> isolation gate + monitor (already exists)
4. Risk Agent -> position sizing and portfolio limits
5. Execution Agent -> entry validation and slippage
6. Orchestrator -> final GO/NO-GO decision

This replaces the monolithic signal processing loop in scanner_executor.py
with a clean, modular pipeline.
"""
import json, os, sys
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from src.agents.structure_agent import StructureAgent
from src.agents.risk_agent import RiskAgent
from src.agents.execution_agent import ExecutionAgent


class Orchestrator:
    """
    Master coordinator that runs all agents in sequence and makes
    the final trade/no-trade decision.

    Replaces the inline logic in scanner_executor.py's main loop.
    """

    def __init__(self, initial_capital=200.0):
        self.structure = StructureAgent()
        self.risk = RiskAgent(initial_capital)
        self.execution = ExecutionAgent()

        # Decision history for analysis
        self.decisions = []
        self.stats = {
            "total_evaluated": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "rejection_reasons": {},
        }

    def evaluate_signal(self, signal, state, gate, monitor, regime="RANGING",
                        regime_confidence=0.5, exchange=None, scan_data=None,
                        deriv_data=None, symbol="ETH/USDT:USDT"):
        """
        Full evaluation pipeline for a proposed signal.

        Args:
            signal: dict with strategy, direction, entry, sl, tp1, conviction, timestamp, cfg
            state: Current portfolio state (capital, positions, etc.)
            gate: IsolationGateGuardian instance
            monitor: LivePerformanceMonitor instance
            regime: Current market regime
            regime_confidence: Confidence in regime classification
            exchange: ccxt exchange instance
            scan_data: Latest scan JSON
            deriv_data: Derivatives history
            symbol: Trading pair

        Returns:
            dict: {
                "approved": bool,
                "action": "OPEN"|"SKIP"|"REJECT",
                "position": dict or None,
                "reason": str,
                "agent_results": dict
            }
        """
        self.stats["total_evaluated"] += 1
        agent_results = {}
        strategy = signal.get("strategy", "unknown")
        direction = signal.get("direction", "LONG")

        # ========================================
        # STAGE 1: Structure Agent Assessment
        # ========================================
        if scan_data:
            structure = self.structure.assess(scan_data, deriv_data)
            agent_results["structure"] = structure

            # Check if structure agrees with signal direction
            if structure["direction"] != "NEUTRAL" and structure["direction"] != direction:
                if structure["conviction"] > 0.6:
                    return self._reject("STRUCTURE_DISAGREES",
                        f"Structure says {structure['direction']} ({structure['conviction']:.2f}) vs signal {direction}",
                        agent_results)

        # ========================================
        # STAGE 2: Validation Agent (Gate + Monitor)
        # ========================================
        gate_ok, gate_reason = gate.is_passed(strategy)
        if not gate_ok:
            return self._reject("GATE_BLOCKED", f"{strategy}: {gate_reason}", agent_results)

        monitor_paused, monitor_reason = monitor.is_paused(strategy)
        if monitor_paused:
            return self._reject("MONITOR_PAUSED", f"{strategy}: {monitor_reason}", agent_results)

        agent_results["validation"] = {"gate": gate_ok, "monitor": not monitor_paused}

        # ========================================
        # STAGE 3: Risk Agent Sizing
        # ========================================
        proposed_pos = {
            "direction": direction,
            "entry": signal.get("entry", 0),
            "sl": signal.get("sl", 0),
            "strategy": strategy,
            "conviction": signal.get("conviction", 0.5),
        }

        risk_result = self.risk.evaluate_position(proposed_pos, state, regime)
        agent_results["risk"] = risk_result

        if not risk_result["approved"]:
            return self._reject("RISK_REJECTED", risk_result["reason"], agent_results)

        # ========================================
        # STAGE 4: Execution Agent Validation
        # ========================================
        exec_result = self.execution.validate_execution(signal, state, exchange, symbol)
        agent_results["execution"] = exec_result

        if not exec_result["execute"]:
            return self._reject("EXECUTION_REJECTED", exec_result["reason"], agent_results)

        # ========================================
        # STAGE 5: Final Decision
        # ========================================
        size = risk_result["adjusted_size"]
        leverage = risk_result["leverage"]
        adjusted_entry = exec_result["adjusted_entry"]

        # Apply regime TP/SL scaling
        tp1 = signal.get("tp1", 0)
        sl = signal.get("sl", 0)
        cfg = signal.get("cfg", {})

        position = {
            "strategy": strategy,
            "direction": direction,
            "fill_price": round(adjusted_entry, 2),
            "tp": round(tp1, 2),
            "sl": round(sl, 2),
            "size": round(size, 6),
            "leverage": leverage,
            "hold_hours": cfg.get("hold_hours", 8),
            "tp_pct": cfg.get("tp_pct", 2.0),
            "sl_pct": cfg.get("sl_pct", 1.5),
            "signal_ts": signal.get("timestamp", ""),
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "order_id": None,
            "trailed": False,
            "agent_score": {
                "structure": agent_results.get("structure", {}).get("conviction", 0),
                "risk_kelly": risk_result.get("kelly_pct", 0),
                "exec_quality": exec_result.get("quality_score", 0),
                "slippage": exec_result.get("slippage_est", 0),
            },
        }

        self.stats["total_approved"] += 1
        return {
            "approved": True,
            "action": "OPEN",
            "position": position,
            "reason": "All agents approved",
            "agent_results": agent_results,
        }

    def _reject(self, reason_code, detail, agent_results):
        """Record rejection and return result."""
        self.stats["total_rejected"] += 1
        self.stats["rejection_reasons"][reason_code] =             self.stats["rejection_reasons"].get(reason_code, 0) + 1

        return {
            "approved": False,
            "action": "REJECT",
            "position": None,
            "reason": f"{reason_code}: {detail}",
            "agent_results": agent_results,
        }

    def get_report(self):
        """Generate orchestrator performance report."""
        total = self.stats["total_evaluated"]
        return {
            "total_evaluated": total,
            "approved": self.stats["total_approved"],
            "rejected": self.stats["total_rejected"],
            "approval_rate": f"{self.stats['total_approved']/total*100:.1f}%" if total > 0 else "N/A",
            "rejection_breakdown": self.stats["rejection_reasons"],
            "risk_report": None,  # Filled by caller
            "execution_stats": self.execution.get_execution_stats(),
            "structure_trend": self.structure.get_trend(),
        }
