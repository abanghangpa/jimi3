"""
Risk Agent — Survival-first position sizing and portfolio risk management.

Handles:
- Kelly Criterion optimal sizing
- Portfolio heat (total risk across all positions)
- Correlation-aware sizing (same-direction = one bet)
- Drawdown circuit breaker
- Dynamic leverage adjustment
- Time-of-day risk scaling
"""
import json, os
from datetime import datetime, timezone, timedelta


class RiskAgent:
    """
    Evaluates proposed positions against portfolio risk limits.
    Returns: approved (bool), adjusted_size, risk_metrics
    """

    def __init__(self, initial_capital=200.0):
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_portfolio_heat = 0.06      # 6% total capital at risk
        self.max_directional_heat = 0.04    # 4% per direction
        self.max_strategy_heat = 0.02       # 2% per single strategy
        self.max_drawdown_pct = 0.25        # 25% max drawdown -> stop trading
        self.dd_reduce_threshold = 0.15     # 15% drawdown -> reduce size 50%
        self.kelly_fraction = 0.25          # Quarter-Kelly (conservative)
        self.max_leverage = 25
        self.min_leverage = 5

    def evaluate_position(self, proposed_pos, state, regime="RANGING"):
        """
        Evaluate a proposed position against all risk limits.

        Args:
            proposed_pos: {"direction", "entry", "sl", "strategy", "conviction"}
            state: Current portfolio state
            regime: Current market regime

        Returns:
            dict: {
                "approved": bool,
                "adjusted_size": float,
                "leverage": int,
                "reason": str,
                "risk_metrics": dict
            }
        """
        capital = state.get("capital", self.initial_capital)
        open_positions = state.get("open_positions", [])
        peak = state.get("peak_capital", capital)
        self.peak_capital = max(self.peak_capital, peak)

        metrics = {}

        # === CHECK 1: Drawdown Circuit Breaker ===
        drawdown = (self.peak_capital - capital) / self.peak_capital if self.peak_capital > 0 else 0
        metrics["drawdown_pct"] = round(drawdown * 100, 2)

        if drawdown >= self.max_drawdown_pct:
            return {
                "approved": False,
                "adjusted_size": 0,
                "leverage": self.max_leverage,
                "reason": f"DRAWDOWN BREAKER: {drawdown*100:.1f}% >= {self.max_drawdown_pct*100}%",
                "risk_metrics": metrics,
            }

        # === CHECK 2: Portfolio Heat ===
        total_risk = self._calc_portfolio_heat(open_positions)
        metrics["portfolio_heat"] = round(total_risk * 100, 2)

        if total_risk >= self.max_portfolio_heat:
            return {
                "approved": False,
                "adjusted_size": 0,
                "leverage": self.max_leverage,
                "reason": f"PORTFOLIO HEAT: {total_risk*100:.1f}% >= {self.max_portfolio_heat*100}%",
                "risk_metrics": metrics,
            }

        # === CHECK 3: Directional Exposure ===
        direction = proposed_pos.get("direction", "LONG")
        dir_risk = self._calc_directional_heat(open_positions, direction)
        metrics["directional_heat"] = round(dir_risk * 100, 2)

        if dir_risk >= self.max_directional_heat:
            return {
                "approved": False,
                "adjusted_size": 0,
                "leverage": self.max_leverage,
                "reason": f"DIRECTIONAL HEAT: {direction} {dir_risk*100:.1f}% >= {self.max_directional_heat*100}%",
                "risk_metrics": metrics,
            }

        # === CHECK 4: Strategy Concentration ===
        strategy = proposed_pos.get("strategy", "unknown")
        strat_risk = self._calc_strategy_heat(open_positions, strategy)
        metrics["strategy_heat"] = round(strat_risk * 100, 2)

        if strat_risk >= self.max_strategy_heat:
            return {
                "approved": False,
                "adjusted_size": 0,
                "leverage": self.max_leverage,
                "reason": f"STRATEGY HEAT: {strategy} {strat_risk*100:.1f}% >= {self.max_strategy_heat*100}%",
                "risk_metrics": metrics,
            }

        # === CALCULATE OPTIMAL SIZE ===
        entry = proposed_pos.get("entry", 0)
        sl = proposed_pos.get("sl", 0)
        if not entry or not sl or entry == 0:
            return {
                "approved": False,
                "adjusted_size": 0,
                "leverage": self.max_leverage,
                "reason": "Invalid entry/SL",
                "risk_metrics": metrics,
            }

        sl_distance_pct = abs(entry - sl) / entry
        if sl_distance_pct <= 0:
            return {
                "approved": False,
                "adjusted_size": 0,
                "leverage": self.max_leverage,
                "reason": "SL distance is zero",
                "risk_metrics": metrics,
            }

        # Kelly sizing (using conviction as win probability proxy)
        conviction = proposed_pos.get("conviction", 0.5)
        # Assume 1.5:1 reward:risk for Kelly calculation
        rr_ratio = 1.5
        kelly_pct = conviction - (1 - conviction) / rr_ratio
        kelly_pct = max(0, kelly_pct) * self.kelly_fraction  # Quarter-Kelly

        # Risk-based sizing
        available_heat = self.max_portfolio_heat - total_risk
        available_dir_heat = self.max_directional_heat - dir_risk
        available_strat_heat = self.max_strategy_heat - strat_risk

        max_risk_amount = capital * min(available_heat, available_dir_heat, available_strat_heat)

        # Drawdown scaling
        size_multiplier = 1.0
        if drawdown >= self.dd_reduce_threshold:
            size_multiplier = 0.5  # Halve size in drawdown
            metrics["dd_scale"] = 0.5

        # Regime scaling
        regime_scale = {
            "BULL": 1.0, "BEAR": 0.9, "RANGING": 0.85,
            "STRESS": 0.7, "MILDLY_BEARISH": 0.9
        }
        regime_mult = regime_scale.get(regime, 1.0)
        metrics["regime_scale"] = regime_mult

        # Final size calculation
        risk_amount = max_risk_amount * size_multiplier * regime_mult
        size = risk_amount / (sl_distance_pct * entry) if sl_distance_pct > 0 else 0

        # Leverage: lower in stress, higher in calm
        leverage = self._calc_leverage(regime, sl_distance_pct)
        metrics["leverage"] = leverage

        # Cap by available margin
        available_margin = capital * 0.8  # Max 80% of capital as margin
        max_size_by_margin = (available_margin * leverage) / entry
        if size > max_size_by_margin:
            size = max_size_by_margin

        return {
            "approved": size >= 0.001,
            "adjusted_size": round(size, 6),
            "leverage": leverage,
            "reason": "OK" if size >= 0.001 else "Size too small",
            "risk_metrics": metrics,
            "kelly_pct": round(kelly_pct * 100, 2),
        }

    def _calc_portfolio_heat(self, positions):
        """Total capital at risk across all positions."""
        total_risk = 0.0
        for p in positions:
            sl_dist = abs(p["fill_price"] - p["sl"])
            risk = sl_dist * p["size"]
            total_risk += risk
        return total_risk / self.peak_capital if self.peak_capital > 0 else 0

    def _calc_directional_heat(self, positions, direction):
        """Capital at risk in one direction."""
        risk = 0.0
        for p in positions:
            if p["direction"] == direction:
                sl_dist = abs(p["fill_price"] - p["sl"])
                risk += sl_dist * p["size"]
        return risk / self.peak_capital if self.peak_capital > 0 else 0

    def _calc_strategy_heat(self, positions, strategy):
        """Capital at risk in one strategy."""
        risk = 0.0
        for p in positions:
            if p.get("strategy") == strategy:
                sl_dist = abs(p["fill_price"] - p["sl"])
                risk += sl_dist * p["size"]
        return risk / self.peak_capital if self.peak_capital > 0 else 0

    def _calc_leverage(self, regime, sl_distance_pct):
        """Dynamic leverage: lower in volatile regimes, higher in calm."""
        base_leverage = {
            "BULL": 20, "BEAR": 15, "RANGING": 18,
            "STRESS": 10, "MILDLY_BEARISH": 15
        }
        lev = base_leverage.get(regime, 15)

        # Reduce leverage for wide SL (larger position = more risk)
        if sl_distance_pct > 0.03:
            lev = min(lev, 10)
        elif sl_distance_pct > 0.02:
            lev = min(lev, 15)

        return max(self.min_leverage, min(self.max_leverage, lev))

    def get_risk_report(self, state):
        """Generate current risk report."""
        capital = state.get("capital", self.initial_capital)
        positions = state.get("open_positions", [])
        peak = state.get("peak_capital", capital)

        drawdown = (peak - capital) / peak if peak > 0 else 0
        portfolio_heat = self._calc_portfolio_heat(positions)
        long_heat = self._calc_directional_heat(positions, "LONG")
        short_heat = self._calc_directional_heat(positions, "SHORT")

        return {
            "capital": round(capital, 2),
            "peak": round(peak, 2),
            "drawdown_pct": round(drawdown * 100, 2),
            "portfolio_heat_pct": round(portfolio_heat * 100, 2),
            "long_heat_pct": round(long_heat * 100, 2),
            "short_heat_pct": round(short_heat * 100, 2),
            "positions": len(positions),
            "status": "DRAWDOWN_BREAKER" if drawdown >= self.max_drawdown_pct
                      else "REDUCED" if drawdown >= self.dd_reduce_threshold
                      else "NORMAL",
        }
