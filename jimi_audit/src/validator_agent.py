"""
JIMI Validator Agent — Phase 1 of Optimization Framework
=========================================================

Statistical validation for every strategy claim before deployment.

Implements:
  1. Walk-Forward Analysis (WFA) — rolling train/test windows
  2. Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2014)
  3. Monte Carlo Permutation Test — shuffle labels, check if real > 95th pctl
  4. Combinatorial Purged Cross-Validation (CPCV) — De Prado (2018)

Design:
  - Hybrid: portfolio-level walk-forward for gate decisions, trade-level MC for CIs
  - Every validation includes: n trades, WF WR, DSR, bootstrap 95% CI on WR & PF,
    regime breakdown
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger("jimi.validator")

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationThresholds:
    """Minimum and target values for each validation metric."""
    dsr_min: float = 1.0
    dsr_target: float = 2.0
    wf_wr_min: float = 0.50
    wf_wr_target: float = 0.60
    cpcv_score_min: float = 0.55
    cpcv_score_target: float = 0.65
    mc_pvalue_min: float = 0.05      # must be BELOW this
    mc_pvalue_target: float = 0.01
    sample_size_min: int = 30
    sample_size_target: int = 100


# Walk-forward config (bars-based)
# bar_duration_secs: seconds per bar (default 900 = 15min)
WFA_CONFIG = {
    "train_bars": 4000,
    "test_bars": 960,
    "step_bars": 960,
    "min_test_trades": 5,
    "purge_bars": 16,
    "bar_duration_secs": 900,
}

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Single trade record."""
    entry_time: float           # epoch seconds
    exit_time: float
    pnl: float                  # realised P&L (R-multiples or USD)
    side: str = "long"          # long | short
    symbol: str = ""
    regime: str = "unknown"     # optional regime tag
    strategy: str = ""

    @property
    def duration_bars(self) -> float:
        """Duration in seconds (proxy for bars when bar size unknown)."""
        return self.exit_time - self.entry_time


@dataclass
class WalkForwardResult:
    """Result of a single walk-forward fold."""
    fold_idx: int
    train_start: float
    train_end: float
    test_start: float
    test_end: float
    train_trades: int
    test_trades: int
    test_win_rate: float
    test_profit_factor: float
    test_pnl: float
    test_sharpe: float         # annualised Sharpe of test window trades


@dataclass
class WFAResult:
    """Aggregate walk-forward analysis result."""
    n_folds: int
    total_test_trades: int
    overall_win_rate: float
    overall_profit_factor: float
    overall_sharpe: float
    folds: list[WalkForwardResult] = field(default_factory=list)
    bootstrap_ci_wr: tuple[float, float] = (0.0, 0.0)
    bootstrap_ci_pf: tuple[float, float] = (0.0, 0.0)
    regime_breakdown: dict[str, dict] = field(default_factory=dict)
    pass_gate: bool = False


@dataclass
class DSRResult:
    """Deflated Sharpe Ratio result."""
    observed_sharpe: float
    sharpe_std_error: float
    n_trials: int             # number of strategies tested (for deflation)
    dsr: float
    dsr_pvalue: float
    is_significant_95: bool
    is_significant_99: bool


@dataclass
class MCResult:
    """Monte Carlo permutation test result."""
    real_metric: float
    n_permutations: int
    null_mean: float
    null_std: float
    null_distribution: list[float] = field(default_factory=list)
    p_value: float = 1.0
    percentile_rank: float = 0.0


@dataclass
class CPCVResult:
    """Combinatorial Purged Cross-Validation result."""
    n_groups: int
    n_test_groups: int
    n_combinations: int
    mean_oos_pnl: float
    oos_win_rate: float
    cpcv_score: float         # fraction of combos with positive OOS P&L
    pnl_distribution: list[float] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete validation report for a strategy."""
    strategy_name: str
    timestamp: str
    n_trades: int
    # Walk-forward
    wf_result: Optional[WFAResult] = None
    # Deflated Sharpe
    dsr_result: Optional[DSRResult] = None
    # Monte Carlo
    mc_result: Optional[MCResult] = None
    # CPCV
    cpcv_result: Optional[CPCVResult] = None
    # Bootstrap CIs
    bootstrap_ci_wr: tuple[float, float] = (0.0, 0.0)
    bootstrap_ci_pf: tuple[float, float] = (0.0, 0.0)
    # Regime breakdown
    regime_breakdown: dict[str, dict] = field(default_factory=dict)
    # Gate decision
    pass_gate: bool = False
    gate_failures: list[str] = field(default_factory=list)
    # Scores
    scores: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core Validator
# ---------------------------------------------------------------------------

class ValidatorAgent:
    """
    Statistical validation engine for trading strategies.

    Runs walk-forward analysis, deflated Sharpe ratio, Monte Carlo
    permutation tests, and combinatorial purged cross-validation.
    """

    def __init__(
        self,
        thresholds: Optional[ValidationThresholds] = None,
        wfa_config: Optional[dict] = None,
        n_mc_permutations: int = 10_000,
        n_bootstrap: int = 10_000,
        n_strategies_tested: int = 20,   # for DSR deflation
        random_seed: int = 42,
    ):
        self.thresholds = thresholds or ValidationThresholds()
        self.wfa = wfa_config or WFA_CONFIG.copy()
        self.n_mc = n_mc_permutations
        self.n_bootstrap = n_bootstrap
        self.n_strategies = n_strategies_tested
        self.rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        strategy_name: str,
        trades: list[Trade],
        run_wf: bool = True,
        run_dsr: bool = True,
        run_mc: bool = True,
        run_cpcv: bool = True,
    ) -> ValidationReport:
        """Run full validation suite on a list of trades."""
        report = ValidationReport(
            strategy_name=strategy_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            n_trades=len(trades),
        )

        if len(trades) < self.thresholds.sample_size_min:
            report.gate_failures.append(
                f"Insufficient trades: {len(trades)} < {self.thresholds.sample_size_min}"
            )
            report.pass_gate = False
            return report

        # Sort by entry time
        trades = sorted(trades, key=lambda t: t.entry_time)

        # Bootstrap CIs (always run)
        report.bootstrap_ci_wr = self._bootstrap_ci_win_rate(trades)
        report.bootstrap_ci_pf = self._bootstrap_ci_profit_factor(trades)
        report.regime_breakdown = self._regime_breakdown(trades)

        # 1. Walk-Forward Analysis
        if run_wf:
            logger.info("Running Walk-Forward Analysis ...")
            report.wf_result = self._walk_forward(trades)
            report.scores["wf_wr"] = report.wf_result.overall_win_rate
            report.scores["wf_pf"] = report.wf_result.overall_profit_factor
            report.scores["wf_sharpe"] = report.wf_result.overall_sharpe

        # 2. Deflated Sharpe Ratio
        if run_dsr:
            logger.info("Running Deflated Sharpe Ratio ...")
            all_pnls = [t.pnl for t in trades]
            report.dsr_result = self._deflated_sharpe(all_pnls)
            report.scores["dsr"] = report.dsr_result.dsr
            report.scores["observed_sharpe"] = report.dsr_result.observed_sharpe

        # 3. Monte Carlo Permutation Test
        if run_mc:
            logger.info("Running Monte Carlo Permutation Test (%d perms) ...", self.n_mc)
            report.mc_result = self._monte_carlo_permutation(trades)
            report.scores["mc_pvalue"] = report.mc_result.p_value
            report.scores["mc_percentile"] = report.mc_result.percentile_rank

        # 4. CPCV
        if run_cpcv:
            logger.info("Running Combinatorial Purged Cross-Validation ...")
            report.cpcv_result = self._cpcv(trades)
            report.scores["cpcv_score"] = report.cpcv_result.cpcv_score

        # Gate decision
        report.pass_gate, report.gate_failures = self._evaluate_gate(report)

        return report

    # ------------------------------------------------------------------
    # 1. Walk-Forward Analysis
    # ------------------------------------------------------------------

    def _walk_forward(self, trades: list[Trade]) -> WFAResult:
        """
        Portfolio-level walk-forward analysis with purged train/test splits.

        Uses time-based windows.  Bar counts are converted to seconds
        via bar_duration_secs (default 900s = 15min bars).
        """
        if not trades:
            return WFAResult(n_folds=0, total_test_trades=0,
                             overall_win_rate=0, overall_profit_factor=0,
                             overall_sharpe=0)

        t_min = trades[0].entry_time
        t_max = trades[-1].exit_time
        bar_secs = self.wfa.get("bar_duration_secs", 900)
        train_len = self.wfa["train_bars"] * bar_secs
        test_len = self.wfa["test_bars"] * bar_secs
        step = self.wfa["step_bars"] * bar_secs
        purge = self.wfa["purge_bars"] * bar_secs
        min_test = self.wfa["min_test_trades"]
        data_span = t_max - t_min

        # Auto-scale: if data is too short for configured windows,
        # shrink proportionally (min 15% of original)
        needed = train_len + purge + test_len
        if data_span < needed:
            scale = max(0.15, (data_span * 0.90) / needed)  # 90% to leave margin
            train_len = max(int(train_len * scale), 5 * bar_secs)  # min 5 bars
            test_len = max(int(test_len * scale), 3 * bar_secs)    # min 3 bars
            step = max(int(step * scale), test_len)
            purge = max(int(purge * scale), bar_secs)
            logger.info(
                "WFA auto-scale: data span %.1f days < needed %.1f days, "
                "scaling to %.0f%% (train→%d bars, test→%d bars)",
                data_span / 86400, needed / 86400,
                scale * 100, int(train_len / bar_secs), int(test_len / bar_secs),
            )

        folds: list[WalkForwardResult] = []
        fold_idx = 0
        cursor = t_min

        while cursor + train_len + purge + test_len <= t_max:
            train_start = cursor
            train_end = cursor + train_len
            test_start = train_end + purge
            test_end = test_start + test_len

            train_trades = [
                t for t in trades
                if train_start <= t.entry_time < train_end
            ]
            test_trades = [
                t for t in trades
                if test_start <= t.entry_time < test_end
            ]

            if len(test_trades) >= min_test:
                test_pnls = [t.pnl for t in test_trades]
                wins = sum(1 for p in test_pnls if p > 0)
                gross_profit = sum(p for p in test_pnls if p > 0)
                gross_loss = abs(sum(p for p in test_pnls if p < 0))
                pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

                # Annualised Sharpe (assumes ~365.25 * 24 * 4 = 35064 bars/year for 15min)
                pnl_arr = np.array(test_pnls)
                if len(pnl_arr) > 1 and pnl_arr.std() > 0:
                    sharpe = (pnl_arr.mean() / pnl_arr.std()) * math.sqrt(min(len(pnl_arr), 35064))
                else:
                    sharpe = 0.0

                folds.append(WalkForwardResult(
                    fold_idx=fold_idx,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_trades=len(train_trades),
                    test_trades=len(test_trades),
                    test_win_rate=wins / len(test_trades),
                    test_profit_factor=pf,
                    test_pnl=sum(test_pnls),
                    test_sharpe=sharpe,
                ))
                fold_idx += 1

            cursor += step

        if not folds:
            # Fallback: trade-count-based WFA for sparse/gapped data
            logger.info("Time-based WFA produced 0 folds, trying trade-count-based ...")
            folds = self._walk_forward_by_count(trades, min_test)

        if not folds:
            return WFAResult(n_folds=0, total_test_trades=0,
                             overall_win_rate=0, overall_profit_factor=0,
                             overall_sharpe=0)

        total_test = sum(f.test_trades for f in folds)
        all_test_wins = sum(f.test_win_rate * f.test_trades for f in folds)
        all_test_pnl = sum(f.test_pnl for f in folds)

        overall_wr = all_test_wins / total_test if total_test > 0 else 0
        overall_pf = self._aggregate_profit_factor(folds)
        overall_sharpe = float(np.mean([f.test_sharpe for f in folds]))

        # Bootstrap CIs on walk-forward test trades
        all_test_trades_list = []
        for f in folds:
            # Re-collect trades for each fold's test window
            pass  # We'll compute CIs from fold-level metrics
        ci_wr = self._bootstrap_ci_from_array(
            [f.test_win_rate for f in folds], self.n_bootstrap
        )
        ci_pf = self._bootstrap_ci_from_array(
            [min(f.test_profit_factor, 10.0) for f in folds], self.n_bootstrap
        )

        # Regime breakdown from test folds
        regime_data: dict[str, list[float]] = defaultdict(list)
        for f in folds:
            regime_data[f"fold_{f.fold_idx}_wr"].append(f.test_win_rate)

        result = WFAResult(
            n_folds=len(folds),
            total_test_trades=total_test,
            overall_win_rate=overall_wr,
            overall_profit_factor=overall_pf,
            overall_sharpe=overall_sharpe,
            folds=folds,
            bootstrap_ci_wr=ci_wr,
            bootstrap_ci_pf=ci_pf,
        )
        return result

    def _walk_forward_by_count(
        self, trades: list[Trade], min_test: int
    ) -> list[WalkForwardResult]:
        """
        Trade-count-based walk-forward fallback.

        Splits trades into rolling train/test windows by index.
        Uses 70/30 train/test ratio with 20% step.
        """
        n = len(trades)
        if n < min_test * 3:  # need at least 3x min_test for train+test
            return []

        train_frac = 0.70
        test_frac = 0.20
        step_frac = 0.20

        train_size = int(n * train_frac)
        test_size = max(int(n * test_frac), min_test)
        step_size = max(int(n * step_frac), min_test)

        folds = []
        fold_idx = 0
        cursor = 0

        while cursor + train_size + test_size <= n:
            train_trades = trades[cursor:cursor + train_size]
            test_trades = trades[cursor + train_size:cursor + train_size + test_size]

            if len(test_trades) >= min_test:
                test_pnls = [t.pnl for t in test_trades]
                wins = sum(1 for p in test_pnls if p > 0)
                gross_profit = sum(p for p in test_pnls if p > 0)
                gross_loss = abs(sum(p for p in test_pnls if p < 0))
                pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

                pnl_arr = np.array(test_pnls)
                if len(pnl_arr) > 1 and pnl_arr.std() > 0:
                    sharpe = (pnl_arr.mean() / pnl_arr.std()) * math.sqrt(min(len(pnl_arr), 252))
                else:
                    sharpe = 0.0

                folds.append(WalkForwardResult(
                    fold_idx=fold_idx,
                    train_start=train_trades[0].entry_time,
                    train_end=train_trades[-1].exit_time,
                    test_start=test_trades[0].entry_time,
                    test_end=test_trades[-1].exit_time,
                    train_trades=len(train_trades),
                    test_trades=len(test_trades),
                    test_win_rate=wins / len(test_trades),
                    test_profit_factor=pf,
                    test_pnl=sum(test_pnls),
                    test_sharpe=sharpe,
                ))
                fold_idx += 1

            cursor += step_size

        logger.info("Trade-count WFA: %d folds from %d trades", len(folds), n)
        return folds

    @staticmethod
    def _aggregate_profit_factor(folds: list[WalkForwardResult]) -> float:
        total_profit = 0.0
        total_loss = 0.0
        for f in folds:
            # Approximate: PF_i * trades_i gives gross profit / gross loss ratio
            # Better to recompute from raw P&L but we only have fold summaries.
            # Use weighted harmonic-like approach.
            if f.test_profit_factor == float("inf"):
                continue
            # We know pnl and win_rate and trades; infer gross profit and loss
            n = f.test_trades
            wr = f.test_win_rate
            pnl = f.test_pnl
            if n == 0:
                continue
            wins = int(round(wr * n))
            losses = n - wins
            if wins > 0 and losses > 0:
                avg_win = pnl / (wr * n - (1 - wr) * n * (pnl / (n * (2 * wr - 1)))) if wr != 0.5 else pnl / n
                # Simpler: use PF directly
                total_profit += f.test_profit_factor * abs(pnl) / (1 + f.test_profit_factor) if f.test_profit_factor > 0 else 0
                total_loss += abs(pnl) / (1 + f.test_profit_factor) if f.test_profit_factor > 0 else 0
            elif wins > 0:
                total_profit += pnl

        return total_profit / total_loss if total_loss > 0 else float("inf")

    # ------------------------------------------------------------------
    # 2. Deflated Sharpe Ratio — Bailey & Lopez de Prado (2014)
    # ------------------------------------------------------------------

    def _deflated_sharpe(self, pnls: list[float]) -> DSRResult:
        """
        Compute the Deflated Sharpe Ratio.

        DSR adjusts the observed Sharpe ratio for multiple-testing bias
        when selecting the best among N strategies.

        Formula:
            DSR = P(SR* < SR_observed)
            where SR* is the expected max Sharpe under the null (no skill)
            corrected by the number of trials.

        Reference: Bailey & Lopez de Prado (2014)
        "The Deflated Sharpe Ratio: Correcting for Selection Bias,
        Backtest Overfitting and Non-Normality"
        """
        arr = np.array(pnls, dtype=np.float64)
        n = len(arr)
        if n < 2:
            return DSRResult(observed_sharpe=0, sharpe_std_error=0,
                             n_trials=self.n_strategies, dsr=0,
                             dsr_pvalue=1.0, is_significant_95=False,
                             is_significant_99=False)

        # Observed annualised Sharpe (using bar-level returns)
        mean_r = arr.mean()
        std_r = arr.std(ddof=1)
        if std_r == 0:
            return DSRResult(observed_sharpe=0, sharpe_std_error=0,
                             n_trials=self.n_strategies, dsr=0,
                             dsr_pvalue=1.0, is_significant_95=False,
                             is_significant_99=False)

        sr_obs = mean_r / std_r
        # Annualise: assume trades are roughly independent, scale by sqrt(trades/year)
        # For 15-min bars: ~35064 bars/year; for trades, use sqrt(n) as conservative
        sr_annual = sr_obs * math.sqrt(min(n, 252))

        # Sharpe standard error (Mertens 2002 / Lo 2002)
        # SE(SR) = sqrt((1 + 0.5 * SR^2) / n)
        sr_se = math.sqrt((1 + 0.5 * sr_annual**2) / n)

        # Expected maximum Sharpe under the null for N independent trials
        # E[max(SR)] ≈ (1 - γ) * Φ^{-1}(1 - 1/N) + γ * Φ^{-1}(1 - 1/(N*e))
        # where γ ≈ 0.5772 (Euler-Mascheroni)
        N = self.n_strategies
        gamma_em = 0.5772156649

        # Using the simpler approximation from B&LdP:
        # E[max_SR] ≈ sqrt(2 * log(N)) for large N
        # More precise: use the Gumbel distribution approximation
        e_max_sr = (1 - gamma_em) * sp_stats.norm.ppf(1 - 1/N) + \
                   gamma_em * sp_stats.norm.ppf(1 - 1 / (N * math.e))
        e_max_sr = max(e_max_sr, 0)

        # Deflated Sharpe Ratio: probability that the true SR > 0
        # given the observed SR and the multiple-testing correction
        # DSR = Φ((SR_obs - E[max_SR]) / SE(SR))
        if sr_se > 0:
            dsr_z = (sr_annual - e_max_sr) / sr_se
            dsr_p = sp_stats.norm.cdf(dsr_z)
        else:
            dsr_z = 0
            dsr_p = 0.5

        return DSRResult(
            observed_sharpe=round(sr_annual, 4),
            sharpe_std_error=round(sr_se, 4),
            n_trials=N,
            dsr=round(dsr_z, 4),
            dsr_pvalue=round(dsr_p, 6),
            is_significant_95=dsr_p > 0.95,
            is_significant_99=dsr_p > 0.99,
        )

    # ------------------------------------------------------------------
    # 3. Monte Carlo Permutation Test
    # ------------------------------------------------------------------

    def _monte_carlo_permutation(self, trades: list[Trade]) -> MCResult:
        """
        Shuffle trade labels (P&L sign assignment) to build a null
        distribution.  Check whether the real win rate / profit factor
        exceeds the 95th percentile of the null.

        We use a sign-flip permutation: randomly flip the sign of each
        trade P&L, then compute the test statistic (profit factor).
        This tests whether the observed profitability is due to chance.
        """
        real_pnls = np.array([t.pnl for t in trades], dtype=np.float64)
        n = len(real_pnls)

        # Test statistic: profit factor
        def profit_factor(pnls: np.ndarray) -> float:
            gp = pnls[pnls > 0].sum()
            gl = abs(pnls[pnls < 0].sum())
            return gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)

        real_pf = profit_factor(real_pnls)

        # Also track win rate as secondary stat
        def win_rate(pnls: np.ndarray) -> float:
            return (pnls > 0).sum() / len(pnls) if len(pnls) > 0 else 0.0

        real_wr = win_rate(real_pnls)

        # Generate null distribution via sign-flip permutations
        null_pfs = np.empty(self.n_mc)
        signs = np.array([1.0] * n)

        for i in range(self.n_mc):
            # Random sign flip: each trade has 50% chance of sign reversal
            flips = self.rng.choice([-1.0, 1.0], size=n)
            perm_pnls = real_pnls * flips
            null_pfs[i] = profit_factor(perm_pnls)

        # p-value: fraction of null PFs >= real PF
        p_value = float((null_pfs >= real_pf).sum() / self.n_mc)

        # Percentile rank of real PF in null distribution
        percentile = float((null_pfs < real_pf).sum() / self.n_mc * 100)

        return MCResult(
            real_metric=round(real_pf, 4),
            n_permutations=self.n_mc,
            null_mean=round(float(null_pfs.mean()), 4),
            null_std=round(float(null_pfs.std()), 4),
            p_value=round(p_value, 6),
            percentile_rank=round(percentile, 2),
        )

    # ------------------------------------------------------------------
    # 4. Combinatorial Purged Cross-Validation (CPCV) — De Prado (2018)
    # ------------------------------------------------------------------

    def _cpcv(self, trades: list[Trade], n_groups: int = 10, n_test_groups: int = 2) -> CPCVResult:
        """
        Combinatorial Purged Cross-Validation.

        Split trades into n_groups, then evaluate all C(n_groups, n_test_groups)
        combinations of test groups.  For each combination, train on the
        remaining groups and test on the selected groups.  Purge overlapping
        trades between train and test sets.

        The CPCV score is the fraction of combinations that produce positive
        out-of-sample P&L.
        """
        n = len(trades)
        if n < n_groups:
            return CPCVResult(n_groups=n_groups, n_test_groups=n_test_groups,
                              n_combinations=0, mean_oos_pnl=0, oos_win_rate=0,
                              cpcv_score=0)

        # Assign trades to groups chronologically
        group_size = n // n_groups
        groups: list[list[Trade]] = []
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size if g < n_groups - 1 else n
            groups.append(trades[start:end])

        # Generate all C(n_groups, n_test_groups) combinations
        test_combos = list(combinations(range(n_groups), n_test_groups))
        n_combos = len(test_combos)

        oos_pnls = []
        oos_wins = 0

        for combo in test_combos:
            test_set_indices = set(combo)
            train_set_indices = set(range(n_groups)) - test_set_indices

            # Collect test trades
            test_trades = []
            for gi in test_set_indices:
                test_trades.extend(groups[gi])

            # Collect train trades (with purge: remove trades whose timestamps
            # overlap with test windows)
            train_trades = []
            test_start = min(t.entry_time for t in test_trades) if test_trades else 0
            test_end = max(t.exit_time for t in test_trades) if test_trades else 0
            bar_secs = self.wfa.get("bar_duration_secs", 900)
            purge_secs = self.wfa.get("purge_bars", 16) * bar_secs

            for gi in train_set_indices:
                for t in groups[gi]:
                    # Purge: skip train trades that overlap with test window
                    if t.exit_time >= test_start - purge_secs and t.entry_time <= test_end + purge_secs:
                        continue
                    train_trades.append(t)

            if not test_trades:
                continue

            # Simple "model": use train set mean P&L as signal direction
            # If train P&L is positive, expect test P&L to be positive
            train_mean_pnl = np.mean([t.pnl for t in train_trades]) if train_trades else 0
            test_pnl_sum = sum(t.pnl for t in test_trades)

            # CPCV uses whether the OOS result is consistent with IS result
            # If both are positive or both negative → consistent → count as "correct"
            if train_mean_pnl != 0:
                consistent = (train_mean_pnl > 0 and test_pnl_sum > 0) or \
                             (train_mean_pnl < 0 and test_pnl_sum < 0)
            else:
                consistent = test_pnl_sum > 0

            oos_pnls.append(test_pnl_sum)
            if consistent:
                oos_wins += 1

        cpcv_score = oos_wins / n_combos if n_combos > 0 else 0
        mean_oos = float(np.mean(oos_pnls)) if oos_pnls else 0
        oos_wr = sum(1 for p in oos_pnls if p > 0) / len(oos_pnls) if oos_pnls else 0

        return CPCVResult(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            n_combinations=n_combos,
            mean_oos_pnl=round(mean_oos, 4),
            oos_win_rate=round(oos_wr, 4),
            cpcv_score=round(cpcv_score, 4),
            pnl_distribution=[round(p, 4) for p in oos_pnls],
        )

    # ------------------------------------------------------------------
    # Bootstrap utilities
    # ------------------------------------------------------------------

    def _bootstrap_ci_win_rate(self, trades: list[Trade], alpha: float = 0.05) -> tuple[float, float]:
        """Bootstrap 95% CI on win rate."""
        pnls = np.array([t.pnl for t in trades])
        return self._bootstrap_ci_from_array(
            pnls, self.n_bootstrap, stat_fn=lambda x: (x > 0).mean(), alpha=alpha
        )

    def _bootstrap_ci_profit_factor(self, trades: list[Trade], alpha: float = 0.05) -> tuple[float, float]:
        """Bootstrap 95% CI on profit factor."""
        pnls = np.array([t.pnl for t in trades])

        def pf_stat(x):
            gp = x[x > 0].sum()
            gl = abs(x[x < 0].sum())
            return gp / gl if gl > 0 else 0.0

        return self._bootstrap_ci_from_array(pnls, self.n_bootstrap, stat_fn=pf_stat, alpha=alpha)

    def _bootstrap_ci_from_array(
        self,
        data,
        n_boot: int,
        stat_fn=None,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Generic bootstrap CI."""
        arr = np.asarray(data, dtype=np.float64)
        if len(arr) < 2:
            val = float(arr[0]) if len(arr) == 1 else 0.0
            return (val, val)

        if stat_fn is None:
            stat_fn = np.mean

        boot_stats = np.empty(n_boot)
        for i in range(n_boot):
            sample = self.rng.choice(arr, size=len(arr), replace=True)
            boot_stats[i] = stat_fn(sample)

        lo = float(np.percentile(boot_stats, 100 * alpha / 2))
        hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        return (round(lo, 6), round(hi, 6))

    # ------------------------------------------------------------------
    # Regime breakdown
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_breakdown(trades: list[Trade]) -> dict[str, dict]:
        """Compute per-regime statistics."""
        regimes: dict[str, list[float]] = defaultdict(list)
        for t in trades:
            regimes[t.regime].append(t.pnl)

        result = {}
        for regime, pnls in regimes.items():
            arr = np.array(pnls)
            wins = (arr > 0).sum()
            gp = arr[arr > 0].sum()
            gl = abs(arr[arr < 0].sum())
            result[regime] = {
                "n_trades": len(pnls),
                "win_rate": round(wins / len(pnls), 4) if pnls else 0,
                "profit_factor": round(gp / gl, 4) if gl > 0 else float("inf"),
                "total_pnl": round(float(arr.sum()), 4),
                "avg_pnl": round(float(arr.mean()), 4),
            }
        return result

    # ------------------------------------------------------------------
    # Gate evaluation
    # ------------------------------------------------------------------

    def _evaluate_gate(self, report: ValidationReport) -> tuple[bool, list[str]]:
        """Check all thresholds and return (pass, [failure reasons])."""
        th = self.thresholds
        failures = []

        # Sample size
        if report.n_trades < th.sample_size_min:
            failures.append(
                f"Sample size {report.n_trades} < min {th.sample_size_min}"
            )

        # Walk-forward WR
        if report.wf_result:
            if report.wf_result.overall_win_rate < th.wf_wr_min:
                failures.append(
                    f"WF WR {report.wf_result.overall_win_rate:.2%} < min {th.wf_wr_min:.0%}"
                )
            if report.wf_result.total_test_trades < th.sample_size_min:
                failures.append(
                    f"WF test trades {report.wf_result.total_test_trades} < min {th.sample_size_min}"
                )

        # DSR
        if report.dsr_result:
            if report.dsr_result.dsr < th.dsr_min:
                failures.append(
                    f"DSR {report.dsr_result.dsr:.2f} < min {th.dsr_min}"
                )

        # Monte Carlo p-value
        if report.mc_result:
            if report.mc_result.p_value > th.mc_pvalue_min:
                failures.append(
                    f"MC p-value {report.mc_result.p_value:.4f} > max {th.mc_pvalue_min}"
                )

        # CPCV score
        if report.cpcv_result:
            if report.cpcv_result.cpcv_score < th.cpcv_score_min:
                failures.append(
                    f"CPCV score {report.cpcv_result.cpcv_score:.4f} < min {th.cpcv_score_min}"
                )

        return (len(failures) == 0, failures)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trades_jsonl(
    path: str | Path,
    strategy_filter: Optional[str] = None,
    outcomes_path: Optional[str | Path] = None,
) -> list[Trade]:
    """
    Load trades from a JSONL signal file.

    Supports two schemas:

    1. Trade-level (has pnl, entry_time, exit_time):
       {"entry_time": ..., "exit_time": ..., "pnl": ..., "side": ..., ...}

    2. Signal-level (JIMI format):
       {"timestamp": "2026-06-27 12:15:00", "strategy": "trade_flow",
        "fired": true, "direction": "LONG", "entry": 1582.74,
        "sl": 1579.09, "tp1": 1587.61, "rr1": 1.33, "outcome": null}

    If outcomes_path is provided, loads resolved outcomes from the
    outcome tracker and uses real PnL instead of simulated values.
    """
    from datetime import datetime as dt

    # Load resolved outcomes if available
    resolved_outcomes: dict[int, dict] = {}
    if outcomes_path:
        op = Path(outcomes_path)
        if op.exists():
            with open(op) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        resolved_outcomes[rec["line_idx"]] = rec
                    except (json.JSONDecodeError, KeyError):
                        continue
            logger.info("Loaded %d resolved outcomes from %s", len(resolved_outcomes), op)

    trades = []
    path = Path(path)
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d", i)
                continue

            if strategy_filter and rec.get("strategy", "") != strategy_filter:
                continue

            # --- Schema 1: trade-level (has explicit pnl) ---
            if "pnl" in rec and "entry_time" in rec:
                try:
                    trades.append(Trade(
                        entry_time=float(rec.get("entry_time", 0)),
                        exit_time=float(rec.get("exit_time", rec.get("entry_time", 0))),
                        pnl=float(rec["pnl"]),
                        side=rec.get("side", "long"),
                        symbol=rec.get("symbol", ""),
                        regime=rec.get("regime", "unknown"),
                        strategy=rec.get("strategy", ""),
                    ))
                except (TypeError, ValueError) as e:
                    logger.warning("Skipping bad record at line %d: %s", i, e)
                continue

            # --- Schema 2: signal-level (JIMI format) ---
            if not rec.get("fired", False):
                continue  # Only process fired signals

            direction = (rec.get("direction") or "").upper()
            if direction not in ("LONG", "SHORT"):
                continue

            entry = rec.get("entry")
            sl = rec.get("sl")
            tp1 = rec.get("tp1")
            rr1 = rec.get("rr1")
            conviction = rec.get("conviction")

            if entry is None or sl is None:
                continue

            # Parse timestamp
            ts_str = rec.get("timestamp", "")
            try:
                ts_dt = dt.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                entry_time = ts_dt.timestamp()
            except (ValueError, TypeError):
                try:
                    entry_time = float(ts_str)
                except (ValueError, TypeError):
                    continue

            # Derive PnL: prefer resolved outcomes, fall back to simulation
            risk = abs(entry - sl)
            if risk <= 0:
                continue

            # Check for resolved outcome (from outcome tracker)
            resolved = resolved_outcomes.get(i)
            if resolved:
                pnl = float(resolved.get("pnl_r", 0))
                outcome_str = resolved.get("outcome", "")
                exit_time_str = resolved.get("exit_time", "")
                if exit_time_str:
                    try:
                        exit_dt = dt.strptime(exit_time_str, "%Y-%m-%d %H:%M:%S")
                        exit_time = exit_dt.timestamp()
                    except ValueError:
                        exit_time = entry_time + 4 * 900
                else:
                    exit_time = entry_time + 4 * 900
                regime = "unknown"  # will be set below
            else:
                # Simulate outcome: use conviction as win probability proxy
                hash_val = hash(ts_str + rec.get("strategy", "")) % 10000
                conv = conviction if conviction and conviction > 0 else 0.5
                win_prob = min(0.70, max(0.40, 0.35 + conv * 0.35))
                is_win = (hash_val / 10000.0) < win_prob

                if is_win:
                    reward_r = float(rr1) if rr1 and rr1 > 0 else 1.5
                    pnl = reward_r
                else:
                    pnl = -1.0

                dur_hash = hash(ts_str + "exit") % 10000
                exit_bars = 4 + int(dur_hash / 10000.0 * 12)
                exit_time = entry_time + exit_bars * 900

            # Regime detection from strategy name (heuristic)
            strategy = rec.get("strategy", "")
            regime = "unknown"
            regime_keywords = {
                "trending": ["momentum", "trend", "breakout"],
                "ranging": ["range", "fade", "positioning"],
                "volatile": ["vol", "cascade", "squeeze"],
                "quiet": ["scalp"],
            }
            for r_name, keywords in regime_keywords.items():
                if any(kw in strategy.lower() for kw in keywords):
                    regime = r_name
                    break

            trades.append(Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=pnl,
                side=direction.lower(),
                symbol="ETH/USDT",
                regime=regime,
                strategy=strategy,
            ))

    logger.info("Loaded %d trades from %s (filter=%s)", len(trades), path, strategy_filter)
    return trades


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------

def _serialize_dataclass(obj) -> Any:
    """Recursively serialize dataclasses, handling tuples and nested structures."""
    if hasattr(obj, "__dataclass_fields__"):
        d = {}
        for k, v in asdict(obj).items():
            d[k] = _serialize_dataclass(v)
        return d
    elif isinstance(obj, (list, tuple)):
        return [_serialize_dataclass(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _serialize_dataclass(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return "NaN"
        return obj
    return obj


def save_report(report: ValidationReport, output_dir: str | Path) -> Path:
    """Save validation report as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = report.strategy_name.replace("/", "_").replace(" ", "_")
    filename = f"validation_{name}_{ts}.json"
    filepath = output_dir / filename

    data = _serialize_dataclass(report)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Report saved to %s", filepath)
    return filepath


def print_summary(report: ValidationReport) -> None:
    """Print a human-readable summary of the validation report."""
    th = ValidationThresholds()

    def status_icon(value, min_val, target_val, higher_is_better=True):
        if higher_is_better:
            if value >= target_val:
                return "✅"
            elif value >= min_val:
                return "⚠️ "
            else:
                return "❌"
        else:
            if value <= target_val:
                return "✅"
            elif value <= min_val:
                return "⚠️ "
            else:
                return "❌"

    print("\n" + "=" * 70)
    print(f"  JIMI VALIDATION REPORT: {report.strategy_name}")
    print(f"  {report.timestamp}")
    print("=" * 70)

    print(f"\n  Trades: {report.n_trades}")
    print(f"  Bootstrap 95% CI — WR: [{report.bootstrap_ci_wr[0]:.2%}, {report.bootstrap_ci_wr[1]:.2%}]")
    print(f"  Bootstrap 95% CI — PF: [{report.bootstrap_ci_pf[0]:.2f}, {report.bootstrap_ci_pf[1]:.2f}]")

    if report.wf_result:
        wf = report.wf_result
        icon = status_icon(wf.overall_win_rate, th.wf_wr_min, th.wf_wr_target)
        print(f"\n  {icon} Walk-Forward Analysis")
        print(f"     Folds: {wf.n_folds}  |  Test trades: {wf.total_test_trades}")
        print(f"     WR: {wf.overall_win_rate:.2%}  (min {th.wf_wr_min:.0%}, target {th.wf_wr_target:.0%})")
        print(f"     PF: {wf.overall_profit_factor:.2f}")
        print(f"     Sharpe: {wf.overall_sharpe:.2f}")
        print(f"     CI-WR: [{wf.bootstrap_ci_wr[0]:.2%}, {wf.bootstrap_ci_wr[1]:.2%}]")
        print(f"     CI-PF: [{wf.bootstrap_ci_pf[0]:.2f}, {wf.bootstrap_ci_pf[1]:.2f}]")

    if report.dsr_result:
        dsr = report.dsr_result
        icon = status_icon(dsr.dsr, th.dsr_min, th.dsr_target)
        print(f"\n  {icon} Deflated Sharpe Ratio")
        print(f"     Observed SR: {dsr.observed_sharpe:.4f}")
        print(f"     DSR z-score: {dsr.dsr:.4f}  (min {th.dsr_min}, target {th.dsr_target})")
        print(f"     DSR p-value: {dsr.dsr_pvalue:.6f}")
        print(f"     Significant @95%: {'Yes' if dsr.is_significant_95 else 'No'}  |  @99%: {'Yes' if dsr.is_significant_99 else 'No'}")
        print(f"     Strategies tested (N): {dsr.n_trials}")

    if report.mc_result:
        mc = report.mc_result
        icon = status_icon(mc.p_value, th.mc_pvalue_min, th.mc_pvalue_target, higher_is_better=False)
        print(f"\n  {icon} Monte Carlo Permutation Test")
        print(f"     Real PF: {mc.real_metric:.4f}")
        print(f"     Null mean: {mc.null_mean:.4f} ± {mc.null_std:.4f}")
        print(f"     p-value: {mc.p_value:.6f}  (max {th.mc_pvalue_min}, target {th.mc_pvalue_target})")
        print(f"     Percentile rank: {mc.percentile_rank:.1f}%")

    if report.cpcv_result:
        cpcv = report.cpcv_result
        icon = status_icon(cpcv.cpcv_score, th.cpcv_score_min, th.cpcv_score_target)
        print(f"\n  {icon} Combinatorial Purged Cross-Validation")
        print(f"     Groups: {cpcv.n_groups}  |  Test groups: {cpcv.n_test_groups}")
        print(f"     Combinations: {cpcv.n_combinations}")
        print(f"     CPCV Score: {cpcv.cpcv_score:.4f}  (min {th.cpcv_score_min}, target {th.cpcv_score_target})")
        print(f"     OOS WR: {cpcv.oos_win_rate:.2%}")
        print(f"     Mean OOS PnL: {cpcv.mean_oos_pnl:.4f}")

    # Regime breakdown
    if report.regime_breakdown:
        print(f"\n  📊 Regime Breakdown")
        for regime, stats in report.regime_breakdown.items():
            print(f"     {regime}: n={stats['n_trades']}, WR={stats['win_rate']:.2%}, "
                  f"PF={stats['profit_factor']:.2f}, PnL={stats['total_pnl']:.4f}")

    # Gate decision
    print(f"\n  {'=' * 50}")
    if report.pass_gate:
        print("  ✅ GATE PASSED — Strategy validated for deployment")
    else:
        print("  ❌ GATE FAILED")
        for f in report.gate_failures:
            print(f"     • {f}")
    print("=" * 70 + "\n")
