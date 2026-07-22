#!/usr/bin/env python3
"""
RegimeClassifierV3 — Research-backed regime detection with persistence.

Based on:
- Shu et al. (2024) "Statistical Jump Model" — regime persistence via jump penalty
- Gupta et al. (2025) "Ensemble-HMM Voting" — multi-model voting for robustness
- Nystrup et al. (2020) — HMM sensitivity to mis-estimation

Key improvements over V2:
1. Jump penalty (λ) — controls regime persistence, tunable via backtest
2. Hysteresis — require sustained evidence across N scans before switching
3. Regime duration + stability tracking
4. Microstructure features — VWAP deviation, spread, vol clustering
5. Ensemble-style voting across 4 signal categories with confidence weighting
6. Regime transition cooldown — minimum bars between switches

Drop-in replacement for RegimeClassifierV2 in agents_v2.py
"""

import json, os, sys, time, math
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RegimeClassifierV3:
    """
    Research-backed regime classifier with jump penalty persistence.

    Signal Categories (Ensemble Voting):
    ┌─────────────────────────────────────────────────┐
    │ Category 1: Derivatives (FR, OI, LS) — SLOW     │
    │ Category 2: Price Momentum (1h/4h) — FAST       │
    │ Category 3: Volatility Structure (ATR, squeeze)  │
    │ Category 4: Microstructure (taker, VWAP, spread) │
    └─────────────────────────────────────────────────┘

    Each category votes BULL/BEAR/STRESS/RANGING with confidence.
    Final regime = weighted vote + jump penalty enforcement.
    """

    # === JUMP PENALTY (from Princeton research) ===
    # Higher λ = fewer transitions = more persistent regimes
    # λ=0: no persistence (flip every scan)
    # λ=5: moderate persistence (need strong evidence to switch)
    # λ=10: high persistence (regime changes are rare)
    JUMP_PENALTY = 3.0  # Default; should be tuned via backtest

    # === HYSTERESIS ===
    # Require N consecutive scans with same new regime before switching
    HYSTERESIS_WINDOW = 3  # Need 3 consecutive votes for new regime

    # === TRANSITION COOLDOWN ===
    # Minimum scans between regime transitions (prevents rapid oscillation)
    MIN_TRANSITION_INTERVAL = 5  # 5 scans * ~60s = ~5 min cooldown

    # === MOMENTUM THRESHOLDS ===
    MOM_OVERRIDE_1H = 0.015   # 1.5% in 1h = hard override
    MOM_OVERRIDE_4H = 0.030   # 3.0% in 4h = force STRESS
    MOM_SOFT_1H = 0.008       # 0.8% = soft boost
    MOM_SOFT_4H = 0.015       # 1.5% = soft boost

    # === MICROSTRUCTURE THRESHOLDS ===
    VWAP_DEVIATION_THRESHOLD = 0.005  # 0.5% from VWAP = notable
    VOL_CLUSTER_HIGH = 1.5            # vol_ratio > 1.5 = high vol
    VOL_CLUSTER_LOW = 0.7             # vol_ratio < 0.7 = low vol

    # === SIGNAL CATEGORY WEIGHTS ===
    CATEGORY_WEIGHTS = {
        "derivatives": 1.0,    # FR, OI, LS — slow but reliable
        "momentum": 1.5,       # Price momentum — fast, highest weight
        "volatility": 0.8,     # Vol structure — moderate
        "microstructure": 0.7, # Taker flow, VWAP — supplementary
    }

    def __init__(self, confluence_checker=None):
        self.cc = confluence_checker
        self.regime = "RANGING"
        self.confidence = 0.5
        self.signals = {}

        # === PERSISTENCE STATE ===
        self._regime_history = deque(maxlen=100)  # [(timestamp, regime, confidence)]
        self._vote_history = deque(maxlen=20)     # [regime] — for hysteresis
        self._last_transition_ts = 0
        self._regime_start_ts = 0
        self._regime_duration = 0  # scans in current regime
        self._regime_stability = 0.0  # % of recent votes matching current

        # === CATEGORY SCORES (for ensemble) ===
        self._category_votes = {}

    def classify(self, scan_data=None, price_history=None, timestamp=None):
        """
        Classify regime with ensemble voting + jump penalty persistence.

        Args:
            scan_data: dict from latest scan
            price_history: list of recent close prices (15m bars)
            timestamp: current timestamp (epoch seconds)

        Returns:
            (regime, confidence, signals_dict)
        """
        now = timestamp or time.time()
        cat_scores = {
            "derivatives": {"bull": 0, "bear": 0, "stress": 0, "ranging": 0},
            "momentum":    {"bull": 0, "bear": 0, "stress": 0, "ranging": 0},
            "volatility":  {"bull": 0, "bear": 0, "stress": 0, "ranging": 0},
            "microstructure": {"bull": 0, "bear": 0, "stress": 0, "ranging": 0},
        }
        signals = {}

        # ═══════════════════════════════════════════════════
        # CATEGORY 1: DERIVATIVES (SLOW — FR, OI, LS)
        # ═══════════════════════════════════════════════════
        deriv = self.cc.deriv_by_ts if self.cc else None
        if deriv:
            sorted_ts = sorted(deriv.keys())
            recent = [deriv[ts] for ts in sorted_ts[-20:]]
            if len(recent) >= 3:
                latest = recent[-1]
                fr = latest.get("fr", 0)
                ls = latest.get("ls", 2.0)
                oi = latest.get("oi", 0)

                prev_oi = recent[-2].get("oi", oi) if len(recent) >= 2 else oi
                oi_roc = ((oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
                avg_fr = sum(d.get("fr", 0) for d in recent) / len(recent)
                ls_vals = [d.get("ls", 2.0) for d in recent]
                ls_trend = ls_vals[-1] - ls_vals[0] if len(ls_vals) > 1 else 0
                fr_vals = [d.get("fr", 0) for d in recent]
                fr_std = (sum((f - avg_fr)**2 for f in fr_vals) / max(len(fr_vals)-1, 1)) ** 0.5

                d = cat_scores["derivatives"]
                # Funding rate
                if fr > 0.000030:
                    d["bull"] += 1.0; signals["fr"] = "BULL"
                elif fr < -0.000010:
                    d["bear"] += 1.0; signals["fr"] = "BEAR"
                else:
                    signals["fr"] = "NEUTRAL"

                # LS ratio
                if ls > 2.2:
                    d["bear"] += 0.8; signals["ls"] = "LONG_CROWDED"
                elif ls < 1.8:
                    d["bull"] += 0.8; signals["ls"] = "SHORT_CROWDED"
                else:
                    signals["ls"] = "NEUTRAL"

                # OI ROC
                if oi_roc < -3:
                    d["stress"] += 2.0; signals["oi"] = "STRESS"
                elif oi_roc > 5:
                    d["bull"] += 0.5; signals["oi"] = "SURGE"
                else:
                    signals["oi"] = "STABLE"

                # FR trend
                if avg_fr > 0.000015:
                    d["bull"] += 0.5; signals["fr_trend"] = "BULL"
                elif avg_fr < -0.000005:
                    d["bear"] += 0.5; signals["fr_trend"] = "BEAR"
                else:
                    signals["fr_trend"] = "NEUTRAL"

                # LS trend
                if ls_trend > 0.1:
                    d["bull"] += 0.2; signals["ls_trend"] = "MORE_LONGS"
                elif ls_trend < -0.1:
                    d["bear"] += 0.2; signals["ls_trend"] = "MORE_SHORTS"
                else:
                    signals["ls_trend"] = "STABLE"

                # FR volatility (stress)
                if fr_std > 0.00003:
                    d["stress"] += 1.0; signals["fr_vol"] = "HIGH"
                else:
                    signals["fr_vol"] = "NORMAL"

        # ═══════════════════════════════════════════════════
        # CATEGORY 2: PRICE MOMENTUM (FAST — 1h/4h ROC)
        # ═══════════════════════════════════════════════════
        m = cat_scores["momentum"]
        momentum_override = False
        if price_history and len(price_history) >= 5:
            cp = price_history[-1]
            mom_1h = (cp - price_history[-5]) / price_history[-5] if price_history[-5] else 0
            mom_4h = (cp - price_history[-17]) / price_history[-17] if len(price_history) >= 17 and price_history[-17] else 0

            signals["mom_1h"] = f"{mom_1h*100:+.2f}%"
            signals["mom_4h"] = f"{mom_4h*100:+.2f}%"

            # Hard overrides (instant regime change)
            if mom_1h < -self.MOM_OVERRIDE_1H:
                m["bear"] += 3.0; m["stress"] += 1.0
                momentum_override = True
                signals["mom_override"] = f"1H_CRASH {mom_1h*100:.1f}%"
            elif mom_1h > self.MOM_OVERRIDE_1H:
                m["bull"] += 2.0
                momentum_override = True
                signals["mom_override"] = f"1H_PUMP {mom_1h*100:.1f}%"

            if mom_4h < -self.MOM_OVERRIDE_4H:
                m["stress"] += 3.0; m["bear"] += 2.0
                momentum_override = True
                signals["mom_override"] = f"4H_CRASH {mom_4h*100:.1f}%"

            # Soft boosts
            if abs(mom_1h) > self.MOM_SOFT_1H:
                m["bull" if mom_1h > 0 else "bear"] += 1.0
            if abs(mom_4h) > self.MOM_SOFT_4H:
                m["bull" if mom_4h > 0 else "bear"] += 0.8

        # ═══════════════════════════════════════════════════
        # CATEGORY 3: VOLATILITY STRUCTURE
        # ═══════════════════════════════════════════════════
        v = cat_scores["volatility"]
        if scan_data:
            vol = scan_data.get("vol_regime", {})
            if vol:
                vol_ratio = vol.get("vol_ratio", 1.0) or 1.0
                vol_trend = vol.get("vol_trend", 0)
                squeeze = scan_data.get("squeeze", {})
                squeeze_status = squeeze.get("squeeze_status", "")

                if vol_ratio > self.VOL_CLUSTER_HIGH:
                    v["stress"] += 0.5; signals["vol"] = "HIGH_VOL"
                elif vol_ratio < self.VOL_CLUSTER_LOW:
                    v["ranging"] += 0.3; signals["vol"] = "LOW_VOL"
                else:
                    signals["vol"] = "NORMAL"

                # Vol trend (rising vol = stress building)
                if vol_trend and vol_trend > 0.1:
                    v["stress"] += 0.3; signals["vol_trend"] = "RISING"
                elif vol_trend and vol_trend < -0.1:
                    signals["vol_trend"] = "FALLING"
                else:
                    signals["vol_trend"] = "STABLE"

                # Squeeze release = momentum
                if squeeze_status == "TRIGGERED":
                    v["bull"] += 0.5; signals["squeeze"] = "TRIGGERED"
                else:
                    signals["squeeze"] = squeeze_status or "NONE"

            # Volatility clustering (GARCH-like: high vol persists)
            # Use FR variance as proxy if vol_regime not available
            if "fr_vol" in signals and signals["fr_vol"] == "HIGH":
                v["stress"] += 0.3

        # ═══════════════════════════════════════════════════
        # CATEGORY 4: MICROSTRUCTURE (taker, VWAP, spread)
        # ═══════════════════════════════════════════════════
        u = cat_scores["microstructure"]
        if scan_data:
            # Taker flow
            taker = scan_data.get("taker", {}) or scan_data.get("taker_summary", {})
            if taker:
                tr = taker.get("regime", "")
                if "BUYING" in tr.upper() or "SURGE" in tr.upper():
                    u["bull"] += 0.5; signals["taker"] = "BUY_SURGE"
                elif "SELLING" in tr.upper():
                    u["bear"] += 0.5; signals["taker"] = "SELL_SURGE"
                else:
                    signals["taker"] = "NEUTRAL"

            # VWAP deviation
            vwap = scan_data.get("vwap")
            price = scan_data.get("price")
            if vwap and price and vwap > 0:
                vwap_dev = (price - vwap) / vwap
                signals["vwap_dev"] = f"{vwap_dev*100:+.2f}%"
                if vwap_dev > self.VWAP_DEVIATION_THRESHOLD:
                    u["bull"] += 0.3  # Price above VWAP = bullish
                elif vwap_dev < -self.VWAP_DEVIATION_THRESHOLD:
                    u["bear"] += 0.3  # Price below VWAP = bearish

            # Cascade signal
            cascade = scan_data.get("cascade", {})
            if cascade:
                cs = cascade.get("combined_signal", "HOLD")
                if cs in ("STRONG_LONG", "STRONG_SHORT"):
                    u["bull" if "LONG" in cs else "bear"] += 0.5
                    signals["cascade"] = cs
                else:
                    signals["cascade"] = "HOLD"

            # Macro calendar
            macro = scan_data.get("macro_calendar", {})
            if macro:
                mp = macro.get("phase", "")
                if "CPI" in mp or "FOMC" in mp:
                    u["stress"] += 0.3; signals["macro"] = "HIGH_IMPACT"
                elif "NFP" in mp:
                    u["stress"] += 0.2; signals["macro"] = "NFP_WEEK"
                else:
                    signals["macro"] = "NORMAL"

            # Direction resolver consensus
            dr = scan_data.get("direction_resolver", {})
            if dr:
                drd = dr.get("direction", "NEUTRAL")
                if drd == "LONG":
                    u["bull"] += 0.3; signals["consensus"] = "LONG"
                elif drd == "SHORT":
                    u["bear"] += 0.3; signals["consensus"] = "SHORT"
                else:
                    signals["consensus"] = "NEUTRAL"

        # ═══════════════════════════════════════════════════
        # ENSEMBLE VOTE: Weighted aggregation across categories
        # ═══════════════════════════════════════════════════
        total_bull = 0.0
        total_bear = 0.0
        total_stress = 0.0
        total_ranging = 0.0

        for cat_name, scores in cat_scores.items():
            w = self.CATEGORY_WEIGHTS.get(cat_name, 1.0)
            total_bull += scores["bull"] * w
            total_bear += scores["bear"] * w
            total_stress += scores["stress"] * w
            total_ranging += scores["ranging"] * w

        # Normalize
        total = total_bull + total_bear + total_stress + total_ranging
        if total > 0:
            bull_pct = total_bull / total
            bear_pct = total_bear / total
            stress_pct = total_stress / total
        else:
            bull_pct = bear_pct = stress_pct = 0.25

        signals["ensemble"] = {
            "bull": round(total_bull, 2),
            "bear": round(total_bear, 2),
            "stress": round(total_stress, 2),
            "categories": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in cat_scores.items()},
        }

        # ═══════════════════════════════════════════════════
        # REGIME DECISION (with jump penalty)
        # ═══════════════════════════════════════════════════
        if total_stress > 2:
            raw_regime = "STRESS"
            raw_confidence = min(0.95, 0.5 + stress_pct * 0.5)
        elif total_bull > total_bear + 0.5:
            raw_regime = "BULL"
            raw_confidence = min(0.95, 0.5 + bull_pct * 0.5)
        elif total_bear > total_bull + 0.5:
            raw_regime = "BEAR"
            raw_confidence = min(0.95, 0.5 + bear_pct * 0.5)
        elif total_bear > total_bull and total_bear >= 1.0:
            raw_regime = "MILDLY_BEARISH"
            raw_confidence = min(0.8, 0.5 + bear_pct * 0.3)
        else:
            raw_regime = "RANGING"
            raw_confidence = 0.5

        # ═══════════════════════════════════════════════════
        # JUMP PENALTY + HYSTERESIS ENFORCEMENT
        # ═══════════════════════════════════════════════════
        regime = self._apply_persistence(raw_regime, raw_confidence, momentum_override, now)

        # ═══════════════════════════════════════════════════
        # UPDATE STATE
        # ═══════════════════════════════════════════════════
        self.regime = regime
        self.confidence = raw_confidence
        self.signals = signals
        self._category_votes = cat_scores

        # Track history
        self._regime_history.append((now, regime, raw_confidence))
        self._vote_history.append(raw_regime)
        if len(self._vote_history) > self.HYSTERESIS_WINDOW * 2:
            self._vote_history = deque(list(self._vote_history)[-self.HYSTERESIS_WINDOW*2:], maxlen=self.HYSTERESIS_WINDOW*2)

        # Duration tracking
        if regime == self._get_previous_regime():
            self._regime_duration += 1
        else:
            self._regime_duration = 1
            self._regime_start_ts = now

        # Stability: % of recent votes matching current regime
        recent_votes = list(self._vote_history)[-10:]
        if recent_votes:
            self._regime_stability = sum(1 for v in recent_votes if v == regime) / len(recent_votes)
        else:
            self._regime_stability = 0.5

        return regime, raw_confidence, signals

    def _apply_persistence(self, raw_regime, raw_confidence, momentum_override, now):
        """
        Apply jump penalty + hysteresis to prevent regime flipping.

        Rules:
        1. Momentum override → instant switch (bypass persistence)
        2. Same as current regime → keep
        3. Different regime → must satisfy ALL:
           a. Hysteresis: N consecutive votes for new regime
           b. Cooldown: enough time since last transition
           c. Jump penalty: confidence must overcome penalty threshold
        """
        current = self._get_previous_regime()

        # Rule 1: Momentum override bypasses everything
        if momentum_override:
            self._last_transition_ts = now
            return raw_regime

        # Rule 2: Same regime → keep
        if raw_regime == current:
            return raw_regime

        # Rule 3: Different regime → apply persistence gates

        # 3a. Hysteresis check
        recent = list(self._vote_history)[-(self.HYSTERESIS_WINDOW):]
        hysteresis_met = len(recent) >= self.HYSTERESIS_WINDOW and all(
            r == raw_regime for r in recent
        )

        # 3b. Cooldown check
        scans_since_transition = len([t for t, _, _ in self._regime_history
                                       if t > self._last_transition_ts])
        cooldown_met = scans_since_transition >= self.MIN_TRANSITION_INTERVAL

        # 3c. Jump penalty check
        # Confidence must exceed: base_threshold + jump_penalty * persistence_factor
        # Higher jump_penalty → higher bar to switch
        duration_factor = min(self._regime_duration / 10.0, 1.0)  # 0→1 over 10 scans
        penalty_threshold = 0.5 + self.JUMP_PENALTY * 0.05 * duration_factor
        penalty_met = raw_confidence >= penalty_threshold

        # All three must be met (unless current regime is RANGING — easier to leave)
        if current == "RANGING":
            # Easier to leave RANGING — only need hysteresis OR high confidence
            if hysteresis_met or raw_confidence > 0.7:
                self._last_transition_ts = now
                return raw_regime
        else:
            # Non-RANGING: need all three gates
            if hysteresis_met and cooldown_met and penalty_met:
                self._last_transition_ts = now
                return raw_regime

        # Persistence holds — keep current regime
        return current

    def _get_previous_regime(self):
        """Get the most recent regime from history."""
        if self._regime_history:
            return self._regime_history[-1][1]
        return "RANGING"

    def get_regime_info(self):
        """Return detailed regime state for logging/monitoring."""
        return {
            "regime": self.regime,
            "confidence": round(self.confidence, 3),
            "duration": self._regime_duration,
            "stability": round(self._regime_stability, 3),
            "jump_penalty": self.JUMP_PENALTY,
            "hysteresis_window": self.HYSTERESIS_WINDOW,
            "scans_since_transition": len([t for t, _, _ in self._regime_history
                                           if t > self._last_transition_ts]),
            "category_votes": self._category_votes,
            "last_transition": self._last_transition_ts,
        }

    # === Convenience methods (compatible with V2) ===
    def is_bullish(self):
        return self.regime == "BULL" and self.confidence >= 0.5

    def is_neutral_or_bearish(self):
        return self.regime in ("BEAR", "STRESS", "MILDLY_BEARISH")

    def is_ranging(self):
        return self.regime == "RANGING"

    def is_bearish(self):
        return self.regime in ("BEAR", "STRESS", "MILDLY_BEARISH") and self.confidence >= 0.5


# ═══════════════════════════════════════════════════════════════
# INTEGRATION: Drop-in replacement in agents_v2.py
# ═══════════════════════════════════════════════════════════════
# To use V3, replace in OrchestratorV2.__init__():
#   self.regime = RegimeClassifierV2(confluence_checker)
# with:
#   from regime_classifier_v3 import RegimeClassifierV3
#   self.regime = RegimeClassifierV3(confluence_checker)
#
# The classify() signature is backward-compatible with V2.
# ═══════════════════════════════════════════════════════════════
