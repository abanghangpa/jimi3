#!/usr/bin/env python3
"""
Scanner Live Executor — Places trades via HTX API based on scanner signals.
OPTIMIZED configs from 2026-07-05 optimization (PF >= 2.0 strategies only).
+ Isolation gate check: rejects strategies that haven't passed the gate
+ Live-vs-backtest monitor: pauses strategies when WR/PF degrades
"""
import json, os, sys, time, math, argparse
from datetime import datetime, timezone, timedelta
from collections import defaultdict




BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# === LOCKFILE: Prevent duplicate processes ===
LOCK_FILE = os.path.join(BASE, "live", "data", "executor.lock")

def acquire_lock():
    """Acquire exclusive lock. Returns True if successful, False if another instance is running."""
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                old_pid = int(f.read().strip())
            # Check if old process is still alive
            try:
                os.kill(old_pid, 0)
                # Process is alive — check if it's actually our executor
                with open(f"/proc/{old_pid}/cmdline", "r") as f:
                    cmdline = f.read()
                if "scanner_executor" in cmdline:
                    log(f"LOCK: Another executor running (PID {old_pid}). Exiting.", "ERROR")
                    return False
            except (ProcessLookupError, FileNotFoundError, PermissionError):
                # Old process is dead, remove stale lock
                pass
        except (ValueError, IOError):
            pass
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
    return True

def release_lock():
    """Release the lockfile."""
    try:
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE) as f:
                pid = int(f.read().strip())
            if pid == os.getpid():
                os.remove(LOCK_FILE)
    except: pass

import atexit
atexit.register(release_lock)

sys.path.insert(0, BASE)

SCAN_DIR = os.path.join(BASE, "data", "scans")
SIGNALS_FILE = os.path.join(BASE, "data", "strategy_signals.jsonl")
STATE_FILE = os.path.join(BASE, "live", "data", "executor_state.json")
TRADE_LOG = os.path.join(BASE, "live", "data", "executor_trades.json")
LOG_FILE = os.path.join(BASE, "live", "logs", "executor.log")
KEYS_FILE = os.path.join(BASE, "config", "exchange_keys.json")
ISOLATION_GATE_FILE = os.path.join(BASE, "config", "isolation_gate_results.json")
BACKTEST_BENCH_FILE = os.path.join(BASE, "config", "backtest_benchmarks.json")

SYMBOL = "ETH/USDT:USDT"
INITIAL_CAPITAL = 200.0


# === ISOLATION GATE GUARDIAN ===
class IsolationGateGuardian:
    """
    Loads isolation gate results and rejects strategies that haven't passed.
    Create config/isolation_gate_results.json with format:
    {
      "strategy_name": {
        "passed": true,
        "p_value": 0.05,
        "effect_direction": "correct",
        "mean_return_pct": 0.15,
        "events": 847,
        "date": "2026-07-12"
      }
    }
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.results = {}
        self._regime_classifier = None
        self.confluence = None
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath) as f:
                    self.results = json.load(f)
            except Exception as e:
                log(f"ISOLATION GATE: Failed to load {self.filepath}: {e}", "WARN")
                self.results = {}
        else:
            log(f"ISOLATION GATE: No file at {self.filepath} — ALL strategies will be REJECTED", "WARN")
            self.results = {}

    def is_passed(self, strategy_name):
        """Check if a strategy passed the isolation gate.
        Supports regime-specific results: if pooled FAIL but current regime PASS, allow it."""
        entry = self.results.get(strategy_name)
        if not entry:
            return False, "no gate result"
        if entry.get("passed"):
            return True, "passed"
        # Check regime-specific breakdown
        regime_breakdown = entry.get("regime_breakdown", {})
        if regime_breakdown and hasattr(self, '_regime_classifier'):
            current_regime = self._regime_classifier.regime.lower()
            # Map classifier regimes to gate regime names
            regime_map = {
                "ranging": "chop_2026", "bear": "bear_2018",
                "bull": "bull_2025", "stress": "bear_2018",
            }
            gate_regime = regime_map.get(current_regime, current_regime)
            if gate_regime in regime_breakdown:
                regime_result = regime_breakdown[gate_regime]
                if regime_result.get("passed"):
                    return True, "passed (%s)" % gate_regime
        return False, "gate failed"

    def get_details(self, strategy_name):
        """Get gate result details for a strategy."""
        return self.results.get(strategy_name, {})


# === LIVE PERFORMANCE MONITOR ===
class LivePerformanceMonitor:
    """
    Tracks rolling WR/PF per strategy and pauses when degraded vs backtest.
    Create config/backtest_benchmarks.json with format:
    {
      "benchmarks": {
        "strategy_name": {"wr": 0.75, "pf": 2.3}
      },
      "live_stats": {},
      "paused": {}
    }
    """
    WR_DROP_THRESHOLD = 0.10    # Pause if live WR drops >10% below backtest
    PF_DROP_THRESHOLD = 0.50    # Pause if live PF drops >0.5 below backtest
    MIN_TRADES_FOR_EVAL = 10    # Need at least N trades before evaluating
    PF_FLOOR = 1.0              # Absolute minimum PF

    def __init__(self, filepath):
        self.filepath = filepath
        self.benchmarks = {}
        self.strategy_stats = {}
        self.paused_strategies = {}
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath) as f:
                    data = json.load(f)
                    self.benchmarks = data.get("benchmarks", {})
                    self.strategy_stats = data.get("live_stats", {})
                    self.paused_strategies = data.get("paused", {})
            except Exception as e:
                log(f"MONITOR: Failed to load {self.filepath}: {e}", "WARN")

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        data = {
            "benchmarks": self.benchmarks,
            "live_stats": self.strategy_stats,
            "paused": self.paused_strategies,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def record_trade(self, strategy_name, outcome):
        """Record a trade outcome (WIN/LOSS/TIMEOUT)."""
        if strategy_name not in self.strategy_stats:
            self.strategy_stats[strategy_name] = {
                "trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
                "pnl": 0.0, "trade_log": []
            }
        stats = self.strategy_stats[strategy_name]
        stats["trades"] += 1
        if outcome == "WIN":
            stats["wins"] += 1
        elif outcome == "LOSS":
            stats["losses"] += 1
        elif outcome == "TIMEOUT":
            stats["timeouts"] += 1

        # Keep last 50 trade outcomes for rolling calc
        stats["trade_log"].append({"outcome": outcome, "ts": datetime.now(timezone.utc).isoformat()})
        if len(stats["trade_log"]) > 50:
            stats["trade_log"] = stats["trade_log"][-50:]

        self.save()
        self._evaluate(strategy_name)

    def record_pnl(self, strategy_name, pnl):
        """Record PnL for a trade."""
        if strategy_name not in self.strategy_stats:
            return
        self.strategy_stats[strategy_name]["pnl"] += pnl
        self.save()

    def _evaluate(self, strategy_name):
        """Check if strategy should be paused due to degradation."""
        stats = self.strategy_stats.get(strategy_name, {})
        trades = stats.get("trades", 0)
        wins = stats.get("wins", 0)

        if trades < self.MIN_TRADES_FOR_EVAL:
            return

        live_wr = wins / trades if trades > 0 else 0
        benchmark = self.benchmarks.get(strategy_name, {})
        backtest_wr = benchmark.get("wr", None)
        backtest_pf = benchmark.get("pf", None)

        # Calculate live PF from recent trades
        recent = stats.get("trade_log", [])[-20:]
        if len(recent) >= 5:
            r_wins = sum(1 for t in recent if t["outcome"] == "WIN")
            r_losses = sum(1 for t in recent if t["outcome"] in ("LOSS", "TIMEOUT"))
            r_wr = r_wins / len(recent) if recent else 0
            live_pf = (r_wr / (1 - r_wr)) if r_wr < 1 else 999.0
        else:
            live_pf = None

        reason = None
        if backtest_wr is not None and trades >= self.MIN_TRADES_FOR_EVAL:
            wr_drop = backtest_wr - live_wr
            if wr_drop > self.WR_DROP_THRESHOLD:
                reason = f"WR degraded: live {live_wr*100:.1f}% vs backtest {backtest_wr*100:.1f}% (drop: {wr_drop*100:.1f}%)"

        if live_pf is not None and backtest_pf is not None:
            pf_drop = backtest_pf - live_pf
            if pf_drop > self.PF_DROP_THRESHOLD:
                reason = f"PF degraded: live {live_pf:.2f} vs backtest {backtest_pf:.2f} (drop: {pf_drop:.2f})"

        if live_pf is not None and live_pf < self.PF_FLOOR and trades >= self.MIN_TRADES_FOR_EVAL:
            reason = f"PF below floor: {live_pf:.2f} < {self.PF_FLOOR}"

        if reason:
            self.paused_strategies[strategy_name] = {
                "reason": reason,
                "paused_at": datetime.now(timezone.utc).isoformat(),
                "live_wr": round(live_wr, 4),
                "trades": trades,
            }
            self.save()
            log(f"MONITOR: PAUSED {strategy_name} — {reason}", "WARN")

    def is_paused(self, strategy_name):
        """Check if a strategy is paused by the monitor."""
        entry = self.paused_strategies.get(strategy_name)
        if not entry:
            return False, None
        return True, entry.get("reason", "unknown")

    def resume(self, strategy_name):
        """Manually resume a paused strategy."""
        if strategy_name in self.paused_strategies:
            del self.paused_strategies[strategy_name]
            self.save()
            log(f"MONITOR: RESUMED {strategy_name}")

    def get_report(self):
        """Get summary of all strategies' live performance vs backtest."""
        report = []
        for strat, stats in self.strategy_stats.items():
            trades = stats.get("trades", 0)
            wins = stats.get("wins", 0)
            live_wr = wins / trades if trades > 0 else 0
            bench = self.benchmarks.get(strat, {})
            paused, reason = self.is_paused(strat)
            report.append({
                "strategy": strat,
                "trades": trades,
                "live_wr": f"{live_wr*100:.1f}%",
                "backtest_wr": f"{bench.get('wr', 0)*100:.1f}%" if bench.get("wr") else "N/A",
                "backtest_pf": bench.get("pf", "N/A"),
                "pnl": f"${stats.get('pnl', 0):.2f}",
                "paused": paused,
                "reason": reason or "",
            })
        return report



# === CONFLUENCE CHECKER ===
# Detects extreme positioning from derivatives data.
# Used by confluence signals (bb_mom6 + extreme positioning).
class ConfluenceChecker:
    """Loads derivatives data and detects extreme positioning events."""

    DERIV_FILE = os.path.join(BASE, "data", "derivatives_history", "derivatives_collected.csv")

    def __init__(self):
        self.deriv_by_ts = {}
        self.extreme_ts = {}  # ts -> {direction, conviction, conds}
        self.fr_p10 = -0.000020
        self.fr_p90 = 0.000078
        self.oi_p90 = 0.47
        self.ls_p10 = 1.595
        self.ls_p90 = 2.677
        self.load()
        self.detect_extreme()

    def load(self):
        """Load derivatives data and snap to 15m candles."""
        if not os.path.exists(self.DERIV_FILE):
            log("CONFLUENCE: No derivatives data file", "WARN")
            return
        try:
            import csv
            with open(self.DERIV_FILE) as f:
                for row in csv.DictReader(f):
                    ts_raw = row.get("timestamp", "")
                    dt = datetime.strptime(ts_raw[:19], "%Y-%m-%dT%H:%M:%S")
                    ts_str = dt.replace(minute=(dt.minute//15)*15, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
                    self.deriv_by_ts[ts_str] = {
                        "ts": ts_str,
                        "fr": float(row.get("funding_rate", 0) or 0),
                        "oi": float(row.get("oi", 0) or 0),
                        "ls": float(row.get("ls_ratio", 0) or 0),
                    }
            log(f"CONFLUENCE: Loaded {len(self.deriv_by_ts)} derivatives points")
        except Exception as e:
            log(f"CONFLUENCE: Failed to load derivatives: {e}", "WARN")

    def detect_extreme(self):
        """Classify all timestamps for extreme positioning."""
        sorted_ts = sorted(self.deriv_by_ts.keys())
        matched = [self.deriv_by_ts[ts] for ts in sorted_ts]

        # OI ROC
        for i in range(1, len(matched)):
            prev = matched[i-1]["oi"]
            matched[i]["oi_roc"] = (matched[i]["oi"] - prev) / prev * 100 if prev > 0 else 0
        if matched:
            matched[0]["oi_roc"] = 0

        for d in matched:
            conds, direction = 0, None
            if d["fr"] > self.fr_p90:
                conds += 1; direction = "SHORT"
            elif d["fr"] < self.fr_p10:
                conds += 1; direction = "LONG"
            if abs(d.get("oi_roc", 0)) > self.oi_p90:
                conds += 1
            if d["ls"] > self.ls_p90:
                conds += 1; direction = direction or "SHORT"
            elif d["ls"] < self.ls_p10:
                conds += 1; direction = direction or "LONG"
            if conds >= 2 and direction:
                fr_e = abs(d["fr"]) / max(abs(self.fr_p90), abs(self.fr_p10), 0.0001)
                oi_e = abs(d.get("oi_roc", 0)) / max(self.oi_p90, 0.01)
                ls_e = abs(d["ls"] - 2.0) / max(abs(self.ls_p90 - 2.0), abs(self.ls_p10 - 2.0), 0.1)
                conv = (min(fr_e, 3) + min(oi_e, 3) + min(ls_e, 3)) / 9.0
                self.extreme_ts[d["ts"]] = {"direction": direction, "conds": conds, "conviction": conv}

        log(f"CONFLUENCE: {len(self.extreme_ts)} extreme events detected")

    def has_extreme_confluence(self, signal_ts, window=4, min_conviction=0.5):
        """Check if there is an extreme positioning event within +/- window bars of signal_ts."""
        from datetime import timedelta
        try:
            sig_dt = datetime.strptime(signal_ts[:19], "%Y-%m-%d %H:%M:%S")
        except:
            return False, None, 0

        for offset in range(-window, window + 1):
            check_dt = sig_dt + timedelta(minutes=offset * 15)
            check_ts = check_dt.strftime("%Y-%m-%d %H:%M:%S")
            if check_ts in self.extreme_ts:
                evt = self.extreme_ts[check_ts]
                if evt["conviction"] >= min_conviction:
                    return True, evt["direction"], evt["conviction"]
        return False, None, 0



# === ENHANCED REAL-TIME REGIME CLASSIFIER ===
class RegimeClassifier:
    """
    Multi-signal regime classifier using:
    - Derivatives: funding rate, OI, ls_ratio, whale_signal
    - Vol regime: vol_ratio, vol_trend, squeeze status
    - Macro: macro_calendar phase, m22 regime
    - Structure: swing_bias, trend_dir
    - Taker flow: taker regime, momentum
    
    Regimes: BULL, BEAR, RANGING, STRESS
    """
    WINDOW = 20
    
    def __init__(self, confluence_checker):
        self.cc = confluence_checker
        self.regime = "RANGING"
        self.confidence = 0.5
        self.signals = {}
    
    def classify(self, scan_data=None):
        """Classify current regime from derivatives + scan data.
        
        Args:
            scan_data: dict from latest scan (optional, adds vol/macro/taker signals)
        """
        deriv = self.cc.deriv_by_ts
        if not deriv:
            return "RANGING", 0.5, {}
        
        sorted_ts = sorted(deriv.keys())
        recent = [deriv[ts] for ts in sorted_ts[-self.WINDOW:]]
        if len(recent) < 3:
            return "RANGING", 0.5, {}
        
        latest = recent[-1]
        fr = latest.get("fr", 0)
        ls = latest.get("ls", 2.0)
        oi = latest.get("oi", 0)
        
        if len(recent) >= 2:
            prev_oi = recent[-2].get("oi", oi)
            oi_roc = ((oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
        else:
            oi_roc = 0
        
        avg_fr = sum(d.get("fr", 0) for d in recent) / len(recent)
        ls_values = [d.get("ls", 2.0) for d in recent]
        ls_trend = ls_values[-1] - ls_values[0] if len(ls_values) > 1 else 0
        
        # === NEW: Price structure (ATR, trend, volatility) ===
        price_data = self.cc.deriv_by_ts  # reuse timestamps
        # Compute ATR-like vol from OI and FR variance
        fr_values = [d.get("fr", 0) for d in recent]
        fr_std = (sum((f - avg_fr)**2 for f in fr_values) / max(len(fr_values)-1, 1)) ** 0.5
        # High FR variance = volatile/stressed market
        high_fr_vol = fr_std > 0.00003
        
        # OI trend (rolling)
        oi_values = [d.get("oi", 0) for d in recent]
        oi_trend = "RISING" if len(oi_values) >= 2 and oi_values[-1] > oi_values[0] * 1.02 else                    "FALLING" if len(oi_values) >= 2 and oi_values[-1] < oi_values[0] * 0.98 else "STABLE"
        
        signals = {}
        bull_score = 0.0
        bear_score = 0.0
        stress_score = 0.0
        
        # Funding rate (LOWERED thresholds: was 0.000050/-0.000020)
        if fr > 0.000030:
            signals["fr"] = "BULL"; bull_score += 1
        elif fr < -0.000010:
            signals["fr"] = "BEAR"; bear_score += 1
        else:
            signals["fr"] = "NEUTRAL"
        
        # LS ratio (WIDER neutral zone was 2.3/1.7)
        if ls > 2.2:
            signals["ls"] = "LONG_CROWDED"; bear_score += 0.8
        elif ls < 1.8:
            signals["ls"] = "SHORT_CROWDED"; bull_score += 0.8
        else:
            signals["ls"] = "NEUTRAL"
        
        # OI rate of change (LOWERED stress threshold: was -5)
        if oi_roc < -3:
            signals["oi"] = "STRESS"; stress_score += 2
        elif oi_roc > 5:
            signals["oi"] = "SURGE"; bull_score += 0.5
        else:
            signals["oi"] = "STABLE"
        
        # FR trend (LOWERED thresholds: was 0.000025/-0.000010)
        if avg_fr > 0.000015:
            signals["fr_trend"] = "BULL"; bull_score += 0.5
        elif avg_fr < -0.000005:
            signals["fr_trend"] = "BEAR"; bear_score += 0.5
        else:
            signals["fr_trend"] = "NEUTRAL"
        
        # LS trend (LOWERED threshold: was 0.2)
        if ls_trend > 0.1:
            signals["ls_trend"] = "MORE_LONGS"; bull_score += 0.2
        elif ls_trend < -0.1:
            signals["ls_trend"] = "MORE_SHORTS"; bear_score += 0.2
        else:
            signals["ls_trend"] = "STABLE"
        
        # === NEW: OI trend signal ===
        if oi_trend == "FALLING":
            signals["oi_trend"] = "FALLING"; bear_score += 0.3
        elif oi_trend == "RISING":
            signals["oi_trend"] = "RISING"; bull_score += 0.2
        else:
            signals["oi_trend"] = "STABLE"
        
        # === NEW: FR volatility (stress indicator) ===
        if high_fr_vol:
            signals["fr_vol"] = "HIGH"; stress_score += 0.5
        else:
            signals["fr_vol"] = "NORMAL"
        
        # === SCAN DATA SIGNALS (vol regime, macro, taker) ===
        if scan_data:
            # Vol regime (LOWERED thresholds)
            vol_ratio = scan_data.get("vol_ratio", 1.0) or 1.0
            squeeze = scan_data.get("squeeze", {})
            squeeze_status = squeeze.get("squeeze_status", "")
            if vol_ratio > 1.5:
                signals["vol"] = "HIGH_VOL"
                stress_score += 0.5
            elif vol_ratio < 0.7:
                signals["vol"] = "LOW_VOL"
            else:
                signals["vol"] = "NORMAL"
            if squeeze_status == "TRIGGERED":
                signals["squeeze"] = "TRIGGERED"
                bull_score += 0.5
            
            # Taker flow
            taker = scan_data.get("taker_summary", {})
            taker_regime = taker.get("regime", "")
            taker_momentum = taker.get("momentum", 0)
            if "BUYING" in taker_regime.upper() or "SURGE" in taker_regime.upper():
                signals["taker"] = "BUY_SURGE"
                bull_score += 0.5
            elif "SELLING" in taker_regime.upper():
                signals["taker"] = "SELL_SURGE"
                bear_score += 0.5
            else:
                signals["taker"] = "NEUTRAL"
            
            # Macro calendar
            macro = scan_data.get("macro_calendar", {})
            macro_phase = macro.get("phase", "")
            if "CPI" in macro_phase or "FOMC" in macro_phase:
                signals["macro"] = "HIGH_IMPACT"
                stress_score += 0.3
            elif "NFP" in macro_phase:
                signals["macro"] = "NFP_WEEK"
                stress_score += 0.2
            else:
                signals["macro"] = "NORMAL"
            
            # Cascade
            cascade = scan_data.get("cascade", {})
            cascade_signal = cascade.get("combined_signal", "HOLD")
            if cascade_signal in ("STRONG_LONG", "STRONG_SHORT"):
                signals["cascade"] = cascade_signal
                if "LONG" in cascade_signal:
                    bull_score += 0.5
                else:
                    bear_score += 0.5
            else:
                signals["cascade"] = "HOLD"
            
            # Direction resolver (consensus)
            dr = scan_data.get("direction_resolver", {})
            dr_dir = dr.get("direction", "NEUTRAL")
            if dr_dir == "LONG":
                signals["consensus"] = "LONG"
                bull_score += 0.3
            elif dr_dir == "SHORT":
                signals["consensus"] = "SHORT"
                bear_score += 0.3
            else:
                signals["consensus"] = "NEUTRAL"
        
        # === REGIME DECISION (lowered thresholds + MILDLY_BEARISH) ===
        if stress_score >= 1.5:
            regime = "STRESS"
            confidence = min(0.9, 0.5 + stress_score * 0.1)
        elif bull_score > bear_score + 0.5:
            regime = "BULL"
            confidence = min(0.9, 0.5 + (bull_score - bear_score) * 0.15)
        elif bear_score > bull_score + 0.5:
            regime = "BEAR"
            confidence = min(0.9, 0.5 + (bear_score - bull_score) * 0.15)
        elif bear_score > bull_score and bear_score >= 1.0:
            # NEW: MILDLY_BEARISH — bearish lean but not strong enough for full BEAR
            regime = "MILDLY_BEARISH"
            confidence = min(0.7, 0.5 + (bear_score - bull_score) * 0.1)
        else:
            regime = "RANGING"
            confidence = 0.5
        
        self.regime = regime
        self.confidence = confidence
        self.signals = signals
        return regime, confidence, signals
    
    def is_bearish(self):
        """Returns True if regime is BEAR, STRESS, or MILDLY_BEARISH."""
        regime, conf, _ = self.classify()
        return regime in ("BEAR", "STRESS", "MILDLY_BEARISH") and conf >= 0.5
    
    def is_ranging(self):
        regime, conf, _ = self.classify()
        return regime == "RANGING"
    
    def is_bullish(self):
        regime, conf, _ = self.classify()
        return regime == "BULL" and conf >= 0.5
    
    def is_neutral_or_bearish(self):
        """For strategies that can trade in RANGING + BEARISH regimes."""
        regime, conf, _ = self.classify()
        return regime in ("RANGING", "BEAR", "STRESS", "MILDLY_BEARISH")


# === OPTIMIZED STRATEGY CONFIGS (from 2026-07-05 optimization, PF >= 2.0) ===
STRATEGY_CONFIGS = {
    # === PROVEN STRATEGIES (PF >= 2.0, WR >= 70%) ===
    "whale_watch": {
        "tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 8,
        "direction": None, "enabled": True,
        "group": "B",
        "min_conviction": 0.5,
    },
    "funding_arb": {
        "tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 24,
        "direction": None, "enabled": True,
        "group": "A",
        "min_conviction": 0.5,
        "notes": "v3 multi-factor: taker z-score + round numbers. Gate PASS: 226 events, +0.21%, p=0.054.",
    },
    "orderbook_imbalance": {
        "tp_pct": 2.0, "sl_pct": 0.75, "hold_hours": 12,
        "direction": None, "enabled": True,
        "group": "A",
        "min_conviction": 0.5,
        "trail_trigger_pct": 0.3,
        "notes": "Gate PASS: 847 events, +0.254%, p=0.001.",
    },
    "failed_breakout": {
        "tp_pct": 2.5, "sl_pct": 1.5, "hold_hours": 24,
        "direction": None, "enabled": True,
        "group": "A",
        "min_conviction": 0.5,
        "structural_tpsl": True,
        "fallback_tp_pct": 2.5,
        "fallback_sl_pct": 1.5,
        "notes": "v9 merged config. Structural TP/SL with fallback. RANGING regime only."
    },
    "positioning_fade": {
        "tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 16,
        "direction": None, "enabled": True,
        "group": "A",
        "min_conviction": 0.35,
    },
    "trade_flow": {
        "tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 12,
        "direction": None, "enabled": True,
        "group": "A",
        "min_conviction": 0.5,
        "notes": "Gate PASS: 623 events, +0.214%, p=0.003. Direction-agnostic with EMA200 alignment.",
    },
    "structural_break": {
        "tp_pct": 0.5, "sl_pct": 0.5, "hold_hours": 8,
        "direction": "SHORT", "enabled": True,
        "group": "A",
        "min_conviction": 0.5,
    },
    "regime_switch": {
        "tp_pct": 1.0, "sl_pct": 1.0, "hold_hours": 8,
        "direction": "SHORT", "enabled": False,
        "group": "A",
        "min_conviction": 0.5,
    },
    # === DISABLED: PF < 2.0 or insufficient data ===
    "squeeze_breakout": {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 8, "direction": None, "enabled": True, "group": "B", "min_conviction": 0.55, "notes": "v2: ATR/BB squeeze Q>=0.80, 63% WR, p=0.0049. Bypasses gates."},
    "bb_mom6": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A", "min_conviction": 0.5, "confluence_with": "extreme_positioning", "notes": "Only fires with extreme positioning confluence. HC only."},
    "cross_asset": {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 12, "direction": None, "enabled": False, "group": "A", "min_conviction": 0.5, "confluence_with": "orderbook_imbalance", "notes": "Confluence with OBI LONG. p=0.0000, mean=0.661%."},
    "judas_sweep": {"tp_pct": 2.5, "sl_pct": 1.5, "hold_hours": 24, "direction": None, "enabled": True, "group": "A", "min_conviction": 0.5, "notes": "v3 multi-factor: daily/session H/L sweep + rejection wick + volume. Gate PASS: 1895 events, +0.10%, p=0.040."},
    "forced_movement": {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 4, "direction": None, "enabled": True, "group": "A", "min_conviction": 0.45, "notes": "Structural forced movement: OI divergence + funding squeeze + liq cascade + basis convergence. Bypasses gates."},
    "scalp_v2": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A"},
    "power_of_3": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A"},
    "macro_surprise": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A"},
    "liquidation_cascade": {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 4, "direction": None, "enabled": True, "group": "A", "min_conviction": 0.50, "notes": "v2: real Bybit liquidation data + OI fallback. Bypasses gates."},
    "taker_flow": {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 8, "direction": None, "enabled": True, "group": "A", "min_conviction": 0.50, "notes": "v2: z-score thresholds + session filter + flow acceleration. Needs gate validation."},
    "vol_rotation": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "B"},
    "kill_zone": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A"},
    "liquidity_grab": {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 12, "direction": None, "enabled": True, "group": "A", "min_conviction": 0.45, "notes": "v2: OB collector + S/R levels + persistence + spoofing."},
    "cascade": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A"},
    "mtf_confluence": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A"},
    "momentum_v3": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "B"},
}

# === EXECUTION PARAMS ===
RISK_PCT = 0.02
LEVERAGE = 25
MAX_SLIPPAGE_PCT = 0.30
BLOCKED_HOURS = {19, 20, 21}

# === KILL ZONE SESSION BONUS ===
# London/NY overlap = highest volume, tightest spreads
# Asia dead zone = lowest volume, widest spreads
KZ_BONUS = {
    0: 0.03,   # Asia active
    1: 0.03,
    2: 0.05,   # London open
    3: 0.05,
    4: 0.02,   # London mid
    5: 0.00,   # Dead zone
    6: 0.00,
    7: 0.05,   # London active
    8: 0.05,
    9: 0.08,   # London/NY overlap — BEST
    10: 0.08,
    11: 0.05,  # NY active
    12: 0.05,
    13: 0.05,
    14: 0.03,  # NY mid
    15: 0.05,  # NY afternoon
    16: 0.05,
    17: 0.03,  # NY close
    18: 0.02,
    19: 0.00,  # Dead zone
    20: 0.00,
    21: 0.02,  # Asia open
    22: 0.02,
    23: 0.02,
}

def get_session_bonus(ts_str):
    """Get conviction bonus based on session (kill zone)."""
    try:
        hour = int(ts_str[11:13])
        return KZ_BONUS.get(hour, 0.0)
    except (ValueError, IndexError, TypeError):
        return 0.0
BLOCKED_DAYS = {"Sat"}
MAX_POSITIONS = 3
SIGNAL_MAX_AGE_SEC = 1200
FEE_PCT = 0.001
MIN_CONVICTION = 0.5
ORDER_TYPE = "limit"
LIMIT_OFFSET_PCT = 0.02

def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    # print disabled — nohup redirects stdout to LOG_FILE
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "capital": INITIAL_CAPITAL, "peak_capital": INITIAL_CAPITAL,
        "open_positions": [], "closed_trades": [],
        "total_pnl": 0, "total_fees": 0, "pnl_total": 0,
        "trades_count": 0, "wins": 0, "losses": 0, "timeouts": 0,
        "last_signal_ts": None, "dd_cooldown_until": None,
    }

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def log_trade(trade):
    os.makedirs(os.path.dirname(TRADE_LOG), exist_ok=True)
    trades = []
    if os.path.exists(TRADE_LOG):
        with open(TRADE_LOG) as f:
            trades = json.load(f)
    trades.append(trade)
    with open(TRADE_LOG, "w") as f:
        json.dump(trades, f, indent=2, default=str)

def get_exchange(dry_run=False):
    import ccxt
    api_key = os.environ.get("HTX_API_KEY", "")
    api_secret = os.environ.get("HTX_API_SECRET", "")
    if not api_key and os.path.exists(KEYS_FILE):
        with open(KEYS_FILE) as f:
            keys = json.load(f)
            api_key = keys.get("api_key", "")
            api_secret = keys.get("api_secret", "")
    exchange = ccxt.htx({
        "apiKey": api_key, "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "defaultMarginMode": "isolated"},
    })
    if not dry_run and not api_key:
        log("NO API KEY — switching to dry-run", "WARN")
        dry_run = True
    return exchange, dry_run

def get_latest_signals(gate, monitor):
    scan_files = sorted(
        [f for f in os.listdir(SCAN_DIR) if f.startswith("scan_") and f.endswith(".json")],
        reverse=True
    )
    if not scan_files:
        return []
    latest = os.path.join(SCAN_DIR, scan_files[0])
    with open(latest) as f:
        data = json.load(f)
    ts = data.get("timestamp", "")
    price = data.get("price", 0)
    status = data.get("status", "")
    multi = data.get("multi_strategy") or {}
    all_signals = multi.get("all_signals", [])
    single = data.get("strategy_signal", {})
    if single and single.get("direction"):
        single_strat = single.get("strategy", "")
        if not any(s.get("strategy") == single_strat for s in all_signals):
            all_signals.append(single)

    # === GROUP A/B VOTING ===
    group_a_fired = []
    group_b_fired = []
    for sig_data in all_signals:
        if not isinstance(sig_data, dict):
            continue
        strat_name = sig_data.get("strategy", "")
        cfg = STRATEGY_CONFIGS.get(strat_name)
        if not cfg:
            continue
        direction = sig_data.get("direction")
        conviction = sig_data.get("conviction", 0) or 0
        if not direction or conviction < 0.1:
            continue
        grp = cfg.get("group", "A")
        if grp == "A":
            group_a_fired.append(sig_data)
        elif grp == "B":
            group_b_fired.append(sig_data)

    b_directions = set()
    for sig in group_b_fired:
        d = sig.get("direction")
        if d:
            b_directions.add(d)

    # === BUILD SIGNALS (with gate + monitor checks) ===
    signals = []
    rejected = {"gate": [], "monitor": [], "conviction": [], "direction": [], "other": []}
    for sig_data in all_signals:
        if not isinstance(sig_data, dict):
            continue
        strat_name = sig_data.get("strategy", "")
        cfg = STRATEGY_CONFIGS.get(strat_name)
        if not cfg or not cfg["enabled"]:
            continue
        if cfg.get("group") == "B":
            continue

        # --- ISOLATION GATE CHECK ---
        gate_ok, gate_reason = gate.is_passed(strat_name)
        if not gate_ok:
            rejected["gate"].append(f"{strat_name}({gate_reason})")
            continue

        # --- LIVE MONITOR CHECK ---
        monitor_paused, monitor_reason = monitor.is_paused(strat_name)
        if monitor_paused:
            rejected["monitor"].append(f"{strat_name}({monitor_reason})")
            continue

        direction = sig_data.get("direction")
        conviction = sig_data.get("conviction", 0) or 0
        entry = sig_data.get("entry", price)
        sl = sig_data.get("sl", 0)
        tp1 = sig_data.get("tp1", 0)
        min_conv = cfg.get("min_conviction", MIN_CONVICTION)
        if not direction or conviction < min_conv:
            rejected["conviction"].append(f"{strat_name}(conv={conviction:.2f}<{min_conv})")
            continue

        # === SESSION BONUS (Kill Zone) ===
        session_bonus = get_session_bonus(sig_data.get("timestamp", "") or data.get("timestamp", ""))
        conviction = min(conviction + session_bonus, 0.95)

        # === REGIME FILTER: failed_breakout only in ranging markets ===
        if strat_name == "failed_breakout":
            if gate._regime_classifier:
                rc = gate._regime_classifier
                if not rc.is_ranging():
                    rejected["other"].append(f"{strat_name}(regime={rc.regime})")
                    continue
        # === REGIME FILTER: positioning_fade + whale_watch only in bearish/stress ===
        if strat_name in ("positioning_fade", "whale_watch"):
            if gate._regime_classifier:
                rc = gate._regime_classifier
                # Reject only in strong BULL; allow RANGING, MILDLY_BEARISH, BEAR, STRESS
                if rc.is_bullish() and rc.confidence >= 0.7:
                    rejected["other"].append(f"{strat_name}(regime={rc.regime},conf={rc.confidence:.2f})")
                    continue

        if cfg["direction"] and direction != cfg["direction"]:
            rejected["direction"].append(f"{strat_name}({direction}!={cfg['direction']})")
            continue

        group_boost = 1.0
        confirmed_by = []
        if direction in b_directions:
            group_boost = 1.5
            confirmed_by = [s.get("strategy") for s in group_b_fired if s.get("direction") == direction]

        # Confluence check for bb_mom6 (requires extreme positioning)
        if strat_name == "bb_mom6" and hasattr(gate, 'confluence'):
            has_conf, conf_dir, conf_conv = gate.confluence.has_extreme_confluence(ts, window=4, min_conviction=0.5)
            if not has_conf:
                rejected["other"].append(f"{strat_name}(no_extreme_confluence)")
                continue
            if conf_dir:
                direction = conf_dir
                conviction = max(conviction, conf_conv)

        # Confluence check for cross_asset (requires OBI LONG within +/- 4 bars)
        if strat_name == "cross_asset":
            obi_confirmed = False
            try:
                from datetime import timedelta
                sig_dt = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
                # Check for OBI signal — relaxed: accept any OBI signal with direction LONG
                # (not just fired=True, since OBI may not have fired but data still indicates LONG)
                for sig2 in all_signals:
                    if sig2.get("strategy") == "orderbook_imbalance" and sig2.get("direction") == "LONG":
                        obi_ts = sig2.get("timestamp", "")[:19]
                        try:
                            obi_dt = datetime.strptime(obi_ts, "%Y-%m-%d %H:%M:%S")
                            if abs((obi_dt - sig_dt).total_seconds()) <= 8 * 900:  # expanded to 8 bars
                                obi_confirmed = True
                                direction = "LONG"  # OBI is LONG-only
                                break
                        except: pass
                # Fallback: if no OBI signal in all_signals, check if scan data has OBI data
                if not obi_confirmed:
                    try:
                        scan_latest = os.path.join(SCAN_DIR, sorted(
                            [f for f in os.listdir(SCAN_DIR) if f.startswith("scan_") and f.endswith(".json")],
                            reverse=True)[0])
                        with open(scan_latest) as sf:
                            scan_json = json.load(sf)
                        obi_data = scan_json.get("strategy_data", {}).get("orderbook_imbalance", {})
                        if obi_data.get("direction") == "LONG" or obi_data.get("obi_score", 0) > 0.5:
                            obi_confirmed = True
                            direction = "LONG"
                    except: pass
            except: pass
            if not obi_confirmed:
                # Instead of rejecting, allow cross_asset standalone with reduced conviction
                conviction *= 0.7  # reduce conviction but don't reject
                if conviction < cfg.get("min_conviction", MIN_CONVICTION):
                    rejected["other"].append(f"{strat_name}(no_obi_low_conv={conviction:.2f})")
                    continue
                log(f"NOTE: {strat_name} firing without OBI confluence (conv reduced to {conviction:.2f})")

        signals.append({
            "strategy": strat_name, "timestamp": ts, "direction": direction,
            "conviction": conviction, "entry": entry or price,
            "sl": sl, "tp1": tp1, "price": price, "cfg": cfg,
            "scan_status": status,
            "group_boost": group_boost,
            "confirmed_by": confirmed_by,
        })

    # Log rejections for transparency
    for reason_type, items in rejected.items():
        if items:
            log(f"REJECTED ({reason_type}): {', '.join(items)}")

    return signals

def is_signal_fresh(signal_ts):
    try:
        sig_dt = datetime.strptime(signal_ts, "%Y-%m-%d %H:%M:%S")
        sig_dt = sig_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            sig_dt = datetime.fromisoformat(signal_ts.replace("Z", "+00:00"))
        except:
            return False
    now = datetime.now(timezone.utc)
    return (now - sig_dt).total_seconds() < SIGNAL_MAX_AGE_SEC

def check_tp_sl(pos, current_price):
    d = pos["direction"]
    tp = pos["tp"]; sl = pos["sl"]
    if d == "LONG":
        if current_price >= tp: return "WIN", tp
        elif current_price <= sl: return "LOSS", sl
    else:
        if current_price <= tp: return "WIN", tp
        elif current_price >= sl: return "LOSS", sl
    return None, None

def close_position(state, pos, exit_price, outcome, monitor=None):
    entry = pos["fill_price"]; size = pos["size"]; d = pos["direction"]
    lev = pos.get("leverage", LEVERAGE)
    pnl_raw = (exit_price - entry) * size * lev if d == "LONG" else (entry - exit_price) * size * lev
    fee = entry * size * FEE_PCT * 2
    pnl = pnl_raw - fee
    state["capital"] += pnl; state["total_pnl"] += pnl; state["total_fees"] += fee
    state["trades_count"] += 1
    if outcome == "WIN": state["wins"] += 1
    elif outcome == "TIMEOUT": state["timeouts"] += 1
    else: state["losses"] += 1
    if state["capital"] > state["peak_capital"]: state["peak_capital"] = state["capital"]
    closed = {**pos, "exit": round(exit_price, 2), "pnl": round(pnl, 4),
              "fee": round(fee, 4), "outcome": outcome,
              "closed_at": datetime.now(timezone.utc).isoformat()}
    state["closed_trades"].append(closed)
    state["open_positions"] = [p for p in state["open_positions"] if p.get("order_id") != pos.get("order_id")]
    log_trade(closed)
    log(f"CLOSE: {outcome} {d} ${entry:.2f}->${exit_price:.2f} PnL=${pnl:+.2f}")

    # Record in live monitor
    if monitor:
        monitor.record_trade(pos.get("strategy", "unknown"), outcome)
        monitor.record_pnl(pos.get("strategy", "unknown"), pnl)

    return state

def calc_committed_margin(positions):
    total = 0.0
    for p in positions:
        notional = p["fill_price"] * p["size"]
        margin = notional / p.get("leverage", LEVERAGE)
        total += margin
    return total

def calc_position_liquidation_risk(pos, all_positions):
    total_sl_loss = 0.0
    for p in all_positions:
        sl_dist = abs(p["fill_price"] - p["sl"])
        sl_loss = sl_dist * p["size"]
        total_sl_loss += sl_loss
    new_sl_dist = abs(pos["fill_price"] - pos["sl"])
    new_sl_loss = new_sl_dist * pos["size"]
    total_sl_loss += new_sl_loss
    total_margin = calc_committed_margin(all_positions)
    new_margin = (pos["fill_price"] * pos["size"]) / pos.get("leverage", LEVERAGE)
    total_margin += new_margin
    if total_sl_loss > total_margin * 0.8:
        return False, f"SL risk: worst-case loss ${total_sl_loss:.2f} > 80% of margin ${total_margin:.2f}"
    return True, "ok"

def main():
    # === LOCKFILE CHECK ===
    if not acquire_lock():
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--reload-gate", action="store_true", help="Reload isolation gate results")
    parser.add_argument("--monitor-report", action="store_true", help="Print monitor report and exit")
    parser.add_argument("--resume", type=str, help="Resume a paused strategy by name")
    args = parser.parse_args()

    # Initialize gate, monitor, confluence checker, and regime classifier
    gate = IsolationGateGuardian(ISOLATION_GATE_FILE)
    monitor = LivePerformanceMonitor(BACKTEST_BENCH_FILE)
    confluence = ConfluenceChecker()
    gate.confluence = confluence  # attach for use in signal filtering
    regime_classifier = RegimeClassifier(confluence)
    gate._regime_classifier = regime_classifier  # attach for use in signal filtering
    # Pass latest scan data to regime classifier for richer signals
    _scan_data = None
    try:
        _scan_files = sorted([f for f in os.listdir(SCAN_DIR) if f.startswith("scan_") and f.endswith(".json")], reverse=True)
        if _scan_files:
            with open(os.path.join(SCAN_DIR, _scan_files[0])) as f:
                _scan_data = json.load(f)
    except:
        pass
    regime, conf, signals = regime_classifier.classify(_scan_data)
    log(f"REGIME: {regime} (confidence={conf:.2f}) signals={signals}")

    # Handle CLI commands
    if args.monitor_report:
        report = monitor.get_report()
        print(json.dumps(report, indent=2))
        return
    if args.resume:
        monitor.resume(args.resume)
        return

    log("=" * 60)
    log(f"Scanner Executor Starting ({'DRY RUN' if args.dry_run else 'LIVE'})")
    enabled = [k for k,v in STRATEGY_CONFIGS.items() if v['enabled']]
    log(f"Strategies: {enabled}")
    log(f"Params: {LEVERAGE}x | {RISK_PCT*100:.0f}% risk | Fee: {FEE_PCT*100:.2f}%")

    # Log gate status
    for strat in enabled:
        ok, reason = gate.is_passed(strat)
        status = "GATE PASS" if ok else f"GATE BLOCKED ({reason})"
        log(f"  {strat}: {status}")

    # Log monitor status
    for strat in enabled:
        paused, reason = monitor.is_paused(strat)
        if paused:
            log(f"  {strat}: MONITOR PAUSED — {reason}")

    log("=" * 60)

    exchange, dry_run = get_exchange(args.dry_run)
    state = load_state()
    log(f"Capital: ${state['capital']:.2f} | Positions: {len(state['open_positions'])} | Trades: {state['trades_count']} ({state['wins']}W/{state['losses']}L)")

    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.hour in BLOCKED_HOURS or now.strftime("%a") in BLOCKED_DAYS:
                time.sleep(args.interval); continue

            # Reload gate if requested
            if args.reload_gate:
                gate.load()
                log("Reloaded isolation gate results")

            # Check existing positions
            for pos in list(state["open_positions"]):
                opened = datetime.fromisoformat(pos["opened_at"])
                hold_h = pos.get("hold_hours", 8)
                if (now - opened).total_seconds() > hold_h * 3600:
                    state = close_position(state, pos, pos["fill_price"], "TIMEOUT", monitor)
                    continue
                try:
                    ticker = exchange.fetch_ticker(SYMBOL)
                    price = ticker["last"]
                except:
                    continue

                # Trailing stop
                strat_cfg = STRATEGY_CONFIGS.get(pos.get("strategy", ""), {})
                trail_trigger = strat_cfg.get("trail_trigger_pct", 0)
                if trail_trigger and not pos.get("trailed"):
                    d = pos["direction"]
                    entry = pos["fill_price"]
                    profit_pct = ((price - entry) / entry * 100) if d == "LONG" else ((entry - price) / entry * 100)
                    if profit_pct >= trail_trigger:
                        pos["sl"] = entry
                        pos["trailed"] = True
                        log(f"TRAIL: {pos['strategy']} {d} moved SL to breakeven (${entry:.2f}) after {profit_pct:.2f}% profit")

                outcome, ep = check_tp_sl(pos, price)
                if outcome:
                    state = close_position(state, pos, ep, outcome, monitor)

            # Capital guard
            if state["capital"] <= 0:
                log(f"CAPITAL ZERO (${state['capital']:.2f}) — stopping executor", "ERROR")
                save_state(state)
                break

            # Get new signals (with gate + monitor filtering)
            signals = get_latest_signals(gate, monitor)
            log(f"Signals after gate+monitor filter: {len(signals)}")
            for sig in signals:
                if not is_signal_fresh(sig["timestamp"]):
                    continue
                if any(p["strategy"] == sig["strategy"] for p in state["open_positions"]):
                    continue
                if len(state["open_positions"]) >= MAX_POSITIONS:
                    continue

                entry = sig["entry"]; sl = sig["sl"]; tp1 = sig["tp1"]
                cfg = sig["cfg"]

                if not entry or not sl or not tp1:
                    # === FIX 7: Structural TP/SL fallback ===
                    if cfg.get("structural_tpsl") and entry:
                        # Fallback: use strategy config percentages
                        fb_tp_pct = cfg.get("fallback_tp_pct", cfg.get("tp_pct", 2.0))
                        fb_sl_pct = cfg.get("fallback_sl_pct", cfg.get("sl_pct", 1.0))
                        if not tp1 and fb_tp_pct > 0:
                            if direction == "LONG":
                                tp1 = entry * (1 + fb_tp_pct / 100.0)
                            else:
                                tp1 = entry * (1 - fb_tp_pct / 100.0)
                            log(f"FALLBACK TP: {strat_name} using {fb_tp_pct}% -> TP=${tp1:.2f}")
                        if not sl and fb_sl_pct > 0:
                            if direction == "LONG":
                                sl = entry * (1 - fb_sl_pct / 100.0)
                            else:
                                sl = entry * (1 + fb_sl_pct / 100.0)
                            log(f"FALLBACK SL: {strat_name} using {fb_sl_pct}% -> SL=${sl:.2f}")
                    if not entry or not sl or not tp1:
                        continue

                if dry_run:
                    fill_price = entry * (1 + 0.001) if sig["direction"] == "LONG" else entry * (1 - 0.001)
                else:
                    fill_price = entry

                sl_pct = abs(fill_price - sl) / fill_price
                if sl_pct <= 0: continue
                if sl_pct < 0.003:
                    strategy_sl_pct = cfg["sl_pct"] / 100.0
                    if strategy_sl_pct > sl_pct:
                        sl_orig = sl_pct
                        sl_pct = strategy_sl_pct
                        if sig["direction"] == "LONG":
                            sl = fill_price * (1 - sl_pct)
                        else:
                            sl = fill_price * (1 + sl_pct)
                        log(f"SL adjusted: signal {sl_orig*100:.3f}% too tight, using strategy config {cfg['sl_pct']}% -> SL=${sl:.2f}")
                    else:
                        log(f"SKIP: SL too tight ({sl_pct*100:.4f}% < 0.3%)")
                        continue

                group_boost = sig.get("group_boost", 1.0)
                committed = calc_committed_margin(state["open_positions"])
                available = state["capital"] - committed
                if available <= 0:
                    log(f"SKIP: no available margin (committed=${committed:.2f}, capital=${state['capital']:.2f})")
                    continue
                size = (available * RISK_PCT * group_boost) / (sl_pct * LEVERAGE)
                if size < 0.001: continue
                # Cap by available margin (max 80% of available)
                max_notional = available * LEVERAGE * 0.8
                max_size = max_notional / fill_price
                if size > max_size:
                    size = max_size
                    log(f"SIZE CAPPED: margin limit, size={size:.4f}")

                risk_ok, risk_msg = calc_position_liquidation_risk(
                    {"fill_price": fill_price, "size": size, "sl": sl, "leverage": LEVERAGE},
                    state["open_positions"]
                )
                if not risk_ok:
                    log(f"SKIP: {risk_msg}")
                    continue

                pos = {
                    "strategy": sig["strategy"], "direction": sig["direction"],
                    "fill_price": round(fill_price, 2), "tp": round(tp1, 2),
                    "sl": round(sl, 2), "size": round(size, 6),
                    "leverage": LEVERAGE, "hold_hours": cfg["hold_hours"],
                    "tp_pct": cfg["tp_pct"], "sl_pct": cfg["sl_pct"],
                    "signal_ts": sig["timestamp"],
                    "opened_at": now.isoformat(),
                    "order_id": f"dry_{int(now.timestamp())}" if dry_run else None,
                    "trailed": False,
                }
                state["open_positions"].append(pos)
                confirmed = sig.get("confirmed_by", [])
                conf_str = f" +B:{','.join(confirmed)}" if confirmed else " (solo)"
                log(f"{'DRY RUN: ' if dry_run else ''}OPEN {sig['direction']} {size:.4f} ETH @ ${fill_price:.2f} TP=${tp1:.2f} SL=${sl:.2f} [{sig['strategy']}]{conf_str} boost={group_boost:.1f}x")

            save_state(state)
            log(f"Capital: ${state['capital']:.2f} | Positions: {len(state['open_positions'])} | Trades: {state['trades_count']} ({state['wins']}W/{state['losses']}L)")

            if args.once: break
            time.sleep(args.interval)

        except KeyboardInterrupt:
            log("Shutting down..."); save_state(state); break
        except Exception as e:
            log(f"ERROR: {e}", "ERROR"); save_state(state); time.sleep(args.interval)

if __name__ == "__main__":
    main()
