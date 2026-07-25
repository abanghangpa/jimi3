#!/usr/bin/env python3
"""
RegimeClassifierV5 — Hybrid daily/15m regime classifier.

Combines:
- V4 daily timeframe (correct for regime detection)
- Contradiction detection (taker/consensus vs FR/LS conflicts)
- Hysteresis (3 consecutive signals + 5-day cooldown)
- Fallback to V3-style 15m when daily data unavailable
- Regime-conditional sizing output (replaces binary COND_GATE)

References:
- Hamilton (1989), Ang & Timmermann (2012), Shu et al. (2024)
- Bieganowski & Ślepaczuk (2026), Nguyen Van (2026)
"""

import json, os, sys, time, math, threading, requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

REGIME_SIZING = {
    "BULL": {
        "tp_scale": 1.0, "sl_scale": 1.0, "size_mult": 1.0,
        "long_ok": True, "short_ok": True, "short_size_mult": 0.5,
        "notes": "Full conviction, trend-friendly",
    },
    "BEAR": {
        "tp_scale": 1.0, "sl_scale": 1.15, "size_mult": 0.7,
        "long_ok": True, "short_ok": True,
        "short_size_mult": 1.0, "long_size_mult": 0.5,
        "notes": "Reduced size, wider SL, shorts favored",
    },
    "RANGING": {
        "tp_scale": 0.9, "sl_scale": 0.9, "size_mult": 1.0,
        "long_ok": True, "short_ok": True, "short_size_mult": 1.0,
        "notes": "Tight TP/SL, mean-reversion friendly",
    },
    "STRESS": {
        "tp_scale": 1.3, "sl_scale": 1.4, "size_mult": 0.5,
        "long_ok": True, "short_ok": True, "short_size_mult": 0.5,
        "notes": "Small size, wide stops, protect capital",
    },
    "MILDLY_BEARISH": {
        "tp_scale": 1.0, "sl_scale": 1.05, "size_mult": 0.85,
        "long_ok": True, "short_ok": True,
        "short_size_mult": 0.8, "long_size_mult": 0.7,
        "notes": "Slight bearish lean, both directions with reduced longs",
    },
}


class RegimeClassifierV5:
    """Hybrid daily/15m regime classifier with contradiction detection."""

    DAILY_CACHE_TTL = 3600
    DAILY_CANDLE_LIMIT = 200
    HYSTERESIS_BARS = 3
    COOLDOWN_SECONDS = 5 * 86400

    def __init__(self, confluence_checker):
        self.cc = confluence_checker
        self.regime = "RANGING"
        self.confidence = 0.5
        self.signals = {}
        self.sizing = REGIME_SIZING["RANGING"].copy()
        self._v4 = None
        self._v4_imported = False
        self._daily_candles = None
        self._daily_candles_ts = 0
        self._weekly_candles = None
        self._regime_start_ts = None
        self._last_transition_ts = None
        self._consecutive_votes = deque(maxlen=10)
        self._fallback_confidence = 0.5
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from regime_classifier_v4 import RegimeClassifierV4
            self._v4_class = RegimeClassifierV4
            self._v4_imported = True
        except ImportError:
            self._v4_class = None

    def classify(self, scan_data=None):
        now = time.time()
        signals = {}
        daily_regime, daily_conf, daily_signals = self._classify_daily(now)
        signals["daily"] = daily_signals
        contradictions = self._detect_contradictions(daily_regime, scan_data)
        signals["contradictions"] = contradictions
        adjusted_regime, adjusted_conf = self._apply_contradictions(daily_regime, daily_conf, contradictions)
        final_regime = self._apply_hysteresis(adjusted_regime, adjusted_conf, now)
        self.sizing = REGIME_SIZING.get(final_regime, REGIME_SIZING["RANGING"]).copy()
        if scan_data:
            direction = scan_data.get("direction_resolver", {}).get("direction", "NEUTRAL")
            if direction == "SHORT" and "short_size_mult" in self.sizing:
                self.sizing["effective_size_mult"] = self.sizing["size_mult"] * self.sizing.get("short_size_mult", 1.0)
            elif direction == "LONG" and "long_size_mult" in self.sizing:
                self.sizing["effective_size_mult"] = self.sizing["size_mult"] * self.sizing.get("long_size_mult", 1.0)
            else:
                self.sizing["effective_size_mult"] = self.sizing["size_mult"]
        self.regime = final_regime
        self.confidence = adjusted_conf
        self.signals = signals
        return final_regime, adjusted_conf, signals

    def _classify_daily(self, now):
        if not self._v4_imported:
            return self._fallback_15m(now), self._fallback_confidence, {"source": "15m_fallback"}
        if self._daily_candles is None or now - self._daily_candles_ts > self.DAILY_CACHE_TTL:
            self._fetch_daily_candles()
            self._daily_candles_ts = now
        if not self._daily_candles or len(self._daily_candles) < 50:
            return self._fallback_15m(now), self._fallback_confidence, {"source": "15m_fallback_no_daily"}
        if self._v4 is None:
            self._v4 = self._v4_class()
        deriv_daily = self._get_deriv_daily()
        regime, conf, signals = self._v4.classify(self._daily_candles, weekly_candles=self._weekly_candles, deriv_daily=deriv_daily, current_ts=now)
        signals["source"] = "v4_daily"
        return regime, conf, signals

    def _fetch_daily_candles(self):
        try:
            r = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": "ETHUSDT", "interval": "1d", "limit": self.DAILY_CANDLE_LIMIT}, timeout=15)
            r.raise_for_status()
            self._daily_candles = []
            for c in r.json():
                self._daily_candles.append({"ts": datetime.fromtimestamp(c[0]/1000, tz=timezone.utc), "open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])})
            r2 = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": "ETHUSDT", "interval": "1w", "limit": 52}, timeout=15)
            r2.raise_for_status()
            self._weekly_candles = []
            for c in r2.json():
                self._weekly_candles.append({"ts": datetime.fromtimestamp(c[0]/1000, tz=timezone.utc), "open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])})
        except Exception:
            pass

    def _get_deriv_daily(self):
        if not self.cc or not hasattr(self.cc, "deriv_by_ts"):
            return None
        deriv = self.cc.deriv_by_ts
        if not deriv:
            return None
        daily_groups = defaultdict(list)
        for ts, data in deriv.items():
            try:
                if isinstance(ts, (int, float)):
                    day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                else:
                    day = str(ts)[:10]
                daily_groups[day].append(data)
            except:
                continue
        if not daily_groups:
            return None
        latest_day = sorted(daily_groups.keys())[-1]
        rows = daily_groups[latest_day]
        if not rows:
            return None
        fr_vals = [r.get("fr", 0) for r in rows if r.get("fr") is not None]
        ls_vals = [r.get("ls", 2.0) for r in rows if r.get("ls") is not None]
        oi_vals = [r.get("oi", 0) for r in rows if r.get("oi") is not None]
        taker_vals = [r.get("taker", 1.0) for r in rows if r.get("taker") is not None]
        return {
            "fr_avg": sum(fr_vals)/len(fr_vals) if fr_vals else 0,
            "ls_avg": sum(ls_vals)/len(ls_vals) if ls_vals else 2.0,
            "taker_avg": sum(taker_vals)/len(taker_vals) if taker_vals else 1.0,
            "oi_trend": ((oi_vals[-1]-oi_vals[0])/oi_vals[0]*100) if len(oi_vals)>=2 and oi_vals[0]>0 else 0,
        }

    def _detect_contradictions(self, daily_regime, scan_data):
        if not scan_data:
            return {"count": 0, "details": []}
        contradictions = []
        bull_signals = 0
        bear_signals = 0
        taker = scan_data.get("taker_summary", {})
        taker_regime = taker.get("regime", "")
        if "BUYING" in taker_regime.upper() or "SURGE" in taker_regime.upper():
            bull_signals += 1
            if daily_regime in ("BEAR", "STRESS"):
                contradictions.append(f"taker=BUY_SURGE but regime={daily_regime}")
        elif "SELLING" in taker_regime.upper():
            bear_signals += 1
            if daily_regime == "BULL":
                contradictions.append(f"taker=SELL_SURGE but regime={daily_regime}")
        dr = scan_data.get("direction_resolver", {})
        dr_dir = dr.get("direction", "NEUTRAL")
        if dr_dir == "LONG":
            bull_signals += 1
            if daily_regime in ("BEAR", "STRESS"):
                contradictions.append(f"consensus=LONG but regime={daily_regime}")
        elif dr_dir == "SHORT":
            bear_signals += 1
            if daily_regime == "BULL":
                contradictions.append(f"consensus=SHORT but regime={daily_regime}")
        cascade = scan_data.get("cascade", {})
        cascade_signal = cascade.get("combined_signal", "HOLD")
        if "LONG" in cascade_signal:
            bull_signals += 1
            if daily_regime in ("BEAR", "STRESS"):
                contradictions.append(f"cascade=LONG but regime={daily_regime}")
        elif "SHORT" in cascade_signal:
            bear_signals += 1
            if daily_regime == "BULL":
                contradictions.append(f"cascade=SHORT but regime={daily_regime}")
        return {"count": len(contradictions), "details": contradictions, "bull_signals": bull_signals, "bear_signals": bear_signals, "net_15m": bull_signals - bear_signals}

    def _apply_contradictions(self, regime, confidence, contradictions):
        count = contradictions.get("count", 0)
        net_15m = contradictions.get("net_15m", 0)
        if count == 0:
            return regime, confidence
        adjusted_conf = confidence * (1 - 0.1 * min(count, 3))
        if count >= 3 and abs(net_15m) >= 2:
            return "RANGING", max(0.5, adjusted_conf)
        if count >= 2:
            if regime == "BEAR":
                return "MILDLY_BEARISH", adjusted_conf
            elif regime == "BULL":
                return "MILDLY_BEARISH", adjusted_conf
        return regime, adjusted_conf

    def _apply_hysteresis(self, raw_regime, confidence, now):
        current = self.regime
        if raw_regime == current:
            self._consecutive_votes.append(raw_regime)
            return current
        self._consecutive_votes.append(raw_regime)
        if self._last_transition_ts:
            elapsed = now - self._last_transition_ts
            if elapsed < self.COOLDOWN_SECONDS:
                return current
        recent = list(self._consecutive_votes)[-self.HYSTERESIS_BARS:]
        hysteresis_met = len(recent) >= self.HYSTERESIS_BARS and all(r == raw_regime for r in recent)
        if raw_regime == "STRESS" and confidence > 0.8:
            self._last_transition_ts = now
            self._regime_start_ts = now
            return raw_regime
        if hysteresis_met:
            self._last_transition_ts = now
            self._regime_start_ts = now
            return raw_regime
        return current

    def _fallback_15m(self, now):
        if not self.cc or not hasattr(self.cc, "deriv_by_ts"):
            return "RANGING"
        deriv = self.cc.deriv_by_ts
        if not deriv:
            return "RANGING"
        sorted_ts = sorted(deriv.keys())
        recent = [deriv[ts] for ts in sorted_ts[-20:]]
        if len(recent) < 3:
            return "RANGING"
        latest = recent[-1]
        fr = latest.get("fr", 0)
        ls = latest.get("ls", 2.0)
        oi = latest.get("oi", 0)
        bull_score = 0
        bear_score = 0
        if fr > 0.00005: bull_score += 1.5
        elif fr < -0.00002: bear_score += 1.5
        if ls > 2.5: bear_score += 1.0
        elif ls < 1.5: bull_score += 1.0
        if len(recent) >= 2:
            prev_oi = recent[-2].get("oi", oi)
            if prev_oi > 0:
                oi_roc = (oi - prev_oi) / prev_oi * 100
                if oi_roc < -5: return "STRESS"
        if bull_score > bear_score + 1.5: return "BULL"
        elif bear_score > bull_score + 1.5: return "BEAR"
        elif bear_score > bull_score + 0.5: return "MILDLY_BEARISH"
        else: return "RANGING"

    def is_bearish(self): return self.regime in ("BEAR", "STRESS", "MILDLY_BEARISH") and self.confidence >= 0.5
    def is_ranging(self): return self.regime == "RANGING"
    def is_bullish(self): return self.regime == "BULL" and self.confidence >= 0.5
    def is_neutral_or_bearish(self): return self.regime in ("BEAR", "STRESS", "MILDLY_BEARISH")

    def get_sizing(self, direction="LONG"):
        base = self.sizing.copy()
        if direction == "SHORT":
            base["effective_size_mult"] = base["size_mult"] * base.get("short_size_mult", 1.0)
        elif direction == "LONG":
            base["effective_size_mult"] = base["size_mult"] * base.get("long_size_mult", 1.0)
        else:
            base["effective_size_mult"] = base["size_mult"]
        return base

    def get_regime_info(self):
        return {"regime": self.regime, "confidence": round(self.confidence, 3), "sizing": self.sizing, "regime_start": self._regime_start_ts, "last_transition": self._last_transition_ts, "consecutive_votes": list(self._consecutive_votes), "v4_available": self._v4_imported, "daily_data_cached": self._daily_candles is not None, "signals": self.signals}
