"""
Structure Agent — Reads market microstructure to detect edge.

Replaces lagging indicators with:
- Liquidation levels (where forced orders happen)
- Open interest delta (who's trapped)
- Funding rate (crowded positioning)
- Order flow imbalance (aggressive buyers vs sellers)
- Whale activity (smart money direction)

Architecture: Single agent that consolidates all structure signals
into a unified "structure score" with direction and conviction.
"""
import json, os, sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)


class StructureAgent:
    """
    Consolidates market structure signals into a unified assessment.
    Returns: direction, conviction, edge_type, confidence
    """

    def __init__(self):
        self.last_assessment = None
        self.history = []  # Last N assessments for trend detection

    def assess(self, scan_data, deriv_data=None, ob_data=None):
        """
        Main entry point: analyze all structure signals and return unified assessment.

        Args:
            scan_data: Latest scan JSON (from scanner)
            deriv_data: Derivatives history (funding, OI, ls_ratio)
            ob_data: Orderbook snapshots

        Returns:
            dict: {
                "direction": "LONG"|"SHORT"|"NEUTRAL",
                "conviction": 0.0-1.0,
                "edge_type": str,
                "components": dict,
                "timestamp": str,
                "regime_alignment": bool
            }
        """
        components = {}
        bull_score = 0.0
        bear_score = 0.0

        # === 1. LIQUIDATION CASCADE DETECTION ===
        liq = self._check_liquidity(scan_data)
        components["liquidity"] = liq
        if liq["signal"] == "CASCADE_LONG":
            bull_score += liq["strength"] * 2.0
        elif liq["signal"] == "CASCADE_SHORT":
            bear_score += liq["strength"] * 2.0

        # === 2. ORDERBOOK IMBALANCE ===
        ob = self._check_orderbook(scan_data, ob_data)
        components["orderbook"] = ob
        if ob["signal"] == "BID_HEAVY":
            bull_score += ob["strength"] * 1.5
        elif ob["signal"] == "ASK_HEAVY":
            bear_score += ob["strength"] * 1.5

        # === 3. FUNDING RATE (contrarian) ===
        fr = self._check_funding(deriv_data)
        components["funding"] = fr
        if fr["signal"] == "EXTREME_LONG_CROWDED":
            bear_score += fr["strength"] * 1.2  # Contrarian
        elif fr["signal"] == "EXTREME_SHORT_CROWDED":
            bull_score += fr["strength"] * 1.2

        # === 4. OPEN INTEREST DIVERGENCE ===
        oi = self._check_oi_divergence(deriv_data, scan_data)
        components["oi_divergence"] = oi
        if oi["signal"] == "TRAPPED_LONGS":
            bear_score += oi["strength"] * 1.8
        elif oi["signal"] == "TRAPPED_SHORTS":
            bull_score += oi["strength"] * 1.8

        # === 5. WHALE ACTIVITY ===
        whale = self._check_whale(scan_data)
        components["whale"] = whale
        if whale["signal"] == "WHALE_LONG":
            bull_score += whale["strength"] * 1.3
        elif whale["signal"] == "WHALE_SHORT":
            bear_score += whale["strength"] * 1.3

        # === 6. TAKER FLOW ===
        taker = self._check_taker(scan_data)
        components["taker"] = taker
        if taker["signal"] == "AGGRESSIVE_BUY":
            bull_score += taker["strength"] * 1.0
        elif taker["signal"] == "AGGRESSIVE_SELL":
            bear_score += taker["strength"] * 1.0

        # === 7. FORCED MOVEMENT DETECTION ===
        forced = self._check_forced_movement(deriv_data, scan_data)
        components["forced_movement"] = forced
        if forced["signal"] == "FORCED_UP":
            bull_score += forced["strength"] * 2.5
        elif forced["signal"] == "FORCED_DOWN":
            bear_score += forced["strength"] * 2.5

        # === CONSENSUS ===
        total = bull_score + bear_score
        if total == 0:
            direction = "NEUTRAL"
            conviction = 0.0
        elif bull_score > bear_score:
            direction = "LONG"
            conviction = min(0.95, bull_score / max(total, 1) * (total / 5.0))
        else:
            direction = "SHORT"
            conviction = min(0.95, bear_score / max(total, 1) * (total / 5.0))

        # Determine primary edge type
        edge_type = self._determine_edge_type(components)

        # Count how many components agree with direction
        agreeing = sum(1 for c in components.values()
                       if c.get("signal", "").endswith("_LONG") and direction == "LONG"
                       or c.get("signal", "").endswith("_SHORT") and direction == "SHORT"
                       or "CROWDED" in c.get("signal", "") and (
                           ("LONG" in c.get("signal", "") and direction == "SHORT") or
                           ("SHORT" in c.get("signal", "") and direction == "LONG")))

        # Boost conviction if multiple signals agree
        if agreeing >= 3:
            conviction = min(0.95, conviction * 1.3)
        elif agreeing >= 2:
            conviction = min(0.95, conviction * 1.15)

        result = {
            "direction": direction,
            "conviction": round(conviction, 3),
            "edge_type": edge_type,
            "components": components,
            "bull_score": round(bull_score, 2),
            "bear_score": round(bear_score, 2),
            "agreeing_signals": agreeing,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.last_assessment = result
        self.history.append(result)
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return result

    def _check_liquidity(self, scan_data):
        """Check for liquidation cascade signals."""
        cascade = scan_data.get("cascade", {})
        liq_data = scan_data.get("strategy_data", {}).get("liquidation_cascade", {})
        combined = cascade.get("combined_signal", "HOLD")

        if "STRONG_LONG" in combined:
            return {"signal": "CASCADE_LONG", "strength": 0.8, "detail": combined}
        elif "STRONG_SHORT" in combined:
            return {"signal": "CASCADE_SHORT", "strength": 0.8, "detail": combined}

        # Check raw liquidation data
        liq_signal = liq_data.get("direction", "")
        if liq_signal == "LONG":
            return {"signal": "CASCADE_LONG", "strength": 0.5, "detail": "liq_data"}
        elif liq_signal == "SHORT":
            return {"signal": "CASCADE_SHORT", "strength": 0.5, "detail": "liq_data"}

        return {"signal": "NEUTRAL", "strength": 0.0}

    def _check_orderbook(self, scan_data, ob_data):
        """Check orderbook imbalance."""
        obi = scan_data.get("strategy_data", {}).get("orderbook_imbalance", {})
        direction = obi.get("direction", "")
        obi_score = obi.get("obi_score", 0) or 0

        if direction == "LONG" and obi_score > 0.5:
            return {"signal": "BID_HEAVY", "strength": min(1.0, obi_score), "score": obi_score}
        elif direction == "SHORT" and obi_score < -0.5:
            return {"signal": "ASK_HEAVY", "strength": min(1.0, abs(obi_score)), "score": obi_score}

        return {"signal": "NEUTRAL", "strength": 0.0, "score": obi_score}

    def _check_funding(self, deriv_data):
        """Check funding rate for crowded positioning (contrarian)."""
        if not deriv_data:
            return {"signal": "NEUTRAL", "strength": 0.0}

        sorted_ts = sorted(deriv_data.keys())
        if not sorted_ts:
            return {"signal": "NEUTRAL", "strength": 0.0}

        latest = deriv_data[sorted_ts[-1]]
        fr = latest.get("fr", 0)

        # Extreme positive funding = longs paying shorts = crowded long = bearish
        if fr > 0.0005:
            return {"signal": "EXTREME_LONG_CROWDED", "strength": min(1.0, fr / 0.001), "fr": fr}
        elif fr > 0.0002:
            return {"signal": "LONG_CROWDED", "strength": min(0.6, fr / 0.0005), "fr": fr}
        elif fr < -0.0003:
            return {"signal": "EXTREME_SHORT_CROWDED", "strength": min(1.0, abs(fr) / 0.0005), "fr": fr}
        elif fr < -0.0001:
            return {"signal": "SHORT_CROWDED", "strength": min(0.6, abs(fr) / 0.0003), "fr": fr}

        return {"signal": "NEUTRAL", "strength": 0.0, "fr": fr}

    def _check_oi_divergence(self, deriv_data, scan_data):
        """Check OI divergence — price up + OI down = trapped shorts closing."""
        if not deriv_data:
            return {"signal": "NEUTRAL", "strength": 0.0}

        sorted_ts = sorted(deriv_data.keys())
        if len(sorted_ts) < 3:
            return {"signal": "NEUTRAL", "strength": 0.0}

        recent = [deriv_data[ts] for ts in sorted_ts[-5:]]
        oi_now = recent[-1].get("oi", 0)
        oi_prev = recent[0].get("oi", 0)
        price_now = scan_data.get("price", 0)

        if oi_prev == 0:
            return {"signal": "NEUTRAL", "strength": 0.0}

        oi_change_pct = (oi_now - oi_prev) / oi_prev * 100

        # Get price direction from scan data
        price_dir = scan_data.get("direction_resolver", {}).get("direction", "NEUTRAL")

        # Price up + OI down = trapped shorts being liquidated
        if price_dir == "LONG" and oi_change_pct < -3:
            return {"signal": "TRAPPED_SHORTS", "strength": min(1.0, abs(oi_change_pct) / 5), "oi_chg": oi_change_pct}
        # Price down + OI down = trapped longs being liquidated
        elif price_dir == "SHORT" and oi_change_pct < -3:
            return {"signal": "TRAPPED_LONGS", "strength": min(1.0, abs(oi_change_pct) / 5), "oi_chg": oi_change_pct}
        # OI surging + price moving = new positions being opened (trend continuation)
        elif oi_change_pct > 5:
            if price_dir == "LONG":
                return {"signal": "NEW_LONGS", "strength": min(0.5, oi_change_pct / 10), "oi_chg": oi_change_pct}
            elif price_dir == "SHORT":
                return {"signal": "NEW_SHORTS", "strength": min(0.5, oi_change_pct / 10), "oi_chg": oi_change_pct}

        return {"signal": "NEUTRAL", "strength": 0.0, "oi_chg": oi_change_pct}

    def _check_whale(self, scan_data):
        """Check whale wallet activity."""
        whale = scan_data.get("strategy_data", {}).get("whale_watch", {})
        direction = whale.get("direction", "")
        conviction = whale.get("conviction", 0) or 0

        if direction == "LONG" and conviction > 0.3:
            return {"signal": "WHALE_LONG", "strength": min(1.0, conviction), "detail": whale}
        elif direction == "SHORT" and conviction > 0.3:
            return {"signal": "WHALE_SHORT", "strength": min(1.0, conviction), "detail": whale}

        return {"signal": "NEUTRAL", "strength": 0.0}

    def _check_taker(self, scan_data):
        """Check taker buy/sell ratio."""
        taker = scan_data.get("taker_summary", {})
        regime = taker.get("regime", "")
        momentum = taker.get("momentum", 0) or 0

        if "BUYING" in regime.upper() or "SURGE" in regime.upper():
            return {"signal": "AGGRESSIVE_BUY", "strength": min(1.0, abs(momentum) / 2), "regime": regime}
        elif "SELLING" in regime.upper():
            return {"signal": "AGGRESSIVE_SELL", "strength": min(1.0, abs(momentum) / 2), "regime": regime}

        return {"signal": "NEUTRAL", "strength": 0.0, "regime": regime}

    def _check_forced_movement(self, deriv_data, scan_data):
        """Detect forced movement: OI divergence + funding squeeze + liquidation cascade."""
        if not deriv_data:
            return {"signal": "NEUTRAL", "strength": 0.0}

        sorted_ts = sorted(deriv_data.keys())
        if len(sorted_ts) < 5:
            return {"signal": "NEUTRAL", "strength": 0.0}

        recent = [deriv_data[ts] for ts in sorted_ts[-5:]]
        fr_now = recent[-1].get("fr", 0)
        oi_now = recent[-1].get("oi", 0)
        oi_prev = recent[0].get("oi", 0)
        ls = recent[-1].get("ls", 2.0)

        # Funding squeeze: both sides paying (high |fr| + high ls_ratio variance)
        fr_squeeze = abs(fr_now) > 0.0003

        # OI collapse: massive position closing
        oi_collapse = oi_prev > 0 and (oi_now - oi_prev) / oi_prev < -0.05

        # Extreme positioning
        extreme_ls = ls > 2.5 or ls < 1.5

        if fr_squeeze and oi_collapse:
            # Determine direction from price action
            price_dir = scan_data.get("direction_resolver", {}).get("direction", "NEUTRAL")
            if price_dir == "LONG":
                return {"signal": "FORCED_UP", "strength": 0.9, "detail": "fr_squeeze+oi_collapse"}
            elif price_dir == "SHORT":
                return {"signal": "FORCED_DOWN", "strength": 0.9, "detail": "fr_squeeze+oi_collapse"}

        if extreme_ls and fr_squeeze:
            if ls > 2.5:
                return {"signal": "FORCED_DOWN", "strength": 0.6, "detail": "extreme_long_crowded+fr"}
            else:
                return {"signal": "FORCED_UP", "strength": 0.6, "detail": "extreme_short_crowded+fr"}

        return {"signal": "NEUTRAL", "strength": 0.0}

    def _determine_edge_type(self, components):
        """Determine the primary edge type from component signals."""
        edges = []
        for name, comp in components.items():
            if comp.get("strength", 0) > 0.3:
                edges.append((name, comp["strength"]))

        if not edges:
            return "none"

        edges.sort(key=lambda x: x[1], reverse=True)
        return edges[0][0]

    def get_trend(self, window=5):
        """Check if structure signals are trending in one direction."""
        if len(self.history) < window:
            return "UNKNOWN", 0.0

        recent = self.history[-window:]
        longs = sum(1 for a in recent if a["direction"] == "LONG")
        shorts = sum(1 for a in recent if a["direction"] == "SHORT")

        if longs >= window * 0.7:
            return "TRENDING_LONG", longs / window
        elif shorts >= window * 0.7:
            return "TRENDING_SHORT", shorts / window
        else:
            return "MIXED", max(longs, shorts) / window
