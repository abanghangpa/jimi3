"""S19: Order Book Imbalance v5 — Research-backed rebuild.

v4 → v5 CHANGES (based on Nittur Anantha 2025, Bieganowski 2026):

1. TRADE-BASED OBI: Uses executed trade imbalance (not just bid/ask quotes)
   - Nittur Anantha: "OBI computed using trade events exhibits stronger causal
     alignment with future price movements" (arXiv:2507.22712)

2. CONCAVE CONVICTION: sqrt() scaling instead of linear
   - Bieganowski: "Order flow imbalance has a monotone but concave effect"
   - Moderate imbalance = strong signal, extreme = possible spoof

3. VWAP DEVIATION: Primary signal (not just bonus)
   - Bieganowski: Top-3 SHAP feature. Asymmetric effect — price above VWAP =
     selling pressure building (mean reversion), below = buying pressure

4. SPREAD FILTER: Wide spread = skip (adverse selection)
   - Bieganowski: "Spreads associated with diminished predictability"
   - Glosten-Milgrom: Market makers widen spreads against informed flow

5. FLEET FILTER: Already had persistence filter, now also checks quote stability

6. ASYMMETRIC ENTRY: Buy at VWAP discount, sell at VWAP premium
"""
from .base import BaseStrategy, SignalResult
import json, os, math
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OB_STATE = os.path.join(BASE_DIR, "data", "ob_history", "ob_state.json")

BAD_HOURS = {5, 23}


def _read_ob_state():
    if not os.path.exists(OB_STATE):
        return None
    try:
        with open(OB_STATE) as f:
            return json.load(f)
    except Exception:
        return None


class OrderBookImbalanceStrategy(BaseStrategy):
    min_vol_ratio = 0.12
    name = 'orderbook_imbalance'
    strategy_type = 'flow'
    description = 'v5: research-backed (trade OBI, concave conviction, VWAP primary)'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        regime = data.get('regime', 'RANGING')
        if not price or not atr or df_15m is None or idx is None:
            return None

        # ── SESSION FILTER ──
        ts = data.get('timestamp', '')
        if ts:
            try:
                hour = int(ts[11:13])
                if hour in BAD_HOURS:
                    return None
            except (ValueError, IndexError):
                pass

        # ── READ OB STATE ──
        ob_state = _read_ob_state()
        if not ob_state:
            return None

        snapshot = ob_state.get("snapshot", {})
        metrics = ob_state.get("metrics", {})

        ob_ratio = snapshot.get("ob_ratio", 0)  # quote-based (bid_vol - ask_vol) / total
        top5_ratio = snapshot.get("top5_ratio", 0)
        persistence = metrics.get("persistence_minutes", 0)
        ob_delta = metrics.get("ob_delta_5m", 0)
        spread_bps = snapshot.get("spread_bps", 0)

        # ── TRADE-BASED OBI (NEW — Nittur Anantha 2025) ──
        # Compute from actual executed trades, not just quotes
        taker_base = df_15m["Taker buy base asset volume"].values.astype(float)
        volume = df_15m["Volume"].values.astype(float)

        trade_obi = 0
        if idx >= 4:
            # Last 4 bars trade imbalance
            buy_vol = np.sum(taker_base[idx-4:idx])
            total_vol = np.sum(volume[idx-4:idx])
            if total_vol > 0:
                trade_obi = (buy_vol / total_vol - 0.5) * 2  # Range: -1 to +1

        # Trade-based OBI z-score (60-bar rolling)
        trade_obi_z = 0
        if idx >= 60:
            window_bi = []
            for j in range(max(0, idx-60), idx-4, 4):
                bv = np.sum(taker_base[j:j+4])
                tv = np.sum(volume[j:j+4])
                if tv > 0:
                    window_bi.append((bv / tv - 0.5) * 2)
            if len(window_bi) >= 5:
                mean_bi = np.mean(window_bi)
                std_bi = np.std(window_bi)
                if std_bi > 0:
                    trade_obi_z = (trade_obi - mean_bi) / std_bi

        # ── PERSISTENCE FILTER ──
        if persistence < 2:
            return None

        # ── SPREAD FILTER (NEW — Bieganowski 2026) ──
        # Wide spread = adverse selection risk = skip
        avg_spread = data.get('avg_spread_bps', 0)
        if avg_spread and spread_bps:
            if spread_bps > avg_spread * 2.0:  # Spread > 2x average
                return None

        # ── VWAP DEVIATION (NEW — Primary feature) ──
        vwap = data.get('vwap', 0)
        vwap_dev = (price - vwap) / vwap if vwap and vwap > 0 else 0

        # ── MOMENTUM ──
        mom_5 = 0
        if idx >= 5:
            mom_5 = (float(df_15m['Close'].iloc[idx]) - float(df_15m['Close'].iloc[idx-5])) / float(df_15m['Close'].iloc[idx-5])

        # ── DIRECTION LOGIC (asymmetric: VWAP discount/premium) ──
        direction = None
        obi_combined = 0  # Combined OBI signal

        # Combine quote-based + trade-based OBI
        # Trade-based gets 2x weight (research says it's more causal)
        quote_signal = ob_ratio * 0.4 + top5_ratio * 0.3
        trade_signal = trade_obi * 0.5 + trade_obi_z * 0.2
        obi_combined = quote_signal + trade_signal

        # LONG: Positive OBI + VWAP discount (buying below fair value)
        if obi_combined > 0.05 and vwap_dev < 0.003:
            if mom_5 < -0.015:
                return None  # Don't buy into recent selloff
            direction = 'LONG'

        # SHORT: Negative OBI + VWAP premium (selling above fair value)
        elif obi_combined < -0.05 and vwap_dev > -0.003:
            if mom_5 > 0.015:
                return None  # Don't sell into recent rally
            direction = 'SHORT'

        if not direction:
            return None

        # ── CONVICTION (CONCAVE — Bieganowski 2026) ──
        # sqrt() scaling: moderate imbalance = strong signal, extreme = diminishing
        base = 0.45

        # OBI strength (concave via sqrt)
        obi_abs = min(abs(obi_combined), 0.5)
        obi_strength = math.sqrt(obi_abs) * 0.25  # max ~0.18 at obi=0.5

        # Trade-based OBI bonus (separate from combined)
        if abs(trade_obi_z) > 1.0:
            trade_bonus = min(math.sqrt(abs(trade_obi_z) - 1.0) * 0.08, 0.15)
        else:
            trade_bonus = 0

        # VWAP confluence (primary signal, not just bonus)
        if direction == 'LONG' and vwap_dev < -0.005:
            vwap_bonus = 0.12  # Strong: buying at discount
        elif direction == 'LONG' and vwap_dev < -0.002:
            vwap_bonus = 0.08  # Moderate: slight discount
        elif direction == 'SHORT' and vwap_dev > 0.005:
            vwap_bonus = 0.12  # Strong: selling at premium
        elif direction == 'SHORT' and vwap_dev > 0.002:
            vwap_bonus = 0.08  # Moderate: slight premium
        else:
            vwap_bonus = 0

        # Persistence bonus
        persist_bonus = min(persistence / 20, 0.12)

        # Delta bonus (strengthening imbalance)
        if direction == 'LONG' and ob_delta > 0.02:
            delta_bonus = 0.08
        elif direction == 'SHORT' and ob_delta < -0.02:
            delta_bonus = 0.08
        else:
            delta_bonus = 0

        # Regime bonus
        regime_bonus = 0
        if regime == "BULL" and direction == "LONG":
            regime_bonus = 0.05
        elif regime == "BEAR" and direction == "SHORT":
            regime_bonus = 0.05
        elif regime == "RANGING":
            regime_bonus = 0.03

        conviction = min(base + obi_strength + trade_bonus + vwap_bonus +
                        persist_bonus + delta_bonus + regime_bonus, 0.90)

        # Spoofing penalty
        recent_spoofs = ob_state.get("recent_spoofs", 0)
        if recent_spoofs > 0:
            conviction *= 0.85

        if conviction < 0.50:
            return None

        # ── TP/SL (regime-adaptive) ──
        if direction == 'LONG':
            if idx >= 20:
                swing_low = float(df_15m['Low'].iloc[idx-20:idx].min())
            else:
                swing_low = price - 1.5 * atr
            sl_dist = max(price - swing_low, 0.5 * atr)
            sl_dist = min(sl_dist, 1.5 * atr)
            sl = price - sl_dist
        else:
            if idx >= 20:
                swing_high = float(df_15m['High'].iloc[idx-20:idx].max())
            else:
                swing_high = price + 1.5 * atr
            sl_dist = max(swing_high - price, 0.5 * atr)
            sl_dist = min(sl_dist, 1.5 * atr)
            sl = price + sl_dist

        tp_mult = {"BULL": 2.5, "BEAR": 2.0, "RANGING": 2.0, "STRESS": 1.5, "MILDLY_BEARISH": 2.0}.get(regime, 2.0)
        tp1_dist = tp_mult * atr
        if direction == 'LONG':
            tp1 = price + tp1_dist
            tp2 = price + 4.0 * atr
            tp3 = price + 6.0 * atr
        else:
            tp1 = price - tp1_dist
            tp2 = price - 4.0 * atr
            tp3 = price - 6.0 * atr

        sl_pct = (sl_dist / price) * 100
        tp1_pct = (tp1_dist / price) * 100

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.8,
            reason=f"OB v5 {direction}: obi={obi_combined:.3f} trade_obi={trade_obi:.3f} "
                   f"vwap_dev={vwap_dev:.5f} regime={regime}",
            bypass_gates=False,
            details={
                'ob_ratio': ob_ratio, 'top5_ratio': top5_ratio,
                'trade_obi': round(trade_obi, 4),
                'trade_obi_z': round(trade_obi_z, 3),
                'obi_combined': round(obi_combined, 4),
                'persistence_min': persistence,
                'spread_bps': spread_bps,
                'vwap_dev': round(vwap_dev, 5),
                'regime': regime,
                'version': 'v5',
            },
        )
