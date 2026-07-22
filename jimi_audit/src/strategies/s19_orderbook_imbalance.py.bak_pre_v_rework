"""S19: Order Book Imbalance v3 — with persistence, delta, spoofing, and SHORT support.

v2 → v3 CHANGES:
1. SHORT direction added with separate filters (below EMA200 + seller-dominated + high vol)
2. OB persistence: requires imbalance held for 3+ minutes (not just a snapshot)
3. Wall delta: tracks if imbalance is strengthening or weakening
4. Spoofing penalty: reduces conviction if recent spoof detected
5. Reads live OB data from data/ob_history/ob_state.json

DATA SOURCE: ob_collector.py (systemd timer, every 60s)
"""
from .base import BaseStrategy, SignalResult
import json, os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OB_STATE = os.path.join(BASE_DIR, "data", "ob_history", "ob_state.json")

# Best/worst hours from analysis (UTC)
GOOD_HOURS = {0, 1, 7, 10, 12, 15, 21}
BAD_HOURS = {5, 11, 13, 14, 23}


def _read_ob_state():
    """Read latest OB state from collector."""
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
    description = 'v3: persistence + delta + spoofing + SHORT support'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        ema_200 = data.get('ema_200', 0)
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

        # — BEAR REGIME FILTER (skip bear trends - S/R breaks through) —
        ema200 = data.get("ema200", 0)
        if ema200:
            try:
                dist_from_ema = (price - ema200) / ema200 * 100
                if dist_from_ema < -2.0:
                    return None  # bear trend - S/R doesn't hold
            except (ValueError, IndexError):
                pass

        # ── READ OB STATE ──
        ob_state = _read_ob_state()
        if not ob_state:
            return None

        snapshot = ob_state.get("snapshot", {})
        metrics = ob_state.get("metrics", {})
        recent_spoofs = ob_state.get("recent_spoofs", 0)

        ob_ratio = snapshot.get("ob_ratio", 0)
        top5_ratio = snapshot.get("top5_ratio", 0)
        persistence = metrics.get("persistence_minutes", 0)
        ob_delta = metrics.get("ob_delta_5m", 0)
        top5_delta = metrics.get("top5_delta_5m", 0)
        trend = metrics.get("trend", "NEUTRAL")

        # ── PERSISTENCE FILTER (new) ──
        # Require imbalance held for at least 3 minutes
        if persistence < 3:
            return None

        # ── VOLUME FILTER ──
        vol_ratio = data.get('vol_ratio', 1.0) or 1.0

        # ── MOMENTUM FILTER ──
        mom_5 = 0
        if idx >= 5:
            mom_5 = (float(df_15m['Close'].iloc[idx]) - float(df_15m['Close'].iloc[idx-5])) / float(df_15m['Close'].iloc[idx-5])

        # ── DIRECTION LOGIC ──
        direction = None
        direction_filters = {}

        # LONG: bid-heavy OB + conditions
        if ob_ratio > 0.10 and top5_ratio > 0.05:
            # Standard LONG filters
            if 1.0 <= vol_ratio < 1.5:
                return None  # Dead zone
            if ema_200 and ema_200 > 0:
                dist_ema = (price - ema_200) / ema_200
                if dist_ema < -0.01:
                    return None  # Below EMA200 by >1%
            if mom_5 < -0.01:
                return None  # Recent selloff

            direction = 'LONG'
            direction_filters = {
                'ob_ratio': ob_ratio,
                'top5_ratio': top5_ratio,
                'vol_ratio': vol_ratio,
                'dist_ema': dist_ema if ema_200 else 0,
            }

        # SHORT: ask-heavy OB + different conditions (stricter)
        elif ob_ratio < -0.15 and top5_ratio < -0.08:
            # SHORT needs stronger signal (42% WR baseline, need more filters)
            if vol_ratio < 1.5:
                return None  # Need high volume for SHORT
            if ema_200 and ema_200 > 0:
                dist_ema = (price - ema_200) / ema_200
                if dist_ema > -0.005:
                    return None  # Must be below EMA200 (or very close)
            if mom_5 > 0.005:
                return None  # Don't SHORT if price rising

            # Taker flow confirmation for SHORT
            taker = data.get('raw_taker_ratio', 0.5)
            if taker > 0.45:
                return None  # Need seller-dominated taker flow

            direction = 'SHORT'
            direction_filters = {
                'ob_ratio': ob_ratio,
                'top5_ratio': top5_ratio,
                'vol_ratio': vol_ratio,
                'dist_ema': dist_ema if ema_200 else 0,
                'taker': taker,
            }

        if not direction:
            return None

        # ── CONVICTION ──
        base = 0.45

        # OB strength bonus
        if direction == 'LONG':
            ob_strength = min(abs(ob_ratio) / 0.3, 0.20)
            top5_strength = min(abs(top5_ratio) / 0.2, 0.10)
        else:
            ob_strength = min(abs(ob_ratio) / 0.4, 0.20)  # Higher bar for SHORT
            top5_strength = min(abs(top5_ratio) / 0.3, 0.10)

        # Persistence bonus (longer = more real)
        persist_bonus = min(persistence / 30, 0.15)  # max at 30 minutes

        # Delta bonus (strengthening imbalance)
        if direction == 'LONG' and ob_delta > 0.03:
            delta_bonus = 0.10
        elif direction == 'SHORT' and ob_delta < -0.03:
            delta_bonus = 0.10
        else:
            delta_bonus = 0

        # EMA bonus
        ema_bonus = 0
        if ema_200 and ema_200 > 0:
            dist_ema = (price - ema_200) / ema_200
            if direction == 'LONG' and 0.01 < dist_ema < 0.03:
                ema_bonus = 0.10
            elif direction == 'SHORT' and -0.03 < dist_ema < -0.01:
                ema_bonus = 0.05

        # Momentum bonus
        mom_bonus = 0
        if direction == 'LONG' and 0 < mom_5 < 0.01:
            mom_bonus = 0.05

        conviction = min(base + ob_strength + top5_strength + persist_bonus + delta_bonus + ema_bonus + mom_bonus, 0.90)

        # ── SPOOFING PENALTY (new) ──
        if recent_spoofs > 0:
            conviction *= 0.85  # 15% penalty if spoof detected recently

        if conviction < 0.50:
            return None

        # ── TP/SL ──
        if direction == 'LONG':
            # Structure-based SL (recent swing low)
            if idx >= 20:
                swing_low = float(df_15m['Low'].iloc[idx-20:idx].min())
            else:
                swing_low = price - 1.5 * atr
            sl_dist = price - swing_low
            if sl_dist <= 0:
                sl_dist = 1.0 * atr
            if sl_dist > 1.5 * atr:
                sl_dist = 1.5 * atr
            sl = price - sl_dist
        else:
            # SHORT: structure-based SL (recent swing high)
            if idx >= 20:
                swing_high = float(df_15m['High'].iloc[idx-20:idx].max())
            else:
                swing_high = price + 1.5 * atr
            sl_dist = swing_high - price
            if sl_dist <= 0:
                sl_dist = 1.0 * atr
            if sl_dist > 1.5 * atr:
                sl_dist = 1.5 * atr
            sl = price + sl_dist

        tp1_dist = 2.5 * atr
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
            reason=f"OB v3 {direction}: ratio={ob_ratio:.3f} top5={top5_ratio:.3f} "
                   f"persist={persistence}m delta={ob_delta:.4f} trend={trend}",
            bypass_gates=False,
            details={
                'ob_ratio': ob_ratio, 'top5_ratio': top5_ratio,
                'persistence_min': persistence, 'ob_delta_5m': ob_delta,
                'top5_delta_5m': top5_delta, 'trend': trend,
                'recent_spoofs': recent_spoofs,
                'vol_ratio': vol_ratio,
                'version': 'v3',
            },
        )
