"""S06: Liquidity Grab v2 — uses OB collector + df_15m for S/R detection.

v1 → v2 CHANGES:
1. Uses OB collector data (persistence, spoofing, wall delta) instead of pipeline
2. Independent S/R level detection from df_15m (no pipeline dependency)
3. Persistence check: wall must be visible for 3+ snapshots
4. Spoofing penalty: reduces conviction if recent spoof detected
5. Wall delta: stronger signal if imbalance is increasing
"""
from .base import BaseStrategy, SignalResult
import numpy as np
import json, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OB_STATE = os.path.join(BASE_DIR, "data", "ob_history", "ob_state.json")

GOOD_HOURS = {0, 1, 7, 8, 9, 10, 12, 15, 16, 21}
BAD_HOURS = {4, 5, 6, 19, 20, 22, 23}


def _read_ob_state():
    """Read latest OB state from collector."""
    if not os.path.exists(OB_STATE):
        return None
    try:
        with open(OB_STATE) as f:
            return json.load(f)
    except Exception:
        return None


def _find_sr_levels(df_15m, idx, lookback=96):
    """
    Find support/resistance levels from swing points in df_15m.
    Returns list of levels with type and strength.
    """
    if idx < lookback:
        return []

    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)

    levels = []

    # Find swing highs (resistance)
    for i in range(3, min(lookback - 3, idx)):
        bar_idx = idx - i
        if bar_idx < 3:
            continue
        # Swing high: higher than 3 bars on each side
        if (highs[bar_idx] > highs[bar_idx-1] and
            highs[bar_idx] > highs[bar_idx-2] and
            highs[bar_idx] > highs[bar_idx-3] and
            highs[bar_idx] > highs[bar_idx+1] and
            highs[bar_idx] > highs[bar_idx+2] and
            highs[bar_idx] > highs[bar_idx+3]):
            # Count touches (price near this level in nearby bars)
            level_price = highs[bar_idx]
            touches = 0
            for j in range(max(0, bar_idx-10), min(len(highs), bar_idx+10)):
                if abs(highs[j] - level_price) / level_price < 0.002:
                    touches += 1
            levels.append({
                "price": level_price,
                "type": "resistance",
                "touches": touches,
                "bars_ago": idx - bar_idx,
                "strength": touches * (1 + np.log1p(volumes[bar_idx])),
            })

    # Find swing lows (support)
    for i in range(3, min(lookback - 3, idx)):
        bar_idx = idx - i
        if bar_idx < 3:
            continue
        if (lows[bar_idx] < lows[bar_idx-1] and
            lows[bar_idx] < lows[bar_idx-2] and
            lows[bar_idx] < lows[bar_idx-3] and
            lows[bar_idx] < lows[bar_idx+1] and
            lows[bar_idx] < lows[bar_idx+2] and
            lows[bar_idx] < lows[bar_idx+3]):
            level_price = lows[bar_idx]
            touches = 0
            for j in range(max(0, bar_idx-10), min(len(lows), bar_idx+10)):
                if abs(lows[j] - level_price) / level_price < 0.002:
                    touches += 1
            levels.append({
                "price": level_price,
                "type": "support",
                "touches": touches,
                "bars_ago": idx - bar_idx,
                "strength": touches * (1 + np.log1p(volumes[bar_idx])),
            })

    # Sort by strength, keep top 10
    levels.sort(key=lambda x: x["strength"], reverse=True)
    return levels[:10]


class LiquidityGrabStrategy(BaseStrategy):
    name = 'liquidity_grab'
    strategy_type = 'structure'
    description = 'v2: OB collector + independent S/R + persistence + spoofing'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
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

        # ── READ OB STATE (new) ──
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

        # Need some OB data
        if abs(ob_ratio) < 0.05 and abs(top5_ratio) < 0.05:
            return None

        # ── FIND S/R LEVELS (new — independent of pipeline) ──
        sr_levels = _find_sr_levels(df_15m, idx, lookback=96)

        if not sr_levels:
            return None

        # ── CHECK IF PRICE IS NEAR A LEVEL ──
        direction = None
        best_level = None
        proximity = 0

        for level in sr_levels:
            dist = abs(price - level["price"]) / atr
            if dist < 1.0:  # Within 1 ATR of level
                if level["type"] == "support" and price > level["price"]:
                    # Price above support — potential bounce
                    direction = 'LONG'
                    best_level = level
                    proximity = 1.0 - dist
                    break
                elif level["type"] == "resistance" and price < level["price"]:
                    # Price below resistance — potential rejection
                    direction = 'SHORT'
                    best_level = level
                    proximity = 1.0 - dist
                    break

        if not direction or not best_level:
            return None

        # ── OB CONFIRMATION ──
        # LONG: need bid-heavy OB (buyers at support)
        # SHORT: need ask-heavy OB (sellers at resistance)
        ob_confirm = 0
        if direction == 'LONG' and ob_ratio > 0.05:
            ob_confirm = 0.15
        elif direction == 'SHORT' and ob_ratio < -0.05:
            ob_confirm = 0.15

        # Top5 confirmation (stronger signal)
        top5_confirm = 0
        if direction == 'LONG' and top5_ratio > 0.10:
            top5_confirm = 0.10
        elif direction == 'SHORT' and top5_ratio < -0.10:
            top5_confirm = 0.10

        # ── PERSISTENCE BONUS (new) ──
        persist_bonus = 0
        if persistence >= 3:
            persist_bonus = min(persistence / 20, 0.10)  # max at 20 minutes

        # ── WALL DELTA BONUS (new) ──
        delta_bonus = 0
        if direction == 'LONG' and ob_delta > 0.03:
            delta_bonus = 0.10  # bid walls increasing
        elif direction == 'SHORT' and ob_delta < -0.03:
            delta_bonus = 0.10  # ask walls increasing

        # ── LEVEL STRENGTH BONUS ──
        level_bonus = min(best_level["strength"] / 20, 0.15)

        # ── CONVICTION ──
        base = 0.35
        conviction = min(base + proximity * 0.15 + ob_confirm + top5_confirm +
                        persist_bonus + delta_bonus + level_bonus, 0.85)

        # ── SPOOFING PENALTY (new) ──
        if recent_spoofs > 0:
            conviction *= 0.85  # 15% penalty

        if conviction < 0.45:
            return None

        # ── TP/SL ──
        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=f"Liq grab v2 {direction}: {best_level['type']} ${best_level['price']:.2f} "
                   f"touches={best_level['touches']} persist={persistence}m ob={ob_ratio:.3f}",
            bypass_gates=False,
            details={
                'level_price': best_level['price'], 'level_type': best_level['type'],
                'touches': best_level['touches'], 'bars_ago': best_level['bars_ago'],
                'ob_ratio': ob_ratio, 'top5_ratio': top5_ratio,
                'persistence': persistence, 'ob_delta': ob_delta,
                'recent_spoofs': recent_spoofs, 'proximity': proximity,
                'version': 'v2',
            },
        )
