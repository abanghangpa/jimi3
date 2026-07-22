"""S20: Liquidation Cascade v3 — real Bybit data + enhanced OI fallback + price-level awareness.

v2 → v3 CHANGES:
1. Lowered liq thresholds: count >= 1, vol > 2 ETH (Bybit liqs are sparse)
2. Enhanced OI fallback: uses derivatives OI + L/S + price structure
3. Price-level awareness: liqs near support/resistance are more significant
4. Added "OI shock" detection: sudden OI drop + price move = cascade in progress
5. Reads OB state for price-level context
"""
from .base import BaseStrategy, SignalResult
import json, os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIQ_LOG = os.path.join(BASE_DIR, "data", "forced_movement", "liquidation_events.jsonl")
OB_STATE = os.path.join(BASE_DIR, "data", "ob_history", "ob_state.json")


def _read_real_liqs(minutes=30):
    """Read real liquidation events from Bybit stream. Extended to 30 min."""
    if not os.path.exists(LIQ_LOG):
        return 0, 0, 0, []
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    cutoff_ms = int(cutoff.timestamp() * 1000)
    long_vol = 0
    short_vol = 0
    count = 0
    events = []
    try:
        with open(LIQ_LOG) as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    if e.get("ts", 0) >= cutoff_ms:
                        count += 1
                        qty = e.get("qty", 0)
                        if e.get("side") == "Sell":
                            long_vol += qty
                        else:
                            short_vol += qty
                        events.append(e)
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception:
        pass
    return count, long_vol, short_vol, events


def _check_oi_shock(data, df_15m, idx):
    """
    Detect OI shock: sudden OI drop + price move = cascade in progress.
    Uses derivatives data (OI ROC) and price momentum.
    """
    deriv = data.get('derivatives', {})
    oi_roc = deriv.get('oi_roc_1h', 0)
    ls_ratio = deriv.get('ls_ratio', 1.0)

    if abs(oi_roc) < 0.01:
        return None  # No significant OI change

    # Price momentum
    closes = df_15m['Close'].values.astype(float)
    if idx < 5:
        return None
    price_change = (closes[idx] - closes[idx-5]) / closes[idx-5]

    # OI dropping + price dropping = long liquidation cascade
    if oi_roc < -0.01 and price_change < -0.005:
        direction = 'SHORT'
        strength = min(abs(oi_roc) * 20, 0.8)
        return {
            "direction": direction, "strength": strength,
            "detail": f"OI shock: OI ROC={oi_roc:.4f}, price_change={price_change:.4f} (long cascade)"
        }

    # OI dropping + price rising = short liquidation cascade
    if oi_roc < -0.01 and price_change > 0.005:
        direction = 'LONG'
        strength = min(abs(oi_roc) * 20, 0.8)
        return {
            "direction": direction, "strength": strength,
            "detail": f"OI shock: OI ROC={oi_roc:.4f}, price_change={price_change:.4f} (short cascade)"
        }

    # Extreme L/S + OI change = potential cascade
    if abs(oi_roc) > 0.015:
        if ls_ratio > 2.0 and oi_roc < -0.015:
            return {"direction": "SHORT", "strength": 0.6,
                    "detail": f"OI shock: crowded long + OI dropping ({oi_roc:.4f})"}
        elif ls_ratio < 0.5 and oi_roc < -0.015:
            return {"direction": "LONG", "strength": 0.6,
                    "detail": f"OI shock: crowded short + OI dropping ({oi_roc:.4f})"}

    return None


def _check_price_levels(price, atr, df_15m, idx):
    """
    Check if price is near key support/resistance levels.
    Liquidations at key levels are more significant.
    """
    if idx < 20:
        return 0, "mid-range"

    closes = df_15m['Close'].values.astype(float)
    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)

    # Recent swing levels
    swing_high = float(np.max(highs[max(0, idx-48):idx]))
    swing_low = float(np.min(lows[max(0, idx-48):idx]))

    # Distance to levels
    dist_to_high = abs(price - swing_high) / atr
    dist_to_low = abs(price - swing_low) / atr

    # Near support (within 0.5 ATR)
    if dist_to_low < 0.5:
        return 0.15, "near_support"
    # Near resistance (within 0.5 ATR)
    elif dist_to_high < 0.5:
        return 0.10, "near_resistance"
    # Near round number (within 0.5%)
    elif price % 50 < 25:
        round_dist = (price % 50) / 50
        if round_dist < 0.02:
            return 0.05, "near_round"
    # In the middle of range
    else:
        range_pct = (price - swing_low) / (swing_high - swing_low) if swing_high > swing_low else 0.5
        if 0.3 < range_pct < 0.7:
            return -0.05, "mid-range"  # Slight penalty for mid-range liqs

    return 0, "normal"


class LiquidationCascadeStrategy(BaseStrategy):
    name = 'liquidation_cascade'
    strategy_type = 'event'
    description = 'v3: lower thresholds + OI shock + price-level awareness'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr:
            return None

        signals = []

        # ── SOURCE 1: Real liquidation data (Bybit stream) ──
        liq_count, long_vol, short_vol, events = _read_real_liqs(minutes=30)
        total_vol = long_vol + short_vol

        # Lowered thresholds: count >= 1, vol > 2 ETH
        if liq_count >= 1 and total_vol > 2:
            if long_vol > short_vol * 1.5:
                direction = 'SHORT'
                strength = min(long_vol / 50, 1.0)
                signals.append({
                    "name": "real_liq",
                    "direction": direction,
                    "strength": strength,
                    "detail": f"Long liq: {long_vol:.1f} ETH ({liq_count} events, 30m)"
                })
            elif short_vol > long_vol * 1.5:
                direction = 'LONG'
                strength = min(short_vol / 50, 1.0)
                signals.append({
                    "name": "real_liq",
                    "direction": direction,
                    "strength": strength,
                    "detail": f"Short liq: {short_vol:.1f} ETH ({liq_count} events, 30m)"
                })

        # ── SOURCE 2: OI shock detection (new) ──
        oi_shock = _check_oi_shock(data, df_15m, idx)
        if oi_shock:
            signals.append({
                "name": "oi_shock",
                "direction": oi_shock["direction"],
                "strength": oi_shock["strength"],
                "detail": oi_shock["detail"]
            })

        # ── SOURCE 3: Extreme OI drop + L/S (enhanced fallback) ──
        if not signals:
            deriv = data.get('derivatives', {})
            oi_roc = deriv.get('oi_roc_1h', 0)
            ls_ratio = deriv.get('ls_ratio', 1.0)

            # More sensitive than v2: oi_roc > 0.01 (was 0.02)
            if abs(oi_roc) > 0.01:
                if oi_roc < -0.01 and ls_ratio > 1.8:
                    signals.append({
                        "name": "oi_estimate",
                        "direction": "SHORT",
                        "strength": min(abs(oi_roc) * 15, 0.7),
                        "detail": f"OI estimate: ROC={oi_roc:.4f} LS={ls_ratio:.2f} (longs closing)"
                    })
                elif oi_roc < -0.01 and ls_ratio < 0.6:
                    signals.append({
                        "name": "oi_estimate",
                        "direction": "LONG",
                        "strength": min(abs(oi_roc) * 15, 0.7),
                        "detail": f"OI estimate: ROC={oi_roc:.4f} LS={ls_ratio:.2f} (shorts closing)"
                    })

        if not signals:
            return None

        # ── PICK STRONGEST SIGNAL ──
        best = max(signals, key=lambda x: x["strength"])
        direction = best["direction"]
        strength = best["strength"]

        # ── PRICE-LEVEL BONUS (new) ──
        level_bonus, level_type = _check_price_levels(price, atr, df_15m, idx)

        conviction = min(0.45 + strength * 0.35 + level_bonus, 0.85)
        if conviction < 0.45:
            return None

        # Cascade moves are fast — tighter TP, wider SL
        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.5)

        source_names = [s["name"] for s in signals]
        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=f"Liq v3 ({'+'.join(source_names)}) {direction}: {best['detail']} [{level_type}]",
            bypass_gates=True,
            details={
                'sources': source_names, 'best_source': best["name"],
                'liq_count': liq_count, 'long_vol': long_vol, 'short_vol': short_vol,
                'strength': strength, 'level_type': level_type, 'level_bonus': level_bonus,
                'version': 'v3',
            },
        )
