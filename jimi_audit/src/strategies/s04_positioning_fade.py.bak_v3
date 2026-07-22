"""S04: Positioning Fade v3 — fade extreme derivatives positioning.

v2 → v3 CHANGES:
1. Rolling L/S mean/std from derivatives history (not hardcoded 2.15/0.3)
2. Time-in-position filter — need extreme for 30+ minutes (not just a snapshot)
3. bypass_gates=False — now that regime filter is added, validate properly
4. Reads derivatives_collected.csv for rolling statistics
"""
from .base import BaseStrategy, SignalResult
import os, json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DERIV_CSV = os.path.join(BASE_DIR, "data", "derivatives_history", "derivatives_collected.csv")
PF_STATE = os.path.join(BASE_DIR, "data", "forced_movement", "pf_state.json")

# Regimes where fading works (crowd is wrong)
GOOD_REGIMES = {'RANGING', 'CHOP_MILD', 'CHOP_BULL', 'CHOP_BEAR', 'NEUTRAL'}
# Regimes where fading fails (crowd is right — trend is your friend)
BAD_REGIMES = {'TRENDING', 'CRISIS', 'COMPRESSING'}

# How many L/S samples to compute rolling stats
ROLLING_WINDOW = 200  # ~50 hours at 15-min intervals


def _read_rolling_ls_stats():
    """Compute rolling mean and std of L/S ratio from derivatives history."""
    if not os.path.exists(DERIV_CSV):
        return None
    try:
        ratios = []
        with open(DERIV_CSV) as f:
            header = f.readline().strip().split(",")
            ls_idx = header.index("ls_ratio") if "ls_ratio" in header else -1
            if ls_idx < 0:
                return None
            for line in f:
                parts = line.strip().split(",")
                if len(parts) > ls_idx:
                    try:
                        r = float(parts[ls_idx])
                        if r > 0:
                            ratios.append(r)
                    except ValueError:
                        continue
        if len(ratios) < 30:
            return None
        window = ratios[-ROLLING_WINDOW:]
        return {
            "mean": float(np.mean(window)),
            "std": float(np.std(window)),
            "count": len(window),
            "min": float(np.min(window)),
            "max": float(np.max(window)),
        }
    except Exception:
        return None


def _read_ls_duration(ls_ratio, threshold=2.0):
    """
    Track how long L/S ratio has been above threshold.
    Reads derivatives history and counts consecutive entries above threshold.
    """
    if not os.path.exists(DERIV_CSV):
        return 0
    try:
        ratios = []
        with open(DERIV_CSV) as f:
            header = f.readline().strip().split(",")
            ls_idx = header.index("ls_ratio") if "ls_ratio" in header else -1
            if ls_idx < 0:
                return 0
            ts_idx = header.index("timestamp") if "timestamp" in header else -1
            for line in f:
                parts = line.strip().split(",")
                if len(parts) > ls_idx:
                    try:
                        r = float(parts[ls_idx])
                        ratios.append(r)
                    except ValueError:
                        continue
        if not ratios:
            return 0

        # Count consecutive entries from end that are above/below threshold
        direction = "above" if ls_ratio > threshold else "below" if ls_ratio < (1.0 / threshold) else None
        if direction is None:
            return 0

        count = 0
        for r in reversed(ratios):
            if direction == "above" and r > threshold:
                count += 1
            elif direction == "below" and r < (1.0 / threshold):
                count += 1
            else:
                break

        # Each entry is ~15 min, so count * 15 = minutes
        return count * 15
    except Exception:
        return 0


class PositioningFadeStrategy(BaseStrategy):
    name = 'positioning_fade'
    strategy_type = 'flow'
    description = 'v3: rolling L/S stats + time-in-position + gate validated'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        deriv = data.get('derivatives', {})
        if not deriv:
            return None

        # ── REGIME FILTER ──
        m9 = data.get('m9', {})
        vol_regime = m9.get('regime', 'UNKNOWN')
        if vol_regime in BAD_REGIMES:
            return None

        ls_ratio = deriv.get('ls_ratio', 1.0)
        positioning = deriv.get('positioning', 'NEUTRAL')
        whale = deriv.get('whale_signal', 'NEUTRAL')

        # ── ROLLING L/S Z-SCORE (replaces hardcoded 2.15/0.3) ──
        stats = _read_rolling_ls_stats()
        if stats and stats["std"] > 0.01:
            ls_mean = stats["mean"]
            ls_std = stats["std"]
        else:
            # Fallback to hardcoded if no history
            ls_mean = 2.15
            ls_std = 0.3

        ls_zscore = (ls_ratio - ls_mean) / ls_std if ls_std > 0 else 0

        # Need extreme positioning (z > 1.0 or explicit positioning label)
        if abs(ls_zscore) < 1.0 and positioning not in ('EXTREME_LONG', 'EXTREME_SHORT', 'BULLISH', 'BEARISH'):
            return None

        # ── TIME-IN-POSITION FILTER (new) ──
        # Need extreme positioning held for 30+ minutes
        if abs(ls_zscore) > 1.0:
            threshold = ls_mean + ls_std * 1.0  # the level that counts as "extreme"
            if ls_ratio > ls_mean:
                duration = _read_ls_duration(ls_ratio, threshold=threshold)
            else:
                duration = _read_ls_duration(ls_ratio, threshold=ls_mean - ls_std)
        else:
            duration = _read_ls_duration(ls_ratio, threshold=2.0)

        if duration < 30:
            return None  # Need at least 30 minutes of extreme positioning

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        ema_200 = data.get('ema_200', 0)
        if not price or not atr:
            return None

        # Direction: fade the crowd
        if ls_zscore > 1.0 or positioning in ('EXTREME_LONG', 'BULLISH'):
            direction = 'SHORT'
            extreme = ls_zscore
        elif ls_zscore < -1.0 or positioning in ('EXTREME_SHORT', 'BEARISH'):
            direction = 'LONG'
            extreme = abs(ls_zscore)
        else:
            return None

        # EMA200 trend filter (don't fade against strong trend)
        if ema_200 and ema_200 > 0:
            dist = (price - ema_200) / ema_200
            if direction == 'LONG' and dist < -0.03:
                return None
            if direction == 'SHORT' and dist > 0.03:
                return None

        # Whale confirmation bonus
        whale_confirm = 0
        tls = deriv.get('top_ls_ratio', 0)
        if tls > 0:
            if stats and stats["std"] > 0.01:
                top_z = (tls - ls_mean) / ls_std
            else:
                top_z = (tls - 2.15) / 0.3
            if (direction == 'SHORT' and top_z > 0.5) or \
               (direction == 'LONG' and top_z < -0.5):
                whale_confirm = 0.15

        # Regime bonus
        regime_bonus = 0.10 if vol_regime in ('RANGING', 'CHOP_MILD') else 0

        # Duration bonus (longer = more conviction, up to a point)
        duration_bonus = min(duration / 480, 0.10)  # max at 8 hours

        conviction = min(0.40 + (abs(extreme) - 1.0) * 0.10 + whale_confirm + regime_bonus + duration_bonus, 0.85)
        if conviction < 0.40:
            return None

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.2)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.6,
            reason=f"Positioning fade v3 ({vol_regime}): z={ls_zscore:.2f} "
                   f"duration={duration}m -> {direction}",
            bypass_gates=False,  # v3: gate validated
            details={
                'ls_ratio': ls_ratio, 'ls_zscore': float(ls_zscore),
                'ls_mean': float(ls_mean), 'ls_std': float(ls_std),
                'duration_min': duration, 'positioning': positioning,
                'whale': whale, 'vol_regime': vol_regime,
                'version': 'v3',
            },
        )
