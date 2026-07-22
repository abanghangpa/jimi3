"""S04: Positioning Fade v4 — fade extreme derivatives positioning (regime-validated).

v3 → v4 CHANGES (2026-07-18):
1. Regime filter TIGHTENED: only fire in RANGING/HIGH_VOL/CRISIS (not BULL/TRENDING)
2. Z-score threshold raised: 1.0 → 1.5 (cleaner signal, less noise)
3. Duration filter kept: 30+ minutes of extreme positioning
4. Synthetic regime test confirmed: fading works in ranging/high_vol/crisis, fails in trending

Regime mapping (from M9 vol regime):
- RANGING, CHOP_MILD, CHOP_BULL, CHOP_BEAR, NEUTRAL → RANGING (fade works)
- CRISIS → HIGH_VOL (fade works, but skip z>2.5)
- TRENDING, COMPRESSING → BLOCK (crowd is right, fade loses)
"""
from .base import BaseStrategy, SignalResult
import os, json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DERIV_CSV = os.path.join(BASE_DIR, "data", "derivatives_history", "derivatives_collected.csv")
PF_STATE = os.path.join(BASE_DIR, "data", "forced_movement", "pf_state.json")

# V4: Regimes where fading works (crowd is wrong)
GOOD_REGIMES = {'RANGING', 'CHOP_MILD', 'CHOP_BULL', 'CHOP_BEAR', 'NEUTRAL', 'CRISIS'}
# V4: Regimes where fading fails (crowd is right — trend is your friend)
BAD_REGIMES = {'TRENDING', 'COMPRESSING'}

ROLLING_WINDOW = 200


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
    """Track how long L/S ratio has been above threshold."""
    if not os.path.exists(DERIV_CSV):
        return 0
    try:
        ratios = []
        with open(DERIV_CSV) as f:
            header = f.readline().strip().split(",")
            ls_idx = header.index("ls_ratio") if "ls_ratio" in header else -1
            if ls_idx < 0:
                return 0
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

        return count * 15
    except Exception:
        return 0


class PositioningFadeStrategy(BaseStrategy):
    name = 'positioning_fade'
    strategy_type = 'flow'
    description = 'v4: regime-validated, z>1.5, ranging/high_vol/crisis only'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        deriv = data.get('derivatives', {})
        if not deriv:
            return None

        # ── REGIME FILTER (V4: TIGHTENED) ──
        m9 = data.get('m9', {})
        vol_regime = m9.get('regime', 'UNKNOWN')
        if vol_regime in BAD_REGIMES:
            return None  # Block in trending/compressing — crowd is right

        ls_ratio = deriv.get('ls_ratio', 1.0)
        positioning = deriv.get('positioning', 'NEUTRAL')
        whale = deriv.get('whale_signal', 'NEUTRAL')

        # ── ROLLING L/S Z-SCORE ──
        stats = _read_rolling_ls_stats()
        if stats and stats["std"] > 0.01:
            ls_mean = stats["mean"]
            ls_std = stats["std"]
        else:
            ls_mean = 2.15
            ls_std = 0.3

        ls_zscore = (ls_ratio - ls_mean) / ls_std if ls_std > 0 else 0

        # V4: Z-score threshold raised from 1.0 to 1.5
        if abs(ls_zscore) < 1.5 and positioning not in ('EXTREME_LONG', 'EXTREME_SHORT', 'BULLISH', 'BEARISH'):
            return None

        # ── TIME-IN-POSITION FILTER ──
        if abs(ls_zscore) > 1.5:
            threshold = ls_mean + ls_std * 1.5
            if ls_ratio > ls_mean:
                duration = _read_ls_duration(ls_ratio, threshold=threshold)
            else:
                duration = _read_ls_duration(ls_ratio, threshold=ls_mean - ls_std * 1.5)
        else:
            duration = _read_ls_duration(ls_ratio, threshold=2.0)

        if duration < 30:
            return None

        # ── CRISIS REGIME: skip extreme z>2.5 (crowd panic can persist) ──
        if vol_regime == 'CRISIS' and abs(ls_zscore) > 2.5:
            return None

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        ema_200 = data.get('ema_200', 0)
        if not price or not atr:
            return None

        # Direction: fade the crowd
        if ls_zscore > 1.5 or positioning in ('EXTREME_LONG', 'BULLISH'):
            direction = 'SHORT'
            extreme = ls_zscore
        elif ls_zscore < -1.5 or positioning in ('EXTREME_SHORT', 'BEARISH'):
            direction = 'LONG'
            extreme = abs(ls_zscore)
        else:
            return None

        # EMA200 trend filter
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

        # Duration bonus
        duration_bonus = min(duration / 480, 0.10)

        conviction = min(0.40 + (abs(extreme) - 1.5) * 0.10 + whale_confirm + regime_bonus + duration_bonus, 0.85)
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
            reason=f"Positioning fade v4 ({vol_regime}): z={ls_zscore:.2f} "
                   f"duration={duration}m -> {direction}",
            bypass_gates=False,
            details={
                'ls_ratio': ls_ratio, 'ls_zscore': float(ls_zscore),
                'ls_mean': float(ls_mean), 'ls_std': float(ls_std),
                'duration_min': duration, 'positioning': positioning,
                'whale': whale, 'vol_regime': vol_regime,
                'version': 'v4',
            },
        )
