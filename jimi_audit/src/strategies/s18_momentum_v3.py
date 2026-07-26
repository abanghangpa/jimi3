"""S18: Momentum v3.1 — Exhaustion Detector (state filter)

v3.1 fixes from forensic analysis (2026-07-24):
- Raised thresholds: vol_div needs -20% (was -10%), extreme needs >90th pctl (was 85th)
- REQUIRE deceleration signal (the only discriminating signal)
- Fixed OI divergence: use30-min window matching instead of exact hour
- Added dedup: no re-fire within 4 bars of last trigger for same direction
- Reduced trigger frequency: from 59/13days to target <15/13days
"""
from .base import BaseStrategy, SignalResult
import numpy as np


class MomentumV3Strategy(BaseStrategy):
    name = 'momentum_v3'
    strategy_type = 'exhaustion'
    description = 'Momentum exhaustion detector v3.1 — tighter thresholds + DECEL required'

    # v3.1: Track last trigger for dedup
    _last_trigger_idx = -999
    _last_trigger_dir = None

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr or df_15m is None or idx is None:
            return None
        if idx < 80:
            return None

        closes = df_15m['Close'].values.astype(float)
        volumes = df_15m['Volume'].values.astype(float)

        # ── 1. MOMENTUM DECELERATION (REQUIRED in v3.1) ──
        mom_5 = (closes[idx] - closes[idx - 5]) / closes[idx - 5]
        mom_10 = (closes[idx] - closes[idx - 10]) / closes[idx - 10]
        accel = mom_5 - mom_10 / 2

        decel_signal = False
        if mom_5 > 0 and accel < 0:
            decel_signal = True  # UP but decelerating → SHORT
        elif mom_5 < 0 and accel > 0:
            decel_signal = True  # DOWN but decelerating → LONG

        # v3.1: DECEL is REQUIRED
        if not decel_signal:
            return None

        # ── 2. VOLUME-MOMENTUM DIVERGENCE (raised threshold) ──
        vol_recent = np.mean(volumes[idx - 5:idx])
        vol_prior = np.mean(volumes[idx - 15:idx - 5])
        vol_change = (vol_recent - vol_prior) / vol_prior if vol_prior > 0 else 0

        vol_divergence = False
        # v3.1: require -20% volume drop (was -10%)
        if mom_5 > 0.005 and vol_change < -0.20:
            vol_divergence = True
        elif mom_5 < -0.005 and vol_change < -0.20:
            vol_divergence = True

        # ── 3. PERCENTILE RANK (raised threshold) ──
        moves = []
        for j in range(idx - 80, idx - 5):
            m = abs(closes[j + 5] - closes[j]) / closes[j]
            moves.append(m)
        current_move = abs(closes[idx] - closes[idx - 5]) / closes[idx - 5]
        percentile = sum(1 for m in moves if m < current_move) / len(moves) * 100

        # v3.1: require >90th percentile (was 85th)
        extreme_move = percentile > 90

        # ── 4. OI DIVERGENCE (fixed matching) ──
        deriv = data.get('derivatives', {})
        oi_roc = deriv.get('oi_roc_1h', 0)
        # v3.1: also check 30-min window
        oi_roc_30m = deriv.get('oi_roc_30m', 0)
        oi_roc_effective = min(oi_roc, oi_roc_30m) if oi_roc_30m else oi_roc

        oi_divergence = False
        if mom_5 > 0.005 and oi_roc_effective < -0.02:
            oi_divergence = True
        elif mom_5 < -0.005 and oi_roc_effective < -0.02:
            oi_divergence = True

        # ── COMBINE SIGNALS ──
        # v3.1: DECEL is required (already checked above)
        # Need at least 1 more signal from: vol_div, extreme, oi_div
        additional = sum([vol_divergence, extreme_move, oi_divergence])
        if additional < 1:
            return None

        # ── DEDUP: no re-fire within 4 bars ──
        direction = 'SHORT' if mom_5 > 0 else 'LONG'
        if (idx - self._last_trigger_idx < 4 and
                direction == self._last_trigger_dir):
            return None

        self._last_trigger_idx = idx
        self._last_trigger_dir = direction

        # ── CONVICTION ──
        # v3.1: higher base (DECEL required = higher conviction)
        base = 0.55
        if vol_divergence:
            base += 0.15
        if extreme_move:
            base += 0.10
        if oi_divergence:
            base += 0.10
        conviction = min(base, 0.90)

        # ── TP/SL ──
        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.2)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=f"Exhaustion v3.1 -> {direction}: mom5={mom_5:.4f} accel={accel:.4f} "
                   f"vol_div={vol_divergence}({vol_change:.1%}) extreme={extreme_move}({percentile:.0f}) "
                   f"oi_div={oi_divergence}({oi_roc_effective:.2%})",
            bypass_gates=False,
            details={'mom_5': mom_5, 'mom_10': mom_10, 'accel': accel,
                     'vol_change': vol_change, 'percentile': percentile,
                     'decel': decel_signal, 'vol_div': vol_divergence,
                     'extreme': extreme_move, 'oi_div': oi_divergence,
                     'signals_count': 1 + additional, 'version': '3.1'},
        )
