"""S09: Power of 3 — trade Wyckoff accumulation/distribution phases."""
from .base import BaseStrategy, SignalResult

class PowerOf3Strategy(BaseStrategy):
    name = 'power_of_3'
    strategy_type = 'structure'
    description = 'Trade based on Wyckoff Power of 3 phase detection'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        p3 = data.get('power_of_3', {})
        if not p3:
            return None

        phase = p3.get('phase', '')
        confidence = p3.get('confidence', 0)
        p3_direction = p3.get('direction', 'NEUTRAL')
        key_level = p3.get('key_level')

        if phase not in ('ACCUMULATION', 'DISTRIBUTION', 'MARKUP', 'MARKDOWN'):
            return None
        if confidence < 0.4:
            return None

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr:
            return None

        # Map phase to direction
        if phase in ('ACCUMULATION', 'MARKUP'):
            direction = 'LONG'
        elif phase in ('DISTRIBUTION', 'MARKDOWN'):
            direction = 'SHORT'
        else:
            return None

        # Bonus if p3_direction agrees
        dir_bonus = 0.15 if p3_direction == direction else 0

        conviction = min(confidence * 0.6 + dir_bonus + 0.2, 0.85)
        if conviction < 0.55:
            return None

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 3.0, 5.0), sl_mult=1.2)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=f"Power of 3: {phase} → {direction} (conf={confidence:.2f})",
            bypass_gates=False,
            details={'phase': phase, 'confidence': confidence,
                     'p3_direction': p3_direction, 'key_level': key_level},
        )
