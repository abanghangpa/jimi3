"""S04: Positioning Fade — fade extreme derivatives positioning."""
from .base import BaseStrategy, SignalResult
import math

class PositioningFadeStrategy(BaseStrategy):
    name = 'positioning_fade'
    strategy_type = 'flow'
    description = 'Fade extreme L/S ratio and whale positioning when crowded'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        deriv = data.get('derivatives', {})
        if not deriv:
            return None

        ls_ratio = deriv.get('ls_ratio', 0)
        # Always compute zscore from ls_ratio (scan has stale pre-computed value)
        ls_ratio = deriv.get('ls_ratio', 1.0)
        ls_zscore = (ls_ratio - 2.15) / 0.3 if ls_ratio > 0 else 0
        positioning = deriv.get('positioning', 'NEUTRAL')
        whale = deriv.get('whale_signal', 'NEUTRAL')

        # Need extreme positioning
        if abs(ls_zscore) < 0.8 and positioning not in ('EXTREME_LONG', 'EXTREME_SHORT', 'BULLISH', 'BEARISH'):
            return None

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr:
            return None

        # Direction: fade the crowd
        if ls_zscore > 1.5 or positioning == 'EXTREME_LONG':
            direction = 'SHORT'
            extreme = ls_zscore
        elif ls_zscore < -1.5 or positioning == 'EXTREME_SHORT':
            direction = 'LONG'
            extreme = abs(ls_zscore)
        else:
            return None

        # Whale confirmation bonus
        whale_confirm = 0
        if (direction == 'SHORT' and whale == 'BEARISH') or \
           (direction == 'LONG' and whale == 'BULLISH'):
            whale_confirm = 0.15

        conviction = min(0.40 + (abs(extreme) - 0.8) * 0.15 + whale_confirm, 0.85)
        if conviction < 0.35:
            return None

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.2)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.6,
            reason=f"Positioning fade: L/S zscore={ls_zscore:.2f} → {direction} "
                   f"(whale={whale}, positioning={positioning})",
            bypass_gates=True,
            details={'ls_ratio': ls_ratio, 'ls_zscore': ls_zscore,
                     'positioning': positioning, 'whale': whale},
        )
