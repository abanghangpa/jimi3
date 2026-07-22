"""S14: Whale Watch — follow whale positioning and smart money flow."""
from .base import BaseStrategy, SignalResult

class WhaleWatchStrategy(BaseStrategy):
    name = 'whale_watch'
    strategy_type = 'flow'
    description = 'Follow whale positioning when whales are clearly positioned one direction'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        deriv = data.get('derivatives', {})
        if not deriv:
            return None

        whale = deriv.get('whale_signal', 'NEUTRAL')
        # Derive from ls_ratio if neutral
        if whale == 'NEUTRAL':
            ls_ratio = deriv.get('ls_ratio', 1.0)
            if ls_ratio > 2.3:
                whale = 'BEARISH'
            elif ls_ratio < 1.8:
                whale = 'BULLISH'
        positioning = deriv.get('positioning', 'NEUTRAL')
        ls_ratio = deriv.get('ls_ratio', 1.0)

        if whale == 'NEUTRAL' or whale == '':
            return None

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr:
            return None

        # Direction from whale signal
        if whale == 'BULLISH':
            direction = 'LONG'
        elif whale == 'BEARISH':
            direction = 'SHORT'
        else:
            return None

        # Confirmation from positioning
        pos_confirm = 0
        if (direction == 'LONG' and positioning in ('BULLISH', 'EXTREME_LONG')) or \
           (direction == 'SHORT' and positioning in ('BEARISH', 'EXTREME_SHORT')):
            pos_confirm = 0.15

        conviction = min(0.40 + pos_confirm + abs(ls_ratio - 1.0) * 0.2, 0.80)
        if conviction < 0.40:
            return None

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=f"Whale watch → {direction}: whale={whale} positioning={positioning}",
            bypass_gates=False,
            details={'whale_signal': whale, 'positioning': positioning,
                     'ls_ratio': ls_ratio},
        )
