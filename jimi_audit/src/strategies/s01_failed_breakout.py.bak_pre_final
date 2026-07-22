"""S01: Failed Breakout — M20 detected failed breakout -> contrarian entry.
Fixes: EMA200 trend filter (don't buy dips in downtrend), wider SL."""
from .base import BaseStrategy, SignalResult

class FailedBreakoutStrategy(BaseStrategy):
    name = 'failed_breakout'
    strategy_type = 'event'
    description = 'Trade contrarian when M20 detects a failed breakout pattern'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        m20 = data.get('m20', {})
        if m20.get('status') != 'PASS':
            return None
        failure = m20.get('failure', {})
        if not failure.get('failed', False):
            return None

        direction = m20.get('contrarian_direction')
        if not direction:
            return None

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        ema_200 = data.get('ema_200', 0)
        if not price or not atr:
            return None

        # CRITICAL FIX: Don't buy dips in downtrend (19/20 losses were LONG)
        if ema_200 and ema_200 > 0:
            if direction == 'LONG' and price < ema_200:
                return None  # don't buy below EMA200
            if direction == 'SHORT' and price > ema_200:
                return None  # don't short above EMA200

        reversal = m20.get('reversal_score', 0.5)
        quality = 1.0 - m20.get('breakout_quality', 0.5)
        bars_since = failure.get('bars_since', 99)
        freshness = max(0, 1.0 - bars_since / 100)
        conviction = (reversal * 0.4 + quality * 0.3 + freshness * 0.3)
        conviction = min(conviction, 0.95)

        # Wider stops for failed breakouts
        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(2.0, 3.5, 5.0), sl_mult=1.5)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.8,
            reason=f"M20 failed BO: {m20.get('breakout_direction')} -> {direction} (rev={reversal:.2f})",
            bypass_gates=True,
            details={'m20_score': m20.get('score'), 'bars_since': bars_since},
        )
