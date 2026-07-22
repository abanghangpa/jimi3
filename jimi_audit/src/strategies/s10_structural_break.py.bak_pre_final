"""S10: Structural Break — trade BOS/ChoCH with momentum confirmation.
Fixes: Require M1+M2 alignment, EMA200 filter (6/8 losses were SHORT in bounce)."""
from .base import BaseStrategy, SignalResult

class StructuralBreakStrategy(BaseStrategy):
    name = 'structural_break'
    strategy_type = 'structure'
    description = 'Trade structural breaks with momentum and trend confirmation'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        m1 = data.get('m1', {})
        m2 = data.get('m2', {})
        m13 = data.get('m13', {})

        m1_dir = m1.get('direction', 'NEUTRAL')
        m1_score = m1.get('score', 0.5)
        m2_status = m2.get('status', 'NEUTRAL')
        m2_score = m2.get('score', 0.5)

        if m1_dir == 'NEUTRAL':
            return None

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        ema_200 = data.get('ema_200', 0)
        if not price or not atr:
            return None

        # Direction from M1 + M2 alignment
        if m1_dir == 'BULLISH' and m2_status in ('BULLISH', 'NEUTRAL'):
            direction = 'LONG'
        elif m1_dir == 'BEARISH' and m2_status in ('BEARISH', 'NEUTRAL'):
            direction = 'SHORT'
        else:
            return None

        # FIX: EMA200 trend filter (6/8 losses were SHORT in bounce)
        if ema_200 and ema_200 > 0:
            if direction == 'LONG' and price < ema_200 * 0.97:
                return None  # too far below EMA
            if direction == 'SHORT' and price > ema_200 * 1.03:
                return None  # too far above EMA

        m13_score = m13.get('score', 0.5)
        conviction = (m1_score * 0.4 + m2_score * 0.3 + m13_score * 0.3)

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=min(conviction, 0.90),
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=f"Struct break -> {direction}: M1={m1_dir} M2={m2_status} M13={m13_score:.2f}",
            bypass_gates=False,
            details={'m1_dir': m1_dir, 'm2_status': m2_status, 'm13_score': m13_score},
        )
