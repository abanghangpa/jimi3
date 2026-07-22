"""S13: Funding Rate Arb — trade funding rate extremes.
Fixes: EMA200 trend filter (12/12 losses were LONG in downtrend)."""
from .base import BaseStrategy, SignalResult

class FundingArbStrategy(BaseStrategy):
    name = 'funding_arb'
    strategy_type = 'flow'
    description = 'Trade funding rate extremes with trend alignment'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        deriv = data.get('derivatives', {})
        if not deriv:
            return None

        oi_roc = deriv.get('oi_roc_1h', 0)
        ls_ratio = deriv.get('ls_ratio', 1.0)

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        ema_200 = data.get('ema_200', 0)
        if not price or not atr:
            return None

        if abs(oi_roc) < 0.02:
            return None

        # Direction from OI flow
        if oi_roc < -0.03 and ls_ratio > 1.5:
            direction = 'LONG'
        elif oi_roc > 0.03 and ls_ratio < 0.7:
            direction = 'SHORT'
        else:
            return None

        # CRITICAL FIX: Don't LONG in downtrend (12/12 losses were LONG)
        if ema_200 and ema_200 > 0:
            if direction == 'LONG' and price < ema_200:
                return None  # don't buy below EMA200
            if direction == 'SHORT' and price > ema_200:
                return None  # don't short above EMA200

        conviction = min(0.35 + abs(oi_roc) * 5, 0.75)

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.6,
            reason=f"Funding arb -> {direction}: OI_roc={oi_roc:.4f} L/S={ls_ratio:.2f}",
            bypass_gates=False,
            details={'oi_roc_1h': oi_roc, 'ls_ratio': ls_ratio},
        )
