"""S11: Cross-Asset Divergence — ETH/BTC diverging from BTC → relative value."""
from .base import BaseStrategy, SignalResult

class CrossAssetStrategy(BaseStrategy):
    min_vol_ratio = 0.12  # require above-average volume
    name = 'cross_asset'
    strategy_type = 'regime'
    description = 'Trade cross-asset divergences between ETH, BTC, and traditional markets'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        m10 = data.get('m10', {})
        m7 = data.get('m7', {})
        ex = data.get('exchange_activity', {})

        m10_score = m10.get('score', 0.5)
        m7_score = m7.get('score', 0.5)
        ex_score = ex.get('score', 0.5)

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr:
            return None

        # Check both directions for cross-asset alignment
        long_alignment = (m10_score + m7_score + ex_score) / 3
        short_alignment = (1 - m10_score + 1 - m7_score + 1 - ex_score) / 3

        # Pick the stronger aligned direction
        if long_alignment >= short_alignment and long_alignment >= 0.55:
            direction = 'LONG'
            alignment = long_alignment
        elif short_alignment > long_alignment and short_alignment >= 0.55:
            direction = 'SHORT'
            alignment = short_alignment
        else:
            return None

        conviction = min(alignment * 0.8 + 0.1, 0.80)

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.6,
            reason=f"Cross-asset alignment → {direction}: M10={m10_score:.2f} "
                   f"M7={m7_score:.2f} EX={ex_score:.2f}",
            bypass_gates=False,
            details={'m10_score': m10_score, 'm7_score': m7_score,
                     'exchange_score': ex_score, 'alignment': alignment,
                     'long_alignment': long_alignment, 'short_alignment': short_alignment},
        )
