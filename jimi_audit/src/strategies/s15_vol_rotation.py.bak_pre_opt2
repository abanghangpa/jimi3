"""S15: Volatility Rotation — switch between mean-reversion and trend-following based on vol regime."""
from .base import BaseStrategy, SignalResult

class VolRotationStrategy(BaseStrategy):
    name = 'vol_rotation'
    strategy_type = 'regime'
    description = 'Switch strategy based on volatility regime: mean-revert in low vol, trend in high vol'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        m9 = data.get('m9', {})
        squeeze = data.get('squeeze', {})

        vol_regime = m9.get('regime', 'UNKNOWN')
        squeeze_active = squeeze.get('squeeze_status') == 'TRIGGERED'
        vol_ratio = data.get('vol_ratio', 1.0)

        direction = data.get('direction')
        if not direction:
            return None

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr:
            return None

        # Strategy selection based on vol regime
        strategy_mode = ''
        if vol_regime in ('COMPRESSING', 'LOW_VOL') or squeeze_active:
            # Mean reversion: fade moves in low vol
            strategy_mode = 'mean_revert'
            conviction = 0.45
        elif vol_regime in ('TRENDING', 'HIGH_VOL') or vol_ratio > 1.5:
            # Trend following: ride momentum in high vol
            strategy_mode = 'trend_follow'
            conviction = 0.50
        else:
            return None

        # Vwap confirmation
        vwap_dist = data.get('vwap_dist', 0)
        if strategy_mode == 'mean_revert':
            # For mean reversion, want price far from VWAP
            if abs(vwap_dist) > 0.5:
                conviction += 0.15
        else:
            # For trend following, want price moving away from VWAP
            if (direction == 'LONG' and vwap_dist > 0) or \
               (direction == 'SHORT' and vwap_dist < 0):
                conviction += 0.10

        if conviction < 0.5:
            return None

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=min(conviction, 0.80),
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=f"Vol rotation ({strategy_mode}) → {direction}: "
                   f"regime={vol_regime} vol_ratio={vol_ratio:.2f}",
            bypass_gates=False,
            details={'strategy_mode': strategy_mode, 'vol_regime': vol_regime,
                     'vol_ratio': vol_ratio, 'squeeze_active': squeeze_active},
        )
