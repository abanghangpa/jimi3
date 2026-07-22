"""S08: Regime Switch — position for regime transitions."""
from .base import BaseStrategy, SignalResult

class RegimeSwitchStrategy(BaseStrategy):
    name = 'regime_switch'
    strategy_type = 'regime'
    description = 'Trade regime transitions (NEUTRAL→TRENDING, COMPRESSING→EXPANSION)'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        m9 = data.get('m9', {})
        m22 = data.get('m22', {})
        m23 = data.get('m23', {})

        vol_regime = m9.get('regime', 'UNKNOWN')
        inflation_regime = m22.get('regime', 'UNKNOWN')
        macro_regime = m23.get('regime', 'UNKNOWN')

        direction = data.get('direction')
        if not direction:
            return None

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr:
            return None

        # Score regime alignment
        regime_score = 0
        regime_reasons = []

        # Volatility regime
        if vol_regime == 'TRENDING':
            regime_score += 0.3
            regime_reasons.append('vol_trending')
        elif vol_regime == 'COMPRESSING':
            regime_score += 0.2
            regime_reasons.append('vol_compressing')

        # Inflation regime
        if inflation_regime in ('STAGFLATION_LITE', 'STAGFLATION_HOT'):
            regime_score += 0.2
            regime_reasons.append(f'inflation_{inflation_regime.lower()}')

        # Macro lifecycle
        macro_phase = data.get('macro_lifecycle', {}).get('phase', '')
        if macro_phase in ('FIRST_US', 'SECOND_US'):
            regime_score += 0.15
            regime_reasons.append(f'macro_{macro_phase.lower()}')

        # M22 severity
        severity = m22.get('severity', '')
        if severity in ('HIGH', 'MEDIUM'):
            regime_score += 0.15
            regime_reasons.append(f'severity_{severity.lower()}')

        if regime_score < 0.35:
            return None

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=min(regime_score, 0.85),
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=f"Regime switch → {direction}: {', '.join(regime_reasons)}",
            bypass_gates=False,
            details={'vol_regime': vol_regime, 'inflation_regime': inflation_regime,
                     'macro_regime': macro_regime, 'severity': severity},
        )
