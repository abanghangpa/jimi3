"""S17: Range Scalper v2 — improved scalp with relaxed ICS and event bypass."""
from .base import BaseStrategy, SignalResult

class ScalpV2Strategy(BaseStrategy):
    name = 'scalp_v2'
    strategy_type = 'structure'
    description = 'Improved range scalper with lower ICS threshold and event-driven bypass'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        direction = data.get('direction')
        ics = data.get('ics', 0)
        threshold = data.get('threshold', 0.50)

        if not direction or ics < 0.45:
            return None

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr:
            return None

        # Check key modules
        m3 = data.get('m3', {})
        m4 = data.get('m4', {})
        m5 = data.get('m5', {})

        m3_ok = m3.get('status') == 'PASS' and m3.get('score', 0) > 0.3
        m4_ok = m4.get('status') == 'PASS'
        m5_ok = m5.get('score', 0) > 0.5

        if not (m3_ok or m5_ok):
            return None

        # Conviction from ICS and module alignment
        module_score = sum([m3_ok, m4_ok, m5_ok]) / 3
        conviction = min(ics * 0.6 + module_score * 0.3 + 0.1, 0.80)

        if conviction < 0.6:
            return None

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=1.0,
            reason=f"Scalp v2 → {direction}: ICS={ics:.3f} M3={m3.get('score', 0):.2f} "
                   f"M5={m5.get('score', 0):.2f}",
            bypass_gates=False,
            details={'ics': ics, 'm3_score': m3.get('score'), 'm5_score': m5.get('score')},
        )
