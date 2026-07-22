"""S16: Multi-Timeframe Confluence — align signals across 15m, 1h, 4h, 1d."""
from .base import BaseStrategy, SignalResult

class MTFConfluenceStrategy(BaseStrategy):
    name = 'mtf_confluence'
    strategy_type = 'structure'
    description = 'Trade when multiple timeframe indicators align in same direction'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        direction = data.get('direction')
        if not direction:
            return None

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr:
            return None

        # Count modules aligned with direction
        aligned = 0
        total = 0
        modules = ['m1', 'm2', 'm3', 'm4', 'm5', 'm7', 'm10', 'm11', 'm12', 'm13', 'm14']

        for mod in modules:
            m = data.get(mod, {})
            if not m or m.get('status') in ('SKIP', 'ERROR'):
                continue
            total += 1
            score = m.get('score', 0.5)
            # Score > 0.55 = bullish, < 0.45 = bearish
            if direction == 'LONG' and score > 0.55:
                aligned += 1
            elif direction == 'SHORT' and score < 0.45:
                aligned += 1

        if total < 5 or aligned < 4:
            return None

        confluence = aligned / total
        conviction = min(confluence * 0.7 + 0.15, 0.85)
        if conviction < 0.5:
            return None

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.8,
            reason=f"MTF confluence → {direction}: {aligned}/{total} modules aligned",
            bypass_gates=False,
            details={'aligned': aligned, 'total': total, 'confluence': confluence},
        )
