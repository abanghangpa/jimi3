"""S21: Trade Flow Momentum — follow aggressive recent trade flow.
Fixes: Don't SHORT when price > EMA200 (uptrend), don't LONG when price < EMA200 (downtrend).
Reduced SHORT sensitivity."""
from .base import BaseStrategy, SignalResult

class TradeFlowStrategy(BaseStrategy):
    min_vol_ratio = 0.15
    name = 'trade_flow'
    strategy_type = 'flow'
    description = 'Follow aggressive trade flow with trend alignment'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        trade_data = kwargs.get('trade_flow', {})
        if not trade_data:
            return None

        taker_ratio = trade_data.get('taker_ratio', 0.5)
        net_flow = trade_data.get('net_flow', 0)
        large_buys = trade_data.get('large_buy_count', 0)
        large_sells = trade_data.get('large_sell_count', 0)

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        ema_200 = data.get('ema_200', 0)
        if not price or not atr:
            return None

        # Direction from flow
        if taker_ratio > 0.60 and net_flow > 0:
            direction = 'LONG'
        elif taker_ratio < 0.40 and net_flow < 0:
            direction = 'SHORT'
        else:
            return None

        # Trend alignment: only trade in direction of EMA200
        if ema_200 and ema_200 > 0:
            if direction == 'LONG' and price < ema_200 * 0.98:
                return None  # too far below EMA for long
            if direction == 'SHORT' and price > ema_200 * 1.02:
                return None  # too far above EMA for short

        flow_strength = abs(taker_ratio - 0.5) * 2
        flow_score = min(flow_strength * 0.4, 0.35)

        if direction == 'LONG' and large_buys > large_sells:
            large_bonus = min((large_buys - large_sells) * 0.05, 0.2)
        elif direction == 'SHORT' and large_sells > large_buys:
            large_bonus = min((large_sells - large_buys) * 0.05, 0.2)
        else:
            large_bonus = 0

        conviction = min(0.45 + flow_score + large_bonus, 0.90)
        if conviction < 0.55:
            return None

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.8,
            reason=f"Trade flow -> {direction}: taker={taker_ratio:.3f} net=${net_flow/1000:.0f}k",
            bypass_gates=True,
            details={'taker_ratio': taker_ratio, 'net_flow': net_flow},
        )
