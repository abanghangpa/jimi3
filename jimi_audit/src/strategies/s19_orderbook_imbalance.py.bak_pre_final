"""S19: Order Book Imbalance — trade multi-exchange order book pressure.
Fixes: 2h cooldown after loss, direction persistence (no flip within2h), max loss cap at1.5%."""
from .base import BaseStrategy, SignalResult
from datetime import datetime, timezone, timedelta

class OrderBookImbalanceStrategy(BaseStrategy):
    min_vol_ratio = 0.15
    name = 'orderbook_imbalance'
    strategy_type = 'flow'
    description = 'Trade when order book shows strong buy/sell imbalance'

    _last_trade_time = None
    _last_direction = None
    _last_loss_time = None
    COOLDOWN_MINUTES = 120  # 2h cooldown after any trade
    LOSS_COOLDOWN_MINUTES = 180  # 3h cooldown after loss
    MAX_LOSS_PCT = 1.5  # wider SL cap (was getting stopped at3.38%)

    def check(self, data, df_15m=None, idx=None, **kwargs):
        now = datetime.now(timezone.utc)

        # Cooldown after loss
        if self._last_loss_time and (now - self._last_loss_time).total_seconds() < self.LOSS_COOLDOWN_MINUTES * 60:
            return None

        # Cooldown after any trade
        if self._last_trade_time and (now - self._last_trade_time).total_seconds() < self.COOLDOWN_MINUTES * 60:
            return None

        ob_data = kwargs.get('order_flow', {})
        if not ob_data:
            return None

        imbalance = ob_data.get('avg_imbalance', 1.0)
        consensus = ob_data.get('consensus', 'NEUTRAL')
        bullish_ex = ob_data.get('bullish_exchanges', 0)
        bearish_ex = ob_data.get('bearish_exchanges', 0)

        if consensus == 'NEUTRAL':
            return None

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr:
            return None

        if consensus == 'BULLISH':
            direction = 'LONG'
            extreme = imbalance - 1.0
        else:
            direction = 'SHORT'
            extreme = 1.0 - imbalance

        # Direction persistence: don't flip within2h
        if self._last_direction and self._last_direction != direction:
            if self._last_trade_time and (now - self._last_trade_time).total_seconds() < self.COOLDOWN_MINUTES * 60:
                return None

        imbalance_score = min(extreme * 2, 0.4)
        exchange_score = min(max(bullish_ex, bearish_ex) / 3, 0.3)

        exchanges = ob_data.get('exchanges', {})
        wall_bonus = 0
        for ex_name, ex_data in exchanges.items():
            if direction == 'LONG' and ex_data.get('bid_wall_count', 0) > 0:
                wall_bonus = 0.15; break
            elif direction == 'SHORT' and ex_data.get('ask_wall_count', 0) > 0:
                wall_bonus = 0.15; break

        conviction = min(0.40 + imbalance_score + exchange_score + wall_bonus, 0.90)
        if conviction < 0.55:
            return None

        # Wider SL for OB strategy (order flow whipsaws)
        sl_mult = 1.5  # wider than default1.0
        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=sl_mult)

        # Cap max loss at1.5%
        if sl_pct > self.MAX_LOSS_PCT:
            return None

        self._last_trade_time = now
        self._last_direction = direction

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.8,  # smaller size (wider stops)
            reason=f"OB {consensus} -> {direction}: imbalance={imbalance:.3f} ({bullish_ex}B/{bearish_ex}S)",
            bypass_gates=True,
            details={'imbalance': imbalance, 'consensus': consensus},
        )

    def record_outcome(self, outcome):
        """Call this when a trade closes to update cooldown state."""
        if outcome in ('LOSS', 'SL_HIT', 'TIMEOUT'):
            self._last_loss_time = datetime.now(timezone.utc)
