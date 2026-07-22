"""
S23: BB Mean Rev + mom6 Combined Strategy
Ported from trader.py v3.
Signal: BB oversold/overbought OR 6h momentum > 3%
Gate: Runner-level vol gate (1.5%)
Params: TP=0.2%, SL=0.1%
"""
from .base import BaseStrategy, SignalResult


class BBMom6Strategy(BaseStrategy):
    min_vol_ratio = 0.0
    name = 'bb_mom6'
    strategy_type = 'mean_rev'
    description = 'BB(20,2.0) mean reversion + 6h momentum > 3%'

    BB_PERIOD = 20
    BB_STD = 2.0
    MOM6_THRESHOLD = 0.03
    TP_PCT = 0.002
    SL_PCT = 0.001

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        if not price:
            return None

        candles_1h = kwargs.get('candles_1h', [])
        if not candles_1h or len(candles_1h) < 60:
            return None

        closes = [float(c[4]) if isinstance(c, (list, tuple)) else float(c) for c in candles_1h]

        # RSI filter for better entry timing
        rsi_val = data.get('rsi', 50)
        if rsi_val and not (40 <= rsi_val <= 60):
            # Only trade when RSI is in neutral zone (not overbought/oversold)
            # This prevents buying at tops and selling at bottoms
            pass  # RSI filter disabled - BB already handles mean reversion

        # BB signal (primary)
        bb_dir, bb_info = self._check_bb(closes, price)
        if bb_dir:
            # RSI confirmation: buy when RSI <50, sell when RSI >50
            rsi_val = data.get('rsi', 50)
            if bb_dir == 'LONG' and rsi_val and rsi_val > 55:
                bb_dir = None  # skip - RSI too high for long
            elif bb_dir == 'SHORT' and rsi_val and rsi_val < 45:
                bb_dir = None  # skip - RSI too low for short
            if bb_dir:
                return self._make_signal(bb_dir, price, bb_info)

        # Mom6 signal (secondary)
        mom_dir, mom_info = self._check_mom6(closes, price)
        if mom_dir:
            return self._make_signal(mom_dir, price, mom_info)

        return None

    def _check_bb(self, closes, price):
        if len(closes) < self.BB_PERIOD + 1:
            return None, "bb_insufficient"
        seg = closes[-self.BB_PERIOD:]
        sma = sum(seg) / self.BB_PERIOD
        std = (sum((x - sma) ** 2 for x in seg) / self.BB_PERIOD) ** 0.5
        upper = sma + self.BB_STD * std
        lower = sma - self.BB_STD * std
        if price < lower:
            return "LONG", "BB_LONG price=%.2f<low=%.2f" % (price, lower)
        elif price > upper:
            return "SHORT", "BB_SHORT price=%.2f>up=%.2f" % (price, upper)
        return None, "BB_NEUTRAL %.2f<%.2f<%.2f" % (lower, price, upper)

    def _check_mom6(self, closes, price):
        if len(closes) < 7:
            return None, "mom6_insufficient"
        current = closes[-1]
        past = closes[-7]
        if past == 0:
            return None, "mom6_zero"
        mom = (current - past) / past
        if mom > self.MOM6_THRESHOLD:
            return "LONG", "mom6=%+.2f%%>3%%" % (mom * 100)
        elif mom < -self.MOM6_THRESHOLD:
            return "SHORT", "mom6=%+.2f%%<-3%%" % (mom * 100)
        return None, "mom6=%+.2f%%<3%%" % (mom * 100)

    def _make_signal(self, direction, price, reason):
        if direction == "LONG":
            entry = price * 1.001
            tp = entry * (1 + self.TP_PCT)
            sl = entry * (1 - self.SL_PCT)
        else:
            entry = price * 0.999
            tp = entry * (1 - self.TP_PCT)
            sl = entry * (1 + self.SL_PCT)
        sl_pct = abs(sl - entry) / entry * 100
        tp_pct = abs(tp - entry) / entry * 100
        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=0.75,
            entry=round(entry, 2), sl=round(sl, 2),
            tp1=round(tp, 2), tp2=round(tp, 2), tp3=round(tp, 2),
            sl_pct=round(sl_pct, 3), tp1_pct=round(tp_pct, 3),
            size_mult=1.0,
            reason="BB+mom6 -> %s: %s" % (direction, reason),
            bypass_gates=True,
            details={'bb_mom6': True},
        )
