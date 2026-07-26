"""S21: Trade Flow Momentum v3 — follow aggressive recent trade flow.

v2 → v3 CHANGES:
1. Z-score thresholds instead of static 0.60/0.40 — adapts to volatility
2. Flow acceleration — tracks if taker ratio is changing rapidly
3. EMA200 filter tightened from 2% to 1.5%
4. Session filter added (good/bad hours)
5. Fallback to computing taker flow from df_15m if kwargs missing
"""
from .base import BaseStrategy, SignalResult
import numpy as np

GOOD_HOURS = {0, 1, 2, 7, 8, 9, 10, 12, 13, 15, 16, 21}
BAD_HOURS = {4, 5, 6, 19, 20, 22, 23}


class TradeFlowStrategy(BaseStrategy):
    min_vol_ratio = 0.15
    name = 'trade_flow'
    strategy_type = 'flow'
    description = 'v3: z-score thresholds + acceleration + session filter + df_15m fallback'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        ema_200 = data.get('ema_200', 0)
        if not price or not atr:
            return None

        # ── SESSION FILTER (new) ──
        ts = data.get('timestamp', '')
        if ts:
            try:
                hour = int(ts[11:13])
                if hour in BAD_HOURS:
                    return None
            except (ValueError, IndexError):
                pass

        # ── GET TAKER DATA ──
        # Primary: from kwargs (pipeline)
        trade_data = kwargs.get('trade_flow', {})
        taker_ratio = trade_data.get('taker_ratio', None)
        net_flow = trade_data.get('net_flow', 0)
        large_buys = trade_data.get('large_buy_count', 0)
        large_sells = trade_data.get('large_sell_count', 0)

        # Fallback: compute from df_15m (new)
        if taker_ratio is None and df_15m is not None and idx is not None and idx >= 60:
            taker_base = df_15m['Taker buy base asset volume'].values.astype(float)
            volumes = df_15m['Volume'].values.astype(float)
            taker_ratios = taker_base / np.maximum(volumes, 1)
            taker_ratio = taker_ratios[idx]

            # Compute net flow from recent bars
            recent_taker = taker_ratios[max(0, idx-5):idx+1]
            recent_vol = volumes[max(0, idx-5):idx+1]
            net_flow = float(np.sum((recent_taker - 0.5) * recent_vol * price))

            # Compute large buys/sells from volume spikes
            vol_ma = np.mean(volumes[max(0, idx-20):idx+1])
            for i in range(max(0, idx-5), idx+1):
                if volumes[i] > vol_ma * 2:
                    if taker_ratios[i] > 0.6:
                        large_buys += 1
                    elif taker_ratios[i] < 0.4:
                        large_sells += 1

        if taker_ratio is None:
            return None

        # ── TREND ALIGNMENT FILTER (v4 upgrade) ──
        # Don't generate counter-trend signals when price is trending strongly
        if df_15m is not None and idx is not None and idx >= 20:
            closes = df_15m['Close'].values.astype(float)
            # 1h trend (4 bars of 15m)
            if idx >= 5:
                mom_1h = (closes[idx] - closes[idx-4]) / closes[idx-4]
            else:
                mom_1h = 0
            # 4h trend (16 bars)
            if idx >= 17:
                mom_4h = (closes[idx] - closes[idx-16]) / closes[idx-16]
            else:
                mom_4h = 0
            
            # If 1h momentum > 1.5%, only allow LONG signals (skip SHORT)
            if mom_1h > 0.015 and taker_ratio < 0.45:
                return None  # Skip SHORT in strong uptrend
            # If 1h momentum < -1.5%, only allow SHORT signals (skip LONG)
            if mom_1h < -0.015 and taker_ratio > 0.55:
                return None  # Skip LONG in strong downtrend
            # If 4h momentum > 3%, boost LONG conviction
            if mom_4h > 0.03:
                pass  # will boost below
            # If 4h momentum < -3%, boost SHORT conviction
            if mom_4h < -0.03:
                pass  # will boost below

        # ── COMPUTE Z-SCORE (new) ──
        # Need df_15m for z-score computation
        if df_15m is None or idx is None or idx < 60:
            return None

        taker_base = df_15m['Taker buy base asset volume'].values.astype(float)
        volumes = df_15m['Volume'].values.astype(float)
        taker_ratios = taker_base / np.maximum(volumes, 1)

        # 60-bar rolling z-score
        window = taker_ratios[max(0, idx-60):idx+1]
        if len(window) < 20:
            return None

        taker_mean = np.mean(window)
        taker_std = np.std(window)
        if taker_std < 0.01:
            return None

        taker_zscore = (taker_ratio - taker_mean) / taker_std

        # ── FLOW ACCELERATION (new) ──
        # Change in z-score over last 5 bars
        if idx >= 5:
            prev_window = taker_ratios[max(0, idx-60):idx-4]
            if len(prev_window) >= 20:
                prev_mean = np.mean(prev_window)
                prev_std = np.std(prev_window)
                if prev_std >= 0.01:
                    prev_zscore = (taker_ratios[idx-5] - prev_mean) / prev_std
                    acceleration = taker_zscore - prev_zscore
                else:
                    acceleration = 0
            else:
                acceleration = 0
        else:
            acceleration = 0

        # ── DIRECTION: z-score based (replaces static thresholds) ──
        if taker_zscore > 0.8:
            direction = 'LONG'
        elif taker_zscore < -0.8:
            direction = 'SHORT'
        else:
            return None

        # ── EMA200 TREND FILTER (tightened) ──
        if ema_200 and ema_200 > 0:
            dist = (price - ema_200) / ema_200
            if direction == 'LONG' and dist < -0.015:
                return None  # too far below EMA
            if direction == 'SHORT' and dist > 0.015:
                return None  # too far above EMA

        # ── CONVICTION ──
        base = 0.40

        # Z-score strength (stronger = more conviction)
        z_strength = min(abs(taker_zscore) / 3.0, 0.25)

        # Acceleration bonus (rapid change = stronger signal)
        accel_bonus = min(abs(acceleration) / 2.0, 0.15) if acceleration != 0 else 0

        # Net flow alignment
        flow_bonus = 0
        if direction == 'LONG' and net_flow > 0:
            flow_bonus = min(abs(net_flow) / 100000, 0.10)
        elif direction == 'SHORT' and net_flow < 0:
            flow_bonus = min(abs(net_flow) / 100000, 0.10)

        # Large trade alignment
        large_bonus = 0
        if direction == 'LONG' and large_buys > large_sells:
            large_bonus = min((large_buys - large_sells) * 0.03, 0.10)
        elif direction == 'SHORT' and large_sells > large_buys:
            large_bonus = min((large_sells - large_buys) * 0.03, 0.10)

        # Volume bonus
        vol_ratio = data.get('vol_ratio', 1.0) or 1.0
        vol_bonus = min((vol_ratio - 1.0) * 0.05, 0.10) if vol_ratio > 1.0 else 0

        conviction = min(base + z_strength + accel_bonus + flow_bonus + large_bonus + vol_bonus, 0.90)
        if conviction < 0.50:
            return None

        # ── TP/SL ──
        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.2)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.8,
            reason=f"Trade flow v3 {direction}: z={taker_zscore:.2f} accel={acceleration:.2f} "
                   f"net=${net_flow/1000:.0f}k vol={vol_ratio:.2f}",
            bypass_gates=True,
            details={
                'taker_zscore': float(taker_zscore), 'acceleration': float(acceleration),
                'taker_ratio': float(taker_ratio), 'net_flow': float(net_flow),
                'large_buys': large_buys, 'large_sells': large_sells,
                'vol_ratio': float(vol_ratio), 'version': 'v3',
            },
        )
