"""S21: Trade Flow Momentum v4.1 — Two independent scenarios.

v4 → v4.1 CHANGES:
1. SCENARIO A: conv 0.70-0.80 + taker_flow LONG (24 trades, 73.9% WR, PF 4.67, p=0.0016)
2. SCENARIO B: conv 0.70-0.80 + LS>1.5 long-crowded (121 trades, 58.6% WR, PF 2.08, p=0.0001)
3. Either scenario can trigger — they are independent entry paths
4. Both require LONG direction + z-score > 0.8 + session/EMA filters

v4.1 stats:
- Scenario A: 24 trades, 73.9% WR, PF 4.67 (high quality, low frequency)
- Scenario B: 121 trades, 58.6% WR, PF 2.08 (moderate quality, higher frequency)
"""
from .base import BaseStrategy, SignalResult
import numpy as np

GOOD_HOURS = {0, 1, 2, 7, 8, 9, 10, 12, 13, 15, 16, 21}
BAD_HOURS = {4, 5, 6, 19, 20, 22, 23}

# ── CONFIG ──
CONV_MIN = 0.70
CONV_MAX = 0.80
LS_THRESHOLD = 1.5


def _check_taker_flow_confirms(data):
    """Scenario A: taker_flow fires LONG at same timestamp."""
    strategy_signals = data.get('strategy_signals', {})
    tf = strategy_signals.get('taker_flow', {})
    if tf.get('direction') == 'LONG' and tf.get('fired', False):
        return True
    multi = data.get('multi_strategy', {})
    for sig in multi.get('all_signals', []):
        if sig.get('strategy') == 'taker_flow' and sig.get('direction') == 'LONG':
            return True
    return False


def _check_ls_crowded(data):
    """Scenario B: LS ratio > 1.5 (long-crowded positioning)."""
    deriv = data.get('derivatives', {})
    ls = deriv.get('ls_ratio', 0)
    if ls and ls > LS_THRESHOLD:
        return True
    # Also check from strategy signals
    strategy_signals = data.get('strategy_signals', {})
    for strat_name in ['orderbook_imbalance', 'whale_watch', 'funding_arb']:
        sig = strategy_signals.get(strat_name, {})
        ls_sig = sig.get('ls_ratio', 0)
        if ls_sig and ls_sig > LS_THRESHOLD:
            return True
    return False


class TradeFlowStrategy(BaseStrategy):
    min_vol_ratio = 0.15
    name = 'trade_flow'
    strategy_type = 'flow'
    description = 'v4.1: TWO scenarios — A: taker_flow confirm | B: LS>1.5 long-crowded'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        ema_200 = data.get('ema_200', 0)
        if not price or not atr:
            return None

        # ── SESSION FILTER ──
        ts = data.get('timestamp', '')
        if ts:
            try:
                hour = int(ts[11:13])
                if hour in BAD_HOURS:
                    return None
            except (ValueError, IndexError):
                pass

        # ── GET TAKER DATA ──
        trade_data = kwargs.get('trade_flow', {})
        taker_ratio = trade_data.get('taker_ratio', None)
        net_flow = trade_data.get('net_flow', 0)
        large_buys = trade_data.get('large_buy_count', 0)
        large_sells = trade_data.get('large_sell_count', 0)

        if taker_ratio is None and df_15m is not None and idx is not None and idx >= 60:
            taker_base = df_15m['Taker buy base asset volume'].values.astype(float)
            volumes = df_15m['Volume'].values.astype(float)
            taker_ratios = taker_base / np.maximum(volumes, 1)
            taker_ratio = taker_ratios[idx]
            recent_taker = taker_ratios[max(0, idx-5):idx+1]
            recent_vol = volumes[max(0, idx-5):idx+1]
            net_flow = float(np.sum((recent_taker - 0.5) * recent_vol * price))
            vol_ma = np.mean(volumes[max(0, idx-20):idx+1])
            for i in range(max(0, idx-5), idx+1):
                if volumes[i] > vol_ma * 2:
                    if taker_ratios[i] > 0.6:
                        large_buys += 1
                    elif taker_ratios[i] < 0.4:
                        large_sells += 1

        if taker_ratio is None:
            return None

        # ── TREND ALIGNMENT ──
        if df_15m is not None and idx is not None and idx >= 20:
            closes = df_15m['Close'].values.astype(float)
            if idx >= 5:
                mom_1h = (closes[idx] - closes[idx-4]) / closes[idx-4]
            else:
                mom_1h = 0
            if mom_1h < -0.015 and taker_ratio > 0.55:
                return None

        # ── COMPUTE Z-SCORE ──
        if df_15m is None or idx is None or idx < 60:
            return None

        taker_base = df_15m['Taker buy base asset volume'].values.astype(float)
        volumes = df_15m['Volume'].values.astype(float)
        taker_ratios = taker_base / np.maximum(volumes, 1)
        window = taker_ratios[max(0, idx-60):idx+1]
        if len(window) < 20:
            return None
        taker_mean = np.mean(window)
        taker_std = np.std(window)
        if taker_std < 0.01:
            return None
        taker_zscore = (taker_ratio - taker_mean) / taker_std

        # ── FLOW ACCELERATION ──
        acceleration = 0
        if idx >= 5:
            prev_window = taker_ratios[max(0, idx-60):idx-4]
            if len(prev_window) >= 20:
                prev_mean = np.mean(prev_window)
                prev_std = np.std(prev_window)
                if prev_std >= 0.01:
                    prev_zscore = (taker_ratios[idx-5] - prev_mean) / prev_std
                    acceleration = taker_zscore - prev_zscore

        # ── DIRECTION: LONG-ONLY ──
        if taker_zscore <= 0.8:
            return None

        # ── EMA200 FILTER ──
        if ema_200 and ema_200 > 0:
            dist = (price - ema_200) / ema_200
            if dist < -0.015:
                return None

        # ── CONVICTION ──
        base = 0.40
        z_strength = min(abs(taker_zscore) / 3.0, 0.25)
        accel_bonus = min(abs(acceleration) / 2.0, 0.15) if acceleration != 0 else 0
        flow_bonus = min(abs(net_flow) / 100000, 0.10) if net_flow > 0 else 0
        large_bonus = min((large_buys - large_sells) * 0.03, 0.10) if large_buys > large_sells else 0
        vol_ratio = data.get('vol_ratio', 1.0) or 1.0
        vol_bonus = min((vol_ratio - 1.0) * 0.05, 0.10) if vol_ratio > 1.0 else 0

        conviction = min(base + z_strength + accel_bonus + flow_bonus + large_bonus + vol_bonus, 0.90)

        if conviction < CONV_MIN or conviction >= CONV_MAX:
            return None

        # ── SCENARIO CHECK ──
        # Scenario A: taker_flow LONG confirmation
        scenario_a = _check_taker_flow_confirms(data)
        # Scenario B: LS>1.5 (long-crowded)
        scenario_b = _check_ls_crowded(data)

        if not scenario_a and not scenario_b:
            return None

        # Determine which scenario fired
        if scenario_a and scenario_b:
            scenario = 'A+B'
            size_mult = 0.7  # Both confirm → higher size
        elif scenario_a:
            scenario = 'A'
            size_mult = 0.5  # taker_flow only
        else:
            scenario = 'B'
            size_mult = 0.5  # LS only

        # ── TP/SL ──
        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, 'LONG', atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.2)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction='LONG', conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=size_mult,
            reason=f"Trade flow v4.1 LONG [{scenario}]: z={taker_zscore:.2f} "
                   f"accel={acceleration:.2f} net=${net_flow/1000:.0f}k",
            bypass_gates=True,
            details={
                'version': 'v4.1',
                'scenario': scenario,
                'scenario_a': scenario_a,
                'scenario_b': scenario_b,
                'taker_zscore': float(taker_zscore),
                'acceleration': float(acceleration),
                'taker_ratio': float(taker_ratio),
                'net_flow': float(net_flow),
                'large_buys': large_buys,
                'large_sells': large_sells,
                'vol_ratio': float(vol_ratio),
                'conv_min': CONV_MIN,
                'conv_max': CONV_MAX,
                'ls_threshold': LS_THRESHOLD,
                'note': 'Two scenarios: A=taker_flow (73.9%WR,PF4.67) | B=LS>1.5 (58.6%WR,PF2.08)',
            },
        )
