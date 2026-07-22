"""S2: Squeeze Breakout V7.1 — Dual Detection (BB + Donchian) with Tiered Boost.

V7.1 changes (2026-07-18):
- Dual squeeze detection: BB width < 3% OR Donchian width < 1%
- Track which detector fired: bb_only, donchian_only, or both
- Conviction boost when both fire simultaneously
- Group B confirmation only (no standalone trades)

Detection:
1. BB(20, 2.0) squeeze: bb_width < 3%
2. Donchian(20) squeeze: (20-bar high - 20-bar low) / price < 1%
3. Both = high conviction squeeze

Fade logic (unchanged from v7):
- Breakout above upper → SHORT (fade back to mean)
- Breakout below lower → LONG (fade back to mean)
- Trend filter: don't fade with-trend breakouts
- Volume filter: skip high-volume breakouts (likely real)
"""
from .base import BaseStrategy, SignalResult
import numpy as np


class SqueezeBreakoutStrategy(BaseStrategy):
    min_vol_ratio = 0.0
    name = 'squeeze_breakout'
    strategy_type = 'structure'
    description = 'v7.1: Dual detection BB+Donchian, Group B only'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        if df_15m is None or idx is None or idx < 60:
            return None

        closes = df_15m['Close'].values.astype(float)
        highs = df_15m['High'].values.astype(float)
        lows = df_15m['Low'].values.astype(float)
        volumes = df_15m['Volume'].values.astype(float)

        price = closes[idx]
        atr = data.get('atr', 0)
        if not price or not atr:
            return None

        # === DETECTOR 1: BB Squeeze ===
        lookback = closes[max(0, idx-19):idx+1]
        if len(lookback) < 20:
            return None
        sma20 = np.mean(lookback)
        std20 = np.std(lookback)
        if std20 == 0 or sma20 == 0:
            return None
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_width = (bb_upper - bb_lower) / sma20
        bb_squeeze = bb_width < 0.03

        # === DETECTOR 2: Donchian Squeeze ===
        dc_high = np.max(highs[max(0, idx-19):idx+1])
        dc_low = np.min(lows[max(0, idx-19):idx+1])
        dc_width = (dc_high - dc_low) / price
        donchian_squeeze = dc_width < 0.01

        # At least one detector must fire
        if not (bb_squeeze or donchian_squeeze):
            return None

        # Track which fired
        detectors = []
        if bb_squeeze:
            detectors.append('bb')
        if donchian_squeeze:
            detectors.append('donchian')
        detection_type = '_'.join(sorted(detectors))

        # === BREAKOUT CHECK ===
        # Use BB bands for breakout (more precise than Donchian)
        is_above = price > bb_upper
        is_below = price < bb_lower

        # Fallback to Donchian if BB didn't detect breakout
        if not (is_above or is_below):
            is_above = price > dc_high
            is_below = price < dc_low

        if not (is_above or is_below):
            return None

        # === EMA TREND FILTER ===
        if idx >= 50:
            ema_50 = np.mean(closes[max(0, idx-49):idx+1])
        else:
            ema_50 = sma20

        # === VOLUME FILTER ===
        avg_vol = np.mean(volumes[max(0, idx-19):idx+1])
        vol_ratio = volumes[idx] / avg_vol if avg_vol > 0 else 1

        # === FADE LOGIC ===
        if is_above:
            direction = 'SHORT'
            if price > ema_50:
                return None  # with-trend breakout, don't fade
            if vol_ratio > 1.5:
                return None  # high volume, likely real
            entry = price
            sl = price + atr * 1.0
            tp = price - atr * 1.5
        elif is_below:
            direction = 'LONG'
            if price < ema_50:
                return None
            if vol_ratio > 1.5:
                return None
            entry = price
            sl = price - atr * 1.0
            tp = price + atr * 1.5
        else:
            return None

        # RR check
        risk = abs(entry - sl)
        reward = abs(entry - tp)
        if risk == 0 or reward / risk < 1.0:
            return None

        # Conviction
        # Base from squeeze quality (normalized to respective thresholds)
        bb_quality = max(0, 1.0 - bb_width / 0.03) if bb_squeeze else 0
        dc_quality = max(0, 1.0 - dc_width / 0.01) if donchian_squeeze else 0

        if len(detectors) == 2:
            # Both fired: higher base conviction
            squeeze_quality = max(bb_quality, dc_quality)
            conviction = 0.60 + squeeze_quality * 0.15
        else:
            squeeze_quality = bb_quality if bb_squeeze else dc_quality
            conviction = 0.50 + squeeze_quality * 0.15

        if vol_ratio < 0.8:
            conviction += 0.1
        conviction = min(conviction, 0.85)

        if conviction < 0.50:
            return None

        sl_pct = abs(entry - sl) / entry * 100
        tp_pct = abs(tp - entry) / entry * 100

        return SignalResult(
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            direction=direction,
            conviction=conviction,
            entry=entry,
            sl=sl,
            tp1=tp,
            tp2=tp,
            tp3=tp,
            sl_pct=sl_pct,
            tp1_pct=tp_pct,
            size_mult=1.0,
            reason=f'SqueezeV7.1 FADE {direction} [{detection_type}] bb_w={bb_width*100:.2f}% dc_w={dc_width*100:.2f}% vol={vol_ratio:.1f}x',
            details={
                'bb_width': bb_width,
                'dc_width': dc_width,
                'vol_ratio': vol_ratio,
                'squeeze_quality': squeeze_quality,
                'ema_50': ema_50,
                'detection_type': detection_type,
                'bb_squeeze': bb_squeeze,
                'donchian_squeeze': donchian_squeeze,
                'version': 'v7.1_dual',
            },
        )
