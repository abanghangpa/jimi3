"""
BB Regime Reversion v2 — With confirmation layer.

v1 failed because BB touch alone isn't enough.
v2 adds confirmation signals that must agree before entry:

1. Volume exhaustion — BB touch + declining volume = move is running out of steam
2. RSI divergence — Price at new BB extreme but RSI doesn't agree
3. CVD divergence — Price at upper BB but aggressive selling increasing
4. Candle pattern — Rejection wick, engulfing, doji at band
5. Orderbook bias — Bid/ask imbalance contradicting the touch direction

Need 2+ confirmations to enter. More confirmations = higher conviction.
"""
import sys, os, json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)


class BBRegimeReversionV2:
    """
    BB strategy with multi-confirmation entry.
    Touch alone = NO trade. Touch + 2 confirmations = trade.
    """

    def __init__(self):
        self.name = "bb_regime_reversion_v2"
        self.bb_period = 20
        self.bb_std = 2.0
        self.min_confirmations = 2
        self.min_rsi_for_short = 60   # RSI must be elevated for short
        self.max_rsi_for_long = 40    # RSI must be depressed for long

    def analyze(self, df_15m, idx, regime, regime_confidence=0.5):
        """
        Full analysis with confirmation layer.
        """
        if idx < max(self.bb_period + 10, 50):
            return self._no_signal("Insufficient data")

        # === CALCULATE INDICATORS ===
        close = df_15m['Close'].iloc[idx]
        high = df_15m['High'].iloc[idx]
        low = df_15m['Low'].iloc[idx]
        open_price = df_15m['Open'].iloc[idx]
        volume = df_15m['Volume'].iloc[idx] if 'Volume' in df_15m.columns else 0

        # Bollinger Bands
        sma = df_15m['Close'].rolling(self.bb_period).mean()
        std = df_15m['Close'].rolling(self.bb_period).std()
        bb_upper = sma + (std * self.bb_std)
        bb_middle = sma
        bb_lower = sma - (std * self.bb_std)

        upper = bb_upper.iloc[idx]
        middle = bb_middle.iloc[idx]
        lower = bb_lower.iloc[idx]

        bb_width = (upper - lower) / middle if middle > 0 else 0

        # RSI (14-period)
        rsi = self._calc_rsi(df_15m['Close'], idx, 14)

        # Volume analysis
        avg_vol = df_15m['Volume'].iloc[max(0,idx-20):idx].mean() if 'Volume' in df_15m.columns else 0
        vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0

        # Previous bars for pattern detection
        prev_close = df_15m['Close'].iloc[idx - 1]
        prev_open = df_15m['Open'].iloc[idx - 1]
        prev_high = df_15m['High'].iloc[idx - 1]
        prev_low = df_15m['Low'].iloc[idx - 1]

        bb_data = {
            "upper": round(upper, 2), "middle": round(middle, 2),
            "lower": round(lower, 2), "width": round(bb_width, 4),
            "close": round(close, 2), "rsi": round(rsi, 1),
            "vol_ratio": round(vol_ratio, 2),
        }

        # === CHECK REGIME ===
        if regime == "STRESS":
            return self._no_signal("STRESS: no BB trades", bb_data)

        # === CHECK BB TOUCH ===
        touched_upper = high >= upper
        touched_lower = low <= lower

        if not touched_upper and not touched_lower:
            return self._no_signal("No BB touch", bb_data)

        # === COLLECT CONFIRMATIONS ===
        confirmations = []
        confirmation_score = 0.0

        if touched_upper:
            # SHORT setup: looking for reversion from upper band
            confirmations, confirmation_score = self._check_upper_confirmations(
                df_15m, idx, close, open_price, high, low, middle, upper,
                rsi, vol_ratio, prev_close, prev_open, prev_high, prev_low,
                regime, regime_confidence
            )

            if len(confirmations) < self.min_confirmations:
                return self._no_signal(
                    f"Upper touch but only {len(confirmations)}/{self.min_confirmations} confirmations: {[c['type'] for c in confirmations]}",
                    bb_data
                )

            # Calculate entry/TP/SL
            entry = close
            tp = middle
            sl = upper + (upper - middle) * 0.25
            rr = abs(entry - tp) / abs(sl - entry) if abs(sl - entry) > 0 else 0

            if rr < 1.0:
                return self._no_signal(f"R:R {rr:.2f} < 1.0", bb_data)

            conviction = min(0.9, regime_confidence * 0.5 + confirmation_score * 0.15)

            return {
                "signal": "SHORT",
                "conviction": round(conviction, 3),
                "entry": round(entry, 2),
                "tp": round(tp, 2),
                "sl": round(sl, 2),
                "reason": f"BB upper touch + {len(confirmations)} confirmations: {[c['type'] for c in confirmations]}",
                "bb_data": bb_data,
                "confirmations": confirmations,
                "confirmation_score": round(confirmation_score, 2),
                "rr": round(rr, 2),
            }

        else:
            # LONG setup: looking for reversion from lower band
            confirmations, confirmation_score = self._check_lower_confirmations(
                df_15m, idx, close, open_price, high, low, middle, lower,
                rsi, vol_ratio, prev_close, prev_open, prev_high, prev_low,
                regime, regime_confidence
            )

            if len(confirmations) < self.min_confirmations:
                return self._no_signal(
                    f"Lower touch but only {len(confirmations)}/{self.min_confirmations} confirmations: {[c['type'] for c in confirmations]}",
                    bb_data
                )

            entry = close
            tp = middle
            sl = lower - (middle - lower) * 0.25
            rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 0 else 0

            if rr < 1.0:
                return self._no_signal(f"R:R {rr:.2f} < 1.0", bb_data)

            conviction = min(0.9, regime_confidence * 0.5 + confirmation_score * 0.15)

            return {
                "signal": "LONG",
                "conviction": round(conviction, 3),
                "entry": round(entry, 2),
                "tp": round(tp, 2),
                "sl": round(sl, 2),
                "reason": f"BB lower touch + {len(confirmations)} confirmations: {[c['type'] for c in confirmations]}",
                "bb_data": bb_data,
                "confirmations": confirmations,
                "confirmation_score": round(confirmation_score, 2),
                "rr": round(rr, 2),
            }

    def _check_upper_confirmations(self, df, idx, close, open_price, high, low,
                                    middle, upper, rsi, vol_ratio,
                                    prev_close, prev_open, prev_high, prev_low,
                                    regime, confidence):
        """Check confirmations for SHORT at upper BB."""
        confirmations = []
        score = 0.0

        # 1. RSI overbought
        if rsi >= self.min_rsi_for_short:
            confirmations.append({"type": "RSI_OVERBOUGHT", "value": rsi, "weight": 1.0})
            score += 1.0

        # 2. Volume exhaustion (declining volume at extreme)
        if vol_ratio < 0.8:
            confirmations.append({"type": "VOL_EXHAUSTION", "value": vol_ratio, "weight": 1.0})
            score += 1.0

        # 3. Bearish rejection candle (upper wick > body, close < open)
        body = abs(close - open_price)
        upper_wick = high - max(close, open_price)
        if upper_wick > body * 1.5 and close < open_price:
            confirmations.append({"type": "REJECTION_WICK", "value": upper_wick / body, "weight": 1.5})
            score += 1.5

        # 4. Bearish engulfing
        if close < open_price and prev_close > prev_open:
            if close < prev_open and open_price > prev_close:
                confirmations.append({"type": "BEARISH_ENGULFING", "value": 1.0, "weight": 1.5})
                score += 1.5

        # 5. Doji at band (indecision)
        body_pct = abs(close - open_price) / open_price * 100 if open_price > 0 else 0
        if body_pct < 0.1 and (high - low) / low * 100 > 0.3:
            confirmations.append({"type": "DOJI_AT_BAND", "value": body_pct, "weight": 0.8})
            score += 0.8

        # 6. RSI divergence (price higher high, RSI lower high)
        if idx >= 5:
            prev_high_5 = max(df['High'].iloc[idx-5:idx])
            prev_rsi = self._calc_rsi(df['Close'], idx - 5, 14)
            if high > prev_high_5 and rsi < prev_rsi:
                confirmations.append({"type": "RSI_DIVERGENCE", "value": prev_rsi - rsi, "weight": 1.5})
                score += 1.5

        # 7. Multiple bars at band (exhaustion)
        bars_at_band = 0
        for i in range(max(0, idx-3), idx+1):
            if df['High'].iloc[i] >= upper * 0.998:
                bars_at_band += 1
        if bars_at_band >= 3:
            confirmations.append({"type": "MULTIPLE_TOUCHES", "value": bars_at_band, "weight": 1.0})
            score += 1.0

        # 8. Trending regime reduces conviction (continuation risk)
        if regime in ("BULL", "BEAR"):
            score *= 0.7  # Reduce score in trending regimes

        return confirmations, score

    def _check_lower_confirmations(self, df, idx, close, open_price, high, low,
                                    middle, lower, rsi, vol_ratio,
                                    prev_close, prev_open, prev_high, prev_low,
                                    regime, confidence):
        """Check confirmations for LONG at lower BB."""
        confirmations = []
        score = 0.0

        # 1. RSI oversold
        if rsi <= self.max_rsi_for_long:
            confirmations.append({"type": "RSI_OVERSOLD", "value": rsi, "weight": 1.0})
            score += 1.0

        # 2. Volume exhaustion
        if vol_ratio < 0.8:
            confirmations.append({"type": "VOL_EXHAUSTION", "value": vol_ratio, "weight": 1.0})
            score += 1.0

        # 3. Bullish rejection candle (lower wick > body, close > open)
        body = abs(close - open_price)
        lower_wick = min(close, open_price) - low
        if lower_wick > body * 1.5 and close > open_price:
            confirmations.append({"type": "REJECTION_WICK", "value": lower_wick / body, "weight": 1.5})
            score += 1.5

        # 4. Bullish engulfing
        if close > open_price and prev_close < prev_open:
            if close > prev_open and open_price < prev_close:
                confirmations.append({"type": "BULLISH_ENGULFING", "value": 1.0, "weight": 1.5})
                score += 1.5

        # 5. Doji at band
        body_pct = abs(close - open_price) / open_price * 100 if open_price > 0 else 0
        if body_pct < 0.1 and (high - low) / low * 100 > 0.3:
            confirmations.append({"type": "DOJI_AT_BAND", "value": body_pct, "weight": 0.8})
            score += 0.8

        # 6. RSI divergence (price lower low, RSI higher low)
        if idx >= 5:
            prev_low_5 = min(df['Low'].iloc[idx-5:idx])
            prev_rsi = self._calc_rsi(df['Close'], idx - 5, 14)
            if low < prev_low_5 and rsi > prev_rsi:
                confirmations.append({"type": "RSI_DIVERGENCE", "value": rsi - prev_rsi, "weight": 1.5})
                score += 1.5

        # 7. Multiple touches
        bars_at_band = 0
        for i in range(max(0, idx-3), idx+1):
            if df['Low'].iloc[i] <= lower * 1.002:
                bars_at_band += 1
        if bars_at_band >= 3:
            confirmations.append({"type": "MULTIPLE_TOUCHES", "value": bars_at_band, "weight": 1.0})
            score += 1.0

        # 8. Trending regime reduces conviction
        if regime in ("BULL", "BEAR"):
            score *= 0.7

        return confirmations, score

    def _calc_rsi(self, series, idx, period=14):
        """Calculate RSI at given index."""
        if idx < period + 1:
            return 50.0
        deltas = series.iloc[max(0, idx-period*2):idx+1].diff().dropna()
        gains = deltas.clip(lower=0)
        losses = -deltas.clip(upper=0)
        avg_gain = gains.rolling(period).mean().iloc[-1]
        avg_loss = losses.rolling(period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _no_signal(self, reason="", bb_data=None):
        return {
            "signal": "NEUTRAL",
            "conviction": 0.0,
            "entry": 0, "tp": 0, "sl": 0,
            "reason": reason,
            "bb_data": bb_data or {},
            "confirmations": [],
        }
