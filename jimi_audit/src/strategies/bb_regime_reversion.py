"""
BB Regime Reversion — Bollinger Band strategy with regime-aware interpretation.

Core insight (from user observation):
- In RANGING: BB touch → mean reversion to middle line (20 SMA)
- In TRENDING: BB touch → continuation signal, NOT a trade
- The middle line (20 SMA) is the directional pivot:
  - Uptrend: middle line = support on pullbacks
  - Downtrend: middle line = resistance on bounces
- Price can ride the upper/lower band multiple times in a trend
- The REAL signal is: band touch + regime confirmation + middle line behavior

This replaces the killed bb_mom6 which tested "BB touch → mean reversion" uniformly.
"""
import json, os, sys
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)


class BBRegimeReversion:
    """
    Regime-aware Bollinger Band strategy.
    
    Rules:
    1. RANGING regime:
       - Touch upper BB + bearish candle → SHORT targeting middle line
       - Touch lower BB + bullish candle → LONG targeting middle line
       - Middle line is the TP (mean reversion)
    
    2. TRENDING regime (BULL/BEAR):
       - DO NOT trade BB touches (continuation, not reversal)
       - Instead: watch for middle line BREAK as trend reversal signal
       - Price breaks below 20 SMA after riding upper band → potential trend end
       - Price breaks above 20 SMA after riding lower band → potential trend end
    
    3. STRESS regime:
       - No BB trades (volatility too high, bands expand, signals unreliable)
    """

    def __init__(self):
        self.name = "bb_regime_reversion"
        self.bb_period = 20
        self.bb_std = 2.0
        self.min_bars_at_band = 2  # Price must touch band for 2+ candles
        self.middle_line_tolerance = 0.002  # 0.2% tolerance for middle line

    def analyze(self, df_15m, idx, regime, regime_confidence=0.5):
        """
        Analyze BB conditions and return signal if applicable.
        
        Args:
            df_15m: DataFrame with OHLCV + BB data
            idx: Current bar index
            regime: Current market regime (BULL, BEAR, RANGING, STRESS, MILDLY_BEARISH)
            regime_confidence: Confidence in regime classification
            
        Returns:
            dict: {
                "signal": "LONG"|"SHORT"|"NEUTRAL",
                "conviction": 0.0-1.0,
                "entry": float,
                "tp": float,
                "sl": float,
                "reason": str,
                "bb_data": dict
            }
        """
        if idx < self.bb_period + 5:
            return self._no_signal("Insufficient data")

        # Get BB values
        close = df_15m['Close'].iloc[idx]
        high = df_15m['High'].iloc[idx]
        low = df_15m['Low'].iloc[idx]
        open_price = df_15m['Open'].iloc[idx]

        # Calculate BB if not already in dataframe
        bb_upper = df_15m.get('BB_upper')
        bb_middle = df_15m.get('BB_middle')
        bb_lower = df_15m.get('BB_lower')

        if bb_upper is None or bb_middle is None or bb_lower is None:
            sma = df_15m['Close'].rolling(self.bb_period).mean()
            std = df_15m['Close'].rolling(self.bb_period).std()
            bb_upper = sma + (std * self.bb_std)
            bb_middle = sma
            bb_lower = sma - (std * self.bb_std)

        upper = bb_upper.iloc[idx]
        middle = bb_middle.iloc[idx]
        lower = bb_lower.iloc[idx]
        prev_close = df_15m['Close'].iloc[idx - 1]

        # BB width (volatility indicator)
        bb_width = (upper - lower) / middle if middle > 0 else 0

        bb_data = {
            "upper": round(upper, 2),
            "middle": round(middle, 2),
            "lower": round(lower, 2),
            "width": round(bb_width, 4),
            "close": round(close, 2),
        }

        # ========================================
        # REGIME: STRESS — No BB trades
        # ========================================
        if regime == "STRESS":
            return self._no_signal("STRESS regime: BB unreliable", bb_data)

        # ========================================
        # REGIME: TRENDING (BULL/BEAR) — Continuation, not reversal
        # ========================================
        if regime in ("BULL", "BEAR"):
            return self._analyze_trending(df_15m, idx, regime, upper, middle, lower,
                                          close, high, low, open_price, bb_data, regime_confidence)

        # ========================================
        # REGIME: RANGING — Mean reversion
        # ========================================
        if regime == "RANGING":
            return self._analyze_ranging(df_15m, idx, upper, middle, lower,
                                         close, high, low, open_price, bb_data, regime_confidence)

        # ========================================
        # REGIME: MILDLY_BEARISH — Conservative reversion
        # ========================================
        if regime == "MILDLY_BEARISH":
            return self._analyze_ranging(df_15m, idx, upper, middle, lower,
                                         close, high, low, open_price, bb_data, regime_confidence * 0.8)

        return self._no_signal(f"Unknown regime: {regime}", bb_data)

    def _analyze_ranging(self, df_15m, idx, upper, middle, lower,
                         close, high, low, open_price, bb_data, confidence):
        """
        RANGING regime: BB mean reversion.
        Touch upper → SHORT targeting middle.
        Touch lower → LONG targeting middle.
        """
        # Check if price touched upper BB
        touched_upper = high >= upper
        # Check if price touched lower BB
        touched_lower = low <= lower

        # Bearish candle at upper band (close < open, near upper)
        bearish_at_upper = (touched_upper and close < open_price and
                           close > upper * 0.995)

        # Bullish candle at lower band (close > open, near lower)
        bullish_at_lower = (touched_lower and close > open_price and
                           close < lower * 1.005)

        if bearish_at_upper:
            # SHORT: upper touch → revert to middle
            entry = close
            tp = middle
            sl = upper + (upper - middle) * 0.3  # SL above upper band
            rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 0 else 0

            if rr < 1.0:
                return self._no_signal(f"R:R {rr:.2f} < 1.0", bb_data)

            conviction = min(0.8, confidence * 0.7)
            # Boost if price has been at band for multiple bars
            bars_at_band = self._count_bars_at_band(df_15m, idx, upper, "upper")
            if bars_at_band >= self.min_bars_at_band:
                conviction = min(0.9, conviction + 0.1)

            return {
                "signal": "SHORT",
                "conviction": round(conviction, 3),
                "entry": round(entry, 2),
                "tp": round(tp, 2),
                "sl": round(sl, 2),
                "reason": f"RANGING: BB upper touch, bearish candle, targeting middle ({middle:.2f})",
                "bb_data": bb_data,
                "bars_at_band": bars_at_band,
            }

        if bullish_at_lower:
            # LONG: lower touch → revert to middle
            entry = close
            tp = middle
            sl = lower - (middle - lower) * 0.3  # SL below lower band
            rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 0 else 0

            if rr < 1.0:
                return self._no_signal(f"R:R {rr:.2f} < 1.0", bb_data)

            conviction = min(0.8, confidence * 0.7)
            bars_at_band = self._count_bars_at_band(df_15m, idx, lower, "lower")
            if bars_at_band >= self.min_bars_at_band:
                conviction = min(0.9, conviction + 0.1)

            return {
                "signal": "LONG",
                "conviction": round(conviction, 3),
                "entry": round(entry, 2),
                "tp": round(tp, 2),
                "sl": round(sl, 2),
                "reason": f"RANGING: BB lower touch, bullish candle, targeting middle ({middle:.2f})",
                "bb_data": bb_data,
                "bars_at_band": bars_at_band,
            }

        return self._no_signal("RANGING: No BB touch", bb_data)

    def _analyze_trending(self, df_15m, idx, regime, upper, middle, lower,
                          close, high, low, open_price, bb_data, confidence):
        """
        TRENDING regime: BB touches are continuation, not reversal.
        DO NOT trade the touch. Instead, watch for middle line break as
        potential trend exhaustion signal.
        """
        # Check if price was riding the band (multiple touches in recent bars)
        riding_upper = self._count_bars_at_band(df_15m, idx, upper, "upper", lookback=10)
        riding_lower = self._count_bars_at_band(df_15m, idx, lower, "lower", lookback=10)

        # Check for middle line break after band ride
        prev_close = df_15m['Close'].iloc[idx - 1]

        # Price broke below middle after riding upper band (potential trend end)
        if riding_upper >= 3 and prev_close > middle and close < middle:
            # This is a TREND REVERSAL signal, not a BB reversion
            entry = close
            if regime == "BULL":
                # Was in uptrend, broke below middle → potential reversal SHORT
                tp = lower
                sl = upper
                rr = abs(entry - tp) / abs(sl - entry) if abs(sl - entry) > 0 else 0
                if rr >= 1.0:
                    return {
                        "signal": "SHORT",
                        "conviction": round(confidence * 0.6, 3),  # Lower conviction for reversal
                        "entry": round(entry, 2),
                        "tp": round(tp, 2),
                        "sl": round(sl, 2),
                        "reason": f"BULL trend exhaustion: {riding_upper} upper band touches, broke below middle ({middle:.2f})",
                        "bb_data": bb_data,
                        "bars_at_band": riding_upper,
                        "signal_type": "trend_reversal",
                    }

        # Price broke above middle after riding lower band (potential trend end)
        if riding_lower >= 3 and prev_close < middle and close > middle:
            if regime == "BEAR":
                # Was in downtrend, broke above middle → potential reversal LONG
                entry = close
                tp = upper
                sl = lower
                rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 0 else 0
                if rr >= 1.0:
                    return {
                        "signal": "LONG",
                        "conviction": round(confidence * 0.6, 3),
                        "entry": round(entry, 2),
                        "tp": round(tp, 2),
                        "sl": round(sl, 2),
                        "reason": f"BEAR trend exhaustion: {riding_lower} lower band touches, broke above middle ({middle:.2f})",
                        "bb_data": bb_data,
                        "bars_at_band": riding_lower,
                        "signal_type": "trend_reversal",
                    }

        # In trending regime: BB touches are NOT signals
        touched_upper = high >= upper
        touched_lower = low <= lower
        if touched_upper or touched_lower:
            return self._no_signal(
                f"{regime}: BB touch is continuation (not reversal), riding upper={riding_upper} lower={riding_lower}",
                bb_data
            )

        return self._no_signal(f"{regime}: No BB conditions met", bb_data)

    def _count_bars_at_band(self, df_15m, idx, band_level, band_type, lookback=5):
        """Count how many recent bars touched the band."""
        count = 0
        for i in range(max(0, idx - lookback), idx + 1):
            if band_type == "upper" and df_15m['High'].iloc[i] >= band_level:
                count += 1
            elif band_type == "lower" and df_15m['Low'].iloc[i] <= band_level:
                count += 1
        return count

    def _no_signal(self, reason="", bb_data=None):
        return {
            "signal": "NEUTRAL",
            "conviction": 0.0,
            "entry": 0,
            "tp": 0,
            "sl": 0,
            "reason": reason,
            "bb_data": bb_data or {},
        }
