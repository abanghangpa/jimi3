"""
BB Regime Reversion v3 — Weighted confirmations, strict entry rules.

Changes from v2:
- Minimum 3 confirmations (was 2)
- HIGH-QUALITY confirmation required (engulfing, divergence, or vol exhaustion)
- Weighted scoring: not all confirmations are equal
- Wider TP targeting (1.2x middle distance instead of exact middle)
- Tighter regime filter (no trades in MILDLY_BEARISH)
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


# Confirmation weights (higher = more valuable)
CONF_WEIGHTS = {
    "BEARISH_ENGULFING": 3.0,   # 58.3% WR in v2
    "BULLISH_ENGULFING": 3.0,   # 37.5% WR in v2
    "RSI_DIVERGENCE": 2.5,      # 32.1% WR in v2
    "VOL_EXHAUSTION": 2.0,      # 31.5% WR in v2
    "REJECTION_WICK": 1.5,      # 29.1% WR in v2
    "DOJI_AT_BAND": 1.0,        # 28.9% WR in v2
    "MULTIPLE_TOUCHES": 0.5,    # 24.1% WR in v2 (weak)
    "RSI_OVERBOUGHT": 0.3,      # 23.2% WR in v2 (noise)
    "RSI_OVERSOLD": 0.3,        # 21.8% WR in v2 (noise)
}

MIN_CONFIRMATIONS = 3
MIN_WEIGHTED_SCORE = 4.0  # Minimum weighted score to enter
REQUIRE_HIGH_QUALITY = True  # Must have at least 1 high-quality confirmation

HIGH_QUALITY = {"BEARISH_ENGULFING", "BULLISH_ENGULFING", "RSI_DIVERGENCE", "VOL_EXHAUSTION"}


class BBRegimeReversionV3:
    def __init__(self):
        self.name = "bb_regime_reversion_v3"
        self.bb_period = 20
        self.bb_std = 2.0

    def analyze(self, df_15m, idx, regime, regime_confidence=0.5):
        if idx < max(self.bb_period + 10, 50):
            return self._no_signal("Insufficient data")

        # No trades in STRESS or MILDLY_BEARISH
        if regime in ("STRESS", "MILDLY_BEARISH"):
            return self._no_signal(f"{regime}: no BB trades", {})

        close = df_15m['Close'].iloc[idx]
        high = df_15m['High'].iloc[idx]
        low = df_15m['Low'].iloc[idx]
        open_price = df_15m['Open'].iloc[idx]
        volume = df_15m['Volume'].iloc[idx] if 'Volume' in df_15m.columns else 0

        # BB
        sma = df_15m['Close'].rolling(self.bb_period).mean()
        std = df_15m['Close'].rolling(self.bb_period).std()
        upper = (sma + std * self.bb_std).iloc[idx]
        middle = sma.iloc[idx]
        lower = (sma - std * self.bb_std).iloc[idx]
        bb_width = (upper - lower) / middle if middle > 0 else 0

        # RSI
        rsi = self._calc_rsi(df_15m['Close'], idx, 14)

        # Volume
        avg_vol = df_15m['Volume'].iloc[max(0,idx-20):idx].mean() if 'Volume' in df_15m.columns else 0
        vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0

        prev_close = df_15m['Close'].iloc[idx - 1]
        prev_open = df_15m['Open'].iloc[idx - 1]

        bb_data = {
            "upper": round(upper, 2), "middle": round(middle, 2),
            "lower": round(lower, 2), "width": round(bb_width, 4),
            "close": round(close, 2), "rsi": round(rsi, 1),
            "vol_ratio": round(vol_ratio, 2),
        }

        touched_upper = high >= upper
        touched_lower = low <= lower
        if not touched_upper and not touched_lower:
            return self._no_signal("No BB touch", bb_data)

        if touched_upper:
            return self._eval_short(df_15m, idx, close, open_price, high, low,
                                     middle, upper, rsi, vol_ratio,
                                     prev_close, prev_open, regime, regime_confidence, bb_data)
        else:
            return self._eval_long(df_15m, idx, close, open_price, high, low,
                                    middle, lower, rsi, vol_ratio,
                                    prev_close, prev_open, regime, regime_confidence, bb_data)

    def _eval_short(self, df, idx, close, open_price, high, low,
                    middle, upper, rsi, vol_ratio, prev_close, prev_open,
                    regime, confidence, bb_data):
        confirmations = []
        weighted_score = 0.0

        # RSI overbought
        if rsi >= 60:
            confirmations.append({"type": "RSI_OVERBOUGHT", "value": rsi})
            weighted_score += CONF_WEIGHTS["RSI_OVERBOUGHT"]

        # Volume exhaustion
        if vol_ratio < 0.8:
            confirmations.append({"type": "VOL_EXHAUSTION", "value": vol_ratio})
            weighted_score += CONF_WEIGHTS["VOL_EXHAUSTION"]

        # Bearish rejection wick
        body = abs(close - open_price)
        upper_wick = high - max(close, open_price)
        if upper_wick > body * 1.5 and close < open_price:
            confirmations.append({"type": "REJECTION_WICK", "value": round(upper_wick / body, 1)})
            weighted_score += CONF_WEIGHTS["REJECTION_WICK"]

        # Bearish engulfing
        if close < open_price and prev_close > prev_open:
            if close < prev_open and open_price > prev_close:
                confirmations.append({"type": "BEARISH_ENGULFING", "value": 1.0})
                weighted_score += CONF_WEIGHTS["BEARISH_ENGULFING"]

        # Doji
        body_pct = abs(close - open_price) / open_price * 100 if open_price > 0 else 0
        if body_pct < 0.1 and (high - low) / low * 100 > 0.3:
            confirmations.append({"type": "DOJI_AT_BAND", "value": body_pct})
            weighted_score += CONF_WEIGHTS["DOJI_AT_BAND"]

        # RSI divergence
        if idx >= 5:
            prev_high_5 = max(df['High'].iloc[idx-5:idx])
            prev_rsi = self._calc_rsi(df['Close'], idx - 5, 14)
            if high > prev_high_5 and rsi < prev_rsi:
                confirmations.append({"type": "RSI_DIVERGENCE", "value": round(prev_rsi - rsi, 1)})
                weighted_score += CONF_WEIGHTS["RSI_DIVERGENCE"]

        # Multiple touches
        bars_at = sum(1 for i in range(max(0,idx-3), idx+1) if df['High'].iloc[i] >= upper * 0.998)
        if bars_at >= 3:
            confirmations.append({"type": "MULTIPLE_TOUCHES", "value": bars_at})
            weighted_score += CONF_WEIGHTS["MULTIPLE_TOUCHES"]

        # Trending reduces score
        if regime in ("BULL", "BEAR"):
            weighted_score *= 0.7

        # Check entry requirements
        conf_types = {c["type"] for c in confirmations}
        has_high_quality = bool(conf_types & HIGH_QUALITY)

        if len(confirmations) < MIN_CONFIRMATIONS:
            return self._no_signal(f"Only {len(confirmations)} confirmations (need {MIN_CONFIRMATIONS})", bb_data)
        if REQUIRE_HIGH_QUALITY and not has_high_quality:
            return self._no_signal(f"No high-quality confirmation (need one of {HIGH_QUALITY})", bb_data)
        if weighted_score < MIN_WEIGHTED_SCORE:
            return self._no_signal(f"Score {weighted_score:.1f} < {MIN_WEIGHTED_SCORE}", bb_data)

        # Entry/TP/SL
        entry = close
        # TP: slightly beyond middle (1.2x distance from entry to middle)
        tp_dist = abs(entry - middle)
        tp = entry - tp_dist * 1.2  # Short: TP below entry
        sl = upper + (upper - middle) * 0.2
        rr = abs(entry - tp) / abs(sl - entry) if abs(sl - entry) > 0 else 0

        if rr < 1.2:  # Higher R:R requirement
            return self._no_signal(f"R:R {rr:.2f} < 1.2", bb_data)

        conviction = min(0.95, confidence * 0.4 + weighted_score * 0.08)

        return {
            "signal": "SHORT", "conviction": round(conviction, 3),
            "entry": round(entry, 2), "tp": round(tp, 2), "sl": round(sl, 2),
            "reason": f"BB upper + {len(confirmations)} conf (score={weighted_score:.1f}): {[c['type'] for c in confirmations]}",
            "bb_data": bb_data, "confirmations": confirmations,
            "weighted_score": round(weighted_score, 2), "rr": round(rr, 2),
        }

    def _eval_long(self, df, idx, close, open_price, high, low,
                   middle, lower, rsi, vol_ratio, prev_close, prev_open,
                   regime, confidence, bb_data):
        confirmations = []
        weighted_score = 0.0

        if rsi <= 40:
            confirmations.append({"type": "RSI_OVERSOLD", "value": rsi})
            weighted_score += CONF_WEIGHTS["RSI_OVERSOLD"]

        if vol_ratio < 0.8:
            confirmations.append({"type": "VOL_EXHAUSTION", "value": vol_ratio})
            weighted_score += CONF_WEIGHTS["VOL_EXHAUSTION"]

        body = abs(close - open_price)
        lower_wick = min(close, open_price) - low
        if lower_wick > body * 1.5 and close > open_price:
            confirmations.append({"type": "REJECTION_WICK", "value": round(lower_wick / body, 1)})
            weighted_score += CONF_WEIGHTS["REJECTION_WICK"]

        if close > open_price and prev_close < prev_open:
            if close > prev_open and open_price < prev_close:
                confirmations.append({"type": "BULLISH_ENGULFING", "value": 1.0})
                weighted_score += CONF_WEIGHTS["BULLISH_ENGULFING"]

        body_pct = abs(close - open_price) / open_price * 100 if open_price > 0 else 0
        if body_pct < 0.1 and (high - low) / low * 100 > 0.3:
            confirmations.append({"type": "DOJI_AT_BAND", "value": body_pct})
            weighted_score += CONF_WEIGHTS["DOJI_AT_BAND"]

        if idx >= 5:
            prev_low_5 = min(df['Low'].iloc[idx-5:idx])
            prev_rsi = self._calc_rsi(df['Close'], idx - 5, 14)
            if low < prev_low_5 and rsi > prev_rsi:
                confirmations.append({"type": "RSI_DIVERGENCE", "value": round(rsi - prev_rsi, 1)})
                weighted_score += CONF_WEIGHTS["RSI_DIVERGENCE"]

        bars_at = sum(1 for i in range(max(0,idx-3), idx+1) if df['Low'].iloc[i] <= lower * 1.002)
        if bars_at >= 3:
            confirmations.append({"type": "MULTIPLE_TOUCHES", "value": bars_at})
            weighted_score += CONF_WEIGHTS["MULTIPLE_TOUCHES"]

        if regime in ("BULL", "BEAR"):
            weighted_score *= 0.7

        conf_types = {c["type"] for c in confirmations}
        has_high_quality = bool(conf_types & HIGH_QUALITY)

        if len(confirmations) < MIN_CONFIRMATIONS:
            return self._no_signal(f"Only {len(confirmations)} confirmations", bb_data)
        if REQUIRE_HIGH_QUALITY and not has_high_quality:
            return self._no_signal(f"No high-quality confirmation", bb_data)
        if weighted_score < MIN_WEIGHTED_SCORE:
            return self._no_signal(f"Score {weighted_score:.1f} < {MIN_WEIGHTED_SCORE}", bb_data)

        entry = close
        tp_dist = abs(entry - middle)
        tp = entry + tp_dist * 1.2
        sl = lower - (middle - lower) * 0.2
        rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 0 else 0

        if rr < 1.2:
            return self._no_signal(f"R:R {rr:.2f} < 1.2", bb_data)

        conviction = min(0.95, confidence * 0.4 + weighted_score * 0.08)

        return {
            "signal": "LONG", "conviction": round(conviction, 3),
            "entry": round(entry, 2), "tp": round(tp, 2), "sl": round(sl, 2),
            "reason": f"BB lower + {len(confirmations)} conf (score={weighted_score:.1f}): {[c['type'] for c in confirmations]}",
            "bb_data": bb_data, "confirmations": confirmations,
            "weighted_score": round(weighted_score, 2), "rr": round(rr, 2),
        }

    def _calc_rsi(self, series, idx, period=14):
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
        return {"signal": "NEUTRAL", "conviction": 0.0, "entry": 0, "tp": 0, "sl": 0,
                "reason": reason, "bb_data": bb_data or {}, "confirmations": []}
