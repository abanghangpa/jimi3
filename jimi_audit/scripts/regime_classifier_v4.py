#!/usr/bin/env python3
"""
RegimeClassifierV4 — Daily/Weekly timeframe regime detection.

Core insight (user feedback): Regimes are macro-structural conditions that
play out over days/weeks. Classifying from 15m data is using a microscope
to see the weather.

Architecture:
┌──────────────────────────────────────────────────────┐
│ DAILY/WEEKLY layer → Regime (BULL/BEAR/RANGING/STRESS)│
│   Input: 1D candles, 1W candles, daily derivatives    │
│   Output: regime + confidence (updates 1x/day)        │
├──────────────────────────────────────────────────────┤
│ 15 MINUTE layer → Signals + Execution                 │
│   Input: 15m scanner, order flow, microstructure      │
│   Gate: filtered by daily regime                       │
│   Output: trades (fast, within regime constraints)     │
└──────────────────────────────────────────────────────┘

References:
- Shu et al. (2024) Princeton — daily returns for regime detection
- Hamilton (1989) — original regime-switching on monthly data
- Ang & Timmermann (2012) — regime changes in financial markets
"""

import json, os, sys, time, math
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RegimeClassifierV4:
    """
    Daily/Weekly regime classifier.

    Features (all computed from daily+ data):
    ┌────────────────────────────────────────────────┐
    │ Trend: EMA50/EMA200 daily, EMA slope, HH/HL    │
    │ Momentum: Daily RSI, weekly ROC                 │
    │ Volatility: ATR regime, Bollinger width         │
    │ Derivatives: Daily avg FR, LS, OI trend         │
    │ Structure: Higher highs/lows, support/resistance │
    └────────────────────────────────────────────────┘

    Regime transitions:
    - Only evaluated on daily close (not intraday)
    - Minimum 3 daily bars before considering change
    - Hysteresis: 3 consecutive daily signals required
    - Cooldown: 5 days minimum between transitions
    """

    # === Timeframe parameters ===
    EMA_FAST = 10       # Fast EMA (daily)
    EMA_SLOW = 50       # Slow EMA (daily)
    EMA_TREND = 50      # Trend EMA (daily)
    RSI_PERIOD = 14     # RSI lookback
    ATR_PERIOD = 14     # ATR lookback

    # === Regime thresholds ===
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    TREND_SLOPE_THRESHOLD = 0.001  # 0.1% daily slope = trending
    VOL_HIGH_ATR_MULT = 1.5       # ATR > 1.5x average = high vol
    VOL_LOW_ATR_MULT = 0.7        # ATR < 0.7x average = low vol

    # === Persistence ===
    MIN_BARS_BEFORE_CHANGE = 3     # Need 3 daily bars in current regime
    HYSTERESIS_BARS = 3            # 3 consecutive daily signals for change
    COOLDOWN_DAYS = 5              # Min days between transitions

    def __init__(self):
        self.regime = "RANGING"
        self.confidence = 0.5
        self.signals = {}
        self._regime_start_ts = None
        self._last_transition_ts = None
        self._consecutive_votes = deque(maxlen=10)
        self._daily_cache = {}  # cache daily computations

    def classify(self, daily_candles, weekly_candles=None, deriv_daily=None, current_ts=None):
        """
        Classify regime from daily+ data.

        Args:
            daily_candles: list of dicts with {ts, open, high, low, close, volume}
                          sorted oldest→newest, at least 60 bars
            weekly_candles: list of dicts (optional, for weekly trend)
            deriv_daily: dict with {fr_avg, ls_avg, oi_trend, taker_avg} (daily aggregates)
            current_ts: current timestamp (epoch seconds)

        Returns:
            (regime, confidence, signals_dict)
        """
        if not daily_candles or len(daily_candles) < self.EMA_TREND:
            return self.regime, self.confidence, {"error": "insufficient_data"}

        now = current_ts or time.time()
        signals = {}

        # ═══════════════════════════════════════════
        # TREND ANALYSIS (daily)
        # ═══════════════════════════════════════════
        closes = [c["close"] for c in daily_candles]
        highs = [c["high"] for c in daily_candles]
        lows = [c["low"] for c in daily_candles]
        volumes = [c.get("volume", 0) for c in daily_candles]

        # EMAs
        ema_fast = self._ema(closes, self.EMA_FAST)
        ema_slow = self._ema(closes, self.EMA_SLOW)
        ema_trend = self._ema(closes, self.EMA_TREND)

        current_price = closes[-1]
        signals["price"] = round(current_price, 2)
        signals["ema_fast"] = round(ema_fast, 2)
        signals["ema_slow"] = round(ema_slow, 2)
        signals["ema_trend"] = round(ema_trend, 2)

        # Price vs EMAs
        above_ema_fast = current_price > ema_fast
        above_ema_slow = current_price > ema_slow
        above_ema_trend = current_price > ema_trend
        ema_fast_above_slow = ema_fast > ema_slow
        ema_fast_above_trend = ema_fast > ema_trend

        signals["above_ema200"] = above_ema_trend
        signals["ema_cross"] = "BULLISH" if ema_fast_above_slow else "BEARISH"

        # EMA slope (5-bar)
        if len(closes) >= 5:
            ema_trend_5ago = self._ema(closes[:-5], self.EMA_TREND) if len(closes) > self.EMA_TREND + 5 else ema_trend
            slope = (ema_trend - ema_trend_5ago) / ema_trend_5ago if ema_trend_5ago else 0
            signals["ema200_slope"] = f"{slope*100:+.3f}%"
        else:
            slope = 0

        # Higher highs / higher lows (last 10 bars)
        hh_hl = self._check_structure(highs[-10:], lows[-10:])
        signals["structure"] = hh_hl

        # ═══════════════════════════════════════════
        # MOMENTUM (daily RSI, weekly ROC)
        # ═══════════════════════════════════════════
        rsi = self._rsi(closes, self.RSI_PERIOD)
        signals["rsi"] = round(rsi, 1)

        # Weekly ROC (if we have enough data)
        weekly_roc = None
        if len(closes) >= 5:
            weekly_roc = (closes[-1] - closes[-5]) / closes[-5]
            signals["weekly_roc"] = f"{weekly_roc*100:+.2f}%"

        # ═══════════════════════════════════════════
        # VOLATILITY (daily ATR regime)
        # ═══════════════════════════════════════════
        atr = self._atr(highs, lows, closes, self.ATR_PERIOD)
        avg_atr = self._atr(highs, lows, closes, 50) if len(closes) >= 50 else atr
        atr_ratio = atr / avg_atr if avg_atr > 0 else 1.0
        signals["atr"] = round(atr, 2)
        signals["atr_ratio"] = round(atr_ratio, 2)

        if atr_ratio > self.VOL_HIGH_ATR_MULT:
            vol_regime = "HIGH_VOL"
        elif atr_ratio < self.VOL_LOW_ATR_MULT:
            vol_regime = "LOW_VOL"
        else:
            vol_regime = "NORMAL"
        signals["vol_regime"] = vol_regime

        # Bollinger width
        if len(closes) >= 20:
            sma20 = sum(closes[-20:]) / 20
            std20 = (sum((c - sma20)**2 for c in closes[-20:]) / 20) ** 0.5
            bb_width = (2 * std20 / sma20) * 100  # percentage
            signals["bb_width"] = f"{bb_width:.2f}%"
        else:
            bb_width = 0

        # ═══════════════════════════════════════════
        # DERIVATIVES (daily aggregates)
        # ═══════════════════════════════════════════
        deriv_score = {"bull": 0, "bear": 0, "stress": 0}
        if deriv_daily:
            fr = deriv_daily.get("fr_avg", 0)
            ls = deriv_daily.get("ls_avg", 2.0)
            oi_trend = deriv_daily.get("oi_trend", 0)  # % change over period
            taker = deriv_daily.get("taker_avg", 1.0)

            signals["deriv_fr"] = f"{fr*10000:.2f}bps"
            signals["deriv_ls"] = round(ls, 2)
            signals["deriv_oi_trend"] = f"{oi_trend:+.2f}%"

            # FR signal (daily avg, wider thresholds)
            if fr > 0.0001: deriv_score["bull"] += 1.5
            elif fr < -0.00005: deriv_score["bear"] += 1.5

            # LS ratio
            if ls > 2.5: deriv_score["bear"] += 1.0  # very long-crowded
            elif ls < 1.5: deriv_score["bull"] += 1.0  # short-crowded

            # OI trend
            if oi_trend < -5: deriv_score["stress"] += 2.0  # major OI drop
            elif oi_trend > 10: deriv_score["bull"] += 1.0  # OI surge

            # Taker flow
            if taker > 1.3: deriv_score["bull"] += 0.5
            elif taker < 0.7: deriv_score["bear"] += 0.5

        signals["deriv_scores"] = deriv_score

        # ═══════════════════════════════════════════
        # REGIME SCORING (weighted ensemble)
        # ═══════════════════════════════════════════
        bull_score = 0.0
        bear_score = 0.0
        stress_score = 0.0

        # Trend (weight: 3.0 — most important)
        if above_ema_trend and ema_fast_above_slow and slope > self.TREND_SLOPE_THRESHOLD:
            bull_score += 3.0
            signals["trend_signal"] = "BULL"
        elif not above_ema_trend and not ema_fast_above_slow and slope < -self.TREND_SLOPE_THRESHOLD:
            bear_score += 3.0
            signals["trend_signal"] = "BEAR"
        elif not above_ema_trend and slope < -self.TREND_SLOPE_THRESHOLD * 2:
            bear_score += 2.0
            signals["trend_signal"] = "BEARISH"
        else:
            signals["trend_signal"] = "NEUTRAL"

        # Structure (weight: 2.0)
        if hh_hl == "HIGHER_HIGHS":
            bull_score += 2.0
        elif hh_hl == "LOWER_LOWS":
            bear_score += 2.0
        elif hh_hl == "MIXED":
            bull_score += 0.5
            bear_score += 0.5

        # RSI (weight: 1.0)
        if rsi > self.RSI_OVERBOUGHT:
            bear_score += 1.0  # overbought = potential reversal
            signals["rsi_signal"] = "OVERBOUGHT"
        elif rsi < self.RSI_OVERSOLD:
            bull_score += 1.0  # oversold = potential bounce
            signals["rsi_signal"] = "OVERSOLD"
        else:
            signals["rsi_signal"] = "NEUTRAL"

        # Volatility (weight: 1.5)
        if vol_regime == "HIGH_VOL":
            stress_score += 1.5
            signals["vol_signal"] = "STRESS"
        elif vol_regime == "LOW_VOL":
            bull_score += 0.3  # low vol = trending-friendly
            signals["vol_signal"] = "CALM"

        # Weekly ROC (weight: 1.5)
        if weekly_roc is not None:
            if weekly_roc > 0.05:   # +5% weekly
                bull_score += 1.5
            elif weekly_roc < -0.05: # -5% weekly
                bear_score += 1.5
            elif weekly_roc < -0.10: # -10% weekly = stress
                stress_score += 1.5

        # Derivatives (weight: 1.0)
        bull_score += deriv_score["bull"]
        bear_score += deriv_score["bear"]
        stress_score += deriv_score["stress"]

        signals["scores"] = {
            "bull": round(bull_score, 2),
            "bear": round(bear_score, 2),
            "stress": round(stress_score, 2),
        }

        # ═══════════════════════════════════════════
        # REGIME DECISION
        # ═══════════════════════════════════════════
        if stress_score > 3:
            raw_regime = "STRESS"
            raw_confidence = min(0.95, 0.5 + stress_score * 0.05)
        elif bull_score > bear_score + 2:
            raw_regime = "BULL"
            raw_confidence = min(0.95, 0.5 + (bull_score - bear_score) * 0.05)
        elif bear_score > bull_score + 2:
            raw_regime = "BEAR"
            raw_confidence = min(0.95, 0.5 + (bear_score - bull_score) * 0.05)
        elif bear_score > bull_score + 1:
            raw_regime = "MILDLY_BEARISH"
            raw_confidence = min(0.8, 0.5 + (bear_score - bull_score) * 0.05)
        else:
            raw_regime = "RANGING"
            raw_confidence = 0.5

        # ═══════════════════════════════════════════
        # PERSISTENCE ENFORCEMENT (daily-level)
        # ═══════════════════════════════════════════
        regime = self._apply_daily_persistence(raw_regime, raw_confidence, now)

        self.regime = regime
        self.confidence = raw_confidence
        self.signals = signals
        self._consecutive_votes.append(raw_regime)

        return regime, raw_confidence, signals

    def _apply_daily_persistence(self, raw_regime, confidence, now):
        """
        Daily-level persistence:
        1. Same regime → keep
        2. New regime → need HYSTERESIS_BARS consecutive daily votes
        3. Cooldown: COOLDOWN_DAYS between transitions
        4. Min bars in current regime before change
        """
        current = self.regime

        # Same → keep
        if raw_regime == current:
            return current

        # Check cooldown
        if self._last_transition_ts:
            days_since = (now - self._last_transition_ts) / 86400
            if days_since < self.COOLDOWN_DAYS:
                return current

        # Check hysteresis (consecutive daily votes)
        recent = list(self._consecutive_votes)[-self.HYSTERESIS_BARS:]
        hysteresis_met = (
            len(recent) >= self.HYSTERESIS_BARS and
            all(r == raw_regime for r in recent)
        )

        # Check min bars in current regime
        if self._regime_start_ts:
            bars_in_regime = (now - self._regime_start_ts) / 86400
            min_bars_met = bars_in_regime >= self.MIN_BARS_BEFORE_CHANGE
        else:
            min_bars_met = True

        # Allow transition
        if hysteresis_met and min_bars_met:
            self._last_transition_ts = now
            self._regime_start_ts = now
            return raw_regime

        # Special: STRESS can override faster (safety)
        if raw_regime == "STRESS" and confidence > 0.8:
            self._last_transition_ts = now
            self._regime_start_ts = now
            return raw_regime

        return current

    # ═══════════════════════════════════════════
    # Technical indicator helpers
    # ═══════════════════════════════════════════

    def _ema(self, data, period):
        """Exponential moving average."""
        if len(data) < period:
            return data[-1] if data else 0
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _rsi(self, closes, period=14):
        """Relative Strength Index."""
        if len(closes) < period + 1:
            return 50  # neutral
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _atr(self, highs, lows, closes, period=14):
        """Average True Range."""
        if len(closes) < period + 1:
            return 0
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        return sum(trs[-period:]) / period

    def _check_structure(self, highs, lows):
        """Check for higher highs/higher lows or lower lows/lower highs."""
        if len(highs) < 4 or len(lows) < 4:
            return "INSUFFICIENT"

        # Compare first half vs second half
        mid = len(highs) // 2
        first_high = max(highs[:mid])
        second_high = max(highs[mid:])
        first_low = min(lows[:mid])
        second_low = min(lows[mid:])

        hh = second_high > first_high
        hl = second_low > first_low
        lh = second_high < first_high
        ll = second_low < first_low

        if hh and hl:
            return "HIGHER_HIGHS"
        elif ll and lh:
            return "LOWER_LOWS"
        elif hh and ll:
            return "EXPANSION"  # volatility expansion
        elif not hh and not ll:
            return "COMPRESSION"
        else:
            return "MIXED"

    def get_regime_info(self):
        """Detailed regime state for monitoring."""
        return {
            "regime": self.regime,
            "confidence": round(self.confidence, 3),
            "regime_start": self._regime_start_ts,
            "last_transition": self._last_transition_ts,
            "consecutive_votes": list(self._consecutive_votes),
            "signals": self.signals,
        }

    # === Convenience methods ===
    def is_bullish(self):
        return self.regime == "BULL" and self.confidence >= 0.5

    def is_bearish(self):
        return self.regime in ("BEAR", "STRESS", "MILDLY_BEARISH") and self.confidence >= 0.5

    def is_ranging(self):
        return self.regime == "RANGING"

    def is_neutral_or_bearish(self):
        return self.regime in ("BEAR", "STRESS", "MILDLY_BEARISH")


# ═══════════════════════════════════════════════════════════════
# BACKTEST: V4 vs V2/V3 on daily data
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import requests
    import random

    print("=" * 70)
    print("REGIME CLASSIFIER V4 — DAILY TIMEFRAME BACKTEST")
    print("=" * 70)

    # Fetch daily candles (200 bars = ~7 months)
    print("\nFetching 1D candles (200 bars)...")
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": "ETHUSDT", "interval": "1d", "limit": 200}, timeout=15)
    r.raise_for_status()
    daily = []
    for c in r.json():
        daily.append({
            "ts": datetime.fromtimestamp(c[0]/1000),
            "open": float(c[1]), "high": float(c[2]),
            "low": float(c[3]), "close": float(c[4]),
            "volume": float(c[5]),
        })
    print(f"  Loaded {len(daily)} daily candles")
    print(f"  Range: {daily[0]['ts'].date()} → {daily[-1]['ts'].date()}")
    print(f"  Price: ${daily[0]['close']:.2f} → ${daily[-1]['close']:.2f}")

    # Fetch weekly candles
    print("\nFetching 1W candles (52 bars)...")
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": "ETHUSDT", "interval": "1w", "limit": 52}, timeout=15)
    r.raise_for_status()
    weekly = []
    for c in r.json():
        weekly.append({
            "ts": datetime.fromtimestamp(c[0]/1000),
            "open": float(c[1]), "high": float(c[2]),
            "low": float(c[3]), "close": float(c[4]),
            "volume": float(c[5]),
        })
    print(f"  Loaded {len(weekly)} weekly candles")

    # Load derivatives (daily aggregates)
    deriv_csv = os.path.join(BASE, "data", "derivatives_history", "derivatives_collected.csv")
    deriv_daily = defaultdict(list)
    if os.path.exists(deriv_csv):
        import csv
        with open(deriv_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row.get("timestamp", "")
                if ts:
                    day = ts[:10]  # YYYY-MM-DD
                    try:
                        deriv_daily[day].append({
                            "fr": float(row.get("funding_rate", 0) or 0),
                            "ls": float(row.get("ls_ratio", 2.0) or 2.0),
                            "oi": float(row.get("oi", 0) or 0),
                            "taker": float(row.get("futures_taker_ratio", 1.0) or 1.0),
                        })
                    except:
                        continue

    # Aggregate derivatives to daily
    deriv_agg = {}
    for day, rows in deriv_daily.items():
        if rows:
            deriv_agg[day] = {
                "fr_avg": sum(r["fr"] for r in rows) / len(rows),
                "ls_avg": sum(r["ls"] for r in rows) / len(rows),
                "taker_avg": sum(r["taker"] for r in rows) / len(rows),
                "oi_start": rows[0]["oi"],
                "oi_end": rows[-1]["oi"],
                "oi_trend": ((rows[-1]["oi"] - rows[0]["oi"]) / rows[0]["oi"] * 100) if rows[0]["oi"] > 0 else 0,
            }
    print(f"  Derivatives daily aggregates: {len(deriv_agg)} days")

    # ═══════════════════════════════════════════
    # Run V4 classifier on daily data
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("RUNNING V4 CLASSIFIER ON DAILY DATA")
    print("=" * 70)

    v4 = RegimeClassifierV4()
    regime_history = []

    for i in range(v4.EMA_TREND, len(daily)):
        window = daily[:i+1]
        day_str = daily[i]["ts"].strftime("%Y-%m-%d")
        dd = deriv_agg.get(day_str)

        regime, conf, signals = v4.classify(
            window,
            weekly_candles=weekly,
            deriv_daily=dd,
            current_ts=daily[i]["ts"].timestamp()
        )
        regime_history.append({
            "date": day_str,
            "price": daily[i]["close"],
            "regime": regime,
            "confidence": conf,
            "trend": signals.get("trend_signal", "?"),
            "rsi": signals.get("rsi", 0),
            "vol": signals.get("vol_regime", "?"),
        })

    # Print regime history
    print(f"\n{'Date':<12} {'Price':>10} {'Regime':<15} {'Conf':>6} {'Trend':<10} {'RSI':>6} {'Vol':<10}")
    print("-" * 75)
    prev = None
    for r in regime_history:
        marker = " ← CHANGED" if prev and r["regime"] != prev else ""
        print(f"{r['date']:<12} ${r['price']:>9.2f} {r['regime']:<15} {r['confidence']:>5.2f} {r['trend']:<10} {r['rsi']:>5.1f} {r['vol']:<10}{marker}")
        prev = r["regime"]

    # ═══════════════════════════════════════════
    # 0/1 Strategy backtest (V4)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("0/1 STRATEGY BACKTEST (V4 daily regime)")
    print("=" * 70)

    capital = 10000.0
    position = None
    trades = []
    returns = []
    peak = capital
    max_dd = 0
    prev_regime = "RANGING"

    for r in regime_history:
        price = r["price"]
        regime = r["regime"]

        should_long = regime in ("BULL", "RANGING", "MILDLY_BEARISH")

        if should_long and position is None:
            position = {"entry": price, "date": r["date"]}
        elif not should_long and position is not None:
            pnl = (price - position["entry"]) / position["entry"]
            capital *= (1 + pnl)
            returns.append(pnl)
            trades.append({
                "entry_date": position["date"],
                "exit_date": r["date"],
                "entry": position["entry"],
                "exit": price,
                "pnl_pct": round(pnl * 100, 2),
                "regime": regime,
            })
            peak = max(peak, capital)
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)
            position = None

        prev_regime = regime

    # Close final
    if position:
        final_pnl = (daily[-1]["close"] - position["entry"]) / position["entry"]
        capital *= (1 + final_pnl)
        returns.append(final_pnl)

    # Buy & hold
    bh = (daily[-1]["close"] - daily[0]["close"]) / daily[0]["close"]

    print(f"\nV4 Results:")
    print(f"  Capital: ${capital:,.2f} (start: $10,000)")
    print(f"  Return: {(capital/10000 - 1)*100:+.2f}%")
    print(f"  Buy & Hold: {bh*100:+.2f}%")
    print(f"  Trades: {len(trades)}")
    print(f"  Max Drawdown: {max_dd*100:.2f}%")
    if returns:
        avg = sum(returns) / len(returns)
        std = (sum((r - avg)**2 for r in returns) / max(len(returns)-1, 1)) ** 0.5
        sharpe = (avg / std) * (365**0.5) if std > 0 else 0
        wr = sum(1 for r in returns if r > 0) / len(returns)
        print(f"  Win Rate: {wr*100:.1f}%")
        print(f"  Sharpe (annualized): {sharpe:.2f}")

    print(f"\n  Trades:")
    for t in trades:
        print(f"    {t['entry_date']} → {t['exit_date']}: {t['pnl_pct']:+.2f}% (regime: {t['regime']})")

    # Monte Carlo
    if returns:
        print(f"\n  Monte Carlo (10k sims, 30-day):")
        sims = []
        for _ in range(10000):
            sampled = random.choices(returns, k=max(1, int(len(returns) * 30 / len(regime_history))))
            cum = 1.0
            for s in sampled:
                cum *= (1 + s)
            sims.append(cum - 1)
        sims.sort()
        n = len(sims)
        print(f"    P5:  {sims[int(n*0.05)]*100:+.2f}%")
        print(f"    P25: {sims[int(n*0.25)]*100:+.2f}%")
        print(f"    P50: {sims[int(n*0.50)]*100:+.2f}%")
        print(f"    P75: {sims[int(n*0.75)]*100:+.2f}%")
        print(f"    P95: {sims[int(n*0.95)]*100:+.2f}%")
        print(f"    Prob(loss): {sum(1 for s in sims if s < 0)/n*100:.1f}%")

    # Regime distribution
    print(f"\n  Regime Distribution:")
    counts = defaultdict(int)
    for r in regime_history:
        counts[r["regime"]] += 1
    total = len(regime_history) or 1
    for regime in ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH"]:
        cnt = counts.get(regime, 0)
        print(f"    {regime:<15} {cnt:>3} days ({cnt/total*100:>5.1f}%)")

    transitions = sum(1 for i in range(1, len(regime_history)) if regime_history[i]["regime"] != regime_history[i-1]["regime"])
    print(f"\n  Transitions: {transitions} ({transitions/len(regime_history)*100:.1f} per day)")
    print(f"  Avg regime duration: {len(regime_history)/max(transitions,1):.1f} days")

    print("\nDone.")
