#!/usr/bin/env python3
"""
Event-Driven Backtest: Replay 15m OHLCV through actual strategy logic.

Unlike the simulation (random signals), this replays real price data through
the same feature computation and strategy logic the executor uses.

Data: eth_15m_merged.csv (Jan-Jul 2026) + derivatives_collected.csv (Apr-Jul)
Strategies: orderbook_imbalance v5, liquidation_cascade v5 (with V4 daily regime)

Process per bar:
1. Compute features (EMA, RSI, ATR, VWAP, taker ratio, etc.)
2. Get V4 daily regime
3. Run strategy check() logic
4. If signal → open position, track TP/SL/timeout
5. Check open positions for exit
"""
import json, os, sys, csv, math, time
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_ohlcv(csv_path, start_date="2026-01-01"):
    """Load 15m OHLCV CSV."""
    bars = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("Open time", "")
            if ts < start_date:
                continue
            try:
                bars.append({
                    "ts": ts,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                    "trades": int(row.get("Number of trades", 0)),
                    "taker_buy_vol": float(row.get("Taker buy base asset volume", 0)),
                })
            except (ValueError, KeyError):
                continue
    return bars


def load_derivatives(csv_path):
    """Load derivatives CSV → daily aggregates."""
    raw = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp", "")
            if not ts:
                continue
            day = ts.replace("T", " ")[:10]
            try:
                raw[day].append({
                    "fr": float(row.get("funding_rate", 0) or 0),
                    "ls": float(row.get("ls_ratio", 2.0) or 2.0),
                    "oi": float(row.get("oi", 0) or 0),
                    "taker": float(row.get("futures_taker_ratio", 1.0) or 1.0),
                })
            except:
                continue
    agg = {}
    for day, rows in raw.items():
        if rows:
            agg[day] = {
                "fr_avg": sum(r["fr"] for r in rows) / len(rows),
                "ls_avg": sum(r["ls"] for r in rows) / len(rows),
                "taker_avg": sum(r["taker"] for r in rows) / len(rows),
                "oi_start": rows[0]["oi"],
                "oi_end": rows[-1]["oi"],
                "oi_trend": ((rows[-1]["oi"] - rows[0]["oi"]) / rows[0]["oi"] * 100) if rows[0]["oi"] > 0 else 0,
            }
    return agg


# ═══════════════════════════════════════════════════════════════
# FEATURE COMPUTATION (mirrors executor logic)
# ═══════════════════════════════════════════════════════════════

class FeatureEngine:
    """Computes the same features the executor uses from 15m bars."""

    def __init__(self):
        self.closes = []
        self.highs = []
        self.lows = []
        self.volumes = []
        self.taker_buys = []
        self.vwap_cum_vol = 0
        self.vwap_cum_pv = 0

    def update(self, bar):
        """Add a new bar and update features."""
        self.closes.append(bar["close"])
        self.highs.append(bar["high"])
        self.lows.append(bar["low"])
        self.volumes.append(bar["volume"])
        self.taker_buys.append(bar["taker_buy_vol"])

        # Running VWAP (session-based, reset daily at 00:00)
        ts = bar["ts"]
        if ts[11:13] == "00" and ts[14:16] == "00":
            self.vwap_cum_vol = 0
            self.vwap_cum_pv = 0
        self.vwap_cum_vol += bar["volume"]
        self.vwap_cum_pv += bar["close"] * bar["volume"]

    def get_features(self, idx):
        """Get all features at bar index."""
        if idx < 50:
            return None

        price = self.closes[idx]
        n = idx + 1

        # EMAs
        ema_200 = self._ema(self.closes[:n], 200)
        ema_50 = self._ema(self.closes[:n], 50)

        # ATR
        atr = self._atr(self.highs[:n], self.lows[:n], self.closes[:n], 14)

        # RSI
        rsi = self._rsi(self.closes[:n], 14)

        # Vol ratio (current vol / 20-bar avg)
        if idx >= 20:
            avg_vol = np.mean(self.volumes[idx-20:idx])
            vol_ratio = self.volumes[idx] / avg_vol if avg_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        # Taker ratio (last 4 bars)
        if idx >= 4:
            buy = sum(self.taker_buys[idx-4:idx])
            total = sum(self.volumes[idx-4:idx])
            taker_ratio = buy / total if total > 0 else 0.5
        else:
            taker_ratio = 0.5

        # Taker z-score
        taker_z = 0
        if idx >= 60:
            window = []
            for j in range(max(0, idx-60), idx-4, 4):
                wb = sum(self.taker_buys[j:j+4])
                wt = sum(self.volumes[j:j+4])
                if wt > 0:
                    window.append(wb / wt)
            if len(window) >= 5:
                mean_r = np.mean(window)
                std_r = np.std(window)
                if std_r > 0:
                    taker_z = (taker_ratio - mean_r) / std_r

        # VWAP
        vwap = self.vwap_cum_pv / self.vwap_cum_vol if self.vwap_cum_vol > 0 else price
        vwap_dev = (price - vwap) / vwap

        # Momentum
        mom_5 = (self.closes[idx] - self.closes[idx-5]) / self.closes[idx-5] if idx >= 5 else 0
        mom_3 = (self.closes[idx] - self.closes[idx-3]) / self.closes[idx-3] if idx >= 3 else 0

        # Swing high/low (20 bars)
        swing_high = max(self.highs[idx-20:idx]) if idx >= 20 else price + atr
        swing_low = min(self.lows[idx-20:idx]) if idx >= 20 else price - atr

        # OBI from taker data (trade-based)
        trade_obi = 0
        if idx >= 4:
            buy_vol = sum(self.taker_buys[idx-4:idx])
            total_vol = sum(self.volumes[idx-4:idx])
            if total_vol > 0:
                trade_obi = (buy_vol / total_vol - 0.5) * 2

        # Trade OBI z-score
        trade_obi_z = 0
        if idx >= 60:
            window_bi = []
            for j in range(max(0, idx-60), idx-4, 4):
                bv = sum(self.taker_buys[j:j+4])
                tv = sum(self.volumes[j:j+4])
                if tv > 0:
                    window_bi.append((bv / tv - 0.5) * 2)
            if len(window_bi) >= 5:
                mean_bi = np.mean(window_bi)
                std_bi = np.std(window_bi)
                if std_bi > 0:
                    trade_obi_z = (trade_obi - mean_bi) / std_bi

        # Bollinger width
        bb_width = 0
        if idx >= 20:
            sma20 = np.mean(self.closes[idx-20:idx])
            std20 = np.std(self.closes[idx-20:idx])
            bb_width = (2 * std20 / sma20) * 100 if sma20 > 0 else 0

        return {
            "price": price, "atr": atr, "ema_200": ema_200, "ema_50": ema_50,
            "rsi": rsi, "vol_ratio": vol_ratio, "taker_ratio": taker_ratio,
            "taker_z": taker_z, "vwap": vwap, "vwap_dev": vwap_dev,
            "mom_5": mom_5, "mom_3": mom_3, "swing_high": swing_high,
            "swing_low": swing_low, "trade_obi": trade_obi,
            "trade_obi_z": trade_obi_z, "bb_width": bb_width,
            "ts": self.closes[idx] if isinstance(self.closes[idx], str) else "",
        }

    def _ema(self, data, period):
        if len(data) < period:
            return data[-1] if data else 0
        mult = 2 / (period + 1)
        ema = sum(data[:period]) / period
        for p in data[period:]:
            ema = (p - ema) * mult + ema
        return ema

    def _rsi(self, data, period=14):
        if len(data) < period + 1:
            return 50
        deltas = [data[i] - data[i-1] for i in range(1, len(data))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        avg_g = sum(gains) / period
        avg_l = sum(losses) / period
        if avg_l == 0:
            return 100
        rs = avg_g / avg_l
        return 100 - (100 / (1 + rs))

    def _atr(self, highs, lows, closes, period=14):
        if len(closes) < period + 1:
            return 0
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        return np.mean(trs[-period:])


# ═══════════════════════════════════════════════════════════════
# STRATEGY LOGIC (simplified, mirrors check() methods)
# ═══════════════════════════════════════════════════════════════

def check_orderbook_imbalance_v5(feat, regime):
    """Simplified OB v5 check."""
    price = feat["price"]
    atr = feat["atr"]
    trade_obi = feat["trade_obi"]
    trade_obi_z = feat["trade_obi_z"]
    taker_ratio = feat["taker_ratio"]
    vwap_dev = feat["vwap_dev"]
    mom_5 = feat["mom_5"]
    vol_ratio = feat["vol_ratio"]

    # Need minimum data
    if atr == 0:
        return None

    # Combined OBI signal (trade-based weighted higher)
    obi_combined = trade_obi * 0.7 + trade_obi_z * 0.3

    # Direction
    direction = None
    if obi_combined > 0.05 and vwap_dev < 0.003:  # Positive OBI + VWAP discount
        if mom_5 < -0.015:
            return None
        direction = "LONG"
    elif obi_combined < -0.05 and vwap_dev > -0.003:  # Negative OBI + VWAP premium
        if mom_5 > 0.015:
            return None
        direction = "SHORT"

    if not direction:
        return None

    # Conviction (concave)
    base = 0.45
    obi_abs = min(abs(obi_combined), 0.5)
    obi_strength = math.sqrt(obi_abs) * 0.25
    trade_bonus = min(math.sqrt(max(abs(trade_obi_z) - 1.0, 0)) * 0.08, 0.15)
    vwap_bonus = 0.12 if abs(vwap_dev) > 0.005 else (0.08 if abs(vwap_dev) > 0.002 else 0)
    vol_bonus = min((vol_ratio - 1.0) * 0.05, 0.10) if vol_ratio > 1.0 else 0
    regime_bonus = 0.05 if (regime == "BULL" and direction == "LONG") or (regime == "BEAR" and direction == "SHORT") else 0.03 if regime == "RANGING" else 0

    conviction = min(base + obi_strength + trade_bonus + vwap_bonus + vol_bonus + regime_bonus, 0.90)
    if conviction < 0.50:
        return None

    # TP/SL
    tp_mult = {"BULL": 2.5, "BEAR": 2.0, "RANGING": 2.0, "STRESS": 1.5}.get(regime, 2.0)
    if direction == "LONG":
        sl = feat["swing_low"]
        sl_dist = max(price - sl, 0.5 * atr)
        sl_dist = min(sl_dist, 1.5 * atr)
        sl = price - sl_dist
        tp1 = price + tp_mult * atr
    else:
        sl = feat["swing_high"]
        sl_dist = max(sl - price, 0.5 * atr)
        sl_dist = min(sl_dist, 1.5 * atr)
        sl = price + sl_dist
        tp1 = price - tp_mult * atr

    return {
        "strategy": "orderbook_imbalance",
        "direction": direction,
        "entry": price, "sl": sl, "tp1": tp1,
        "sl_pct": (sl_dist / price) * 100,
        "tp_pct": (tp_mult * atr / price) * 100,
        "conviction": conviction,
        "hold_bars": 48,  # 12h
    }


def check_liquidation_cascade_v5(feat, regime, deriv):
    """Simplified cascade v5 check."""
    price = feat["price"]
    atr = feat["atr"]
    mom_3 = feat["mom_3"]

    if not deriv or atr == 0:
        return None

    oi_roc = deriv.get("oi_trend", 0) / 100  # Convert % to decimal
    ls_ratio = deriv.get("ls_avg", 2.0)

    # Regime-adaptive thresholds
    if regime in ("STRESS", "BEAR"):
        oi_thresh = -0.008
        ls_thresh = 1.3
    elif regime == "RANGING":
        oi_thresh = -0.01
        ls_thresh = 1.5
    else:
        oi_thresh = -0.012
        ls_thresh = 1.6

    # SHORT cascade
    direction = None
    strength = 0
    source = ""

    if oi_roc < oi_thresh and ls_ratio > ls_thresh:
        if mom_3 > 0.005:
            return None
        direction = "SHORT"
        strength = min(abs(oi_roc) * 15, 0.9) if oi_roc < oi_thresh * 1.5 else min(abs(oi_roc) * 10, 0.7)
        source = "cascade_short"

    # LONG cascade
    oi_surge = 0.015 if regime not in ("STRESS", "BEAR") else 0.012
    ls_short = 0.7 if regime not in ("BULL",) else 0.6

    if oi_roc > oi_surge and ls_ratio < ls_short:
        if mom_3 < -0.005:
            return None
        direction = "LONG"
        strength = min(abs(oi_roc) * 8, 0.8)
        source = "cascade_long"

    if not direction:
        return None

    conviction = min(0.45 + strength * 0.40, 0.90)
    if conviction < 0.45:
        return None

    tp_mult = {"BULL": 2.0, "BEAR": 1.8, "RANGING": 2.0, "STRESS": 1.5}.get(regime, 2.0)
    sl_mult = {"BULL": 1.0, "BEAR": 1.2, "RANGING": 1.0, "STRESS": 0.8}.get(regime, 1.0)

    if direction == "LONG":
        sl = price - sl_mult * atr
        tp1 = price + tp_mult * atr
    else:
        sl = price + sl_mult * atr
        tp1 = price - tp_mult * atr

    return {
        "strategy": "liquidation_cascade",
        "direction": direction,
        "entry": price, "sl": sl, "tp1": tp1,
        "sl_pct": (sl_mult * atr / price) * 100,
        "tp_pct": (tp_mult * atr / price) * 100,
        "conviction": conviction,
        "hold_bars": 16,  # 4h
    }


# ═══════════════════════════════════════════════════════════════
# DAILY REGIME (simplified V4)
# ═══════════════════════════════════════════════════════════════

def compute_daily_regimes(bars):
    """Compute V4 daily regime for each day."""
    # Aggregate to daily
    daily = defaultdict(lambda: {"highs": [], "lows": [], "closes": [], "volumes": []})
    for bar in bars:
        day = bar["ts"][:10]
        daily[day]["highs"].append(bar["high"])
        daily[day]["lows"].append(bar["low"])
        daily[day]["closes"].append(bar["close"])
        daily[day]["volumes"].append(bar["volume"])

    # Compute daily features
    sorted_days = sorted(daily.keys())
    daily_closes = []
    regimes = {}

    for day in sorted_days:
        d = daily[day]
        daily_closes.append(d["closes"][-1])

        if len(daily_closes) < 50:
            regimes[day] = "RANGING"
            continue

        closes = daily_closes
        ema50 = _ema(closes, 50)
        ema10 = _ema(closes, 10)
        price = closes[-1]

        # RSI
        rsi = _rsi(closes, 14)

        # Trend
        above_ema50 = price > ema50
        ema_cross = ema10 > ema50
        slope = (ema50 - _ema(closes[:-5], 50)) / _ema(closes[:-5], 50) if len(closes) > 55 else 0

        # Structure
        if len(d["highs"]) >= 10:
            mid = len(d["highs"]) // 2
            hh = max(d["highs"][mid:]) > max(d["highs"][:mid])
            hl = min(d["lows"][mid:]) > min(d["lows"][:mid])
            ll = min(d["lows"][mid:]) < min(d["lows"][:mid])
            lh = max(d["highs"][mid:]) < max(d["highs"][:mid])
        else:
            hh = hl = ll = lh = False

        bull = bear = stress = 0

        if above_ema50 and ema_cross and slope > 0.001:
            bull += 3
        elif not above_ema50 and not ema_cross and slope < -0.001:
            bear += 3
        elif not above_ema50 and slope < -0.002:
            bear += 2

        if hh and hl:
            bull += 2
        elif ll and lh:
            bear += 2

        if rsi > 70:
            bear += 1
        elif rsi < 30:
            bull += 1

        # Weekly ROC
        if len(closes) >= 5:
            weekly_roc = (closes[-1] - closes[-5]) / closes[-5]
            if weekly_roc > 0.05:
                bull += 1.5
            elif weekly_roc < -0.05:
                bear += 1.5
            elif weekly_roc < -0.10:
                stress += 1.5

        if stress > 3:
            regimes[day] = "STRESS"
        elif bull > bear + 2:
            regimes[day] = "BULL"
        elif bear > bull + 2:
            regimes[day] = "BEAR"
        elif bear > bull + 1:
            regimes[day] = "MILDLY_BEARISH"
        else:
            regimes[day] = "RANGING"

    return regimes


def _ema(data, period):
    if len(data) < period:
        return data[-1] if data else 0
    mult = 2 / (period + 1)
    ema = sum(data[:period]) / period
    for p in data[period:]:
        ema = (p - ema) * mult + ema
    return ema

def _rsi(data, period=14):
    if len(data) < period + 1:
        return 50
    deltas = [data[i] - data[i-1] for i in range(1, len(data))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    if avg_l == 0:
        return 100
    return 100 - (100 / (1 + avg_g / avg_l))


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def run_backtest(bars, deriv_agg, regimes):
    """Event-driven backtest replaying real bars through strategy logic."""
    engine = FeatureEngine()
    positions = []
    trades = []
    capital = 10000.0
    initial_capital = capital
    peak = capital
    max_dd = 0
    equity = [capital]
    cooldown = {}  # strategy+direction -> last entry bar

    for i, bar in enumerate(bars):
        engine.update(bar)
        day = bar["ts"][:10]
        regime = regimes.get(day, "RANGING")
        price = bar["close"]

        # Check open positions for exit
        closed = []
        for pos in positions:
            bars_held = i - pos["entry_bar"]

            # Check SL
            if pos["direction"] == "LONG":
                hit_sl = bar["low"] <= pos["sl"]
                hit_tp = bar["high"] >= pos["tp1"]
            else:
                hit_sl = bar["high"] >= pos["sl"]
                hit_tp = bar["low"] <= pos["tp1"]

            if hit_sl:
                pnl_pct = -pos["sl_pct"] / 100
                closed.append((pos, "SL", pnl_pct, bars_held))
            elif hit_tp:
                pnl_pct = pos["tp_pct"] / 100
                closed.append((pos, "TP", pnl_pct, bars_held))
            elif bars_held >= pos["hold_bars"]:
                pnl_pct = (price - pos["entry"]) / pos["entry"] if pos["direction"] == "LONG" else (pos["entry"] - price) / pos["entry"]
                closed.append((pos, "TIMEOUT", pnl_pct, bars_held))

        for pos, outcome, pnl_pct, bars_held in closed:
            pnl_dollar = capital * 0.02 * (pnl_pct / (pos["sl_pct"] / 100))  # Risk-adjusted
            capital += pnl_dollar
            trades.append({
                "strategy": pos["strategy"],
                "direction": pos["direction"],
                "entry": pos["entry"],
                "exit": price,
                "pnl_pct": round(pnl_pct * 100, 2),
                "pnl_dollar": round(pnl_dollar, 2),
                "outcome": outcome,
                "regime": pos["regime"],
                "bars_held": bars_held,
                "entry_bar": pos["entry_bar"],
                "exit_bar": i,
            })
            positions.remove(pos)

        peak = max(peak, capital)
        dd = (peak - capital) / peak
        max_dd = max(max_dd, dd)
        equity.append(capital)

        # Get features
        feat = engine.get_features(i)
        if not feat:
            continue

        # Cooldown check (1 signal per strategy per 48 bars)
        def can_trade(strat, direction):
            key = f"{strat}_{direction}"
            return cooldown.get(key, -999) < i - 48

        def record_trade(strat, direction):
            cooldown[f"{strat}_{direction}"] = i

        # Run strategies
        deriv = deriv_agg.get(day)

        # OB v5
        if can_trade("ob", "LONG") or can_trade("ob", "SHORT"):
            sig = check_orderbook_imbalance_v5(feat, regime)
            if sig and can_trade("ob", sig["direction"]):
                positions.append({
                    **sig,
                    "regime": regime,
                    "entry_bar": i,
                })
                record_trade("ob", sig["direction"])

        # Cascade v5
        if can_trade("cascade", "LONG") or can_trade("cascade", "SHORT"):
            sig = check_liquidation_cascade_v5(feat, regime, deriv)
            if sig and can_trade("cascade", sig["direction"]):
                positions.append({
                    **sig,
                    "regime": regime,
                    "entry_bar": i,
                })
                record_trade("cascade", sig["direction"])

    # Close remaining positions
    final_price = bars[-1]["close"]
    for pos in positions:
        pnl_pct = (final_price - pos["entry"]) / pos["entry"] if pos["direction"] == "LONG" else (pos["entry"] - final_price) / pos["entry"]
        pnl_dollar = capital * 0.02 * (pnl_pct / (pos["sl_pct"] / 100))
        capital += pnl_dollar
        trades.append({
            "strategy": pos["strategy"], "direction": pos["direction"],
            "entry": pos["entry"], "exit": final_price,
            "pnl_pct": round(pnl_pct * 100, 2), "pnl_dollar": round(pnl_dollar, 2),
            "outcome": "OPEN", "regime": pos["regime"],
            "bars_held": len(bars) - pos["entry_bar"],
        })

    return trades, capital, max_dd, equity


# ═══════════════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════════════

def monte_carlo(trades, n_sims=10000, horizon_days=30):
    returns = [t["pnl_pct"] / 100 for t in trades]
    if len(returns) < 2:
        return None
    trades_per_day = len(returns) / 150
    n_trades = max(1, int(horizon_days * trades_per_day))
    finals = []
    for _ in range(n_sims):
        sampled = random.choices(returns, k=n_trades)
        cum = 1.0
        for r in sampled:
            cum *= (1 + r)
        finals.append(cum - 1)
    finals.sort()
    n = len(finals)
    return {
        "horizon": horizon_days,
        "trades_per_sim": n_trades,
        "p5": round(np.percentile(finals, 5) * 100, 2),
        "p25": round(np.percentile(finals, 25) * 100, 2),
        "p50": round(np.percentile(finals, 50) * 100, 2),
        "p75": round(np.percentile(finals, 75) * 100, 2),
        "p95": round(np.percentile(finals, 95) * 100, 2),
        "mean": round(np.mean(finals) * 100, 2),
        "prob_loss": round(sum(1 for f in finals if f < 0) / n * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

import random
random.seed(42)

if __name__ == "__main__":
    print("=" * 70)
    print("EVENT-DRIVEN BACKTEST: Replay 15m OHLCV through strategy logic")
    print("=" * 70)

    # Load data
    ohlcv_path = os.path.join(BASE, "data", "eth_15m_merged.csv")
    deriv_path = os.path.join(BASE, "data", "derivatives_history", "derivatives_collected.csv")

    print(f"\nLoading OHLCV: {ohlcv_path}")
    bars = load_ohlcv(ohlcv_path, "2026-04-01")  # Start from Apr (when we have derivatives)
    print(f"  Loaded {len(bars)} bars ({bars[0]['ts']} → {bars[-1]['ts']})")

    print(f"Loading derivatives: {deriv_path}")
    deriv_agg = load_derivatives(deriv_path)
    print(f"  Loaded {len(deriv_agg)} daily aggregates")

    # Compute daily regimes
    print("\nComputing daily regimes...")
    regimes = compute_daily_regimes(bars)
    regime_counts = defaultdict(int)
    for r in regimes.values():
        regime_counts[r] += 1
    total_days = len(regimes)
    print(f"  {total_days} days:")
    for regime in ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH"]:
        cnt = regime_counts.get(regime, 0)
        print(f"    {regime:<15} {cnt:>3} ({cnt/total_days*100:.1f}%)")

    # Run backtest
    print("\nRunning event-driven backtest...")
    trades, final_capital, max_dd, equity = run_backtest(bars, deriv_agg, regimes)

    # Results
    ret = (final_capital / 10000 - 1) * 100
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Capital:     ${final_capital:,.2f} (start: $10,000)")
    print(f"  Return:      {ret:+.2f}%")
    print(f"  Trades:      {len(trades)}")
    print(f"  Win Rate:    {len(wins)/max(len(trades),1)*100:.1f}%")
    print(f"  Avg Win:     {np.mean([t['pnl_pct'] for t in wins]):+.2f}%" if wins else "  Avg Win:     N/A")
    print(f"  Avg Loss:    {np.mean([t['pnl_pct'] for t in losses]):+.2f}%" if losses else "  Avg Loss:    N/A")
    print(f"  Max DD:      {max_dd*100:.2f}%")
    if wins and losses:
        pf = abs(sum(t["pnl_pct"] for t in wins) / sum(t["pnl_pct"] for t in losses))
        print(f"  Profit Factor: {pf:.2f}")

    # By strategy
    print(f"\n  By Strategy:")
    print(f"  {'Strategy':<25} {'Trades':>7} {'Win%':>7} {'Avg PnL':>10} {'Total':>10}")
    print(f"  {'-'*60}")
    for strat in sorted(set(t["strategy"] for t in trades)):
        st = [t for t in trades if t["strategy"] == strat]
        sw = [t for t in st if t["pnl_pct"] > 0]
        print(f"  {strat:<25} {len(st):>7} {len(sw)/len(st)*100:>6.1f}% {np.mean([t['pnl_pct'] for t in st]):>+9.2f}% {sum(t['pnl_pct'] for t in st):>+9.2f}%")

    # By regime
    print(f"\n  By Regime:")
    print(f"  {'Regime':<15} {'Trades':>7} {'Win%':>7} {'Avg PnL':>10} {'Total':>10}")
    print(f"  {'-'*50}")
    for regime in ["BULL", "BEAR", "RANGING", "STRESS", "MILDLY_BEARISH"]:
        rt = [t for t in trades if t["regime"] == regime]
        if not rt:
            continue
        rw = [t for t in rt if t["pnl_pct"] > 0]
        print(f"  {regime:<15} {len(rt):>7} {len(rw)/len(rt)*100:>6.1f}% {np.mean([t['pnl_pct'] for t in rt]):>+9.2f}% {sum(t['pnl_pct'] for t in rt):>+9.2f}%")

    # By outcome
    print(f"\n  By Outcome:")
    for outcome in ["TP", "SL", "TIMEOUT", "OPEN"]:
        ot = [t for t in trades if t["outcome"] == outcome]
        if ot:
            print(f"  {outcome:<10} {len(ot):>5} ({len(ot)/len(trades)*100:.1f}%) avg {np.mean([t['pnl_pct'] for t in ot]):+.2f}%")

    # Monte Carlo
    if len(trades) >= 2:
        print(f"\n{'='*70}")
        print(f"MONTE CARLO (10,000 sims)")
        print(f"{'='*70}")
        for h in [7, 30, 90]:
            mc = monte_carlo(trades, 10000, h)
            if mc:
                print(f"\n  {h}-day ({mc['trades_per_sim']} trades/sim):")
                print(f"    P5={mc['p5']:+.1f}%  P25={mc['p25']:+.1f}%  P50={mc['p50']:+.1f}%  P75={mc['p75']:+.1f}%  P95={mc['p95']:+.1f}%")
                print(f"    Mean={mc['mean']:+.1f}%  P(loss)={mc['prob_loss']:.1f}%")

    # Trade log
    print(f"\n{'='*70}")
    print(f"TRADE LOG ({len(trades)} trades)")
    print(f"{'='*70}")
    print(f"  {'#':<4} {'Strategy':<20} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'PnL%':>8} {'Outcome':<10} {'Regime':<12} {'Bars':>5}")
    print(f"  {'-'*85}")
    for j, t in enumerate(trades):
        print(f"  {j:<4} {t['strategy']:<20} {t['direction']:<6} ${t['entry']:>9.2f} ${t['exit']:>9.2f} {t['pnl_pct']:>+7.2f}% {t['outcome']:<10} {t['regime']:<12} {t['bars_held']:>5}")

    # Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bars_tested": len(bars),
        "date_range": f"{bars[0]['ts']} → {bars[-1]['ts']}",
        "total_trades": len(trades),
        "win_rate": round(len(wins)/max(len(trades),1)*100, 1),
        "return_pct": round(ret, 2),
        "max_dd_pct": round(max_dd*100, 2),
        "trades": trades,
    }
    out_path = os.path.join(BASE, "data", "5agent_backtest", "event_driven_backtest.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    print("Done.")
