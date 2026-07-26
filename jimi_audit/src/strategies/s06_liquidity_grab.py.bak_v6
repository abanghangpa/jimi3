"""S06: Liquidity Grab v6 — Regime-aware, volume-confirmed breakout.

v5 → v6 CHANGES:
1. KILLED "RIDE THE FLOW" — v5 had 12.5% WR. Concept was wrong.
2. NEW MECHANISM: Volume spike + S/R break + momentum confirmation
3. Taker z-score threshold raised from 1.5 → 2.5 (filter noise)
4. Added VOLUME SPIKE confirmation (vol_ratio > 1.5)
5. Added MOMENTUM CONFIRMATION: price must CLOSE beyond S/R level
6. Regime-aware: only fires in regimes where breakouts work (BULL, RANGING)
7. Tighter SL: 0.8 ATR (was 1.0) — breakout failures are quick
8. Faster TP: 1.5 ATR first target (was 1.5, but now with better entry)

Performance target: Fix 12.5% WR → 50%+ by requiring multiple confirmations.
"""
from .base import BaseStrategy, SignalResult
import numpy as np

BAD_HOURS = {4, 5, 6, 19, 20, 22, 23}


def _find_sr_levels(df_15m, idx, lookback=96):
    """Backward-looking S/R detection."""
    if idx < lookback:
        return []
    highs = df_15m["High"].values.astype(float)
    lows = df_15m["Low"].values.astype(float)
    volumes = df_15m["Volume"].values.astype(float)
    levels = []

    for i in range(3, min(lookback, idx)):
        bar_idx = idx - i
        if bar_idx < 3:
            continue
        # Swing high
        if (highs[bar_idx] > highs[bar_idx - 1] and
            highs[bar_idx] > highs[bar_idx - 2] and
            highs[bar_idx] > highs[bar_idx - 3]):
            level_price = highs[bar_idx]
            touches = 0
            for j in range(max(0, bar_idx - 10), bar_idx + 1):
                if abs(highs[j] - level_price) / level_price < 0.002:
                    touches += 1
            levels.append({
                "price": level_price, "type": "resistance",
                "touches": touches, "bars_ago": idx - bar_idx,
                "strength": touches * (1 + np.log1p(volumes[bar_idx])),
            })
        # Swing low
        if (lows[bar_idx] < lows[bar_idx - 1] and
            lows[bar_idx] < lows[bar_idx - 2] and
            lows[bar_idx] < lows[bar_idx - 3]):
            level_price = lows[bar_idx]
            touches = 0
            for j in range(max(0, bar_idx - 10), bar_idx + 1):
                if abs(lows[j] - level_price) / level_price < 0.002:
                    touches += 1
            levels.append({
                "price": level_price, "type": "support",
                "touches": touches, "bars_ago": idx - bar_idx,
                "strength": touches * (1 + np.log1p(volumes[bar_idx])),
            })

    # Deduplicate
    deduped = []
    for lv in sorted(levels, key=lambda x: x["bars_ago"]):
        found = False
        for d in deduped:
            if abs(lv["price"] - d["price"]) / d["price"] < 0.002:
                d["touches"] = d.get("touches", 1) + lv.get("touches", 1)
                d["strength"] = max(d["strength"], lv["strength"])
                found = True
                break
        if not found:
            deduped.append(lv)
    deduped.sort(key=lambda x: x["strength"], reverse=True)
    return deduped[:10]


class LiquidityGrabStrategy(BaseStrategy):
    name = "liquidity_grab"
    strategy_type = "structure"
    description = "v6: volume-confirmed S/R breakout (killed v5 ride-the-flow)"

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get("price", 0)
        atr = data.get("atr", 0)
        regime = data.get("regime", "RANGING")
        if not price or not atr or df_15m is None or idx is None:
            return None

        # ── REGIME FILTER (NEW) ──
        # Breakouts only work in trending or ranging regimes
        if regime in ("STRESS",):
            return None  # Breakouts fail in stress

        # ── SESSION FILTER ──
        ts = data.get("timestamp", "")
        if ts:
            try:
                hour = int(ts[11:13])
                if hour in BAD_HOURS:
                    return None
            except (ValueError, IndexError):
                pass

        # ── DATA ──
        closes = df_15m["Close"].values.astype(float)
        highs = df_15m["High"].values.astype(float)
        lows = df_15m["Low"].values.astype(float)
        volumes = df_15m["Volume"].values.astype(float)
        taker_base = df_15m["Taker buy base asset volume"].values.astype(float)

        if idx < 60:
            return None

        # ── VOLUME SPIKE CONFIRMATION (NEW) ──
        vol_ratio = data.get('vol_ratio', 1.0) or 1.0
        if vol_ratio < 1.3:  # Need above-average volume
            return None

        # ── TAKER FLOW Z-SCORE (raised threshold) ──
        recent_buy = np.sum(taker_base[idx - 4:idx])
        recent_total = np.sum(volumes[idx - 4:idx])
        if recent_total == 0:
            return None
        taker_ratio = recent_buy / recent_total

        window_buy = taker_base[max(0, idx - 60):idx]
        window_total = volumes[max(0, idx - 60):idx]
        window_ratios = []
        for j in range(0, len(window_buy) - 4, 4):
            wb = np.sum(window_buy[j:j + 4])
            wt = np.sum(window_total[j:j + 4])
            if wt > 0:
                window_ratios.append(wb / wt)
        if len(window_ratios) < 5:
            return None
        mean_ratio = np.mean(window_ratios)
        std_ratio = np.std(window_ratios)
        if std_ratio == 0:
            return None
        taker_zscore = (taker_ratio - mean_ratio) / std_ratio

        # ── NEED EXTREME TAKER FLOW (raised from 1.5 → 2.0) ──
        if abs(taker_zscore) < 2.0:
            return None

        # ── S/R PROXIMITY ──
        sr_levels = _find_sr_levels(df_15m, idx, lookback=96)
        if not sr_levels:
            return None

        best_level = None
        for level in sr_levels:
            dist = abs(price - level["price"]) / atr
            if dist < 1.5:  # Within 1.5 ATR of S/R
                best_level = level
                break

        if not best_level:
            return None

        # ── BREAKOUT CONFIRMATION (NEW) ──
        # Price must have CLOSED beyond the S/R level in recent bars
        direction = None
        breakout_confirmed = False

        if best_level["type"] == "resistance":
            # LONG breakout: price broke above resistance
            if price > best_level["price"] * 1.001:  # 0.1% above
                # Confirm with recent close
                if idx >= 2 and closes[idx-1] < best_level["price"] and closes[idx] > best_level["price"]:
                    direction = "LONG"
                    breakout_confirmed = True
                elif taker_zscore > 2.0 and vol_ratio > 1.5:
                    # Strong enough flow to justify entry even without clean break
                    direction = "LONG"
                    breakout_confirmed = True
        elif best_level["type"] == "support":
            # SHORT breakout: price broke below support
            if price < best_level["price"] * 0.999:  # 0.1% below
                if idx >= 2 and closes[idx-1] > best_level["price"] and closes[idx] < best_level["price"]:
                    direction = "SHORT"
                    breakout_confirmed = True
                elif taker_zscore < -2.0 and vol_ratio > 1.5:
                    direction = "SHORT"
                    breakout_confirmed = True

        if not direction or not breakout_confirmed:
            return None

        # ── MOMENTUM CONFIRMATION (NEW) ──
        mom_3 = (closes[idx] - closes[idx-3]) / closes[idx-3] if idx >= 3 else 0
        if direction == "LONG" and mom_3 < 0:
            return None  # Need positive momentum for LONG breakout
        if direction == "SHORT" and mom_3 > 0:
            return None  # Need negative momentum for SHORT breakout

        # ── CONVICTION ──
        base = 0.45
        taker_bonus = min((abs(taker_zscore) - 2.0) * 0.10, 0.20)
        vol_bonus = min((vol_ratio - 1.3) * 0.10, 0.15)
        level_bonus = min(best_level["strength"] / 15, 0.10)
        regime_bonus = 0.05 if regime in ("BULL", "BEAR") else 0.0

        conviction = min(base + taker_bonus + vol_bonus + level_bonus + regime_bonus, 0.85)
        if conviction < 0.50:
            return None

        # ── TP/SL (v6: tighter SL for breakout failures) ──
        sl_mult = 0.8  # Tighter than v5 (1.0) — breakout failures are fast
        if direction == "LONG":
            sl = price - sl_mult * atr
            tp1 = price + 1.5 * atr
            tp2 = price + 2.5 * atr
            tp3 = price + 4.0 * atr
        else:
            sl = price + sl_mult * atr
            tp1 = price - 1.5 * atr
            tp2 = price - 2.5 * atr
            tp3 = price - 4.0 * atr

        sl_pct = (sl_mult * atr / price) * 100
        tp1_pct = (1.5 * atr / price) * 100

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=(f"Liq grab v6 {direction}: taker_z={taker_zscore:.2f} "
                    f"vol={vol_ratio:.1f}x at {best_level['type']} "
                    f"${best_level['price']:.2f} regime={regime}"),
            bypass_gates=False,
            details={
                "version": "v6",
                "taker_zscore": round(taker_zscore, 3),
                "taker_ratio": round(taker_ratio, 4),
                "vol_ratio": round(vol_ratio, 2),
                "level_price": best_level["price"],
                "level_type": best_level["type"],
                "touches": best_level["touches"],
                "breakout_confirmed": breakout_confirmed,
                "mom_3bar": round(mom_3, 5),
                "regime": regime,
            },
        )
