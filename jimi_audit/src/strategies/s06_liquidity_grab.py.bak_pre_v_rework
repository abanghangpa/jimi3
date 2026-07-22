"""S06: Liquidity Grab v5 — Taker flow momentum at S/R.

Mechanism: Ride the taker flow through S/R levels.

When taker flow z-score > 1.5 at an S/R level, it means the crowd is pushing
THROUGH the level (breakout), not bouncing off it. The S/R level becomes a
launching pad. Follow the flow.

v2 -> v3: Derivatives-filtered S/R reversal (killed — false positive)
v3 -> v4: Regime-adaptive (killed — edge doesn't exist in any regime)
v4 -> v5: Taker flow momentum at S/R (Agent 4 discovery)
  - 49 events, +0.356% mean (24bar), WR=63.3%, PF=2.33, p=0.034
  - No derivatives filter needed — taker flow IS the signal
  - Follow the crowd, don't fade them
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

    for i in range(3, min(lookback, idx)):
        bar_idx = idx - i
        if bar_idx < 3:
            continue
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
    description = "v5: taker flow momentum at S/R (ride the flow)"

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get("price", 0)
        atr = data.get("atr", 0)
        if not price or not atr or df_15m is None or idx is None:
            return None

        # ── SESSION FILTER ──
        ts = data.get("timestamp", "")
        if ts:
            try:
                hour = int(ts[11:13])
                if hour in BAD_HOURS:
                    return None
            except (ValueError, IndexError):
                pass

        # ── TAKER FLOW Z-SCORE ──
        taker_base = df_15m["Taker buy base asset volume"].values.astype(float)
        volume = df_15m["Volume"].values.astype(float)

        if idx < 60:
            return None

        # Compute taker ratio for last 4 bars
        recent_buy = np.sum(taker_base[idx - 4:idx])
        recent_total = np.sum(volume[idx - 4:idx])
        if recent_total == 0:
            return None
        taker_ratio = recent_buy / recent_total

        # Z-score from rolling 60-bar window
        window_buy = taker_base[max(0, idx - 60):idx]
        window_total = volume[max(0, idx - 60):idx]
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

        # ── NEED EXTREME TAKER FLOW ──
        if abs(taker_zscore) < 1.5:
            return None

        # ── DIRECTION: follow the taker flow ──
        direction = "LONG" if taker_zscore > 0 else "SHORT"

        # ── S/R PROXIMITY ──
        sr_levels = _find_sr_levels(df_15m, idx, lookback=96)
        if not sr_levels:
            return None

        best_level = None
        proximity = 0

        for level in sr_levels:
            dist = abs(price - level["price"]) / atr
            if dist < 1.0:
                best_level = level
                proximity = 1.0 - dist
                break

        if not best_level:
            return None

        # ── CONVICTION ──
        base = 0.40
        taker_bonus = min((abs(taker_zscore) - 1.5) * 0.15, 0.25)  # stronger flow = higher conviction
        proximity_bonus = proximity * 0.10
        level_bonus = min(best_level["strength"] / 20, 0.10)
        conviction = min(base + taker_bonus + proximity_bonus + level_bonus, 0.85)

        if conviction < 0.50:
            return None

        # ── TP/SL ──
        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=(f"Liq grab v5 {direction}: taker_z={taker_zscore:.2f} "
                    f"at {best_level['type']} ${best_level['price']:.2f} "
                    f"proximity={proximity:.2f}"),
            bypass_gates=False,
            details={
                "version": "v5",
                "taker_zscore": round(taker_zscore, 3),
                "taker_ratio": round(taker_ratio, 4),
                "level_price": best_level["price"],
                "level_type": best_level["type"],
                "touches": best_level["touches"],
                "proximity": round(proximity, 3),
            },
        )
