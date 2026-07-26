"""S13: Funding Rate Arbitrage v6 — Hybrid (v5 gate logic + cumulative FR + SHORT support)

v5 → v6 CHANGES:
1. Round number filter widened from 3% to 5% — more opportunities
2. v3 now also checks actual funding rate (not just taker z-score)
3. v4 uses 72h cumulative funding instead of instantaneous FR
4. v4 now supports SHORT direction (when shorts are squeezed)
5. Removed fr > 0.001 cap — high funding IS the signal
6. Added trend context — don't fade strong trends
"""
from .base import BaseStrategy, SignalResult
import numpy as np
import os, json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FUNDING_CSV = os.path.join(BASE_DIR, "data", "forced_movement", "funding_history.csv")

GOOD_HOURS = {2, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16}
BAD_HOURS = {4, 6, 19, 20, 21, 22, 23}


def _read_cumulative_funding(hours=72):
    """Sum funding rates over N hours."""
    if not os.path.exists(FUNDING_CSV):
        return 0, 0
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_ms = int(cutoff.timestamp() * 1000)
    total = 0.0
    count = 0
    try:
        with open(FUNDING_CSV) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    try:
                        ts = int(parts[1])
                        rate = float(parts[2])
                        if ts >= cutoff_ms:
                            total += rate
                            count += 1
                    except (ValueError, IndexError):
                        continue
    except Exception:
        pass
    return total, count


class FundingArbStrategy(BaseStrategy):
    name = 'funding_arb'
    strategy_type = 'flow'
    description = 'v6: cumulative FR + SHORT support + wider round numbers + trend context'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr or df_15m is None or idx is None:
            return None

        ts = data.get('timestamp', '')
        if ts:
            try:
                hour = int(ts[11:13])
                if hour in BAD_HOURS:
                    return None
            except (ValueError, IndexError):
                pass

        closes = df_15m['Close'].values
        volumes = df_15m['Volume'].values
        taker_base = df_15m['Taker buy base asset volume'].values

        if idx < 100:
            return None

        # Try v3 first (taker z-score + round numbers + FR confirmation)
        result = self._check_v3(data, closes, volumes, taker_base, idx, price, atr)
        if result:
            return result

        # Fallback to v4 (cumulative FR + derivatives)
        return self._check_v4(data, df_15m, idx, price, atr)

    def _check_v3(self, data, closes, volumes, taker_base, idx, price, atr):
        """v3: Taker z-score near round numbers + funding rate confirmation."""
        taker_ratio = taker_base[idx] / max(volumes[idx], 1)
        window_start = max(0, idx - 100)
        taker_window = taker_base[window_start:idx+1] / np.maximum(volumes[window_start:idx+1], 1)
        taker_ma = np.mean(taker_window)
        taker_std = np.std(taker_window)
        if taker_std < 0.001:
            return None
        taker_zscore = (taker_ratio - taker_ma) / taker_std

        vol_ma = np.mean(volumes[max(0, idx-20):idx+1])
        vol_ratio = volumes[idx] / max(vol_ma, 1)

        # Round number filter — widened from 3% to 5%
        round_dist = (price % 50) / 50
        near_round = round_dist < 0.05 or round_dist > 0.95

        if not near_round:
            return None
        if vol_ratio < 1.0:
            return None

        direction = None
        if taker_zscore < -1.25:
            direction = 'LONG'
        elif taker_zscore > 1.25:
            direction = 'SHORT'
        else:
            return None

        # Funding rate confirmation (new)
        cum_fr, fr_count = _read_cumulative_funding(72)
        deriv = data.get('derivatives', {})
        ls_ratio = deriv.get('ls_ratio', 1.0)

        # Bonus if funding confirms the direction
        fr_bonus = 0
        if direction == 'LONG' and cum_fr > 0.001:
            # Longs paying high funding = longs squeezed = LONG has edge
            fr_bonus = 0.10
        elif direction == 'SHORT' and cum_fr < -0.001:
            # Shorts paying high funding = shorts squeezed = SHORT has edge
            fr_bonus = 0.10

        # Trend context (new) — don't fade strong trends
        ema_200 = data.get('ema_200', 0)
        if ema_200 and ema_200 > 0:
            dist = (price - ema_200) / ema_200
            # Don't go LONG in strong downtrend, don't go SHORT in strong uptrend
            if direction == 'LONG' and dist < -0.03:
                return None
            if direction == 'SHORT' and dist > 0.03:
                return None

        base = 0.50
        base += min(abs(taker_zscore) - 1.25, 1.0) * 0.15
        base += min(vol_ratio - 1.0, 1.0) * 0.10
        if round_dist < 0.02 or round_dist > 0.98:
            base += 0.10  # very close to round number
        base += fr_bonus
        conviction = min(base, 0.85)

        if conviction < 0.50:
            return None

        sl_dist = 1.5 * atr
        tp_dist = 2.0 * atr

        if direction == 'LONG':
            sl = price - sl_dist
            tp1 = price + tp_dist
            tp2 = price + tp_dist * 1.5
            tp3 = price + tp_dist * 2.0
        else:
            sl = price + sl_dist
            tp1 = price - tp_dist
            tp2 = price - tp_dist * 1.5
            tp3 = price - tp_dist * 2.0

        sl_pct = (sl_dist / price) * 100
        tp1_pct = (tp_dist / price) * 100

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=f"Funding arb v3 {direction}: z={taker_zscore:.2f} "
                   f"round={round_dist:.3f} vol={vol_ratio:.2f} cumFR={cum_fr:.5f}",
            bypass_gates=False,
            details={
                'version': 'v3', 'taker_zscore': float(taker_zscore),
                'round_dist': float(round_dist), 'vol_ratio': float(vol_ratio),
                'cum_funding_72h': float(cum_fr), 'fr_bonus': fr_bonus,
            },
        )

    def _check_v4(self, data, df_15m, idx, price, atr):
        """v4: Cumulative funding squeeze + derivatives + trend context.
        Now supports both LONG and SHORT."""
        deriv = data.get('derivatives', {})
        if not deriv:
            return None

        # 72h cumulative funding (replaces instantaneous FR)
        cum_fr, fr_count = _read_cumulative_funding(72)
        ls_ratio = deriv.get('ls_ratio', 1.0)
        taker = deriv.get('futures_taker_ratio', 0.5)
        ema_200 = data.get('ema_200', 0)

        # Trend context (new)
        trend_dir = None
        if ema_200 and ema_200 > 0:
            dist = (price - ema_200) / ema_200
            if dist > 0.02:
                trend_dir = 'LONG'
            elif dist < -0.02:
                trend_dir = 'SHORT'

        direction = None

        # LONG: longs paying high funding + crowded long → squeeze potential
        if cum_fr > 0.002 and ls_ratio > 1.5:
            # Don't LONG if strong downtrend
            if trend_dir == 'SHORT':
                return None
            direction = 'LONG'

        # SHORT: shorts paying high funding + crowded short → squeeze potential
        elif cum_fr < -0.002 and ls_ratio < 0.65:
            # Don't SHORT if strong uptrend
            if trend_dir == 'LONG':
                return None
            direction = 'SHORT'

        if not direction:
            return None

        # Volume check
        vol_ratio = data.get('vol_ratio', 0) or 0
        if vol_ratio < 0.5:
            return None

        # No FR cap — high funding IS the signal (removed fr > 0.001 check)

        # Conviction
        base = 0.40
        if abs(cum_fr) > 0.005:
            base += 0.15  # strong cumulative FR
        if abs(cum_fr) > 0.01:
            base += 0.10  # very strong
        if ls_ratio > 2.0 or ls_ratio < 0.5:
            base += 0.15  # extreme positioning
        if taker > 1.2 or taker < 0.8:
            base += 0.10  # taker alignment
        if ema_200 and price > ema_200 and direction == 'LONG':
            base += 0.05
        if ema_200 and price < ema_200 and direction == 'SHORT':
            base += 0.05
        if vol_ratio > 1.0:
            base += 0.05

        conviction = min(base, 0.90)
        if conviction < 0.50:
            return None

        # TP/SL
        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(2.0, 3.5, 5.0), sl_mult=1.2)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=f"Funding arb v4 {direction}: cumFR72h={cum_fr:.5f} "
                   f"LS={ls_ratio:.2f} taker={taker:.3f} trend={trend_dir}",
            bypass_gates=False,
            details={
                'version': 'v4', 'cum_funding_72h': float(cum_fr),
                'fr_count': fr_count, 'ls_ratio': float(ls_ratio),
                'taker_ratio': float(taker), 'vol_ratio': float(vol_ratio),
                'trend_dir': trend_dir,
            },
        )
