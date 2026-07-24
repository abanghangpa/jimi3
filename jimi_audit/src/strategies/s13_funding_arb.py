"""S13: Funding Rate Arbitrage v7.1 — Three independent scenarios.

v7 → v7.1 CHANGES:
1. SCENARIO A: squeeze_breakout LONG (14 trades, 78.6% WR, PF 5.56, p=0.0049)
2. SCENARIO B: 1h momentum >= 1.0% (17 trades, 76.9% WR, PF 3.33, p=0.0302)
3. SCENARIO C: squeeze_breakout LONG + 1h mom >= 0.1% (8 trades, 100% WR, PF inf, p=0.0042)
4. Any scenario can trigger — independent entry paths

v7.1 stats:
- A: 14 trades, 78.6% WR, PF 5.56
- B: 17 trades, 76.9% WR, PF 3.33
- C: 8 trades, 100% WR, PF inf
"""
from .base import BaseStrategy, SignalResult
import numpy as np
import os, json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FUNDING_CSV = os.path.join(BASE_DIR, "data", "forced_movement", "funding_history.csv")

GOOD_HOURS = {2, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16}
BAD_HOURS = {4, 6, 19, 20, 21, 22, 23}

MOM_1H_THRESHOLD_B = 0.01   # 1.0% for Scenario B
MOM_1H_THRESHOLD_C = 0.001  # 0.1% for Scenario C


def _read_cumulative_funding(hours=72):
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


def _check_squeeze_breakout_confirms(data):
    """Scenario A/C: squeeze_breakout fires LONG."""
    strategy_signals = data.get('strategy_signals', {})
    sq = strategy_signals.get('squeeze_breakout', {})
    if sq.get('direction') == 'LONG' and sq.get('fired', False):
        return True
    multi = data.get('multi_strategy', {})
    for sig in multi.get('all_signals', []):
        if sig.get('strategy') == 'squeeze_breakout' and sig.get('direction') == 'LONG':
            return True
    return False


def _check_1h_momentum(df_15m, idx, threshold):
    """Scenario B/C: 1h momentum >= threshold."""
    if df_15m is None or idx is None or idx < 4:
        return False
    closes = df_15m['Close'].values.astype(float)
    mom_1h = (closes[idx] - closes[idx-4]) / closes[idx-4]
    return mom_1h >= threshold


class FundingArbStrategy(BaseStrategy):
    name = 'funding_arb'
    strategy_type = 'flow'
    description = 'v7.1: A: squeeze_breakout | B: 1h_mom>=1.0% | C: squeeze+1h_mom'

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

        # ── SCENARIO CHECKS ──
        scenario_a = _check_squeeze_breakout_confirms(data)
        scenario_b = _check_1h_momentum(df_15m, idx, MOM_1H_THRESHOLD_B)
        scenario_c = _check_squeeze_breakout_confirms(data) and _check_1h_momentum(df_15m, idx, MOM_1H_THRESHOLD_C)

        if not scenario_a and not scenario_b and not scenario_c:
            return None

        # Determine scenario and size
        if scenario_c:
            scenario = 'C'
            size_mult = 0.7  # Both confirm
        elif scenario_a and scenario_b:
            scenario = 'A+B'
            size_mult = 0.7
        elif scenario_a:
            scenario = 'A'
            size_mult = 0.5
        else:
            scenario = 'B'
            size_mult = 0.5

        # Try v3 first
        result = self._check_v3(data, closes, volumes, taker_base, idx, price, atr, scenario, size_mult)
        if result:
            return result

        # Fallback to v4
        return self._check_v4(data, df_15m, idx, price, atr, scenario, size_mult)

    def _check_v3(self, data, closes, volumes, taker_base, idx, price, atr, scenario, size_mult):
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

        cum_fr, fr_count = _read_cumulative_funding(72)
        deriv = data.get('derivatives', {})
        ls_ratio = deriv.get('ls_ratio', 1.0)

        fr_bonus = 0
        if direction == 'LONG' and cum_fr > 0.001:
            fr_bonus = 0.10
        elif direction == 'SHORT' and cum_fr < -0.001:
            fr_bonus = 0.10

        ema_200 = data.get('ema_200', 0)
        if ema_200 and ema_200 > 0:
            dist = (price - ema_200) / ema_200
            if direction == 'LONG' and dist < -0.03:
                return None
            if direction == 'SHORT' and dist > 0.03:
                return None

        base = 0.50
        base += min(abs(taker_zscore) - 1.25, 1.0) * 0.15
        base += min(vol_ratio - 1.0, 1.0) * 0.10
        if round_dist < 0.02 or round_dist > 0.98:
            base += 0.10
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
            size_mult=size_mult,
            reason=f"Funding arb v7.1 [{scenario}] {direction}: z={taker_zscore:.2f} "
                   f"round={round_dist:.3f} cumFR={cum_fr:.5f}",
            bypass_gates=False,
            details={
                'version': 'v7.1',
                'scenario': scenario,
                'provisional': True,
                'taker_zscore': float(taker_zscore),
                'round_dist': float(round_dist),
                'vol_ratio': float(vol_ratio),
                'cum_funding_72h': float(cum_fr),
                'fr_bonus': fr_bonus,
                'note': 'v7.1: A=sq_breakout (78.6%WR,PF5.56) | B=1h_mom>=1% (76.9%WR,PF3.33) | C=sq+1h_mom (100%WR,PFinf)',
            },
        )

    def _check_v4(self, data, df_15m, idx, price, atr, scenario, size_mult):
        deriv = data.get('derivatives', {})
        if not deriv:
            return None

        cum_fr, fr_count = _read_cumulative_funding(72)
        ls_ratio = deriv.get('ls_ratio', 1.0)
        taker = deriv.get('futures_taker_ratio', 0.5)
        ema_200 = data.get('ema_200', 0)

        trend_dir = None
        if ema_200 and ema_200 > 0:
            dist = (price - ema_200) / ema_200
            if dist > 0.02:
                trend_dir = 'LONG'
            elif dist < -0.02:
                trend_dir = 'SHORT'

        direction = None
        if cum_fr > 0.002 and ls_ratio > 1.5:
            if trend_dir == 'SHORT':
                return None
            direction = 'LONG'
        elif cum_fr < -0.002 and ls_ratio < 0.65:
            if trend_dir == 'LONG':
                return None
            direction = 'SHORT'

        if not direction:
            return None

        vol_ratio = data.get('vol_ratio', 0) or 0
        if vol_ratio < 0.5:
            return None

        base = 0.40
        if abs(cum_fr) > 0.005:
            base += 0.15
        if abs(cum_fr) > 0.01:
            base += 0.10
        if ls_ratio > 2.0 or ls_ratio < 0.5:
            base += 0.15
        if taker > 1.2 or taker < 0.8:
            base += 0.10
        if ema_200 and price > ema_200 and direction == 'LONG':
            base += 0.05
        if ema_200 and price < ema_200 and direction == 'SHORT':
            base += 0.05
        if vol_ratio > 1.0:
            base += 0.05

        conviction = min(base, 0.90)
        if conviction < 0.50:
            return None

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(2.0, 3.5, 5.0), sl_mult=1.2)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=size_mult,
            reason=f"Funding arb v7.1 [{scenario}] {direction}: cumFR={cum_fr:.5f} "
                   f"LS={ls_ratio:.2f}",
            bypass_gates=False,
            details={
                'version': 'v7.1',
                'scenario': scenario,
                'provisional': True,
                'cum_funding_72h': float(cum_fr),
                'fr_count': fr_count,
                'ls_ratio': float(ls_ratio),
                'taker_ratio': float(taker),
                'vol_ratio': float(vol_ratio),
                'trend_dir': trend_dir,
                'note': 'v7.1: A=sq_breakout (78.6%WR,PF5.56) | B=1h_mom>=1% (76.9%WR,PF3.33) | C=sq+1h_mom (100%WR,PFinf)',
            },
        )
