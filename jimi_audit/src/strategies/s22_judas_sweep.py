"""S22: Judas Sweep v3 — Hybrid (v3 structural levels + v2 swing clustering)

PRIMARY (v3 — gate tested, 1895 events, +0.10%, p=0.040):
- Price wicks through daily/session high/low
- Rejection wick > 1.5x candle body
- Volume > 1.0x average
- Closes back inside the level (the trap)

FALLBACK (v2 — swing point clustering):
- Fractal swing detection + volume-weighted clustering
"""
from .base import BaseStrategy, SignalResult
import numpy as np
import json, os
from datetime import datetime, timezone

SIGNAL_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          'data', 'judas_sweep_signals.jsonl')

def _log_signal(signal_data):
    try:
        os.makedirs(os.path.dirname(SIGNAL_LOG), exist_ok=True)
        with open(SIGNAL_LOG, 'a') as f:
            f.write(json.dumps(signal_data, default=str) + '\n')
    except Exception:
        pass

def _find_swing_points(series, period=3, mode='high'):
    swings = []
    for i in range(period, len(series) - period):
        if mode == 'high':
            if all(series[i] >= series[i-j] for j in range(1, period+1)) and \
               all(series[i] >= series[i+j] for j in range(1, period+1)):
                swings.append(i)
        else:
            if all(series[i] <= series[i-j] for j in range(1, period+1)) and \
               all(series[i] <= series[i+j] for j in range(1, period+1)):
                swings.append(i)
    return swings

def _cluster_levels(prices, volumes, cluster_pct=0.002, min_touches=2):
    if len(prices) < min_touches:
        return []
    sorted_indices = np.argsort(prices)
    sorted_prices = prices[sorted_indices]
    sorted_vols = volumes[sorted_indices] if volumes is not None else np.ones(len(prices))
    clusters = []
    cp = [sorted_prices[0]]
    cv = [sorted_vols[0]]
    for i in range(1, len(sorted_prices)):
        if (sorted_prices[i] - cp[-1]) / cp[-1] < cluster_pct:
            cp.append(sorted_prices[i])
            cv.append(sorted_vols[i])
        else:
            if len(cp) >= min_touches:
                vw = np.array(cv) / sum(cv)
                clusters.append({'price': np.average(cp, weights=vw), 'touches': len(cp), 'volume': sum(cv), 'strength': len(cp) * (1 + np.log1p(sum(cv)))})
            cp = [sorted_prices[i]]
            cv = [sorted_vols[i]]
    if len(cp) >= min_touches:
        vw = np.array(cv) / sum(cv)
        clusters.append({'price': np.average(cp, weights=vw), 'touches': len(cp), 'volume': sum(cv), 'strength': len(cp) * (1 + np.log1p(sum(cv)))})
    clusters.sort(key=lambda x: x['strength'], reverse=True)
    return clusters[:10]


class JudasSweepStrategy(BaseStrategy):
    name = 'judas_sweep'
    strategy_type = 'event'
    description = 'Hybrid: v3 structural levels (primary) + v2 swing clustering (fallback)'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr or df_15m is None or idx is None:
            return None

        closes = df_15m['Close'].values
        highs = df_15m['High'].values
        lows = df_15m['Low'].values
        volumes = df_15m['Volume'].values
        taker_base = df_15m['Taker buy base asset volume'].values

        if idx < 100:
            return None
        # BEAR REGIME FILTER (skip bear trends - S/R breaks through)
        ema200 = data.get('ema200', 0)
        if ema200:
            dist_from_ema = (price - ema200) / ema200 * 100
            if dist_from_ema < -2.0:
                return None  # bear trend - S/R doesn't hold

        result = self._check_v3_structural(data, closes, highs, lows, volumes, taker_base, idx, price, atr)
        if result:
            return result

        return self._check_v2_clustering(data, df_15m, idx, price, atr)

    def _check_v3_structural(self, data, closes, highs, lows, volumes, taker_base, idx, price, atr):
        current_high = highs[idx]
        current_low = lows[idx]
        current_close = closes[idx]
        vol_ma = np.mean(volumes[max(0, idx-20):idx+1])
        vol_ratio = volumes[idx] / max(vol_ma, 1)

        if vol_ratio < 1.0:
            return None

        daily_high = np.max(highs[max(0, idx-96):idx]) if idx >= 96 else 0
        daily_low = np.min(lows[max(0, idx-96):idx]) if idx >= 96 else 0
        session_high = np.max(highs[max(0, idx-32):idx]) if idx >= 32 else 0
        session_low = np.min(lows[max(0, idx-32):idx]) if idx >= 32 else 0

        levels_high = [l for l in [daily_high, session_high] if l > 0]
        levels_low = [l for l in [daily_low, session_low] if l > 0]

        direction = None
        level_price = None
        level_type = ''

        for level in levels_high:
            if current_high > level * 1.001 and current_close < level:
                wick_up = current_high - current_close
                body = abs(current_close - closes[idx-1]) if idx > 0 else 0.001
                if wick_up > body * 1.5:
                    direction = 'SHORT'
                    level_price = level
                    level_type = 'daily_high' if level == daily_high else 'session_high'
                    break

        if direction is None:
            for level in levels_low:
                if current_low < level * 0.999 and current_close > level:
                    wick_down = current_close - current_low
                    body = abs(current_close - closes[idx-1]) if idx > 0 else 0.001
                    if wick_down > body * 1.5:
                        direction = 'LONG'
                        level_price = level
                        level_type = 'daily_low' if level == daily_low else 'session_low'
                        break

        if direction is None:
            return None

        base = 0.50
        base += min(vol_ratio - 1.0, 1.0) * 0.15
        if 'daily' in level_type:
            base += 0.10
        conviction = min(base, 0.85)

        if conviction < 0.50:
            return None

        sl_dist = 1.5 * atr
        tp_dist = 2.5 * atr

        if direction == 'LONG':
            sl = current_low - atr * 0.3
            tp1 = price + tp_dist
            tp2 = price + tp_dist * 1.5
            tp3 = price + tp_dist * 2.0
        else:
            sl = current_high + atr * 0.3
            tp1 = price - tp_dist
            tp2 = price - tp_dist * 1.5
            tp3 = price - tp_dist * 2.0

        sl_pct = abs(sl - price) / price * 100
        tp1_pct = abs(tp1 - price) / price * 100

        _log_signal({
            'timestamp': str(data.get('timestamp', '')),
            'strategy': self.name, 'direction': direction,
            'entry': price, 'sl': sl, 'tp1': tp1,
            'conviction': conviction, 'level_price': level_price,
            'level_type': level_type, 'vol_ratio': float(vol_ratio), 'version': 'v3',
        })

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.8,
            reason=f"Judas v3 {direction}: swept {level_type} ${level_price:.2f}, "
                   f"wick rejected, vol={vol_ratio:.2f}",
            bypass_gates=True,
            details={'version': 'v3', 'level_price': float(level_price), 'level_type': level_type, 'vol_ratio': float(vol_ratio)},
        )

    def _check_v2_clustering(self, data, df_15m, idx, price, atr):
        closes = df_15m['Close'].values
        highs = df_15m['High'].values
        lows = df_15m['Low'].values
        volumes = df_15m['Volume'].values
        taker_base = df_15m['Taker buy base asset volume'].values

        lookback = 200
        if idx < lookback + 50:
            return None

        current_price = closes[idx]
        current_high = highs[idx]
        current_low = lows[idx]

        atr_pct = atr / current_price
        sweep_min_pct = max(0.0003, atr_pct * 0.2)
        sweep_max_pct = min(0.02, atr_pct * 3.0)
        compression_max_pct = atr_pct * 100
        compression_bars = max(24, int(48 / max(atr_pct / 0.003, 0.5)))

        window_start = max(0, idx - compression_bars)
        range_high = np.max(highs[window_start:idx+1])
        range_low = np.min(lows[window_start:idx+1])
        if range_low <= 0:
            return None
        compression = (range_high - range_low) / range_low * 100
        if compression > compression_max_pct:
            return None

        swing_highs = _find_swing_points(highs[window_start:idx+1], period=3, mode='high')
        swing_lows = _find_swing_points(lows[window_start:idx+1], period=3, mode='low')
        swing_highs = [s + window_start for s in swing_highs]
        swing_lows = [s + window_start for s in swing_lows]

        sh_prices = highs[swing_highs] if swing_highs else np.array([])
        sl_prices = lows[swing_lows] if swing_lows else np.array([])
        sh_vols = volumes[swing_highs] if swing_highs else np.array([])
        sl_vols = volumes[swing_lows] if swing_lows else np.array([])

        resistance_clusters = _cluster_levels(sh_prices, sh_vols, cluster_pct=0.003, min_touches=1)
        support_clusters = _cluster_levels(sl_prices, sl_vols, cluster_pct=0.003, min_touches=1)

        direction = None
        level_price = None
        sweep_pct = 0
        level_type = ''

        if resistance_clusters:
            near_res = [c for c in resistance_clusters if abs(c['price'] - current_price) / current_price < 0.015]
            if near_res:
                best_res = min(near_res, key=lambda x: abs(x['price'] - current_price))
                sp = (current_high - best_res['price']) / best_res['price']
                if sweep_min_pct <= sp <= sweep_max_pct:
                    direction = 'SHORT'
                    level_price = best_res['price']
                    sweep_pct = sp
                    level_type = 'resistance'

        if direction is None and support_clusters:
            near_sup = [c for c in support_clusters if abs(c['price'] - current_price) / current_price < 0.015]
            if near_sup:
                best_sup = min(near_sup, key=lambda x: abs(x['price'] - current_price))
                sp = (best_sup['price'] - current_low) / best_sup['price']
                if sweep_min_pct <= sp <= sweep_max_pct:
                    direction = 'LONG'
                    level_price = best_sup['price']
                    sweep_pct = sp
                    level_type = 'support'

        if direction is None:
            return None

        taker_window = max(0, idx - 3)
        taker_avg = np.mean(taker_base[taker_window:idx+1]) / max(np.mean(volumes[taker_window:idx+1]), 1)

        if direction == 'SHORT' and taker_avg > 0.52:
            return None
        if direction == 'LONG' and taker_avg < 0.48:
            return None

        if direction == 'SHORT':
            rejected = closes[idx] < level_price
        else:
            rejected = closes[idx] > level_price

        if not rejected:
            if direction == 'SHORT' and taker_avg < 0.45:
                rejected = True
            elif direction == 'LONG' and taker_avg > 0.55:
                rejected = True
            else:
                return None

        sweep_score = min(sweep_pct / (sweep_max_pct * 0.5), 1.0) * 0.20
        compression_ratio = compression / compression_max_pct
        compression_score = max(0, (1.0 - compression_ratio)) * 0.15
        if direction == 'SHORT':
            taker_score = max(0, (0.5 - taker_avg) / 0.5) * 0.20
        else:
            taker_score = max(0, (taker_avg - 0.5) / 0.5) * 0.20
        if direction == 'SHORT':
            rejection_dist = (level_price - closes[idx]) / atr if closes[idx] < level_price else 0
        else:
            rejection_dist = (closes[idx] - level_price) / atr if closes[idx] > level_price else 0
        rejection_score = min(rejection_dist / 1.0, 1.0) * 0.20
        level_strength = 0
        if level_type == 'resistance' and resistance_clusters:
            matched = [c for c in resistance_clusters if abs(c['price'] - level_price) / level_price < 0.005]
            if matched:
                level_strength = min(matched[0]['strength'] / 10.0, 1.0) * 0.15
        elif level_type == 'support' and support_clusters:
            matched = [c for c in support_clusters if abs(c['price'] - level_price) / level_price < 0.005]
            if matched:
                level_strength = min(matched[0]['strength'] / 10.0, 1.0) * 0.15

        conviction = sweep_score + compression_score + taker_score + rejection_score + level_strength
        conviction = min(conviction, 0.95)
        if conviction < 0.40:
            return None

        entry = current_price
        if direction == 'SHORT':
            sl = current_high + atr * 0.3
            tp1 = entry - atr * 1.5
            tp2 = entry - atr * 2.5
            tp3 = entry - atr * 4.0
        else:
            sl = current_low - atr * 0.3
            tp1 = entry + atr * 1.5
            tp2 = entry + atr * 2.5
            tp3 = entry + atr * 4.0

        sl_pct = abs(sl - entry) / entry * 100
        tp1_pct = abs(tp1 - entry) / entry * 100
        rr1 = tp1_pct / sl_pct if sl_pct > 0 else 0
        if rr1 < 1.0:
            return None

        _log_signal({
            'timestamp': str(data.get('timestamp', '')),
            'strategy': self.name, 'direction': direction,
            'entry': entry, 'sl': sl, 'tp1': tp1,
            'conviction': conviction, 'level_price': level_price,
            'level_type': level_type, 'sweep_pct': sweep_pct * 100,
            'version': 'v2',
        })

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.8,
            reason=f"Judas v2 {direction}: swept {sweep_pct*100:.2f}% {level_type} ${level_price:.2f}, "
                   f"comp={compression:.2f}%, taker={taker_avg:.3f}",
            bypass_gates=True,
            details={'version': 'v2', 'level_price': float(level_price), 'level_type': level_type,
                     'sweep_pct': float(sweep_pct * 100), 'compression': float(compression), 'taker_avg': float(taker_avg)},
        )
