"""
M20: Failed Breakout Detector

Detects the "breakout → trap → breakdown" pattern where:
  1. Price breaks above/below a significant level (sweep of liquidity)
  2. The breakout FAILS — price cannot hold above/below the level
  3. Trapped participants are forced to exit, causing a cascade in the
     opposite direction

This pattern is distinct from M14 (sweep-retest-reclaim) which detects
successful reclaim after a sweep. M20 specifically catches the FAILURE
case — when the sweep doesn't reclaim and instead traps participants.

Pattern taxonomy:
  FAILED_BREAKOUT  — breakout attempt + failure + reversal → high score (contrarian)
  WEAK_BREAKOUT    — breakout with poor quality but hasn't failed yet → warning
  NO_BREAKOUT      — no breakout attempt detected → neutral (pass-through)
  HOLDING          — breakout is holding so far → neutral/low score

Signal flow:
  Phase 1: Detect breakout attempt (price crosses significant level)
  Phase 2: Score breakout quality (wick, taker, volume, body)
  Phase 3: Detect failure (price returns below breakout level)
  Phase 4: Score reversal conviction (volume spike, taker flip, momentum)
  Phase 5: Compute contrarian ICS score
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# DEFAULTS
# ═══════════════════════════════════════════════════════════════

_M20_DEFAULTS = {
    'M20_ENABLED': True,
    'M20_WEIGHT': 0.10,

    # Breakout detection
    'M20_BREAKOUT_LOOKBACK': 48,        # bars to scan for breakout attempt
    'M20_BREAKOUT_LEVEL_ATR_MULT': 0.5, # min distance from level to count as breakout
    'M20_BREAKOUT_MIN_RANGE_PCT': 0.3,  # min candle range % for breakout bar

    # Breakout quality scoring
    'M20_WICK_REJECTION_RATIO': 0.40,   # wick/body ratio that signals rejection
    'M20_TAKER_SELL_THRESHOLD': 0.42,   # taker ratio below this = sellers active
    'M20_TAKER_BUY_THRESHOLD': 0.58,    # taker ratio above this = buyers active
    'M20_VOL_FADE_THRESHOLD': 0.7,      # vol < 0.7x avg on breakout = fading

    # Failure detection
    'M20_FAILURE_BARS': 8,              # max bars after breakout to detect failure
    'M20_FAILURE_RETURN_PCT': 0.3,      # price must return this % below breakout level
    'M20_HOLD_BARS': 4,                 # bars breakout must hold to be "holding"

    # Reversal conviction
    'M20_REVERSAL_VOL_MULT': 1.3,       # reversal candle needs 1.3x avg volume
    'M20_REVERSAL_TAKER_FLIP': True,    # reversal taker must flip direction
    'M20_REVERSAL_BODY_MIN': 0.5,       # min body/range ratio for conviction

    # Scoring
    'M20_STRONG_SCORE': 0.85,           # strong failed breakout (contrarian)
    'M20_MODERATE_SCORE': 0.65,         # moderate failed breakout
    'M20_WEAK_BREAKOUT_SCORE': 0.35,    # weak breakout detected (warning)
    'M20_HOLDING_SCORE': 0.50,          # breakout holding — neutral
    'M20_NO_BREAKOUT_SCORE': 0.50,      # no breakout detected — neutral
}


def _cfg(config, key):
    if config and key in config:
        return config[key]
    return _M20_DEFAULTS[key]


# ═══════════════════════════════════════════════════════════════
# PHASE 1: BREAKOUT DETECTION
# ═══════════════════════════════════════════════════════════════

def _find_breakout_levels(df_15m, idx, lookback):
    """Find significant price levels near current price.

    Uses recent swing highs/lows and high-volume nodes as reference levels.
    Returns list of (level_price, level_type) where type is 'HIGH' or 'LOW'.
    """
    if idx < lookback:
        lookback = idx

    highs = df_15m['High'].values[max(0, idx-lookback):idx+1].astype(float)
    lows = df_15m['Low'].values[max(0, idx-lookback):idx+1].astype(float)
    closes = df_15m['Close'].values[max(0, idx-lookback):idx+1].astype(float)

    levels = []

    # Find local swing highs/lows using fractal detection
    for i in range(2, len(highs) - 2):
        # Swing high: higher than 2 bars on each side
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
            highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            levels.append((float(highs[i]), 'HIGH'))

        # Swing low: lower than 2 bars on each side
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
            lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            levels.append((float(lows[i]), 'LOW'))

    # Deduplicate nearby levels (within 0.1%)
    if not levels:
        return []

    levels.sort(key=lambda x: x[0])
    deduped = [levels[0]]
    for lvl in levels[1:]:
        if abs(lvl[0] - deduped[-1][0]) / deduped[-1][0] > 0.001:
            deduped.append(lvl)

    return deduped


def _detect_breakout_attempt(df_15m, idx, levels, atr, config):
    """Detect if price has attempted a breakout of any significant level.

    Returns list of breakout attempts:
      (level_price, level_type, breakout_bar_idx, breakout_direction, quality_signals)
    """
    if not levels or atr is None or atr <= 0:
        return []

    cfg_breakout_lookback = _cfg(config, 'M20_BREAKOUT_LOOKBACK')
    cfg_min_range = _cfg(config, 'M20_BREAKOUT_MIN_RANGE_PCT') / 100.0
    cfg_atr_mult = _cfg(config, 'M20_BREAKOUT_LEVEL_ATR_MULT')

    lookback = min(cfg_breakout_lookback, idx)
    attempts = []

    for level_price, level_type in levels:
        # Scan recent bars for breakout attempts
        for offset in range(lookback, 0, -1):
            bar_idx = idx - offset
            if bar_idx < 0:
                continue

            bar_high = float(df_15m['High'].iloc[bar_idx])
            bar_low = float(df_15m['Low'].iloc[bar_idx])
            bar_close = float(df_15m['Close'].iloc[bar_idx])
            bar_open = float(df_15m['Open'].iloc[bar_idx])
            bar_range = (bar_high - bar_low) / bar_close if bar_close > 0 else 0

            # Skip tiny candles
            if bar_range < cfg_min_range:
                continue

            min_distance = atr * cfg_atr_mult

            # Upside breakout: price crosses above resistance
            if level_type == 'HIGH' and bar_high > level_price + min_distance:
                # Check it wasn't already broken long ago
                prev_high = float(df_15m['High'].iloc[max(0, bar_idx-5):bar_idx].max()) if bar_idx > 5 else 0
                if prev_high > level_price + min_distance:
                    continue  # already broken, not a fresh breakout

                attempts.append({
                    'level': level_price,
                    'level_type': level_type,
                    'bar_idx': bar_idx,
                    'direction': 'UPSIDE',
                    'break_high': bar_high,
                    'break_close': bar_close,
                    'break_open': bar_open,
                    'break_low': bar_low,
                    'break_range_pct': bar_range,
                })
                break  # one attempt per level

            # Downside breakout: price crosses below support
            elif level_type == 'LOW' and bar_low < level_price - min_distance:
                prev_low = float(df_15m['Low'].iloc[max(0, bar_idx-5):bar_idx].min()) if bar_idx > 5 else float('inf')
                if prev_low < level_price - min_distance:
                    continue

                attempts.append({
                    'level': level_price,
                    'level_type': level_type,
                    'bar_idx': bar_idx,
                    'direction': 'DOWNSIDE',
                    'break_low': bar_low,
                    'break_close': bar_close,
                    'break_open': bar_open,
                    'break_high': bar_high,
                    'break_range_pct': bar_range,
                })
                break

    return attempts


# ═══════════════════════════════════════════════════════════════
# PHASE 2: BREAKOUT QUALITY SCORING
# ═══════════════════════════════════════════════════════════════

def _score_breakout_quality(df_15m, attempt, config):
    """Score the quality of a breakout attempt.

    Low quality = likely to fail (long wicks, low taker, fading volume).
    High quality = likely to hold (strong body, high volume, taker aligned).

    Returns dict with quality signals and overall quality score (0-1).
    """
    bar_idx = attempt['bar_idx']
    direction = attempt['direction']

    bar_high = float(df_15m['High'].iloc[bar_idx])
    bar_low = float(df_15m['Low'].iloc[bar_idx])
    bar_close = float(df_15m['Close'].iloc[bar_idx])
    bar_open = float(df_15m['Open'].iloc[bar_idx])
    bar_body = bar_close - bar_open
    bar_range = bar_high - bar_low

    if bar_range == 0:
        return {'quality': 0.5, 'signals': []}

    # Taker ratio
    taker = float(df_15m['Taker buy base asset volume'].iloc[bar_idx]) if 'Taker buy base asset volume' in df_15m.columns else 0.5
    vol = float(df_15m['Volume'].iloc[bar_idx])
    taker_ratio = taker / vol if vol > 0 else 0.5

    # Volume vs MA
    vol_ma = float(df_15m['Volume'].iloc[max(0, bar_idx-19):bar_idx+1].mean())
    vol_ratio = vol / vol_ma if vol_ma > 0 else 1.0

    signals = []
    quality = 0.5  # start neutral

    if direction == 'UPSIDE':
        # Wick analysis for upside breakout
        wick_up = bar_high - max(bar_open, bar_close)
        wick_down = min(bar_open, bar_close) - bar_low
        wick_ratio = wick_up / bar_range if bar_range > 0 else 0

        # Long upper wick = rejection
        if wick_ratio > _cfg(config, 'M20_WICK_REJECTION_RATIO'):
            signals.append(f'wick_rejection={wick_ratio:.2f}')
            quality -= 0.15

        # Red close on breakout = weak
        if bar_body < 0:
            signals.append('red_close')
            quality -= 0.15

        # Low taker = sellers defending
        if taker_ratio < _cfg(config, 'M20_TAKER_SELL_THRESHOLD'):
            signals.append(f'sellers_active={taker_ratio:.3f}')
            quality -= 0.10

        # Volume fade
        if vol_ratio < _cfg(config, 'M20_VOL_FADE_THRESHOLD'):
            signals.append(f'vol_fade={vol_ratio:.2f}x')
            quality -= 0.10

        # Strong body = conviction
        body_ratio = abs(bar_body) / bar_range
        if body_ratio > 0.6 and bar_body > 0:
            signals.append(f'strong_body={body_ratio:.2f}')
            quality += 0.10

        # High volume = conviction
        if vol_ratio > 1.5:
            signals.append(f'vol_spike={vol_ratio:.2f}x')
            quality += 0.05

    elif direction == 'DOWNSIDE':
        wick_down = min(bar_open, bar_close) - bar_low
        wick_up = bar_high - max(bar_open, bar_close)
        wick_ratio = wick_down / bar_range if bar_range > 0 else 0

        if wick_ratio > _cfg(config, 'M20_WICK_REJECTION_RATIO'):
            signals.append(f'wick_rejection={wick_ratio:.2f}')
            quality -= 0.15

        if bar_body > 0:  # green close on downside breakout = weak
            signals.append('green_close')
            quality -= 0.15

        if taker_ratio > _cfg(config, 'M20_TAKER_BUY_THRESHOLD'):
            signals.append(f'buyers_active={taker_ratio:.3f}')
            quality -= 0.10

        if vol_ratio < _cfg(config, 'M20_VOL_FADE_THRESHOLD'):
            signals.append(f'vol_fade={vol_ratio:.2f}x')
            quality -= 0.10

        body_ratio = abs(bar_body) / bar_range
        if body_ratio > 0.6 and bar_body < 0:
            signals.append(f'strong_body={body_ratio:.2f}')
            quality += 0.10

        if vol_ratio > 1.5:
            signals.append(f'vol_spike={vol_ratio:.2f}x')
            quality += 0.05

    quality = max(0.0, min(1.0, quality))

    return {
        'quality': quality,
        'signals': signals,
        'taker_ratio': taker_ratio,
        'vol_ratio': vol_ratio,
        'wick_ratio': wick_ratio if 'wick_ratio' in dir() else 0,
    }


# ═══════════════════════════════════════════════════════════════
# PHASE 3: FAILURE DETECTION
# ═══════════════════════════════════════════════════════════════

def _detect_breakout_failure(df_15m, idx, attempt, config):
    """Check if a breakout attempt has failed.

    A breakout fails when price returns below/above the breakout level
    within N bars.

    Returns dict with failure status and details.
    """
    bar_idx = attempt['bar_idx']
    level = attempt['level']
    direction = attempt['direction']
    failure_bars = _cfg(config, 'M20_FAILURE_BARS')
    return_pct = _cfg(config, 'M20_FAILURE_RETURN_PCT') / 100.0

    # Always scan from the breakout bar to the current bar.
    # The old code capped end_idx at bar_idx + failure_bars + 1, which meant
    # that for breakouts older than failure_bars, current_close was read from
    # an intermediate bar instead of the actual current bar — causing stale
    # breakouts (e.g. 14 bars ago) to be judged on outdated price data.
    end_idx = idx + 1

    if end_idx <= bar_idx + 2:
        return {'failed': False, 'status': 'TOO_EARLY', 'bars_since': 0}

    bars_since = end_idx - bar_idx - 1

    if direction == 'UPSIDE':
        # Failed if price returned below the breakout level
        post_closes = df_15m['Close'].values[bar_idx+1:end_idx].astype(float)
        post_lows = df_15m['Low'].values[bar_idx+1:end_idx].astype(float)

        if len(post_closes) == 0:
            return {'failed': False, 'status': 'TOO_EARLY', 'bars_since': 0}

        min_close = float(np.min(post_closes))
        min_low = float(np.min(post_lows))
        return_distance = (level - min_close) / level if level > 0 else 0

        # Check if price is still above the breakout level
        current_close = float(df_15m['Close'].iloc[end_idx-1])
        if current_close > level:
            # Still holding above — check if weakening
            hold_bars = _cfg(config, 'M20_HOLD_BARS')
            if bars_since >= hold_bars:
                return {
                    'failed': False,
                    'status': 'HOLDING',
                    'bars_since': bars_since,
                    'current_above': current_close - level,
                }
            return {'failed': False, 'status': 'HOLDING_WEAK', 'bars_since': bars_since}

        # Price returned below level
        if return_distance >= return_pct:
            # Find the failure bar (first close below level)
            failure_bar = bar_idx + 1
            for i, c in enumerate(post_closes):
                if c < level:
                    failure_bar = bar_idx + 1 + i
                    break

            return {
                'failed': True,
                'status': 'FAILED',
                'bars_since': bars_since,
                'failure_bar': failure_bar,
                'return_distance': return_distance,
                'min_close': min_close,
            }

        # Price dipped below but not enough
        if min_low < level:
            return {
                'failed': False,
                'status': 'TESTING',
                'bars_since': bars_since,
                'min_low': min_low,
            }

    elif direction == 'DOWNSIDE':
        post_closes = df_15m['Close'].values[bar_idx+1:end_idx].astype(float)
        post_highs = df_15m['High'].values[bar_idx+1:end_idx].astype(float)

        if len(post_closes) == 0:
            return {'failed': False, 'status': 'TOO_EARLY', 'bars_since': 0}

        max_close = float(np.max(post_closes))
        return_distance = (max_close - level) / level if level > 0 else 0

        current_close = float(df_15m['Close'].iloc[end_idx-1])

        # Current price still below level — breakout is holding
        if current_close < level:
            hold_bars = _cfg(config, 'M20_HOLD_BARS')
            if bars_since >= hold_bars:
                return {
                    'failed': False,
                    'status': 'HOLDING',
                    'bars_since': bars_since,
                    'current_below': level - current_close,
                }
            return {'failed': False, 'status': 'HOLDING_WEAK', 'bars_since': bars_since}

        # Current price is above level — breakout may have failed.
        # But require that price SUSTAINED above the level (max close far
        # enough above) to filter out brief wicks that immediately reversed.
        if return_distance >= return_pct:
            failure_bar = bar_idx + 1
            for i, c in enumerate(post_closes):
                if c > level:
                    failure_bar = bar_idx + 1 + i
                    break

            return {
                'failed': True,
                'status': 'FAILED',
                'bars_since': bars_since,
                'failure_bar': failure_bar,
                'return_distance': return_distance,
                'max_close': max_close,
            }

        if float(np.max(post_highs)) > level:
            return {
                'failed': False,
                'status': 'TESTING',
                'bars_since': bars_since,
            }

    return {'failed': False, 'status': 'TOO_EARLY', 'bars_since': 0}


# ═══════════════════════════════════════════════════════════════
# PHASE 4: REVERSAL CONVICTION
# ═══════════════════════════════════════════════════════════════

def _score_reversal_conviction(df_15m, idx, attempt, failure, trade_direction, config):
    """Score the conviction of the reversal after a failed breakout.

    The reversal should show:
    - Volume expansion (trapped participants exiting)
    - Taker ratio aligned with reversal direction
    - Strong body candles (no indecision)

    Returns score 0-1 and details.
    """
    if not failure.get('failed'):
        return 0.5, {'status': 'NO_FAILURE'}

    failure_bar = failure.get('failure_bar', idx)
    direction = attempt['direction']

    # Look at candles from failure bar to current
    start = max(failure_bar, idx - 8)
    end = idx + 1

    if end - start < 2:
        return 0.5, {'status': 'TOO_FEW_BARS'}

    closes = df_15m['Close'].values[start:end].astype(float)
    opens = df_15m['Open'].values[start:end].astype(float)
    highs = df_15m['High'].values[start:end].astype(float)
    lows = df_15m['Low'].values[start:end].astype(float)
    volumes = df_15m['Volume'].values[start:end].astype(float)

    # Taker for the reversal bars
    if 'Taker buy base asset volume' in df_15m.columns:
        takers = df_15m['Taker buy base asset volume'].values[start:end].astype(float)
    else:
        takers = volumes * 0.5

    # Volume baseline
    vol_ma = float(df_15m['Volume'].iloc[max(0, start-19):start].mean()) if start > 19 else float(np.mean(volumes))

    score = 0.5
    details = []

    # 1. Volume expansion on reversal
    reversal_vol = float(np.mean(volumes))
    vol_mult = reversal_vol / vol_ma if vol_ma > 0 else 1.0
    if vol_mult > _cfg(config, 'M20_REVERSAL_VOL_MULT'):
        score += 0.15
        details.append(f'vol_expansion={vol_mult:.2f}x')

    # 2. Taker alignment with reversal direction
    if trade_direction == 'SHORT':
        # Reversal should show selling pressure
        avg_taker = float(np.mean(takers / np.where(volumes > 0, volumes, 1)))
        if avg_taker < 0.45:
            score += 0.10
            details.append(f'seller_dominant={avg_taker:.3f}')
    elif trade_direction == 'LONG':
        avg_taker = float(np.mean(takers / np.where(volumes > 0, volumes, 1)))
        if avg_taker > 0.55:
            score += 0.10
            details.append(f'buyer_dominant={avg_taker:.3f}')

    # 3. Body conviction (strong directional candles)
    bodies = np.abs(closes - opens)
    ranges = highs - lows
    ranges = np.where(ranges > 0, ranges, 1)
    body_ratios = bodies / ranges
    avg_body_ratio = float(np.mean(body_ratios))

    if avg_body_ratio > _cfg(config, 'M20_REVERSAL_BODY_MIN'):
        score += 0.10
        details.append(f'body_conviction={avg_body_ratio:.2f}')

    # 4. Consecutive candles in reversal direction
    if trade_direction == 'SHORT':
        red_count = sum(1 for i in range(len(closes)) if closes[i] < opens[i])
    else:
        red_count = sum(1 for i in range(len(closes)) if closes[i] > opens[i])

    consec_ratio = red_count / len(closes) if len(closes) > 0 else 0
    if consec_ratio > 0.6:
        score += 0.05
        details.append(f'consecutive={red_count}/{len(closes)}')

    # 5. Speed of failure (faster = more trapped = bigger cascade)
    bars_to_fail = failure.get('bars_since', 99)
    if bars_to_fail <= 3:
        score += 0.10
        details.append(f'fast_failure={bars_to_fail}bars')
    elif bars_to_fail <= 5:
        score += 0.05
        details.append(f'moderate_failure={bars_to_fail}bars')

    score = max(0.0, min(1.0, score))

    return score, {
        'status': 'SCORED',
        'vol_mult': round(vol_mult, 2),
        'avg_body_ratio': round(avg_body_ratio, 2),
        'bars_to_fail': bars_to_fail,
        'details': details,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════

def score_m20(df_15m, idx, direction, sr_levels=None, magnets=None,
              config=None, atr_1h=None):
    """Score the Failed Breakout Detector for the current bar.

    Args:
        df_15m: 15m OHLCV DataFrame
        idx: current bar index
        direction: trade direction ('LONG', 'SHORT', 'NEUTRAL')
        sr_levels: optional list of (price, strength, type, touches, bounces) from M5
        magnets: optional list of (price, volume, strength) from volume profile
        config: config dict
        atr_1h: 1H ATR for breakout distance calculation

    Returns:
        (status, score, details)
        status: 'PASS', 'FAIL', 'SKIP', 'NEUTRAL'
        score: 0.0 to 1.0 (contrarian — high score = failed breakout = trade in that direction)
        details: dict with diagnostic info
    """
    cfg = config or _M20_DEFAULTS

    if not cfg.get('M20_ENABLED', _M20_DEFAULTS['M20_ENABLED']):
        return 'SKIP', 0.5, {'status': 'DISABLED'}

    if idx < 50:
        return 'SKIP', 0.5, {'status': 'INSUFFICIENT_DATA'}

    # Compute ATR if not provided
    if atr_1h is None or (isinstance(atr_1h, float) and np.isnan(atr_1h)):
        atr_series = df_15m['atr'] if 'atr' in df_15m.columns else None
        if atr_series is not None:
            atr_1h = float(atr_series.iloc[idx]) if not np.isnan(atr_series.iloc[idx]) else None
        if atr_1h is None:
            # Fallback: compute from recent bars
            highs = df_15m['High'].values[max(0,idx-13):idx+1].astype(float)
            lows = df_15m['Low'].values[max(0,idx-13):idx+1].astype(float)
            closes = df_15m['Close'].values[max(0,idx-13):idx+1].astype(float)
            tr = np.maximum(highs[1:] - lows[1:],
                           np.maximum(np.abs(highs[1:] - closes[:-1]),
                                     np.abs(lows[1:] - closes[:-1])))
            atr_1h = float(np.mean(tr)) if len(tr) > 0 else 0

    # Build breakout reference levels
    levels = _find_breakout_levels(df_15m, idx, _cfg(cfg, 'M20_BREAKOUT_LOOKBACK'))

    # Add S/R levels if provided
    if sr_levels:
        for sr in sr_levels:
            if len(sr) >= 3:
                price, strength, sr_type = sr[0], sr[1], sr[2]
                level_type = 'HIGH' if sr_type == 'RESISTANCE' else 'LOW'
                levels.append((float(price), level_type))

    # Add magnet levels if provided
    if magnets:
        for mag in magnets[:5]:
            if len(mag) >= 2:
                price = mag[0]
                # Determine if above or below current price
                current = float(df_15m['Close'].iloc[idx])
                level_type = 'HIGH' if price > current else 'LOW'
                levels.append((float(price), level_type))

    if not levels:
        return 'NEUTRAL', 0.5, {'status': 'NO_LEVELS'}

    # Phase 1: Detect breakout attempts
    attempts = _detect_breakout_attempt(df_15m, idx, levels, atr_1h, cfg)

    if not attempts:
        return 'NEUTRAL', _cfg(cfg, 'M20_NO_BREAKOUT_SCORE'), {
            'status': 'NO_BREAKOUT',
            'levels_checked': len(levels),
        }

    # Phase 2-4: Score each attempt
    best_result = None
    best_score = 0.5

    for attempt in attempts:
        # Phase 2: Breakout quality
        quality = _score_breakout_quality(df_15m, attempt, cfg)

        # Phase 3: Failure detection
        failure = _detect_breakout_failure(df_15m, idx, attempt, cfg)

        # Phase 4: Reversal conviction
        reversal_score, reversal_details = _score_reversal_conviction(
            df_15m, idx, attempt, failure, direction, cfg)

        # Determine the contrarian direction
        if attempt['direction'] == 'UPSIDE':
            contrarian_dir = 'SHORT'
        else:
            contrarian_dir = 'LONG'

        # Compute final score
        if failure['status'] == 'FAILED':
            # Failed breakout — score based on quality (low = more likely trap)
            # and reversal conviction
            base_score = _cfg(cfg, 'M20_STRONG_SCORE')

            # Adjust by breakout quality (lower quality = stronger signal)
            quality_adj = (1.0 - quality['quality']) * 0.20

            # Adjust by reversal conviction
            reversal_adj = (reversal_score - 0.5) * 0.30

            final_score = base_score + quality_adj + reversal_adj

            # Boost if trade direction matches contrarian direction
            if direction == contrarian_dir:
                final_score = min(1.0, final_score + 0.05)
            elif direction != 'NEUTRAL' and direction != contrarian_dir:
                final_score = max(0.0, final_score - 0.20)

            status = 'PASS'

        elif failure['status'] == 'HOLDING':
            # Breakout is holding — not a failed breakout
            final_score = _cfg(cfg, 'M20_HOLDING_SCORE')
            status = 'NEUTRAL'

        elif failure['status'] in ('HOLDING_WEAK', 'TESTING'):
            # Weak hold — potential failure brewing
            final_score = _cfg(cfg, 'M20_WEAK_BREAKOUT_SCORE')
            if quality['quality'] < 0.4:
                final_score -= 0.10  # low quality breakout, likely to fail
            status = 'NEUTRAL'

        else:
            # TOO_EARLY — breakout just happened
            if quality['quality'] < 0.35:
                final_score = _cfg(cfg, 'M20_WEAK_BREAKOUT_SCORE')
                status = 'NEUTRAL'
            else:
                final_score = _cfg(cfg, 'M20_NO_BREAKOUT_SCORE')
                status = 'NEUTRAL'

        final_score = max(0.0, min(1.0, final_score))

        result = {
            'status': 'FAILED' if failure.get('failed') else failure['status'],
            'breakout_direction': attempt['direction'],
            'contrarian_direction': contrarian_dir,
            'level': round(attempt['level'], 2),
            'breakout_quality': round(quality['quality'], 3),
            'quality_signals': quality['signals'],
            'failure': failure,
            'reversal_score': round(reversal_score, 3),
            'reversal_details': reversal_details,
            'bars_since_breakout': idx - attempt['bar_idx'],
            'final_score': round(final_score, 4),
        }

        # Track best (most actionable) result
        if abs(final_score - 0.5) > abs(best_score - 0.5):
            best_score = final_score
            best_result = result

    if best_result is None:
        return 'NEUTRAL', 0.5, {'status': 'NO_ACTIONABLE'}

    # Determine status
    if best_result['status'] == 'FAILED':
        if direction == best_result['contrarian_direction']:
            status = 'PASS'
        elif direction == 'NEUTRAL':
            status = 'PASS'  # failed breakout detected — signal is valid
        else:
            status = 'FAIL'  # failed breakout is against our trade direction
    else:
        status = 'NEUTRAL'

    return status, best_score, best_result


# ═══════════════════════════════════════════════════════════════
# FORMATTER
# ═══════════════════════════════════════════════════════════════

def format_failed_breakout(result):
    """Format failed breakout result for display."""
    if not result or result.get('status') in ('NO_LEVELS', 'NO_ACTIONABLE',
                                                'DISABLED', 'INSUFFICIENT_DATA'):
        return ''

    status = result.get('status', '?')
    direction = result.get('breakout_direction', '?')
    contrarian = result.get('contrarian_direction', '?')
    level = result.get('level', 0)
    quality = result.get('breakout_quality', 0)
    failure = result.get('failure', {})
    reversal = result.get('reversal_details', {})
    score = result.get('final_score', 0.5)
    bars_ago = result.get('bars_since_breakout', 0)

    lines = []
    lines.append(f"\n  💥 FAILED BREAKOUT DETECTOR (M20)")

    if status == 'FAILED':
        lines.append(f"  Status: ❌ FAILED {direction} BREAKOUT at ${level:.2f}")
        lines.append(f"  Contrarian: → {contrarian}  (score={score:.3f})")
        lines.append(f"  Breakout quality: {quality:.2f} (low = more likely trap)")
        if result.get('quality_signals'):
            lines.append(f"    Signals: {', '.join(result['quality_signals'])}")
        lines.append(f"  Failure speed: {failure.get('bars_since', '?')} bars")
        lines.append(f"  Return distance: {failure.get('return_distance', 0)*100:.2f}%")
        if reversal.get('details'):
            for d in reversal['details']:
                lines.append(f"    ✅ {d}")
    elif status in ('HOLDING', 'HOLDING_WEAK'):
        lines.append(f"  Status: ⏳ {direction} breakout HOLDING at ${level:.2f}")
        lines.append(f"  Quality: {quality:.2f}  |  {bars_ago} bars ago")
    elif status == 'TESTING':
        lines.append(f"  Status: 🔍 {direction} breakout TESTING at ${level:.2f}")
        lines.append(f"  Quality: {quality:.2f}  |  {bars_ago} bars ago")
    else:
        lines.append(f"  Status: {status} at ${level:.2f}")

    return '\n'.join(lines)
