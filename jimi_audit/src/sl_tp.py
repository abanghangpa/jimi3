"""
Liquidity-Aware SL/TP Placement

Shared by both scanner and engine so backtest matches live behavior.

Framework:
  SL  → Place in nearest liquidity void (no magnets/S/R/stops nearby)
        Fallback to ATR-based if no void found
  TP1 → Nearest unswept magnet/pool in trade direction
  TP2 → Next unswept pool beyond TP1
  TP3 → Furthest unswept pool or ATR extension
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# DEFAULTS
# ═══════════════════════════════════════════════════════════════

_SL_TP_DEFAULTS = {
    # SL void detection
    'SL_VOID_BUFFER_PCT': 0.003,       # 0.3% buffer around levels = "clustered"
    'SL_VOID_MIN_DIST_PCT': 0.002,     # min SL distance as % of price
    'SL_VOID_MAX_DIST_PCT': 0.025,     # max SL distance (2.5% cap)
    'SL_ATR_STD': 1.0,  # ATR fallback multiplier (SL floor is 30)                 # ATR fallback multiplier
    'SL_HARD_MAX_PCT': 0.02,           # hard max SL distance (2%)
    'SL_MIN_DOLLAR': 30.0,              # minimum SL distance in dollars

    # TP targeting
    'TP1_USE_MAGNET': True,            # target nearest unswept magnet for TP1
    'TP1_MAGNET_MIN_DIST_PCT': 0.002,  # min distance to magnet (avoid entry-adjacent)
    'TP1_ATR': 1.6,                    # ATR fallback for TP1 (~$15 at current ATR)
    'TP1_MIN_DOLLAR': 10.0,            # minimum TP1 distance in dollars
    'TP2_ATR': 2.5,                    # ATR fallback for TP2
    'TP3_ATR': 4.0,                    # ATR fallback for TP3

    # Sweep gate
    'M14_ENTRY_GATE': False,           # require M14 sweep before signaling
}


def _cfg(config, key, default=None):
    if config and key in config:
        return config[key]
    if key in _SL_TP_DEFAULTS:
        return _SL_TP_DEFAULTS.get(key)
    return default


# ═══════════════════════════════════════════════════════════════
# LIQUIDITY VOID DETECTION (for SL)
# ═══════════════════════════════════════════════════════════════

def _collect_levels(price, direction, magnets, sr_levels, liq_levels, cfg):
    """Collect all liquidity levels near price that define 'clusters'."""
    buffer_pct = _cfg(cfg, 'SL_VOID_BUFFER_PCT')
    levels = []

    # Volume profile magnets (HVNs)
    if magnets:
        for m in magnets:
            if isinstance(m, (list, tuple, np.ndarray)) and len(m) >= 1:
                p = m[0]
            elif isinstance(m, (float, int, np.float64, np.int64)):
                p = float(m)
            else:
                continue
            if abs(p - price) / price < 0.05:  # within 5%
                levels.append(p)

    # S/R levels
    if sr_levels:
        for sr in sr_levels:
            p = sr[0]  # (price, strength, touches, bounces, type)
            if abs(p - price) / price < 0.05:
                levels.append(p)

    # Liquidation/stop clusters from M15
    if liq_levels and isinstance(liq_levels, dict):
        for side in ('above', 'below'):
            for lvl in liq_levels.get(side, []):
                p = lvl.get('price', 0)
                if p > 0 and abs(p - price) / price < 0.05:
                    levels.append(p)

    return sorted(set(levels))


def find_liquidity_void(price, direction, magnets, sr_levels, liq_levels, atr_1h, cfg=None):
    """Find the nearest price level in a liquidity void for SL placement.

    A void = a zone with no magnets, S/R, or stop clusters nearby.

    Returns: SL price (float) or None if no void found.
    """
    cfg = cfg or {}
    buffer_pct = _cfg(cfg, 'SL_VOID_BUFFER_PCT')
    min_dist_pct = _cfg(cfg, 'SL_VOID_MIN_DIST_PCT')
    max_dist_pct = _cfg(cfg, 'SL_VOID_MAX_DIST_PCT')
    hard_max_pct = _cfg(cfg, 'SL_HARD_MAX_PCT')

    levels = _collect_levels(price, direction, magnets, sr_levels, liq_levels, cfg)

    if not levels:
        return None  # no levels → use ATR fallback

    # Sort levels by distance from price
    levels.sort(key=lambda p: abs(p - price))

    # Find voids between levels
    # A void = midpoint between two adjacent levels, if the gap > 2 * buffer
    voids = []

    # Add boundary voids (beyond the nearest level away from price)
    if direction == 'LONG':
        # SL goes below → look for voids below price
        below_levels = sorted([p for p in levels if p < price], reverse=True)
        if below_levels:
            # Void below the lowest nearby level
            lowest = below_levels[-1]
            void_candidate = lowest - price * buffer_pct
            if abs(price - void_candidate) / price >= min_dist_pct:
                voids.append(void_candidate)
        # Voids between levels
        for i in range(len(below_levels) - 1):
            gap = below_levels[i] - below_levels[i+1]
            if gap > 2 * price * buffer_pct:
                mid = (below_levels[i] + below_levels[i+1]) / 2
                if abs(price - mid) / price >= min_dist_pct:
                    voids.append(mid)
    else:
        # SL goes above → look for voids above price
        above_levels = sorted([p for p in levels if p > price])
        if above_levels:
            highest = above_levels[-1]
            void_candidate = highest + price * buffer_pct
            if abs(void_candidate - price) / price >= min_dist_pct:
                voids.append(void_candidate)
        for i in range(len(above_levels) - 1):
            gap = above_levels[i+1] - above_levels[i]
            if gap > 2 * price * buffer_pct:
                mid = (above_levels[i] + above_levels[i+1]) / 2
                if abs(mid - price) / price >= min_dist_pct:
                    voids.append(mid)

    if not voids:
        return None

    # Pick the closest void that's within max distance
    voids_in_range = [v for v in voids if abs(v - price) / price <= max_dist_pct]
    if not voids_in_range:
        return None

    # Pick closest to price
    best_void = min(voids_in_range, key=lambda v: abs(v - price))

    # Enforce hard max
    if abs(best_void - price) / price > hard_max_pct:
        if direction == 'LONG':
            best_void = price - price * hard_max_pct
        else:
            best_void = price + price * hard_max_pct

    return best_void


# ═══════════════════════════════════════════════════════════════
# UNSWEPT POOL DETECTION (for TP)
# ═══════════════════════════════════════════════════════════════

def find_next_unswept(price, direction, magnets, liq_levels, exclude_below=None, cfg=None):
    """Find the next unswept liquidity pool in trade direction.

    Args:
        price: current price
        direction: 'LONG' or 'SHORT'
        magnets: volume profile magnets [(price, vol, strength), ...]
        liq_levels: dict with 'above'/'below' lists of level dicts
        exclude_below: skip pools closer than this price (for TP2/TP3)
        cfg: config dict

    Returns: target price (float) or None
    """
    cfg = cfg or {}
    min_dist_pct = _cfg(cfg, 'TP1_MAGNET_MIN_DIST_PCT')

    candidates = []

    # From magnets (HVNs — absorption zones)
    if magnets:
        for m in magnets:
            if isinstance(m, (list, tuple, np.ndarray)) and len(m) >= 1:
                p = m[0]
                strength = m[2] if len(m) >= 3 else (m[1] if len(m) >= 2 else 10.0)
            elif isinstance(m, (float, int, np.float64, np.int64)):
                p = float(m)
                strength = 10.0
            else:
                continue
            dist_pct = abs(p - price) / price
            if dist_pct < min_dist_pct:
                continue
            if direction == 'LONG' and p > price:
                if exclude_below and p <= exclude_below:
                    continue
                candidates.append((p, dist_pct, strength))  # (price, dist, strength)
            elif direction == 'SHORT' and p < price:
                if exclude_below and p >= exclude_below:
                    continue
                candidates.append((p, dist_pct, strength))

    # From liquidation levels (unswept stops/liquidations)
    if liq_levels and isinstance(liq_levels, dict):
        side = 'above' if direction == 'LONG' else 'below'
        for lvl in liq_levels.get(side, []):
            p = lvl.get('price', 0)
            swept = lvl.get('swept', False)
            if p <= 0 or swept:
                continue
            dist_pct = abs(p - price) / price
            if dist_pct < min_dist_pct:
                continue
            if exclude_below:
                if direction == 'LONG' and p <= exclude_below:
                    continue
                if direction == 'SHORT' and p >= exclude_below:
                    continue
            strength = lvl.get('strength', 1)
            candidates.append((p, dist_pct, strength))

    if not candidates:
        return None

    # Sort by distance, pick closest
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION: CALCULATE ALL LEVELS
# ═══════════════════════════════════════════════════════════════


def find_strongest_liquidity(entry_price, direction, liq_levels, sr_levels=None,
                              min_dist_pct=0.003, max_targets=3, cfg=None):
    """Find TP targets based on liquidity STRENGTH, not distance.

    Picks the strongest unswept liquidity pools as TP targets.
    Much better than distance-based for capturing real moves.

    Args:
        entry_price: entry price
        direction: 'LONG' or 'SHORT'
        liq_levels: dict with 'above'/'below' lists
        sr_levels: S/R levels for additional targets
        min_dist_pct: minimum distance from entry (0.3% default)
        max_targets: max TP targets to return
        cfg: config dict

    Returns:
        list of dicts: [{'price': float, 'strength': float, 'source': str, 'dist_pct': float}, ...]
        sorted by price (ascending for SHORT, descending for LONG)
    """
    cfg = cfg or {}
    candidates = []

    # From liquidation levels (primary source)
    if liq_levels and isinstance(liq_levels, dict):
        side = 'below' if direction == 'SHORT' else 'above'
        for lvl in liq_levels.get(side, []):
            p = lvl.get('price', 0)
            swept = lvl.get('swept', False)
            if p <= 0 or swept:
                continue
            dist_pct = abs(p - entry_price) / entry_price
            if dist_pct < min_dist_pct:
                continue
            strength = lvl.get('strength', 1)
            cascade = lvl.get('cascade_risk', 'LOW')
            # Boost cascade risk levels
            if cascade == 'HIGH':
                strength *= 1.3
            elif cascade == 'MED':
                strength *= 1.1
            candidates.append({
                'price': p, 'strength': strength,
                'source': lvl.get('type', 'LIQ'),
                'dist_pct': dist_pct, 'cascade': cascade
            })

    # From S/R levels (secondary source — lower weight)
    if sr_levels:
        for sr in sr_levels:
            if len(sr) < 3:
                continue
            p, strength, sr_type = sr[0], sr[1], sr[2]
            if sr_type == 'SUPPORT' and direction == 'SHORT':
                dist_pct = abs(p - entry_price) / entry_price
                if dist_pct >= min_dist_pct:
                    candidates.append({
                        'price': p, 'strength': strength * 0.5,  # S/R weighted lower
                        'source': 'SR_SUPPORT',
                        'dist_pct': dist_pct, 'cascade': 'LOW'
                    })
            elif sr_type == 'RESISTANCE' and direction == 'LONG':
                dist_pct = abs(p - entry_price) / entry_price
                if dist_pct >= min_dist_pct:
                    candidates.append({
                        'price': p, 'strength': strength * 0.5,
                        'source': 'SR_RESISTANCE',
                        'dist_pct': dist_pct, 'cascade': 'LOW'
                    })

    if not candidates:
        return []

    # Sort by strength (strongest first)
    candidates.sort(key=lambda x: x['strength'], reverse=True)

    # Pick top N, then sort by price for proper TP ordering
    selected = candidates[:max_targets * 2]  # take extra, filter later

    # Remove duplicates (within 0.2% of each other)
    filtered = []
    for c in selected:
        too_close = False
        for f in filtered:
            if abs(c['price'] - f['price']) / entry_price < 0.002:
                too_close = True
                break
        if not too_close:
            filtered.append(c)

    # Take top N by strength
    filtered = filtered[:max_targets]

    # Sort by STRENGTH (strongest = TP1, furthest = TP3)
    # TP1 = strongest pool (most likely to attract price)
    # TP3 = weakest pool (furthest target)
    filtered.sort(key=lambda x: x['strength'], reverse=True)

    return filtered

def calc_trade_levels(entry_price, direction, atr_1h, vol_ratio,
                      magnets, sr_levels, liq_levels, cfg=None):
    """Calculate SL/TP using liquidity-aware logic with ATR fallback.

    Args:
        entry_price: entry price
        direction: 'LONG' or 'SHORT'
        atr_1h: 1-hour ATR value
        vol_ratio: volume ratio (for TP multipliers)
        magnets: volume profile magnets [(price, vol, strength), ...]
        sr_levels: S/R levels [(price, strength, touches, bounces, type), ...]
        liq_levels: dict with 'above'/'below' lists (from M15) or None
        cfg: config dict

    Returns:
        dict with sl, tp1, tp2, tp3, sl_source, tp1_source, tp2_source, tp3_source
    """
    cfg = cfg or {}
    atr = float(atr_1h) if not np.isnan(atr_1h) else entry_price * 0.01

    # ── SL: Strongest invalidation level (not nearest) ──
    # Pick the STRONGEST liquidity level in opposite direction as SL.
    # Nearest level is often noise; strongest represents real invalidation.
    sl_candidates = []

    if liq_levels and isinstance(liq_levels, dict):
        side = 'above' if direction == 'SHORT' else 'below'
        for lvl in liq_levels.get(side, []):
            p = lvl.get('price', 0)
            swept = lvl.get('swept', False)
            if p <= 0 or swept:
                continue
            dist_pct = abs(p - entry_price) / entry_price
            # Skip too-close levels (< 0.1%) — those are noise
            if dist_pct < 0.001:
                continue
            strength = lvl.get('strength', 1)
            sl_candidates.append({'price': p, 'strength': strength, 'dist_pct': dist_pct})

    if sl_candidates:
        # Sort by strength to find the strongest cluster
        sl_candidates.sort(key=lambda x: x['strength'], reverse=True)
        strongest = sl_candidates[0]

        # SL = weak level AFTER the strongest (survives the sweep)
        # Price will sweep the strongest cluster (triggering stops),
        # then reverse. SL beyond that cluster avoids the sweep.
        beyond = [c for c in sl_candidates if c['dist_pct'] > strongest['dist_pct']]
        if beyond:
            # Pick the weakest level beyond the strongest cluster
            beyond.sort(key=lambda x: x['strength'])
            sl = beyond[0]['price']
            sl_source = 'LIQUIDITY_BEYOND_SWEEP'
        else:
            # No level beyond strongest — add buffer above strongest
            buffer_pct = 0.003  # 0.3% buffer
            if direction == 'SHORT':
                sl = strongest['price'] * (1 + buffer_pct)
            else:
                sl = strongest['price'] * (1 - buffer_pct)
            sl_source = 'LIQUIDITY_SWEEP_BUFFER'
    else:
        # ATR fallback
        sl_atr_std = _cfg(cfg, 'SL_ATR_STD')
        sl_min_dollar = _cfg(cfg, 'SL_MIN_DOLLAR')
        if sl_min_dollar and sl_atr_std < sl_min_dollar:
            sl_atr_std = sl_min_dollar
        sl_dist = min(sl_atr_std,
                      _cfg(cfg, 'SL_HARD_MAX_PCT') * entry_price)
        if direction == 'LONG':
            sl = entry_price - sl_dist
        else:
            sl = entry_price + sl_dist
        sl_source = 'ATR'

    # ── TP: Liquidity-strength-based targets ──
    # Pick TPs by strongest liquidity pools, not nearest distance
    liq_targets = find_strongest_liquidity(
        entry_price, direction, liq_levels, sr_levels,
        min_dist_pct=_cfg(cfg, 'TP1_MAGNET_MIN_DIST_PCT', 0.003),
        max_targets=3, cfg=cfg)

    if len(liq_targets) >= 1:
        tp1 = liq_targets[0]['price']
        tp1_source = liq_targets[0]['source']
    else:
        # ATR fallback
        tp1_dist = _cfg(cfg, 'TP1_ATR') * atr
        tp1_min_dollar = _cfg(cfg, 'TP1_MIN_DOLLAR')
        if tp1_min_dollar and tp1_dist < tp1_min_dollar:
            tp1_dist = tp1_min_dollar
        if direction == 'LONG':
            tp1 = entry_price + tp1_dist
        else:
            tp1 = entry_price - tp1_dist
        tp1_source = 'ATR'

    if len(liq_targets) >= 2:
        tp2 = liq_targets[1]['price']
        tp2_source = liq_targets[1]['source']
    else:
        tp2_mult = _cfg(cfg, 'TP2_ATR')
        tp2_dist = tp2_mult * atr
        if direction == 'LONG':
            tp2 = entry_price + tp2_dist
        else:
            tp2 = entry_price - tp2_dist
        tp2_source = 'ATR'

    if len(liq_targets) >= 3:
        tp3 = liq_targets[2]['price']
        tp3_source = liq_targets[2]['source']
    else:
        tp3_mult = _cfg(cfg, 'TP3_ATR')
        tp3_dist = tp3_mult * atr
        if direction == 'LONG':
            tp3 = entry_price + tp3_dist
        else:
            tp3 = entry_price - tp3_dist
        tp3_source = 'ATR'

    return {
        'sl': float(sl),
        'tp1': float(tp1),
        'tp2': float(tp2),
        'tp3': float(tp3),
        'sl_source': sl_source,
        'tp1_source': tp1_source,
        'tp2_source': tp2_source,
        'tp3_source': tp3_source,
        'sl_pct': abs(entry_price - sl) / entry_price * 100,
        'tp1_pct': abs(tp1 - entry_price) / entry_price * 100,
    }


def check_sweep_gate(m14_status, m14_score, cfg=None):
    """Check if M14 sweep gate is enabled and passed.

    Returns: (passed, reason)
    """
    cfg = cfg or {}
    if not _cfg(cfg, 'M14_ENTRY_GATE'):
        return True, 'gate_disabled'

    if m14_status == 'PASS':
        return True, 'sweep_confirmed'

    return False, f'M14 gate: {m14_status} (sweep required)'


# ═══════════════════════════════════════════════════════════════
# LIMIT ENTRY — Better entry price via support/resistance
# ═══════════════════════════════════════════════════════════════

_LIMIT_ENTRY_DEFAULTS = {
    'LIMIT_ENTRY_ENABLED': True,
    'LIMIT_ENTRY_MAX_DIST_PCT': 0.02,    # max 2.0% from current price
    'LIMIT_ENTRY_MIN_DIST_PCT': 0.001,   # min 0.1% (avoid too-close levels)
    'LIMIT_ENTRY_ATR_MIN_MULT': 0.2,     # min distance as ATR multiple
    'LIMIT_ENTRY_ATR_MAX_MULT': 2.0,     # max distance as ATR multiple
    'LIMIT_ENTRY_PREFER_SR': True,       # prefer S/R over HVN
    'LIMIT_ENTRY_SR_MIN_STRENGTH': 15,   # min S/R strength to consider
}


def calc_limit_entry(current_price, direction, magnets, sr_levels,
                     atr_1h=None, cfg=None):
    """Compute a limit entry price using support/resistance and volume profile.

    For LONG: finds the nearest strong support or HVN below current price.
    For SHORT: finds the nearest strong resistance or HVN above current price.

    Returns:
        dict with:
            entry_price: recommended limit entry price
            entry_source: 'SUPPORT' | 'RESISTANCE' | 'HVN' | 'MARKET'
            entry_level: the S/R or magnet price used
            entry_dist_pct: distance from current price (%)
            reason: human-readable explanation
    """
    cfg = cfg or {}
    if not cfg.get('LIMIT_ENTRY_ENABLED', _LIMIT_ENTRY_DEFAULTS['LIMIT_ENTRY_ENABLED']):
        return {
            'entry_price': current_price, 'entry_source': 'MARKET',
            'entry_level': current_price, 'entry_dist_pct': 0.0,
            'reason': 'limit entry disabled',
        }

    max_dist = cfg.get('LIMIT_ENTRY_MAX_DIST_PCT', _LIMIT_ENTRY_DEFAULTS['LIMIT_ENTRY_MAX_DIST_PCT'])
    min_dist = cfg.get('LIMIT_ENTRY_MIN_DIST_PCT', _LIMIT_ENTRY_DEFAULTS['LIMIT_ENTRY_MIN_DIST_PCT'])
    atr_min = cfg.get('LIMIT_ENTRY_ATR_MIN_MULT', _LIMIT_ENTRY_DEFAULTS['LIMIT_ENTRY_ATR_MIN_MULT'])
    atr_max = cfg.get('LIMIT_ENTRY_ATR_MAX_MULT', _LIMIT_ENTRY_DEFAULTS['LIMIT_ENTRY_ATR_MAX_MULT'])
    prefer_sr = cfg.get('LIMIT_ENTRY_PREFER_SR', _LIMIT_ENTRY_DEFAULTS['LIMIT_ENTRY_PREFER_SR'])
    sr_min_str = cfg.get('LIMIT_ENTRY_SR_MIN_STRENGTH', _LIMIT_ENTRY_DEFAULTS['LIMIT_ENTRY_SR_MIN_STRENGTH'])

    atr = float(atr_1h) if atr_1h is not None and not np.isnan(atr_1h) else current_price * 0.005

    candidates = []

    # ── Collect S/R candidates ──
    if sr_levels:
        for level in sr_levels:
            price, strength, touches, bounces, ltype = level
            dist_pct = abs(price - current_price) / current_price
            dist_atr = abs(price - current_price) / atr if atr > 0 else 0

            # Filter by direction
            if direction == 'LONG' and ltype != 'SUPPORT':
                continue
            if direction == 'SHORT' and ltype != 'RESISTANCE':
                continue

            # Filter by distance
            if dist_pct < min_dist or dist_pct > max_dist:
                continue
            if dist_atr < atr_min or dist_atr > atr_max:
                continue

            # Filter by strength
            if strength < sr_min_str:
                continue

            candidates.append({
                'price': float(price),
                'source': ltype,
                'strength': float(strength),
                'touches': touches,
                'bounces': bounces,
                'dist_pct': dist_pct,
                'dist_atr': dist_atr,
                'priority': 0 if prefer_sr else 1,  # S/R gets priority
            })

    # ── Collect HVN (magnet) candidates ──
    if magnets:
        for mag in magnets:
            # Handle both tuple/list (price, vol, strength) and simple float (price)
            if isinstance(mag, (list, tuple, np.ndarray)) and len(mag) >= 1:
                price = mag[0]
                strength = mag[2] if len(mag) >= 3 else (mag[1] if len(mag) >= 2 else 10.0)
            elif isinstance(mag, (float, int, np.float64, np.int64)):
                price = float(mag)
                strength = 10.0  # Default strength for simple price magnets
            else:
                continue

            dist_pct = abs(price - current_price) / current_price
            dist_atr = abs(price - current_price) / atr if atr > 0 else 0

            # Filter by direction
            if direction == 'LONG' and price >= current_price:
                continue
            if direction == 'SHORT' and price <= current_price:
                continue

            # Filter by distance
            if dist_pct < min_dist or dist_pct > max_dist:
                continue
            if dist_atr < atr_min or dist_atr > atr_max:
                continue

            candidates.append({
                'price': float(price),
                'source': 'HVN',
                'strength': float(strength),
                'touches': 0,
                'bounces': 0,
                'dist_pct': dist_pct,
                'dist_atr': dist_atr,
                'priority': 1,
            })

    if not candidates:
        return {
            'entry_price': current_price, 'entry_source': 'MARKET',
            'entry_level': current_price, 'entry_dist_pct': 0.0,
            'reason': 'no qualifying S/R or HVN within range',
        }

    # ── Score candidates: prefer closer, stronger levels ──
    for c in candidates:
        # Score = strength * proximity bonus - distance penalty
        proximity_bonus = 1.0 / (c['dist_pct'] * 100 + 0.1)  # closer = higher
        strength_score = c['strength'] / 100.0  # normalize
        c['score'] = (strength_score * 0.6 + proximity_bonus * 0.4) - c['priority'] * 0.1

    # Sort by score descending
    candidates.sort(key=lambda x: -x['score'])
    best = candidates[0]

    entry_price = best['price']
    entry_dist_pct = (current_price - entry_price) / current_price * 100

    return {
        'entry_price': round(entry_price, 2),
        'entry_source': best['source'],
        'entry_level': round(best['price'], 2),
        'entry_dist_pct': round(entry_dist_pct, 4),
        'reason': (f"{best['source']} @ ${best['price']:.2f} "
                   f"(str={best['strength']:.0f}, dist={entry_dist_pct:+.2f}%)"),
    }
