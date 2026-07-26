"""S01: Failed Breakout v3 — Liquidity Trap Detection.

Uses existing scanner modules to detect REAL false breakouts (liquidity traps):
- M14: Liquidity sweep detection (stop-hunt identification)
- M21: Wyckoff phase + zone (accumulation/distribution context)
- M5: Structural levels (HVN, FVG, order blocks)
- Liquidity levels: Where stops are clustered
- Derivatives: Positioning extremes (crowded side)
- Taker flow: Who is pushing (informed vs retail)

v2 → v3 CHANGES:
1. Uses M14 sweep detection instead of raw price-action breakout
2. Requires Wyckoff context (spring/upthrust in accumulation/distribution)
3. Uses liquidity_levels for significant S/R (not every minor swing)
4. Checks derivatives for positioning extremes (crowded stops)
5. Requires taker flow divergence (informed traders on reversal side)
6. Structural SL at sweep extreme, TP at next M5 level

Research basis:
- Brunnermeier & Pedersen (2009): Stop-loss cascades at liquidity clusters
- Wyckoff: Springs (false breaks below accumulation) = high-conviction LONG
- SSRN 6579278: Cascade = liquidity event, not price event
"""
from .base import BaseStrategy, SignalResult
import numpy as np

GOOD_HOURS = {9, 10, 11, 12, 14, 15, 16, 18}


def _check_sweep(data, df_15m, idx):
    """Check M14 for liquidity sweep (stop-hunt).
    
    M14 detects: stop-hunt identification, sweep velocity.
    A sweep = price moves beyond a level to trigger stops, then reverses.
    """
    m14 = data.get('m14', {})
    if not m14:
        return None
    
    sweep_detected = m14.get('sweep_detected', False)
    sweep_direction = m14.get('sweep_direction', '')
    sweep_velocity = m14.get('sweep_velocity', 0)
    sweep_level = m14.get('sweep_level', 0)
    
    if not sweep_detected or not sweep_direction:
        return None
    
    return {
        'direction': sweep_direction,  # 'LONG' or 'SHORT' (direction of the sweep)
        'velocity': sweep_velocity,
        'level': sweep_level,
        'm14_score': m14.get('score', 0),
    }


def _check_wyckoff(data):
    """Check M21 for Wyckoff phase context.
    
    Highest-conviction false breakouts:
    - Spring (false break below accumulation) → LONG
    - Upthrust (false break above distribution) → SHORT
    
    Also check zone: Premium/Discount/Equilibrium
    - Discount + spring = highest conviction LONG
    - Premium + upthrust = highest conviction SHORT
    """
    m21 = data.get('m21', {})
    if not m21:
        return None
    
    phase = m21.get('phase', '')  # Accumulation, Markup, Distribution, Markdown
    zone = m21.get('zone', '')    # Premium, Discount, Equilibrium
    spring = m21.get('spring', False)
    upthrust = m21.get('upthrust', False)
    score = m21.get('score', 0)
    
    # Map phase to regime
    phase_map = {
        'Accumulation': 'ACCUMULATION',
        'Markup': 'MARKUP',
        'Distribution': 'DISTRIBUTION',
        'Markdown': 'MARKDOWN',
    }
    wyckoff_phase = phase_map.get(phase, 'UNKNOWN')
    
    return {
        'phase': wyckoff_phase,
        'zone': zone,
        'spring': spring,
        'upthrust': upthrust,
        'score': score,
    }


def _check_structural_levels(data):
    """Get structural levels from M5 + liquidity_levels.
    
    These are the SIGNIFICANT levels where stops cluster:
    - HVN (High Volume Node) = absorption zone
    - FVG (Fair Value Gap) = reversion target
    - Order Block = institutional rejection level
    - Liquidity levels = where stops are resting
    """
    m5 = data.get('m5', {})
    liq = data.get('liquidity_levels', {})
    
    levels = {
        'above': [],  # levels above price (resistance / TP for SHORT)
        'below': [],  # levels below price (support / TP for LONG)
    }
    
    # From M5
    if m5:
        magnets = m5.get('magnets', [])
        for m in magnets:
            if isinstance(m, dict):
                lvl = m.get('level', 0)
                strength = m.get('strength', 0)
                ltype = m.get('type', 'unknown')
                if lvl > 0:
                    levels['above' if lvl > data.get('price', 0) else 'below'].append({
                        'level': lvl, 'strength': strength, 'type': f'm5_{ltype}',
                    })
    
    # From liquidity_levels (these are where stops are clustered)
    if liq:
        for direction in ['above', 'below']:
            for lvl_data in liq.get(direction, []):
                if isinstance(lvl_data, dict):
                    lvl = lvl_data.get('level', 0)
                    strength = lvl_data.get('strength', 0)
                    if lvl > 0:
                        levels[direction].append({
                            'level': lvl, 'strength': strength, 'type': 'liquidity',
                        })
    
    # Sort by distance from price
    price = data.get('price', 0)
    for direction in ['above', 'below']:
        levels[direction].sort(key=lambda x: abs(x['level'] - price))
    
    return levels


def _check_positioning(data):
    """Check derivatives for positioning extremes.
    
    Crowded positioning = more stops to trigger = bigger cascade.
    - LS ratio > 2.0 = long-crowded → SHORT sweep likely
    - LS ratio < 0.5 = short-crowded → LONG sweep likely
    - |FR| > 0.05% = overleveraged → bigger cascade when stops hit
    """
    deriv = data.get('derivatives', {})
    if not deriv:
        return None
    
    ls = deriv.get('ls_ratio', 1.0) or 1.0
    fr = deriv.get('funding_rate', 0) or 0
    oi_roc = deriv.get('oi_roc_1h', 0) or 0
    
    crowded_side = 'NEUTRAL'
    if ls > 2.0:
        crowded_side = 'LONG_CROWDED'
    elif ls < 0.5:
        crowded_side = 'SHORT_CROWDED'
    
    overleveraged = abs(fr) > 0.0005
    
    return {
        'ls_ratio': ls,
        'funding_rate': fr,
        'oi_roc': oi_roc,
        'crowded_side': crowded_side,
        'overleveraged': overleveraged,
    }


def _check_taker_divergence(data):
    """Check taker flow for informed trader activity.
    
    Divergence = price goes one way, takers go the other.
    - Price down + taker buying = informed accumulating on dip (LONG signal)
    - Price up + taker selling = informed distributing on rally (SHORT signal)
    """
    taker = data.get('taker_summary', {})
    if not taker:
        return None
    
    regime = taker.get('regime', '')
    momentum = taker.get('momentum', 0)
    ratio = taker.get('ratio', 1.0) or 1.0
    
    # Price direction (recent)
    price = data.get('price', 0)
    ema200 = data.get('ema_200', 0)
    
    taker_direction = 'NEUTRAL'
    if 'BUYING' in regime.upper() or 'SURGE' in regime.upper():
        taker_direction = 'BUYING'
    elif 'SELLING' in regime.upper():
        taker_direction = 'SELLING'
    
    return {
        'direction': taker_direction,
        'momentum': momentum,
        'ratio': ratio,
        'regime': regime,
    }


class FailedBreakoutStrategy(BaseStrategy):
    name = 'failed_breakout'
    strategy_type = 'event'
    description = 'v3: Liquidity trap detection via M14+M21+M5+derivatives+taker'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr or df_15m is None or idx is None:
            return None

        # ── SESSION FILTER ──
        ts = data.get('timestamp', '')
        if ts:
            try:
                hour = int(ts[11:13])
                if hour not in GOOD_HOURS:
                    return None
            except (ValueError, IndexError):
                pass

        # ── CHECK 1: LIQUIDITY SWEEP (M14) ──
        sweep = _check_sweep(data, df_15m, idx)
        if not sweep:
            return None

        # ── CHECK 2: WYCKOFF CONTEXT (M21) ──
        wyckoff = _check_wyckoff(data)
        if not wyckoff:
            return None

        # ── CHECK 3: POSITIONING (derivatives) ──
        positioning = _check_positioning(data)
        
        # ── CHECK 4: TAKER DIVERGENCE ──
        taker = _check_taker_divergence(data)
        
        # ── CHECK 5: STRUCTURAL LEVELS (M5 + liquidity) ──
        levels = _check_structural_levels(data)

        # ── DIRECTION DETERMINATION ──
        # Sweep direction tells us WHERE price went to trigger stops
        # We trade AGAINST the sweep (the reversal)
        
        sweep_dir = sweep['direction']
        
        # Match sweep to Wyckoff context
        if sweep_dir == 'SHORT' and wyckoff.get('spring'):
            # Spring (false break below accumulation) → LONG
            direction = 'LONG'
            wyckoff_bonus = 0.20
        elif sweep_dir == 'LONG' and wyckoff.get('upthrust'):
            # Upthrust (false break above distribution) → SHORT
            direction = 'SHORT'
            wyckoff_bonus = 0.20
        elif sweep_dir == 'SHORT' and wyckoff.get('phase') == 'ACCUMULATION':
            # Sweep below + accumulation context → LONG
            direction = 'LONG'
            wyckoff_bonus = 0.15
        elif sweep_dir == 'LONG' and wyckoff.get('phase') == 'DISTRIBUTION':
            # Sweep above + distribution context → SHORT
            direction = 'SHORT'
            wyckoff_bonus = 0.15
        elif sweep_dir == 'SHORT':
            # Sweep below without Wyckoff context → LONG but lower conviction
            direction = 'LONG'
            wyckoff_bonus = 0.0
        elif sweep_dir == 'LONG':
            # Sweep above without Wyckoff context → SHORT but lower conviction
            direction = 'SHORT'
            wyckoff_bonus = 0.0
        else:
            return None

        # ── POSITIONING ALIGNMENT ──
        positioning_bonus = 0
        if positioning:
            if direction == 'LONG' and positioning['crowded_side'] == 'SHORT_CROWDED':
                positioning_bonus = 0.10  # shorts are crowded = squeeze potential
            elif direction == 'SHORT' and positioning['crowded_side'] == 'LONG_CROWDED':
                positioning_bonus = 0.10  # longs are crowded = squeeze potential
            if positioning['overleveraged']:
                positioning_bonus += 0.05  # bigger cascade when stops hit

        # ── TAKER DIVERGENCE ──
        taker_bonus = 0
        if taker:
            if direction == 'LONG' and taker['direction'] == 'BUYING':
                taker_bonus = 0.10  # informed buyers on dip
            elif direction == 'SHORT' and taker['direction'] == 'SELLING':
                taker_bonus = 0.10  # informed sellers on rally
            # Divergence: price going against taker direction
            if direction == 'LONG' and taker['direction'] == 'BUYING':
                taker_bonus += 0.05  # price down + taker buying = strong divergence
            elif direction == 'SHORT' and taker['direction'] == 'SELLING':
                taker_bonus += 0.05  # price up + taker selling = strong divergence

        # ── SWEEP QUALITY ──
        sweep_bonus = min(sweep.get('velocity', 0) / 5.0, 0.10)  # faster sweep = more stops triggered

        # ── CONVICTION ──
        base = 0.35
        conviction = base + wyckoff_bonus + positioning_bonus + taker_bonus + sweep_bonus
        conviction = min(conviction, 0.90)

        if conviction < 0.50:
            return None

        # ── TP/SL ──
        # SL: at the sweep extreme (if price goes back to sweep level, trap failed)
        sweep_level = sweep.get('level', 0)
        if sweep_level and sweep_level > 0:
            if direction == 'LONG':
                sl_price = sweep_level * 0.998  # just below sweep low
                sl_pct = abs(price - sl_price) / price * 100
            else:
                sl_price = sweep_level * 1.002  # just above sweep high
                sl_pct = abs(sl_price - price) / price * 100
            
            # Cap SL at 2x ATR
            max_sl_pct = (atr / price * 100) * 2.0
            if sl_pct > max_sl_pct:
                sl_pct = max_sl_pct
                sl_price = price * (1 - sl_pct/100) if direction == 'LONG' else price * (1 + sl_pct/100)
        else:
            # Fallback: ATR-based
            sl_pct = (atr / price * 100) * 1.2
            sl_price = price * (1 - sl_pct/100) if direction == 'LONG' else price * (1 + sl_pct/100)

        # TP: nearest structural level in trade direction
        tp1_price = None
        if levels:
            tp_levels = levels['below'] if direction == 'LONG' else levels['above']
            for lvl in tp_levels:
                if direction == 'LONG' and lvl['level'] > price:
                    tp1_price = lvl['level']
                    break
                elif direction == 'SHORT' and lvl['level'] < price:
                    tp1_price = lvl['level']
                    break
        
        if tp1_price and tp1_price > 0:
            tp1_pct = abs(tp1_price - price) / price * 100
        else:
            # Fallback: 2.5x ATR
            tp1_pct = sl_pct * 2.5
            tp1_price = price * (1 + tp1_pct/100) if direction == 'LONG' else price * (1 - tp1_pct/100)

        tp2_pct = tp1_pct * 1.5
        tp3_pct = tp1_pct * 2.5
        tp2_price = price * (1 + tp2_pct/100) if direction == 'LONG' else price * (1 - tp2_pct/100)
        tp3_price = price * (1 + tp3_pct/100) if direction == 'LONG' else price * (1 - tp3_pct/100)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl_price, tp1=tp1_price, tp2=tp2_price, tp3=tp3_price,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.8,
            reason=f"Failed BO v3 {direction}: sweep={sweep_dir} wyckoff={wyckoff.get('phase','?')} "
                   f"zone={wyckoff.get('zone','?')} crowded={positioning.get('crowded_side','?') if positioning else '?'} "
                   f"taker={taker.get('direction','?') if taker else '?'}",
            bypass_gates=True,
            details={
                'version': 'v3',
                'sweep': sweep,
                'wyckoff': wyckoff,
                'positioning': positioning,
                'taker': taker,
                'structural_levels': {
                    'above_count': len(levels.get('above', [])),
                    'below_count': len(levels.get('below', [])),
                    'tp_source': 'structural' if tp1_price else 'atr_fallback',
                },
                'conviction_breakdown': {
                    'base': base,
                    'wyckoff_bonus': wyckoff_bonus,
                    'positioning_bonus': positioning_bonus,
                    'taker_bonus': taker_bonus,
                    'sweep_bonus': sweep_bonus,
                },
                'sl_source': 'sweep_level' if sweep_level else 'atr_fallback',
            },
        )
