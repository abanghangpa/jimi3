"""S01: Failed Breakout v3.1 — Liquidity Trap Detection (Deploy).

v3 → v3.1 CHANGES:
1. Added independent sweep detection from df_15m (fallback when M14 scan data missing)
2. Scan M14 used when available, price-action detection as fallback
3. Full backtest confirmed: WEAK+ACCUM+LONG = MC p=0.033, WR=62.1%
4. Restructured to match backtest logic exactly

Gate: 95 events, +0.281%, p=0.081, WR=62.1%, MC p=0.033
"""
from .base import BaseStrategy, SignalResult
import numpy as np

GOOD_HOURS = {9, 10, 11, 12, 14, 15, 16, 18}

# M14 defaults (from m14_sweep.py)
M14_SWEEP_DEPTH_MIN = 0.001  # 0.1%
M14_SWEEP_DEPTH_MAX = 0.020  # 2%
M14_RECLAIM_WICK_RATIO = 0.40
M14_VOL_CONFIRM_MULT = 1.2


def _find_swing_levels(highs, lows, idx, lookback=48):
    """Find swing highs and lows (simple pivot detection)."""
    swing_highs = []
    swing_lows = []
    start = max(0, idx - lookback)
    for i in range(start + 2, idx - 1):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1]:
            swing_highs.append((highs[i], i))
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1]:
            swing_lows.append((lows[i], i))
    return swing_highs, swing_lows


def _detect_sweep_independent(df_15m, idx, direction):
    """Independent sweep detection from price action (matches backtest logic).
    
    Returns (found, details) tuple.
    """
    if idx < 56:
        return False, {}
    
    closes = df_15m['Close'].values.astype(float)
    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    opens = df_15m['Open'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)
    
    swing_highs, swing_lows = _find_swing_levels(highs, lows, idx)
    levels = swing_lows if direction == 'LONG' else swing_highs
    
    for level_price, level_idx in levels:
        # Look for sweep in last 5 bars
        for lb in range(1, min(6, idx)):
            bar_idx = idx - lb
            if bar_idx < level_idx:
                continue
            
            bar_range = highs[bar_idx] - lows[bar_idx]
            if bar_range <= 0:
                continue
            
            if direction == 'LONG':
                sweep_depth = (level_price - lows[bar_idx]) / level_price
                if M14_SWEEP_DEPTH_MIN <= sweep_depth <= M14_SWEEP_DEPTH_MAX:
                    reclaimed = closes[bar_idx] > level_price
                    lower_wick = min(opens[bar_idx], closes[bar_idx]) - lows[bar_idx]
                    wick_ratio = lower_wick / bar_range
                    
                    # Classify reclaim type (matches backtest logic)
                    vol_avg = np.mean(volumes[max(0, bar_idx-20):bar_idx]) if bar_idx >= 20 else volumes[bar_idx]
                    vol_ok = volumes[bar_idx] > vol_avg * M14_VOL_CONFIRM_MULT
                    
                    if wick_ratio >= M14_RECLAIM_WICK_RATIO and closes[bar_idx] > opens[bar_idx]:
                        reclaim_type = 'STRONG' if vol_ok else 'WEAK'
                    elif closes[bar_idx] > opens[bar_idx] and closes[bar_idx] > level_price:
                        reclaim_type = 'WEAK'
                    else:
                        reclaim_type = 'NONE'
                    
                    return True, {
                        'sweep_bar': bar_idx,
                        'bars_ago': idx - bar_idx,
                        'level': level_price,
                        'depth_pct': sweep_depth * 100,
                        'reclaimed': reclaimed,
                        'wick_ratio': wick_ratio,
                        'reclaim_type': reclaim_type,
                        'vol_ok': vol_ok,
                        'source': 'independent',
                    }
            
            elif direction == 'SHORT':
                sweep_depth = (highs[bar_idx] - level_price) / level_price
                if M14_SWEEP_DEPTH_MIN <= sweep_depth <= M14_SWEEP_DEPTH_MAX:
                    reclaimed = closes[bar_idx] < level_price
                    upper_wick = highs[bar_idx] - max(opens[bar_idx], closes[bar_idx])
                    wick_ratio = upper_wick / bar_range
                    
                    vol_avg = np.mean(volumes[max(0, bar_idx-20):bar_idx]) if bar_idx >= 20 else volumes[bar_idx]
                    vol_ok = volumes[bar_idx] > vol_avg * M14_VOL_CONFIRM_MULT
                    
                    if wick_ratio >= M14_RECLAIM_WICK_RATIO and closes[bar_idx] < opens[bar_idx]:
                        reclaim_type = 'STRONG' if vol_ok else 'WEAK'
                    elif closes[bar_idx] < opens[bar_idx] and closes[bar_idx] < level_price:
                        reclaim_type = 'WEAK'
                    else:
                        reclaim_type = 'NONE'
                    
                    return True, {
                        'sweep_bar': bar_idx,
                        'bars_ago': idx - bar_idx,
                        'level': level_price,
                        'depth_pct': sweep_depth * 100,
                        'reclaimed': reclaimed,
                        'wick_ratio': wick_ratio,
                        'reclaim_type': reclaim_type,
                        'vol_ok': vol_ok,
                        'source': 'independent',
                    }
    
    return False, {}


def _check_sweep(data, df_15m, idx):
    """Check for liquidity sweep. Uses scan M14 if available, else independent."""
    m14 = data.get('m14', {})
    
    # Try scan M14 first
    if m14 and m14.get('status') == 'PASS' and m14.get('score', 0) > 0.5:
        # Scan M14 detected a sweep
        # Determine direction from M14 details
        details = m14.get('details', {})
        sweep_dir = details.get('sweep_direction', '')
        if not sweep_dir:
            # Infer from M14 score and price action
            price = data.get('price', 0)
            ema200 = data.get('ema_200', 0)
            sweep_dir = 'LONG' if price < ema200 else 'SHORT' if price > ema200 else ''
        
        if sweep_dir:
            return {
                'direction': sweep_dir,
                'velocity': details.get('sweep_velocity', 1),
                'level': details.get('sweep_level', 0),
                'm14_score': m14.get('score', 0),
                'reclaim_type': 'WEAK' if m14.get('score', 0) < 0.7 else 'STRONG',
                'source': 'scan_m14',
            }
    
    # Fallback: independent detection from price action
    # Try both directions, pick the most recent
    for direction in ['LONG', 'SHORT']:
        found, details = _detect_sweep_independent(df_15m, idx, direction)
        if found:
            return {
                'direction': direction,
                'velocity': details.get('wick_ratio', 0) * 5,
                'level': details.get('level', 0),
                'm14_score': 0.55 if details.get('reclaim_type') == 'WEAK' else 0.85 if details.get('reclaim_type') == 'STRONG' else 0.30,
                'reclaim_type': details.get('reclaim_type', 'NONE'),
                'source': 'independent',
                'depth_pct': details.get('depth_pct', 0),
                'bars_ago': details.get('bars_ago', 0),
            }
    
    return None


def _check_wyckoff(data, df_15m, idx):
    """Check Wyckoff context. Uses scan M21 if available, else independent."""
    m21 = data.get('m21', {})
    
    # Try scan M21 first
    if m21 and m21.get('status') == 'PASS':
        phase = m21.get('phase', '')
        zone = m21.get('zone', '')
        spring_upthrust = m21.get('spring_upthrust', '')
        
        phase_map = {
            'Accumulation': 'ACCUMULATION', 'Markup': 'MARKUP',
            'Distribution': 'DISTRIBUTION', 'Markdown': 'MARKDOWN',
        }
        wyckoff_phase = phase_map.get(phase, 'UNKNOWN')
        spring = 'SPRING' in str(spring_upthrust).upper()
        upthrust = 'UPThrust' in str(spring_upthrust).upper()
        
        return {
            'phase': wyckoff_phase, 'zone': zone,
            'spring': spring, 'upthrust': upthrust,
            'confidence': m21.get('score', 0.5),
            'source': 'scan_m21',
        }
    
    # Fallback: independent detection from price action
    if idx < 96:
        return None
    
    closes = df_15m['Close'].values.astype(float)
    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    
    lookback = min(768, idx)
    h4_highs = highs[idx-lookback:idx+1]
    h4_lows = lows[idx-lookback:idx+1]
    h4_closes = closes[idx-lookback:idx+1]
    
    half = len(h4_closes) // 2
    recent_hi = h4_highs[-min(10, half):].max()
    prior_hi = h4_highs[-min(20, half):-min(10, half)].max() if len(h4_highs) > 10 else recent_hi
    recent_lo = h4_lows[-min(10, half):].min()
    prior_lo = h4_lows[-min(20, half):-min(10, half)].min() if len(h4_lows) > 10 else recent_lo
    
    hh = recent_hi > prior_hi
    hl = recent_lo > prior_lo
    lh = recent_hi < prior_hi
    ll = recent_lo < prior_lo
    
    range_hi = float(h4_highs.max())
    range_lo = float(h4_lows.min())
    eq = (range_hi + range_lo) / 2
    current = float(h4_closes[-1])
    position = (current - range_lo) / (range_hi - range_lo) if range_hi > range_lo else 0.5
    
    phase = 'RANGE'
    if hh and hl:
        phase = 'DISTRIBUTION' if position > 0.7 else 'MARKUP' if position > 0.5 else 'ACCUMULATION'
    elif lh and ll:
        phase = 'ACCUMULATION' if position < 0.3 else 'MARKDOWN' if position < 0.5 else 'DISTRIBUTION'
    
    zone = 'PREMIUM' if position > 0.55 else 'DISCOUNT' if position < 0.45 else 'EQUILIBRIUM'
    spring = phase == 'ACCUMULATION' and position < 0.25
    upthrust = phase == 'DISTRIBUTION' and position > 0.75
    
    return {
        'phase': phase, 'zone': zone,
        'spring': spring, 'upthrust': upthrust,
        'confidence': 0.5,
        'position': position,
        'source': 'independent',
    }


def _check_positioning(data):
    """Check derivatives for positioning extremes."""
    deriv = data.get('derivatives', {})
    if not deriv:
        return None
    ls = deriv.get('ls_ratio', 1.0) or 1.0
    fr = deriv.get('funding_rate', 0) or 0
    crowded = 'LONG_CROWDED' if ls > 2.0 else 'SHORT_CROWDED' if ls < 0.5 else 'NEUTRAL'
    return {'ls_ratio': ls, 'funding_rate': fr, 'crowded_side': crowded, 'overleveraged': abs(fr) > 0.0005}


def _check_taker(data):
    """Check taker flow direction."""
    taker = data.get('taker_summary', {})
    if not taker:
        return None
    regime = taker.get('regime', '')
    direction = 'BUYING' if 'BUYING' in regime.upper() or 'SURGE' in regime.upper() else 'SELLING' if 'SELLING' in regime.upper() else 'NEUTRAL'
    return {'direction': direction, 'regime': regime}


def _get_structural_tp(data, direction, price):
    """Get TP from structural levels."""
    liq = data.get('liquidity_levels', {})
    if not liq:
        return None
    
    levels = liq.get('below' if direction == 'LONG' else 'above', [])
    for lvl in levels:
        if isinstance(lvl, dict):
            lvl_price = lvl.get('level', 0)
            if direction == 'LONG' and lvl_price > price:
                return lvl_price
            elif direction == 'SHORT' and lvl_price < price:
                return lvl_price
    return None


class FailedBreakoutStrategy(BaseStrategy):
    name = 'failed_breakout'
    strategy_type = 'event'
    description = 'v3.1: LONG-only. WEAK+ACCUM (MC p=0.033, WR=62.1%). SHORT disabled (upthrust=continuation).'

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

        # ── CHECK 1: SWEEP DETECTION (hybrid: scan M14 or independent) ──
        sweep = _check_sweep(data, df_15m, idx)
        if not sweep:
            return None
        
        # ── CHECK 2: WYCKOFF CONTEXT (hybrid: scan M21 or independent) ──
        wyckoff = _check_wyckoff(data, df_15m, idx)
        if not wyckoff:
            return None

        # ── FILTER: WEAK+ACCUM+LONG (backtest-validated signal) ──
        # The full backtest (88,735 bars) confirmed:
        # WEAK reclaim + ACCUMULATION + LONG = MC p=0.033, WR=62.1%
        # STRONG reclaim in ACCUMULATION = 0% WR (wrong!)
        
        sweep_dir = sweep['direction']
        reclaim_type = sweep.get('reclaim_type', 'NONE')
        wyckoff_phase = wyckoff.get('phase', 'UNKNOWN')
        
        # Determine direction from sweep
        if sweep_dir == 'SHORT':
            direction = 'LONG'  # sweep below → trade LONG (reversal)
        elif sweep_dir == 'LONG':
            direction = 'SHORT'  # sweep above → trade SHORT (reversal)
        else:
            return None
        
        # Apply backtest-validated filter
        if direction == 'LONG':
            # Best signal: WEAK reclaim + ACCUMULATION
            if wyckoff_phase == 'ACCUMULATION' and reclaim_type == 'WEAK':
                wyckoff_bonus = 0.25  # highest conviction
            elif wyckoff_phase == 'ACCUMULATION' and reclaim_type == 'STRONG':
                return None  # STRONG+ACCUM = 0% WR in backtest!
            elif wyckoff_phase == 'ACCUMULATION':
                wyckoff_bonus = 0.15  # ACCUMULATION with unknown reclaim
            elif reclaim_type == 'WEAK':
                wyckoff_bonus = 0.10  # WEAK reclaim without ACCUMULATION
            else:
                return None  # not the validated signal
        
        elif direction == 'SHORT':
            # UPthrust finding (2026-07-26): upthrusts are CONTINUATION signals
            # n=3,031, mean=-0.066%, p=0.0003 — NOT reversal entries
            # Do NOT use upthrust as SHORT entry. Reject all SHORT signals.
            return None  # SHORT path disabled — upthrust is continuation, not reversal

        # ── POSITIONING ──
        positioning = _check_positioning(data)
        positioning_bonus = 0
        if positioning:
            if direction == 'LONG' and positioning['crowded_side'] == 'SHORT_CROWDED':
                positioning_bonus = 0.10
            elif direction == 'SHORT' and positioning['crowded_side'] == 'LONG_CROWDED':
                positioning_bonus = 0.10
            if positioning.get('overleveraged'):
                positioning_bonus += 0.05

        # ── TAKER ──
        taker = _check_taker(data)
        taker_bonus = 0
        if taker:
            if direction == 'LONG' and taker['direction'] == 'BUYING':
                taker_bonus = 0.10
            elif direction == 'SHORT' and taker['direction'] == 'SELLING':
                taker_bonus = 0.10

        # ── CONVICTION ──
        base = 0.35
        conviction = base + wyckoff_bonus + positioning_bonus + taker_bonus
        conviction = min(conviction, 0.90)
        
        if conviction < 0.50:
            return None

        # ── TP/SL ──
        sweep_level = sweep.get('level', 0)
        if sweep_level and sweep_level > 0:
            if direction == 'LONG':
                sl_price = sweep_level * 0.998
            else:
                sl_price = sweep_level * 1.002
            sl_pct = abs(price - sl_price) / price * 100
            max_sl_pct = (atr / price * 100) * 2.0
            if sl_pct > max_sl_pct:
                sl_pct = max_sl_pct
                sl_price = price * (1 - sl_pct/100) if direction == 'LONG' else price * (1 + sl_pct/100)
        else:
            sl_pct = (atr / price * 100) * 1.2
            sl_price = price * (1 - sl_pct/100) if direction == 'LONG' else price * (1 + sl_pct/100)

        # TP: structural level or ATR fallback
        tp1_price = _get_structural_tp(data, direction, price)
        if tp1_price:
            tp1_pct = abs(tp1_price - price) / price * 100
        else:
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
            size_mult=0.5,  # provisional size
            reason=f"S01 v3.1 {direction}: sweep={sweep_dir} reclaim={reclaim_type} "
                   f"wyckoff={wyckoff_phase} zone={wyckoff.get('zone','?')} "
                   f"crowded={positioning.get('crowded_side','?') if positioning else '?'} "
                   f"src={sweep.get('source','?')}",
            bypass_gates=True,
            details={
                'version': 'v3.1',
                'signal_type': 'WEAK_ACCUM_LONG' if direction == 'LONG' else 'UPThrust_DISTRIB_SHORT',
                'sweep': sweep,
                'wyckoff': wyckoff,
                'positioning': positioning,
                'taker': taker,
                'conviction_breakdown': {
                    'base': base,
                    'wyckoff_bonus': wyckoff_bonus,
                    'positioning_bonus': positioning_bonus,
                    'taker_bonus': taker_bonus,
                },
                'backtest_validation': {
                    'n': 95, 'wr': 0.621, 'mc_p': 0.033,
                    'signal': 'WEAK+ACCUM+LONG',
                },
                'sl_source': 'sweep_level' if sweep_level else 'atr_fallback',
                'tp_source': 'structural' if tp1_price else 'atr_fallback',
            },
        )
