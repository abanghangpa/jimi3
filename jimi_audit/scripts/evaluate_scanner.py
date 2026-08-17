#!/usr/bin/env python3
"""
JIMI Scanner Performance Evaluator

Replays the scanner pipeline on historical 15m data, records every signal,
then checks forward price action for TP1/TP2/TP3/SL hit rates.

Usage:
    python scripts/evaluate_scanner.py eth_15m_merged.csv
    python scripts/evaluate_scanner.py eth_15m_merged.csv --start 2026-01-01 --end 2026-04-30
    python scripts/evaluate_scanner.py eth_15m_merged.csv --lookforward 200
"""

import sys, os, argparse, json
from datetime import timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import CONFIG
from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import (
    calc_ema, calc_macd, calc_rsi, calc_atr, calc_vwap, calc_vol_ratio,
    calc_swing_bias, calc_phase0, calc_phase0_1h, calc_phase0_crypto, calc_trend_state,
)
from src.modules.m1_macd_v2 import score_m1_v2 as score_m1
from src.modules.m2_ema import score_m2
from src.modules.m3_vwap import score_m3
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross, score_m4
from src.modules.m5_liquidation import (
    build_volume_profile, find_magnets, find_gaps, score_m5, find_support_resistance,
)
from src.modules.m7_market_regime import m7_prepare_data, m7_get_row, score_m7
from src.modules.m8_funding import score_m8_funding
from src.engine import calc_ics, check_entry_filters, run_gatekeepers
from src.modules.m9_volatility import RegimeState, compute_vol_regime, score_vol_regime
from src.modules.m11_momentum import score_m11_mtf_momentum
from src.modules.m13_structure import score_m13
from src.modules.m14_sweep import score_m14
from src.modules.direction_resolver import resolve_direction, score_targets
from src.modules.veto_system import evaluate_vetoes
from src.modules.coherence_liquidity import check_coherence
from src.sl_tp import calc_trade_levels


def compute_all_indicators(df_15m, cfg):
    """Compute all indicators on the full dataset."""
    df_15m['vwap'] = calc_vwap(df_15m['High'], df_15m['Low'], df_15m['Close'], df_15m['Volume'], cfg['VWAP_LOOKBACK'])
    df_15m['vol_ma20'] = df_15m['Volume'].rolling(20).mean()
    taker_base = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    df_15m['taker_ratio'] = (taker_base / total_vol.replace(0, np.nan)).fillna(cfg['TAKER_FILLNA'])
    df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], cfg['ATR_PERIOD'])
    df_15m['vol_ratio'] = calc_vol_ratio(df_15m['Volume'])
    df_15m['rsi'] = calc_rsi(df_15m['Close'], 14)

    df_1h = resample_ohlcv(df_15m, '1H')
    df_2h = resample_ohlcv(df_15m, '2H')
    df_4h = resample_ohlcv(df_15m, '4H')
    df_1d = resample_ohlcv(df_15m, '1D')

    df_1h['macd_line'], df_1h['macd_signal'], df_1h['macd_hist'] = calc_macd(
        df_1h['Close'], cfg['MACD_FAST'], cfg['MACD_SLOW'], cfg['MACD_SIGNAL'])
    df_1h['ema_fast'] = calc_ema(df_1h['Close'], cfg['EMA_FAST'])
    df_1h['ema_slow'] = calc_ema(df_1h['Close'], cfg['EMA_SLOW'])
    df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], cfg['ATR_PERIOD'])
    df_1h['rsi'] = calc_rsi(df_1h['Close'], 14)

    df_4h['ema_fast'] = calc_ema(df_4h['Close'], cfg['EMA_FAST'])
    df_4h['ema_slow'] = calc_ema(df_4h['Close'], cfg['EMA_SLOW'])
    df_2h['ema_fast'] = calc_ema(df_2h['Close'], cfg['EMA_FAST'])
    df_2h['ema_slow'] = calc_ema(df_2h['Close'], cfg['EMA_SLOW'])

    df_15m['cvd_15m'] = calc_cvd_15m(df_15m)
    df_15m['cvd_divergence_15m'] = detect_cvd_divergence_15m(df_15m, cfg['CVD_LOOKBACK'], cfg['CVD_DIVERGENCE_WINDOW'])

    df_2h['cvd_2h'] = calc_cvd_2h(df_2h)
    df_2h['cvd_zl_state'], df_2h['cvd_zl_cross_bar'], df_2h['cvd_zl_cross_dir'] = detect_cvd_zero_cross(df_2h)

    df_1d['swing_bias'] = calc_swing_bias(df_1d)
    df_1h['phase0'] = calc_phase0_1h(df_1h)
    df_1d['trend'], df_1d['trend_score'] = calc_trend_state(df_1d)

    df_4h['macd_line'], df_4h['macd_signal'], df_4h['macd_hist'] = calc_macd(
        df_4h['Close'], cfg['MACD_FAST'], cfg['MACD_SLOW'], cfg['MACD_SIGNAL'])

    return df_15m, df_1h, df_2h, df_4h, df_1d


def scan_bar(df_15m, df_1h, df_2h, df_4h, df_1d, idx, idx_1h, idx_2h, idx_4h, idx_1d, cfg):
    """Run scanner pipeline on a single bar. Returns signal dict or None."""
    row = df_15m.iloc[idx]

    atr_1h = df_1h['atr'].iloc[idx_1h] if idx_1h >= 0 else np.nan
    swing_bias = df_1d['swing_bias'].iloc[idx_1d] if idx_1d >= 0 else None
    phase0_val = df_1h['phase0'].iloc[idx_1h] if idx_1h >= 0 else None
    trend_dir = df_1d['trend'].iloc[idx_1d] if idx_1d >= 0 else 'NEUTRAL'

    # M9
    regime_state = RegimeState(config=cfg)
    vol_regime, m9_raw, _ = compute_vol_regime(df_15m, df_1h, idx, idx_1h, regime_state=regime_state, config=cfg)
    if vol_regime in cfg.get('M9_BLOCK_REGIMES', ['CRISIS']):
        return None

    # M13
    m13_status, m13_score_raw, m13_details = score_m13(df_1h, idx_1h, 'NEUTRAL', df_15m, idx)
    m13_bias = m13_details.get('m13_bias', 'NEUTRAL')

    # M7
    m7_score = 0.5; m7_status = 'SKIP'
    # Skip M7 in evaluation (needs live data)

    # Targets
    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    closes = df_15m['Close'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)
    bin_centers, vol_profile, bin_edges = build_volume_profile(
        highs[:idx+1], lows[:idx+1], closes[:idx+1], volumes[:idx+1],
        n_bins=cfg['M5_VP_BINS'], lookback=cfg['M5_VP_LOOKBACK'])
    magnets = find_magnets(bin_centers, vol_profile) if bin_centers is not None else []
    gaps = find_gaps(bin_centers, vol_profile) if bin_centers is not None else []
    sr_levels = find_support_resistance(df_15m, idx)

    atr_1h_val = atr_1h
    current_price = float(row['Close'])
    long_tgt_score, long_tgt_details = score_targets(
        current_price, magnets, gaps, sr_levels, 'LONG', atr_1h=atr_1h_val)
    short_tgt_score, short_tgt_details = score_targets(
        current_price, magnets, gaps, sr_levels, 'SHORT', atr_1h=atr_1h_val)

    # Direction
    direction, dir_size_mult, dir_details = resolve_direction(
        vol_regime, m9_raw if m9_raw else 0.5,
        m13_bias, m13_score_raw, m13_details,
        m7_score=m7_score, m7_status=m7_status,
        swing_bias_1d=swing_bias, trend_dir=trend_dir, config=cfg,
        long_target_score=long_tgt_score, short_target_score=short_tgt_score,
        long_target_details=long_tgt_details, short_target_details=short_tgt_details,
    )
    if direction == 'NEUTRAL':
        return None

    # Re-score with direction
    m9_status, m9_score, _ = score_vol_regime(vol_regime, m9_raw, direction, trend_dir)
    m13_status, m13_score, m13_details = score_m13(df_1h, idx_1h, direction, df_15m, idx)

    # Modules
    m1_dir, m1_score, _ = score_m1(df_1h, idx_1h, cfg, df_15m=df_15m, idx_15m=idx)
    m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)

    # M2 veto
    if cfg.get('M2_VETO_ENABLED', False):
        m2_veto_thresh = cfg.get('M2_VETO_THRESHOLD', 0.40)
        if direction == 'LONG' and m2_status == 'BEARISH' and m2_score < m2_veto_thresh:
            return None
        if direction == 'SHORT' and m2_status == 'BULLISH' and m2_score < m2_veto_thresh:
            return None

    m3_status, m3_score, _ = score_m3(df_15m, idx, direction, cfg)
    m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, idx_2h, direction, cfg)
    # v7.2: Extract and normalize divergence string from details dict
    m4_div_str = 'NONE'
    if isinstance(m4_div, dict):
        m4_div_str = m4_div.get('layer_a_div', 'NONE')
    if m4_div_str.endswith('_BASE'):
        m4_div_str = m4_div_str.replace('_BASE', '')
    m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction, cfg,
        n_bins=cfg['M5_VP_BINS'], lookback=cfg['M5_VP_LOOKBACK'])

    # M8 — skip (needs live funding)
    m8_score = 0.5; m8_status = 'SKIP'

    # M11
    m11_score = 0.5; m11_status = 'SKIP'
    if cfg.get('M11_ENABLED', False):
        try:
            m11_status, m11_score, _ = score_m11_mtf_momentum(df_15m, df_1h, df_4h, idx, idx_1h, idx_4h, direction)
        except:
            pass

    # M14
    m14_score = 0.5; m14_status = 'SKIP'
    if cfg.get('M14_ENABLED', True):
        _swing_levels = m13_details.get('swing_lows', []) if direction == 'LONG' else m13_details.get('swing_highs', [])
        if _swing_levels:
            try:
                m14_status, m14_score, _ = score_m14(df_15m, idx, direction, _swing_levels, config=cfg)
            except:
                pass

    # ICS
    ics, effective_floor = calc_ics(
        m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
        m7_score=m7_score, m8_score=m8_score,
        use_m7=False, use_m8=False,
        m9_score=m9_score, use_m9=True,
        m10_score=0.5, use_m10=False,
        m11_score=m11_score, use_m11=m11_status != 'SKIP',
        m13_score=m13_score, use_m13=cfg.get('M13_ENABLED', False),
        m14_score=m14_score, use_m14=m14_status == 'PASS',
        config=cfg,
    )

    # Veto
    if cfg.get('VETO_ENABLED', False):
        m4_disagree = (direction == 'LONG' and m4_div_str == 'BEARISH') or (direction == 'SHORT' and m4_div_str == 'BULLISH')
        m5_disagree = (m5_status == 'FAIL')
        dir_veto = m4_disagree and m5_disagree
        veto = evaluate_vetoes(cfg, vol_regime=vol_regime, dir_veto=dir_veto,
                               m9_status=m9_status, m10_status='SKIP', m11_status=m11_status)
        if veto.hard_blocked:
            return None

    # Coherence
    if cfg.get('COHERENCE_CHECK_ENABLED', True):
        is_coherent, conflicts, coherence_penalty = check_coherence(
            direction, m4_div, m5_details if isinstance(m5_details, dict) else {},
            m13_bias, vol_regime, m7_score=m7_score, m2_status=m2_status, config=cfg)
        if not is_coherent:
            return None
        ics -= coherence_penalty

    # Threshold
    threshold = cfg['ICS_THRESHOLD_CAUTION'] if phase0_val and phase0_val >= 0.40 else cfg['ICS_THRESHOLD_NORMAL']

    # M3 hard fail
    if m3_status == 'FAIL':
        return None

    # ICS check
    if ics < effective_floor or ics < threshold:
        return None

    # Gatekeepers
    gatekeeper = run_gatekeepers(
        direction, vol_regime, m7_score, m7_status, {},
        m9_score, m9_status, 0.5, 'SKIP', trend_dir, config=cfg)
    if not gatekeeper.passed:
        return None

    # Entry filters
    passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h, config=cfg)
    if not passed:
        return None

    # Calculate TP/SL — Liquidity-Aware
    entry_price = float(row['Close'])
    atr_for_sl = float(atr_1h) if not pd.isna(atr_1h) else float(row['atr'])

    trade_levels = calc_trade_levels(
        entry_price, direction, atr_for_sl,
        row.get('vol_ratio', np.nan),
        magnets=magnets,
        sr_levels=sr_levels,
        liq_levels=None,
        cfg=cfg,
    )

    return {
        'bar_idx': idx,
        'timestamp': str(row['Open time']),
        'direction': direction,
        'entry': entry_price,
        'sl': trade_levels['sl'],
        'tp1': trade_levels['tp1'],
        'tp2': trade_levels['tp2'],
        'tp3': trade_levels['tp3'],
        'sl_source': trade_levels['sl_source'],
        'tp1_source': trade_levels['tp1_source'],
        'ics': round(float(ics), 4),
        'dir_size_mult': round(float(dir_size_mult), 3),
        'vol_regime': vol_regime,
        'swing_bias': swing_bias,
        'm1_score': round(float(m1_score), 3),
        'm3_score': round(float(m3_score), 3),
        'm4_score': round(float(m4_score), 3),
        'm5_score': round(float(m5_score), 3),
        'm9_score': round(float(m9_score), 3),
        'm11_score': round(float(m11_score), 3),
        'm13_score': round(float(m13_score), 3),
    }


def check_outcome(df_15m, signal, lookforward_bars):
    """Check what happened after the signal. Returns outcome dict."""
    idx = signal['bar_idx']
    direction = signal['direction']
    entry = signal['entry']
    sl = signal['sl']
    tp1 = signal['tp1']
    tp2 = signal['tp2']
    tp3 = signal['tp3']

    end_idx = min(idx + lookforward_bars, len(df_15m))

    tp1_hit = False; tp2_hit = False; tp3_hit = False; sl_hit = False
    tp1_bar = None; tp2_bar = None; tp3_bar = None; sl_bar = None
    max_favorable = 0.0; max_adverse = 0.0
    exit_price = None; exit_reason = None; bars_held = 0

    for j in range(idx + 1, end_idx):
        high = float(df_15m['High'].iloc[j])
        low = float(df_15m['Low'].iloc[j])
        bars_held = j - idx

        if direction == 'LONG':
            favorable = (high - entry) / entry * 100
            adverse = (entry - low) / entry * 100
            if not tp1_hit and high >= tp1:
                tp1_hit = True; tp1_bar = bars_held
            if not tp2_hit and high >= tp2:
                tp2_hit = True; tp2_bar = bars_held
            if not tp3_hit and high >= tp3:
                tp3_hit = True; tp3_bar = bars_held
                if not exit_reason:
                    exit_price = tp3; exit_reason = 'TP3'
            if low <= sl:
                sl_hit = True; sl_bar = bars_held
                if not exit_reason:
                    exit_price = sl; exit_reason = 'SL'
                break
        else:
            favorable = (entry - low) / entry * 100
            adverse = (high - entry) / entry * 100
            if not tp1_hit and low <= tp1:
                tp1_hit = True; tp1_bar = bars_held
            if not tp2_hit and low <= tp2:
                tp2_hit = True; tp2_bar = bars_held
            if not tp3_hit and low <= tp3:
                tp3_hit = True; tp3_bar = bars_held
                if not exit_reason:
                    exit_price = tp3; exit_reason = 'TP3'
            if high >= sl:
                sl_hit = True; sl_bar = bars_held
                if not exit_reason:
                    exit_price = sl; exit_reason = 'SL'
                break

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

    # If neither TP3 nor SL hit, mark as open
    if not exit_reason:
        exit_price = float(df_15m['Close'].iloc[end_idx - 1])
        exit_reason = 'OPEN'
        bars_held = end_idx - idx

    pnl_pct = (exit_price - entry) / entry if direction == 'LONG' else (entry - exit_price) / entry

    return {
        'tp1_hit': tp1_hit, 'tp2_hit': tp2_hit, 'tp3_hit': tp3_hit, 'sl_hit': sl_hit,
        'tp1_bar': tp1_bar, 'tp2_bar': tp2_bar, 'tp3_bar': tp3_bar, 'sl_bar': sl_bar,
        'exit_price': exit_price, 'exit_reason': exit_reason,
        'pnl_pct': round(pnl_pct * 100, 4),
        'bars_held': bars_held,
        'max_favorable': round(max_favorable, 4),
        'max_adverse': round(max_adverse, 4),
    }


def print_report(results):
    """Print comprehensive scanner evaluation report."""
    signals = [r for r in results if r.get('signal')]
    if not signals:
        print("\n  No signals generated.")
        return

    total = len(signals)
    outcomes = [r['outcome'] for r in signals]
    winners = [o for o in outcomes if o['pnl_pct'] > 0]
    losers = [o for o in outcomes if o['pnl_pct'] < 0]
    opens = [o for o in outcomes if o['exit_reason'] == 'OPEN']

    longs = [r for r in signals if r['signal']['direction'] == 'LONG']
    shorts = [r for r in signals if r['signal']['direction'] == 'SHORT']

    tp1_hits = sum(1 for o in outcomes if o['tp1_hit'])
    tp2_hits = sum(1 for o in outcomes if o['tp2_hit'])
    tp3_hits = sum(1 for o in outcomes if o['tp3_hit'])
    sl_hits = sum(1 for o in outcomes if o['sl_hit'])

    print("\n" + "═" * 70)
    print("  JIMI SCANNER — PERFORMANCE EVALUATION")
    print("═" * 70)

    print(f"\n  Total Signals:     {total}")
    print(f"  LONG signals:      {len(longs)}")
    print(f"  SHORT signals:     {len(shorts)}")

    print(f"\n  ── Outcome Distribution ──")
    print(f"  Winners:           {len(winners)} ({len(winners)/total*100:.1f}%)")
    print(f"  Losers:            {len(losers)} ({len(losers)/total*100:.1f}%)")
    print(f"  Still Open:        {len(opens)} ({len(opens)/total*100:.1f}%)")

    closed = [o for o in outcomes if o['exit_reason'] != 'OPEN']
    if closed:
        closed_wr = sum(1 for o in closed if o['pnl_pct'] > 0) / len(closed) * 100
        print(f"\n  ── Closed Trades Only ──")
        print(f"  Win Rate:          {closed_wr:.1f}%")
        print(f"  Avg PnL:           {np.mean([o['pnl_pct'] for o in closed]):.2f}%")
        print(f"  Avg Win:           {np.mean([o['pnl_pct'] for o in closed if o['pnl_pct'] > 0]):.2f}%" if any(o['pnl_pct'] > 0 for o in closed) else "")
        print(f"  Avg Loss:          {np.mean([o['pnl_pct'] for o in closed if o['pnl_pct'] < 0]):.2f}%" if any(o['pnl_pct'] < 0 for o in closed) else "")

    print(f"\n  ── TP/SL Hit Rates ──")
    print(f"  TP1 hit:           {tp1_hits}/{total} ({tp1_hits/total*100:.1f}%)")
    print(f"  TP2 hit:           {tp2_hits}/{total} ({tp2_hits/total*100:.1f}%)")
    print(f"  TP3 hit:           {tp3_hits}/{total} ({tp3_hits/total*100:.1f}%)")
    print(f"  SL hit:            {sl_hits}/{total} ({sl_hits/total*100:.1f}%)")

    # Avg bars to TP
    tp1_bars = [o['tp1_bar'] for o in outcomes if o['tp1_bar'] is not None]
    tp2_bars = [o['tp2_bar'] for o in outcomes if o['tp2_bar'] is not None]
    sl_bars = [o['sl_bar'] for o in outcomes if o['sl_bar'] is not None]
    if tp1_bars:
        print(f"\n  Avg bars to TP1:   {np.mean(tp1_bars):.1f}")
    if tp2_bars:
        print(f"  Avg bars to TP2:   {np.mean(tp2_bars):.1f}")
    if sl_bars:
        print(f"  Avg bars to SL:    {np.mean(sl_bars):.1f}")

    # Direction breakdown
    print(f"\n  ── Direction Breakdown ──")
    for d, label in [('LONG', 'LONG'), ('SHORT', 'SHORT')]:
        d_signals = [r for r in signals if r['signal']['direction'] == d]
        if not d_signals:
            continue
        d_outcomes = [r['outcome'] for r in d_signals]
        d_closed = [o for o in d_outcomes if o['exit_reason'] != 'OPEN']
        d_wr = sum(1 for o in d_closed if o['pnl_pct'] > 0) / len(d_closed) * 100 if d_closed else 0
        d_pnl = np.mean([o['pnl_pct'] for o in d_closed]) if d_closed else 0
        d_tp1 = sum(1 for o in d_outcomes if o['tp1_hit'])
        d_tp2 = sum(1 for o in d_outcomes if o['tp2_hit'])
        d_sl = sum(1 for o in d_outcomes if o['sl_hit'])
        print(f"  {label}: {len(d_signals)} signals | WR {d_wr:.1f}% | Avg PnL {d_pnl:+.2f}% | TP1 {d_tp1} TP2 {d_tp2} SL {d_sl}")

    # ICS distribution
    ics_vals = [r['signal']['ics'] for r in signals]
    print(f"\n  ── ICS Distribution ──")
    print(f"  Mean:              {np.mean(ics_vals):.4f}")
    print(f"  Median:            {np.median(ics_vals):.4f}")
    print(f"  Min:               {np.min(ics_vals):.4f}")
    print(f"  Max:               {np.max(ics_vals):.4f}")

    # ICS vs performance
    print(f"\n  ── ICS vs Win Rate ──")
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]:
        bucket = [r for r in signals if lo <= r['signal']['ics'] < hi]
        if not bucket:
            continue
        b_closed = [r['outcome'] for r in bucket if r['outcome']['exit_reason'] != 'OPEN']
        b_wr = sum(1 for o in b_closed if o['pnl_pct'] > 0) / len(b_closed) * 100 if b_closed else 0
        b_pnl = np.mean([o['pnl_pct'] for o in b_closed]) if b_closed else 0
        print(f"  ICS {lo:.1f}-{hi:.1f}: {len(bucket):>4} signals | WR {b_wr:.1f}% | Avg PnL {b_pnl:+.2f}%")

    # Monthly breakdown
    monthly = {}
    for r in signals:
        ts = r['signal']['timestamp'][:7]  # YYYY-MM
        if ts not in monthly:
            monthly[ts] = {'count': 0, 'wins': 0, 'pnl': 0}
        monthly[ts]['count'] += 1
        if r['outcome']['pnl_pct'] > 0:
            monthly[ts]['wins'] += 1
        if r['outcome']['exit_reason'] != 'OPEN':
            monthly[ts]['pnl'] += r['outcome']['pnl_pct']

    print(f"\n  ── Monthly Performance ──")
    print(f"  {'Month':<10} {'Signals':>8} {'WR':>7} {'Avg PnL':>10}")
    print(f"  {'─'*10} {'─'*8} {'─'*7} {'─'*10}")
    for month in sorted(monthly.keys()):
        m = monthly[month]
        wr = m['wins'] / m['count'] * 100 if m['count'] > 0 else 0
        avg_pnl = m['pnl'] / m['count'] if m['count'] > 0 else 0
        print(f"  {month:<10} {m['count']:>8} {wr:>6.1f}% {avg_pnl:>+9.2f}%")

    # Max drawdown from signal sequence
    equity = [0]
    for r in sorted(signals, key=lambda x: x['signal']['timestamp']):
        if r['outcome']['exit_reason'] != 'OPEN':
            equity.append(equity[-1] + r['outcome']['pnl_pct'])
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    max_dd = abs((equity - peak).min()) if len(equity) > 1 else 0
    total_pnl = equity[-1] if len(equity) > 1 else 0

    print(f"\n  ── Portfolio Stats ──")
    print(f"  Cumulative PnL:    {total_pnl:+.2f}%")
    print(f"  Max Drawdown:      {max_dd:.2f}%")
    if max_dd > 0:
        print(f"  Return/DD Ratio:   {total_pnl/max_dd:.1f}×")

    print("\n" + "═" * 70)


def main():
    parser = argparse.ArgumentParser(description='JIMI Scanner Performance Evaluator')
    parser.add_argument('csv', help='Path to 15m OHLCV CSV')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--lookforward', type=int, default=200, help='Bars to look forward for outcome (default: 200 = ~50h on 15m)')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--export', help='Export signals to JSON file')
    parser.add_argument('--step', type=int, default=1, help='Evaluate every Nth bar (for speed, default=1)')
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else CONFIG

    print("Loading data...")
    df_15m = load_data(args.csv)
    print(f"  Loaded {len(df_15m):,} bars: {df_15m['Open time'].iloc[0]} → {df_15m['Open time'].iloc[-1]}")

    if args.start:
        df_15m = df_15m[df_15m['Open time'] >= args.start].reset_index(drop=True)
    if args.end:
        df_15m = df_15m[df_15m['Open time'] <= args.end].reset_index(drop=True)
    if args.start or args.end:
        print(f"  Filtered to {len(df_15m):,} bars: {df_15m['Open time'].iloc[0]} → {df_15m['Open time'].iloc[-1]}")

    print("Computing indicators...")
    df_15m, df_1h, df_2h, df_4h, df_1d = compute_all_indicators(df_15m, cfg)

    # Build index maps
    df_1h['_ts'] = df_1h['Open time'].values.astype('datetime64[ns]')
    df_2h['_ts'] = df_2h['Open time'].values.astype('datetime64[ns]')
    df_4h['_ts'] = df_4h['Open time'].values.astype('datetime64[ns]')
    df_1d['_ts'] = df_1d['Open time'].values.astype('datetime64[ns]')

    def find_tf_idx(ts, df_tf):
        idx = df_tf['_ts'].searchsorted(ts, side='right') - 1
        return max(idx, 0)

    warmup = cfg.get('WARMUP_BARS_1H', 50)
    warmup_time = df_1h['Open time'].iloc[min(warmup, len(df_1h)-1)]
    start_idx = df_15m[df_15m['Open time'] >= warmup_time].index[0]
    print(f"  Warmup: skip until bar {start_idx} ({warmup_time})")

    total_bars = len(df_15m) - start_idx
    print(f"\nScanning {total_bars:,} bars (step={args.step})...")

    results = []
    signals_found = 0
    last_pct = -1

    for i in range(start_idx, len(df_15m), args.step):
        ts = df_15m['Open time'].iloc[i]
        idx_1h = find_tf_idx(ts, df_1h)
        idx_2h = find_tf_idx(ts, df_2h)
        idx_4h = find_tf_idx(ts, df_4h)
        idx_1d = find_tf_idx(ts, df_1d)

        signal = scan_bar(df_15m, df_1h, df_2h, df_4h, df_1d, i, idx_1h, idx_2h, idx_4h, idx_1d, cfg)

        if signal:
            outcome = check_outcome(df_15m, signal, args.lookforward)
            results.append({'signal': signal, 'outcome': outcome})
            signals_found += 1

        # Progress
        pct = int((i - start_idx) / total_bars * 100)
        if pct != last_pct and pct % 5 == 0:
            last_pct = pct
            print(f"  {pct}% — {signals_found} signals so far...", flush=True)

    print(f"\n  Done. {signals_found} signals from {total_bars:,} bars.")

    print_report(results)

    if args.export:
        with open(args.export, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Exported to {args.export}")


if __name__ == '__main__':
    main()
