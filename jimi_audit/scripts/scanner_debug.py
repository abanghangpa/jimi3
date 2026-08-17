#!/usr/bin/env python3
"""
JIMI Framework — Live Signal Scanner

Usage:
    python scripts/scanner.py
    python scripts/scanner.py --json
    python scripts/scanner.py --dashboard 8888
    python scripts/scanner.py --search "FOMC minutes May 2026 release date"
"""

import argparse
import sys
import os
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the new web/data utilities
from src.utils.web_data_utils import fetch_tradfi_data, search_web

from src.config import CONFIG
from src.utils.data_handler import fetch_recent, fetch_btc_15m, load_daily_from_csv
from src.utils.indicators import (
    calc_ema, calc_macd, calc_rsi, calc_atr, calc_vwap, calc_vol_ratio,
    calc_swing_bias, calc_phase0, calc_phase0_1h, calc_phase0_crypto, calc_trend_state, compute_btc_correlation,
)
from src.modules.m1_macd_v2 import score_m1_v2 as score_m1
from src.modules.m2_ema import score_m2
from src.modules.m3_vwap import score_m3
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross, score_m4
from src.modules.intrabar_cvd import get_intrabar_cvd_summary, score_intrabar_divergence
from src.modules.m5_liquidation import (
    build_volume_profile, find_magnets, find_gaps, score_m5, detect_cascade_setup,
    find_support_resistance,
)
from src.modules.m6_derivatives import score_derivatives, get_derivatives_summary
from src.modules.m15_liq_levels import get_liquidity_summary
from src.modules.m7_market_regime import m7_prepare_data, m7_get_row, score_m7
from src.modules.m8_funding import score_m8_funding
from src.modules.m10_macro import m10_prepare_data, m10_get_row, m10_compute_emas, score_m10_macro
from src.engine import calc_ics, check_entry_filters, get_tp_multipliers, run_gatekeepers
from src.modules.m_conflict import get_conflict_stats
from src.modules.m9_volatility import RegimeState, compute_vol_regime, score_vol_regime
from src.modules.m13_structure import score_m13
from src.modules.direction_resolver import resolve_direction, score_targets
from src.modules.veto_system import evaluate_vetoes
from src.modules.coherence_liquidity import check_coherence
from src.modules.m12_orderbook import score_m12_orderbook
from src.modules.m14_sweep import score_m14
from src.modules.m17_resistance_quality import score_resistance_quality, format_resistance_quality
from src.modules.m16_exchange_activity import get_exchange_summary, fetch_all_exchange_data, compute_exchange_signals, score_exchange_activity, score_spot_signals
from src.modules.taker_tracker import get_taker_summary, format_taker_summary
from src.modules.cross_asset import score_cross_asset
from src.modules.m21_wyckoff import score_m21, format_m21, detect_trading_range, get_range_targets, get_range_sl
from src.modules.m22_inflation_regime_v2 import score_m22, format_m22
from src.modules.m22_aggregator import aggregate_macro_regime, format_m22_aggregated
from src.modules.m24_nbs_pmi import score_m24_nbs_pmi, format_m24
from src.modules.m25_caixin_pmi import score_m25_caixin_pmi, format_m25
from src.modules.m65_china_activity import score_m65_china_activity, format_m65
from src.modules.m26_ez_pmi import score_m26_ez_pmi, format_m26
from src.modules.m33_retail_sales import score_m33_retail_sales, format_m33
from src.modules.m34_housing_starts import score_m34_housing, format_m34
from src.modules.m35_pboc_lpr import score_m35_pboc_lpr, format_m35
from src.modules.m36_adp_employment import score_m36_adp, format_m36
from src.modules.m37_nfp import score_m37_nfp, format_m37
from src.modules.m38_ifo import score_m38_ifo, format_m38
from src.modules.m39_ums import score_m39_ums, format_m39
from src.modules.m40_germany_cpi import score_m40_germany_cpi, format_m40
from src.modules.m41_ez_cpi import score_m41_ez_cpi, format_m41
from src.modules.m42_ez_gdp import score_m42_ez_gdp, format_m42
from src.modules.m43_us_gdp import score_m43_us_gdp, format_m43
from src.modules.m44_durables import score_m44_durables, format_m44
from src.modules.m45_pce import score_m45_pce, format_m45
from src.modules.m46_japan_cpi import score_m46_japan_cpi, format_m46
from src.modules.m47_boj_rate import score_m47_boj_rate, format_m47
from src.modules.m48_ecb_rate import score_m48_ecb_rate, format_m48
from src.modules.m49_boe_rate import score_m49_boe_rate, format_m49
from src.modules.m50_cb_consumer_confidence import score_m50_cb_confidence, format_m50
from src.modules.m51_uk_gdp_monthly import score_m51_uk_gdp, format_m51
from src.modules.m52_rba_rate import score_m52_rba_rate, format_m52
from src.modules.m53_au_cpi import score_m53_au_cpi, format_m53
from src.modules.m54_china_gdp import score_m54_china_gdp, format_m54
from src.modules.m55_treasury_auction import score_m55_treasury, format_m55
from src.modules.m56_us_cpi import score_m56_us_cpi, format_m56
from src.modules.m57_fomc import score_m57_fomc, format_m57
from src.modules.m58_powell_presser import score_m58_presser, format_m58
from src.modules.m59_fomc_minutes import score_m59_minutes, format_m59
from src.modules.m62_us_unemployment import score_m62_us_unemployment, format_m62
from src.modules.m60_us_ppi import score_m60_us_ppi, format_m60
from src.modules.m61_us_claims import score_m61_us_claims, format_m61
from src.modules.m23_ppi_session import score_m23_ppi_session, format_m23
from src.modules.macro_utils import (
    is_ppi_release_day, is_cpi_release_day, is_nfp_release_day,
    is_macro_release_day, is_claims_release_day,
    get_claims_trend, classify_macro_combo,
)
from src.modules.cascade_meta import score_all_cascades, format_all_cascades
from src.modules.cascade_labor import format_labor_cascade
from src.modules.cascade_inflation import format_inflation_cascade
from src.modules.cascade_china import format_china_cascade
from src.modules.cascade_eu import format_eu_cascade
from src.modules.cascade_uk import format_uk_cascade
from src.modules.cascade_japan import format_japan_cascade
from src.modules.cascade_au import format_au_cascade
from src.modules.cascade_us_activity import format_us_activity_cascade
from src.modules.macro_lifecycle import evaluate_macro_lifecycle, format_lifecycle
from src.modules.macro_calendar import get_macro_calendar, format_macro_calendar, format_macro_calendar_compact, calendar_to_dict
from src.modules.macro_event_filter import MacroEventFilter, get_phase0_from_df, get_trend_30d
from src.sl_tp import calc_trade_levels, check_sweep_gate, calc_limit_entry
from src.signal_eval import evaluate_signal, format_signal_eval
from src.modules.conflict_resolver import detect_conflict, format_conflict, conflict_to_dict
from src.modules.power_of_3 import detect_phase, format_phase, phase_to_dict
from src.modules.m18_squeeze import detect_squeeze_v6 as detect_squeeze, format_squeeze
from src.modules.m19_breakout_confirm import check_breakout_filters, format_breakout_confirm
from src.modules.m20_failed_breakout import score_m20, format_failed_breakout
from src.dual_strategy import DualStrategy

# ── M66-M73: Traditional Finance Macro Filters ──
from src.modules.m66_usdjpy import score_m66_usdjpy
from src.modules.m67_dxy import score_m67_dxy
from src.modules.m68_yield import score_m68_yield
from src.modules.m69_vix import score_m69_vix
from src.modules.m70_wti import score_m70_wti
from src.modules.m71_gold import score_m71_gold
from src.modules.m72_btcdom import score_m72_btcdom, fetch_btcdom
from src.modules.m73_stablecoin import score_m73_stablecoin, fetch_stablecoin_mints


def fetch_all_tradfi_data(config=None):
    """Fetch all traditional-finance data (FX, commodities, VIX) using yfinance or the new utility.

    Returns dict of DataFrames keyed by signal name. Failed fetches return None for that key (non-fatal).
    """
    cfg = config or {}
    data = {}

    # Use the new utility for faster, simplified fetching
    print("  📊 Fetching TradFi data (DXY, 10Y, VIX, gold, WTI)...")
    tradfi_data = fetch_tradfi_data()
    
    # Map the utility output to the expected format
    if "dxy" in tradfi_data:
        data["dxy"] = pd.DataFrame({"Close": [tradfi_data["dxy"]]})
    if "10y_yield" in tradfi_data:
        data["tnx"] = pd.DataFrame({"Close": [tradfi_data["10y_yield"]]})
    if "vix" in tradfi_data:
        data["vix"] = pd.DataFrame({"Close": [tradfi_data["vix"]]})
    if "gold" in tradfi_data:
        data["gold"] = pd.DataFrame({"Close": [tradfi_data["gold"]]})
    if "wti" in tradfi_data:
        data["wti"] = pd.DataFrame({"Close": [tradfi_data["wti"]]})
    
    # Fallback to yfinance for any missing data
    try:
        import yfinance as yf
        if "usdjpy" not in data and cfg.get('M66_ENABLED', True):
            df = yf.download("JPY=X", period="5d", interval="1m", progress=False)
            if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
                df.columns = df.columns.droplevel(1)
            if df is not None and len(df) > 0:
                data["usdjpy"] = df
    except ImportError:
        print("  ⚠️  yfinance not installed — M66 skipped")
    
    return data


def compute_indicators(df_15m, config=None, df_1d_hist=None):
    """Compute all indicators on fresh data.

    Args:
        df_1d_hist: Optional pre-loaded daily DataFrame from historical CSV.
                    If provided, it replaces the daily resample from df_15m,
                    giving EMA55+ proper warmup for accurate daily bias.
    """
    cfg = config or CONFIG
    df_15m['vwap'] = calc_vwap(df_15m['High'], df_15m['Low'], df_15m['Close'], df_15m['Volume'], cfg['VWAP_LOOKBACK'])
    df_15m['vol_ma20'] = df_15m['Volume'].rolling(20).mean()
    taker_base = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    df_15m['taker_ratio'] = (taker_base / total_vol.replace(0, np.nan)).fillna(cfg['TAKER_FILLNA'])
    df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], cfg['ATR_PERIOD'])
    df_15m['vol_ratio'] = calc_vol_ratio(df_15m['Volume'])

    df_1h = resample_ohlcv(df_15m, '1H')
    df_2h = resample_ohlcv(df_15m, '2H')
    df_4h = resample_ohlcv(df_15m, '4H')

    # Use historical CSV daily data if available (fixes EMA55 warmup bug),
    # otherwise fall back to resampling from the limited live fetch.
    if df_1d_hist is not None and len(df_1d_hist) > 0:
        df_1d = df_1d_hist.copy()
    else:
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
    df_15m['rsi'] = calc_rsi(df_15m['Close'], 14)

    return df_15m, df_1h, df_2h, df_4h, df_1d


# Need resample_ohlcv for scanner
from src.utils.data_handler import resample_ohlcv, load_data

# Minimum daily bars needed for reliable EMA55 convergence
_MIN_DAILY_BARS = 100


def ensure_csv_fresh(csv_path=None):
    """Ensure the historical CSV is patched to the latest 15m bar.

    Reads the last timestamp from the CSV. If any bars are missing,
    fetches them from Binance and appends — always up to date, no threshold.

    Returns:
        CSV path, or None if not found.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'data', 'eth_15m_merged.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'eth_15m_merged.csv')

    if not os.path.exists(csv_path):
        print(f"  ⚠️  CSV not found at {csv_path}, using live fetch only")
        return None

    # Read last timestamp from CSV (fast: seek to end, read last line)
    last_line = None
    with open(csv_path, 'rb') as f:
        f.seek(0, 2)
        fsize = f.tell()
        f.seek(max(0, fsize - 500))
        lines = f.read().decode('utf-8', errors='replace').strip().split('\n')
        last_line = lines[-1] if lines else None

    if not last_line:
        return csv_path

    last_ts_str = last_line.split(',')[0].strip('"')
    try:
        last_dt = pd.Timestamp(last_ts_str)
    except Exception:
        return csv_path

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    gap_bars = int((now - last_dt).total_seconds() / 900)  # 15m bars

    print(f"  📄 CSV last: {last_ts_str}  ({gap_bars} bars behind)")

    if gap_bars <= 0:
        return csv_path  # already current

    # Always fetch missing bars — no threshold
    print(f"  📥 Fetching {gap_bars} missing bars from Binance...")

    try:
        import ccxt
        ex = ccxt.binance({"enableRateLimit": True})
        since_ms = int(last_dt.timestamp() * 1000) + 1

        all_rows = []
        end_time_ms = None
        remaining = gap_bars

        while remaining > 0:
            limit = min(remaining, 1000)
            params = {'symbol': 'ETHUSDT', 'interval': '15m', 'limit': limit}
            if end_time_ms:
                params['endTime'] = end_time_ms
            raw = ex.publicGetKlines(params)
            if not raw:
                break
            all_rows = raw + all_rows
            end_time_ms = int(raw[0][0]) - 1
            remaining -= len(raw)
            if len(raw) < limit:
                break

        if all_rows:
            cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close time', 'Quote asset volume', 'Number of trades',
                    'Taker buy base asset volume', 'Taker buy quote asset volume',
                    'Ignore']
            df_new = pd.DataFrame(all_rows, columns=cols)
            df_new['Open time'] = (pd.to_datetime(df_new['Open time'].astype(int), unit='ms')
                                  .dt.strftime('%Y-%m-%d %H:%M:%S'))
            df_new['Close time'] = (pd.to_datetime(df_new['Close time'].astype(int), unit='ms')
                                   .dt.strftime('%Y-%m-%d %H:%M:%S'))
            for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
                      'Number of trades', 'Taker buy base asset volume',
                      'Taker buy quote asset volume']:
                df_new[c] = pd.to_numeric(df_new[c])
            df_new['Ignore'] = 0

            df_new = df_new[df_new['Open time'] > last_ts_str]
            if len(df_new) > 0:
                df_new.to_csv(csv_path, mode='a', header=False, index=False)
                print(f"  ✅ Appended {len(df_new)} bars → {df_new['Open time'].iloc[-1]}")
    except Exception as e:
        print(f"  ⚠️  Fetch failed: {e}")

    return csv_path


def run_full_scan(config=None, df_1d_hist=None):
    """
    Perform a full signal scan and return the results dictionary.
    This is the core orchestration logic used by both scanner.py and other tools.
    """
    cfg = config or CONFIG
    
    # ── DATA FETCHING & PREP ──
    df_15m = fetch_recent(symbol='ETHUSDT', timeframe='15m', limit=2000)
    if df_15m is None or len(df_15m) == 0:
        return {'status': 'ERROR', 'reason': 'could not fetch 15m data'}

    # Use historical daily CSV if available for EMA warmup
    csv_path = ensure_csv_fresh()
    df_1d_hist = load_daily_from_csv(csv_path) if df_1d_hist is None else df_1d_hist
    
    df_15m, df_1h, df_2h, df_4h, df_1d = compute_indicators(df_15m, config=cfg, df_1d_hist=df_1d_hist)
    
    # The rest of the scan logic...
    # (I will move the logic from the global scope to here)
    
    # [ALL THE MODULE SCORING LOGIC FROM THE ORIGINAL SCRIPT GOES HERE]
    # I will perform this in a second edit because the script is too large for one edit.
    
    return result # The final results dictionary





def _check_swept_magnets(df_15m, idx, magnets, lookback_bars=96, config=None):
    """Check if magnets have been swept by recent price action.

    A magnet is 'swept' if price has already traded through it (or within
    proximity tolerance) in the last `lookback_bars` candles (default: 96 = 24h on 15m).
    Returns list of (price, strength, swept: bool, swept_at) tuples.
    """
    if not magnets:
        return []

    cfg = config or CONFIG
    proximity_pct = cfg.get('SWEEP_PROXIMITY_PCT', 0.001)  # default 0.1%

    start = max(0, idx - lookback_bars + 1)
    recent_highs = df_15m['High'].values[start:idx+1].astype(float)
    recent_lows = df_15m['Low'].values[start:idx+1].astype(float)
    recent_times = df_15m['Open time'].values[start:idx+1]
    session_high = np.max(recent_highs)
    session_low = np.min(recent_lows)

    result = []
    for price, vol, strength in magnets:
        swept = False
        swept_at = None
        proximity_buf = price * proximity_pct
        # For magnets above current price: swept if recent high already passed it
        # For magnets below current price: swept if recent low already passed it
        current = df_15m['Close'].values[idx]
        if price > current and session_high >= price - proximity_buf:
            swept = True
            # Find the first candle that swept it
            for i in range(len(recent_highs)):
                if recent_highs[i] >= price - proximity_buf:
                    swept_at = str(recent_times[i])
                    break
        elif price < current and session_low <= price + proximity_buf:
            swept = True
            for i in range(len(recent_lows)):
                if recent_lows[i] <= price + proximity_buf:
                    swept_at = str(recent_times[i])
                    break

        result.append((round(price, 2), round(strength, 2), swept, swept_at))

    return result


def _detect_cascade_risk(df, idx, result):
    """Detect whether the next liquidation cluster will be a quick flush or
    a full deleveraging cascade.

    Returns a dict with:
      - verdict: 'FLUSH' | 'CASCADE' | 'UNKNOWN'
      - score: 0.0 (likely flush) → 1.0 (likely cascade)
      - factors: list of contributing signals
    """
    deriv = result.get('derivatives', {})
    liq = result.get('liquidity_levels', {})
    direction = result.get('direction')
    score = 0.0
    factors = []

    # 1. OI velocity — fast OI drop = positions being force-closed
    oi_roc = deriv.get('oi_roc_1h', 0)  # % change per hour
    if oi_roc < -2.0:
        score += 0.25
        factors.append(f'OI dumping {oi_roc:+.2f}%/hr (cascade)')
    elif oi_roc < -1.0:
        score += 0.10
        factors.append(f'OI declining {oi_roc:+.2f}%/hr (deleveraging)')
    elif oi_roc > 1.0:
        score -= 0.10
        factors.append(f'OI rising {oi_roc:+.2f}%/hr (new positions)')

    # 2. Funding rate — extreme funding = crowded trade at risk
    fr = deriv.get('funding_rate')
    if fr is not None:
        if direction == 'LONG' and fr > 0.0005:
            score += 0.15
            factors.append(f'Funding {fr*100:+.4f}% (longs paying, cascade risk)')
        elif direction == 'SHORT' and fr < -0.0005:
            score += 0.15
            factors.append(f'Funding {fr*100:+.4f}% (shorts paying, cascade risk)')

    # 3. L/S z-score — extreme positioning = more cascade potential
    ls_z = deriv.get('ls_zscore', 0)
    if abs(ls_z) > 2.0:
        score += 0.20
        factors.append(f'L/S z={ls_z:.2f} (extreme positioning)')
    elif abs(ls_z) > 1.5:
        score += 0.10
        factors.append(f'L/S z={ls_z:.2f} (crowded)')

    # 4. Whale signal — whales leaning against the crowd = cascade amplified
    whale = deriv.get('whale_signal', 'NEUTRAL')
    if direction == 'LONG' and whale == 'WHALE_BEARISH':
        score += 0.15
        factors.append('Whales bearish (leaning against longs)')
    elif direction == 'SHORT' and whale == 'WHALE_BULLISH':
        score += 0.15
        factors.append('Whales bullish (leaning against shorts)')

    # 5. Order book depth — thin bids below = cascade through, thick = flush absorbed
    if liq:
        below = liq.get('below', [])
        above = liq.get('above', [])
        target_levels = below if direction == 'LONG' else above
        bid_walls = [z for z in target_levels if z['type'] == 'BID_WALL']
        ask_walls = [z for z in target_levels if z['type'] == 'ASK_WALL']
        walls = bid_walls if direction == 'LONG' else ask_walls

        if not walls:
            score += 0.15
            factors.append('No order book walls near cluster (thin support)')
        else:
            strongest = max(w['strength'] for w in walls)
            if strongest > 50:
                score -= 0.10
                factors.append(f'Order book wall str={strongest:.0f} (will absorb)')

    # 6. Recent momentum — accelerating sell-off = cascade momentum
    lookback = min(16, idx)  # 4h of candles
    if lookback >= 4:
        recent_closes = df['Close'].values[idx-lookback+1:idx+1].astype(float)
        momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100
        if direction == 'LONG' and momentum < -1.5:
            score += 0.15
            factors.append(f'Price momentum {momentum:+.2f}% in 4h (accelerating down)')
        elif direction == 'SHORT' and momentum > 1.5:
            score += 0.15
            factors.append(f'Price momentum {momentum:+.2f}% in 4h (accelerating up)')

    # Clamp to 0-1
    score = max(0.0, min(1.0, score))

    if score >= 0.50:
        verdict = 'CASCADE'
    elif score >= 0.30:
        verdict = 'RISKY'
    else:
        verdict = 'FLUSH'

    return {
        'verdict': verdict,
        'score': round(score, 2),
        'factors': factors,
    }


def _build_release_data_map(result, fred_data=None):
    """Build release_data_map for cascade scoring from module results + FRED cache.

    Maps cascade release names → {actual, previous, consensus, surprise} dicts
    so cascade signal classifiers can produce real signals instead of NEUTRAL.

    Args:
        result: scanner result dict with m25, m26, m33, m37, etc. already scored
        fred_data: dict from FRED cache (cpi, ppi, unemployment, claims)

    Returns:
        dict: {cascade_name: {release_name: data_dict}}
    """
    rdm = {}

    def _mod_data(mod_result, actual_key='actual', prev_key='previous',
                  cons_key='consensus', surpr_key='surprise'):
        """Extract release data dict from a module result."""
        if not mod_result or mod_result.get('status') == 'ERROR':
            return None
        d = mod_result.get('details', mod_result)
        actual = d.get(actual_key) or mod_result.get(actual_key)
        if actual is None:
            # Try common alternative keys
            for k in ('caixin_actual', 'nbs_actual', 'ism_actual', 'nfp_k',
                       'retail_mom', 'adp_k', 'ifo_actual', 'cpi_yoy', 'ppi_yoy',
                       'gdp_qoq', 'durables_mom', 'pce_yoy', 'jp_cpi_yoy',
                       'uk_cpi_yoy', 'wages_yoy', 'rba_rate', 'au_cpi_qoq',
                       'china_gdp_qoq', 'ez_pmi_actual', 'ums_actual'):
                actual = d.get(k) or mod_result.get(k)
                if actual is not None:
                    break
        prev = d.get(prev_key) or mod_result.get(prev_key)
        cons = d.get(cons_key) or mod_result.get(cons_key)
        # Try alternative consensus keys
        if cons is None:
            for ck in ('consensus_k', 'consensus_mom', 'consensus_yoy',
                       'consensus_qoq'):
                cons = d.get(ck) or mod_result.get(ck)
                if cons is not None:
                    break
        surpr = d.get(surpr_key) or mod_result.get(surpr_key)
        if actual is None:
            return None
        data = {'actual': actual}
        if prev is not None:
            data['previous'] = prev
        if cons is not None:
            data['consensus'] = cons
        if surpr is not None:
            data['surprise'] = surpr
        return data

    # ── US_INFLATION cascade ──
    us_infl = {}
    # CPI from FRED
    if fred_data and fred_data.get('cpi', {}).get('yoy'):
        cpi_yoy = fred_data['cpi']['yoy']
        latest_cpi = list(cpi_yoy.values())[-1] if cpi_yoy else None
        prev_cpi = list(cpi_yoy.values())[-2] if len(cpi_yoy) >= 2 else None
        if latest_cpi is not None:
            us_infl['CPI'] = {'actual': latest_cpi, 'previous': prev_cpi}
    # PPI from FRED
    if fred_data and fred_data.get('ppi', {}).get('yoy'):
        ppi_yoy = fred_data['ppi']['yoy']
        latest_ppi = list(ppi_yoy.values())[-1] if ppi_yoy else None
        prev_ppi = list(ppi_yoy.values())[-2] if len(ppi_yoy) >= 2 else None
        if latest_ppi is not None:
            us_infl['PPI'] = {'actual': latest_ppi, 'previous': prev_ppi}
    # PCE from M45
    pce_d = _mod_data(result.get('m45'))
    if pce_d:
        us_infl['PCE'] = pce_d
    if us_infl:
        rdm['US_INFLATION'] = us_infl

    # ── US_LABOR cascade ──
    us_labor = {}
    adp_d = _mod_data(result.get('m36'))
    if adp_d:
        us_labor['ADP'] = adp_d
    # Claims from FRED
    if fred_data and fred_data.get('icsa', {}).get('monthly_avg'):
        claims = fred_data['icsa']['monthly_avg']
        latest_claims = list(claims.values())[-1] if claims else None
        if latest_claims is not None:
            us_labor['CLAIMS'] = {'actual': latest_claims}
    if us_labor:
        rdm['US_LABOR'] = us_labor

    # ── US_ACTIVITY cascade ──
    us_act = {}
    ism_d = _mod_data(result.get('m27') if 'm27' in result else None)
    if ism_d:
        us_act['ISM_MFG'] = ism_d
    ism_svc_d = _mod_data(result.get('m28') if 'm28' in result else None)
    if ism_svc_d:
        us_act['ISM_SVC'] = ism_svc_d
    retail_d = _mod_data(result.get('m33'))
    if retail_d:
        us_act['RETAIL_SALES'] = retail_d
    durables_d = _mod_data(result.get('m44'))
    if durables_d:
        us_act['DURABLES'] = durables_d
    gdp_d = _mod_data(result.get('m43'))
    if gdp_d:
        us_act['US_GDP'] = gdp_d
    if us_act:
        rdm['US_ACTIVITY'] = us_act

    # ── CHINA_MACRO cascade ──
    cn = {}
    nbs_d = _mod_data(result.get('m24'))
    if nbs_d:
        cn['NBS_PMI'] = nbs_d
    caixin_d = _mod_data(result.get('m25'))
    if caixin_d:
        cn['CAIXIN_PMI'] = caixin_d
    cn_cpi_d = _mod_data(result.get('m30') if 'm30' in result else None)
    if cn_cpi_d:
        cn['CHINA_CPI_PPI'] = cn_cpi_d
    cn_act_d = _mod_data(result.get('m55') if 'm55' in result else None)
    if cn_act_d:
        cn['CHINA_ACTIVITY'] = cn_act_d
    pboc_d = _mod_data(result.get('m35'))
    if pboc_d:
        cn['PBOC_LPR'] = pboc_d
    cn_gdp_d = _mod_data(result.get('m54'))
    if cn_gdp_d:
        cn['GDP'] = cn_gdp_d
    if cn:
        rdm['CHINA_MACRO'] = cn

    # ── EU_MACRO cascade ──
    eu = {}
    ez_pmi_d = _mod_data(result.get('m26'))
    if ez_pmi_d:
        eu['EZ_PMI'] = ez_pmi_d
    de_cpi_d = _mod_data(result.get('m40'))
    if de_cpi_d:
        eu['GERMANY_CPI'] = de_cpi_d
    ez_cpi_d = _mod_data(result.get('m41'))
    if ez_cpi_d:
        eu['EZ_CPI'] = ez_cpi_d
    ez_gdp_d = _mod_data(result.get('m42'))
    if ez_gdp_d:
        eu['EZ_GDP'] = ez_gdp_d
    ifo_d = _mod_data(result.get('m38'))
    if ifo_d:
        eu['IFO'] = ifo_d
    ecb_d = _mod_data(result.get('m48'))
    if ecb_d:
        eu['ECB'] = ecb_d
    if eu:
        rdm['EU_MACRO'] = eu

    # ── UK_MACRO cascade ──
    uk = {}
    uk_cpi_d = _mod_data(result.get('m31') if 'm31' in result else None)
    if uk_cpi_d:
        uk['UK_CPI'] = uk_cpi_d
    uk_wages_d = _mod_data(result.get('m32') if 'm32' in result else None)
    if uk_wages_d:
        uk['UK_WAGES'] = uk_wages_d
    uk_gdp_d = _mod_data(result.get('m51'))
    if uk_gdp_d:
        uk['UK_GDP'] = uk_gdp_d
    boe_d = _mod_data(result.get('m49'))
    if boe_d:
        uk['BOE'] = boe_d
    if uk:
        rdm['UK_MACRO'] = uk

    # ── JAPAN_MACRO cascade ──
    jp = {}
    jp_cpi_d = _mod_data(result.get('m46'))
    if jp_cpi_d:
        jp['JAPAN_CPI'] = jp_cpi_d
    jp_gdp_d = _mod_data(result.get('m55') if 'm55' in result else None)
    if jp_gdp_d:
        jp['JAPAN_GDP'] = jp_gdp_d
    boj_d = _mod_data(result.get('m47'))
    if boj_d:
        jp['BOJ'] = boj_d
    if jp:
        rdm['JAPAN_MACRO'] = jp

    # ── AU_MACRO cascade ──
    au = {}
    au_cpi_d = _mod_data(result.get('m53'))
    if au_cpi_d:
        au['AU_CPI'] = au_cpi_d
    rba_d = _mod_data(result.get('m52'))
    if rba_d:
        au['RBA'] = rba_d
    if au:
        rdm['AU_MACRO'] = au

    return rdm


def scan_signal(df_15m, df_1h, df_2h, df_4h, df_1d, config=None,
                btc_15m_df=None, btc_corr_series=None):
    """Scan current market for trading signals.

    Uses the same pipeline as the backtest engine:
      Phase 1: M9 (regime) → market climate
      Phase 2: M13 (structure) → swing direction
      Phase 2: M7 (macro) → ETH/BTC + BTC
      Phase 3: resolve_direction() → unified direction
      Phase 4: Score all modules (M1–M14) for ICS
      Phase 5: Veto + Coherence + Entry filters
    """
    cfg = config or CONFIG
    idx = len(df_15m) - 1
    row = df_15m.iloc[idx]
    ts = row['Open time']
    idx_1h = len(df_1h) - 1
    idx_2h = len(df_2h) - 1
    idx_4h = len(df_4h) - 1
    idx_1d = len(df_1d) - 1
    atr_1h = df_1h['atr'].iloc[idx_1h]
    swing_bias = df_1d['swing_bias'].iloc[idx_1d]
    phase0_val = df_1h['phase0'].iloc[idx_1h]
    trend_dir = df_1d['trend'].iloc[idx_1d]
    trend_val = df_1d['trend_score'].iloc[idx_1d]

    result = {
        'timestamp': str(ts), 'price': float(row['Close']),
        'swing_bias': swing_bias, 'phase0': float(phase0_val) if not pd.isna(phase0_val) else None,
        'trend_dir': trend_dir, 'trend_val': float(trend_val),
    }

    # Get current UTC time for use in multiple modules
    from datetime import datetime, timezone
    _now_utc = datetime.now(timezone.utc)

    # ── Prepare 15m data for M9/M13 (they expect 15m bars, not base TF) ──
    base_tf = cfg.get('_base_timeframe', '15m')
    if base_tf == '15m':
        df_15m_for_m9 = df_15m
        idx_15m_for_m9 = idx
    elif base_tf == '1h':
        # 1h data — M13 uses it directly as df_1h, pass same as df_15m (FVGs/OBs on 1h bars)
        df_15m_for_m9 = df_15m
        idx_15m_for_m9 = idx
    else:
        # 5m/1m — resample to 15m for M9/M13
        df_tmp = df_15m.copy().set_index('Open time')
        agg = {
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            'Volume': 'sum', 'Quote asset volume': 'sum',
            'Number of trades': 'sum',
            'Taker buy base asset volume': 'sum',
            'Taker buy quote asset volume': 'sum',
        }
        df_15m_for_m9 = df_tmp.resample('15min').agg(agg).dropna(subset=['Open']).reset_index()
        df_15m_for_m9['atr'] = calc_atr(df_15m_for_m9['High'], df_15m_for_m9['Low'], df_15m_for_m9['Close'], cfg['ATR_PERIOD'])
        idx_15m_for_m9 = len(df_15m_for_m9) - 1

    # ── Phase 1: M9 Volatility Regime ──
    regime_state = RegimeState(config=cfg)
    vol_regime, m9_raw, m9_vol_details = compute_vol_regime(
        df_15m_for_m9, df_1h, idx_15m_for_m9, idx_1h, regime_state=regime_state, config=cfg)
    result['m9'] = {'regime': vol_regime, 'raw': round(float(m9_raw), 3) if m9_raw else None}

    # Regime block — record but don't early-return (all modules still score)
    block_regimes = cfg.get('M9_BLOCK_REGIMES', ['CRISIS'])
    regime_blocked = vol_regime in block_regimes
    if regime_blocked:
        result['regime_blocked'] = True

    # ── Phase 2: M13 Structural Bias ──
    m13_status, m13_score_raw, m13_details = score_m13(df_1h, idx_1h, 'NEUTRAL', df_15m_for_m9, idx_15m_for_m9)
    m13_bias = m13_details.get('m13_bias', 'NEUTRAL')
    result['m13'] = {'bias': m13_bias, 'score': round(float(m13_score_raw), 3), 'status': m13_status}

    # ── Phase 2: M7 Macro (ETH/BTC + BTC) ──
    m7_ethbtc_df, m7_btc_df = None, None
    m7_score = 0.5
    m7_status = 'SKIP'
    m7_details = {}
    if cfg.get('M7_ENABLED', False):
        try:
            m7_ethbtc_df, m7_btc_df = m7_prepare_data(df_15m)
            eb_row, bt_row = m7_get_row(m7_ethbtc_df, m7_btc_df, ts)
            m7_status, m7_score, m7_details = score_m7(eb_row, bt_row, row.get('vol_ratio', np.nan), 'NEUTRAL')
            result['m7'] = {'score': round(float(m7_score), 3), 'status': m7_status}
        except Exception as e:
            result['m7'] = {'score': 0.5, 'status': 'SKIP', 'error': str(e)}

    # ── Phase 2d: Pre-compute targets for direction resolver ──
    # Volume profile + S/R computed early so targets can inform direction
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

    atr_1h_val = df_1h['atr'].iloc[idx_1h] if idx_1h >= 0 else None
    current_price = float(row['Close'])
    long_tgt_score, long_tgt_details = score_targets(
        current_price, magnets, gaps, sr_levels, 'LONG', atr_1h=atr_1h_val)
    short_tgt_score, short_tgt_details = score_targets(
        current_price, magnets, gaps, sr_levels, 'SHORT', atr_1h=atr_1h_val)

    # ── Phase 3: Resolve Direction (now with target awareness) ──
    # Compute nearest_liq_direction from unswept magnets
    nearest_liq_dir = None
    if magnets:
        above = [(p, s) for p, v, s in magnets if p > float(row['Close'])]
        below = [(p, s) for p, v, s in magnets if p < float(row['Close'])]
        if above and below:
            nearest_above_dist = min(above, key=lambda x: x[0] - float(row['Close']))
            nearest_below_dist = min(below, key=lambda x: float(row['Close']) - x[0])
            above_dist = nearest_above_dist[0] - float(row['Close'])
            below_dist = float(row['Close']) - nearest_below_dist[0]
            if above_dist < below_dist * 0.7:
                nearest_liq_dir = 'LONG'  # closer liquidity above → go long to grab it
            elif below_dist < above_dist * 0.7:
                nearest_liq_dir = 'SHORT'  # closer liquidity below → go short to grab it

    # ── Phase 2e: M20 Pre-compute (before direction resolver) ──
    # M20 detects failed breakouts independently. We run it early so its
    # contrarian direction can feed into the direction resolver as an override
    # when a strong failed breakout is detected.
    _m20_pre_score = None
    _m20_pre_dir = None
    if cfg.get('M20_ENABLED', True):
        try:
            _sr_for_m20 = sr_levels if sr_levels else None
            _mag_for_m20 = magnets if magnets else None
            _m20_pre_status, _m20_pre_score, _m20_pre_result = score_m20(
                df_15m, idx, 'NEUTRAL',
                sr_levels=_sr_for_m20, magnets=_mag_for_m20,
                config=cfg, atr_1h=atr_1h)
            if _m20_pre_result and _m20_pre_result.get('status') == 'FAILED':
                _m20_pre_dir = _m20_pre_result.get('contrarian_direction')
                if _m20_pre_dir:
                    print(f"  💥 M20 pre-compute: failed breakout → {_m20_pre_dir} (score={_m20_pre_score:.3f})")
        except Exception:
            pass

    _rsi_val = float(df_15m['rsi'].iloc[idx]) if 'rsi' in df_15m.columns and not pd.isna(df_15m['rsi'].iloc[idx]) else None

    direction, dir_size_mult, dir_details = resolve_direction(
        vol_regime, m9_raw if m9_raw else 0.5,
        m13_bias, m13_score_raw, m13_details,
        m7_score=m7_score, m7_status=m7_status,
        swing_bias_1d=swing_bias, trend_dir=trend_dir, config=cfg,
        long_target_score=long_tgt_score, short_target_score=short_tgt_score,
        long_target_details=long_tgt_details, short_target_details=short_tgt_details,
        nearest_liq_direction=nearest_liq_dir,
        m20_score=_m20_pre_score, m20_direction=_m20_pre_dir,
        rsi_value=_rsi_val,
    )
    # ── Gather market data (always, regardless of signal status) ──
    swept_magnets = _check_swept_magnets(df_15m, idx, magnets[:5], config=cfg)
    result['magnets'] = swept_magnets
    result['gaps'] = [round(p, 2) for p, _ in gaps[:5]]

    # Target scores from direction resolver
    result['target_scores'] = {
        'LONG': round(long_tgt_score, 3), 'SHORT': round(short_tgt_score, 3),
        'long_details': long_tgt_details, 'short_details': short_tgt_details,
    }

    sr_levels.sort(key=lambda x: x[1], reverse=True)
    result['sr_levels'] = [(round(p, 2), round(s, 2), t, touches, bounces)
                           for p, s, touches, bounces, t in sr_levels[:8]]

    try:
        deriv_summary = get_derivatives_summary()
        if 'error' not in deriv_summary:
            result['derivatives'] = deriv_summary
    except Exception:
        pass

    result['cascade_risk'] = _detect_cascade_risk(df_15m, idx, result)

    # ── Intrabar CVD (compute early — needed by squeeze confirmation gate) ──
    m4b_status = 'SKIP'
    m4b_score = 0.5
    m4b_details = {}
    m4b_divergence = 'NONE'
    _intrabar_result_early = None
    if cfg.get('M4B_INTRABAR_ENABLED', True):
        try:
            _tf_map = {'15m': '15min', '1h': '1h', '5m': '5min', '1m': '1min'}
            _target_tf = _tf_map.get(cfg.get('_base_timeframe', '15m'), '15min')
            _hours = cfg.get('M4B_INTRABAR_HOURS', 48)
            _intrabar_df, _intrabar_result_early = get_intrabar_cvd_summary(
                symbol='ETHUSDT', target_tf=_target_tf, hours=_hours)
            if _intrabar_result_early and 'error' not in _intrabar_result_early:
                m4b_status, m4b_score, m4b_details = score_intrabar_divergence(
                    _intrabar_result_early, direction)
                m4b_divergence = _intrabar_result_early.get('divergence', 'NONE')
                result['m4b'] = {
                    'status': m4b_status,
                    'score': round(float(m4b_score), 3),
                    'divergence': m4b_divergence,
                    'bars_ago': _intrabar_result_early.get('bars_ago', -1),
                    'cvd_slope': round(_intrabar_result_early.get('cvd_slope_12', 0), 2),
                    'details': m4b_details,
                }
        except Exception as e:
            result['m4b'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # ── Liquidity levels (compute early — needed by squeeze TP/SL) ──
    _liq_for_squeeze = None
    _liq_direction = direction if direction != 'NEUTRAL' else None
    if _liq_direction:
        try:
            oi_usd = result.get('derivatives', {}).get('oi_usd', 0)
            ls_ratio = result.get('derivatives', {}).get('ls_ratio', 1.0)
            _liq_for_squeeze = get_liquidity_summary(
                df_15m, idx, sr_levels, oi_usd, ls_ratio, _liq_direction)
            result['liquidity_levels'] = _liq_for_squeeze
        except Exception:
            pass

    # ── M18 Squeeze Detection (after derivatives data is available) ──
    result['rsi'] = float(df_15m['rsi'].iloc[idx]) if 'rsi' in df_15m.columns else 50
    result['vol_trend'] = float(df_15m['Volume'].iloc[idx] / df_15m['vol_ma20'].iloc[idx]) if 'vol_ma20' in df_15m.columns else 1.0
    result['atr'] = float(df_15m['atr'].iloc[idx]) if 'atr' in df_15m.columns else 0

    # Squeeze quality: compute from raw market features
    roll_high_48 = float(df_15m['High'].iloc[max(0,idx-47):idx+1].max())
    roll_low_48 = float(df_15m['Low'].iloc[max(0,idx-47):idx+1].min())
    range_width = (roll_high_48 - roll_low_48) / float(df_15m['Close'].iloc[idx]) * 100

    vol_ratio_val = float(df_15m['vol_ratio'].iloc[idx]) if 'vol_ratio' in df_15m.columns and not pd.isna(df_15m['vol_ratio'].iloc[idx]) else 0.15
    taker_base = float(df_15m['Taker buy base asset volume'].iloc[idx]) if 'Taker buy base asset volume' in df_15m.columns else 0
    total_vol = float(df_15m['Volume'].iloc[idx])
    taker_ratio = taker_base / total_vol if total_vol > 0 else 0.5

    # VWAP distance
    vwap_val = float(df_15m['vwap'].iloc[idx]) if 'vwap' in df_15m.columns else float(df_15m['Close'].iloc[idx])
    vwap_dist = (float(df_15m['Close'].iloc[idx]) - vwap_val) / vwap_val * 100 if vwap_val > 0 else 0

    # OI proxy: volume accumulation relative to average
    vol_cumsum_48 = float(df_15m['Volume'].iloc[max(0,idx-47):idx+1].sum())
    vol_cumsum_ma = float(df_15m['Volume'].iloc[max(0,idx-67):idx+1].rolling(20).mean().iloc[-1]) if idx > 67 else vol_cumsum_48
    oi_proxy = vol_cumsum_48 / vol_cumsum_ma if vol_cumsum_ma > 0 else 1.0

    # Bar-level ignition
    bar_vol_spike = float(df_15m['Volume'].iloc[idx] / df_15m['vol_ma20'].iloc[idx]) if 'vol_ma20' in df_15m.columns else 1.0
    bar_range = (float(df_15m['High'].iloc[idx]) - float(df_15m['Low'].iloc[idx])) / float(df_15m['Close'].iloc[idx]) * 100
    bar_range_ma = float(df_15m['High'].iloc[max(0,idx-19):idx+1].sub(df_15m['Low'].iloc[max(0,idx-19):idx+1]).div(df_15m['Close'].iloc[max(0,idx-19):idx+1]).mean() * 100) if idx > 19 else bar_range
    bar_range_expansion = bar_range / bar_range_ma if bar_range_ma > 0 else 1.0
    bar_taker_extreme = taker_ratio > 0.65 or taker_ratio < 0.35

    # Quality percentile (use rolling rank)
    # For now, use a simplified quality score
    # Lower range_width, lower vol_ratio, higher oi_proxy, closer to VWAP = better
    result['range_width'] = range_width
    result['vol_ratio'] = vol_ratio_val
    result['vol_ma20'] = float(df_15m['vol_ma20'].iloc[idx]) if 'vol_ma20' in df_15m.columns else 0
    result['oi_proxy'] = oi_proxy
    result['vwap_dist'] = vwap_dist
    result['bar_vol_spike'] = bar_vol_spike
    result['bar_range_expansion'] = bar_range_expansion
    result['bar_taker_extreme'] = bar_taker_extreme

    # Simplified quality: normalize each factor to 0-1 and combine
    # Using rough thresholds from backtest analysis
    rw_score = max(0, min(1, 1 - (range_width - 1.5) / 4.0))  # 1.5% = best, 5.5% = worst
    vr_score = max(0, min(1, 1 - (vol_ratio_val - 0.05) / 0.20))  # 0.05 = best, 0.25 = worst
    oip_score = max(0, min(1, (oi_proxy - 0.7) / 0.5))  # 0.7 = worst, 1.2 = best
    vd_score = max(0, min(1, 1 - abs(vwap_dist) / 1.0))  # 0 = best, 1.0% = worst

    result['squeeze_quality'] = (rw_score * 0.30 + vr_score * 0.25 +
                                  oip_score * 0.25 + vd_score * 0.20)

    # Build compression history for squeeze detector (last 48 bars)
    compression_history = []
    if idx >= 48:
        for i in range(max(0, idx - 47), idx):
            r48_h = float(df_15m['High'].iloc[max(0, i-47):i+1].max())
            r48_l = float(df_15m['Low'].iloc[max(0, i-47):i+1].min())
            r48_pct = (r48_h - r48_l) / float(df_15m['Close'].iloc[i]) * 100 if float(df_15m['Close'].iloc[i]) > 0 else 5.0
            vr = float(df_15m['Volume'].iloc[i] / df_15m['vol_ma20'].iloc[i]) if 'vol_ma20' in df_15m.columns and float(df_15m['vol_ma20'].iloc[i]) > 0 else 1.0
            br = (float(df_15m['High'].iloc[i]) - float(df_15m['Low'].iloc[i])) / float(df_15m['Close'].iloc[i]) * 100 if float(df_15m['Close'].iloc[i]) > 0 else 0.5
            tr_val = float(df_15m['Taker buy base asset volume'].iloc[i]) / float(df_15m['Volume'].iloc[i]) if float(df_15m['Volume'].iloc[i]) > 0 else 0.5
            compression_history.append((r48_pct, vr, br, tr_val))

    # Pass raw taker_ratio and bar_range to result for squeeze detector
    result['raw_taker_ratio'] = taker_ratio
    result['raw_bar_range_pct'] = bar_range

    # Taker flow analysis
    try:
        result['taker_summary'] = get_taker_summary(df_15m, idx)
    except Exception:
        result['taker_summary'] = None

    squeeze_result = detect_squeeze(result, config=cfg,
                                     last_signal_bar=result.get('_last_squeeze_bar', -1),
                                     current_bar=idx,
                                     compression_history=compression_history,
                                     df_15m=df_15m,
                                     magnets=magnets,
                                     sr_levels=sr_levels,
                                     liq_levels=_liq_for_squeeze)
    result['squeeze'] = squeeze_result
    if squeeze_result['squeeze_status'] == 'TRIGGERED':
        result['_last_squeeze_bar'] = idx

    # Exchange Activity (cross-exchange funding, OI, L/S)
    try:
        exchange_summary = get_exchange_summary()
        if 'error' not in exchange_summary:
            result['exchange_activity'] = exchange_summary
            # Score derivatives exchange data with resolved direction
            _ex_signals = exchange_summary.get('signals', {})
            if _ex_signals:
                _ex_status, _ex_score, _ex_details = score_exchange_activity(
                    _ex_signals, direction if direction != 'NEUTRAL' else 'LONG')
                result['exchange_activity']['score'] = round(_ex_score, 3)
                result['exchange_activity']['status'] = _ex_status
                result['exchange_activity']['direction_details'] = _ex_details
            # Score spot data
            _spot_signals = exchange_summary.get('spot_signals', {})
            if _spot_signals:
                _sp_status, _sp_score, _sp_details = score_spot_signals(
                    _spot_signals, direction if direction != 'NEUTRAL' else 'LONG')
                result['exchange_activity']['spot_score'] = round(_sp_score, 3)
                result['exchange_activity']['spot_status'] = _sp_status
                result['exchange_activity']['spot_details'] = _sp_details
    except Exception:
        pass

    # ── M19: Breakout Confirmation Filters ──
    # Only run when a squeeze is detected (PENDING or TRIGGERED)
    breakout_result = None
    sq_type = squeeze_result.get('squeeze_type', 'NONE')
    if sq_type != 'NONE' and squeeze_result.get('direction', 'NEUTRAL') != 'NEUTRAL':
        breakout_result = check_breakout_filters(result, df_15m=df_15m, config=cfg)
        result['breakout_confirm'] = breakout_result

        # If breakout is REJECTED and squeeze was going to trigger,
        # downgrade to PENDING (wait for filters to pass)
        if breakout_result['status'] == 'REJECTED':
            if squeeze_result['squeeze_status'] == 'TRIGGERED':
                # Don't let a rejected breakout trigger a signal
                squeeze_confirmed = False
                result['squeeze_confirmed'] = False
                result['breakout_rejected'] = True
                print(f"  ⚠️  Breakout REJECTED ({breakout_result['passed']}/{breakout_result['total']}) "
                      f"— squeeze trigger suppressed")

        elif breakout_result['status'] == 'WEAK':
            # Reduce ICS boost by half for weak breakouts
            if squeeze_result.get('ics_boost', 0) > 0:
                squeeze_result['ics_boost'] *= 0.5
                result['squeeze_ics_boost_half'] = True

    # When regime blocks direction, use squeeze direction if available, else force LONG for display
    if direction == 'NEUTRAL' and result.get('regime_blocked'):
        sq_dir = squeeze_result.get('direction', 'NEUTRAL')
        if sq_dir != 'NEUTRAL':
            direction = sq_dir
            dir_details['reason'] = f"Squeeze {squeeze_result['squeeze_type']} override → {direction}"
        else:
            direction = 'LONG'
            dir_details['reason'] = f"regime={vol_regime} blocked — scoring LONG for display"

    result['direction'] = direction
    result['direction_resolver'] = {
        'direction': direction, 'size_mult': round(float(dir_size_mult), 3),
        'action': dir_details.get('action', '?'),
        'reason': dir_details.get('reason', '?'),
    }

    # Conflict check (liq summary already computed above before squeeze)
    _liq_direction = direction if direction != 'NEUTRAL' else None
    if _liq_direction and swing_bias:
        try:
            conflict = get_conflict_stats(
                'BULLISH' if _liq_direction == 'LONG' else 'BEARISH',
                swing_bias)
            if conflict:
                result['conflict'] = conflict
        except Exception:
            pass

    # Re-score M9, M7, M13 with actual direction
    m9_status, m9_score, m9_details = score_vol_regime(vol_regime, m9_raw, direction, trend_dir)
    if cfg.get('M7_ENABLED', False) and m7_ethbtc_df is not None:
        eb_row, bt_row = m7_get_row(m7_ethbtc_df, m7_btc_df, ts)
        m7_status, m7_score, m7_details = score_m7(eb_row, bt_row, row.get('vol_ratio', np.nan), direction)
    _chop_regimes_m13 = ('CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL', 'CHOP_HARD')
    _in_chop = cfg.get('M13_DEFER_IN_CHOP', True) and vol_regime in _chop_regimes_m13
    if cfg.get('M13_ENABLED', True):
        _m13_status2, _m13_score2, m13_details = score_m13(df_1h, idx_1h, direction, df_15m_for_m9, idx_15m_for_m9)
        if not _in_chop:
            m13_status = _m13_status2
            m13_score = _m13_score2
            m13_bias = m13_details.get('m13_bias', 'NEUTRAL')
        # During chop: keep m13_score at 0.5 for ICS, but m13_details
        # is updated with direction-aware swing levels for M14
    result['m9']['score'] = round(float(m9_score), 3)
    result['m9']['status'] = m9_status
    result['m7']['score'] = round(float(m7_score), 3)
    result['m13']['score'] = round(float(m13_score), 3)

    # ── M21: Wyckoff Phase + Premium/Discount + Kill Zone ──
    m21_status = 'SKIP'
    m21_score = 0.5
    m21_details = {}
    range_info_m21 = None
    if cfg.get('M21_ENABLED', True):
        try:
            m21_status, m21_score, m21_details = score_m21(
                df_15m, df_1h, df_4h, df_1d, idx, direction, config=cfg)
            range_info_m21 = m21_details.get('range_info')
            result['m21'] = {
                'status': m21_status,
                'score': round(float(m21_score), 3),
                'phase': m21_details.get('phase', 'RANGE'),
                'zone': m21_details.get('premium_discount', {}).get('zone', 'UNKNOWN'),
                'kill_zone': m21_details.get('kill_zone', {}).get('session', 'ANY'),
                'spring_upthrust': m21_details.get('spring_upthrust', {}).get('type', 'NONE'),
                'details': m21_details,
            }
            # When M21 blocks a direction, try flipping to the opposite side
            # instead of just blocking. ACCUMULATION blocking SHORT → try LONG.
            # DISTRIBUTION blocking LONG → try SHORT.
            if m21_status == 'BLOCKED':
                flip_dir = 'LONG' if direction == 'SHORT' else 'SHORT'
                try:
                    m21_flip_status, m21_flip_score, m21_flip_details = score_m21(
                        df_15m, df_1h, df_4h, df_1d, idx, flip_dir, config=cfg)
                    if m21_flip_status != 'BLOCKED':
                        # Flipped direction is viable — use it
                        direction = flip_dir
                        m21_status = m21_flip_status
                        m21_score = m21_flip_score
                        m21_details = m21_flip_details
                        range_info_m21 = m21_details.get('range_info')
                        result['m21'] = {
                            'status': m21_status,
                            'score': round(float(m21_score), 3),
                            'phase': m21_details.get('phase', 'RANGE'),
                            'zone': m21_details.get('premium_discount', {}).get('zone', 'UNKNOWN'),
                            'kill_zone': m21_details.get('kill_zone', {}).get('session', 'ANY'),
                            'spring_upthrust': m21_details.get('spring_upthrust', {}).get('type', 'NONE'),
                            'details': m21_details,
                        }
                        result['m21_direction_flip'] = f'M21 flipped {flip_dir} → {direction}'
                        # Update the stored direction to reflect the flip
                        result['direction'] = direction
                        result['direction_resolver']['direction'] = direction
                        result['direction_resolver']['reason'] = (
                            f"{dir_details.get('reason', '?')} → M21 flip to {direction}"
                        )
                    else:
                        # Both directions blocked — no signal
                        result['status'] = 'NO_SIGNAL'
                        result['reason'] = f'M21 block: {", ".join(m21_details.get("reasons", []))}'
                        return result
                except Exception:
                    result['status'] = 'NO_SIGNAL'
                    result['reason'] = f'M21 block: {", ".join(m21_details.get("reasons", []))}'
                    return result
        except Exception as e:
            result['m21'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # ── M22: Macro Regime (placeholder — aggregated after M23-M65) ──
    # M22 now runs AFTER all macro modules to aggregate their outputs.
    # Placeholder values set here; real scoring happens after M62.
    m22_score = 0.5
    m22_status = 'SKIP'
    m22_details = {}

    # ── M23: NFP + PPI + CPI Session Analysis ──
    m23_score = 0.5
    m23_status = 'SKIP'
    m23_details = {}
    try:
        m23_status, m23_score, m23_details, m23_decay_mult = score_m23_ppi_session(
            df_15m, current_time=_now_utc, config=cfg)
        if m23_details and m23_details.get('regime') not in ('DISABLED', 'NO_PPI', 'NO_DATA'):
            result['m23'] = {
                'status': m23_status,
                'score': round(float(m23_score), 3),
                'regime': m23_details.get('regime', '?'),
                'release_type': m23_details.get('release_type', 'PPI'),
                'release_date': m23_details.get('release_date', m23_details.get('ppi_date')),
                'us_direction': m23_details.get('us_direction'),
                'us_magnitude': m23_details.get('us_magnitude'),
                'fade_rate': m23_details.get('fade_rate'),
                'spike_accuracy': m23_details.get('spike_accuracy'),
                'type_strength': m23_details.get('type_strength', 1.0),
                'regime_sensitivity': m23_details.get('regime_sensitivity', 0.80),
                'claims_context': m23_details.get('claims_context'),
                'claims_today': m23_details.get('claims_today', False),
                # NFP-specific fields
                'nfp_seasonality': m23_details.get('nfp_seasonality'),
                'nfp_asia_fade_rate': m23_details.get('nfp_asia_fade_rate'),
                'decay_mult': m23_decay_mult,
                'details': m23_details,
            }
            # Apply decay multiplier from macro release timing
            if m23_decay_mult < 1.0:
                result['_m23_decay_mult'] = m23_decay_mult
    except Exception as e:
        result['m23'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # ── M24: NBS PMI Session Bias (regime-conditional) ──
    m24_score_adj = 0.0
    m24_size_mult = 1.0
    m24_details = {}
    try:
        _wyckoff_for_m24 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m24 = result.get('m9', {}).get('regime', 'CHOP')
        m24_status, m24_score_adj, m24_size_mult, m24_details = score_m24_nbs_pmi(
            wyckoff_phase=_wyckoff_for_m24,
            vol_regime=_vol_for_m24,
            direction=direction,
            config=cfg)
        if m24_details and m24_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m24'] = {
                'status': m24_status,
                'score_adj': m24_score_adj,
                'size_mult': m24_size_mult,
                'regime': m24_details.get('regime', '?'),
                'bias': m24_details.get('bias', '?'),
                'mfg_pmi': m24_details.get('mfg_pmi'),
                'services_pmi': m24_details.get('services_pmi'),
                'pmi_signal': m24_details.get('pmi_signal'),
                'economic_regime': m24_details.get('economic_regime'),
                'avg_ret_24h': m24_details.get('avg_ret_24h'),
                'win_rate': m24_details.get('win_rate'),
                'sample_size': m24_details.get('sample_size'),
                'confidence': m24_details.get('confidence'),
                'source': m24_details.get('source'),
                'release_date': m24_details.get('release_date'),
                'details': m24_details,
            }
            # Apply M24 size multiplier
            if m24_size_mult < 1.0:
                result['_m24_size_mult'] = m24_size_mult
    except Exception as e:
        result['m24'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M25: Caixin PMI Session Bias (regime-conditional) ──
    m25_score_adj = 0.0
    m25_size_mult = 1.0
    m25_details = {}
    try:
        # Get Phase0 and 30d trend for Caixin filters
        _phase0_for_m25 = result.get('m21', {}).get('phase0')
        if _phase0_for_m25 is None:
            _phase0_for_m25 = result.get('phase0')
        _trend30_for_m25 = None
        _price_now = float(df_15m['Close'].iloc[-1]) if len(df_15m) > 2880 else None
        if _price_now and len(df_15m) > 2880:
            _price_30d = float(df_15m['Close'].iloc[-2880])
            _trend30_for_m25 = (_price_now - _price_30d) / _price_30d * 100

        # Check if Caixin PMI data is available (from macro_fetch cache)
        _caixin_surprise = None
        try:
            from src.utils.macro_fetch import get_surprise_for_event
            _caixin_surprise = get_surprise_for_event('cn_caixin_mfg_pmi')
            if _caixin_surprise == 'UNKNOWN':
                _caixin_surprise = None
        except Exception:
            pass

        m25_status, m25_score_adj, m25_size_mult, m25_details = score_m25_caixin_pmi(
            surprise=_caixin_surprise,
            phase0=_phase0_for_m25,
            trend_30d=_trend30_for_m25,
            direction=direction,
            config=cfg)
        if m25_details and m25_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_DATA'):
            result['m25'] = {
                'status': m25_status,
                'score_adj': m25_score_adj,
                'size_mult': m25_size_mult,
                'regime': m25_details.get('regime', '?'),
                'bias': m25_details.get('bias', '?'),
                'surprise': m25_details.get('surprise'),
                'caixin_actual': m25_details.get('caixin_actual'),
                'caixin_prev': m25_details.get('caixin_prev'),
                'avg_ret_24h': m25_details.get('avg_ret_24h'),
                'win_rate': m25_details.get('win_rate'),
                'sample_size': m25_details.get('sample_size'),
                'phase0_bucket': m25_details.get('phase0_bucket'),
                'trend_bucket': m25_details.get('trend_bucket'),
                'release_date': m25_details.get('release_date'),
                'details': m25_details,
            }
            # Apply M25 size multiplier
            if m25_size_mult < 1.0:
                result['_m25_size_mult'] = m25_size_mult
    except Exception as e:
        result['m25'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M26: Eurozone Flash PMI Session Bias (regime-conditional) ──
    m26_score_adj = 0.0
    m26_size_mult = 1.0
    m26_details = {}
    try:
        m26_status, m26_score_adj, m26_size_mult, m26_details = score_m26_ez_pmi(
            wyckoff_phase=_wyckoff_for_m24,  # reuse M24's wyckoff lookup
            vol_regime=_vol_for_m24,          # reuse M24's vol lookup
            direction=direction,
            config=cfg,
            now_utc=_now_utc)
        if m26_details and m26_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m26'] = {
                'status': m26_status,
                'score_adj': m26_score_adj,
                'size_mult': m26_size_mult,
                'regime': m26_details.get('regime', '?'),
                'bias': m26_details.get('bias', '?'),
                'actual': m26_details.get('actual'),
                'consensus': m26_details.get('consensus'),
                'surprise': m26_details.get('surprise'),
                'signal': m26_details.get('signal'),
                'avg_ret_entry_to_exit': m26_details.get('avg_ret_entry_to_exit'),
                'win_rate': m26_details.get('win_rate'),
                'sample_size': m26_details.get('sample_size'),
                'confidence': m26_details.get('confidence'),
                'session_phase': m26_details.get('session_phase'),
                'persistence_eu_uk': m26_details.get('persistence_eu_uk'),
                'persistence_uk_us': m26_details.get('persistence_uk_us'),
                'entry_session': m26_details.get('entry_session'),
                'exit_session': m26_details.get('exit_session'),
                'release_date': m26_details.get('release_date'),
                'details': m26_details,
            }
            # Apply M26 size multiplier
            if m26_size_mult < 1.0:
                result['_m26_size_mult'] = m26_size_mult
    except Exception as e:
        result['m26'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M33: US Retail Sales Session Bias (regime-conditional) ──
    m33_score_adj = 0.0
    m33_size_mult = 1.0
    m33_details = {}
    try:
        _wyckoff_for_m33 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m33 = result.get('m9', {}).get('regime', 'CHOP')
        m33_status, m33_score_adj, m33_size_mult, m33_details = score_m33_retail_sales(
            wyckoff_phase=_wyckoff_for_m33,
            vol_regime=_vol_for_m33,
            direction=direction,
            config=cfg)
        if m33_details and m33_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m33'] = {
                'status': m33_status,
                'score_adj': m33_score_adj,
                'size_mult': m33_size_mult,
                'regime': m33_details.get('regime', '?'),
                'bias': m33_details.get('bias', '?'),
                'retail_mom': m33_details.get('retail_mom'),
                'core_mom': m33_details.get('core_mom'),
                'consensus': m33_details.get('consensus'),
                'surprise': m33_details.get('surprise'),
                'signal': m33_details.get('signal'),
                'avg_ret_24h': m33_details.get('avg_ret_24h'),
                'win_rate': m33_details.get('win_rate'),
                'sample_size': m33_details.get('sample_size'),
                'confidence': m33_details.get('confidence'),
                'source': m33_details.get('source'),
                'release_date': m33_details.get('release_date'),
                'details': m33_details,
            }
            if m33_size_mult < 1.0:
                result['_m33_size_mult'] = m33_size_mult
    except Exception as e:
        result['m33'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M34: US Housing Starts Session Bias (regime-conditional) ──
    m34_score_adj = 0.0
    m34_size_mult = 1.0
    m34_details = {}
    try:
        _wyckoff_for_m34 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m34 = result.get('m9', {}).get('regime', 'CHOP')
        m34_status, m34_score_adj, m34_size_mult, m34_details = score_m34_housing(
            wyckoff_phase=_wyckoff_for_m34,
            vol_regime=_vol_for_m34,
            direction=direction,
            config=cfg)
        if m34_details and m34_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m34'] = {
                'status': m34_status,
                'score_adj': m34_score_adj,
                'size_mult': m34_size_mult,
                'regime': m34_details.get('regime', '?'),
                'bias': m34_details.get('bias', '?'),
                'starts_k': m34_details.get('starts_k'),
                'permits_k': m34_details.get('permits_k'),
                'starts_mom': m34_details.get('starts_mom'),
                'permits_mom': m34_details.get('permits_mom'),
                'signal': m34_details.get('signal'),
                'avg_ret_24h': m34_details.get('avg_ret_24h'),
                'win_rate': m34_details.get('win_rate'),
                'sample_size': m34_details.get('sample_size'),
                'confidence': m34_details.get('confidence'),
                'source': m34_details.get('source'),
                'release_date': m34_details.get('release_date'),
                'details': m34_details,
            }
            if m34_size_mult < 1.0:
                result['_m34_size_mult'] = m34_size_mult
    except Exception as e:
        result['m34'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M35: PBoC LPR Decision Session Bias (regime-conditional) ──
    m35_score_adj = 0.0
    m35_size_mult = 1.0
    m35_details = {}
    try:
        _wyckoff_for_m35 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m35 = result.get('m9', {}).get('regime', 'CHOP')
        m35_status, m35_score_adj, m35_size_mult, m35_details = score_m35_pboc_lpr(
            wyckoff_phase=_wyckoff_for_m35,
            vol_regime=_vol_for_m35,
            direction=direction,
            config=cfg)
        if m35_details and m35_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m35'] = {
                'status': m35_status,
                'score_adj': m35_score_adj,
                'size_mult': m35_size_mult,
                'regime': m35_details.get('regime', '?'),
                'bias': m35_details.get('bias', '?'),
                'lpr_1y': m35_details.get('lpr_1y'),
                'lpr_5y': m35_details.get('lpr_5y'),
                'cut_1y': m35_details.get('cut_1y'),
                'cut_5y': m35_details.get('cut_5y'),
                'signal': m35_details.get('signal'),
                'avg_ret_24h': m35_details.get('avg_ret_24h'),
                'win_rate': m35_details.get('win_rate'),
                'sample_size': m35_details.get('sample_size'),
                'confidence': m35_details.get('confidence'),
                'source': m35_details.get('source'),
                'release_date': m35_details.get('release_date'),
                'details': m35_details,
            }
            if m35_size_mult < 1.0:
                result['_m35_size_mult'] = m35_size_mult
    except Exception as e:
        result['m35'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M36: ADP Employment Report Session Bias (regime-conditional) ──
    m36_score_adj = 0.0
    m36_size_mult = 1.0
    m36_details = {}
    try:
        _wyckoff_for_m36 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m36 = result.get('m9', {}).get('regime', 'CHOP')
        m36_status, m36_score_adj, m36_size_mult, m36_details = score_m36_adp(
            wyckoff_phase=_wyckoff_for_m36,
            vol_regime=_vol_for_m36,
            direction=direction,
            config=cfg)
        if m36_details and m36_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m36'] = {
                'status': m36_status,
                'score_adj': m36_score_adj,
                'size_mult': m36_size_mult,
                'regime': m36_details.get('regime', '?'),
                'bias': m36_details.get('bias', '?'),
                'adp_k': m36_details.get('adp_k'),
                'consensus_k': m36_details.get('consensus_k'),
                'surprise': m36_details.get('surprise'),
                'signal': m36_details.get('signal'),
                'avg_ret_24h': m36_details.get('avg_ret_24h'),
                'win_rate': m36_details.get('win_rate'),
                'sample_size': m36_details.get('sample_size'),
                'confidence': m36_details.get('confidence'),
                'source': m36_details.get('source'),
                'release_date': m36_details.get('release_date'),
                'details': m36_details,
            }
            if m36_size_mult < 1.0:
                result['_m36_size_mult'] = m36_size_mult
    except Exception as e:
        result['m36'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── today_str: date string for M37-M62 macro modules ──
    today_str = _now_utc.strftime('%Y-%m-%d')

    # ── M37: US Non-Farm Payrolls Session Bias (regime-conditional) ──
    m37_score_adj = 0.0
    m37_size_mult = 1.0
    m37_status = 'SKIP'
    m37_details = {}
    try:
        _wyckoff_for_m37 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m37 = result.get('m9', {}).get('regime', 'CHOP')
        m37_status, m37_score_adj, m37_size_mult, m37_details = score_m37_nfp(
            wyckoff_phase=_wyckoff_for_m37,
            vol_regime=_vol_for_m37,
            direction=direction, today_str=today_str, config=cfg)
        if m37_details and m37_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m37'] = {
                'status': m37_status,
                'score_adj': m37_score_adj,
                'size_mult': m37_size_mult,
                'regime': m37_details.get('regime', '?'),
                'bias': m37_details.get('bias', '?'),
                'nfp_k': m37_details.get('nfp_k'),
                'consensus_k': m37_details.get('consensus_k'),
                'surprise': m37_details.get('surprise'),
                'signal': m37_details.get('signal'),
                'avg_ret_24h': m37_details.get('avg_ret_24h'),
                'win_rate': m37_details.get('win_rate'),
                'sample_size': m37_details.get('sample_size'),
                'confidence': m37_details.get('confidence'),
                'source': m37_details.get('source'),
                'release_date': m37_details.get('release_date'),
                'details': m37_details,
            }
            if m37_size_mult < 1.0:
                result['_m37_size_mult'] = m37_size_mult
    except Exception as e:
        result['m37'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M38: Germany Ifo Business Climate Session Bias (regime-conditional) ──
    m38_score_adj = 0.0
    m38_size_mult = 1.0
    m38_status = 'SKIP'
    m38_details = {}
    try:
        _wyckoff_for_m38 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m38 = result.get('m9', {}).get('regime', 'CHOP')
        m38_status, m38_score_adj, m38_size_mult, m38_details = score_m38_ifo(
            wyckoff_phase=_wyckoff_for_m38,
            vol_regime=_vol_for_m38,
            direction=direction, today_str=today_str, config=cfg)
        if m38_details and m38_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m38'] = {
                'status': m38_status,
                'score_adj': m38_score_adj,
                'size_mult': m38_size_mult,
                'regime': m38_details.get('regime', '?'),
                'bias': m38_details.get('bias', '?'),
                'actual': m38_details.get('actual'),
                'consensus': m38_details.get('consensus'),
                'surprise': m38_details.get('surprise'),
                'signal': m38_details.get('signal'),
                'trend': m38_details.get('trend'),
                'avg_ret_24h': m38_details.get('avg_ret_24h'),
                'win_rate': m38_details.get('win_rate'),
                'sample_size': m38_details.get('sample_size'),
                'confidence': m38_details.get('confidence'),
                'source': m38_details.get('source'),
                'release_date': m38_details.get('release_date'),
                'details': m38_details,
            }
            if m38_size_mult < 1.0:
                result['_m38_size_mult'] = m38_size_mult
    except Exception as e:
        result['m38'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M39: Michigan Consumer Sentiment Session Bias (regime-conditional) ──
    m39_score_adj = 0.0
    m39_size_mult = 1.0
    m39_status = 'SKIP'
    m39_details = {}
    try:
        _wyckoff_for_m39 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m39 = result.get('m9', {}).get('regime', 'CHOP')
        m39_status, m39_score_adj, m39_size_mult, m39_details = score_m39_ums(
            wyckoff_phase=_wyckoff_for_m39,
            vol_regime=_vol_for_m39,
            direction=direction, today_str=today_str, config=cfg)
        if m39_details and m39_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m39'] = {
                'status': m39_status,
                'score_adj': m39_score_adj,
                'size_mult': m39_size_mult,
                'regime': m39_details.get('regime', '?'),
                'bias': m39_details.get('bias', '?'),
                'headline': m39_details.get('headline'),
                'consensus': m39_details.get('consensus'),
                'surprise': m39_details.get('surprise'),
                'signal': m39_details.get('signal'),
                'infl_5yr': m39_details.get('infl_5yr'),
                'infl_signal': m39_details.get('infl_signal'),
                'type': m39_details.get('type'),
                'avg_ret_24h': m39_details.get('avg_ret_24h'),
                'win_rate': m39_details.get('win_rate'),
                'sample_size': m39_details.get('sample_size'),
                'confidence': m39_details.get('confidence'),
                'source': m39_details.get('source'),
                'release_date': m39_details.get('release_date'),
                'details': m39_details,
            }
            if m39_size_mult < 1.0:
                result['_m39_size_mult'] = m39_size_mult
    except Exception as e:
        result['m39'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M40: Germany CPI Session Bias (regime-conditional) ──
    m40_score_adj = 0.0
    m40_size_mult = 1.0
    m40_status = 'SKIP'
    m40_details = {}
    try:
        _wyckoff_for_m40 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m40 = result.get('m9', {}).get('regime', 'CHOP')
        m40_status, m40_score_adj, m40_size_mult, m40_details = score_m40_germany_cpi(
            wyckoff_phase=_wyckoff_for_m40,
            vol_regime=_vol_for_m40,
            direction=direction, today_str=today_str, config=cfg)
        if m40_details and m40_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m40'] = {
                'status': m40_status,
                'score_adj': m40_score_adj,
                'size_mult': m40_size_mult,
                'regime': m40_details.get('regime', '?'),
                'bias': m40_details.get('bias', '?'),
                'cpi_yoy': m40_details.get('cpi_yoy'),
                'consensus': m40_details.get('consensus'),
                'surprise': m40_details.get('surprise'),
                'signal': m40_details.get('signal'),
                'cpi_level': m40_details.get('cpi_level'),
                'cpi_direction': m40_details.get('cpi_direction'),
                'avg_ret_24h': m40_details.get('avg_ret_24h'),
                'win_rate': m40_details.get('win_rate'),
                'sample_size': m40_details.get('sample_size'),
                'confidence': m40_details.get('confidence'),
                'source': m40_details.get('source'),
                'release_date': m40_details.get('release_date'),
                'details': m40_details,
            }
            if m40_size_mult < 1.0:
                result['_m40_size_mult'] = m40_size_mult
    except Exception as e:
        result['m40'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M41: Eurozone CPI Flash Session Bias (regime-conditional) ──
    m41_score_adj = 0.0
    m41_size_mult = 1.0
    m41_status = 'SKIP'
    m41_details = {}
    try:
        _wyckoff_for_m41 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m41 = result.get('m9', {}).get('regime', 'CHOP')
        m41_status, m41_score_adj, m41_size_mult, m41_details = score_m41_ez_cpi(
            wyckoff_phase=_wyckoff_for_m41,
            vol_regime=_vol_for_m41,
            direction=direction, today_str=today_str, config=cfg)
        if m41_details and m41_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m41'] = {
                'status': m41_status,
                'score_adj': m41_score_adj,
                'size_mult': m41_size_mult,
                'regime': m41_details.get('regime', '?'),
                'bias': m41_details.get('bias', '?'),
                'hicp_yoy': m41_details.get('hicp_yoy'),
                'core_yoy': m41_details.get('core_yoy'),
                'consensus': m41_details.get('consensus'),
                'signal': m41_details.get('signal'),
                'cpi_level': m41_details.get('cpi_level'),
                'cpi_direction': m41_details.get('cpi_direction'),
                'core_spread': m41_details.get('core_spread'),
                'avg_ret_24h': m41_details.get('avg_ret_24h'),
                'win_rate': m41_details.get('win_rate'),
                'sample_size': m41_details.get('sample_size'),
                'confidence': m41_details.get('confidence'),
                'source': m41_details.get('source'),
                'release_date': m41_details.get('release_date'),
                'details': m41_details,
            }
            if m41_size_mult < 1.0:
                result['_m41_size_mult'] = m41_size_mult
    except Exception as e:
        result['m41'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M42: Eurozone GDP Flash Session Bias (regime-conditional) ──
    m42_score_adj = 0.0
    m42_size_mult = 1.0
    m42_status = 'SKIP'
    m42_details = {}
    try:
        _wyckoff_for_m42 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m42 = result.get('m9', {}).get('regime', 'CHOP')
        m42_status, m42_score_adj, m42_size_mult, m42_details = score_m42_ez_gdp(
            wyckoff_phase=_wyckoff_for_m42,
            vol_regime=_vol_for_m42,
            direction=direction, today_str=today_str, config=cfg)
        if m42_details and m42_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m42'] = {
                'status': m42_status,
                'score_adj': m42_score_adj,
                'size_mult': m42_size_mult,
                'regime': m42_details.get('regime', '?'),
                'bias': m42_details.get('bias', '?'),
                'gdp_qoq': m42_details.get('gdp_qoq'),
                'gdp_yoy': m42_details.get('gdp_yoy'),
                'consensus_qoq': m42_details.get('consensus_qoq'),
                'signal': m42_details.get('signal'),
                'health': m42_details.get('health'),
                'momentum': m42_details.get('momentum'),
                'avg_ret_24h': m42_details.get('avg_ret_24h'),
                'win_rate': m42_details.get('win_rate'),
                'sample_size': m42_details.get('sample_size'),
                'confidence': m42_details.get('confidence'),
                'source': m42_details.get('source'),
                'release_date': m42_details.get('release_date'),
                'details': m42_details,
            }
            if m42_size_mult < 1.0:
                result['_m42_size_mult'] = m42_size_mult
    except Exception as e:
        result['m42'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M43: US GDP Advance Estimate Session Bias (regime-conditional) ──
    m43_score_adj = 0.0
    m43_size_mult = 1.0
    m43_status = 'SKIP'
    m43_details = {}
    try:
        _wyckoff_for_m43 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m43 = result.get('m9', {}).get('regime', 'CHOP')
        m43_status, m43_score_adj, m43_size_mult, m43_details = score_m43_us_gdp(
            wyckoff_phase=_wyckoff_for_m43,
            vol_regime=_vol_for_m43,
            direction=direction, today_str=today_str, config=cfg)
        if m43_details and m43_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m43'] = {
                'status': m43_status,
                'score_adj': m43_score_adj,
                'size_mult': m43_size_mult,
                'regime': m43_details.get('regime', '?'),
                'bias': m43_details.get('bias', '?'),
                'gdp_qoq': m43_details.get('gdp_qoq'),
                'consensus': m43_details.get('consensus'),
                'signal': m43_details.get('signal'),
                'health': m43_details.get('health'),
                'momentum': m43_details.get('momentum'),
                'avg_ret_24h': m43_details.get('avg_ret_24h'),
                'win_rate': m43_details.get('win_rate'),
                'sample_size': m43_details.get('sample_size'),
                'confidence': m43_details.get('confidence'),
                'source': m43_details.get('source'),
                'release_date': m43_details.get('release_date'),
                'contrarian': m43_details.get('contrarian', False),
                'details': m43_details,
            }
            if m43_size_mult < 1.0:
                result['_m43_size_mult'] = m43_size_mult
    except Exception as e:
        result['m43'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M44: US Durable Goods Orders Session Bias (regime-conditional) ──
    m44_score_adj = 0.0
    m44_size_mult = 1.0
    m44_status = 'SKIP'
    m44_details = {}
    try:
        _wyckoff_for_m44 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m44 = result.get('m9', {}).get('regime', 'CHOP')
        m44_status, m44_score_adj, m44_size_mult, m44_details = score_m44_durables(
            wyckoff_phase=_wyckoff_for_m44,
            vol_regime=_vol_for_m44,
            direction=direction, today_str=today_str, config=cfg)
        if m44_details and m44_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m44'] = {
                'status': m44_status,
                'score_adj': m44_score_adj,
                'size_mult': m44_size_mult,
                'regime': m44_details.get('regime', '?'),
                'bias': m44_details.get('bias', '?'),
                'headline_mom': m44_details.get('headline_mom'),
                'core_mom': m44_details.get('core_mom'),
                'consensus': m44_details.get('consensus'),
                'signal': m44_details.get('signal'),
                'capex_health': m44_details.get('capex_health'),
                'momentum': m44_details.get('momentum'),
                'avg_ret_24h': m44_details.get('avg_ret_24h'),
                'win_rate': m44_details.get('win_rate'),
                'sample_size': m44_details.get('sample_size'),
                'confidence': m44_details.get('confidence'),
                'source': m44_details.get('source'),
                'release_date': m44_details.get('release_date'),
                'details': m44_details,
            }
            if m44_size_mult < 1.0:
                result['_m44_size_mult'] = m44_size_mult
    except Exception as e:
        result['m44'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M45: Core PCE + Personal Spending Session Bias (regime-conditional) ──
    m45_score_adj = 0.0
    m45_size_mult = 1.0
    m45_status = 'SKIP'
    m45_details = {}
    try:
        _wyckoff_for_m45 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m45 = result.get('m9', {}).get('regime', 'CHOP')
        m45_status, m45_score_adj, m45_size_mult, m45_details = score_m45_pce(
            wyckoff_phase=_wyckoff_for_m45,
            vol_regime=_vol_for_m45,
            direction=direction, today_str=today_str, config=cfg)
        if m45_details and m45_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m45'] = {
                'status': m45_status,
                'score_adj': m45_score_adj,
                'size_mult': m45_size_mult,
                'regime': m45_details.get('regime', '?'),
                'bias': m45_details.get('bias', '?'),
                'core_pce_yoy': m45_details.get('core_pce_yoy'),
                'core_pce_mom': m45_details.get('core_pce_mom'),
                'spending_mom': m45_details.get('spending_mom'),
                'consensus_yoy': m45_details.get('consensus_yoy'),
                'signal': m45_details.get('signal'),
                'inflation': m45_details.get('inflation'),
                'spending_signal': m45_details.get('spending_signal'),
                'avg_ret_24h': m45_details.get('avg_ret_24h'),
                'win_rate': m45_details.get('win_rate'),
                'sample_size': m45_details.get('sample_size'),
                'confidence': m45_details.get('confidence'),
                'source': m45_details.get('source'),
                'release_date': m45_details.get('release_date'),
                'fomc_metric': m45_details.get('fomc_metric', False),
                'details': m45_details,
            }
            if m45_size_mult < 1.0:
                result['_m45_size_mult'] = m45_size_mult
    except Exception as e:
        result['m45'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M46: Japan CPI (Tokyo Flash) Session Bias (regime-conditional) ──
    m46_score_adj = 0.0
    m46_size_mult = 1.0
    m46_status = 'SKIP'
    m46_details = {}
    try:
        _wyckoff_for_m46 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m46 = result.get('m9', {}).get('regime', 'CHOP')
        m46_status, m46_score_adj, m46_size_mult, m46_details = score_m46_japan_cpi(
            wyckoff_phase=_wyckoff_for_m46,
            vol_regime=_vol_for_m46,
            direction=direction, today_str=today_str, config=cfg)
        if m46_details and m46_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m46'] = {
                'status': m46_status,
                'score_adj': m46_score_adj,
                'size_mult': m46_size_mult,
                'regime': m46_details.get('regime', '?'),
                'bias': m46_details.get('bias', '?'),
                'cpi_yoy': m46_details.get('cpi_yoy'),
                'core_yoy': m46_details.get('core_yoy'),
                'consensus': m46_details.get('consensus'),
                'signal': m46_details.get('signal'),
                'inflation': m46_details.get('inflation'),
                'core_pressure': m46_details.get('core_pressure'),
                'avg_ret_24h': m46_details.get('avg_ret_24h'),
                'win_rate': m46_details.get('win_rate'),
                'sample_size': m46_details.get('sample_size'),
                'confidence': m46_details.get('confidence'),
                'source': m46_details.get('source'),
                'release_date': m46_details.get('release_date'),
                'boj_metric': m46_details.get('boj_metric', False),
                'details': m46_details,
            }
            if m46_size_mult < 1.0:
                result['_m46_size_mult'] = m46_size_mult
    except Exception as e:
        result['m46'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M47: BoJ Rate Decision Session Bias (regime-conditional) ──
    m47_score_adj = 0.0
    m47_size_mult = 1.0
    m47_status = 'SKIP'
    m47_details = {}
    try:
        _wyckoff_for_m47 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m47 = result.get('m9', {}).get('regime', 'CHOP')
        m47_status, m47_score_adj, m47_size_mult, m47_details = score_m47_boj_rate(
            wyckoff_phase=_wyckoff_for_m47,
            vol_regime=_vol_for_m47,
            direction=direction, today_str=today_str, config=cfg)
        if m47_details and m47_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m47'] = {
                'status': m47_status,
                'score_adj': m47_score_adj,
                'size_mult': m47_size_mult,
                'regime': m47_details.get('regime', '?'),
                'bias': m47_details.get('bias', '?'),
                'rate': m47_details.get('rate'),
                'prev_rate': m47_details.get('prev_rate'),
                'rate_change': m47_details.get('rate_change'),
                'signal': m47_details.get('signal'),
                'signal_hint': m47_details.get('signal_hint'),
                'avg_ret_24h': m47_details.get('avg_ret_24h'),
                'win_rate': m47_details.get('win_rate'),
                'sample_size': m47_details.get('sample_size'),
                'confidence': m47_details.get('confidence'),
                'source': m47_details.get('source'),
                'release_date': m47_details.get('release_date'),
                'details': m47_details,
            }
            if m47_size_mult < 1.0:
                result['_m47_size_mult'] = m47_size_mult
    except Exception as e:
        result['m47'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M48: ECB Rate Decision + Lagarde Presser Session Bias (regime-conditional) ──
    m48_score_adj = 0.0
    m48_size_mult = 1.0
    m48_status = 'SKIP'
    m48_details = {}
    try:
        _wyckoff_for_m48 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m48 = result.get('m9', {}).get('regime', 'CHOP')
        m48_status, m48_score_adj, m48_size_mult, m48_details = score_m48_ecb_rate(
            wyckoff_phase=_wyckoff_for_m48,
            vol_regime=_vol_for_m48,
            direction=direction, today_str=today_str, config=cfg)
        if m48_details and m48_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m48'] = {
                'status': m48_status,
                'score_adj': m48_score_adj,
                'size_mult': m48_size_mult,
                'regime': m48_details.get('regime', '?'),
                'bias': m48_details.get('bias', '?'),
                'deposit_rate': m48_details.get('deposit_rate'),
                'prev_rate': m48_details.get('prev_rate'),
                'rate_change': m48_details.get('rate_change'),
                'signal': m48_details.get('signal'),
                'signal_hint': m48_details.get('signal_hint'),
                'avg_ret_24h': m48_details.get('avg_ret_24h'),
                'win_rate': m48_details.get('win_rate'),
                'sample_size': m48_details.get('sample_size'),
                'confidence': m48_details.get('confidence'),
                'source': m48_details.get('source'),
                'release_date': m48_details.get('release_date'),
                'details': m48_details,
            }
            if m48_size_mult < 1.0:
                result['_m48_size_mult'] = m48_size_mult
    except Exception as e:
        result['m48'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M49: BoE Rate Decision + MPC Vote Split Session Bias (regime-conditional) ──
    m49_score_adj = 0.0
    m49_size_mult = 1.0
    m49_status = 'SKIP'
    m49_details = {}
    try:
        _wyckoff_for_m49 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m49 = result.get('m9', {}).get('regime', 'CHOP')
        m49_status, m49_score_adj, m49_size_mult, m49_details = score_m49_boe_rate(
            wyckoff_phase=_wyckoff_for_m49,
            vol_regime=_vol_for_m49,
            direction=direction, today_str=today_str, config=cfg)
        if m49_details and m49_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m49'] = {
                'status': m49_status,
                'score_adj': m49_score_adj,
                'size_mult': m49_size_mult,
                'regime': m49_details.get('regime', '?'),
                'bias': m49_details.get('bias', '?'),
                'rate': m49_details.get('rate'),
                'prev_rate': m49_details.get('prev_rate'),
                'rate_change': m49_details.get('rate_change'),
                'signal': m49_details.get('signal'),
                'signal_hint': m49_details.get('signal_hint'),
                'vote_split': m49_details.get('vote_split'),
                'avg_ret_24h': m49_details.get('avg_ret_24h'),
                'win_rate': m49_details.get('win_rate'),
                'sample_size': m49_details.get('sample_size'),
                'confidence': m49_details.get('confidence'),
                'source': m49_details.get('source'),
                'release_date': m49_details.get('release_date'),
                'details': m49_details,
            }
            if m49_size_mult < 1.0:
                result['_m49_size_mult'] = m49_size_mult
    except Exception as e:
        result['m49'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M50: CB Consumer Confidence Session Bias (regime-conditional) ──
    m50_score_adj = 0.0
    m50_size_mult = 1.0
    m50_status = 'SKIP'
    m50_details = {}
    try:
        _wyckoff_for_m50 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m50 = result.get('m9', {}).get('regime', 'CHOP')
        m50_status, m50_score_adj, m50_size_mult, m50_details = score_m50_cb_confidence(
            wyckoff_phase=_wyckoff_for_m50,
            vol_regime=_vol_for_m50,
            direction=direction, date_str=today_str)
        if m50_details and m50_status in ('ACTIVE', 'NO_EDGE'):
            result['m50'] = {
                'status': m50_status,
                'score_adj': m50_score_adj,
                'size_mult': m50_size_mult,
                'signal': m50_details.get('signal'),
                'level': m50_details.get('level'),
                'actual': m50_details.get('actual'),
                'consensus': m50_details.get('consensus'),
                'surprise_pct': m50_details.get('surprise_pct'),
                'edge_key': m50_details.get('edge_key'),
                'edge_dir': m50_details.get('edge_dir'),
                'edge_avg': m50_details.get('edge_avg'),
                'edge_wr': m50_details.get('edge_wr'),
                'edge_n': m50_details.get('edge_n'),
                'details': m50_details,
            }
            if m50_size_mult < 1.0:
                result['_m50_size_mult'] = m50_size_mult
    except Exception as e:
        result['m50'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M51: UK Monthly GDP Session Bias (regime-conditional) ──
    m51_score_adj = 0.0
    m51_size_mult = 1.0
    m51_status = 'SKIP'
    m51_details = {}
    try:
        _wyckoff_for_m51 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m51 = result.get('m9', {}).get('regime', 'CHOP')
        m51_status, m51_score_adj, m51_size_mult, m51_details = score_m51_uk_gdp(
            wyckoff_phase=_wyckoff_for_m51,
            vol_regime=_vol_for_m51,
            direction=direction, date_str=today_str)
        if m51_details and m51_status in ('ACTIVE', 'NO_EDGE'):
            result['m51'] = {
                'status': m51_status,
                'score_adj': m51_score_adj,
                'size_mult': m51_size_mult,
                'signal': m51_details.get('signal'),
                'level': m51_details.get('level'),
                'gdp_mom': m51_details.get('gdp_mom'),
                'consensus_mom': m51_details.get('consensus_mom'),
                'surprise': m51_details.get('surprise'),
                'gdp_yoy': m51_details.get('gdp_yoy'),
                'edge_key': m51_details.get('edge_key'),
                'edge_dir': m51_details.get('edge_dir'),
                'edge_avg': m51_details.get('edge_avg'),
                'edge_wr': m51_details.get('edge_wr'),
                'edge_n': m51_details.get('edge_n'),
                'details': m51_details,
            }
            if m51_size_mult < 1.0:
                result['_m51_size_mult'] = m51_size_mult
    except Exception as e:
        result['m51'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M52: RBA Rate Decision Session Bias (regime-conditional) ──
    m52_score_adj = 0.0
    m52_size_mult = 1.0
    m52_status = 'SKIP'
    m52_details = {}
    try:
        _wyckoff_for_m52 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m52 = result.get('m9', {}).get('regime', 'CHOP')
        m52_status, m52_score_adj, m52_size_mult, m52_details = score_m52_rba_rate(
            wyckoff_phase=_wyckoff_for_m52,
            vol_regime=_vol_for_m52,
            direction=direction, date_str=today_str)
        if m52_details and m52_status in ('ACTIVE', 'NO_EDGE'):
            result['m52'] = {
                'status': m52_status,
                'score_adj': m52_score_adj,
                'size_mult': m52_size_mult,
                'signal': m52_details.get('signal'),
                'level': m52_details.get('level'),
                'cycle': m52_details.get('cycle'),
                'rate': m52_details.get('rate'),
                'prev_rate': m52_details.get('prev_rate'),
                'rate_change': m52_details.get('rate_change'),
                'edge_key': m52_details.get('edge_key'),
                'edge_dir': m52_details.get('edge_dir'),
                'edge_avg': m52_details.get('edge_avg'),
                'edge_wr': m52_details.get('edge_wr'),
                'edge_n': m52_details.get('edge_n'),
                'details': m52_details,
            }
            if m52_size_mult < 1.0:
                result['_m52_size_mult'] = m52_size_mult
    except Exception as e:
        result['m52'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M53: Australia Quarterly CPI Session Bias (regime-conditional) ──
    m53_score_adj = 0.0
    m53_size_mult = 1.0
    m53_status = 'SKIP'
    m53_details = {}
    try:
        _wyckoff_for_m53 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m53 = result.get('m9', {}).get('regime', 'CHOP')
        m53_status, m53_score_adj, m53_size_mult, m53_details = score_m53_au_cpi(
            wyckoff_phase=_wyckoff_for_m53,
            vol_regime=_vol_for_m53,
            direction=direction, date_str=today_str)
        if m53_details and m53_status in ('ACTIVE', 'NO_EDGE'):
            result['m53'] = {
                'status': m53_status,
                'score_adj': m53_score_adj,
                'size_mult': m53_size_mult,
                'signal': m53_details.get('signal'),
                'level': m53_details.get('level'),
                'trimmed_mean_yoy': m53_details.get('trimmed_mean_yoy'),
                'prev_trimmed_yoy': m53_details.get('prev_trimmed_yoy'),
                'headline_yoy': m53_details.get('headline_yoy'),
                'quarter': m53_details.get('quarter'),
                'edge_dir': m53_details.get('edge_dir'),
                'edge_avg': m53_details.get('edge_avg'),
                'edge_wr': m53_details.get('edge_wr'),
                'edge_n': m53_details.get('edge_n'),
                'details': m53_details,
            }
            if m53_size_mult < 1.0:
                result['_m53_size_mult'] = m53_size_mult
    except Exception as e:
        result['m53'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M54: China Quarterly GDP Session Bias (regime-conditional) ──
    m54_score_adj = 0.0
    m54_size_mult = 1.0
    m54_status = 'SKIP'
    m54_details = {}
    try:
        _wyckoff_for_m54 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m54 = result.get('m9', {}).get('regime', 'CHOP')
        m54_status, m54_score_adj, m54_size_mult, m54_details = score_m54_china_gdp(
            wyckoff_phase=_wyckoff_for_m54,
            vol_regime=_vol_for_m54,
            direction=direction, date_str=today_str)
        if m54_details and m54_status in ('ACTIVE', 'NO_EDGE'):
            result['m54'] = {
                'status': m54_status,
                'score_adj': m54_score_adj,
                'size_mult': m54_size_mult,
                'signal': m54_details.get('signal'),
                'level': m54_details.get('level'),
                'gdp_yoy': m54_details.get('gdp_yoy'),
                'consensus_yoy': m54_details.get('consensus_yoy'),
                'surprise': m54_details.get('surprise'),
                'retail_yoy': m54_details.get('retail_yoy'),
                'industrial_yoy': m54_details.get('industrial_yoy'),
                'quarter': m54_details.get('quarter'),
                'edge_dir': m54_details.get('edge_dir'),
                'edge_avg': m54_details.get('edge_avg'),
                'edge_wr': m54_details.get('edge_wr'),
                'edge_n': m54_details.get('edge_n'),
                'details': m54_details,
            }
            if m54_size_mult < 1.0:
                result['_m54_size_mult'] = m54_size_mult
    except Exception as e:
        result['m54'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M55: US Treasury Auction Session Bias (regime-conditional) ──
    m55_score_adj = 0.0
    m55_size_mult = 1.0
    m55_status = 'SKIP'
    m55_details = {}
    try:
        _wyckoff_for_m55 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m55 = result.get('m9', {}).get('regime', 'CHOP')
        m55_status, m55_score_adj, m55_size_mult, m55_details = score_m55_treasury(
            wyckoff_phase=_wyckoff_for_m55,
            vol_regime=_vol_for_m55,
            direction=direction, date_str=today_str)
        if m55_details and m55_status in ('ACTIVE', 'NO_EDGE'):
            result['m55'] = {
                'status': m55_status,
                'score_adj': m55_score_adj,
                'size_mult': m55_size_mult,
                'demand': m55_details.get('demand'),
                'type': m55_details.get('type'),
                'bid_to_cover': m55_details.get('bid_to_cover'),
                'tail_bps': m55_details.get('tail_bps'),
                'edge_key': m55_details.get('edge_key'),
                'edge_dir': m55_details.get('edge_dir'),
                'edge_avg': m55_details.get('edge_avg'),
                'edge_wr': m55_details.get('edge_wr'),
                'edge_n': m55_details.get('edge_n'),
                'details': m55_details,
            }
            if m55_size_mult < 1.0:
                result['_m55_size_mult'] = m55_size_mult
    except Exception as e:
        result['m55'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M56: US CPI Session Bias (regime-conditional) ──
    m56_score_adj = 0.0
    m56_size_mult = 1.0
    m56_status = 'SKIP'
    m56_details = {}
    try:
        _wyckoff_for_m56 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m56 = result.get('m9', {}).get('regime', 'CHOP')
        m56_status, m56_score_adj, m56_size_mult, m56_details = score_m56_us_cpi(
            wyckoff_phase=_wyckoff_for_m56,
            vol_regime=_vol_for_m56,
            direction=direction, date_str=today_str)
        if m56_details and m56_status in ('ACTIVE', 'WEAK', 'NO_EDGE'):
            result['m56'] = {
                'status': m56_status,
                'score_adj': m56_score_adj,
                'size_mult': m56_size_mult,
                'cpi_yoy': m56_details.get('cpi_yoy'),
                'consensus_yoy': m56_details.get('consensus_yoy'),
                'signal': m56_details.get('signal'),
                'inflation': m56_details.get('inflation'),
                'bias': m56_details.get('bias'),
                'avg_ret_24h': m56_details.get('avg_ret_24h'),
                'win_rate': m56_details.get('win_rate'),
                'sample_size': m56_details.get('sample_size'),
                'edge_key': m56_details.get('source'),
                'details': m56_details,
            }
            if m56_size_mult < 1.0:
                result['_m56_size_mult'] = m56_size_mult
    except Exception as e:
        result['m56'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M57: FOMC Rate Decision Session Bias (regime-conditional) ──
    m57_score_adj = 0.0
    m57_size_mult = 1.0
    m57_status = 'SKIP'
    m57_details = {}
    try:
        _wyckoff_for_m57 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m57 = result.get('m9', {}).get('regime', 'CHOP')
        m57_status, m57_score_adj, m57_size_mult, m57_details = score_m57_fomc(
            wyckoff_phase=_wyckoff_for_m57,
            vol_regime=_vol_for_m57,
            direction=direction, date_str=today_str)
        if m57_details and m57_status in ('ACTIVE', 'WEAK', 'NO_EDGE'):
            result['m57'] = {
                'status': m57_status,
                'score_adj': m57_score_adj,
                'size_mult': m57_size_mult,
                'rate': m57_details.get('rate'),
                'prior_rate': m57_details.get('prior_rate'),
                'rate_action': m57_details.get('rate_action'),
                'signal': m57_details.get('signal'),
                'stance': m57_details.get('stance'),
                'dot_plot': m57_details.get('dot_plot'),
                'vote': m57_details.get('vote'),
                'bias': m57_details.get('bias'),
                'avg_ret_24h': m57_details.get('avg_ret_24h'),
                'win_rate': m57_details.get('win_rate'),
                'sample_size': m57_details.get('sample_size'),
                'edge_key': m57_details.get('source'),
                'details': m57_details,
            }
            if m57_size_mult < 1.0:
                result['_m57_size_mult'] = m57_size_mult
    except Exception as e:
        result['m57'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M58: Powell Press Conference Session Bias (regime-conditional) ──
    m58_score_adj = 0.0
    m58_size_mult = 1.0
    m58_status = 'SKIP'
    m58_details = {}
    try:
        _wyckoff_for_m58 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m58 = result.get('m9', {}).get('regime', 'CHOP')
        m58_status, m58_score_adj, m58_size_mult, m58_details = score_m58_presser(
            wyckoff_phase=_wyckoff_for_m58,
            vol_regime=_vol_for_m58,
            direction=direction, date_str=today_str)
        if m58_details and m58_status in ('ACTIVE', 'WEAK', 'NO_EDGE'):
            result['m58'] = {
                'status': m58_status,
                'score_adj': m58_score_adj,
                'size_mult': m58_size_mult,
                'fomc_stance': m58_details.get('fomc_stance'),
                'powell_tone': m58_details.get('powell_tone'),
                'tone_vs_statement': m58_details.get('tone_vs_statement'),
                'signal': m58_details.get('signal'),
                'bias': m58_details.get('bias'),
                'avg_ret_24h': m58_details.get('avg_ret_24h'),
                'win_rate': m58_details.get('win_rate'),
                'sample_size': m58_details.get('sample_size'),
                'edge_key': m58_details.get('source'),
                'details': m58_details,
            }
            if m58_size_mult < 1.0:
                result['_m58_size_mult'] = m58_size_mult
    except Exception as e:
        result['m58'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M59: FOMC Meeting Minutes Session Bias (regime-conditional) ──
    m59_score_adj = 0.0
    m59_size_mult = 1.0
    m59_status = 'SKIP'
    m59_details = {}
    try:
        _wyckoff_for_m59 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m59 = result.get('m9', {}).get('regime', 'CHOP')
        m59_status, m59_score_adj, m59_size_mult, m59_details = score_m59_minutes(
            wyckoff_phase=_wyckoff_for_m59,
            vol_regime=_vol_for_m59,
            direction=direction, date_str=today_str)
        if m59_details and m59_status in ('ACTIVE', 'WEAK', 'NO_EDGE'):
            result['m59'] = {
                'status': m59_status,
                'score_adj': m59_score_adj,
                'size_mult': m59_size_mult,
                'meeting_date': m59_details.get('meeting_date'),
                'meeting_stance': m59_details.get('meeting_stance'),
                'minutes_surprise': m59_details.get('minutes_surprise'),
                'key_revelation': m59_details.get('key_revelation'),
                'signal': m59_details.get('signal'),
                'bias': m59_details.get('bias'),
                'avg_ret_24h': m59_details.get('avg_ret_24h'),
                'win_rate': m59_details.get('win_rate'),
                'sample_size': m59_details.get('sample_size'),
                'edge_key': m59_details.get('source'),
                'details': m59_details,
            }
            if m59_size_mult < 1.0:
                result['_m59_size_mult'] = m59_size_mult
    except Exception as e:
        result['m59'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M65: China NBS Activity Session Bias (regime-conditional) ──
    m65_score_adj = 0.0
    m65_size_mult = 1.0
    m65_status = 'SKIP'
    m65_details = {}
    try:
        _wyckoff_for_m65 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m65 = result.get('m9', {}).get('regime', 'CHOP')
        m65_status, m65_score_adj, m65_size_mult, m65_details = score_m65_china_activity(
            today_str=today_str,
            wyckoff_phase=_wyckoff_for_m65,
            vol_regime=_vol_for_m65,
            direction=direction,
            config=cfg)
        if m65_details and m65_details.get('regime') not in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
            result['m65'] = {
                'status': m65_status,
                'score_adj': m65_score_adj,
                'size_mult': m65_size_mult,
                'regime': m65_details.get('regime', '?'),
                'composite': m65_details.get('composite', '?'),
                'release_date': m65_details.get('release_date'),
                'wyckoff': m65_details.get('wyckoff'),
                'vol': m65_details.get('vol'),
                'edge_direction': m65_details.get('edge_direction'),
                'edge_avg_ret': m65_details.get('edge_avg_ret'),
                'edge_win_rate': m65_details.get('edge_win_rate'),
                'edge_n': m65_details.get('edge_n'),
                'edge_confidence': m65_details.get('edge_confidence'),
                'details': m65_details,
            }
            if m65_size_mult < 1.0:
                result['_m65_size_mult'] = m65_size_mult
    except Exception as e:
        result['m65'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M60: US PPI Session Bias (regime-conditional) ──
    m60_score_adj = 0.0
    m60_size_mult = 1.0
    m60_status = 'SKIP'
    m60_details = {}
    try:
        _wyckoff_for_m60 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m60 = result.get('m9', {}).get('regime', 'CHOP')
        m60_status, m60_score_adj, m60_size_mult, m60_details = score_m60_us_ppi(
            wyckoff_phase=_wyckoff_for_m60,
            vol_regime=_vol_for_m60,
            direction=direction, today_str=today_str, config=cfg)
        if m60_details and m60_status in ('PASS', 'WEAK'):
            result['m60'] = {
                'status': m60_status,
                'score_adj': m60_score_adj,
                'size_mult': m60_size_mult,
                'ppi_yoy': m60_details.get('ppi_yoy'),
                'consensus_yoy': m60_details.get('consensus_yoy'),
                'surprise': m60_details.get('surprise'),
                'ppi_signal': m60_details.get('ppi_signal'),
                'infl_level': m60_details.get('infl_level'),
                'bias': m60_details.get('bias'),
                'avg_ret_24h': m60_details.get('avg_ret_24h'),
                'win_rate': m60_details.get('win_rate'),
                'sample_size': m60_details.get('sample_size'),
                'source': m60_details.get('source'),
                'release_date': m60_details.get('release_date'),
                'details': m60_details,
            }
            if m60_size_mult < 1.0:
                result['_m60_size_mult'] = m60_size_mult
    except Exception as e:
        result['m60'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M61: US Weekly Jobless Claims Session Bias (regime-conditional) ──
    m61_score_adj = 0.0
    m61_size_mult = 1.0
    m61_status = 'SKIP'
    m61_details = {}
    try:
        _wyckoff_for_m61 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m61 = result.get('m9', {}).get('regime', 'CHOP')
        m61_status, m61_score_adj, m61_size_mult, m61_details = score_m61_us_claims(
            wyckoff_phase=_wyckoff_for_m61,
            vol_regime=_vol_for_m61,
            direction=direction, today_str=today_str, config=cfg)
        if m61_details and m61_details.get('regime') not in ('DISABLED', 'NOT_CLAIMS_DAY', 'NO_CLAIMS_DATA', 'NO_EDGE'):
            result['m61'] = {
                'status': m61_status,
                'score_adj': m61_score_adj,
                'size_mult': m61_size_mult,
                'regime': m61_details.get('regime', '?'),
                'bias': m61_details.get('bias', '?'),
                'claims_k': m61_details.get('claims_k'),
                'claims_signal': m61_details.get('claims_signal'),
                'claims_trend': m61_details.get('claims_trend'),
                'avg_ret_24h': m61_details.get('avg_ret_24h'),
                'win_rate': m61_details.get('win_rate'),
                'sample_size': m61_details.get('sample_size'),
                'confidence': m61_details.get('confidence'),
                'trend_mult': m61_details.get('trend_mult'),
                'details': m61_details,
            }
            if m61_size_mult < 1.0:
                result['_m61_size_mult'] = m61_size_mult
    except Exception as e:
        result['m61'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── M62: US Unemployment Rate Session Bias (regime-conditional) ──
    m62_score_adj = 0.0
    m62_size_mult = 1.0
    m62_status = 'SKIP'
    m62_details = {}
    try:
        _wyckoff_for_m62 = result.get('m21', {}).get('phase', 'RANGE')
        _vol_for_m62 = result.get('m9', {}).get('regime', 'CHOP')
        m62_status, m62_score_adj, m62_size_mult, m62_details = score_m62_us_unemployment(
            wyckoff_phase=_wyckoff_for_m62,
            vol_regime=_vol_for_m62,
            direction=direction, today_str=today_str, config=cfg)
        if m62_details and m62_status in ('PASS', 'WEAK'):
            result['m62'] = {
                'status': m62_status,
                'score_adj': m62_score_adj,
                'size_mult': m62_size_mult,
                'regime': m62_details.get('regime', '?'),
                'bias': m62_details.get('bias', '?'),
                'unemp_rate': m62_details.get('unemp_rate'),
                'unemp_signal': m62_details.get('unemp_signal'),
                'unemp_change': m62_details.get('unemp_change'),
                'sahm_triggered': m62_details.get('sahm_triggered'),
                'avg_ret_24h': m62_details.get('avg_ret_24h'),
                'win_rate': m62_details.get('win_rate'),
                'sample_size': m62_details.get('sample_size'),
                'confidence': m62_details.get('confidence'),
                'release_date': m62_details.get('release_date'),
                'details': m62_details,
            }
            if m62_size_mult < 1.0:
                result['_m62_size_mult'] = m62_size_mult
    except Exception as e:
        result['m62'] = {'status': 'ERROR', 'score_adj': 0.0, 'error': str(e)}

    # ── Fetch Traditional Finance Data (M66-M73) ──
    _tradfi_data = fetch_all_tradfi_data(config=cfg)
    _dxy_df = _tradfi_data.get('dxy')

    # ── M66: USD/JPY Carry Trade Proxy ──
    m66_score = 0.5
    m66_status = 'SKIP'
    if cfg.get('M66_ENABLED', True):
        try:
            _usdjpy_df = _tradfi_data.get('usdjpy')
            m66_status, m66_score, m66_details = score_m66_usdjpy(
                _usdjpy_df, _dxy_df, direction, config=cfg)
            if m66_status == 'PASS':
                result['m66'] = {'status': m66_status, 'score': round(float(m66_score), 3),
                                 'details': m66_details}
        except Exception as e:
            result['m66'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # ── M67: DXY Divergence ──
    m67_score = 0.5
    m67_status = 'SKIP'
    if cfg.get('M67_ENABLED', True):
        try:
            eth_now = float(row['Close'])
            eth_prev = float(df_15m['Close'].iloc[max(0, idx - 1)])
            m67_status, m67_score, m67_details = score_m67_dxy(
                _dxy_df, eth_now, eth_prev, direction, config=cfg)
            if m67_status == 'PASS':
                result['m67'] = {'status': m67_status, 'score': round(float(m67_score), 3),
                                 'details': m67_details}
        except Exception as e:
            result['m67'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # ── M68: 10Y Yield + TIPS ──
    m68_score = 0.5
    m68_status = 'SKIP'
    if cfg.get('M68_ENABLED', True):
        try:
            _tnx_df = _tradfi_data.get('tnx')
            _tips_df = None
            try:
                from src.modules.m68_yield import fetch_tips_yield
                _tips_df = fetch_tips_yield()
            except Exception:
                pass
            m68_status, m68_score, m68_details = score_m68_yield(
                _tnx_df, _tips_df, direction, config=cfg)
            if m68_status == 'PASS':
                result['m68'] = {'status': m68_status, 'score': round(float(m68_score), 3),
                                 'details': m68_details}
        except Exception as e:
            result['m68'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # ── M69: VIX Regime ──
    m69_score = 0.5
    m69_status = 'SKIP'
    if cfg.get('M69_ENABLED', True):
        try:
            _vix_df = _tradfi_data.get('vix')
            m69_status, m69_score, m69_details = score_m69_vix(
                _vix_df, direction, config=cfg, df_dxy=_dxy_df)
            if m69_status == 'PASS':
                result['m69'] = {'status': m69_status, 'score': round(float(m69_score), 3),
                                 'details': m69_details}
        except Exception as e:
            result['m69'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # ── M70: WTI Crude Oil ──
    m70_score = 0.5
    m70_status = 'SKIP'
    if cfg.get('M70_ENABLED', True):
        try:
            _wti_df = _tradfi_data.get('wti')
            m70_status, m70_score, m70_details = score_m70_wti(
                _wti_df, _dxy_df, direction, config=cfg)
            if m70_status == 'PASS':
                result['m70'] = {'status': m70_status, 'score': round(float(m70_score), 3),
                                 'details': m70_details}
        except Exception as e:
            result['m70'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # ── M71: Gold + DXY Geopolitical ──
    m71_score = 0.5
    m71_status = 'SKIP'
    if cfg.get('M71_ENABLED', True):
        try:
            _gold_df = _tradfi_data.get('gold')
            m71_status, m71_score, m71_details = score_m71_gold(
                _gold_df, _dxy_df, direction, config=cfg)
            if m71_status == 'PASS':
                result['m71'] = {'status': m71_status, 'score': round(float(m71_score), 3),
                                 'details': m71_details}
        except Exception as e:
            result['m71'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # ── M72: BTC Dominance ──
    m72_score = 0.5
    m72_status = 'SKIP'
    if cfg.get('M72_ENABLED', True):
        try:
            _btcdom = fetch_btcdom()
            m72_status, m72_score, m72_details = score_m72_btcdom(
                _btcdom, direction, config=cfg)
            if m72_status == 'PASS':
                result['m72'] = {'status': m72_status, 'score': round(float(m72_score), 3),
                                 'details': m72_details}
        except Exception as e:
            result['m72'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # ── M73: Stablecoin Mints ──
    m73_score = 0.5
    m73_status = 'SKIP'
    if cfg.get('M73_ENABLED', True):
        try:
            _mint_data = fetch_stablecoin_mints()
            m73_status, m73_score, m73_details = score_m73_stablecoin(
                _mint_data, direction, config=cfg)
            if m73_status == 'PASS':
                result['m73'] = {'status': m73_status, 'score': round(float(m73_score), 3),
                                 'details': m73_details}
        except Exception as e:
            result['m73'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # ── M22: Macro Regime Aggregator (reads M23-M65 outputs) ──
    # M22 is now a pure downstream aggregator — it does NOT fetch its own data.
    # It reads the scores/classifications from M23-M65 and produces a regime label.
    try:
        m22_status, m22_score, m22_details = aggregate_macro_regime(result, config=cfg)
        if m22_details and m22_details.get('regime') not in ('DISABLED', 'NO_DATA'):
            result['m22'] = {
                'status': m22_status,
                'score': round(float(m22_score), 3),
                'regime': m22_details.get('regime', '?'),
                'severity': m22_details.get('severity', '?'),
                'inflation_label': m22_details.get('inflation_label', '?'),
                'labor_label': m22_details.get('labor_label', '?'),
                'policy_label': m22_details.get('policy_label', '?'),
                'global_label': m22_details.get('global_label', '?'),
                'size_mult': m22_details.get('size_mult', 1.0),
                'factors': m22_details.get('factors', []),
                'details': m22_details,
            }
            _m22_size_mult = m22_details.get('size_mult', 1.0)
            if _m22_size_mult < 1.0:
                result['_m22_size_mult'] = _m22_size_mult
    except Exception as e:
        result['m22'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # ── Cascade Meta-Aggregator (all regional cascades) ──
    # Runs after M22 so it can use the regime label.
    # Each cascade scores its release chain independently, then
    # the meta-aggregator produces a weighted global macro score.
    try:
        _cascade_regime = result.get('m22', {}).get('regime', 'UNKNOWN')
        # Build release_data_map from FRED cache + module results
        _fred_data = None
        try:
            _fred_cache_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'data', 'fred', 'claims_cache.json')
            if os.path.exists(_fred_cache_path):
                with open(_fred_cache_path) as _f:
                    _fred_data = json.load(_f)
        except Exception:
            pass
        _release_data_map = _build_release_data_map(result, fred_data=_fred_data)
        _cascade_results = score_all_cascades(
            df_15m, current_time=_now_utc, config=cfg, regime=_cascade_regime,
            release_data_map=_release_data_map)
        if _cascade_results and _cascade_results.get('active_count', 0) > 0:
            result['cascade'] = {
                'combined_score': _cascade_results.get('combined_score', 0.5),
                'combined_signal': _cascade_results.get('combined_signal', 'HOLD'),
                'active_count': _cascade_results.get('active_count', 0),
                'total_weight': _cascade_results.get('total_weight', 0),
                'active_cascades': [
                    {
                        'name': c['name'],
                        'score': round(c['score'], 3),
                        'weight': c['weight'],
                        'signal': c['signal'],
                        'expected_move': c['expected_move'],
                        'confidence': c['confidence'],
                    }
                    for c in _cascade_results.get('active_cascades', [])
                ],
                'all_results': {
                    name: {'status': r.get('status'), 'score': r.get('score', 0.5)}
                    for name, r in _cascade_results.get('all_results', {}).items()
                },
                'details': _cascade_results,
            }
            # Store individual cascade details for formatting
            for _ac in _cascade_results.get('active_cascades', []):
                result[f'cascade_{_ac["name"].lower()}'] = _ac.get('details', {})
    except Exception as e:
        result['cascade'] = {'status': 'ERROR', 'combined_score': 0.5, 'error': str(e)}

    # ── Macro Lifecycle (event cascade tracking) ──
    try:
        lifecycle_state = evaluate_macro_lifecycle(df_15m, config=cfg)
        if lifecycle_state:
            result['macro_lifecycle'] = {
                'phase': lifecycle_state.get('phase'),
                'release_type': lifecycle_state.get('release_type'),
                'release_date': lifecycle_state.get('release_date'),
                'hours_since': lifecycle_state.get('hours_since_release'),
                'trade_action': lifecycle_state.get('trade_action'),
                'size_multiplier': lifecycle_state.get('size_multiplier', 1.0),
                'session_data': {
                    'regime': lifecycle_state.get('session_data', {}).get('regime'),
                    'us_direction': lifecycle_state.get('session_data', {}).get('us_direction'),
                    'asia_pattern': lifecycle_state.get('session_data', {}).get('asia_pattern'),
                },
            }
            # Apply lifecycle size multiplier if not in normal mode
            _lc_action = lifecycle_state.get('trade_action', 'NORMAL')
            _lc_size = lifecycle_state.get('size_multiplier', 1.0)
            if _lc_action == 'REDUCE_SIZE' and _lc_size < 1.0:
                result['_lifecycle_size_mult'] = _lc_size
    except Exception:
        pass

    # ── Phase 4: Score all modules ──
    # M1 (now an ICS contributor, not the gate)
    m1_dir, m1_score, _m1_details = score_m1(df_1h, idx_1h, cfg, df_15m=df_15m, idx_15m=idx)
    # Direction-aware scoring: flip M1 score when direction disagrees with trade
    # M1 returns score 0.5-1.0 (distance from neutral), but doesn't encode trade direction.
    # If M1 says BEARISH and trade is LONG, high score should penalize, not boost.
    if m1_dir == 'BEARISH' and direction == 'LONG':
        m1_score = 1.0 - m1_score
    elif m1_dir == 'BULLISH' and direction == 'SHORT':
        m1_score = 1.0 - m1_score
    result['m1'] = {'direction': m1_dir, 'score': round(float(m1_score), 3)}

    # M2
    m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)
    result['m2'] = {'status': m2_status, 'score': round(float(m2_score), 3)}

    # M2 Veto
    if cfg.get('M2_VETO_ENABLED', False):
        m2_veto_thresh = cfg.get('M2_VETO_THRESHOLD', 0.40)
        if direction == 'LONG' and m2_status == 'BEARISH' and m2_score < m2_veto_thresh:
            result['status'] = 'NO_SIGNAL'
            result['reason'] = f'M2 veto: {m2_status} score={m2_score:.3f}'
            return result
        if direction == 'SHORT' and m2_status == 'BULLISH' and m2_score < m2_veto_thresh:
            result['status'] = 'NO_SIGNAL'
            result['reason'] = f'M2 veto: {m2_status} score={m2_score:.3f}'
            return result

    # M3
    m3_status, m3_score, _ = score_m3(df_15m, idx, direction, cfg)
    result['m3'] = {'status': m3_status, 'score': round(float(m3_score), 3)}

    # M4
    m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, idx_2h, direction, cfg)
    result['m4'] = {'status': m4_status, 'score': round(float(m4_score), 3), 'div': m4_div}

    # M4b blend into M4: if intrabar catches div that taker CVD missed
    if m4b_divergence != 'NONE':
        m4_div_detected = m4_div.get('layer_a_div', 'NONE') if isinstance(m4_div, dict) else 'NONE'
        if m4_div_detected == 'NONE':
            m4_score = m4b_score
            if isinstance(m4_div, dict):
                m4_div['intrabar_div'] = m4b_divergence
                m4_div['intrabar_source'] = 'lucf_style'
            print(f"  📊 Intrabar CVD override: {m4b_divergence} div ({m4b_details.get('bars_ago', '?')} bars ago)")

    # ── Squeeze 5-filter confirmation gate (backtested: 84.6% WR on 4h) ──
    # Runs AFTER M4b so CVD filter reads fresh intrabar data.
    squeeze_confirmed = False
    squeeze_filters = {}
    if squeeze_result['squeeze_type'] != 'NONE' and squeeze_result['direction'] != 'NEUTRAL':
        sq_dir = squeeze_result['direction']

        # Filter 1: EMA trend — REGIME-ADAPTIVE (2026-05-06)
        if cfg.get('SQUEEZE_CONFIRM_EMA', True) and len(df_15m) >= 55:
            _close = df_15m['Close']
            _ema21 = float(_close.ewm(span=21, adjust=False).mean().iloc[-1])
            _ema55 = float(_close.ewm(span=55, adjust=False).mean().iloc[-1])
            _ema_spread = (_ema21 - _ema55) / _ema55 * 100 if _ema55 > 0 else 0
            _ema_trend = 'BULL' if _ema21 > _ema55 else 'BEAR'
            _contrarian = (sq_dir == 'LONG' and _ema_trend == 'BEAR') or \
                          (sq_dir == 'SHORT' and _ema_trend == 'BULL')
            _aligned = not _contrarian
            if _ema_spread < 0:
                squeeze_filters['ema_regime'] = _contrarian
            else:
                squeeze_filters['ema_regime'] = _aligned
        else:
            squeeze_filters['ema_regime'] = True

        # Filter 2: CVD divergence agrees (now reads fresh M4b data)
        if cfg.get('SQUEEZE_CONFIRM_CVD', True):
            _m4b_div = result.get('m4b', {}).get('divergence', 'NONE')
            squeeze_filters['cvd_agrees'] = not ((sq_dir == 'LONG' and _m4b_div == 'BEARISH') or
                                                 (sq_dir == 'SHORT' and _m4b_div == 'BULLISH'))
        else:
            squeeze_filters['cvd_agrees'] = True

        # Filter 3: RSI not extreme against direction
        if cfg.get('SQUEEZE_CONFIRM_RSI', True) and 'rsi' in df_15m.columns:
            _rsi = float(df_15m['rsi'].iloc[-1]) if not pd.isna(df_15m['rsi'].iloc[-1]) else 50
            squeeze_filters['rsi_ok'] = (sq_dir == 'LONG' and _rsi < 75) or \
                                         (sq_dir == 'SHORT' and _rsi > 25)
        else:
            squeeze_filters['rsi_ok'] = True

        # Filter 4: Quality score >= 0.5
        if cfg.get('SQUEEZE_CONFIRM_QUALITY', True):
            squeeze_filters['quality_high'] = squeeze_result.get('squeeze_score', 0) >= 0.5
        else:
            squeeze_filters['quality_high'] = True

        # Filter 5: ATR floor — skip signals when vol is too low for a real move
        if cfg.get('SQUEEZE_CONFIRM_ATR_FLOOR', True):
            _atr_now = float(df_15m['atr'].iloc[-1]) if 'atr' in df_15m.columns and not pd.isna(df_15m['atr'].iloc[-1]) else 0
            _atr_hard_floor = cfg.get('SQUEEZE_MIN_ATR', 5.0)
            _atr_lookback = cfg.get('SQUEEZE_ATR_LOOKBACK', 8640)
            _atr_pctile = cfg.get('SQUEEZE_ATR_FLOOR_PCTILE', 15)
            _atr_series = df_15m['atr'].dropna()
            if len(_atr_series) > 100:
                _atr_window = _atr_series.iloc[-min(len(_atr_series), _atr_lookback):]
                _atr_threshold = float(np.percentile(_atr_window, _atr_pctile))
            else:
                _atr_threshold = _atr_hard_floor
            _atr_effective = max(_atr_hard_floor, _atr_threshold)
            squeeze_filters['atr_floor'] = _atr_now >= _atr_effective
            squeeze_filters['_atr_value'] = round(_atr_now, 2)
            squeeze_filters['_atr_threshold'] = round(_atr_effective, 2)
        else:
            squeeze_filters['atr_floor'] = True

        squeeze_confirmed = all(squeeze_filters.values())

    result['squeeze_filters'] = squeeze_filters
    result['squeeze_confirmed'] = squeeze_confirmed

    # Only apply regime override + ICS boost if squeeze is CONFIRMED
    if squeeze_result['squeeze_status'] == 'TRIGGERED' and squeeze_confirmed and squeeze_result.get('overrides_regime'):
        regime_blocked = False
        result['regime_blocked'] = False
        result['squeeze_override'] = True

    # M5
    m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction, cfg,
        n_bins=cfg['M5_VP_BINS'], lookback=cfg['M5_VP_LOOKBACK'])
    result['m5'] = {'status': m5_status, 'score': round(float(m5_score), 3)}

    # M5 Regime Gate — neutralize M5 in unfavorable regimes (forensic P1)
    if cfg.get('M5_REGIME_GATE_ENABLED', False):
        _m5_favorable = ('NEUTRAL', 'TRENDING', 'CHOP_MILD_BEAR')
        if vol_regime not in _m5_favorable:
            m5_score = 0.5

    # M8 (funding)
    m8_score = 0.5
    m8_status = 'SKIP'
    if cfg.get('M8_ENABLED', False):
        try:
            from src.modules.m6_derivatives import fetch_funding_rate
            fr_df = fetch_funding_rate("ETHUSDT", limit=1)
            if fr_df is not None and len(fr_df) > 0:
                fr = float(fr_df.iloc[-1].get('funding_rate', fr_df.iloc[-1].get('lastFundingRate', np.nan)))
                if not np.isnan(fr):
                    m8_status, m8_score, _ = score_m8_funding(fr, direction, cfg)
                    result['m8'] = {'status': m8_status, 'score': round(float(m8_score), 3), 'rate': round(fr, 6)}
        except Exception:
            pass

    # M10 (cross-asset macro) — BTC trend + ETH/BTC relative strength
    m10_score = 0.5
    m10_status = 'SKIP'
    m10_details = {}
    m10_data = None
    if cfg.get('M10_ENABLED', False):
        try:
            m10_data = m10_prepare_data(df_15m)
            if m10_data:
                m10_data = m10_compute_emas(m10_data)
                macro_row = m10_get_row(m10_data, ts)
                if macro_row:
                    m10_status, m10_score, m10_details = score_m10_macro(macro_row, direction, trend_dir)
                    result['m10'] = {'status': m10_status, 'score': round(float(m10_score), 3), 'details': m10_details}
        except Exception as e:
            result['m10'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # M11 (MTF momentum)
    m11_score = 0.5
    m11_status = 'SKIP'
    if cfg.get('M11_ENABLED', False):
        try:
            from src.modules.m11_momentum import score_m11_mtf_momentum
            m11_status, m11_score, _ = score_m11_mtf_momentum(
                df_15m, df_1h, df_4h, idx, idx_1h, idx_4h, direction)
            result['m11'] = {'status': m11_status, 'score': round(float(m11_score), 3)}
        except Exception:
            pass

    # M12 (order book imbalance) — live only
    m12_score = 0.5
    m12_status = 'SKIP'
    if cfg.get('M12_ENABLED', False) and cfg.get('M12_LIVE_ONLY', True):
        try:
            m12_status, m12_score, m12_details = score_m12_orderbook(direction, live=True)
            result['m12'] = {'status': m12_status, 'score': round(float(m12_score), 3), 'details': m12_details}
        except Exception as e:
            result['m12'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # M14 (sweep-retest-reclaim)
    m14_score = 0.5
    m14_status = 'SKIP'
    if cfg.get('M14_ENABLED', True):
        _swing_levels = m13_details.get('swing_lows', []) if direction == 'LONG' else m13_details.get('swing_highs', [])
        if _swing_levels:
            m14_status, m14_score, _ = score_m14(df_15m, idx, direction, _swing_levels, config=cfg, magnets=magnets)
            result['m14'] = {'status': m14_status, 'score': round(float(m14_score), 3)}

    # M17 (resistance quality) — validate nearest S/R level
    m17_score = 0.5
    m17_status = 'SKIP'
    m17_result = None
    if cfg.get('M17_ENABLED', True) and sr_levels:
        if direction == 'LONG':
            resistances = [sr for sr in sr_levels if sr[4] == 'RESISTANCE']
            if resistances:
                nearest_res = min(resistances, key=lambda x: abs(x[0] - current_price))
                m17_result = score_resistance_quality(
                    nearest_res[0], df_15m, idx, bin_centers, vol_profile,
                    result.get('derivatives', {}), 'LONG', config=cfg)
        elif direction == 'SHORT':
            supports = [sr for sr in sr_levels if sr[4] == 'SUPPORT']
            if supports:
                nearest_sup = min(supports, key=lambda x: abs(x[0] - current_price))
                m17_result = score_resistance_quality(
                    nearest_sup[0], df_15m, idx, bin_centers, vol_profile,
                    result.get('derivatives', {}), 'SHORT', config=cfg)
        if m17_result:
            m17_score = m17_result['composite']
            m17_status = 'PASS'
            result['m17'] = {**m17_result, 'status': m17_status, 'score': round(float(m17_score), 4)}

    # ── M20: Failed Breakout Detector ──
    m20_score = 0.5
    m20_status = 'SKIP'
    m20_result = None
    if cfg.get('M20_ENABLED', True):
        try:
            _sr_for_m20 = sr_levels if sr_levels else None
            _mag_for_m20 = magnets if magnets else None
            m20_status, m20_score, m20_result = score_m20(
                df_15m, idx, direction,
                sr_levels=_sr_for_m20, magnets=_mag_for_m20,
                config=cfg, atr_1h=atr_1h)
            if m20_result:
                result['m20'] = {**m20_result, 'status': m20_status, 'score': round(float(m20_score), 4)}
        except Exception as e:
            result['m20'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # TAKER: Taker flow momentum + regime scoring
    taker_score = 0.5
    use_taker = False
    taker_data = result.get('taker_summary')
    if cfg.get('TAKER_ENABLED', False) and taker_data:
        _taker_dir = taker_data.get('direction', 'NEUTRAL')
        _taker_sc = taker_data.get('score', 0)
        if _taker_dir == 'LONG':
            taker_score = 0.5 + _taker_sc * 0.5
            use_taker = True
        elif _taker_dir == 'SHORT':
            taker_score = 0.5 - _taker_sc * 0.5
            use_taker = True
        else:
            taker_score = 0.5
            use_taker = True  # include neutral

    # ── ICS ──
    # Extract cascade params from M5 details (aligned with engine)
    cascade_dir = m5_details.get('cascade_dir', 'NONE') if isinstance(m5_details, dict) else 'NONE'
    cascade_strength = m5_details.get('cascade_strength', 0.0) if isinstance(m5_details, dict) else 0.0

    # Cross-asset scoring (aligned with engine)
    cross_asset_score = 0.5
    use_cross_asset = False
    if cfg.get('CROSS_ASSET_ENABLED', False) and btc_15m_df is not None and btc_corr_series is not None:
        btc_corr_val = btc_corr_series.iloc[-1] if len(btc_corr_series) > 0 else 0.5
        btc_change = 0.0
        if len(btc_15m_df) > 4:
            btc_close_now = float(btc_15m_df['Close'].iloc[-1])
            btc_close_1h_ago = float(btc_15m_df['Close'].iloc[max(0, len(btc_15m_df) - 5)])
            if btc_close_1h_ago > 0:
                btc_change = (btc_close_now - btc_close_1h_ago) / btc_close_1h_ago
        cross_asset_score = score_cross_asset(
            float(df_15m['Close'].iloc[-1]), btc_close_now, btc_corr_val, btc_change, direction)
        use_cross_asset = True

    ics, effective_floor = calc_ics(
        m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
        m7_score=m7_score, m8_score=m8_score, cross_asset_score=cross_asset_score,
        use_m7=cfg.get('M7_ENABLED', False) and m7_ethbtc_df is not None,
        use_m8=m8_status != 'SKIP', use_cross_asset=use_cross_asset,
        cascade_dir=cascade_dir, cascade_strength=cascade_strength,
        m9_score=m9_score, use_m9=True,
        m10_score=m10_score, use_m10=m10_status != 'SKIP',
        m11_score=m11_score, use_m11=m11_status != 'SKIP',
        m12_score=m12_score, use_m12=m12_status != 'SKIP',
        m13_score=m13_score, use_m13=cfg.get('M13_ENABLED', False),
        m14_score=m14_score, use_m14=m14_status == 'PASS',
        m17_score=m17_score, use_m17=m17_status == 'PASS',
        m20_score=m20_score, use_m20=m20_status == 'PASS',
        m22_score=m22_score, use_m22=m22_status != 'SKIP',
        taker_score=taker_score, use_taker=use_taker,
        m66_score=m66_score, use_m66=m66_status == 'PASS',
        m67_score=m67_score, use_m67=m67_status == 'PASS',
        m68_score=m68_score, use_m68=m68_status == 'PASS',
        m69_score=m69_score, use_m69=m69_status == 'PASS',
        m70_score=m70_score, use_m70=m70_status == 'PASS',
        m71_score=m71_score, use_m71=m71_status == 'PASS',
        m72_score=m72_score, use_m72=m72_status == 'PASS',
        m73_score=m73_score, use_m73=m73_status == 'PASS',
        config=cfg,
    )

    # M5 sweet-spot boost (aligned with engine)
    m5_spot_low = cfg.get('M5_SWEET_SPOT_LOW', 0.30)
    m5_spot_high = cfg.get('M5_SWEET_SPOT_HIGH', 0.50)
    if m5_spot_low <= m5_score <= m5_spot_high:
        ics += cfg.get('M5_SWEET_SPOT_BOOST', 0.04)

    result['ics'] = round(float(ics), 4)
    result['effective_floor'] = round(float(effective_floor), 4)
    result['taker_score'] = round(float(taker_score), 4)
    if use_cross_asset:
        result['cross_asset'] = {'score': round(float(cross_asset_score), 3)}

    # ── Squeeze ICS boost (only when TRIGGERED + CONFIRMED) ──
    if squeeze_result['squeeze_status'] == 'TRIGGERED' and squeeze_confirmed and squeeze_result['ics_boost'] > 0:
        ics += squeeze_result['ics_boost']
        result['ics'] = round(float(ics), 4)
        result['squeeze_ics_boost'] = squeeze_result['ics_boost']

    # ── M24 NBS PMI ICS adjustment (regime-conditional bias on release days) ──
    if m24_score_adj != 0.0 and m24_status in ('PASS', 'WEAK'):
        ics += m24_score_adj
        ics = max(0.0, min(1.0, ics))  # clamp to [0, 1]
        result['ics'] = round(float(ics), 4)
        result['m24_ics_adj'] = m24_score_adj

    # ── M25 Caixin PMI ICS adjustment (regime-conditional bias on release days) ──
    if m25_score_adj != 0.0 and m25_status == 'PASS':
        ics += m25_score_adj
        ics = max(0.0, min(1.0, ics))  # clamp to [0, 1]
        result['ics'] = round(float(ics), 4)
        result['m25_ics_adj'] = m25_score_adj

    # ── M26 EZ PMI ICS adjustment (session-conditional bias on release days) ──
    if m26_score_adj != 0.0 and m26_status in ('PASS', 'WEAK'):
        ics += m26_score_adj
        ics = max(0.0, min(1.0, ics))  # clamp to [0, 1]
        result['ics'] = round(float(ics), 4)
        result['m26_ics_adj'] = m26_score_adj

    # ── M33 US Retail Sales ICS adjustment (regime-conditional bias on release days) ──
    if m33_score_adj != 0.0 and m33_status in ('PASS', 'WEAK'):
        ics += m33_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m33_ics_adj'] = m33_score_adj

    # ── M34 US Housing ICS adjustment (regime-conditional bias on release days) ──
    if m34_score_adj != 0.0 and m34_status in ('PASS', 'WEAK'):
        ics += m34_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m34_ics_adj'] = m34_score_adj

    # ── M35 PBoC LPR ICS adjustment (regime-conditional bias on release days) ──
    if m35_score_adj != 0.0 and m35_status in ('PASS', 'WEAK'):
        ics += m35_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m35_ics_adj'] = m35_score_adj

    # ── M36 ADP Employment ICS adjustment (regime-conditional bias on release days) ──
    if m36_score_adj != 0.0 and m36_status in ('PASS', 'WEAK'):
        ics += m36_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m36_ics_adj'] = m36_score_adj

    # ── M37 NFP ICS adjustment (regime-conditional bias on release days) ──
    if m37_score_adj != 0.0 and m37_status in ('PASS', 'WEAK'):
        ics += m37_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m37_ics_adj'] = m37_score_adj

    # ── M38 Ifo ICS adjustment (regime-conditional bias on release days) ──
    if m38_score_adj != 0.0 and m38_status in ('PASS', 'WEAK'):
        ics += m38_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m38_ics_adj'] = m38_score_adj

    # ── M39 UMS ICS adjustment (regime-conditional bias on release days) ──
    if m39_score_adj != 0.0 and m39_status in ('PASS', 'WEAK'):
        ics += m39_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m39_ics_adj'] = m39_score_adj

    # ── M40 Germany CPI ICS adjustment (regime-conditional bias on release days) ──
    if m40_score_adj != 0.0 and m40_status in ('PASS', 'WEAK'):
        ics += m40_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m40_ics_adj'] = m40_score_adj

    # ── M41 EZ CPI Flash ICS adjustment (regime-conditional bias on release days) ──
    if m41_score_adj != 0.0 and m41_status in ('PASS', 'WEAK'):
        ics += m41_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m41_ics_adj'] = m41_score_adj

    # ── M42 EZ GDP Flash ICS adjustment (regime-conditional bias on release days) ──
    if m42_score_adj != 0.0 and m42_status in ('PASS', 'WEAK'):
        ics += m42_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m42_ics_adj'] = m42_score_adj

    # ── M43 US GDP Advance ICS adjustment (contrarian: miss→rally, beat→sell) ──
    if m43_score_adj != 0.0 and m43_status in ('PASS', 'WEAK'):
        ics += m43_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m43_ics_adj'] = m43_score_adj

    # ── M44 US Durable Goods ICS adjustment (regime-conditional bias) ──
    if m44_score_adj != 0.0 and m44_status in ('PASS', 'WEAK'):
        ics += m44_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m44_ics_adj'] = m44_score_adj

    # ── M45 Core PCE ICS adjustment (Fed's preferred gauge, Friday weekend lock) ──
    if m45_score_adj != 0.0 and m45_status in ('PASS', 'WEAK'):
        ics += m45_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m45_ics_adj'] = m45_score_adj

    # ── M46 Japan CPI ICS adjustment (BoJ metric, carry-trade de-risking) ──
    if m46_score_adj != 0.0 and m46_status in ('PASS', 'WEAK'):
        ics += m46_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m46_ics_adj'] = m46_score_adj

    # ── M47 BoJ Rate ICS adjustment (carry-trade regime, YCC impact) ──
    if m47_score_adj != 0.0 and m47_status in ('PASS', 'WEAK'):
        ics += m47_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m47_ics_adj'] = m47_score_adj

    # ── M48 ECB Rate ICS adjustment (DXY repricing, EUR/USD impact) ──
    if m48_score_adj != 0.0 and m48_status in ('PASS', 'WEAK'):
        ics += m48_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m48_ics_adj'] = m48_score_adj

    # ── M49 BoE Rate ICS adjustment (GBP→DXY→ETH mechanism) ──
    if m49_score_adj != 0.0 and m49_status in ('PASS', 'WEAK'):
        ics += m49_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m49_ics_adj'] = m49_score_adj

    # ── M50 CB Consumer Confidence ICS adjustment ──
    if m50_score_adj != 0.0 and m50_status == 'ACTIVE':
        ics += m50_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m50_ics_adj'] = m50_score_adj

    # ── M51 UK Monthly GDP ICS adjustment ──
    if m51_score_adj != 0.0 and m51_status == 'ACTIVE':
        ics += m51_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m51_ics_adj'] = m51_score_adj

    # ── M52 RBA Rate ICS adjustment ──
    if m52_score_adj != 0.0 and m52_status == 'ACTIVE':
        ics += m52_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m52_ics_adj'] = m52_score_adj

    # ── M53 Australia CPI ICS adjustment ──
    if m53_score_adj != 0.0 and m53_status == 'ACTIVE':
        ics += m53_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m53_ics_adj'] = m53_score_adj

    # ── M54 China GDP ICS adjustment ──
    if m54_score_adj != 0.0 and m54_status == 'ACTIVE':
        ics += m54_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m54_ics_adj'] = m54_score_adj

    # ── M55 Treasury Auction ICS adjustment ──
    if m55_score_adj != 0.0 and m55_status == 'ACTIVE':
        ics += m55_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m55_ics_adj'] = m55_score_adj

    # ── M56 US CPI ICS adjustment ──
    if m56_score_adj != 0.0 and m56_status in ('ACTIVE', 'WEAK'):
        ics += m56_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m56_ics_adj'] = m56_score_adj

    # ── M57 FOMC Rate Decision ICS adjustment ──
    if m57_score_adj != 0.0 and m57_status in ('ACTIVE', 'WEAK'):
        ics += m57_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m57_ics_adj'] = m57_score_adj

    # ── M58 Powell Presser ICS adjustment ──
    if m58_score_adj != 0.0 and m58_status in ('ACTIVE', 'WEAK'):
        ics += m58_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m58_ics_adj'] = m58_score_adj

    # ── M59 FOMC Minutes ICS adjustment ──
    if m59_score_adj != 0.0 and m59_status in ('ACTIVE', 'WEAK'):
        ics += m59_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m59_ics_adj'] = m59_score_adj

    # ── M60 US PPI ICS adjustment ──
    if m60_score_adj != 0.0 and m60_status in ('PASS', 'WEAK'):
        ics += m60_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m60_ics_adj'] = m60_score_adj

    # ── M61 US Jobless Claims ICS adjustment (regime-conditional bias on Thursdays) ──
    if m61_score_adj != 0.0 and m61_status in ('PASS', 'WEAK'):
        ics += m61_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m61_ics_adj'] = m61_score_adj
    if m62_score_adj != 0.0 and m62_status in ('PASS', 'WEAK'):
        ics += m62_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m62_ics_adj'] = m62_score_adj

    # ── M65 China Activity ICS adjustment (regime-conditional on release days) ──
    if m65_score_adj != 0.0 and m65_status in ('PASS', 'WEAK'):
        ics += m65_score_adj
        ics = max(0.0, min(1.0, ics))
        result['ics'] = round(float(ics), 4)
        result['m65_ics_adj'] = m65_score_adj

    # ── Phase 5: Veto + Coherence + Filters ──
    # Veto
    if cfg.get('VETO_ENABLED', False):
        # Extract actual divergence string from M4 details dict
        m4_div_str = 'NONE'
        if isinstance(m4_div, dict):
            m4_div_str = m4_div.get('layer_a_div', 'NONE')
        elif isinstance(m4_div, str):
            m4_div_str = m4_div
        # v7.2: Normalize _BASE variants (BULLISH_BASE → BULLISH, etc.)
        if m4_div_str.endswith('_BASE'):
            m4_div_str = m4_div_str.replace('_BASE', '')
        # Also consider intrabar divergence
        if m4_div_str == 'NONE' and m4b_divergence != 'NONE':
            m4_div_str = m4b_divergence

        m4_disagree = (direction == 'LONG' and m4_div_str == 'BEARISH') or \
                      (direction == 'SHORT' and m4_div_str == 'BULLISH')
        m5_disagree = (m5_status == 'FAIL')
        dir_veto = m4_disagree and m5_disagree

        veto = evaluate_vetoes(
            cfg, vol_regime=vol_regime,
            dir_veto=dir_veto,
            m9_status=m9_status, m10_status=m10_status, m11_status=m11_status,
        )
        if veto.hard_blocked:
            result['status'] = 'NO_SIGNAL'
            result['reason'] = f'Veto: {veto.summary()}'
            result['veto'] = veto.summary()
            return result
        result['veto'] = veto.summary() if veto.soft_vetoes else 'CLEAR'

    # Coherence check
    # In chop regimes, suppress M13 bias for coherence (M13 is anti-predictive
    # when it agrees with M9 during chop) — aligned with engine behavior
    # _in_chop already computed above during M13 re-scoring
    _coherence_m13_bias = 'NEUTRAL' if _in_chop else m13_bias

    if cfg.get('COHERENCE_CHECK_ENABLED', True):
        is_coherent, conflicts, coherence_penalty = check_coherence(
            direction, m4_div_str, m5_details if isinstance(m5_details, dict) else {},
            _coherence_m13_bias, vol_regime, m7_score=m7_score, m2_status=m2_status, config=cfg,
        )
        if not is_coherent:
            result['status'] = 'NO_SIGNAL'
            result['reason'] = f'Coherence block: {", ".join(conflicts)}'
            return result
        ics -= coherence_penalty
        result['ics'] = round(float(ics), 4)

    # Threshold
    threshold = cfg['ICS_THRESHOLD_CAUTION'] if phase0_val and phase0_val >= 0.40 else cfg['ICS_THRESHOLD_NORMAL']
    result['threshold'] = round(float(threshold), 4)

    # M3 hard fail — but not when M20 direct signal is active
    # (failed breakouts naturally have poor VWAP scores)
    m20_override_active_for_m3 = dir_details.get('m20_override') is not None
    m20_strong_for_m3 = m20_status == 'PASS' and m20_score >= cfg.get('M20_DIRECT_SIGNAL_THRESHOLD', 0.85)
    if m3_status == 'FAIL' and not (m20_override_active_for_m3 and m20_strong_for_m3):
        result['status'] = 'NO_SIGNAL'
        result['reason'] = 'M3 VWAP fail'
        return result

    # ICS check
    if ics < effective_floor or ics < threshold:
        # ── M20 Direct Signal Path ──
        # When normal ICS fails but M20 detected a strong failed breakout
        # that overrode the direction, allow M20 to generate a signal directly.
        # Failed breakouts are high-conviction contrarian events that the normal
        # module pipeline doesn't score well (M3/M4/M13 work against the flip).
        m20_direct_threshold = cfg.get('M20_DIRECT_SIGNAL_THRESHOLD', 0.85)
        m20_override_active = dir_details.get('m20_override') is not None
        if (m20_status == 'PASS' and m20_score >= m20_direct_threshold and
                m20_result and m20_result.get('status') == 'FAILED' and
                m20_override_active):
            # M20 strong failed breakout — bypass ICS gate
            result['m20_direct_signal'] = True
            result['m20_direct_score'] = round(float(m20_score), 4)
            result['ics_bypassed_by_m20'] = True
            print(f"  💥 M20 DIRECT SIGNAL: failed breakout score={m20_score:.3f} ≥ {m20_direct_threshold} — bypassing ICS gate")
            # Continue to trade level computation below (don't return NO_SIGNAL)
        else:
            result['status'] = 'NO_SIGNAL'
            result['reason'] = f'ICS {ics:.4f} < threshold {threshold:.2f}'
            return result

    # Gatekeepers
    gatekeeper = run_gatekeepers(
        direction, vol_regime, m7_score, m7_status, m7_details,
        m9_score, m9_status, m10_score, m10_status, trend_dir, config=cfg,
    )
    if not gatekeeper.passed:
        result['status'] = 'NO_SIGNAL'
        result['reason'] = f'Gatekeeper: {gatekeeper.summary()}'
        return result
    # Apply gatekeeper ICS boost (e.g., M7 strong agree) — aligned with engine
    ics += gatekeeper.ics_boost
    result['ics'] = round(float(ics), 4)

    # ── Compute trade levels (always, even on NO_SIGNAL) ──
    entry_price = float(row['Close'])
    atr_for_sl = float(atr_1h) if not pd.isna(atr_1h) else float(row['atr'])
    _liq_for_levels = result.get('liquidity_levels')

    # ── Limit Entry: find better entry at nearest support/resistance ──
    limit_entry = calc_limit_entry(
        entry_price, direction, magnets, sr_levels,
        atr_1h=atr_for_sl, cfg=cfg)
    result['limit_entry'] = limit_entry
    # Use limit entry price if available, otherwise market price
    effective_entry = limit_entry['entry_price'] if limit_entry['entry_source'] != 'MARKET' else entry_price

    levels = calc_trade_levels(
        effective_entry, direction, atr_for_sl,
        row.get('vol_ratio', np.nan),
        magnets=magnets,
        sr_levels=sr_levels,
        liq_levels=_liq_for_levels,
        cfg=cfg,
    )

    # ── Range-Aware TP/SL Override ──
    # When M21 detects a trading range, override TP/SL with structure-based levels
    if cfg.get('RANGE_TP_ENABLED', True) and range_info_m21:
        range_width_pct = range_info_m21.get('width_pct', 0)
        min_range = cfg.get('STRUCTURE_TP_MIN_RANGE_PCT', 1.5)
        max_range = cfg.get('STRUCTURE_TP_MAX_RANGE_PCT', 6.0)
        if min_range <= range_width_pct <= max_range:
            range_targets = get_range_targets(effective_entry, direction, range_info_m21, magnets)
            if range_targets:
                levels['tp1'] = range_targets['tp1']
                levels['tp2'] = range_targets['tp2']
                levels['tp3'] = range_targets['tp3']
                levels['tp1_source'] = range_targets['tp1_source']
                levels['tp2_source'] = range_targets['tp2_source']
                levels['tp3_source'] = range_targets['tp3_source']
                levels['tp1_pct'] = range_targets['tp1_pct']
                result['range_tp_override'] = True

    if cfg.get('RANGE_SL_ENABLED', True) and range_info_m21:
        range_sl = get_range_sl(effective_entry, direction, range_info_m21, atr_for_sl)
        if range_sl:
            levels['sl'] = range_sl['sl']
            levels['sl_pct'] = range_sl['sl_pct']
            levels['sl_source'] = range_sl['sl_source']
            result['range_sl_override'] = True

    # ── Build invalidation conditions ──
    invalidation = []

    # Bearish divergence active
    m4_div_str = 'NONE'
    if isinstance(m4_div, dict):
        m4_div_str = m4_div.get('layer_a_div', 'NONE')
    # v7.2: Normalize _BASE variants
    if m4_div_str.endswith('_BASE'):
        m4_div_str = m4_div_str.replace('_BASE', '')
    if m4_div_str == 'NONE' and m4b_divergence != 'NONE':
        m4_div_str = m4b_divergence
    if (direction == 'LONG' and m4_div_str == 'BEARISH') or \
       (direction == 'SHORT' and m4_div_str == 'BULLISH'):
        invalidation.append(f'{m4_div_str} CVD divergence active — momentum against you')

    # Whales leaning against
    whale = result.get('derivatives', {}).get('whale_signal', 'NEUTRAL')
    if (direction == 'LONG' and whale == 'WHALE_BEARISH') or \
       (direction == 'SHORT' and whale == 'WHALE_BULLISH'):
        invalidation.append(f'Whales {whale.replace("WHALE_", "").lower()} — smart money against direction')

    # Crowded positioning
    pos = result.get('derivatives', {}).get('positioning', 'NEUTRAL')
    if (direction == 'LONG' and pos == 'CROWDED_LONG') or \
       (direction == 'SHORT' and pos == 'CROWDED_SHORT'):
        invalidation.append(f'Crowded {direction.lower()} positioning — squeeze risk')

    # Conflict history
    conflict = result.get('conflict')
    if conflict and conflict.get('is_conflict'):
        invalidation.append(f'Historical conflict: similar setups reverse {conflict["historical"]["windows"].get("24h", {}).get("reversal_rate", 0):.0f}% at 24h')

    # Phase0 death zone
    phase0_max = cfg.get('PHASE0_MAX_BLOCK', 0.30)
    if phase0_val is not None and phase0_val > phase0_max:
        invalidation.append(f'Phase0={phase0_val:.3f} (crowded zone >{phase0_max}) — high positioning noise, reduced edge')

    # M2 EMA failure
    if m2_status == 'FAIL':
        invalidation.append('M2 EMA confluence FAIL — multi-TF trend disagreement')

    # Price below key S/R
    supports = [p for p, s, t, _, _ in sr_levels if t == 'SUPPORT']
    resistances = [p for p, s, t, _, _ in sr_levels if t == 'RESISTANCE']
    nearest_support = min(supports, key=lambda x: abs(x - entry_price)) if supports else 0
    nearest_resist = min(resistances, key=lambda x: abs(x - entry_price)) if resistances else 0

    result['what_if'] = {
        'direction': direction,
        'entry': effective_entry,
        'entry_source': limit_entry['entry_source'],
        'entry_reason': limit_entry['reason'],
        'market_entry': entry_price,
        'sl': levels['sl'],
        'tp1': levels['tp1'],
        'tp2': levels['tp2'],
        'tp3': levels['tp3'],
        'sl_pct': levels['sl_pct'],
        'tp1_pct': levels['tp1_pct'],
        'sl_source': levels['sl_source'],
        'tp1_source': levels['tp1_source'],
        'tp2_source': levels['tp2_source'],
        'tp3_source': levels['tp3_source'],
        'rr1': abs(levels['tp1_pct'] / levels['sl_pct']) if levels['sl_pct'] != 0 else 0,
        'nearest_support': round(nearest_support, 2),
        'nearest_resist': round(nearest_resist, 2),
        'invalidation': invalidation,
    }

    # ── Entry filters ──
    # Bypass for M20 direct signals and squeeze-triggered entries
    _m20_direct = result.get('m20_direct_signal', False)
    _squeeze_active_for_filters = squeeze_result['squeeze_status'] == 'TRIGGERED' and squeeze_confirmed

    # ── Macro Event Pre-Filter (backtested Phase0 + trend + cascade) ──
    if not _m20_direct and not _squeeze_active_for_filters and cfg.get('MACRO_EVENT_FILTER_ENABLED', True):
        _phase0_for_mef = phase0_val
        _trend_30d_for_mef = get_trend_30d(df_15m, idx)
        _mef = MacroEventFilter(config=cfg)
        _mef_result = _mef.check(ts, df_15m, idx, direction,
                                  phase0=_phase0_for_mef, trend_30d=_trend_30d_for_mef)
        if _mef_result['blocked']:
            result['status'] = 'FILTERED'
            result['reason'] = _mef_result['reason']
            result['macro_event_filter'] = _mef_result
            return result
        if _mef_result['size_mult'] < 1.0:
            result['macro_event_filter'] = _mef_result
            # Apply size reduction later (after signal confirmed)
        elif _mef_result['regime_notes']:
            result['macro_event_filter'] = _mef_result

    if not _m20_direct and not _squeeze_active_for_filters:
        passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h, config=cfg)
        if not passed:
            result['status'] = 'FILTERED'
            result['reason'] = reason
            return result

    # ── M14 Sweep Gate ──
    # Bypass for M20 direct signals and squeeze-triggered entries
    if not _m20_direct and not _squeeze_active_for_filters:
        sweep_passed, sweep_reason = check_sweep_gate(m14_status, m14_score, cfg)
        if not sweep_passed:
            result['status'] = 'NO_SIGNAL'
            result['reason'] = sweep_reason
            return result

    # ── Minimum R:R filter ──
    # Reject signals where TP1 risk-reward is below threshold
    min_rr = cfg.get('MIN_RR_RATIO', 0.0)
    rr1 = abs(levels['tp1_pct'] / levels['sl_pct']) if levels['sl_pct'] != 0 else 0
    if min_rr > 0 and rr1 < min_rr:
        result['status'] = 'NO_SIGNAL'
        result['reason'] = f'R:R {rr1:.2f}x < min {min_rr:.2f}x (TP1 {levels["tp1_pct"]:.2f}% vs SL {levels["sl_pct"]:.2f}%)'
        return result

    # ── Regime block override (all modules scored, but regime kills the signal) ──
    # M20 direct signals can override regime block (failed breakout = strong event)
    if result.get('regime_blocked') and not _m20_direct:
        result['status'] = 'NO_SIGNAL'
        result['reason'] = f'M9 regime={vol_regime} (blocked)'
        return result

    # ── SIGNAL: set levels ──
    result.update({
        'status': 'SIGNAL', 'entry': effective_entry,
        'market_entry': entry_price,
        'limit_entry': limit_entry,
        'sl': levels['sl'], 'tp1': levels['tp1'],
        'tp2': levels['tp2'], 'tp3': levels['tp3'],
        'sl_pct': levels['sl_pct'], 'tp1_pct': levels['tp1_pct'],
        'sl_source': levels['sl_source'],
        'tp1_source': levels['tp1_source'],
        'tp2_source': levels['tp2_source'],
        'tp3_source': levels['tp3_source'],
    })

    return result


def print_signal(result):
    """Print detailed signal analysis with all module data."""
    print("\n" + "═" * 60)
    print("  JIMI — LIVE SIGNAL SCAN")
    print("═" * 60)
    print(f"\n  Time:   {result['timestamp']}")
    print(f"  Price:  ${result['price']:.2f}")
    print(f"  Bias:   {result['swing_bias']}")
    print(f"  Phase0: {result.get('phase0', 'N/A')}")

    # Market Data
    print(f"\n  Market Data:")
    vwap = result.get('vwap')
    vwap_dist = result.get('vwap_dist_pct')
    taker = result.get('taker_ratio')
    atr = result.get('atr_1h')
    vol_r = result.get('vol_ratio')
    if vwap:
        print(f"    VWAP:           ${vwap:.2f}  ({vwap_dist:+.2f}% from price)" if vwap_dist else f"    VWAP: ${vwap:.2f}")
    if taker is not None:
        taker_label = "buyers" if taker > 0.52 else "sellers" if taker < 0.48 else "neutral"
        print(f"    Taker Ratio:    {taker:.4f}  ({taker_label}, {taker*100:.1f}% buy)")
    if atr:
        print(f"    ATR (1H):       ${atr:.2f}  ({atr/result['price']*100:.2f}% of price)")
    if vol_r:
        print(f"    Vol Ratio:      {vol_r:.2f}x  (24h vs 7d)")

    # Taker Flow Analysis
    taker_data = result.get('taker_summary')
    if taker_data:
        print()
        print(format_taker_summary(taker_data))

    # Direction Resolver
    dr = result.get('direction_resolver', {})
    print(f"\n  Direction Resolver:")
    print(f"    Regime:        {result.get('m9', {}).get('regime', '?')}  (score={result.get('m9', {}).get('score', 0):.3f})")
    print(f"    Structure:     {result.get('m13', {}).get('bias', '?')}  (score={result.get('m13', {}).get('score', 0):.3f})")
    if 'm7' in result and result['m7'].get('status') != 'SKIP':
        print(f"    Macro M7:      {result['m7']['score']:.3f}  ({result['m7']['status']})")
    # Target scores
    tgt = result.get('target_scores', {})
    if tgt:
        print(f"    Targets:       LONG={tgt.get('LONG', 0):.3f}  SHORT={tgt.get('SHORT', 0):.3f}")
        # Show top target for each direction
        for d in ('LONG', 'SHORT'):
            det = tgt.get(f'{d.lower()}_details', {})
            top = sorted(det.get('targets', []), key=lambda x: -x.get('contrib', 0))[:1]
            top_sr = sorted(det.get('sr', []), key=lambda x: -x.get('contrib', 0))[:1]
            parts = []
            if top:
                parts.append(f"HVN ${top[0]['price']:.0f} ({top[0]['dist_pct']:+.1f}%)")
            if top_sr:
                parts.append(f"S/R ${top_sr[0]['price']:.0f} ({top_sr[0]['dist_pct']:+.1f}%)")
            if parts:
                print(f"      {d:>6} best: {', '.join(parts)}")
    print(f"    → Direction:   {dr.get('direction', '?')}  size_mult={dr.get('size_mult', 0):.2f}")
    if dr.get('target_tiebreaker'):
        print(f"    🎯 Tiebreaker: {dr.get('target_tiebreaker')}")
    print(f"    Reason:        {dr.get('reason', '?')}")

    # M21: Wyckoff Phase + Premium/Discount
    if 'm21' in result and result['m21'].get('status') != 'SKIP':
        m21 = result['m21']
        m21_details = m21.get('details', {})
        if m21_details:
            print(format_m21(m21_details))
        else:
            phase_icon = {'ACCUMULATION': '🟢', 'MARKUP': '📈', 'DISTRIBUTION': '🔴',
                         'MARKDOWN': '📉', 'RANGE': '↔️'}.get(m21.get('phase'), '❓')
            print(f"    {phase_icon} Wyckoff: {m21.get('phase', '?')}  "
                  f"Zone: {m21.get('zone', '?')}  "
                  f"KillZone: {m21.get('kill_zone', '?')}  "
                  f"score={m21.get('score', 0):.3f}")

    # M22 Macro Regime (aggregated from M23-M65)
    if 'm22' in result and result['m22'].get('status') != 'SKIP':
        m22_out = format_m22_aggregated(result['m22'].get('details', {}))
        if m22_out:
            print(m22_out)

    # Cascade Meta-Aggregator (all regional cascades)
    if 'cascade' in result and result['cascade'].get('active_count', 0) > 0:
        cascade_out = format_all_cascades(result['cascade'].get('details', {}))
        if cascade_out:
            print(cascade_out)

    # M23 PPI + CPI Session
    if 'm23' in result and result['m23'].get('status') not in ('SKIP', 'NO_PPI', 'NO_DATA'):
        m23_out = format_m23(result['m23'].get('details', {}))
        if m23_out:
            print(m23_out)

    # M24 NBS PMI Session Bias
    if 'm24' in result and result['m24'].get('status') not in ('SKIP', 'NO_EDGE'):
        m24_out = format_m24(result['m24'].get('details', {}))
        if m24_out:
            print(m24_out)

    # M25 Caixin PMI Session Bias
    if 'm25' in result and result['m25'].get('status') not in ('SKIP', 'NO_DATA'):
        m25_out = format_m25(result['m25'].get('details', {}))
        if m25_out:
            print(m25_out)

    # M65 China Activity Session Bias
    if 'm65' in result and result['m65'].get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m65_out = format_m65(result['m65'].get('details', {}))
        if m65_out:
            print(m65_out)

    # M66–M73: Traditional Finance Macro Filters
    for _m_num, _m_name in [(66, 'USD/JPY'), (67, 'DXY'), (68, '10Y Yield'),
                             (69, 'VIX'), (70, 'WTI Oil'), (71, 'Gold'),
                             (72, 'BTC.D'), (73, 'Stablecoins')]:
        _m_key = f'm{_m_num}'
        if _m_key in result and result[_m_key].get('status') not in ('SKIP', 'ERROR', None):
            _m = result[_m_key]
            _det = _m.get('details', {})
            _cls = _det.get('classification', '?')
            _sc = _m.get('score', 0.5)
            _icon = '🟢' if _sc > 0.55 else '🔴' if _sc < 0.45 else '⚪'
            # Build detail string from details dict
            _parts = []
            for k, v in _det.items():
                if k != 'classification' and v is not None:
                    if isinstance(v, float):
                        _parts.append(f'{k}={v:+.2f}')
                    elif isinstance(v, bool) and v:
                        _parts.append(k)
                    elif not isinstance(v, bool):
                        _parts.append(f'{k}={v}')
            _detail_str = '  '.join(_parts[:4])
            print(f"    {_icon} M{_m_num} ({_m_name:>10}): {_cls:>24}  score={_sc:.3f}  {_detail_str}")

    # M26 Eurozone Flash PMI Session Bias
    if 'm26' in result and result['m26'].get('status') not in ('SKIP', 'NO_EDGE'):
        m26_out = format_m26(result['m26'].get('details', {}))
        if m26_out:
            print(m26_out)

    # M33 US Retail Sales Session Bias
    if 'm33' in result and result['m33'].get('status') not in ('SKIP', 'NO_EDGE'):
        m33_out = format_m33(result['m33'].get('details', {}))
        if m33_out:
            print(m33_out)

    # M34 US Housing Starts Session Bias
    if 'm34' in result and result['m34'].get('status') not in ('SKIP', 'NO_EDGE'):
        m34_out = format_m34(result['m34'].get('details', {}))
        if m34_out:
            print(m34_out)

    # M35 PBoC LPR Session Bias
    if 'm35' in result and result['m35'].get('status') not in ('SKIP', 'NO_EDGE'):
        m35_out = format_m35(result['m35'].get('details', {}))
        if m35_out:
            print(m35_out)

    # M36 ADP Employment Session Bias
    if 'm36' in result and result['m36'].get('status') not in ('SKIP', 'NO_EDGE'):
        m36_out = format_m36(result['m36'].get('details', {}))
        if m36_out:
            print(m36_out)

    # M37 NFP Session Bias
    if 'm37' in result and result['m37'].get('status') not in ('SKIP', 'NO_EDGE'):
        m37_out = format_m37(result['m37'].get('details', {}))
        if m37_out:
            print(m37_out)

    # M38 Ifo Session Bias
    if 'm38' in result and result['m38'].get('status') not in ('SKIP', 'NO_EDGE'):
        m38_out = format_m38(result['m38'].get('details', {}))
        if m38_out:
            print(m38_out)

    # M39 Michigan Sentiment Session Bias
    if 'm39' in result and result['m39'].get('status') not in ('SKIP', 'NO_EDGE'):
        m39_out = format_m39(result['m39'].get('details', {}))
        if m39_out:
            print(m39_out)

    # M40 Germany CPI Session Bias
    if 'm40' in result and result['m40'].get('status') not in ('SKIP', 'NO_EDGE'):
        m40_out = format_m40(result['m40'].get('details', {}))
        if m40_out:
            print(m40_out)

    # M41 EZ CPI Flash Session Bias
    if 'm41' in result and result['m41'].get('status') not in ('SKIP', 'NO_EDGE'):
        m41_out = format_m41(result['m41'].get('details', {}))
        if m41_out:
            print(m41_out)

    # M42 EZ GDP Flash Session Bias
    if 'm42' in result and result['m42'].get('status') not in ('SKIP', 'NO_EDGE'):
        m42_out = format_m42(result['m42'].get('details', {}))
        if m42_out:
            print(m42_out)

    # M43 US GDP Advance Session Bias
    if 'm43' in result and result['m43'].get('status') not in ('SKIP', 'NO_EDGE'):
        m43_out = format_m43(result['m43'].get('details', {}))
        if m43_out:
            print(m43_out)

    # M44 US Durable Goods Orders Session Bias
    if 'm44' in result and result['m44'].get('status') not in ('SKIP', 'NO_EDGE'):
        m44_out = format_m44(result['m44'].get('details', {}))
        if m44_out:
            print(m44_out)

    # M45 Core PCE + Personal Spending Session Bias
    if 'm45' in result and result['m45'].get('status') not in ('SKIP', 'NO_EDGE'):
        m45_out = format_m45(result['m45'].get('details', {}))
        if m45_out:
            print(m45_out)

    # M46 Japan CPI (Tokyo Flash) Session Bias
    if 'm46' in result and result['m46'].get('status') not in ('SKIP', 'NO_EDGE'):
        m46_out = format_m46(result['m46'].get('details', {}))
        if m46_out:
            print(m46_out)

    # M47 BoJ Rate Decision Session Bias
    if 'm47' in result and result['m47'].get('status') not in ('SKIP', 'NO_EDGE'):
        m47_out = format_m47(result['m47'].get('details', {}))
        if m47_out:
            print(m47_out)

    # M48 ECB Rate Decision + Lagarde Presser Session Bias
    if 'm48' in result and result['m48'].get('status') not in ('SKIP', 'NO_EDGE'):
        m48_out = format_m48(result['m48'].get('details', {}))
        if m48_out:
            print(m48_out)

    # M49 BoE Rate Decision + MPC Vote Split Session Bias
    if 'm49' in result and result['m49'].get('status') not in ('SKIP', 'NO_EDGE'):
        m49_out = format_m49(result['m49'].get('details', {}))
        if m49_out:
            print(m49_out)

    # Module Scores
    print(f"\n  Module Scores:")
    if 'm1' in result:
        print(f"    M1 (1H MACD):  {result['m1']['direction']:>8}  score={result['m1']['score']:.2f}")
    if 'm2' in result:
        print(f"    M2 (EMA conf): {result['m2']['status']:>8}  score={result['m2']['score']:.2f}")
    if 'm3' in result:
        print(f"    M3 (VWAP):     {result['m3']['status']:>8}  score={result['m3']['score']:.2f}")
    if 'm4' in result:
        m4 = result['m4']
        det = m4.get('div', {}) or m4.get('details', {})
        if isinstance(det, dict):
            div_str = det.get('layer_a_div', 'NONE')
            zl_str = det.get('layer_b_cross', 'NONE')
            la = det.get('layer_a_score', 0)
            lb = det.get('layer_b_score', 0)
            bars = det.get('layer_b_bars_since', '')
            print(f"    M4 (CVD):      {m4['status']:>8}  score={m4['score']:.2f}  "
                  f"div={div_str}({la:.2f})  zl={zl_str}({lb:.2f}) {bars}bars")
        else:
            print(f"    M4 (CVD):      {m4['status']:>8}  score={m4['score']:.2f}")
    if 'm4b' in result:
        m4b = result['m4b']
        m4b_div = m4b.get('divergence', 'NONE')
        m4b_ago = m4b.get('bars_ago', -1)
        m4b_slope = m4b.get('cvd_slope', 0)
        m4b_icon = {'BEARISH': '🔻', 'BULLISH': '🔺', 'NONE': '—'}.get(m4b_div, '—')
        ago_str = f"{m4b_ago}bars ago" if m4b_ago >= 0 else ""
        print(f"    M4b(IntraCVD): {m4b['status']:>8}  score={m4b['score']:.2f}  "
              f"div={m4b_div} {m4b_icon}  slope={m4b_slope:.1f}  {ago_str}")
    if 'm5' in result:
        m5 = result['m5']
        print(f"    M5 (LiqtMag):  {m5['status']:>8}  score={m5['score']:.2f}")
    if 'm8' in result:
        print(f"    M8 (Funding):  {result['m8']['status']:>8}  score={result['m8']['score']:.2f}  rate={result['m8'].get('rate', 'N/A')}")
    if 'm9' in result:
        m9 = result['m9']
        m9_st = m9.get('status', '—')
        m9_sc = m9.get('score', 0)
        print(f"    M9 (VolRegime):{m9_st:>8}  score={m9_sc:.2f}  regime={m9['regime']}")
    if 'm10' in result:
        m10 = result['m10']
        m10_comp = m10.get('details', {}).get('m10_components', {})
        m10_agree = m10.get('details', {}).get('macro_agreement', '')
        print(f"    M10 (Macro):   {m10['status']:>8}  score={m10['score']:.2f}  {m10_agree}")
    if 'm11' in result:
        print(f"    M11 (MTF Mom): {result['m11']['status']:>8}  score={result['m11']['score']:.2f}")
    if 'm13' in result:
        m13 = result['m13']
        print(f"    M13 (Struct):  {m13['status']:>8}  score={m13['score']:.2f}  bias={m13['bias']}")
    if 'm12' in result:
        m12 = result['m12']
        m12_ob = m12.get('details', {}).get('bid_ask_ratio', '')
        ob_str = f"  OB={m12_ob:.2f}" if m12_ob else ""
        print(f"    M12 (OrderBook): {m12['status']:>8}  score={m12['score']:.2f}{ob_str}")
    if 'm14' in result:
        print(f"    M14 (Sweep):   {result['m14']['status']:>8}  score={result['m14']['score']:.2f}")
    if 'm17' in result:
        m17 = result['m17']
        m17_zv = m17.get('zone_volume', {}).get('zone_vol_ratio', '?')
        m17_rej = m17.get('rejection', {}).get('status', '?')
        m17_dfn = m17.get('defender', {}).get('status', '?')
        print(f"    M17 (ResQual): {m17['status']:>8}  score={m17['score']:.3f}  "
              f"zone={m17_zv}x  reject={m17_rej}  defender={m17_dfn}  → {m17['verdict']}")
    if 'm20' in result:
        m20 = result['m20']
        m20_st = m20.get('status', '—')
        m20_sc = m20.get('score', 0.5)
        m20_bd = m20.get('breakout_direction', '')
        m20_ct = m20.get('contrarian_direction', '')
        m20_lv = m20.get('level', 0)
        m20_q = m20.get('breakout_quality', 0)
        m20_fs = m20.get('failure', {}).get('status', '')
        if m20_st not in ('SKIP', 'NO_LEVELS', 'NO_ACTIONABLE', 'DISABLED', 'INSUFFICIENT_DATA'):
            print(f"    M20 (FailBO):  {m20_st:>8}  score={m20_sc:.3f}  "
                  f"{m20_bd} @ ${m20_lv:.0f}  quality={m20_q:.2f}  fail={m20_fs}  → {m20_ct}")
    # Taker flow module score
    if CONFIG.get('TAKER_ENABLED', False):
        _td = result.get('taker_summary', {})
        _td_dir = _td.get('direction', 'NEUTRAL') if _td else 'NEUTRAL'
        _td_sc = _td.get('score', 0) if _td else 0
        _td_regime = _td.get('regime', '?') if _td else '?'
        _td_icon = {'LONG': '🟢', 'SHORT': '🔴', 'NEUTRAL': '⚪'}.get(_td_dir, '⚪')
        _taker_ics = result.get('taker_score', 0.5)
        print(f"    Taker (Flow): {_td_icon} {_td_dir:>8}  score={_taker_ics:.3f}  regime={_td_regime}")
    if 'cascade' in result and result['cascade'].get('cascade'):
        c = result['cascade']
        print(f"    ⚡ CASCADE:    momentum={c['momentum']}% vol_spike={c['vol_spike']}x range={c['range_expansion']}x")

    # Liquidation Magnets
    magnets = result.get('magnets', [])
    if magnets:
        print(f"\n  Liquidation Magnets (volume clusters):")
        price = result['price']
        for i, mag in enumerate(magnets[:5]):
            if len(mag) == 4:
                p, s, swept, swept_at = mag
            else:
                p, s = mag[0], mag[1]
                swept, swept_at = False, None
            dist = (p - price) / price * 100
            direction = "↑" if dist > 0 else "↓"
            swept_tag = f"  ✅ SWEPT @ {swept_at}" if swept else ""
            print(f"    #{i+1}: ${p:.2f}  strength={s:.2f}x  ({direction}{abs(dist):.2f}%){swept_tag}")

    # Support / Resistance
    sr = result.get('sr_levels', [])
    if sr:
        supports = [(p, s, t, tb, bb) for p, s, t, tb, bb in sr if t == 'SUPPORT']
        resistances = [(p, s, t, tb, bb) for p, s, t, tb, bb in sr if t == 'RESISTANCE']
        supports.sort(key=lambda x: x[1], reverse=True)
        resistances.sort(key=lambda x: x[1], reverse=True)
        if supports:
            print(f"  Support Levels:")
            for i, (p, s, _, touches, bounces) in enumerate(supports[:4]):
                dist = (p - price) / price * 100
                print(f"    #{i+1}: ${p:.2f}  strength={s:.1f}  touches={touches} bounces={bounces}  ({dist:+.2f}%)")
        if resistances:
            print(f"  Resistance Levels:")
            for i, (p, s, _, touches, bounces) in enumerate(resistances[:4]):
                dist = (p - price) / price * 100
                print(f"    #{i+1}: ${p:.2f}  strength={s:.1f}  touches={touches} bounces={bounces}  ({dist:+.2f}%)")

    # M17 Resistance Quality
    if 'm17' in result and result['m17'].get('status') != 'SKIP':
        print(format_resistance_quality(result['m17']))

    # M20 Failed Breakout
    if 'm20' in result:
        m20_fmt = format_failed_breakout(result['m20'])
        if m20_fmt:
            print(m20_fmt)

    # Macro Indicators (fetched live)
    macro_ind = result.get('macro_indicators', {})
    if macro_ind:
        has_data = any(v and v.get('actual') is not None for v in macro_ind.values())
        if has_data:
            print(f"\n  Macro Indicators (live):")
            for name, data in macro_ind.items():
                if data and data.get('actual') is not None:
                    actual = data['actual']
                    prev = data.get('previous', '?')
                    surprise = data.get('surprise', '?')
                    source = data.get('source', '?')
                    icon = {'STRONG_BEAT': '🟢🟢', 'BEAT': '🟢', 'INLINE': '⚪',
                            'MISS': '🔴', 'BIG_MISS': '🔴🔴'}.get(surprise, '❓')
                    print(f"    {name:24s} actual={actual}  prev={prev}  {surprise} {icon}  [{source}]")

    # Derivatives
    deriv = result.get('derivatives', {})
    if deriv and 'error' not in deriv:
        print(f"\n  Derivatives Data:")
        oi_usd = deriv.get('oi_usd', 0)
        print(f"    OI:             {deriv.get('oi', 0):,.0f} ETH  (${oi_usd/1e9:.2f}B)  1h Δ: {deriv.get('oi_roc_1h', 0):+.3f}%")
        print(f"    L/S Ratio:      {deriv.get('ls_ratio', 0):.4f}  (long {deriv.get('long_pct', 0):.1f}% / short {deriv.get('short_pct', 0):.1f}%)  z={deriv.get('ls_zscore', 0):.2f}")
        pos = deriv.get('positioning', 'NEUTRAL')
        pos_icon = {'CROWDED_LONG': '🔴', 'CROWDED_SHORT': '🟢'}.get(pos, '⚪')
        print(f"    Positioning:    {pos_icon} {pos}")
        print(f"    Top Traders:    L/S={deriv.get('top_ls_ratio', 0):.4f}  whale={deriv.get('whale_signal', 'NEUTRAL')}  gap={deriv.get('whale_retail_gap', 0):+.4f}")
        print(f"    Futures Taker:  {deriv.get('futures_taker_ratio', 0):.4f}  flow={deriv.get('futures_flow', 'NEUTRAL')}")
        fr = deriv.get('funding_rate')
        if fr is not None:
            fr_label = "longs pay" if fr > 0 else "shorts pay"
            print(f"    Funding Rate:   {fr*100:+.4f}%  ({fr_label})")
        oi_div = deriv.get('oi_price_div', 'NONE')
        if oi_div != 'NONE':
            print(f"    ⚡ OI Divergence: {oi_div}")

    # Exchange Activity (cross-exchange)
    exch = result.get('exchange_activity', {})
    if exch and 'error' not in exch:
        print(f"\n  Exchange Activity (cross-exchange):")
        snaps = exch.get('snapshots', {})
        sigs = exch.get('signals', {})

        # Per-exchange snapshot table
        print(f"    {'Exchange':<10} {'Funding':>12} {'OI (ETH)':>14} {'L/S Ratio':>10}")
        for ex_name in ['binance', 'okx', 'bybit', 'htx', 'phemex', 'kraken']:
            s = snaps.get(ex_name, {})
            if not s or s.get('error'):
                continue
            fr = s.get('funding_rate')
            oi = s.get('oi')
            ls = s.get('ls_ratio')
            fr_str = f"{fr*100:+.4f}%" if fr is not None else "N/A"
            oi_str = f"{oi:,.0f}" if oi else "N/A"
            ls_str = f"{ls:.4f}" if ls else "N/A"
            print(f"    {ex_name:<10} {fr_str:>12} {oi_str:>14} {ls_str:>10}")

        # Funding spread
        spread = sigs.get('funding_spread', 0)
        spread_trend = sigs.get('funding_spread_trend', '?')
        print(f"    Funding spread: {spread*100:.4f}%  trend={spread_trend}")

        # OI shares
        oi_shares = sigs.get('oi_shares', {})
        if oi_shares:
            shares_str = '  '.join(f"{k}={v*100:.1f}%" for k, v in sorted(oi_shares.items()))
            print(f"    OI share:  {shares_str}")
            print(f"    OI dominant: {sigs.get('oi_dominant_exchange', '?')}  "
                  f"concentration={sigs.get('oi_concentration', 0):.3f}")

        # OI migration
        migration = sigs.get('oi_migration')
        if migration and migration != 'BALANCED':
            print(f"    ⚡ OI Migration: {migration}")

        # L/S divergence
        ls_spread = sigs.get('ls_spread', 0)
        if ls_spread > 0.3:
            ls_by = sigs.get('ls_by_exchange', {})
            ls_str = '  '.join(f"{k}={v:.2f}" for k, v in sorted(ls_by.items()))
            print(f"    L/S divergence: {ls_spread:.3f}  ({ls_str})")

        # Scoring
        ex_score = exch.get('score', 0)
        ex_status = exch.get('status', '?')
        ex_details = exch.get('direction_details', {})
        ex_factors = ex_details.get('factors', [])
        print(f"    Score: {ex_score:.3f} ({ex_status})")
        for f in ex_factors:
            print(f"      • {f}")

        # Spot data
        spot = exch.get('spot', {})
        spot_sigs = exch.get('spot_signals', {})
        spot_details = exch.get('spot_details', {})
        if spot:
            print(f"\n    Spot Markets:")
            print(f"      {'Exchange':<10} {'Price':>10} {'24h Vol':>12} {'OB Ratio':>10} {'Flow':>8}")
            for ex_name in ['binance', 'okx', 'bybit', 'kraken', 'coinbase', 'htx']:
                s = spot.get(ex_name, {})
                if not s or not s.get('price'):
                    continue
                p = s.get('price', 0)
                v = s.get('vol_24h') or 0
                ob = s.get('ob_ratio') or 0
                buy_pct = s.get('buy_pct') or 50
                flow = f"{buy_pct:.0f}% buy"
                print(f"      {ex_name:<10} ${p:>9.2f} {v:>11,.0f} {ob:>10.3f} {flow:>8}")

            # Basis
            basis = spot_sigs.get('basis', {})
            if basis:
                basis_str = '  '.join(f"{k}={v:+.3f}%" for k, v in sorted(basis.items()))
                print(f"      Basis: {basis_str}")
                print(f"      Avg basis: {spot_sigs.get('basis_avg', 0):+.4f}% ({spot_sigs.get('basis_state', '?')})")

            # Spot walls
            sell_walls = spot_sigs.get('spot_sell_walls', [])
            bid_support = spot_sigs.get('spot_bid_support', [])
            if sell_walls or bid_support:
                parts = []
                if sell_walls:
                    parts.append(f"sell walls: {', '.join(sell_walls)}")
                if bid_support:
                    parts.append(f"bid support: {', '.join(bid_support)}")
                print(f"      Book: {' | '.join(parts)}")

            # Spot scoring
            sp_score = exch.get('spot_score', 0)
            sp_status = exch.get('spot_status', '?')
            sp_factors = spot_details.get('factors', [])
            print(f"      Spot score: {sp_score:.3f} ({sp_status})")
            for f in sp_factors:
                print(f"        • {f}")

    # Real Liquidity Levels (liquidation + stops + order book)
    liq = result.get('liquidity_levels', {})
    if liq:
        price = result['price']
        n_fresh = 0
        for z in liq.get('below', []) + liq.get('above', []):
            if z.get('formed_at') is not None and not z.get('swept'):
                n_fresh += 1
        print(f"\n  Liquidity Levels (estimated):")
        print(f"    High cascade zones: {liq.get('high_cascade_zones', 0)}  |  "
              f"Bid walls: {liq.get('bid_walls', 0)}  Ask walls: {liq.get('ask_walls', 0)}  |  "
              f"🆕 Fresh unswept: {n_fresh}")

        below = liq.get('below', [])
        if below:
            print(f"    ▼ Below ${price:.0f} (long liquidations / stops):")
            for z in below[:8]:
                icon = {'LONG_LIQ': '💥', 'LONG_STOP': '🛑', 'BID_WALL': '🟢',
                        'SHORT_LIQ': '💥', 'SHORT_STOP': '🛑', 'ASK_WALL': '🔴'}.get(z['type'], '•')
                cascade = z.get('cascade_risk', '')
                swept_tag = f"  ✅ SWEPT" if z.get('swept') else ""
                is_fresh = z.get('formed_at') is not None
                age = z.get('age_bars')
                fresh_tag = f"  🆕 {age}bars" if is_fresh and age else ""
                dist_pct = z.get('dist_pct', (z['price'] - price) / price * 100)
                source = z.get('source', '')
                source_tag = f"  [{source}]" if source and is_fresh else ""
                print(f"      {icon} ${z['price']:.2f}  {z['type']}  "
                      f"str={z['strength']:.0f}  cascade={cascade}  ({dist_pct:+.2f}%){swept_tag}{fresh_tag}{source_tag}")

        above = liq.get('above', [])
        if above:
            print(f"    ▲ Above ${price:.0f} (short liquidations / stops):")
            for z in above[:8]:
                icon = {'LONG_LIQ': '💥', 'LONG_STOP': '🛑', 'BID_WALL': '🟢',
                        'SHORT_LIQ': '💥', 'SHORT_STOP': '🛑', 'ASK_WALL': '🔴'}.get(z['type'], '•')
                cascade = z.get('cascade_risk', '')
                swept_tag = f"  ✅ SWEPT" if z.get('swept') else ""
                is_fresh = z.get('formed_at') is not None
                age = z.get('age_bars')
                fresh_tag = f"  🆕 {age}bars" if is_fresh and age else ""
                dist_pct = z.get('dist_pct', (z['price'] - price) / price * 100)
                source = z.get('source', '')
                source_tag = f"  [{source}]" if source and is_fresh else ""
                print(f"      {icon} ${z['price']:.2f}  {z['type']}  "
                      f"str={z['strength']:.0f}  cascade={cascade}  ({dist_pct:+.2f}%){swept_tag}{fresh_tag}{source_tag}")

    # Conflict History
    conflict = result.get('conflict')
    if conflict and conflict.get('historical'):
        hist = conflict['historical']
        is_conflict = conflict['is_conflict']
        label = "CONFLICT" if is_conflict else "ALIGNED"
        print(f"\n  Conflict History ({conflict['m1_direction']} M1 vs {conflict['daily_bias']} daily — {label}):")
        print(f"    Historical signals: {hist['total_signals']}  ({hist['first_seen'][:10]} → {hist['last_seen'][:10]})")
        windows = hist.get('windows', {})
        if windows:
            print(f"    {'Window':>6}  {'Rev%':>6}  {'Win%':>6}  {'AvgNet':>8}  {'Avg↓':>8}  {'Avg↑':>8}  {'n':>4}")
            for wname in ['4h', '12h', '24h', '48h', '72h']:
                w = windows.get(wname)
                if w:
                    rev_icon = "⬇️" if w['reversal_rate'] > 55 else "⬆️" if w['reversal_rate'] < 45 else "↔️"
                    print(f"    {wname:>6}  {w['reversal_rate']:>5.1f}%  {w['win_rate']:>5.1f}%  "
                          f"{w['avg_net']:>+7.2f}%  {-w['avg_down']:>+7.2f}%  {w['avg_up']:>+7.2f}%  {w['n']:>4}  {rev_icon}")

    # Cascade Risk
    cr = result.get('cascade_risk', {})
    if cr:
        verdict = cr.get('verdict', 'UNKNOWN')
        icon = {'CASCADE': '🌊', 'RISKY': '⚠️', 'FLUSH': '💧'}.get(verdict, '❓')
        print(f"\n  Cascade Risk: {icon} {verdict}  (score={cr.get('score', 0):.2f})")
        for f in cr.get('factors', []):
            print(f"    • {f}")

    # Squeeze Detector
    sq = result.get('squeeze', {})
    if sq and sq.get('squeeze_type', 'NONE') != 'NONE':
        sq_output = format_squeeze(sq)
        if sq_output:
            print(sq_output)
        # Show 5-filter confirmation status
        sq_filters = result.get('squeeze_filters', {})
        sq_confirmed = result.get('squeeze_confirmed', False)
        if sq_filters:
            icons = {True: '✅', False: '❌'}
            _sq_status_label = sq.get('squeeze_status', 'NONE')
            _gate_label = f" (only matters when TRIGGERED)" if _sq_status_label != 'TRIGGERED' else ""
            print(f"\n  Squeeze Confirmation Gate{_gate_label}:")
            print(f"    EMA regime:   {icons.get(sq_filters.get('ema_regime'), '?')}")
            print(f"    CVD agrees:   {icons.get(sq_filters.get('cvd_agrees'), '?')}")
            print(f"    RSI ok:       {icons.get(sq_filters.get('rsi_ok'), '?')}")
            print(f"    Quality ≥0.5: {icons.get(sq_filters.get('quality_high'), '?')}")
            _atr_val = sq_filters.get('_atr_value', '?')
            _atr_thr = sq_filters.get('_atr_threshold', '?')
            print(f"    ATR floor:    {icons.get(sq_filters.get('atr_floor'), '?')}  (ATR=${_atr_val} vs floor=${_atr_thr})")
            if _sq_status_label != 'TRIGGERED':
                print(f"    → ⏳ PENDING — gate will be checked on trigger")
            elif sq_confirmed:
                print(f"    → ✅ CONFIRMED (backtested 84.6% WR on 4h)")
            else:
                print(f"    → ❌ NOT CONFIRMED — regime override & ICS boost skipped")
        if result.get('squeeze_override'):
            print(f"\n  ⚡ SQUEEZE OVERRIDE: regime block lifted!")
    else:
        # Always show squeeze status even when no active squeeze
        sq_gates = sq.get('gates_failed', []) if sq else []
        sq_reason = sq_gates[0] if sq_gates else 'no detection'
        print(f"\n  🔥 Squeeze: ⚪ NONE  ({sq_reason})")

    # Breakout Confirmation (M19)
    bc = result.get('breakout_confirm')
    if bc and bc.get('filters'):
        print(format_breakout_confirm(bc))
        if result.get('breakout_rejected'):
            print(f"    ⚠️  Squeeze trigger suppressed — breakout filters failed")
        if result.get('squeeze_ics_boost_half'):
            print(f"    ⚠️  ICS boost halved — weak breakout confirmation")

    # ICS & Signal
    if 'ics' in result:
        boost_str = f"  (squeeze +{result.get('squeeze_ics_boost', 0):.4f})" if result.get('squeeze_ics_boost') else ''
        print(f"\n  ICS: {result['ics']:.3f}  (floor={result['effective_floor']:.3f}){boost_str}")

    status = result['status']
    if status == 'SIGNAL':
        print(f"\n  ✅ SIGNAL: {result['direction']}")
        sl_src = result.get('sl_source', 'ATR')
        tp1_src = result.get('tp1_source', 'ATR')
        tp2_src = result.get('tp2_source', 'ATR')
        tp3_src = result.get('tp3_source', 'ATR')
        le = result.get('limit_entry', {})
        if le.get('entry_source', 'MARKET') != 'MARKET':
            print(f"    Entry: ${result['entry']:.2f}  [{le.get('entry_source', '?')}]  (market: ${result.get('market_entry', result['entry']):.2f})")
            print(f"           {le.get('reason', '')}")
        else:
            print(f"    Entry: ${result['entry']:.2f}")
        if result.get('range_sl_override'):
            print(f"    SL:    ${result['sl']:.2f}  ({result['sl_pct']:.2f}%)  [{sl_src}]  🔧 RANGE OVERRIDE")
        else:
            print(f"    SL:    ${result['sl']:.2f}  ({result['sl_pct']:.2f}%)  [{sl_src}]")
        if result.get('range_tp_override'):
            print(f"    TP1:   ${result['tp1']:.2f}  ({result['tp1_pct']:.2f}%)  [{tp1_src}]  🔧 RANGE OVERRIDE")
        else:
            print(f"    TP1:   ${result['tp1']:.2f}  ({result['tp1_pct']:.2f}%)  [{tp1_src}]")
        print(f"    TP2:   ${result['tp2']:.2f}  [{tp2_src}]")
        print(f"    TP3:   ${result['tp3']:.2f}  [{tp3_src}]")
    else:
        print(f"\n  ⛔ {status}: {result.get('reason', 'N/A')}")

    # ── What-if trade levels (always show) ──
    w = result.get('what_if')
    if w:
        print(f"\n  {'─' * 56}")
        print(f"  IF YOU WERE TO TRADE ({w['direction']}):")
        if w.get('entry_source', 'MARKET') != 'MARKET':
            print(f"    Entry: ${w['entry']:.2f}  [{w['entry_source']}]  ← limit order")
            print(f"           {w.get('entry_reason', '')}")
            print(f"    Market: ${w.get('market_entry', w['entry']):.2f}  ← current price")
        else:
            print(f"    Entry: ${w['entry']:.2f}  [MARKET]")
        print(f"    SL:    ${w['sl']:.2f}  ({w['sl_pct']:.2f}%)  [{w['sl_source']}]")
        print(f"    TP1:   ${w['tp1']:.2f}  ({w['tp1_pct']:.2f}%)  [{w['tp1_source']}]")
        print(f"    TP2:   ${w['tp2']:.2f}  [{w['tp2_source']}]")
        print(f"    TP3:   ${w['tp3']:.2f}  [{w['tp3_source']}]")
        print(f"    R:R:   {w['rr1']:.2f}x (entry→TP1 vs SL)")
        print(f"    Support:    ${w['nearest_support']:.2f}")
        print(f"    Resistance: ${w['nearest_resist']:.2f}")
        if w['invalidation']:
            print(f"    ⚠️  INVALIDATION:")
            for inv in w['invalidation']:
                print(f"      • {inv}")
    print("═" * 60)


def print_summary(result):
    """Print clean one-page summary with table + verdict."""
    price = result['price']
    deriv = result.get('derivatives', {})
    liq = result.get('liquidity_levels', {})

    # Module score table
    m1_dir = result.get('m1', {}).get('direction', 'N/A')
    m1_sc = result.get('m1', {}).get('score', 0)
    m2_st = result.get('m2', {}).get('status', 'N/A')
    m2_sc = result.get('m2', {}).get('score', 0)
    m3_st = result.get('m3', {}).get('status', 'N/A') if 'm3' in result else '—'
    m3_sc = result.get('m3', {}).get('score', 0) if 'm3' in result else 0
    m4_st = result.get('m4', {}).get('status', 'N/A') if 'm4' in result else '—'
    m4_sc = result.get('m4', {}).get('score', 0) if 'm4' in result else 0
    m5_st = result.get('m5', {}).get('status', 'N/A') if 'm5' in result else '—'
    m5_sc = result.get('m5', {}).get('score', 0) if 'm5' in result else 0
    m8_st = result.get('m8', {}).get('status', '—') if 'm8' in result else '—'
    m8_sc = result.get('m8', {}).get('score', 0) if 'm8' in result else 0
    m9_st = result.get('m9', {}).get('status', '—') if 'm9' in result else '—'
    m9_sc = result.get('m9', {}).get('score', 0) if 'm9' in result else 0
    m11_st = result.get('m11', {}).get('status', '—') if 'm11' in result else '—'
    m11_sc = result.get('m11', {}).get('score', 0) if 'm11' in result else 0
    m13_st = result.get('m13', {}).get('status', '—') if 'm13' in result else '—'
    m13_sc = result.get('m13', {}).get('score', 0) if 'm13' in result else 0
    m14_st = result.get('m14', {}).get('status', '—') if 'm14' in result else '—'
    m14_sc = result.get('m14', {}).get('score', 0) if 'm14' in result else 0

    print("\n" + "─" * 55)
    print("  SUMMARY")
    print("─" * 55)
    print(f"  {'Module':<22} {'Status':>10}  {'Score':>6}")
    print(f"  {'M1 MACD (1H)':<22} {m1_dir:>10}  {m1_sc:>6.2f}")
    print(f"  {'M2 EMA confluence':<22} {m2_st:>10}  {m2_sc:>6.2f}")
    print(f"  {'M3 VWAP':<22} {m3_st:>10}  {m3_sc:>6.2f}")
    print(f"  {'M4 CVD':<22} {m4_st:>10}  {m4_sc:>6.2f}")
    m4b_st = result.get('m4b', {}).get('status', '—') if 'm4b' in result else '—'
    m4b_sc = result.get('m4b', {}).get('score', 0) if 'm4b' in result else 0
    print(f"  {'M4b Intrabar CVD':<22} {m4b_st:>10}  {m4b_sc:>6.2f}")
    print(f"  {'M5 Liquidation':<22} {m5_st:>10}  {m5_sc:>6.2f}")
    print(f"  {'M8 Funding':<22} {m8_st:>10}  {m8_sc:>6.2f}")
    print(f"  {'M9 Vol Regime':<22} {m9_st:>10}  {m9_sc:>6.2f}")
    m10_st = result.get('m10', {}).get('status', '—') if 'm10' in result else '—'
    m10_sc = result.get('m10', {}).get('score', 0) if 'm10' in result else 0
    print(f"  {'M10 Cross-Asset':<22} {m10_st:>10}  {m10_sc:>6.2f}")
    print(f"  {'M11 MTF Momentum':<22} {m11_st:>10}  {m11_sc:>6.2f}")
    print(f"  {'M13 Structure':<22} {m13_st:>10}  {m13_sc:>6.2f}")
    print(f"  {'M14 Sweep':<22} {m14_st:>10}  {m14_sc:>6.2f}")
    m17_st = result.get('m17', {}).get('status', '—') if 'm17' in result else '—'
    m17_sc = result.get('m17', {}).get('score', 0) if 'm17' in result else 0
    m17_vd = result.get('m17', {}).get('verdict', '') if 'm17' in result else ''
    print(f"  {'M17 Resist Qual':<22} {m17_st:>10}  {m17_sc:>6.3f}  {m17_vd}")
    m20_st = result.get('m20', {}).get('status', '—') if 'm20' in result else '—'
    m20_sc = result.get('m20', {}).get('score', 0) if 'm20' in result else 0
    m20_bd = result.get('m20', {}).get('breakout_direction', '') if 'm20' in result else ''
    m20_ct = result.get('m20', {}).get('contrarian_direction', '') if 'm20' in result else ''
    m20_fs = result.get('m20', {}).get('failure', {}).get('status', '') if 'm20' in result else ''
    if m20_st not in ('SKIP', '—'):
        print(f"  {'M20 Failed Breakout':<22} {m20_st:>10}  {m20_sc:>6.3f}  {m20_bd}→{m20_ct} ({m20_fs})")
    m12_st = result.get('m12', {}).get('status', '—') if 'm12' in result else '—'
    m12_sc = result.get('m12', {}).get('score', 0) if 'm12' in result else 0
    print(f"  {'M12 Order Book':<22} {m12_st:>10}  {m12_sc:>6.2f}")
    m16_exch = result.get('exchange_activity', {})
    m16_sc = m16_exch.get('score', 0) if m16_exch else 0
    m16_st = m16_exch.get('status', '—') if m16_exch else '—'
    print(f"  {'M16 Exch Derivs':<22} {m16_st:>10}  {m16_sc:>6.2f}")
    m16_sp_sc = m16_exch.get('spot_score', 0) if m16_exch else 0
    m16_sp_st = m16_exch.get('spot_status', '—') if m16_exch else '—'
    print(f"  {'M16 Exch Spot':<22} {m16_sp_st:>10}  {m16_sp_sc:>6.2f}")

    # M66–M73: Traditional Finance Macro Filters
    for _m_num, _m_name in [(66, 'USD/JPY'), (67, 'DXY'), (68, '10Y Yield'),
                             (69, 'VIX'), (70, 'WTI Oil'), (71, 'Gold'),
                             (72, 'BTC.D'), (73, 'Stablecoins')]:
        _m_key = f'm{_m_num}'
        _m = result.get(_m_key, {})
        _m_st = _m.get('status', '—') if _m else '—'
        _m_sc = _m.get('score', 0) if _m else 0
        _m_cls = _m.get('details', {}).get('classification', '') if _m else ''
        if _m_st not in ('SKIP', '—', None):
            print(f"  {f'M{_m_num} {_m_name}':<22} {_m_st:>10}  {_m_sc:>6.3f}  {_m_cls}")

    # Squeeze in summary
    sq = result.get('squeeze', {})
    if sq and sq.get('squeeze_type', 'NONE') != 'NONE':
        sq_status = sq.get('squeeze_status', 'NONE')
        sq_st = f"{'STRONG' if sq.get('squeeze_strong') else 'ACTIVE'} {sq_status}"
        sq_sc = sq.get('squeeze_score', 0)
        sq_dir = sq.get('direction', '?')
        print(f"  {'M18 Squeeze':<22} {sq_st:>10}  {sq_sc:>6.3f}  → {sq_dir}")
        # Box type (Phase 1)
        box_type = sq.get('box_type', 'UNKNOWN')
        if box_type != 'UNKNOWN':
            box_bias = sq.get('box_bias', 'NEUTRAL')
            bias_str = f'  bias={box_bias}' if box_bias != 'NEUTRAL' else ''
            print(f"  {'  Box':<22} {box_type:>10}{bias_str}")
        # Lifecycle (Phase 3)
        lc = sq.get('lifecycle_stage', '')
        if lc:
            print(f"  {'  Lifecycle':<22} {lc:>10}")
        # Maturity
        mat = sq.get('compression_maturity', 0)
        if mat > 0:
            print(f"  {'  Maturity':<22} {mat:>9.0%}")
        if sq.get('entry_condition'):
            print(f"  {'  Entry':<22} {sq['entry_condition']}")
        # Breakout quality (Phase 2)
        bq = sq.get('breakout_quality_score')
        if bq is not None:
            bq_icon = '✅' if sq.get('breakout_quality_passed') else '⚠️'
            print(f"  {'  Breakout Quality':<22} {bq_icon} {bq:.2f}")
        # Retest
        if sq.get('retest_detected'):
            print(f"  {'  Retest':<22} ✅ {sq.get('retest_quality', 0):.2f}")
        sq_confirmed = result.get('squeeze_confirmed', False)
        sq_filters = result.get('squeeze_filters', {})
        if sq_filters:
            n_pass = sum(1 for k, v in sq_filters.items() if v is True)
            n_total = sum(1 for k in sq_filters if not k.startswith('_'))
            conf_label = f"✅ CONFIRMED ({n_pass}/{n_total})" if sq_confirmed else f"❌ ({n_pass}/{n_total})"
            print(f"  {'  Confirmation':<22} {conf_label}")
        if result.get('squeeze_override'):
            print(f"  {'⚡ Regime Override':<22} {'ACTIVE':>10}")
        # Squeeze trade plan
        if sq.get('tp', 0) > 0 and sq.get('entry_price', 0) > 0:
            sq_entry = sq['entry_price']
            sq_tp = sq['tp']
            sq_sl = sq['sl']
            sq_tp_pct = sq.get('tp_pct', 0)
            sq_sl_pct = sq.get('sl_pct', 0)
            sq_tp_dist = abs(sq_tp - sq_entry)
            sq_sl_dist = abs(sq_entry - sq_sl)
            sq_rr = sq_tp_dist / sq_sl_dist if sq_sl_dist > 0 else 0
            print(f"  {'  ── Trade Plan':<22} {'──────────':>10}")
            print(f"  {'  Entry':<22} ${sq_entry:<10.2f} (trigger)")
            print(f"  {'  TP':<22} ${sq_tp:<10.2f} (+{sq_tp_pct:.2f}%)")
            print(f"  {'  SL':<22} ${sq_sl:<10.2f} ({sq_sl_pct:.2f}%)")
            print(f"  {'  R:R':<22} {sq_rr:.2f}x")
            print(f"  {'  Target Source':<22} {sq.get('tp1_source', 'ATR'):>10}")
    else:
        sq_gates = sq.get('gates_failed', []) if sq else []
        sq_reason = sq_gates[0] if sq_gates else 'no detection'
        print(f"  {'M18 Squeeze':<22} {'NONE':>10}  —  ({sq_reason})")

    # M21 Wyckoff Phase summary
    if 'm21' in result and result['m21'].get('status') != 'SKIP':
        m21 = result['m21']
        phase = m21.get('phase', '?')
        zone = m21.get('zone', '?')
        kz = m21.get('kill_zone', '?')
        spring = m21.get('spring_upthrust', 'NONE')
        phase_icons = {'ACCUMULATION': '🟢', 'MARKUP': '📈', 'DISTRIBUTION': '🔴',
                       'MARKDOWN': '📉', 'RANGE': '↔️'}
        p_icon = phase_icons.get(phase, '❓')
        zone_icons = {'PREMIUM': '🔴', 'DISCOUNT': '🟢', 'EQUILIBRIUM': '⚪'}
        z_icon = zone_icons.get(zone, '?')
        kz_icon = '✅' if kz in ('LONDON_OPEN', 'NY_OPEN', 'LONDON_CLOSE') else '⚠️'
        spring_str = f' ⚡{spring}' if spring != 'NONE' else ''
        print(f"  {'M21 Wyckoff':<22} {p_icon} {phase:>12}  {z_icon} {zone}{spring_str}")
        print(f"  {'  Kill Zone':<22} {kz_icon} {kz:>12}")
        if result.get('range_tp_override') or result.get('range_sl_override'):
            print(f"  {'  Range Override':<22} {'✅':>12}  TP/SL adjusted to range structure")

    # M22 Macro Regime summary (aggregated from M23-M65)
    m22 = result.get('m22', {})
    if m22 and m22.get('status') != 'SKIP':
        m22_regime = m22.get('regime', '?')
        m22_sev = m22.get('severity', '?')
        m22_sc = m22.get('score', 0.5)
        m22_sm = m22.get('size_mult', 1.0)
        infl = m22.get('inflation_label', '?')
        labor = m22.get('labor_label', '?')
        policy = m22.get('policy_label', '?')
        glob = m22.get('global_label', '?')
        sev_icons = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'}
        sev_icon = sev_icons.get(m22_sev, '⚪')
        print(f"  {'M22 Macro Regime':<22} {sev_icon} {m22_regime:>16}  score={m22_sc:.3f}")
        print(f"  {'  Signals':<22} infl={infl}  labor={labor}  policy={policy}")
        if glob != 'NO_DATA':
            print(f"  {'  Global CBs':<22} {glob}")
        if m22_sm < 1.0:
            print(f"  {'  ⚠️ Size Reduce':<22} {m22_sm:.2f}x")

    # M23 PPI + CPI + Claims Session summary
    m23 = result.get('m23', {})
    if m23 and m23.get('status') not in ('SKIP', 'NO_PPI', 'NO_DATA', 'ERROR'):
        m23_regime = m23.get('regime', '?')
        m23_fade = m23.get('fade_rate', 0)
        m23_us_dir = m23.get('us_direction', '?')
        m23_us_mag = m23.get('us_magnitude', '?')
        m23_rel_type = m23.get('release_type', 'PPI')
        m23_spike_acc = m23.get('spike_accuracy', 0.70)
        claims_today = m23.get('claims_today', False)

        if m23_regime == 'CLAIMS_RELEASE':
            # Standalone claims release
            claims = m23.get('claims', {})
            combo = m23.get('combo', {})
            cls_icons = {'LOW': '🟢', 'NORMAL': '⚪', 'ELEVATED': '🟡', 'SPIKE': '🟠', 'CRISIS': '🔴'}
            cls_icon = cls_icons.get(claims.get('classification', ''), '⚪')
            trend_icons = {'RISING': '📈', 'FALLING': '📉', 'STABLE': '➡️'}
            trend_icon = trend_icons.get(claims.get('trend', ''), '➡️')
            sig_icons = {'STRONG_BUY': '🟢🟢', 'BUY': '🟢', 'HOLD': '⚪', 'SELL': '🔴', 'STRONG_SELL': '🔴🔴'}
            sig_icon = sig_icons.get(combo.get('signal', ''), '⚪')
            print(f"  {'M23 Claims Release':<22} {cls_icon} {claims.get('current', 0)}K ({claims.get('classification', '?')})  {trend_icon} {claims.get('trend', '?')}")
            print(f"  {'  Macro Combo':<22} {sig_icon} {combo.get('signal', '?')}  expected={combo.get('expected_eth_move', 0):+.1f}%  Fed={combo.get('fed_action', '?')}")
            if claims.get('sahm_triggered'):
                print(f"  {'  🚨 Sahm Rule':<22} TRIGGERED — recession indicator")
        else:
            # PPI/CPI release (with optional claims context)
            us_icon = {'DUMP': '🔴', 'RALLY': '🟢', 'FLAT': '⚪'}.get(m23_us_dir, '⚪')
            type_icons = {'PPI': '🏭', 'CPI': '🛒', 'BOTH': '📊'}
            type_icon = type_icons.get(m23_rel_type, '📊')
            claims_tag = ' + 📋' if claims_today else ''
            print(f"  {'M23 ' + m23_rel_type + claims_tag + ' Release':<22} {us_icon} {m23_us_dir:>8} {m23_us_mag}  fade={m23_fade:.0%}  spike_acc={m23_spike_acc:.0%}")

            # Claims context inline
            claims_ctx = m23.get('claims_context')
            if claims_ctx:
                claims = claims_ctx.get('claims', {})
                combo = claims_ctx.get('combo', {})
                cls_icons = {'LOW': '🟢', 'NORMAL': '⚪', 'ELEVATED': '🟡', 'SPIKE': '🟠', 'CRISIS': '🔴'}
                cls_icon = cls_icons.get(claims.get('classification', ''), '⚪')
                sig_icons = {'STRONG_BUY': '🟢🟢', 'BUY': '🟢', 'HOLD': '⚪', 'SELL': '🔴', 'STRONG_SELL': '🔴🔴'}
                sig_icon = sig_icons.get(combo.get('signal', ''), '⚪')
                print(f"  {'  Claims Context':<22} {cls_icon} {claims.get('current', 0)}K ({claims.get('classification', '?')})  combo={sig_icon}{combo.get('signal', '?')}")
                if combo.get('ppi_leading_cpi'):
                    print(f"  {'  ⚠️ PPI→CPI lag':<22} PPI leads CPI by {combo.get('ppi_cpi_gap', 0):.1f}pp — CPI will follow UP")
                if claims.get('sahm_triggered'):
                    print(f"  {'  🚨 Sahm Rule':<22} TRIGGERED — recession indicator")

        # Show prediction or analysis
        m23_det = m23.get('details', {})
        m23_pred = m23_det.get('asia_prediction', {})
        m23_asia = m23_det.get('asia_analysis', {})
        if m23_pred:
            bias = m23_pred.get('regime_bias', '?')
            conf = m23_pred.get('confidence', '?')
            expected = m23_pred.get('expected_asia_move', 0)
            conf_icon = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(conf, '⚪')
            print(f"  {'  Asia Predict':<22} {conf_icon} {bias:>8}  expected={expected:+.2f}%")
        elif m23_asia:
            asia_move = m23_asia.get('asia_move', 0)
            gap_held = m23_asia.get('gap_held', False)
            pattern = m23_asia.get('pattern', '?')
            gap_icon = '✅' if gap_held else '❌'
            asia_icon = '🟢' if asia_move > 0 else '🔴'
            print(f"  {'  Asia Actual':<22} {asia_icon} {asia_move:+.2f}%  gap={gap_icon}  {pattern}")

    # Cascade Meta-Aggregator summary
    cascade = result.get('cascade', {})
    if cascade and cascade.get('active_count', 0) > 0:
        c_score = cascade.get('combined_score', 0.5)
        c_signal = cascade.get('combined_signal', 'HOLD')
        c_count = cascade.get('active_count', 0)
        c_weight = cascade.get('total_weight', 0)
        sig_icons = {'STRONG_BUY': '🟢🟢', 'BUY': '🟢', 'HOLD': '⚪', 'SELL': '🔴', 'STRONG_SELL': '🔴🔴'}
        sig_icon = sig_icons.get(c_signal, '⚪')
        print(f"  {'Cascade Global':<22} {sig_icon} {c_signal:>12}  score={c_score:.3f}  active={c_count}  wt={c_weight:.0%}")
        for ac in cascade.get('active_cascades', []):
            ac_name = ac.get('name', '?')
            ac_sig = ac.get('signal', '?')
            ac_sc = ac.get('score', 0.5)
            ac_wt = ac.get('weight', 0)
            ac_icon = sig_icons.get(ac_sig, '⚪')
            print(f"    {ac_name:<20} {ac_icon} {ac_sig:>12}  score={ac_sc:.3f}  wt={ac_wt:.0%}")

    # M24 NBS PMI Session Bias summary
    m24 = result.get('m24', {})
    if m24 and m24.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m24_bias = m24.get('bias', '?')
        m24_mfg = m24.get('mfg_pmi', 0)
        m24_svc = m24.get('services_pmi', 0)
        m24_signal = m24.get('pmi_signal', '?')
        m24_econ = m24.get('economic_regime', '?')
        m24_avg = m24.get('avg_ret_24h', 0)
        m24_win = m24.get('win_rate', 0)
        m24_n = m24.get('sample_size', 0)
        m24_conf = m24.get('confidence', 0)
        m24_adj = m24.get('score_adj', 0)
        m24_sm = m24.get('size_mult', 1.0)
        m24_src = m24.get('source', '?')
        bias_icon = '🟢' if m24_bias == 'LONG' else '🔴' if m24_bias == 'SHORT' else '⚪'
        conf_icon = '🟢' if m24_conf >= 0.7 else '🟡' if m24_conf >= 0.4 else '🟠'
        print(f"  {'M24 NBS PMI':<22} {bias_icon} {m24_bias:>8}  mfg={m24_mfg:.1f}({m24_signal}) svc={m24_svc:.1f}  econ={m24_econ}")
        print(f"  {'  Backtest':<22} {conf_icon} 24h={m24_avg:+.2f}%  win={m24_win*100:.0f}%  n={m24_n}  src={m24_src}  adj={m24_adj:+.3f}  size={m24_sm:.2f}x")

    # M25 Caixin PMI Session Bias summary
    m25 = result.get('m25', {})
    if m25 and m25.get('status') not in ('SKIP', 'NO_DATA', 'ERROR'):
        m25_regime = m25.get('regime', '?')
        m25_surprise = m25.get('surprise', '?')
        m25_bias = m25.get('bias', '?')
        m25_actual = m25.get('caixin_actual', 0)
        m25_prev = m25.get('caixin_prev', 0)
        m25_avg = m25.get('avg_ret_24h', 0)
        m25_win = m25.get('win_rate', 0)
        m25_n = m25.get('sample_size', 0)
        m25_adj = m25.get('score_adj', 0)
        m25_sm = m25.get('size_mult', 1.0)
        m25_p0b = m25.get('phase0_bucket', '?')
        m25_tb = m25.get('trend_bucket', '?')
        bias_icon = '🟢' if m25_bias == 'LONG' else '🔴' if m25_bias == 'SHORT' else '⚪'
        if m25_bias in ('LONG', 'SHORT'):
            print(f"  {'M25 Caixin PMI':<22} {bias_icon} {m25_surprise:>12}  actual={m25_actual:.1f} prev={m25_prev:.1f}  24h={m25_avg:+.2f}%  win={m25_win*100:.0f}%  n={m25_n}")
            print(f"  {'  Filters':<22} Phase0={m25_p0b}  Trend={m25_tb}  adj={m25_adj:+.3f}  size={m25_sm:.2f}x")
        else:
            print(f"  {'M25 Caixin PMI':<22} {bias_icon} {m25_surprise:>12}  {m25_regime}")

    # M33 US Retail Sales Session Bias summary
    m33 = result.get('m33', {})
    if m33 and m33.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m33_bias = m33.get('bias', '?')
        m33_retail_mom = m33.get('retail_mom', 0)
        m33_core = m33.get('core_mom', 0)
        m33_consensus = m33.get('consensus', 0)
        m33_surprise = m33.get('surprise', 0)
        m33_signal = m33.get('signal', '?')
        m33_avg = m33.get('avg_ret_24h', 0)
        m33_win = m33.get('win_rate', 0)
        m33_n = m33.get('sample_size', 0)
        m33_conf = m33.get('confidence', 0)
        m33_adj = m33.get('score_adj', 0)
        m33_sm = m33.get('size_mult', 1.0)
        m33_src = m33.get('source', '?')
        bias_icon = '🟢' if m33_bias == 'LONG' else '🔴' if m33_bias == 'SHORT' else '⚪'
        conf_icon = '🟢' if m33_conf >= 0.7 else '🟡' if m33_conf >= 0.4 else '🟠'
        print(f"  {'M33 Retail Sales':<22} {bias_icon} {m33_bias:>8}  retail={m33_retail_mom:+.1f}% cons={m33_consensus:+.1f}% surp={m33_surprise:+.2f}")
        print(f"  {'  Backtest':<22} {conf_icon} 24h={m33_avg:+.2f}%  win={m33_win*100:.0f}%  n={m33_n}  src={m33_src}  adj={m33_adj:+.3f}  size={m33_sm:.2f}x")

    # M34 US Housing Starts Session Bias summary
    m34 = result.get('m34', {})
    if m34 and m34.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m34_bias = m34.get('bias', '?')
        m34_starts = m34.get('starts_k', 0)
        m34_permits = m34.get('permits_k', 0)
        m34_starts_mom = m34.get('starts_mom', 0)
        m34_permits_mom = m34.get('permits_mom', 0)
        m34_signal = m34.get('signal', '?')
        m34_avg = m34.get('avg_ret_24h', 0)
        m34_win = m34.get('win_rate', 0)
        m34_n = m34.get('sample_size', 0)
        m34_conf = m34.get('confidence', 0)
        m34_adj = m34.get('score_adj', 0)
        m34_sm = m34.get('size_mult', 1.0)
        m34_src = m34.get('source', '?')
        bias_icon = '🟢' if m34_bias == 'LONG' else '🔴' if m34_bias == 'SHORT' else '⚪'
        conf_icon = '🟢' if m34_conf >= 0.7 else '🟡' if m34_conf >= 0.4 else '🟠'
        print(f"  {'M34 Housing Starts':<22} {bias_icon} {m34_bias:>8}  starts={m34_starts}K({m34_starts_mom:+.1f}%) permits={m34_permits}K({m34_permits_mom:+.1f}%)  signal={m34_signal}")
        print(f"  {'  Backtest':<22} {conf_icon} 24h={m34_avg:+.2f}%  win={m34_win*100:.0f}%  n={m34_n}  src={m34_src}  adj={m34_adj:+.3f}  size={m34_sm:.2f}x")

    # M35 PBoC LPR Session Bias summary
    m35 = result.get('m35', {})
    if m35 and m35.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m35_bias = m35.get('bias', '?')
        m35_lpr1y = m35.get('lpr_1y', 0)
        m35_lpr5y = m35.get('lpr_5y', 0)
        m35_cut1y = m35.get('cut_1y', 0)
        m35_cut5y = m35.get('cut_5y', 0)
        m35_signal = m35.get('signal', '?')
        m35_avg = m35.get('avg_ret_24h', 0)
        m35_win = m35.get('win_rate', 0)
        m35_n = m35.get('sample_size', 0)
        m35_conf = m35.get('confidence', 0)
        m35_adj = m35.get('score_adj', 0)
        m35_sm = m35.get('size_mult', 1.0)
        bias_icon = '🟢' if m35_bias == 'LONG' else '🔴' if m35_bias == 'SHORT' else '⚪'
        conf_icon = '🟢' if m35_conf >= 0.7 else '🟡' if m35_conf >= 0.4 else '🟠'
        cut_icon = '📉' if m35_cut1y > 0 or m35_cut5y > 0 else '📈' if m35_cut1y < 0 else '➡️'
        print(f"  {'M35 PBoC LPR':<22} {bias_icon} {m35_bias:>8}  1Y={m35_lpr1y:.2f}%({m35_cut1y:+.2f}%) 5Y={m35_lpr5y:.2f}%({m35_cut5y:+.2f}%) {cut_icon}  signal={m35_signal}")
        print(f"  {'  Backtest':<22} {conf_icon} 24h={m35_avg:+.2f}%  win={m35_win*100:.0f}%  n={m35_n}  adj={m35_adj:+.3f}  size={m35_sm:.2f}x")

    # M36 ADP Employment Session Bias summary
    m36 = result.get('m36', {})
    if m36 and m36.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m36_bias = m36.get('bias', '?')
        m36_adp = m36.get('adp_k', 0)
        m36_cons = m36.get('consensus_k', 0)
        m36_surprise = m36.get('surprise', 0)
        m36_signal = m36.get('signal', '?')
        m36_avg = m36.get('avg_ret_24h', 0)
        m36_win = m36.get('win_rate', 0)
        m36_n = m36.get('sample_size', 0)
        m36_conf = m36.get('confidence', 0)
        m36_adj = m36.get('score_adj', 0)
        m36_sm = m36.get('size_mult', 1.0)
        bias_icon = '🟢' if m36_bias == 'LONG' else '🔴' if m36_bias == 'SHORT' else '⚪'
        conf_icon = '🟢' if m36_conf >= 0.7 else '🟡' if m36_conf >= 0.4 else '🟠'
        print(f"  {'M36 ADP Employment':<22} {bias_icon} {m36_bias:>8}  ADP={m36_adp}K cons={m36_cons}K surp={m36_surprise:+.0f}K  signal={m36_signal}")
        print(f"  {'  Backtest':<22} {conf_icon} 24h={m36_avg:+.2f}%  win={m36_win*100:.0f}%  n={m36_n}  adj={m36_adj:+.3f}  size={m36_sm:.2f}x  ⚠️ trap risk")

    # M37 NFP Session Bias summary
    m37 = result.get('m37', {})
    if m37 and m37.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m37_bias = m37.get('bias', '?')
        m37_nfp = m37.get('nfp_k', 0)
        m37_cons = m37.get('consensus_k', 0)
        m37_surprise = m37.get('surprise', 0)
        m37_signal = m37.get('signal', '?')
        m37_avg = m37.get('avg_ret_24h', 0)
        m37_win = m37.get('win_rate', 0)
        m37_n = m37.get('sample_size', 0)
        m37_conf = m37.get('confidence', 0)
        m37_adj = m37.get('score_adj', 0)
        m37_sm = m37.get('size_mult', 1.0)
        bias_icon = '🟢' if m37_bias == 'LONG' else '🔴' if m37_bias == 'SHORT' else '⚪'
        conf_icon = '🟢' if m37_conf >= 0.7 else '🟡' if m37_conf >= 0.4 else '🟠'
        print(f"  {'M37 NFP':<22} {bias_icon} {m37_bias:>8}  NFP={m37_nfp:,}K cons={m37_cons:,}K surp={m37_surprise:+,}K  signal={m37_signal}")
        print(f"  {'  Backtest':<22} {conf_icon} 24h={m37_avg:+.2f}%  win={m37_win*100:.0f}%  n={m37_n}  adj={m37_adj:+.3f}  size={m37_sm:.2f}x  chain: London→NY 73-90%")

    # M38 Germany Ifo Session Bias summary
    m38 = result.get('m38', {})
    if m38 and m38.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m38_bias = m38.get('bias', '?')
        m38_actual = m38.get('actual', 0)
        m38_cons = m38.get('consensus', 0)
        m38_surprise = m38.get('surprise', 0)
        m38_signal = m38.get('signal', '?')
        m38_trend = m38.get('trend', '?')
        m38_avg = m38.get('avg_ret_24h', 0)
        m38_win = m38.get('win_rate', 0)
        m38_n = m38.get('sample_size', 0)
        m38_conf = m38.get('confidence', 0)
        m38_adj = m38.get('score_adj', 0)
        m38_sm = m38.get('size_mult', 1.0)
        bias_icon = '🟢' if m38_bias == 'LONG' else '🔴' if m38_bias == 'SHORT' else '⚪'
        conf_icon = '🟢' if m38_conf >= 0.7 else '🟡' if m38_conf >= 0.4 else '🟠'
        trend_icon = '📈' if m38_trend == 'IMPROVING' else '📉' if m38_trend == 'DETERIORATING' else '➡️'
        print(f"  {'M38 Germany Ifo':<22} {bias_icon} {m38_bias:>8}  actual={m38_actual} cons={m38_cons} trend={trend_icon}{m38_trend}")
        print(f"  {'  Backtest':<22} {conf_icon} 24h={m38_avg:+.2f}%  win={m38_win*100:.0f}%  n={m38_n}  adj={m38_adj:+.3f}  size={m38_sm:.2f}x  chain: Fr→London→NY 78-88%")

    # M39 Michigan Consumer Sentiment summary
    m39 = result.get('m39', {})
    if m39 and m39.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m39_bias = m39.get('bias', '?')
        m39_headline = m39.get('headline', 0)
        m39_cons = m39.get('consensus', 0)
        m39_surprise = m39.get('surprise', 0)
        m39_signal = m39.get('signal', '?')
        m39_infl = m39.get('infl_5yr', 0)
        m39_infl_sig = m39.get('infl_signal', '?')
        m39_type = m39.get('type', '?')
        m39_avg = m39.get('avg_ret_24h', 0)
        m39_win = m39.get('win_rate', 0)
        m39_n = m39.get('sample_size', 0)
        m39_conf = m39.get('confidence', 0)
        m39_adj = m39.get('score_adj', 0)
        m39_sm = m39.get('size_mult', 1.0)
        bias_icon = '🟢' if m39_bias == 'LONG' else '🔴' if m39_bias == 'SHORT' else '⚪'
        conf_icon = '🟢' if m39_conf >= 0.7 else '🟡' if m39_conf >= 0.4 else '🟠'
        infl_icon = '🔴' if m39_infl_sig == 'INFL_UP' else '🟢' if m39_infl_sig == 'INFL_DOWN' else '⚪'
        print(f"  {'M39 Michigan Sent':<22} {bias_icon} {m39_bias:>8}  {m39_type} headline={m39_headline} cons={m39_cons} 5yr={m39_infl}%{infl_icon}")
        print(f"  {'  Backtest':<22} {conf_icon} 24h={m39_avg:+.2f}%  win={m39_win*100:.0f}%  n={m39_n}  adj={m39_adj:+.3f}  size={m39_sm:.2f}x  weekend: 68.2% continuity")

    # M40 Germany CPI Session Bias summary
    m40 = result.get('m40', {})
    if m40 and m40.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m40_bias = m40.get('bias', '?')
        m40_cpi = m40.get('cpi_yoy', 0)
        m40_cons = m40.get('consensus', 0)
        m40_signal = m40.get('signal', '?')
        m40_level = m40.get('cpi_level', '?')
        m40_dir = m40.get('cpi_direction', '?')
        m40_avg = m40.get('avg_ret_24h', 0)
        m40_win = m40.get('win_rate', 0)
        m40_n = m40.get('sample_size', 0)
        m40_conf = m40.get('confidence', 0)
        m40_adj = m40.get('score_adj', 0)
        m40_sm = m40.get('size_mult', 1.0)
        bias_icon = '🟢' if m40_bias == 'LONG' else '🔴' if m40_bias == 'SHORT' else '⚪'
        conf_icon = '🟢' if m40_conf >= 0.7 else '🟡' if m40_conf >= 0.4 else '🟠'
        dir_icon = '📈' if m40_dir == 'RISING' else '📉' if m40_dir == 'FALLING' else '➡️'
        print(f"  {'M40 Germany CPI':<22} {bias_icon} {m40_bias:>8}  CPI={m40_cpi:.1f}% cons={m40_cons:.1f}% {dir_icon}{m40_dir} level={m40_level} signal={m40_signal}")
        print(f"  {'  Backtest':<22} {conf_icon} 24h={m40_avg:+.2f}%  win={m40_win*100:.0f}%  n={m40_n}  adj={m40_adj:+.3f}  size={m40_sm:.2f}x  chain: Asia 89-93%, Lon→NY 77-89%")

    # M41 Eurozone CPI Flash Session Bias summary
    m41 = result.get('m41', {})
    if m41 and m41.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m41_bias = m41.get('bias', '?')
        m41_hicp = m41.get('hicp_yoy', 0)
        m41_core = m41.get('core_yoy', 0)
        m41_cons = m41.get('consensus', 0)
        m41_signal = m41.get('signal', '?')
        m41_level = m41.get('cpi_level', '?')
        m41_dir = m41.get('cpi_direction', '?')
        m41_spread = m41.get('core_spread', '?')
        m41_avg = m41.get('avg_ret_24h', 0)
        m41_win = m41.get('win_rate', 0)
        m41_n = m41.get('sample_size', 0)
        m41_conf = m41.get('confidence', 0)
        m41_adj = m41.get('score_adj', 0)
        m41_sm = m41.get('size_mult', 1.0)
        bias_icon = '🟢' if m41_bias == 'LONG' else '🔴' if m41_bias == 'SHORT' else '⚪'
        conf_icon = '🟢' if m41_conf >= 0.7 else '🟡' if m41_conf >= 0.4 else '🟠'
        dir_icon = '📈' if m41_dir == 'RISING' else '📉' if m41_dir == 'FALLING' else '➡️'
        spread_icon = '🔴' if 'STICKY' in m41_spread else '🟢' if m41_spread == 'CORE_BELOW' else '⚪'
        print(f"  {'M41 EZ CPI Flash':<22} {bias_icon} {m41_bias:>8}  HICP={m41_hicp:.1f}% cons={m41_cons:.1f}% {dir_icon}{m41_dir} core={m41_core:.1f}%{spread_icon}{m41_spread}")
        print(f"  {'  Backtest':<22} {conf_icon} 24h={m41_avg:+.2f}%  win={m41_win*100:.0f}%  n={m41_n}  adj={m41_adj:+.3f}  size={m41_sm:.2f}x  chain: Asia 88-99%, Lon→NY 75-100%  p=0.013✅")

    # M42 Eurozone GDP Flash Session Bias summary
    m42 = result.get('m42', {})
    if m42 and m42.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m42_bias = m42.get('bias', '?')
        m42_gdp = m42.get('gdp_qoq', 0)
        m42_yoy = m42.get('gdp_yoy', 0)
        m42_cons = m42.get('consensus_qoq', 0)
        m42_signal = m42.get('signal', '?')
        m42_health = m42.get('health', '?')
        m42_mom = m42.get('momentum', '?')
        m42_avg = m42.get('avg_ret_24h', 0)
        m42_win = m42.get('win_rate', 0)
        m42_n = m42.get('sample_size', 0)
        m42_conf = m42.get('confidence', 0)
        m42_adj = m42.get('score_adj', 0)
        m42_sm = m42.get('size_mult', 1.0)
        bias_icon = '🟢' if m42_bias == 'LONG' else '🔴' if m42_bias == 'SHORT' else '⚪'
        conf_icon = '🟠'  # always orange — small sample
        health_icon = {'RECESSION': '🔴', 'CONTRACTION': '🟠', 'STAGNANT': '🟡',
                       'MODERATE': '🟢', 'STRONG': '🟢🟢'}.get(m42_health, '⚪')
        mom_icon = {'ACCELERATING': '📈', 'DECELERATING': '📉', 'STABLE': '➡️'}.get(m42_mom, '➡️')
        print(f"  {'M42 EZ GDP Flash':<22} {bias_icon} {m42_bias:>8}  GDP={m42_gdp:+.1f}%(qoq) cons={m42_cons:+.1f}% {health_icon}{m42_health} {mom_icon}{m42_mom}")
        print(f"  {'  Backtest':<22} {conf_icon} 24h={m42_avg:+.2f}%  win={m42_win*100:.0f}%  n={m42_n}  adj={m42_adj:+.3f}  size={m42_sm:.2f}x  chain: Asia 81-100%, Lon→NY 78-100%  (small n)")

    # M43 US GDP Advance Estimate Session Bias summary
    m43 = result.get('m43', {})
    if m43 and m43.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m43_bias = m43.get('bias', '?')
        m43_gdp = m43.get('gdp_qoq', 0)
        m43_cons = m43.get('consensus', 0)
        m43_signal = m43.get('signal', '?')
        m43_health = m43.get('health', '?')
        m43_mom = m43.get('momentum', '?')
        m43_avg = m43.get('avg_ret_24h', 0)
        m43_win = m43.get('win_rate', 0)
        m43_n = m43.get('sample_size', 0)
        m43_conf = m43.get('confidence', 0)
        m43_adj = m43.get('score_adj', 0)
        m43_sm = m43.get('size_mult', 1.0)
        bias_icon = '🟢' if m43_bias == 'LONG' else '🔴' if m43_bias == 'SHORT' else '⚪'
        conf_icon = '🟠'
        health_icon = {'RECESSION': '🔴', 'CONTRACTION': '🟠', 'SLOW': '🟡',
                       'MODERATE': '🟢', 'STRONG': '🟢🟢'}.get(m43_health, '⚪')
        mom_icon = {'ACCELERATING': '📈', 'DECELERATING': '📉', 'STABLE': '➡️'}.get(m43_mom, '➡️')
        print(f"  {'M43 US GDP Advance':<22} {bias_icon} {m43_bias:>8}  GDP={m43_gdp:+.1f}%(qoq) cons={m43_cons:+.1f}% {health_icon}{m43_health} {mom_icon}{m43_mom}  ⚡CONTRARIAN")
        print(f"  {'  Backtest':<22} {conf_icon} 24h={m43_avg:+.2f}%  win={m43_win*100:.0f}%  n={m43_n}  adj={m43_adj:+.3f}  size={m43_sm:.2f}x  miss→rally, beat→sell")

    # M44 US Durable Goods Orders Session Bias summary
    m44 = result.get('m44', {})
    if m44 and m44.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m44_bias = m44.get('bias', '?')
        m44_headline = m44.get('headline_mom', 0)
        m44_core = m44.get('core_mom', 0)
        m44_cons = m44.get('consensus', 0)
        m44_signal = m44.get('signal', '?')
        m44_capex = m44.get('capex_health', '?')
        m44_mom = m44.get('momentum', '?')
        m44_avg = m44.get('avg_ret_24h', 0)
        m44_win = m44.get('win_rate', 0)
        m44_n = m44.get('sample_size', 0)
        m44_adj = m44.get('score_adj', 0)
        m44_sm = m44.get('size_mult', 1.0)
        bias_icon = '🟢' if m44_bias == 'LONG' else '🔴' if m44_bias == 'SHORT' else '⚪'
        capex_icon = {'BOOMING': '🟢🟢', 'GROWING': '🟢', 'STABLE': '⚪',
                      'WEAK': '🟠', 'COLLAPSING': '🔴'}.get(m44_capex, '⚪')
        mom_icon = {'ACCELERATING': '📈', 'DECELERATING': '📉', 'STABLE': '➡️'}.get(m44_mom, '➡️')
        print(f"  {'M44 Durables':<22} {bias_icon} {m44_bias:>8}  headline={m44_headline:+.1f}% cons={m44_cons:+.1f}% core={m44_core:+.1f}% {capex_icon}{m44_capex} {mom_icon}{m44_mom}")
        print(f"  {'  Backtest':<22} 24h={m44_avg:+.2f}%  win={m44_win*100:.0f}%  n={m44_n}  adj={m44_adj:+.3f}  size={m44_sm:.2f}x  chain: Asia 88-100%, breaks at Lon→NY")

    # M45 Core PCE + Personal Spending Session Bias summary
    m45 = result.get('m45', {})
    if m45 and m45.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m45_bias = m45.get('bias', '?')
        m45_core = m45.get('core_pce_yoy', 0)
        m45_mom = m45.get('core_pce_mom', 0)
        m45_spend = m45.get('spending_mom', 0)
        m45_cons = m45.get('consensus_yoy', 0)
        m45_signal = m45.get('signal', '?')
        m45_infl = m45.get('inflation', '?')
        m45_spend_sig = m45.get('spending_signal', '?')
        m45_avg = m45.get('avg_ret_24h', 0)
        m45_win = m45.get('win_rate', 0)
        m45_n = m45.get('sample_size', 0)
        m45_adj = m45.get('score_adj', 0)
        m45_sm = m45.get('size_mult', 1.0)
        bias_icon = '🟢' if m45_bias == 'LONG' else '🔴' if m45_bias == 'SHORT' else '⚪'
        infl_icon = {'RUNAWAY': '🔴🔴', 'HOT': '🔴', 'WARM': '🟠',
                     'TARGET': '🟢', 'COOL': '🟢🟢', 'DEFLATION': '⚪'}.get(m45_infl, '⚪')
        spend_icon = {'SURGING': '📈', 'STRONG': '🟢', 'MODERATE': '⚪', 'WEAK': '🔴'}.get(m45_spend_sig, '⚪')
        print(f"  {'M45 Core PCE':<22} {bias_icon} {m45_bias:>8}  PCE={m45_core:.1f}%(yoy) cons={m45_cons:.1f}% {infl_icon}{m45_infl} spend={m45_spend:+.1f}%{spend_icon}{m45_spend_sig}")
        print(f"  {'  Backtest':<22} 24h={m45_avg:+.2f}%  win={m45_win*100:.0f}%  n={m45_n}  adj={m45_adj:+.3f}  size={m45_sm:.2f}x  🏛️FOMC metric  weekend lock")

    # M46 Japan CPI (Tokyo Flash) Session Bias summary
    m46 = result.get('m46', {})
    if m46 and m46.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m46_bias = m46.get('bias', '?')
        m46_cpi = m46.get('cpi_yoy', 0)
        m46_core = m46.get('core_yoy', 0)
        m46_cons = m46.get('consensus', 0)
        m46_signal = m46.get('signal', '?')
        m46_infl = m46.get('inflation', '?')
        m46_core_p = m46.get('core_pressure', '?')
        m46_avg = m46.get('avg_ret_24h', 0)
        m46_win = m46.get('win_rate', 0)
        m46_n = m46.get('sample_size', 0)
        m46_adj = m46.get('score_adj', 0)
        m46_sm = m46.get('size_mult', 1.0)
        bias_icon = '🟢' if m46_bias == 'LONG' else '🔴' if m46_bias == 'SHORT' else '⚪'
        core_icon = {'EXTREME': '🔴🔴', 'HAWKISH': '🔴', 'MODERATE': '🟡',
                     'DOVISH': '🟢'}.get(m46_core_p, '⚪')
        print(f"  {'M46 Japan CPI':<22} {bias_icon} {m46_bias:>8}  CPI={m46_cpi:.1f}% cons={m46_cons:.1f}% core={m46_core:.1f}% {core_icon}BoJ:{m46_core_p}")
        print(f"  {'  Backtest':<22} 24h={m46_avg:+.2f}%  win={m46_win*100:.0f}%  n={m46_n}  adj={m46_adj:+.3f}  size={m46_sm:.2f}x  chain: Tokyo→NY 67-100%  🏛️BoJ metric")

    # M26 Eurozone Flash PMI Session Bias summary
    m26 = result.get('m26', {})
    if m26 and m26.get('status') not in ('SKIP', 'NO_EDGE', 'ERROR'):
        m26_bias = m26.get('bias', '?')
        m26_actual = m26.get('actual', 0)
        m26_consensus = m26.get('consensus', 0)
        m26_surprise = m26.get('surprise', 0)
        m26_signal = m26.get('signal', '?')
        m26_avg = m26.get('avg_ret_entry_to_exit', 0)
        m26_win = m26.get('win_rate', 0)
        m26_n = m26.get('sample_size', 0)
        m26_conf = m26.get('confidence', 0)
        m26_adj = m26.get('score_adj', 0)
        m26_sm = m26.get('size_mult', 1.0)
        m26_phase = m26.get('session_phase', '?')
        m26_entry = m26.get('entry_session', '?')
        m26_exit = m26.get('exit_session', '?')
        m26_eu_uk = m26.get('persistence_eu_uk', 0)
        m26_uk_us = m26.get('persistence_uk_us', 0)
        bias_icon = '🟢' if m26_bias == 'LONG' else '🔴' if m26_bias == 'SHORT' else '⚪'
        conf_icon = '🟢' if m26_conf >= 0.7 else '🟡' if m26_conf >= 0.4 else '🟠'
        phase_icon = {'IN_WINDOW': '✅', 'FADING': '⚠️', 'OUT_OF_WINDOW': '❌'}.get(m26_phase, '❓')
        print(f"  {'M26 EZ PMI':<22} {bias_icon} {m26_bias:>8}  actual={m26_actual:.1f} cons={m26_consensus:.1f} surp={m26_surprise:+.1f}  signal={m26_signal}")
        print(f"  {'  Chain':<22} {conf_icon} {m26_entry}→{m26_exit}  EU→UK={m26_eu_uk:.0f}%  UK→US={m26_uk_us:.0f}%  avg={m26_avg:+.2f}%  win={m26_win*100:.0f}%  n={m26_n}")
        print(f"  {'  Session':<22} {phase_icon} {m26_phase}  adj={m26_adj:+.3f}  size={m26_sm:.2f}x")

    # Breakout Confirmation summary
    bc = result.get('breakout_confirm')
    if bc and bc.get('filters'):
        bc_status = bc.get('status', '?')
        bc_passed = bc.get('passed', 0)
        bc_total = bc.get('total', 0)
        bc_icons = {'CONFIRMED': '✅', 'WEAK': '⚠️', 'REJECTED': '❌'}
        bc_icon = bc_icons.get(bc_status, '?')
        print(f"  {'M19 Breakout':<22} {bc_status:>10}  {bc_passed}/{bc_total} {bc_icon}")
        for f in bc.get('filters', []):
            check = '✅' if f['passed'] else '❌'
            print(f"    {check} {f['name']}")

    ics = result.get('ics', 0)
    status = result.get('status', 'N/A')
    direction = result.get('direction', 'N/A')
    print(f"\n  ICS: {ics:.3f}", end="")
    if 'threshold' in result:
        print(f" (threshold {result['threshold']:.2f})", end="")
    print(f" → {'SIGNAL: ' + direction if status == 'SIGNAL' else 'NO SIGNAL'}")

    # Key observations
    print(f"\n  Key observations:")

    # Macro Event Filter
    mef = result.get('macro_event_filter', {})
    if mef:
        if mef.get('blocked'):
            print(f"  • 🚫 MACRO EVENT BLOCKED: {mef.get('reason', '')}")
        elif mef.get('size_mult', 1.0) < 1.0:
            print(f"  • ⚠️ MACRO EVENT SIZE REDUCTION: {mef.get('size_mult', 1.0):.2f}x")
        for note in mef.get('regime_notes', []):
            print(f"    {note}")
        for evt in mef.get('active_events', []):
            print(f"    📡 Active: {evt['name']} ({evt['country']}) {evt['hours_since']:+.1f}h — {evt['phase']}")

    # PPI/CPI session context
    m23 = result.get('m23', {})
    if m23 and m23.get('status') not in ('SKIP', 'NO_PPI', 'NO_RELEASE', 'NO_DATA', 'ERROR'):
        m23_det = m23.get('details', {})
        m23_pred = m23_det.get('asia_prediction', {})
        m23_asia = m23_det.get('asia_analysis', {})
        m23_type = m23.get('release_type', 'PPI')
        if m23_pred:
            bias = m23_pred.get('regime_bias', '?')
            conf = m23_pred.get('confidence', '?')
            expected = m23_pred.get('expected_asia_move', 0)
            us_dir = m23_pred.get('us_direction', '?')
            print(f"  • {m23_type} release: US {us_dir}, Asia bias={bias} ({conf}), expected {expected:+.2f}%")
        elif m23_asia:
            pattern = m23_asia.get('pattern', '?')
            gap_held = m23_asia.get('gap_held', False)
            print(f"  • {m23_type} release: Asia {pattern} (gap {'held' if gap_held else 'failed'})")

    # Direction resolver
    dr = result.get('direction_resolver', {})
    m9 = result.get('m9', {})
    m13 = result.get('m13', {})
    if dr:
        print(f"  • Direction: {dr.get('direction', '?')} via resolver (regime={m9.get('regime', '?')}, structure={m13.get('bias', '?')})")

    # Price & bias
    swing = result.get('swing_bias', '')
    phase0 = result.get('phase0')
    phase_str = f", phase0={phase0}" if phase0 else ""
    print(f"  • Price ${price:.2f}, daily bias {swing}{phase_str}")

    # Derivatives
    if deriv and 'error' not in deriv:
        ls = deriv.get('ls_ratio', 0)
        lp = deriv.get('long_pct', 0)
        pos = deriv.get('positioning', '')
        whale = deriv.get('whale_signal', '')
        taker = deriv.get('futures_flow', '')
        fr = deriv.get('funding_rate')
        oi_usd = deriv.get('oi_usd', 0)

        crowd_label = f"({'crowded' if abs(deriv.get('ls_zscore', 0)) > 1.5 else 'neutral'})"
        print(f"  • L/S ratio {ls:.2f} ({lp:.1f}% long {crowd_label}), whales={whale}")
        print(f"  • Futures taker: {taker}")
        if fr is not None:
            fr_dir = "longs pay" if fr > 0 else "shorts pay"
            print(f"  • Funding rate: {fr*100:+.4f}% ({fr_dir})")
        print(f"  • OI: ${oi_usd/1e9:.2f}B  Δ1h: {deriv.get('oi_roc_1h', 0):+.3f}%")

    # M10 macro summary
    m10 = result.get('m10', {})
    if m10 and m10.get('status') not in ('SKIP', 'ERROR', None):
        m10_comp = m10.get('details', {}).get('m10_components', {})
        m10_agree = m10.get('details', {}).get('macro_agreement', '')
        btc_roc = m10.get('details', {}).get('btc_roc7', '')
        ethbtc_roc = m10.get('details', {}).get('ethbtc_roc7', '')
        parts = []
        if btc_roc:
            parts.append(f"BTC 7d ROC={btc_roc:+.1%}")
        if ethbtc_roc:
            parts.append(f"ETH/BTC 7d ROC={ethbtc_roc:+.1%}")
        if m10_agree:
            parts.append(m10_agree)
        if parts:
            print(f"  • M10 macro: {m10['status']} ({', '.join(parts)})")

    # Exchange Activity summary
    exch = result.get('exchange_activity', {})
    if exch and 'error' not in exch:
        sigs = exch.get('signals', {})
        snaps = exch.get('snapshots', {})
        spread = sigs.get('funding_spread', 0)
        spread_trend = sigs.get('funding_spread_trend', '?')
        oi_shares = sigs.get('oi_shares', {})
        migration = sigs.get('oi_migration', 'BALANCED')
        ls_spread = sigs.get('ls_spread', 0)

        parts = []
        if spread > 0:
            parts.append(f"funding spread {spread*100:.4f}% ({spread_trend})")
        if oi_shares:
            dominant = sigs.get('oi_dominant_exchange', '?')
            parts.append(f"OI dominant: {dominant} ({oi_shares.get(dominant, 0)*100:.1f}%)")
        if migration and migration != 'BALANCED':
            parts.append(f"migration: {migration}")
        if ls_spread > 0.3:
            ls_by = sigs.get('ls_by_exchange', {})
            parts.append(f"L/S spread {ls_spread:.2f}")
        if parts:
            print(f"  • Exchange: {'; '.join(parts)}")

    # Spot market summary
    spot = exch.get('spot', {}) if exch else {}
    spot_sigs = exch.get('spot_signals', {}) if exch else {}
    if spot and spot_sigs:
        spot_parts = []
        basis_avg = spot_sigs.get('basis_avg', 0)
        basis_state = spot_sigs.get('basis_state', '?')
        if basis_avg:
            spot_parts.append(f"basis {basis_avg:+.3f}% ({basis_state})")
        ob_avg = spot_sigs.get('spot_ob_avg', 0)
        if ob_avg:
            spot_parts.append(f"OB ratio {ob_avg:.3f}")
        sell_walls = spot_sigs.get('spot_sell_walls', [])
        if sell_walls:
            spot_parts.append(f"sell walls: {', '.join(sell_walls)}")
        flow = spot_sigs.get('spot_flow', '?')
        if flow != '?':
            spot_parts.append(f"flow: {flow}")
        if spot_parts:
            print(f"  • Spot: {'; '.join(spot_parts)}")

    # Breakout confirmation summary
    bc = result.get('breakout_confirm')
    if bc and bc.get('filters'):
        bc_passed = bc.get('passed', 0)
        bc_total = bc.get('total', 0)
        bc_status = bc.get('status', '?')
        failed = [f['name'] for f in bc.get('filters', []) if not f['passed']]
        if bc_status == 'CONFIRMED':
            print(f"  • Breakout: ✅ CONFIRMED ({bc_passed}/{bc_total})")
        elif bc_status == 'WEAK':
            print(f"  • Breakout: ⚠️ WEAK ({bc_passed}/{bc_total}), failed: {', '.join(failed)}")
        else:
            print(f"  • Breakout: ❌ REJECTED ({bc_passed}/{bc_total}), failed: {', '.join(failed)}")

    # Liquidity — unswept only
    if liq:
        above = [z for z in liq.get('above', []) if not z.get('swept')]
        below = [z for z in liq.get('below', []) if not z.get('swept')]

        if above:
            targets = []
            for z in above[:3]:
                icon = '💥' if 'LIQ' in z['type'] else '🛑'
                targets.append(f"{icon}${z['price']:.0f}({z['dist_pct']:+.1f}%)")
            print(f"  • Unswept above: {', '.join(targets)}")
        else:
            print(f"  • Unswept above: none — all swept")

        if below:
            targets = []
            for z in below[:3]:
                icon = '💥' if 'LIQ' in z['type'] else '🛑'
                targets.append(f"{icon}${z['price']:.0f}({z['dist_pct']:+.1f}%)")
            print(f"  • Unswept below: {', '.join(targets)}")
        else:
            print(f"  • Unswept below: none — all swept")

    # Support/Resistance
    sr = result.get('sr_levels', [])
    if sr:
        sup = [p for p, s, t, _, _ in sr if t == 'SUPPORT'][:2]
        res = [p for p, s, t, _, _ in sr if t == 'RESISTANCE'][:2]
        if sup or res:
            parts = []
            if sup:
                parts.append(f"support: {', '.join(f'${p:.0f}' for p in sup)}")
            if res:
                parts.append(f"resistance: {', '.join(f'${p:.0f}' for p in res)}")
            print(f"  • {' | '.join(parts)}")

    # Conflict history
    conflict = result.get('conflict')
    if conflict and conflict.get('historical'):
        hist = conflict['historical']
        windows = hist.get('windows', {})
        is_conflict = conflict['is_conflict']
        if is_conflict:
            label = "CONFLICT"
        else:
            label = "ALIGNED"
        w24 = windows.get('24h')
        w48 = windows.get('48h')
        parts_c = [f"{hist['total_signals']} historical signals"]
        if w24:
            parts_c.append(f"24h rev={w24['reversal_rate']:.0f}% net={w24['avg_net']:+.2f}%")
        if w48:
            parts_c.append(f"48h rev={w48['reversal_rate']:.0f}% net={w48['avg_net']:+.2f}%")
        print(f"  • {label} {conflict['m1_direction']} M1 vs {conflict['daily_bias']} daily: {'; '.join(parts_c)}")

    # Verdict
    print(f"\n  Verdict: ", end="")
    if status == 'SIGNAL':
        print(f"✅ {direction} signal — entry ${result['entry']:.2f}, "
              f"SL ${result['sl']:.2f} ({result['sl_pct']:.2f}%)")
    else:
        reasons = []
        if ics < result.get('threshold', 0.5):
            reasons.append(f"ICS {ics:.3f} below threshold")
        if result.get('swing_bias') == 'BEARISH' and direction == 'LONG':
            reasons.append("conflict: M1 bullish but daily bearish")
        if result.get('swing_bias') == 'BULLISH' and direction == 'SHORT':
            reasons.append("conflict: M1 bearish but daily bullish")
        if deriv.get('positioning') == 'CROWDED_LONG' and direction == 'LONG':
            reasons.append("crowded long")
        if deriv.get('positioning') == 'CROWDED_SHORT' and direction == 'SHORT':
            reasons.append("crowded short")
        if deriv.get('whale_signal') == 'WHALE_BEARISH' and direction == 'LONG':
            reasons.append("whales bearish")
        if deriv.get('whale_signal') == 'WHALE_BULLISH' and direction == 'SHORT':
            reasons.append("whales bullish")
        if deriv.get('futures_flow') == 'SELLERS_DOMINANT' and direction == 'LONG':
            reasons.append("seller-dominated flow")
        if deriv.get('futures_flow') == 'BUYERS_DOMINANT' and direction == 'SHORT':
            reasons.append("buyer-dominated flow")
        if result.get('breakout_rejected'):
            bc = result.get('breakout_confirm', {})
            reasons.append(f"breakout rejected ({bc.get('passed', 0)}/{bc.get('total', 0)} filters)")
        if not reasons:
            reasons.append(result.get('reason', 'unknown'))
        print(f"No trade — {'; '.join(reasons)}")

        # What-if levels in summary
        w = result.get('what_if')
        if w:
            print(f"\n  If you trade anyway ({w['direction']}):")
            print(f"    Entry ${w['entry']:.2f} | SL ${w['sl']:.2f} ({w['sl_pct']:.2f}%) | "
                  f"TP1 ${w['tp1']:.2f} ({w['tp1_pct']:.2f}%) | R:R {w['rr1']:.2f}x")
            if w['invalidation']:
                for inv in w['invalidation']:
                    print(f"    ⚠️  {inv}")
    print("─" * 55)


def print_dual_strategy(result):
    """Print dual strategy (scalp + momentum) results."""
    dual = result.get('dual_strategy')
    if not dual:
        return

    sa = dual.get('strategy_a') or {}
    sb = dual.get('strategy_b') or {}

    print(f"\n  {'═' * 56}")
    print(f"  DUAL STRATEGY")
    print(f"  {'═' * 56}")

    # Strategy A: Range Scalper
    print(f"\n  {'─' * 56}")
    print(f"  📐 STRATEGY A: RANGE SCALPER")
    print(f"  {'─' * 56}")
    _print_dual_sub(sa, 'scalp')

    # Strategy B: Momentum Rider
    print(f"\n  {'─' * 56}")
    print(f"  🚀 STRATEGY B: MOMENTUM RIDER")
    print(f"  {'─' * 56}")
    _print_dual_sub(sb, 'momentum')

    # Combined verdict
    print(f"\n  {'─' * 56}")
    signals = []
    if sa.get('status') == 'SIGNAL':
        signals.append(f"📐 SCALP {sa['direction']} ${sa['entry']:.2f} → TP1 ${sa['tp1']:.2f}")
    if sb.get('status') == 'SIGNAL':
        signals.append(f"🚀 MOMENTUM {sb['direction']} ${sb['entry']:.2f} → TP1 ${sb['tp1']:.2f}")

    if signals:
        print(f"  ✅ DUAL SIGNALS:")
        for s in signals:
            print(f"     {s}")
    else:
        reasons = []
        if sa.get('reason'):
            reasons.append(f"Scalp: {sa['reason']}")
        if sb.get('reason'):
            reasons.append(f"Mom: {sb['reason']}")
        print(f"  ⛔ NO DUAL SIGNALS — {'; '.join(reasons)}")
    print(f"  {'═' * 56}")


def _print_dual_sub(s, mode):
    """Print one dual-strategy sub-result."""
    if not s:
        print(f"  ⚪ No data")
        return

    status = s.get('status', '?')
    if status == 'SIGNAL':
        direction = s.get('direction', '?')
        entry = s.get('entry', 0)
        sl = s.get('sl', 0)
        tp1 = s.get('tp1', 0)
        tp2 = s.get('tp2', 0)
        tp3 = s.get('tp3', 0)
        sl_pct = s.get('sl_pct', 0)
        tp1_pct = s.get('tp1_pct', 0)
        ics = s.get('ics', 0)
        size = s.get('size', 0)
        entry_mode = s.get('mode', '')

        print(f"  ✅ SIGNAL: {direction}")
        if 'pullback' in entry_mode:
            print(f"     📐 Entry Mode: PULLBACK RETEST")
            print(f"     📐 Retrace: {s.get('pullback_retrace', 0)*100:.1f}%  Wait: {s.get('pullback_bars', 0)} bars")
        print(f"     Entry: ${entry:.2f}  (size={size:.1f})")
        print(f"     SL:    ${sl:.2f}  ({sl_pct:.2f}%)")
        print(f"     TP1:   ${tp1:.2f}  ({tp1_pct:.2f}%)", end="")
        if mode == 'momentum':
            tp1_close = s.get('tp1_close', 0.15)
            print(f"  [close {tp1_close*100:.0f}%]")
        else:
            print(f"  [close 30%]")
        print(f"     TP2:   ${tp2:.2f}")
        print(f"     TP3:   ${tp3:.2f}")
        print(f"     ICS:   {ics:.4f}  Regime: {s.get('regime', '?')}")

        if mode == 'momentum':
            strength = s.get('momentum_strength', 0)
            reason = s.get('momentum_reason', '')
            rr = abs(tp1_pct / sl_pct) if sl_pct > 0 else 0
            print(f"     📊 Momentum: {strength:.2f}  ({reason})")
            print(f"     📊 R:R:     {rr:.2f}x (TP1 vs SL)")

    elif status == 'FILTERED':
        print(f"  🔶 FILTERED: {s.get('reason', '?')}")
    else:
        reason = s.get('reason', '?')
        entry_mode = s.get('mode', '')
        if 'pullback_pending' in entry_mode:
            strength = s.get('strength', 0)
            print(f"  ⏳ PENDING: {reason}")
            print(f"     📊 Ignition strength: {strength:.2f}")
        else:
            print(f"  ⛔ NO SIGNAL: {reason}")


def main():
    print("DEBUG: Scanner started")
    parser = argparse.ArgumentParser(description='JIMI Live Scanner')
    parser.add_argument('--json', action='store_true', help='Output JSON only')
    parser.add_argument('--dashboard', type=int, help='Run dashboard on port')
    parser.add_argument('--tf', default='15m', choices=['1m', '5m', '15m', '1h'],
                        help='Base timeframe (default: 15m)')
    parser.add_argument('--cached', action='store_true',
                        help='Skip all macro fetching — use whatever is in cache')
    parser.add_argument('--refresh', action='store_true',
                        help='Force-refresh all macro data (ignore cache TTL)')
    parser.add_argument('--search', type=str,
                        help='Run a web search query before scanning (e.g., "FOMC minutes May 2026 release date")')
    args = parser.parse_args()

    if args.dashboard:
        print(f"Dashboard mode on port {args.dashboard} (not implemented in refactored version)")
        print("Use the legacy scanner for dashboard mode.")
        return

    from datetime import datetime

    # Timeframe scaling: lookback bars are tuned for 15m, scale for other TFs
    tf_multipliers = {'1m': 15, '5m': 3, '15m': 1, '1h': 0.25}
    tf_mult = tf_multipliers[args.tf]
    bars_map = {'1m': 10, '5m': 10, '15m': 10, '1h': 10}
    bars = bars_map[args.tf]

    # Scale config lookbacks for the selected timeframe
    scaled_config = dict(CONFIG)
    lookback_keys = [
        'VWAP_LOOKBACK', 'CVD_LOOKBACK', 'M4_ZL_LOOKBACK',
        'M14_SWEEP_LOOKBACK', 'CROSS_ASSET_LOOKBACK',
    ]
    for k in lookback_keys:
        if k in scaled_config:
            scaled_config[k] = max(int(CONFIG[k] * tf_mult), 10)
    scaled_config['_base_timeframe'] = args.tf

    # Step 1: Ensure historical CSV is fresh (fetch gap if stale)
    csv_path = ensure_csv_fresh()

    # Step 1b: Fetch live FRED claims/unemployment data (no API key needed)
    # Skip if --cached flag is set; check cache age before subprocess call.
    _fred_cache_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'data', 'fred', 'macro_cache.json')
    _fred_cache_fresh = False
    if os.path.exists(_fred_cache_path):
        try:
            _fred_mtime = os.path.getmtime(_fred_cache_path)
            _fred_age_hours = (time.time() - _fred_mtime) / 3600
            _fred_ttl_days = scaled_config.get('MACRO_CACHE_TTL_DAYS', 7)
            _fred_cache_fresh = _fred_age_hours < (_fred_ttl_days * 24)
        except Exception:
            pass

    if args.cached:
        print(f"  📦 FRED: --cached mode, using existing cache")
    elif _fred_cache_fresh and not args.refresh:
        print(f"  📦 FRED: cache fresh ({_fred_age_hours:.1f}h old), skipping fetch")
    else:
        try:
            import subprocess as _sp
            _fred_script = os.path.join(os.path.dirname(__file__), 'fetch_fred_claims.py')
            # Ensure we run the script from the workspace root to maintain proper context
            workspace_root = os.path.dirname(os.path.dirname(__file__))
            _r = _sp.run([sys.executable, _fred_script], capture_output=True, text=True, timeout=30, cwd=workspace_root)
            if _r.returncode == 0:
                for line in _r.stdout.strip().split('\n'):
                    print(f"  {line.strip()}")
            else:
                print(f"  ⚠️  FRED claims fetch failed (return code {_r.returncode}) (using cached/hardcoded data)")
                if _r.stderr:
                    print(f"  🔍 FRED stderr: {_r.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"  ⏰  FRED claims fetch timed out after 30 seconds (using cached/hardcoded data)")
        except Exception as e:
            print(f"  ⚠️  FRED claims fetch failed: {e} (using cached/hardcoded data)")

    # Step 1c: Fetch live macro indicators (Caixin PMI, NBS PMI, etc.)
    # Skip if --cached; force-refresh if --refresh.
    if args.cached:
        print(f"  📦 Macro indicators: --cached mode, using existing cache")
    else:
        _force_refresh = args.refresh
        if _force_refresh:
            print(f"  🔄 Macro indicators: --refresh mode, forcing fresh fetch")
        try:
            from src.utils.macro_fetch import get_latest_macro_indicators
            _macro_indicators = get_latest_macro_indicators(force_refresh=_force_refresh)
            for _name, _data in _macro_indicators.items():
                if _data and _data.get('actual') is not None:
                    _surprise = _data.get('surprise', '?')
                    print(f"  📊 {_name}: actual={_data['actual']} prev={_data.get('previous', '?')} surprise={_surprise}")
        except Exception as e:
            print(f"  ⚠️  Macro indicator fetch failed: {e}")

    # Step 2: Load daily data from CSV for reliable EMA55 bias
    df_1d_hist = load_daily_from_csv(csv_path)
    if df_1d_hist is not None:
        print(f"  📊 Daily bias from CSV ({len(df_1d_hist)} bars)")

    # Step 3: Fetch live base-timeframe data
    print(f"Fetching {args.tf} data ({bars} bars)...")
    df_base = fetch_recent(bars=bars, timeframe=args.tf)
    print("Computing indicators...")
    df_base, df_1h, df_2h, df_4h, df_1d = compute_indicators(
        df_base, config=scaled_config, df_1d_hist=df_1d_hist)

    # Step 3b: Fetch BTC 15m for cross-asset correlation (aligned with engine)
    btc_15m_df, btc_corr_series = None, None
    if scaled_config.get('CROSS_ASSET_ENABLED', False):
        try:
            print("Fetching BTC/USDT 15m for cross-asset...")
            btc_15m_df = fetch_btc_15m(df_base['Open time'].iloc[0], df_base['Open time'].iloc[-1])
            if btc_15m_df is not None and len(btc_15m_df) > 100:
                btc_corr_series = compute_btc_correlation(
                    df_base, btc_15m_df, scaled_config.get('CROSS_ASSET_LOOKBACK', 48))
                print(f"  BTC data: {len(btc_15m_df)} bars, correlation computed")
            else:
                print("  BTC data: insufficient, cross-asset disabled")
                btc_15m_df = None
        except Exception as e:
            print(f"  BTC data: fetch failed ({e}), cross-asset disabled")
            btc_15m_df = None

    print(f"Scanning [{args.tf}]...")
    result = scan_signal(df_base, df_1h, df_2h, df_4h, df_1d, config=scaled_config,
                         btc_15m_df=btc_15m_df, btc_corr_series=btc_corr_series)

    # Tag the result with timeframe
    result['timeframe'] = args.tf

    # ── Dual Strategy Scan ──
    try:
        ds = DualStrategy(config=scaled_config)
        dual_result = ds.scan(df_base, df_1h, df_2h, df_4h, df_1d)
        result['dual_strategy'] = dual_result
    except Exception as e:
        print(f"  ⚠️  Dual strategy error: {e}")
        result['dual_strategy'] = None

    # ── Macro Calendar ──
    try:
        macro_cal = get_macro_calendar()
        result['macro_calendar'] = calendar_to_dict(macro_cal)
    except Exception as e:
        print(f"  ⚠️  Macro calendar error: {e}")
        result['macro_calendar'] = None

    # Add macro indicators to result
    try:
        from src.utils.macro_fetch import get_latest_macro_indicators
        result['macro_indicators'] = get_latest_macro_indicators()
    except Exception:
        result['macro_indicators'] = None

    # Always save scan result to data/scans/
    scan_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'scans')
    os.makedirs(scan_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    scan_file = os.path.join(scan_dir, f'scan_{ts}.json')
    with open(scan_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  💾 Saved: {scan_file}")

    if args.json:
        # Add conflict resolution and phase detection to JSON output
        cr = detect_conflict(result, config=scaled_config)
        result['conflict_resolution'] = conflict_to_dict(cr)
        p3 = detect_phase(result, config=scaled_config, df_15m=df_base)
        result['power_of_3'] = phase_to_dict(p3)
        print(json.dumps(result, indent=2, default=str))
    else:
        print_signal(result)
        print_summary(result)
        print_dual_strategy(result)

        # ── Signal Evaluation (post-generation actionable check) ──
        # Price may have moved since data was fetched — evaluate freshness
        try:
            _live_price = float(df_base['Close'].iloc[-1])
            _eval = evaluate_signal(result, current_price=_live_price, config=scaled_config)
            if _eval:
                print(format_signal_eval(_eval))
                result['signal_eval'] = _eval
        except Exception:
            pass

        # ── Macro Lifecycle ──
        _lc = result.get('macro_lifecycle')
        if _lc and _lc.get('phase') not in ('NO_RELEASE', 'RESOLVED', None):
            try:
                _lc_state = evaluate_macro_lifecycle(df_base, config=scaled_config)
                if _lc_state:
                    print(format_lifecycle(_lc_state))
            except Exception:
                pass

        # ── Power of 3 Phase Detection ──
        p3 = detect_phase(result, config=scaled_config, df_15m=df_base)
        print(format_phase(p3))
        result['power_of_3'] = phase_to_dict(p3)

        # ── Conflict Resolution ──
        cr = detect_conflict(result, config=scaled_config)
        if cr.has_conflict:
            print(format_conflict(cr))
            result['conflict_resolution'] = conflict_to_dict(cr)

            # Re-save with all analysis data
            with open(scan_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            # Offer to spawn forward test
            if cr.forward_test:
                ft = cr.forward_test
                print(f"\n  💡 To start forward test, run:")
                print(f"     python3 scripts/forward_test.py --level {ft.key_level_low:.0f} {ft.key_level_high:.0f} "
                      f"--scenarios {','.join(s.name.lower() for s in ft.scenarios)}")
        else:
            print(f"\n  ✅ No conflict — scanner verdict stands.")

        # Re-save with phase data
        with open(scan_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    # ── Macro Calendar ──
    try:
        if result.get('macro_calendar'):
            macro_cal = get_macro_calendar()
            _current_regime = result.get('m9', {}).get('regime', None)
            print(format_macro_calendar(macro_cal, current_regime=_current_regime))
    except Exception:
        pass

    # ── Collect derivatives snapshot ──
    try:
        from scripts.collect_derivatives import collect, mark_ready
        print()
        collect()
        mark_ready()
    except Exception as e:
        print(f"\n  ⚠️  Derivatives collection failed: {e}")


if __name__ == '__main__':
    main()
