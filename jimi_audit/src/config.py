"""
JIMI Framework — Configuration Loader
Loads settings from YAML, falls back to built-in defaults.
Usage:
    from src.config import CONFIG
    # or
    from src.config import load_config
    cfg = load_config("config/v615.yaml")
"""

import os
import sys
import yaml


# ═══════════════════════════════════════════════════════════════
# CONFIG VALIDATOR
# ═══════════════════════════════════════════════════════════════

# Keys that are set dynamically (scanner, engine) — not errors if absent
_DYNAMIC_KEYS = {'_base_timeframe', 'M22_ENABLED', 'M22_WEIGHT', 'M22_VETO_THRESHOLD',
                  'M22_FAIL_THRESHOLD', 'M22_LS_CROWDED_THRESHOLD',
                  'M22_SIZE_CRITICAL', 'M22_SIZE_HIGH', 'M22_SIZE_MEDIUM',
                  'M22_CPI_HOT_THRESHOLD', 'M22_PPI_YOY', 'M22_PPI_PREV_YOY',
                  'M22_PPI_PREV_PREV_YOY', 'M22_PPI_MOM', 'M22_CPI_YOY',
                  'M22_CPI_PREV_YOY', 'M22_CPI_EXPECTED', 'M22_PPI_EXPECTED',
                  'M22_FED_STANCE', 'M22_FED_FUNDS_RATE',
                  'M22_CLAIMS_CLASSIFICATION', 'M22_CLAIMS_TREND',
                  'M22_UNEMPLOYMENT_RATE', 'M22_SAHM_TRIGGERED',
                  'TAKER_ENABLED', 'TAKER_WEIGHT', 'TAKER_DEVIATION_THRESHOLD',
                  'TAKER_MOMENTUM_BONUS', 'TAKER_12H_AGREE_BONUS',
                  'TAKER_DIVERGENCE_PENALTY', 'MIN_RR_RATIO',
                  'M23_CLAIMS_ENABLED', 'M23_CLAIMS_WEIGHT', 'M23_COMBO_ENABLED',
                  'M24_ENABLED', 'M24_WINDOW_DAYS',
                  'M25_ENABLED', 'M25_WINDOW_DAYS',
                  'M26_ENABLED', 'M26_WINDOW_DAYS',
                  'M27_ENABLED', 'M27_WINDOW_DAYS',
                  'M28_ENABLED', 'M28_WINDOW_DAYS',
                  'M29_ENABLED', 'M29_WINDOW_DAYS',
                  'M30_ENABLED', 'M30_WINDOW_DAYS',
                  'M31_ENABLED', 'M31_WINDOW_DAYS',
                  'M32_ENABLED', 'M32_WINDOW_DAYS',
                  'M33_ENABLED', 'M33_WINDOW_DAYS',
                  'M34_ENABLED', 'M34_WINDOW_DAYS',
                  'M35_ENABLED', 'M35_WINDOW_DAYS',
                  'M36_ENABLED', 'M36_WINDOW_DAYS',
                  'M37_ENABLED', 'M37_WINDOW_DAYS',
                  'M38_ENABLED', 'M38_WINDOW_DAYS',
                  'M39_ENABLED', 'M39_WINDOW_DAYS',
                  'M40_ENABLED', 'M40_WINDOW_DAYS',
                  'M41_ENABLED', 'M41_WINDOW_DAYS',
                  'M42_ENABLED', 'M42_WINDOW_DAYS',
                  'M43_ENABLED', 'M43_WINDOW_DAYS',
                  'M44_ENABLED', 'M44_WINDOW_DAYS',
                  'M45_ENABLED', 'M45_WINDOW_DAYS',
                  'M46_ENABLED', 'M46_WINDOW_DAYS',
                  'M47_ENABLED', 'M47_WINDOW_DAYS',
                  'M48_ENABLED', 'M48_WINDOW_DAYS',
                  'M49_ENABLED', 'M49_WINDOW_DAYS',
                  'M50_ENABLED', 'M50_WINDOW_DAYS',
                  'M51_ENABLED', 'M51_WINDOW_DAYS',
                  'M52_ENABLED', 'M52_WINDOW_DAYS',
                  'M53_ENABLED', 'M53_WINDOW_DAYS',
                  'M54_ENABLED', 'M54_WINDOW_DAYS',
                  'M55_ENABLED', 'M55_WINDOW_DAYS',
                  'M56_ENABLED', 'M56_WINDOW_DAYS',
                  'M57_ENABLED', 'M57_WINDOW_DAYS',
                  'M58_ENABLED', 'M58_WINDOW_DAYS',
                  'M59_ENABLED', 'M59_WINDOW_DAYS',
                  'M60_ENABLED', 'M60_WINDOW_DAYS',
                  'M61_ENABLED', 'M61_WINDOW_DAYS',
                  'M62_ENABLED', 'M62_WINDOW_DAYS',
                  'M65_ENABLED', 'M65_WINDOW_DAYS',
                  # Cascade config keys
                  'CASCADE_ENABLED',
                  'CASCADE_US_INFLATION_ENABLED',
                  'CASCADE_US_LABOR_ENABLED',
                  'CASCADE_US_ACTIVITY_ENABLED',
                  'CASCADE_CHINA_MACRO_ENABLED',
                  'CASCADE_EU_MACRO_ENABLED',
                  'CASCADE_UK_MACRO_ENABLED',
                  'CASCADE_JAPAN_MACRO_ENABLED',
                  'CASCADE_AU_MACRO_ENABLED',
                  'M_CHINA_ACTIVITY_ENABLED',
                  # Macro event filter keys
                  'MACRO_EVENT_FILTER_ENABLED',
                  'PHASE0_CAUTION',
                  'MACRO_CASCADE_SIZE_MULT',
                  'MACRO_LOOKBACK_HOURS',
                  'MACRO_LOOKAHEAD_HOURS',
                  # M66-M73 Tradfi config keys
                  'M66_ENABLED', 'M66_WEIGHT', 'M66_ALERT_THRESH', 'M66_CONFIRMED_THRESH',
                  'M67_ENABLED', 'M67_WEIGHT', 'M67_DXY_THRESH', 'M67_DIVERGENCE_ETH_THRESH',
                  'M68_ENABLED', 'M68_WEIGHT', 'M68_ALERT_BPS', 'M68_EXTREME_BPS',
                  'M69_ENABLED', 'M69_WEIGHT', 'M69_COMPLACENT_THRESH', 'M69_ELEVATED_THRESH',
                  'M69_FEAR_THRESH', 'M69_CRISIS_THRESH', 'M69_SPIKE_DELTA',
                  'M70_ENABLED', 'M70_WEIGHT', 'M70_ALERT_THRESH',
                  'M71_ENABLED', 'M71_WEIGHT', 'M71_ALERT_THRESH',
                  'M72_ENABLED', 'M72_WEIGHT', 'M72_CACHE_BARS', 'M72_HIGH_THRESH', 'M72_LOW_THRESH',
                  'M73_ENABLED', 'M73_WEIGHT', 'M73_LARGE_MINT_THRESH',
                  'M74_ENABLED', 'M74_WEIGHT', 'M74_HIGH_THRESH', 'M74_LOW_THRESH',
                  'M75_ENABLED', 'M75_WEIGHT', 'M75_CVD_WINDOW', 'M75_TAKER_WINDOW', 'M75_OI_WINDOW', 'M75_FUNDING_WINDOW', 'M75_TAKER_PERSIST_THRESH', 'M75_CVD_DIV_THRESH', 'M75_OI_EXPAND_THRESH', 'M75_FUNDING_EXTREME', 'M75_W_TAKER', 'M75_W_CVD', 'M75_W_OI', 'M75_W_FUNDING', 'M75_BULL_THRESHOLD', 'M75_BEAR_THRESHOLD',
                  'M73_MEGA_MINT_THRESH', 'M73_LARGE_BURN_THRESH',
                  'M66_CARRY_ALERT_SCORE', 'M66_CARRY_CONFIRMED_SCORE',
                  'SIGNAL_EVAL_TRADFI_ACTIVE',
                  'MACRO_CACHE_TTL_DAYS'}

# Module → (enabled_key, weight_key, required_when_enabled)
_MODULE_WEIGHT_PAIRS = [
    ('M7_ENABLED',  'M7_WEIGHT',  False),   # direction/sizing only is OK
    ('M8_ENABLED',  'M8_WEIGHT',  True),
    ('M9_ENABLED',  'M9_WEIGHT',  False),   # direction/sizing only is OK
    ('M10_ENABLED', 'M10_WEIGHT', True),
    ('M11_ENABLED', 'M11_WEIGHT', True),
    ('M12_ENABLED', 'M12_WEIGHT', True),
    ('M13_ENABLED', 'M13_WEIGHT', True),
    ('M14_ENABLED', 'M14_WEIGHT', True),
    ('M17_ENABLED', 'M17_WEIGHT', True),
    ('M18_ENABLED', 'M18_WEIGHT', True),
    ('M75_ENABLED',  'M75_WEIGHT',  True),
    ('M74_ENABLED',  'M74_WEIGHT',  False),   # direction/sizing only is OK
]

# Expected types for each key (None = any)
_TYPE_MAP = {
    'ICS_THRESHOLD_NORMAL': (int, float),
    'ICS_THRESHOLD_CAUTION': (int, float),
    'ICS_FLOOR': (int, float),
    'ICS_FLOOR_M4_FALSE': (int, float),
    'ICS_CEILING': (int, float),
    'M1_WEIGHT': (int, float),
    'M2_WEIGHT': (int, float),
    'M3_WEIGHT': (int, float),
    'M4_WEIGHT': (int, float),
    'M5_WEIGHT': (int, float),
    'M6_WEIGHT': (int, float),
    'M7_WEIGHT': (int, float),
    'M8_WEIGHT': (int, float),
    'M9_WEIGHT': (int, float),
    'M10_WEIGHT': (int, float),
    'M11_WEIGHT': (int, float),
    'M12_WEIGHT': (int, float),
    'M13_WEIGHT': (int, float),
    'M14_WEIGHT': (int, float),
    'SWEEP_PROXIMITY_PCT': (int, float),
    'M17_WEIGHT': (int, float),
    'MACD_FAST': int, 'MACD_SLOW': int, 'MACD_SIGNAL': int,
    'EMA_FAST': int, 'EMA_SLOW': int, 'RSI_PERIOD': int, 'ATR_PERIOD': int,
    'VWAP_LOOKBACK': int, 'CVD_LOOKBACK': int, 'M4_ZL_LOOKBACK': int,
    'M5_VP_LOOKBACK': int, 'M5_VP_BINS': int,
    'SUMMER_MONTHS': list, 'SHOULDER_MONTHS': list,
    'M9_BLOCK_REGIMES': list,
    'M18_WEIGHT': (int, float),
    'SQUEEZE_RANGE48_MAX': (int, float),
    'SQUEEZE_COIL_DELTA_MAX': (int, float),
    'SQUEEZE_COIL_BARS_MIN': int,
    'SQUEEZE_COMPRESSION_BARS_MIN': int,
    'SQUEEZE_DRY_BARS_MIN': int,
    'SQUEEZE_DOJI_BARS_MIN': int,
    'SQUEEZE_REQUIRE_HIST_FLIP': bool,
    'SQUEEZE_EMA_FILTER': bool,
    'SQUEEZE_ENTRY_BUFFER_PCT': (int, float),
    'SQUEEZE_ENTRY_EXPIRY_BARS': int,
    'SQUEEZE_BREAKOUT_VOL_MULT': (int, float),
    'SQUEEZE_ENTRY_MODE': str,
    'SQUEEZE_COIL_HOURS_MIN': int,
    'SQUEEZE_CONFIRM_EMA': bool,
    'SQUEEZE_CONFIRM_CVD': bool,
    'SQUEEZE_CONFIRM_RSI': bool,
    'SQUEEZE_CONFIRM_QUALITY': bool,
    'SQUEEZE_CONFIRM_ATR_FLOOR': bool,
    'SQUEEZE_MIN_ATR': (int, float),
    'SQUEEZE_ATR_FLOOR_PCTILE': int,
    'SQUEEZE_ATR_LOOKBACK': int,
    'SQUEEZE_TP_ATR_MULT': (int, float),
    'SQUEEZE_SL_ATR_MULT': (int, float),
    'SQUEEZE_TP_MIN_PCT': (int, float),
    'SQUEEZE_TP_MAX_PCT': (int, float),
    'SQUEEZE_OVERRIDE_REGIME': bool,
    'SQUEEZE_ICS_BOOST': (int, float),
    'SQUEEZE_SIZE_MULT': (int, float),
    'SQUEEZE_COOLDOWN_BARS': int,
    'PAIR': str,
    'WARMUP_BARS_1H': int,
}

# Numeric range checks: key → (min, max)
_RANGE_CHECKS = {
    'ICS_THRESHOLD_NORMAL': (0.0, 1.0),
    'ICS_THRESHOLD_CAUTION': (0.0, 1.0),
    'ICS_FLOOR': (0.0, 1.0),
    'ICS_CEILING': (0.0, 1.0),
    'SL_HARD_MAX_PCT': (0.0, 0.10),
    'SL_ATR_STD': (0.5, 5.0),
    'TP1_ATR': (0.5, 10.0),
    'TP2_ATR': (0.5, 10.0),
    'TP3_ATR': (0.5, 10.0),
    'TP1_CLOSE': (0.0, 1.0),
    'TP2_CLOSE': (0.0, 1.0),
    'MAX_DAILY_LOSS': (0.0, 1.0),
    'MAX_TRADES_DAY': (1, 100),
}


def validate_config(cfg, strict=False):
    """Validate config against known schema. Returns (errors, warnings).

    Args:
        cfg: Config dict to validate
        strict: If True, warnings become errors

    Returns:
        (errors: list[str], warnings: list[str])
    """
    errors = []
    warnings = []

    # 1. Unknown keys
    known = set(_DEFAULTS.keys()) | _DYNAMIC_KEYS
    for key in cfg:
        if key not in known:
            warnings.append(f"Unknown config key: {key} (typo or deprecated?)")

    # 2. Type checks
    for key, expected_type in _TYPE_MAP.items():
        if key in cfg and cfg[key] is not None:
            if not isinstance(cfg[key], expected_type):
                errors.append(
                    f"Type mismatch: {key} expected {expected_type}, "
                    f"got {type(cfg[key]).__name__} = {cfg[key]!r}"
                )

    # 3. Range checks
    for key, (lo, hi) in _RANGE_CHECKS.items():
        if key in cfg and isinstance(cfg[key], (int, float)):
            if not (lo <= cfg[key] <= hi):
                warnings.append(
                    f"Out of range: {key}={cfg[key]} (expected {lo}–{hi})"
                )

    # 4. Module weight consistency
    base_weight_keys = ['M1_WEIGHT', 'M2_WEIGHT', 'M3_WEIGHT', 'M4_WEIGHT', 'M5_WEIGHT']
    base_total = sum(cfg.get(k, 0) for k in base_weight_keys)

    for enabled_key, weight_key, required in _MODULE_WEIGHT_PAIRS:
        enabled = cfg.get(enabled_key, False)
        weight = cfg.get(weight_key, 0)
        if enabled and weight == 0 and required:
            warnings.append(
                f"Module {enabled_key}=True but {weight_key}=0 — "
                f"module runs but contributes nothing to ICS"
            )
        if not enabled and weight > 0:
            warnings.append(
                f"Module {enabled_key}=False but {weight_key}={weight} — "
                f"weight is wasted"
            )

    # 5. ICS weight sanity
    enabled_extra_weights = []
    for enabled_key, weight_key, _ in _MODULE_WEIGHT_PAIRS:
        if cfg.get(enabled_key, False) and cfg.get(weight_key, 0) > 0:
            enabled_extra_weights.append((weight_key, cfg[weight_key]))
    if enabled_extra_weights:
        extra_total = sum(w for _, w in enabled_extra_weights)
        total = base_total + extra_total
        if abs(total - 1.0) > 0.05:
            warnings.append(
                f"ICS weights sum to {total:.3f} "
                f"(base={base_total:.3f} + extras={extra_total:.3f}) — "
                f"should be ~1.0 for proper scoring"
            )

    # 6. Structural checks
    if cfg.get('ICS_THRESHOLD_CAUTION', 0) < cfg.get('ICS_THRESHOLD_NORMAL', 0):
        errors.append(
            f"ICS_THRESHOLD_CAUTION ({cfg['ICS_THRESHOLD_CAUTION']}) < "
            f"ICS_THRESHOLD_NORMAL ({cfg['ICS_THRESHOLD_NORMAL']}) — caution should be >= normal"
        )

    if cfg.get('EMA_FAST', 0) >= cfg.get('EMA_SLOW', 0):
        errors.append(
            f"EMA_FAST ({cfg.get('EMA_FAST')}) >= EMA_SLOW ({cfg.get('EMA_SLOW')}) — "
            f"fast should be < slow"
        )

    if cfg.get('MACD_FAST', 0) >= cfg.get('MACD_SLOW', 0):
        errors.append(
            f"MACD_FAST ({cfg.get('MACD_FAST')}) >= MACD_SLOW ({cfg.get('MACD_SLOW')}) — "
            f"fast should be < slow"
        )

    tp1 = cfg.get('TP1_ATR', 0)
    tp2 = cfg.get('TP2_ATR', 0)
    tp3 = cfg.get('TP3_ATR', 0)
    if not (tp1 < tp2 < tp3):
        warnings.append(
            f"TP ladder not ascending: TP1={tp1} TP2={tp2} TP3={tp3}"
        )

    # 7. Veto consistency
    if cfg.get('DIR_VETO_ENABLED', False) and not cfg.get('VETO_ENABLED', False):
        warnings.append("DIR_VETO_ENABLED=true but VETO_ENABLED=false — veto won't run")

    return errors, warnings


def print_validation_report(errors, warnings):
    """Print config validation results."""
    if not errors and not warnings:
        print("  ✅ Config validation: all checks passed")
        return

    if warnings:
        print(f"\n  ⚠️  Config warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    • {w}")

    if errors:
        print(f"\n  ❌ Config errors ({len(errors)}):")
        for e in errors:
            print(f"    • {e}")


# ═══════════════════════════════════════════════════════════════
# CONFIG LOADER
# ═══════════════════════════════════════════════════════════════

_DEFAULTS = {
    # Gate thresholds
    "ICS_THRESHOLD_NORMAL": 0.50,
    "ICS_THRESHOLD_CAUTION": 0.52,
    "ICS_FLOOR": 0.50,
    "ICS_FLOOR_M4_FALSE": 0.50,
    "ICS_CEILING": 0.70,
    # Module weights
    "M1_WEIGHT": 0.08,
    "M2_WEIGHT": 0.00,
    "M3_WEIGHT": 0.22,
    "M4_WEIGHT": 0.20,
    "M5_WEIGHT": 0.15,
    "M6_WEIGHT": 0.10,
    "M7_WEIGHT": 0.00,
    "M8_WEIGHT": 0.10,
    "M9_WEIGHT": 0.00,
    "M10_WEIGHT": 0.25,
    "M11_WEIGHT": 0.12,
    "M12_WEIGHT": 0.05,
    # v6.12
    "BIAS_GATE_ENABLED": True,
    "BIAS_GATE_LONG_ICS": 0.65,
    "TREND_FILTER_ENABLED": True,
    "TREND_BLOCK_COUNTER_TREND": False,
    "TREND_STRONG_ONLY": False,
    "TREND_MIN_SCORE": 0.15,
    "MONTHLY_DD_CIRCUIT": 0.05,
    # v6.13 Seasonal
    "SUMMER_MONTHS": [6, 7, 8, 9],
    "SUMMER_SIZE_MULT": 0.60,
    "SUMMER_ICS_BOOST": 0.03,
    "TP1_CLOSE_BASE": 0.30,
    "TP1_CLOSE_SUMMER": 0.45,
    "TP1_ATR_SUMMER": 0.7,
    "MAX_CONSEC_LOSS_SUMMER": 2,
    "CONSEC_LOSS_PAUSE_SUMMER": 12,
    "PHASE0_SUMMER_BLOCK": 0.60,
    "SHOULDER_MONTHS": [3, 10],
    "SHOULDER_SIZE_MULT": 0.50,
    "SHOULDER_SL_ATR": 1.4,
    "SHOULDER_SL_HARD_MAX": 0.016,
    "SHOULDER_TP1_ATR": 0.8,
    "SHOULDER_TP1_CLOSE": 0.45,
    "SHOULDER_ICS_BOOST": 0.02,
    "SHOULDER_MAX_TRADES_DAY": 3,
    # Module gates
    "M2_VETO_ENABLED": True,
    "M2_VETO_THRESHOLD": 0.40,
    "M5_FAIL_ICS_BOOST": 0.06,
    "M5_FAIL_HARD_THRESHOLD": 0.25,
    "M7_HARD_GATE": False,
    "M7_GATE_THRESHOLD": 0.30,
    "M7_GATE_STRONG_THRESHOLD": 0.60,
    "M7_GATE_STRONG_BOOST": 0.04,
    "SESSION_ASIAN_BLOCK": False,
    "POST_CRASH_COOLDOWN": True,
    "POST_CRASH_THRESHOLD": 0.10,
    "POST_CRASH_BARS": 48,
    "CASCADE_MULTIPLIER": 1.12,
    "CASCADE_PENALTY": 0.85,
    "DIR_VETO_ENABLED": True,
    # Indicators
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "EMA_FAST": 21,
    "EMA_SLOW": 55,
    "RSI_PERIOD": 14,
    "ATR_PERIOD": 14,
    # M3 VWAP
    "VWAP_LOOKBACK": 96,
    "VWAP_ZONE_PCT": 0.012,
    "VOL_THRESHOLD": 0.25,
    "TAKER_LONG": 0.52,
    "TAKER_SHORT": 0.48,
    "TAKER_FILLNA": 0.50,
    "SIGNAL_EXPIRY": 3,
    # M4 CVD
    "CVD_LOOKBACK": 36,
    "CVD_DIVERGENCE_WINDOW": 12,
    "M4_ZL_LOOKBACK": 18,
    "M4_ZL_MOMENTUM_BARS": 8,
    "M4_DIV_WEIGHT": 0.40,
    "M4_ZL_WEIGHT": 0.60,
    # M5 Liquidation
    "M5_VP_LOOKBACK": 672,
    "M5_VP_BINS": 50,
    "M5_MIN_SCORE": 0.25,
    # Stop loss
    "SL_ATR_STD": 1.3,
    "SL_ATR_STD_SUMMER": 1.6,
    "SL_ATR_TRANSITION": 1.0,
    "TP1_ATR_TRANSITION": 1.2,
    "TRANSITION_SIZE_MULT": 0.50,
    "TRANSITION_SCORE_RANGE": 0.20,
    "SL_HARD_MAX_PCT": 0.018,
    "SL_HARD_MAX_SUMMER": 0.018,
    "SL_BREAKEVEN_AFTER_TP1": True,
    "EARLY_EXIT_BARS": 16,
    "EARLY_EXIT_MIN_LOSS": 0.003,
    "EARLY_EXIT_BARS_SUMMER": 12,
    "EARLY_EXIT_MIN_LOSS_SUMMER": 0.002,
    # Take profit
    "TP1_ATR": 1.5,
    "TP2_ATR": 2.0,
    "TP3_ATR": 3.5,
    "TP1_CLOSE": 0.30,
    "TP2_CLOSE": 0.30,
    # Position sizing
    "SIZE_STD": 5.0,
    "SIZE_LONG": 5.0,
    "SIZE_M2_NEUTRAL": 2.5,
    "SIZE_CAUTION": 3.5,
    # Entry filters
    "BAR_MOVE_ATR": 0.5,
    "ATR_FILTER_MAX": 0.035,
    "MIN_ENTRY_DIST_PCT": 0.002,
    # Risk management
    "MAX_TRADES_DAY": 5,
    "MAX_TRADES_DAY_SUMMER": 3,
    "MAX_DAILY_LOSS": 0.05,
    "MAX_DAILY_LOSS_SUMMER": 0.03,
    "COOLDOWN_BARS": 2,
    "COOLDOWN_BARS_SUMMER": 4,
    "SHOULDER_COOLDOWN_BARS": 3,
    "ROLLING_WR_WINDOW": 20,
    "ROLLING_WR_MIN": 0.15,
    "MAX_CONSEC_LOSS": 3,
    "CONSEC_LOSS_PAUSE_BARS": 8,
    "LONG_MIN_ICS": 0.55,
    # M6 Derivatives
    "M3_WEIGHT_DERIV": 0.25,
    "M4_WEIGHT_DERIV": 0.20,
    "DERIV_ENABLED": True,
    # M7
    "M7_ENABLED": True,
    "M7_SIZE_REDUCTION": 0.70,
    "M7_SIZE_MILD": 0.85,
    # M8
    "M8_ENABLED": True,
    "M8_HIGH_FUNDING": 0.05,
    "M8_LOW_FUNDING": -0.05,
    "M8_FLIP_BONUS": 0.15,
    # M9
    "M9_ENABLED": True,
    "M9_BLOCK_REGIMES": ["CRISIS"],
    "M9_SIZE_CHOP": 0.50,
    "M9_SIZE_COMPRESSING": 0.80,
    # M9 regime detection thresholds (v2)
    "M9_WHIPSAW_THRESHOLD": 0.45,
    "M9_RETRACE_THRESHOLD": 0.50,
    "M9_TRENDING_MIN_DIR": 0.45,
    "M9_TRENDING_MIN_VOLUME": 0.50,
    "M9_TRENDING_MAX_WHIPSAW": 0.35,
    "M9_TRENDING_MAX_RETRACE": 0.45,
    "M9_TRENDING_MIN_COHERENCE": 0.50,
    # M9 regime classification thresholds
    "M9_CRISIS_ATR_THRESHOLD": 0.85,
    "M9_CRISIS_BB_THRESHOLD": 0.90,
    "M9_CHOP_HARD_CHOP_SCORE": 0.72,
    "M9_CHOP_HARD_WHIPSAW": 0.70,
    "M9_CHOP_HARD_RETRACE": 0.80,
    "M9_CHOP_MILD_CHOP_SCORE": 0.50,
    "M9_CHOP_MILD_TREND_CEILING": 0.35,
    "M9_TRENDING_TREND_SCORE": 0.40,
    "M9_TRENDING_MAX_WHIPSAW_SCORE": 0.65,
    "M9_TRENDING_MAX_RETRACE_SCORE": 0.80,
    "M9_COMPRESSING_BB": 0.30,
    "M9_COMPRESSING_ATR": 0.40,
    "M9_COMPRESSING_RANGE": 0.60,
    "M9_HYSTERESIS_CONFIRM_BARS": 2,
    "M9_TRENDING_CONFIRM_BARS": 3,
    "M9_COMPRESSING_CONFIRM_BARS": 3,
    "M9_CRISIS_COOLDOWN": 8,
    "M9_TRENDING_COOLDOWN": 4,
    "M9_COMPRESSING_COOLDOWN": 6,
    "M9_CHOP_HARD_COOLDOWN": 6,
    "M9_CHOP_MILD_COOLDOWN": 4,
    "M9_SIZE_CHOP_MILD": 0.55,
    "M9_CHOP_SPLIT_ENABLED": True,
    "M9_CHOP_MAX_BARS": 96,
    "M9_CHOP_EXIT_COOLDOWN": 12,
    # M10
    "M10_ENABLED": True,
    "M10_FETCH_ON_BACKTEST": True,
    # M11
    "M11_ENABLED": True,
    "M11_RSI_PERIOD_15M": 14,
    "M11_RSI_PERIOD_1H": 14,
    "M11_REQUIRE_AGREEMENT": True,
    # M12
    "M12_ENABLED": True,
    "M12_LIVE_ONLY": True,
    # M13 Structure (HTF swing direction)
    "M13_ENABLED": True,
    "M13_WEIGHT": 0.10,
    "M13_DEFER_IN_CHOP": True,
    # Coherence check
    "COHERENCE_CHECK_ENABLED": True,
    "COHERENCE_MAX_CONFLICTS": 3,
    "COHERENCE_M4_PENALTY": 0.04,
    "COHERENCE_M5_PENALTY": 0.03,
    "COHERENCE_M13_PENALTY": 0.03,
    "COHERENCE_M7_PENALTY": 0.02,
    "COHERENCE_MAX_PENALTY": 0.10,
    # Liquidity-aware TP
    "LIQUIDITY_TP_ENABLED": True,
    "LIQUIDITY_FRICTION_THRESHOLD": 0.60,
    "LIQUIDITY_VOID_THRESHOLD": 0.20,
    "LIQUIDITY_TP1_ADJUST_UP": 0.08,
    "LIQUIDITY_TP1_ADJUST_DOWN": 0.05,
    "STOP_RISK_THRESHOLD": 0.55,
    "STOP_RISK_TIGHTEN": True,
    "STOP_RISK_TIGHTEN_FACTOR": 0.85,
    # Adaptive direction
    "ADAPTIVE_DIR_ENABLED": True,
    "ADAPTIVE_DIR_MIN_BIAS": 0.10,
    "ADAPTIVE_DIR_BLOCK_THRESHOLD": 0.60,
    # M14 Sweep-Retest-Reclaim
    "M14_ENABLED": True,
    "M14_WEIGHT": 0.08,
    "M23_ENABLED": True,
    "M14_SWEEP_LOOKBACK": 20,
    "M14_SWEEP_DEPTH_MIN": 0.001,
    "M14_SWEEP_DEPTH_MAX": 0.020,
    "M14_RECLAIM_BARS": 3,
    "M14_RECLAIM_WICK_RATIO": 0.40,
    "M14_VOL_CONFIRM_MULT": 1.2,
    "M14_STRONG_SCORE": 0.85,
    "M14_WEAK_SCORE": 0.55,
    "M14_SLICE_PENALTY": 0.30,
    "M14_NO_SWEEP_SCORE": 0.50,
    "WICK_SLICE_BLOCK": False,
    # M75 Toxic Order Flow
    "M75_ENABLED": False,
    "M75_WEIGHT": 0.10,
    "M75_CVD_WINDOW": 20,
    "M75_TAKER_WINDOW": 8,
    "M75_OI_WINDOW": 10,
    "M75_FUNDING_WINDOW": 3,
    "M75_TAKER_PERSIST_THRESH": 0.62,
    "M75_CVD_DIV_THRESH": 0.015,
    "M75_OI_EXPAND_THRESH": 0.005,
    "M75_FUNDING_EXTREME": 0.0003,
    "M75_W_TAKER": 0.30,
    "M75_W_CVD": 0.35,
    "M75_W_OI": 0.20,
    "M75_W_FUNDING": 0.15,
    "M75_BULL_THRESHOLD": 0.35,
    "M75_BEAR_THRESHOLD": -0.35,
    # M17 Resistance Quality
    "M17_ENABLED": True,
    "M17_WEIGHT": 0.05,
    "M17_ZONE_LOOKBACK": 960,
    "M17_ZONE_WIDTH_PCT": 0.5,
    "M17_ZONE_THIN_THRESHOLD": 0.5,
    "M17_ZONE_THICK_THRESHOLD": 1.5,
    "M17_REJECT_MIN_TOUCHES": 3,
    "M17_REJECT_VOL_STRONG": 2.5,
    "M17_REJECT_VOL_WEAK": 1.2,
    "M17_REJECT_WICK_STRONG": 5.0,
    "M17_DEFENDER_TRAPPED_Z": -2.0,
    "M17_DEFENDER_TRAPPED_Z_LONG": 2.0,
    "M17_DEFENDER_LEAVING_OI_ROC": -0.5,
    "M17_DEFENDER_ADDING_OI_ROC": 1.0,
    "M17_BREAKOUT_VOL_MIN": 1.2,
    "M17_BREAKOUT_MOMENTUM_MIN": 0.3,
    "M17_W_ZONE": 0.25,
    "M17_W_REJECT": 0.30,
    "M17_W_DEFENDER": 0.25,
    "M17_W_READINESS": 0.20,
    # Data freshness
    "DATA_FRESHNESS_ENABLED": True,
    "DATA_FRESHNESS_MAX_AGE_MIN": 20,
    "DATA_FRESHNESS_CHECK_INTERVAL": 5,
    # Veto system
    "VETO_ENABLED": True,
    "VETO_CRISIS_HARD": True,
    "VETO_CHOP_HARD": False,
    "VETO_STALE_DATA_HARD": True,
    "VETO_MONTHLY_DD_HARD": True,
    "VETO_DIR_CONFLICT_HARD": True,
    # Adaptive weights
    "ADAPTIVE_WEIGHTS_ENABLED": True,
    "ADAPTIVE_DECAY": 0.97,
    "ADAPTIVE_MIN_MULT": 0.8,
    "ADAPTIVE_MAX_MULT": 1.2,
    "ADAPTIVE_WARMUP_TRADES": 15,
    # Adaptive TP
    "ADAPTIVE_TP_ENABLED": True,
    "ADAPTIVE_RR_MILESTONE_1": 1.0,
    "ADAPTIVE_RR_MILESTONE_2": 2.0,
    "ADAPTIVE_RR_CLOSE_1": 0.40,
    "ADAPTIVE_RR_CLOSE_2": 0.30,
    "ADAPTIVE_VOL_ENABLED": True,
    "ADAPTIVE_VOL_LOOKBACK": 16,
    "ADAPTIVE_VOL_EXPAND_MULT": 1.3,
    "ADAPTIVE_VOL_COMPRESS_MULT": 0.7,
    "ADAPTIVE_VOL_TIGHTEN_FRAC": 0.5,
    "ADAPTIVE_MOMENTUM_ENABLED": True,
    "ADAPTIVE_MOM_BARS": 12,
    "ADAPTIVE_MOM_MIN_MOVE": 0.001,
    "ADAPTIVE_MOM_EXIT_AFTER": 24,
    "ADAPTIVE_OPPOSING_ENABLED": True,
    "ADAPTIVE_OPPOSING_MIN_FLIP": 0.3,
    # Session awareness
    "SESSION_AWARENESS_ENABLED": True,
    "SESSION_ASIAN_MULT": 0.85,
    "SESSION_EU_MULT": 1.0,
    "SESSION_US_MULT": 1.05,
    "SESSION_LATE_US_MULT": 0.90,
    "SESSION_US_OPEN_BOOST": 1.10,
    # Multi-TF confirmation
    "MTF_CONFIRM_ENABLED": True,
    "MTF_1H_CANDLE_CHECK": True,
    "MTF_4H_EMA_CHECK": False,
    # M1 enhancement
    "M1_RSI_ENABLED": True,
    "M1_MOMENTUM_ENABLED": True,
    "M1_RSI_PERIOD": 14,
    "M1_RSI_OVERBOUGHT": 70,
    "M1_RSI_OVERSOLD": 30,
    "M1_MOMENTUM_LOOKBACK": 6,
    # Cross-asset
    "CROSS_ASSET_ENABLED": True,
    "CROSS_ASSET_BTC_WEIGHT": 0.08,
    "CROSS_ASSET_LOOKBACK": 48,
    # Data
    "PAIR": "ETHUSDT",
    "WARMUP_BARS_1H": 168,
    # Pre-check thresholds
    "ICS_PRECHECK_FLOOR": 0.42,
    "ICS_PRECHECK_THRESHOLD": 0.45,
    # Trend filter (regime-aware)
    "TREND_MIN_SCORE_NEUTRAL": 0.03,
    # M4 sigmoid gating
    "M4_SIGMOID_CENTER": 0.65,
    "M4_SIGMOID_STEEPNESS": 12,
    "M4_ATR_SCALING_ENABLED": True,
    # M5 sweet-spot boost
    "M5_SWEET_SPOT_LOW": 0.30,
    "M5_SWEET_SPOT_HIGH": 0.50,
    "M5_SWEET_SPOT_BOOST": 0.04,
    # Chop regime TP/SL overrides
    "CHOP_TP1_ATR": 0.6,
    "CHOP_TP1_CLOSE": 0.90,
    "CHOP_SL_ATR": 0.5,
    "CHOP_SL_HARD_MAX": 0.008,
    # Liquidity-aware SL/TP
    "LIQUIDITY_LEVELS_ENABLED": True,
    "SL_VOID_BUFFER_PCT": 0.003,
    "SL_VOID_MIN_DIST_PCT": 0.002,
    "SL_VOID_MAX_DIST_PCT": 0.025,
    "TP1_USE_MAGNET": True,
    "TP1_MAGNET_MIN_DIST_PCT": 0.002,
    # M14 entry gate
    "M14_ENTRY_GATE": False,
    "SWEEP_PROXIMITY_PCT": 0.001,   # 0.1% near-miss tolerance for swept detection
    # Forensic recommendations (P0-P2)
    "PHASE0_MAX_BLOCK": 0.30,
    "M5_REGIME_GATE_ENABLED": True,
    # M18 Squeeze Detector v5
    "M18_ENABLED": True,
    "M18_WEIGHT": 0.08,
    "SQUEEZE_RANGE48_MAX": 1.5,
    "SQUEEZE_COMPRESSION_BARS_MIN": 6,
    "SQUEEZE_DRY_BARS_MIN": 2,
    "SQUEEZE_DOJI_BARS_MIN": 3,
    "SQUEEZE_COIL_DELTA_MAX": 0.05,
    "SQUEEZE_COIL_BARS_MIN": 6,
    "SQUEEZE_REQUIRE_HIST_FLIP": True,
    "SQUEEZE_EMA_FILTER": True,
    "SQUEEZE_ENTRY_BUFFER_PCT": 0.001,
    "SQUEEZE_ENTRY_EXPIRY_BARS": 32,
    "SQUEEZE_BREAKOUT_VOL_MULT": 1.0,
    "SQUEEZE_ENTRY_MODE": "TWO_BAR",
    "SQUEEZE_COIL_HOURS_MIN": 12,
    "SQUEEZE_CONFIRM_EMA": True,
    "SQUEEZE_CONFIRM_CVD": True,
    "SQUEEZE_CONFIRM_RSI": True,
    "SQUEEZE_CONFIRM_QUALITY": True,
    "SQUEEZE_CONFIRM_ATR_FLOOR": True,
    "SQUEEZE_MIN_ATR": 5.0,
    "SQUEEZE_ATR_FLOOR_PCTILE": 15,
    "SQUEEZE_ATR_LOOKBACK": 8640,
    "SQUEEZE_TP_ATR_MULT": 2.5,
    "SQUEEZE_SL_ATR_MULT": 1.0,
    "SQUEEZE_TP_MIN_PCT": 0.3,
    "SQUEEZE_TP_MAX_PCT": 2.0,
    "SQUEEZE_OVERRIDE_REGIME": True,
    "SQUEEZE_ICS_BOOST": 0.10,
    "SQUEEZE_SIZE_MULT": 0.80,
    "SQUEEZE_COOLDOWN_BARS": 32,
    # M4b Intrabar CVD
    "M4B_INTRABAR_ENABLED": True,
    "M4B_INTRABAR_HOURS": 48,
    # M66-M73 Tradfi modules (total weight budget: 0.20)
    "M66_ENABLED": True,
    "M66_WEIGHT": 0.02,  # USD/JPY carry trade proxy
    "M67_ENABLED": True,
    "M67_WEIGHT": 0.02,  # DXY divergence filter
    "M68_ENABLED": True,
    "M68_WEIGHT": 0.03,  # 10Y yield + TIPS
    "M69_ENABLED": True,
    "M69_WEIGHT": 0.03,  # VIX regime classifier
    "M70_ENABLED": True,
    "M70_WEIGHT": 0.02,  # WTI crude oil
    "M71_ENABLED": True,
    "M71_WEIGHT": 0.02,  # Gold + DXY geopolitical
    "M72_ENABLED": True,
    "M72_WEIGHT": 0.04,  # BTC dominance (reduced from 0.10)
    "M72_CACHE_BARS": 96,  # re-fetch BTC.D every 24h (96 x 15m)
    "M73_ENABLED": True,
    "M73_WEIGHT": 0.02,  # Stablecoin mint flows
    "M73_LARGE_MINT_THRESH": 500_000_000,
    "M73_MEGA_MINT_THRESH": 1_000_000_000,
    "M73_LARGE_BURN_THRESH": 500_000_000,
    # M9 Neutral sub-classification
    "M9_NEUTRAL_DIR_THRESHOLD": 0.40,
    "M9_NEUTRAL_VOL_THRESHOLD": 1.05,
    # Survival filter
    "SURVIVAL_FILTER_ENABLED": True,
    "SURVIVAL_LOOKBACK": 12,
    "SURVIVAL_MIN_MOVE": 0.002,
    "SURVIVAL_MIN_CONSISTENCY": 0.35,
    "SURVIVAL_VOL_RATIO": 0.30,
    # Time stop (adaptive)
    "TIME_STOP_BARS": 20,
    "TIME_STOP_BASE_BARS": 20,
    "TIME_STOP_PER_PCT": 25,
    "TIME_STOP_MAX_BARS": 60,
    # M1 MACD divergence
    "M1_MACD_DIV_ENABLED": True,
    "M1_MACD_DIV_LOOKBACK": 40,
    "M1_W_RSI_DIV": 0.25,
    "M1_W_MACD_DIV": 0.20,
    "M1_W_CROSSOVER": 0.35,
    "M1_W_MOMENTUM": 0.20,
    "M1_DIV_AGREE_BOOST": 0.03,
    # Power of 3
    "P3_MIN_KEY_LEVEL_DIST_PCT": 0.005,
    "P3_SWEEP_LOOKBACK": 96,
    # M20 Failed Breakout
    "M20_ENABLED": True,
    "M20_WEIGHT": 0.1,
    "M20_BREAKOUT_LOOKBACK": 48,
    "M20_BREAKOUT_LEVEL_ATR_MULT": 0.5,
    "M20_BREAKOUT_MIN_RANGE_PCT": 0.3,
    "M20_WICK_REJECTION_RATIO": 0.4,
    "M20_TAKER_SELL_THRESHOLD": 0.42,
    "M20_TAKER_BUY_THRESHOLD": 0.58,
    "M20_VOL_FADE_THRESHOLD": 0.7,
    "M20_FAILURE_BARS": 8,
    "M20_FAILURE_RETURN_PCT": 0.3,
    "M20_HOLD_BARS": 4,
    "M20_REVERSAL_VOL_MULT": 1.3,
    "M20_REVERSAL_TAKER_FLIP": True,
    "M20_REVERSAL_BODY_MIN": 0.5,
    "M20_STRONG_SCORE": 0.85,
    "M20_MODERATE_SCORE": 0.65,
    "M20_WEAK_BREAKOUT_SCORE": 0.35,
    "M20_HOLDING_SCORE": 0.5,
    "M20_NO_BREAKOUT_SCORE": 0.5,
    "M20_DIRECTION_OVERRIDE_THRESHOLD": 0.65,
    "M20_DIRECT_SIGNAL_THRESHOLD": 0.85,
    # M21 Wyckoff
    "M21_ENABLED": True,
    "M21_WEIGHT": 0.0,
    "M21_RANGE_LOOKBACK": 48,
    "M21_SPRING_LOOKBACK": 12,
    "M21_DISTRIBUTION_LONG_BLOCK": True,
    "M21_ACCUMULATION_SHORT_BLOCK": True,
    # Kill Zone
    "KILL_ZONE_FILTER_ENABLED": True,
    "KILL_ZONE_LONDON_OPEN_START": 7,
    "KILL_ZONE_LONDON_OPEN_END": 10,
    "KILL_ZONE_NY_OPEN_START": 12.5,
    "KILL_ZONE_NY_OPEN_END": 15.5,
    "KILL_ZONE_LONDON_CLOSE_START": 15,
    "KILL_ZONE_LONDON_CLOSE_END": 17,
    "KILL_ZONE_ASIAN_MULT": 0.6,
    "KILL_ZONE_OFF_MULT": 0.7,
    # Range TP/SL overrides
    "RANGE_TP_ENABLED": True,
    "RANGE_SL_ENABLED": True,
    "RANGE_SL_BUFFER_PCT": 0.05,
    "RANGE_TP3_EXTENSION": 0.25,
    "RANGE_SL_HARD_MAX_PCT": 0.025,
    # Structure TP
    "STRUCTURE_TP_ENABLED": True,
    "STRUCTURE_TP_MIN_RANGE_PCT": 1.5,
    "STRUCTURE_TP_MAX_RANGE_PCT": 6.0,
    # Squeeze v6 additions
    "SQUEEZE_SWING_PERIOD": 3,
    "SQUEEZE_SLOPE_THRESHOLD": 0.0008,
    "SQUEEZE_CONVERGENCE_MIN": 0.3,
    "SQUEEZE_MATURITY_WINDOW": 8,
    "SQUEEZE_MATURITY_THRESHOLD": 1.15,
    "SQUEEZE_BREAKOUT_CLOSE_PCT": 0.75,
    "SQUEEZE_BREAKOUT_BODY_MIN": 0.5,
    "SQUEEZE_BREAKOUT_WICK_MAX": 0.3,
    "SQUEEZE_VOL_EXPANSION_MIN": 1.3,
    "SQUEEZE_TAKER_ALIGN": True,
    "SQUEEZE_TAKER_LONG": 0.54,
    "SQUEEZE_TAKER_SHORT": 0.46,
    "SQUEEZE_RETEST_ENABLED": True,
    "SQUEEZE_RETEST_BARS": 8,
    "SQUEEZE_RETEST_TOLERANCE": 0.002,
    "SQUEEZE_RETEST_HOLD_BARS": 2,
}


def load_config(path=None, validate=True):
    """Load config from YAML file, merged with defaults.

    Args:
        path: Path to YAML config file
        validate: Run validation on load (default: True)

    Returns:
        Merged config dict
    """
    cfg = dict(_DEFAULTS)
    if path and os.path.exists(path):
        with open(path) as f:
            user = yaml.safe_load(f) or {}
            cfg.update(user)

    if validate:
        errors, warnings = validate_config(cfg)
        print_validation_report(errors, warnings)
        if errors:
            print(f"\n  💀 Config has {len(errors)} error(s) — aborting.")
            sys.exit(1)

    return cfg


# Singleton — importable as `from src.config import CONFIG`
CONFIG = load_config(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.yaml"),
    validate=False # TEMPORARY: Bypass validation to find hang
)
