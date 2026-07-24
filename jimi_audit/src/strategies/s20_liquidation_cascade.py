"""S20: Liquidation Cascade v6 — LONG-only + whale_watch confirmation.

v5 → v6 CHANGES:
1. LONG-ONLY: Removed SHORT cascade detection entirely
2. WHALE_WATCH CONFIRMATION: Requires whale_watch to fire LONG at same timestamp
3. Rationale: v5 backtest showed SHORT signals lose money (44.2% WR against whale LONG)
   Same-direction (LONG+LONG) = 59.5% WR on 52 trades
4. Kept: regime-adaptive thresholds, vol regime, momentum confirmation, cooldown

v5 stats: 1109 signals, 23.1% WR, PF 0.06 (catastrophic)
v6 target: ~52 signals, 59.5% WR (based on revalidation with whale_watch filter)
"""
from .base import BaseStrategy, SignalResult
import json, os, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cooldown tracking
_last_cascade_ts = 0
CASCADE_COOLDOWN = 1800  # 30 min


def _check_whale_watch_confirms(data):
    """
    Check if whale_watch strategy fired LONG at the same timestamp.
    Returns True if whale_watch confirms LONG direction.
    """
    # Check in strategy_signals from the current scan
    strategy_signals = data.get('strategy_signals', {})
    whale = strategy_signals.get('whale_watch', {})
    if whale.get('direction') == 'LONG' and whale.get('fired', False):
        return True

    # Also check in multi_strategy output
    multi = data.get('multi_strategy', {})
    all_signals = multi.get('all_signals', [])
    for sig in all_signals:
        if sig.get('strategy') == 'whale_watch' and sig.get('direction') == 'LONG':
            return True

    return False


def _check_long_cascade(data, df_15m, idx, regime):
    """
    Detect LONG cascade (crowded shorts squeezed).
    - OI surging (new longs + shorts covering)
    - LS < 0.7 (short-crowded)
    - Price rising (confirming)
    """
    deriv = data.get('derivatives', {})
    oi_roc = deriv.get('oi_roc_1h', 0)
    ls_ratio = deriv.get('ls_ratio', 1.0)

    closes = df_15m['Close'].values.astype(float)
    if idx < 5:
        return None
    price_change = (closes[idx] - closes[idx-5]) / closes[idx-5]

    # ── REGIME-ADAPTIVE THRESHOLDS ──
    if regime in ("STRESS", "BEAR"):
        oi_surge_threshold = 0.012
        ls_short_threshold = 0.8
    elif regime == "BULL":
        oi_surge_threshold = 0.020
        ls_short_threshold = 0.6
    else:  # RANGING, MILDLY_BEARISH
        oi_surge_threshold = 0.015
        ls_short_threshold = 0.7

    if oi_roc <= oi_surge_threshold or ls_ratio >= ls_short_threshold:
        return None

    # Confirm: price must be rising
    if price_change < -0.005:
        return None

    vol_regime = _check_vol_regime(df_15m, idx)
    strength = min(abs(oi_roc) * 8, 0.8)
    source = 'cascade_long'

    return {
        "direction": "LONG",
        "strength": strength,
        "oi_roc": oi_roc,
        "ls_ratio": ls_ratio,
        "price_change": price_change,
        "vol_regime": vol_regime,
        "source": source,
        "detail": f"LONG cascade: OI ROC={oi_roc:.4f}, LS={ls_ratio:.2f}, price={price_change:.4f}, vol={vol_regime}"
    }


def _check_vol_regime(df_15m, idx):
    """Check vol regime (LOW/MID/HIGH)."""
    if idx < 20:
        return 'UNKNOWN'
    closes = df_15m['Close'].values.astype(float)[:idx+1]
    returns = np.diff(np.log(closes))
    if len(returns) < 20:
        return 'UNKNOWN'
    vol_20bar = np.std(returns[-20:])
    vols = [np.std(returns[i-20:i]) for i in range(20, len(returns))]
    if len(vols) < 30:
        return 'UNKNOWN'
    vols = np.array(vols)
    p33 = np.percentile(vols, 33)
    p67 = np.percentile(vols, 67)
    if vol_20bar < p33:
        return 'LOW'
    elif vol_20bar < p67:
        return 'MID'
    else:
        return 'HIGH'


class LiquidationCascadeStrategy(BaseStrategy):
    name = 'liquidation_cascade'
    strategy_type = 'event'
    description = 'v6: LONG-only + whale_watch confirmation. No SHORT signals.'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        global _last_cascade_ts

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        regime = data.get('regime', 'RANGING')
        if not price or not atr or df_15m is None or idx is None:
            return None

        # ── COOLDOWN ──
        now = time.time()
        if now - _last_cascade_ts < CASCADE_COOLDOWN:
            return None

        # ── WHALE_WATCH CONFIRMATION (REQUIRED) ──
        if not _check_whale_watch_confirms(data):
            return None

        # ── LONG CASCADE DETECTION ONLY ──
        cascade = _check_long_cascade(data, df_15m, idx, regime)
        if not cascade:
            return None

        direction = cascade['direction']  # Always LONG
        strength = cascade['strength']
        source = cascade['source']

        # ── MOMENTUM CONFIRMATION ──
        closes = df_15m['Close'].values.astype(float)
        if idx >= 3:
            mom_3 = (closes[idx] - closes[idx-3]) / closes[idx-3]
            if mom_3 < -0.003:
                return None  # Need positive momentum for LONG cascade

        # ── CONVICTION ──
        conviction = min(0.45 + strength * 0.40, 0.90)
        if conviction < 0.45:
            return None

        # ── TP/SL (regime-adaptive) ──
        tp_mult_base = {"BULL": 2.0, "BEAR": 1.8, "RANGING": 2.0, "STRESS": 1.5, "MILDLY_BEARISH": 1.8}.get(regime, 2.0)
        sl_mult_base = {"BULL": 1.0, "BEAR": 1.2, "RANGING": 1.0, "STRESS": 0.8, "MILDLY_BEARISH": 1.0}.get(regime, 1.0)

        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(tp_mult_base, tp_mult_base * 1.5, tp_mult_base * 2.5), sl_mult=sl_mult_base)

        # Update cooldown
        _last_cascade_ts = now

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=f"Liq cascade v6 ({source}) LONG: {cascade['detail']} | whale_watch confirms",
            bypass_gates=True,
            details={
                'version': 'v6',
                'source': source,
                'oi_roc': cascade.get('oi_roc', 0),
                'ls_ratio': cascade.get('ls_ratio', 0),
                'price_change': cascade.get('price_change', 0),
                'vol_regime': cascade.get('vol_regime', 'UNKNOWN'),
                'strength': strength,
                'regime': regime,
                'whale_confirmed': True,
            },
        )
