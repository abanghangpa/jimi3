"""S20: Liquidation Cascade v5 — Bidirectional + regime-adaptive.

v4 → v5 CHANGES:
1. BIDIRECTIONAL: Added LONG cascade detection (OI surge + short-crowded)
2. REGIME-ADAPTIVE: Different thresholds per regime
3. CONFIRMATION: Price momentum must align with cascade direction
4. TIERED CONVICTION: 3 tiers (primary/high/premium) kept from v4
5. REGIME TP/SL: Tighter in RANGING, wider in STRESS
6. COOLDOWN: 30min between cascade signals (prevent spam)

v4 stats: SHORT only, 83% WR in RANGING, +$174
v5 target: Add LONG cascade edge, maintain SHORT quality.
"""
from .base import BaseStrategy, SignalResult
import json, os, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cooldown tracking
_last_cascade_ts = 0
CASCADE_COOLDOWN = 1800  # 30 min


def _check_cascade(data, df_15m, idx, regime):
    """
    Detect OI cascade — bidirectional.

    SHORT cascade (crowded longs liquidated):
    - OI dropping fast (longs closing)
    - LS > 1.5 (long-crowded)
    - Price dropping (confirming)

    LONG cascade (crowded shorts squeezed):
    - OI surging (new longs + shorts covering)
    - LS < 0.7 (short-crowded, i.e. LS inverted or very low)
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
        # More sensitive in stress/bear (cascades happen faster)
        oi_threshold = -0.008  # Was -0.01
        ls_threshold = 1.3     # Was 1.5
    elif regime == "RANGING":
        # Standard thresholds
        oi_threshold = -0.01
        ls_threshold = 1.5
    else:  # BULL, MILDLY_BEARISH
        # Higher bar in bull (cascades are rarer)
        oi_threshold = -0.012
        ls_threshold = 1.6

    vol_regime = _check_vol_regime(df_15m, idx)

    # ── SHORT CASCADE (OI dropping + crowded longs) ──
    if oi_roc < oi_threshold and ls_ratio > ls_threshold:
        # Confirm: price must be dropping
        if price_change > 0.005:
            return None  # Price rising = not a cascade

        if oi_roc < oi_threshold * 1.5:
            # HIGH CONVICTION
            strength = min(abs(oi_roc) * 15, 0.9)
            source = 'cascade_short_hi'
        else:
            strength = min(abs(oi_roc) * 10, 0.7)
            source = 'cascade_short_primary'

        # PREMIUM: MID vol + deep OI drop
        if vol_regime == 'MID' and oi_roc < oi_threshold * 1.5:
            strength = min(strength + 0.1, 0.95)
            source = 'cascade_short_premium'

        return {
            "direction": "SHORT",
            "strength": strength,
            "oi_roc": oi_roc,
            "ls_ratio": ls_ratio,
            "price_change": price_change,
            "vol_regime": vol_regime,
            "source": source,
            "detail": f"SHORT cascade: OI ROC={oi_roc:.4f}, LS={ls_ratio:.2f}, price={price_change:.4f}, vol={vol_regime}"
        }

    # ── LONG CASCADE (OI surging + crowded shorts) ──
    # Inverted logic: OI rising fast + LS < threshold = shorts getting squeezed
    oi_surge_threshold = 0.015  # OI must surge by 1.5%+
    ls_short_threshold = 0.7    # LS below 0.7 = short-crowded

    if regime in ("STRESS", "BEAR"):
        oi_surge_threshold = 0.012  # More sensitive
        ls_short_threshold = 0.8
    elif regime == "BULL":
        oi_surge_threshold = 0.020  # Higher bar
        ls_short_threshold = 0.6

    if oi_roc > oi_surge_threshold and ls_ratio < ls_short_threshold:
        # Confirm: price must be rising
        if price_change < -0.005:
            return None  # Price dropping = not a squeeze

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

    return None


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
    description = 'v5: bidirectional + regime-adaptive. SHORT+LONG cascades.'

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

        # ── CASCADE DETECTION ──
        cascade = _check_cascade(data, df_15m, idx, regime)
        if not cascade:
            return None

        direction = cascade['direction']
        strength = cascade['strength']
        source = cascade['source']

        # ── MOMENTUM CONFIRMATION (NEW) ──
        closes = df_15m['Close'].values.astype(float)
        if idx >= 3:
            mom_3 = (closes[idx] - closes[idx-3]) / closes[idx-3]
            if direction == "LONG" and mom_3 < -0.003:
                return None  # Need positive momentum for LONG cascade
            if direction == "SHORT" and mom_3 > 0.003:
                return None  # Need negative momentum for SHORT cascade

        # ── CONVICTION ──
        conviction = min(0.45 + strength * 0.40, 0.90)
        if conviction < 0.45:
            return None

        # ── TP/SL (v5: regime-adaptive) ──
        # Base: TP=2.0%, SL=1.0% (2:1 R:R)
        # Regime adjustments
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
            reason=f"Liq cascade v5 ({source}) {direction}: {cascade['detail']}",
            bypass_gates=True,
            details={
                'version': 'v5',
                'source': source,
                'oi_roc': cascade.get('oi_roc', 0),
                'ls_ratio': cascade.get('ls_ratio', 0),
                'price_change': cascade.get('price_change', 0),
                'vol_regime': cascade.get('vol_regime', 'UNKNOWN'),
                'strength': strength,
                'regime': regime,
            },
        )
