"""S20: Liquidation Cascade v6.1 — Provisional deployment.

v6 → v6.1 CHANGES:
1. PROVISIONAL CONFIG: TP=1.5xATR, SL=1.5xATR (equal multipliers)
2. CONVICTION THRESHOLD: 0.70 (up from 0.45)
3. Based on optimization: 18 trades, 70% WR, PF 2.05
4. NOT STATISTICALLY SIGNIFICANT (MC p=0.14, CI includes zero)
5. Provisional flag: will be validated with 30+ live trades

v6.1 stats (backtest): 13 trades (conv>=0.70), 70% WR, PF 2.05, MC p=0.14
Status: PROVISIONAL — needs 30+ trades for statistical confirmation
"""
from .base import BaseStrategy, SignalResult
import json, os, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cooldown tracking
_last_cascade_ts = 0
CASCADE_COOLDOWN = 1800  # 30 min

# ── PROVISIONAL CONFIG ──
PROVISIONAL = True
PROVISIONAL_TP_MULT = 1.5   # x ATR
PROVISIONAL_SL_MULT = 1.5   # x ATR
PROVISIONAL_CONV_THRESHOLD = 0.70
PROVISIONAL_MIN_TRADES_FOR_CONFIRM = 30


def _check_whale_watch_confirms(data):
    """Check if whale_watch fired LONG."""
    strategy_signals = data.get('strategy_signals', {})
    whale = strategy_signals.get('whale_watch', {})
    if whale.get('direction') == 'LONG' and whale.get('fired', False):
        return True
    multi = data.get('multi_strategy', {})
    for sig in multi.get('all_signals', []):
        if sig.get('strategy') == 'whale_watch' and sig.get('direction') == 'LONG':
            return True
    return False


def _check_long_cascade(data, df_15m, idx, regime):
    """Detect LONG cascade (crowded shorts squeezed)."""
    deriv = data.get('derivatives', {})
    oi_roc = deriv.get('oi_roc_1h', 0)
    ls_ratio = deriv.get('ls_ratio', 1.0)

    closes = df_15m['Close'].values.astype(float)
    if idx < 5:
        return None
    price_change = (closes[idx] - closes[idx-5]) / closes[idx-5]

    if regime in ("STRESS", "BEAR"):
        oi_surge_threshold = 0.012
        ls_short_threshold = 0.8
    elif regime == "BULL":
        oi_surge_threshold = 0.020
        ls_short_threshold = 0.6
    else:
        oi_surge_threshold = 0.015
        ls_short_threshold = 0.7

    if oi_roc <= oi_surge_threshold or ls_ratio >= ls_short_threshold:
        return None

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
    description = 'v6.1 PROVISIONAL: LONG-only + whale_watch + TP=SL=1.5xATR, conv>=0.70'

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
                return None

        # ── CONVICTION (provisional: 0.70 threshold) ──
        conviction = min(0.45 + strength * 0.40, 0.90)
        if conviction < PROVISIONAL_CONV_THRESHOLD:
            return None

        # ── TP/SL (provisional: fixed 1.5x ATR both sides) ──
        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr,
            tp_mults=(PROVISIONAL_TP_MULT, PROVISIONAL_TP_MULT * 1.5, PROVISIONAL_TP_MULT * 2.5),
            sl_mult=PROVISIONAL_SL_MULT)

        # Update cooldown
        _last_cascade_ts = now

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.5,  # Reduced size for provisional
            reason=f"Liq cascade v6.1 PROVISIONAL ({source}) LONG: {cascade['detail']} | whale confirms | TP=SL=1.5xATR",
            bypass_gates=True,
            details={
                'version': 'v6.1-provisional',
                'provisional': True,
                'source': source,
                'oi_roc': cascade.get('oi_roc', 0),
                'ls_ratio': cascade.get('ls_ratio', 0),
                'price_change': cascade.get('price_change', 0),
                'vol_regime': cascade.get('vol_regime', 'UNKNOWN'),
                'strength': strength,
                'regime': regime,
                'whale_confirmed': True,
                'tp_mult': PROVISIONAL_TP_MULT,
                'sl_mult': PROVISIONAL_SL_MULT,
                'conv_threshold': PROVISIONAL_CONV_THRESHOLD,
                'mc_p_value': 0.14,
                'sample_size': 18,
                'confirmation_threshold': PROVISIONAL_MIN_TRADES_FOR_CONFIRM,
                'note': 'PROVISIONAL: 70% WR / PF 2.05 on 18 trades. MC not significant (p=0.14). Needs 30+ live trades to confirm.',
            },
        )
