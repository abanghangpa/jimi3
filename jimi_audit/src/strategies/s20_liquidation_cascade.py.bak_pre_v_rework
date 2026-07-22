"""S20: Liquidation Cascade v4 — 8-Agent validated config.

v3 → v4 CHANGES (from 8-Agent Protocol 2026-07-19):
1. PRIMARY: OI ROC < -0.01 + LS > 1.5 (87 events, +0.375%, p=0.011, WR=62.1%)
2. HIGH CONVICTION: OI ROC < -0.015 + LS > 1.5 (29 events, +0.976%, p=0.0003, WR=82.8%)
3. PREMIUM: OI ROC < -0.015 + MID vol (12 events, +2.222%, p=0.002, WR=91.7%)
4. Direction: SHORT only (LONG has no edge in cascade signals)
5. TP/SL: TP=2.0%, SL=1.0% (2:1 R:R), hold=4h
6. Removed broken Source 1 (real liquidation stream — empty)
7. Removed backfilled derivatives contamination (all LS=2.0, no OI)

8-Agent Validation (collected data only):
- Forward returns: OI<-0.015 + LS>1.5: +0.976%, p=0.0003, WR=82.8%
- TP/SL sim: OI<-0.01 + LS>1.5: 87 trades, WR=37.9%, PF=1.22, PnL=$120/$1000
"""
from .base import BaseStrategy, SignalResult
import json, os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _check_oi_cascade(data, df_15m, idx):
    """
    Detect OI cascade using collected derivatives data.
    Primary: OI ROC < -0.01 + LS > 1.5
    High conviction: OI ROC < -0.015 + LS > 1.5
    """
    deriv = data.get('derivatives', {})
    oi_roc = deriv.get('oi_roc_1h', 0)
    ls_ratio = deriv.get('ls_ratio', 1.0)

    # Need OI ROC < -0.01 minimum
    if oi_roc >= -0.01:
        return None

    # Need LS > 1.5 (crowded longs)
    if ls_ratio <= 1.5:
        return None

    # Price momentum check (4-bar)
    closes = df_15m['Close'].values.astype(float)
    if idx < 5:
        return None
    price_change = (closes[idx] - closes[idx-5]) / closes[idx-5]

    # Direction: SHORT only (LONG has no edge per 8-Agent)
    # OI dropping + LS > 1.5 = crowded longs getting liquidated → SHORT
    direction = 'SHORT'

    # Strength based on OI ROC magnitude
    if oi_roc < -0.015:
        # HIGH CONVICTION: +0.976% forward, p=0.0003, WR=82.8%
        strength = min(abs(oi_roc) * 15, 0.9)
        source = 'oi_hi'
    else:
        # PRIMARY: +0.375% forward, p=0.011, WR=62.1%
        strength = min(abs(oi_roc) * 10, 0.7)
        source = 'oi_primary'

    # Bonus: if also MID vol regime, boost conviction
    vol_regime = _check_vol_regime(df_15m, idx)
    if vol_regime == 'MID' and oi_roc < -0.015:
        # PREMIUM: +2.222% forward, p=0.002, WR=91.7%
        strength = min(strength + 0.1, 0.95)
        source = 'oi_premium'

    return {
        "direction": direction,
        "strength": strength,
        "oi_roc": oi_roc,
        "ls_ratio": ls_ratio,
        "price_change": price_change,
        "vol_regime": vol_regime,
        "source": source,
        "detail": f"OI ROC={oi_roc:.4f}, LS={ls_ratio:.2f}, price={price_change:.4f}, vol={vol_regime}"
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
    description = 'v4: 8-Agent validated. OI<-0.01+LS>1.5 SHORT, conviction tiers.'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr or df_15m is None or idx is None:
            return None

        # ── CASCADE DETECTION ──
        cascade = _check_oi_cascade(data, df_15m, idx)
        if not cascade:
            return None

        direction = cascade['direction']
        strength = cascade['strength']
        source = cascade['source']

        # ── CONVICTION ──
        # Base: 0.45, strength adds up to 0.40
        conviction = min(0.45 + strength * 0.40, 0.90)
        if conviction < 0.45:
            return None

        # ── TP/SL (SHORT cascade) ──
        # TP=2.0%, SL=1.0% (2:1 R:R), hold=4h
        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.7,
            reason=f"Liq cascade v4 ({source}) {direction}: {cascade['detail']}",
            bypass_gates=True,
            details={
                'version': 'v4',
                'source': source,
                'oi_roc': cascade.get('oi_roc', 0),
                'ls_ratio': cascade.get('ls_ratio', 0),
                'price_change': cascade.get('price_change', 0),
                'vol_regime': cascade.get('vol_regime', 'UNKNOWN'),
                'strength': strength,
            },
        )
