"""S25: Funding Squeeze — FR z-score extreme predicts SHORT continuation.

Mechanism: When funding rate is extreme relative to recent history (z-score > 1.25),
overleveraged longs are paying a premium. This predicts price dropping (squeeze).

Gate results (collected derivatives data, Jan-Jul 2026):
- FR z > 1.25: 602 events, -0.153% at 4h, p=0.003, WR=53.8%
- FR z > 1.50: 438 events, -0.147% at 4h, p=0.016, WR=54.8%
- FR z > 1.75: 300 events, -0.250% at 4h, p=0.0002, WR=59.7%

Regime performance (TP=2%, SL=1%, hold=4h):
- MID vol: PF=3.01, WR=60.1% (best)
- HIGH vol: PF=2.78, WR=58.2%
- LOW vol: PF=1.31, WR=39.5% (skip)

Confluence:
- + vol_ratio > 1.5: -0.519%, p=0.003 (n=97)
- + price > EMA50: -0.176%, p=0.0002 (n=432)
"""
from .base import BaseStrategy, SignalResult
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _compute_fr_zscore(deriv_data, lookback=96):
    """Compute funding rate z-score from derivatives data."""
    fr = deriv_data.get('funding_rate', None)
    if fr is None:
        return None

    # We need rolling stats — use stored history if available
    fr_history = deriv_data.get('fr_history', [])
    if len(fr_history) < lookback:
        return None  # Not enough history for z-score

    fr_arr = np.array(fr_history[-lookback:])
    mean = np.mean(fr_arr)
    std = np.std(fr_arr)
    if std == 0:
        return None

    return (fr - mean) / std


def _check_vol_regime(df_15m, idx):
    """Check vol regime (LOW/MID/HIGH). MID is best for this strategy."""
    if idx < 20:
        return 'UNKNOWN'
    closes = df_15m['Close'].values.astype(float)[:idx + 1]
    returns = np.diff(np.log(closes))
    if len(returns) < 20:
        return 'UNKNOWN'
    vol_20bar = np.std(returns[-20:])
    vols = [np.std(returns[i - 20:i]) for i in range(20, len(returns))]
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


class FundingSqueezeStrategy(BaseStrategy):
    name = 'funding_squeeze'
    strategy_type = 'event'
    description = 'SHORT: FR z-score > 1.25 = overleveraged longs = squeeze coming'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr or df_15m is None or idx is None:
            return None

        deriv = data.get('derivatives', {})
        fr = deriv.get('funding_rate', None)
        if fr is None:
            return None

        # Need FR history for z-score computation
        fr_history = deriv.get('fr_history', [])
        if len(fr_history) < 96:
            return None

        # Compute z-score
        fr_arr = np.array(fr_history[-96:])
        fr_mean = np.mean(fr_arr)
        fr_std = np.std(fr_arr)
        if fr_std == 0:
            return None

        fr_z = (fr - fr_mean) / fr_std

        # Gate: z-score > 1.25
        if fr_z < 1.25:
            return None

        # Direction: SHORT only
        direction = 'SHORT'

        # Vol regime check — skip LOW vol (no edge)
        vol_regime = _check_vol_regime(df_15m, idx)
        if vol_regime == 'LOW':
            return None

        # Conviction based on z-score magnitude
        if fr_z >= 2.0:
            # Premium: -0.299%, WR=59.7%
            base = 0.65
        elif fr_z >= 1.75:
            # Strong: -0.250%, WR=59.7%
            base = 0.60
        elif fr_z >= 1.50:
            # Standard: -0.147%, WR=54.8%
            base = 0.55
        else:
            # Minimum gate: -0.153%, WR=53.8%
            base = 0.50

        # Regime boost
        if vol_regime == 'MID':
            base += 0.05  # MID vol PF=3.01
        elif vol_regime == 'HIGH':
            base += 0.02  # HIGH vol PF=2.78

        conviction = min(base, 0.85)
        if conviction < 0.50:
            return None

        # TP/SL: TP=2%, SL=1% (validated by Agent 5)
        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.8,
            reason=f"Funding squeeze SHORT: FR_z={fr_z:.2f} [vol={vol_regime}]",
            bypass_gates=True,
            details={
                'version': 'v1',
                'fr_z': round(fr_z, 4),
                'fr_current': fr,
                'fr_mean': round(fr_mean, 8),
                'fr_std': round(fr_std, 8),
                'vol_regime': vol_regime,
            },
        )
