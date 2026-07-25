"""S25: Funding Squeeze v2 — Research-backed FR extreme + OI confirmation.

v1 → v2 CHANGES (based on forensic analysis + research):
1. KILLED ASIA SESSION — 44.3% WR at 4h, 39.1% at 8h. Money incinerator.
2. TP AT 8H — edge is 2-3x stronger at 8h vs 4h (z>1.75: 60.5% WR at 8h)
3. RAISED Z-SCORE THRESHOLD — 1.25 → 1.75 (60.5% WR vs 52.3%)
4. ADDED OI CONFIRMATION — OI rising = overleveraged longs adding = stronger edge
5. ADDED CUMULATIVE FR — FR z > 1.5 for 3+ readings filters flash spikes
6. VOL REGIME: MID/HIGH only (LOW has no edge)

Research basis:
- Zhang (SSRN): "Funding Rate Mechanism in Perpetual Futures" — FR mean-reverts
- Presto Research (2024): FR extremes are contrarian signals
- QuantJourney (2025): "FR + OI together are stronger than FR alone"
- Forensic (2026-07-25): z>1.75 at 8h = 60.5% WR, +0.154% mean, n=466

Gate targets: 60%+ WR, PF > 1.5, p < 0.05
"""
from .base import BaseStrategy, SignalResult
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Session filter: EU + US only (ASIA killed)
EU_HOURS = set(range(8, 14))
US_HOURS = set(range(14, 22))
ALLOWED_HOURS = EU_HOURS | US_HOURS

# Thresholds
FR_Z_THRESH = 1.75          # Instantaneous z-score gate
FR_Z_CUMULATIVE = 1.5       # Cumulative threshold (3+ bars)
FR_CUMULATIVE_BARS = 3      # Consecutive bars above cumulative threshold
TP_BARS = 8                 # 8h = 32 x 15min bars (but we use 8-bar TP in executor)
SL_MULT = 0.8               # Slightly wider SL for 8h hold
TP1_MULT = 1.5
TP2_MULT = 2.5
TP3_MULT = 4.0


class FundingSqueezeStrategy(BaseStrategy):
    min_vol_ratio = 0.0  # We check vol regime internally
    name = 'funding_squeeze'
    strategy_type = 'event'
    description = 'v2: SHORT FR extreme + OI rising + EU/US only + 8h TP'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        if not price or not atr or df_15m is None or idx is None:
            return None

        # ── SESSION FILTER: EU + US only ──
        ts = data.get('timestamp', '')
        if ts:
            try:
                hour = int(ts[11:13])
                if hour not in ALLOWED_HOURS:
                    return None
            except (ValueError, IndexError):
                return None

        # ── READ DERIVATIVES DATA ──
        deriv = data.get('derivatives', {})
        fr = deriv.get('funding_rate', None)
        if fr is None:
            return None

        fr_history = deriv.get('fr_history', [])
        if len(fr_history) < 96:
            return None

        # ── COMPUTE FR Z-SCORE ──
        fr_arr = np.array(fr_history[-96:])
        fr_mean = np.mean(fr_arr)
        fr_std = np.std(fr_arr)
        if fr_std == 0:
            return None

        fr_z = (fr - fr_mean) / fr_std

        # ── INSTANTANEOUS GATE: z > 1.75 ──
        if fr_z < FR_Z_THRESH:
            return None

        # ── CUMULATIVE FR CHECK: z > 1.5 for 3+ consecutive bars ──
        if len(fr_history) >= FR_CUMULATIVE_BARS:
            recent_fr = np.array(fr_history[-FR_CUMULATIVE_BARS:])
            recent_z = (recent_fr - fr_mean) / fr_std
            if not np.all(recent_z > FR_Z_CUMULATIVE):
                return None  # Not sustained — flash spike, skip

        # ── OI CONFIRMATION: OI must be rising ──
        oi = deriv.get('oi', None)
        oi_history = deriv.get('oi_history', [])
        if oi is not None and len(oi_history) >= 4:
            oi_arr = np.array(oi_history[-4:])
            oi_change = (oi - oi_arr[0]) / oi_arr[0] if oi_arr[0] > 0 else 0
            if oi_change < 0:
                return None  # OI falling = longs closing = edge weakens
        # If no OI data, proceed without OI filter (degrade gracefully)

        # ── VOL REGIME: MID or HIGH only ──
        vol_regime = self._check_vol_regime(df_15m, idx)
        if vol_regime == 'LOW':
            return None

        # ── DIRECTION: SHORT only ──
        direction = 'SHORT'

        # ── CONVICTION ──
        if fr_z >= 2.5:
            base = 0.70  # Extreme: 61% WR at 8h
        elif fr_z >= 2.0:
            base = 0.65  # Strong: 58% WR at 8h
        elif fr_z >= 1.75:
            base = 0.60  # Standard: 60.5% WR at 8h
        else:
            base = 0.50

        # Vol regime boost
        if vol_regime == 'MID':
            base += 0.05  # MID vol best for this strategy
        elif vol_regime == 'HIGH':
            base += 0.03

        # OI boost
        if oi is not None and len(oi_history) >= 4:
            oi_arr = np.array(oi_history[-4:])
            oi_change = (oi - oi_arr[0]) / oi_arr[0] if oi_arr[0] > 0 else 0
            if oi_change > 0.02:
                base += 0.05  # Strong OI rise = more overleveraged
            elif oi_change > 0:
                base += 0.02

        conviction = min(base, 0.85)
        if conviction < 0.55:
            return None

        # ── TP/SL ──
        sl = price + SL_MULT * atr
        tp1 = price - TP1_MULT * atr
        tp2 = price - TP2_MULT * atr
        tp3 = price - TP3_MULT * atr

        sl_pct = (SL_MULT * atr / price) * 100
        tp1_pct = (TP1_MULT * atr / price) * 100

        return SignalResult(
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            direction=direction,
            conviction=conviction,
            entry=price,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            sl_pct=sl_pct,
            tp1_pct=tp1_pct,
            size_mult=0.8,
            reason=(f"Funding squeeze v2 SHORT: FR_z={fr_z:.2f} "
                    f"[vol={vol_regime}] OI_chg={oi_change:+.3f}" if oi else
                    f"Funding squeeze v2 SHORT: FR_z={fr_z:.2f} [vol={vol_regime}]"),
            bypass_gates=True,
            details={
                'version': 'v2',
                'fr_z': round(fr_z, 4),
                'fr_current': fr,
                'fr_mean': round(fr_mean, 8),
                'fr_std': round(fr_std, 8),
                'vol_regime': vol_regime,
                'oi_change': round(oi_change, 4) if oi else None,
                'cumulative_check': True,
            },
        )

    @staticmethod
    def _check_vol_regime(df_15m, idx):
        """Check vol regime (LOW/MID/HIGH)."""
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
