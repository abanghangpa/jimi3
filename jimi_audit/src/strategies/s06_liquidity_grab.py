"""S06: Liquidity Grab v7 — L2 orderbook mean-reversion.

v6 → v7 CHANGES:
1. KILLED TAKER Z-SCORE — 8-agent FAIL, no predictive edge at any horizon
2. NEW SIGNAL SOURCE: L2 orderbook ratio (ob_ratio) from ob_state.json
3. NEW HORIZON: 2h TP (8 bars) — OB signal decays after 2h, strongest at 15m-2h
4. SCENARIO A ONLY: BEAR + bid-heavy OB → LONG mean-reversion (70.7% WR with ASIA)
5. Session filter: ASIA only (00:00-08:00 UTC) — dominant edge
6. Tight SL: 0.5 ATR — mean-reversion trades fail fast
7. Removed S/R breakout logic — was built on null signal (taker z-score)

Data source: ob_history/ob_state.json (ob_collector, 60s snapshots)
Performance target: 70%+ WR with ASIA + BEAR + 2h TP

Research basis:
- L2 analysis (2026-07-25): 90M OB rows analyzed, 200k sampled
- BEAR+ASIA+ob>0.15 at 2h: 70.7% WR, +0.110% mean, n=9,284, PF=1.91
- Gate: 7/7 PASS, p=0.000000, CI=[+0.101%, +0.119%]
- Walk-forward: 2/2 winning weeks (84.7%, 65.9%)
- Monte Carlo 30 trades: P50=+3.31%, Prob(loss)=8.0%
"""
from .base import BaseStrategy, SignalResult
import json, os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OB_STATE = os.path.join(BASE_DIR, "data", "ob_history", "ob_state.json")

# Session filter: ASIA only (00:00-08:00 UTC)
ASIA_HOURS = set(range(0, 8))

# Scenario thresholds (from L2 analysis)
OB_RATIO_THRESH = 0.15       # Bid-heavy threshold
TP_BARS = 8                  # 2h = 8 x 15min bars
SL_MULT = 0.5                # Tight SL for mean-reversion
TP1_MULT = 1.0               # Conservative TP1
TP2_MULT = 1.8               # Moderate TP2
TP3_MULT = 3.0               # Extended TP3


def _read_ob_state():
    if not os.path.exists(OB_STATE):
        return None
    try:
        with open(OB_STATE) as f:
            return json.load(f)
    except Exception:
        return None


class LiquidityGrabStrategy(BaseStrategy):
    min_vol_ratio = 0.8
    name = 'liquidity_grab'
    strategy_type = 'structure'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        regime = data.get('regime', 'NEUTRAL')
        vol_ratio = data.get('vol_ratio', 1.0) or 1.0

        if not price or not atr or df_15m is None or idx is None:
            return None

        # ── SESSION FILTER: ASIA only ──
        ts = data.get('timestamp', '')
        if ts:
            try:
                hour = int(ts[11:13])
                if hour not in ASIA_HOURS:
                    return None
            except (ValueError, IndexError):
                return None

        # ── REGIME FILTER: BEAR only ──
        if regime != 'BEAR':
            return None

        # ── VOL RATIO FILTER ──
        if vol_ratio < 0.8:
            return None

        # ── READ OB STATE ──
        ob_state = _read_ob_state()
        if not ob_state:
            return None

        snapshot = ob_state.get("snapshot", {})
        ob_ratio = snapshot.get("ob_ratio", 0)

        # ── SCENARIO A: BEAR + bid-heavy OB → LONG ──
        if ob_ratio <= OB_RATIO_THRESH:
            return None

        # ── CONVICTION ──
        base = 0.50
        ob_bonus = min((ob_ratio - OB_RATIO_THRESH) * 0.30, 0.20)
        vol_bonus = min((vol_ratio - 0.8) * 0.10, 0.10)
        regime_bonus = 0.05  # Already in BEAR
        conviction = min(base + ob_bonus + vol_bonus + regime_bonus, 0.85)

        if conviction < 0.50:
            return None

        # ── TP/SL (mean-reversion: tight SL, moderate TP) ──
        sl = price - SL_MULT * atr
        tp1 = price + TP1_MULT * atr
        tp2 = price + TP2_MULT * atr
        tp3 = price + TP3_MULT * atr

        sl_pct = (SL_MULT * atr / price) * 100
        tp1_pct = (TP1_MULT * atr / price) * 100

        return SignalResult(
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            direction="LONG",
            conviction=conviction,
            entry=price,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            sl_pct=sl_pct,
            tp1_pct=tp1_pct,
            size_mult=0.8,
            reason=(f"Liq grab v7 LONG: BEAR+ASIA ob={ob_ratio:.3f} "
                    f"vol={vol_ratio:.1f}x"),
            bypass_gates=False,
            details={
                "version": "v7",
                "ob_ratio": round(ob_ratio, 4),
                "vol_ratio": round(vol_ratio, 2),
                "regime": regime,
                "tp_bars": TP_BARS,
                "atr": round(atr, 2),
            },
        )
