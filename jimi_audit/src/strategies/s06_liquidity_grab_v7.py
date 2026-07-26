"""S06: Liquidity Grab v7 — L2 orderbook mean-reversion.

v6 → v7 CHANGES:
1. KILLED TAKER Z-SCORE — 8-agent FAIL, no predictive edge at any horizon
2. NEW SIGNAL SOURCE: L2 orderbook ratio (ob_ratio) from ob_collector
3. NEW HORIZON: 2h TP (8 bars) — OB signal decays after 2h, strongest at 15m-2h
4. TWO SCENARIOS:
   A. BEAR + bid-heavy OB → LONG mean-reversion (57% WR, n=45k, p≈0)
   B. OB spike reversal → LONG after sudden ask wall disappears (54.4% WR, n=60k)
5. Regime-aware: Scenario A requires BEAR regime (mean-reversion setup)
6. Tight SL: 0.5 ATR — mean-reversion trades fail fast
7. Removed S/R breakout logic — was built on null signal (taker z-score)

Data source: ob_history/ob_historical.csv (ob_collector, 60s snapshots)
Performance target: 55%+ WR with 2h horizon, statistical significance p<0.05

Research basis:
- L2 analysis (2026-07-25): 90M OB rows analyzed, 200k sampled
- ob_ratio at 2h: 53.9% WR LONG, 45.2% WR SHORT (8.7pp gap)
- BEAR+LONG(ob>0.15) at 2h: 57.0% WR, +0.036% mean, n=45,370
- OB spike reversal at 2h: 54.4% WR, +0.037% mean, n=60,646
"""
from .base import BaseStrategy, SignalResult
import numpy as np

BAD_HOURS = {4, 5, 6, 19, 20, 22, 23}

# Scenario thresholds (from L2 analysis)
OB_RATIO_BID_THRESH = 0.15      # Scenario A: bid-heavy threshold
OB_SPIKE_THRESH = 0.5           # Scenario B: spike detection threshold
OB_SPIKE_REVERSAL_BARS = 3      # Scenario B: bars to detect reversal
TP_BARS = 8                     # 2h = 8 x 15min bars
SL_MULT = 0.5                   # Tight SL for mean-reversion
TP1_MULT = 1.0                  # Conservative TP1
TP2_MULT = 1.8                  # Moderate TP2
TP3_MULT = 3.0                  # Extended TP3


class LiquidityGrabV7(BaseStrategy):
    name = "liquidity_grab"
    strategy_type = "structure"
    version = "v7"

    def check(self, df_15m, idx, **kwargs):
        """Check for liquidity grab v7 signals.

        Requires kwargs:
            ob_ratio: current orderbook bid/ask ratio
            ob_ratio_prev: ob_ratio N bars ago (for delta)
            regime: current market regime (BULL/BEAR/RANGING/STRESS)
        """
        if idx < 20:
            return None

        price = df_15m["Close"].iloc[idx]
        ts = df_15m.index[idx] if hasattr(df_15m.index[idx], 'hour') else None
        hour = ts.hour if ts else kwargs.get('hour', 12)
        if hour in BAD_HOURS:
            return None

        regime = kwargs.get('regime', 'NEUTRAL')
        if regime == 'STRESS':
            return None

        atr = kwargs.get('atr', None)
        if atr is None or atr <= 0:
            # Calculate from recent bars
            highs = df_15m["High"].iloc[max(0, idx-16):idx].values.astype(float)
            lows = df_15m["Low"].iloc[max(0, idx-16):idx].values.astype(float)
            atr = np.mean(highs - lows)
        if atr <= 0:
            return None

        ob_ratio = kwargs.get('ob_ratio', None)
        if ob_ratio is None:
            return None

        ob_ratio_prev = kwargs.get('ob_ratio_prev', ob_ratio)
        ob_delta = ob_ratio - ob_ratio_prev

        vol_ratio = kwargs.get('vol_ratio', 1.0)

        # ── SCENARIO A: BEAR + bid-heavy OB → LONG ──
        # Mean-reversion: price dropped into BEAR, OB shows bid wall → bounce expected
        scenario_a = False
        if regime == 'BEAR' and ob_ratio > OB_RATIO_BID_THRESH:
            # Additional confirmation: volume should be normal+ (not dead)
            if vol_ratio >= 0.8:
                scenario_a = True

        # ── SCENARIO B: OB spike reversal → LONG ──
        # Sudden disappearance of ask wall (negative spike that reverses)
        scenario_b = False
        if ob_delta < -OB_SPIKE_THRESH:
            # Check if previous bars had even more negative delta (reversal pattern)
            ob_delta_prev = kwargs.get('ob_delta_prev', 0)
            if ob_delta_prev < -OB_SPIKE_THRESH and ob_delta > ob_delta_prev:
                # Delta is negative but less negative than before → reversal
                scenario_b = True

        if not scenario_a and not scenario_b:
            return None

        # ── CONVICTION ──
        if scenario_a:
            # Scenario A: BEAR + bid-heavy
            base = 0.50
            ob_bonus = min((ob_ratio - OB_RATIO_BID_THRESH) * 0.30, 0.20)
            vol_bonus = min((vol_ratio - 0.8) * 0.10, 0.10)
            regime_bonus = 0.05  # Already in BEAR, that's the setup
            conviction = min(base + ob_bonus + vol_bonus + regime_bonus, 0.85)
            scenario_label = "A"
            reason_detail = f"BEAR+bid_heavy ob={ob_ratio:.3f}"
        else:
            # Scenario B: spike reversal
            base = 0.48
            spike_bonus = min(abs(ob_delta) * 0.15, 0.20)
            conviction = min(base + spike_bonus, 0.80)
            scenario_label = "B"
            reason_detail = f"spike_reversal delta={ob_delta:.3f}"

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
            direction="LONG",  # Both scenarios are LONG-only
            conviction=conviction,
            entry=price,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            sl_pct=sl_pct,
            tp1_pct=tp1_pct,
            size_mult=0.8,
            reason=(f"Liq grab v7-{scenario_label} LONG: {reason_detail} "
                    f"regime={regime} vol={vol_ratio:.1f}x"),
            bypass_gates=False,
            details={
                "version": "v7",
                "scenario": scenario_label,
                "ob_ratio": round(ob_ratio, 4),
                "ob_delta": round(ob_delta, 4),
                "vol_ratio": round(vol_ratio, 2),
                "regime": regime,
                "tp_bars": TP_BARS,
                "atr": round(atr, 2),
            },
        )
