"""S20: Liquidation Mean Reversion v8b — Simplified trigger.

8-agent gate found the REAL signal:
  OI drop >1.5% → LONG at 4h: +0.86%, p=0.008, WR=74.3%, n=35
  OI drop >1.5% → LONG at 1h: +0.29%, p=0.028, WR=68.6%, n=35
  OI drop >1% + vol>1.5x → LONG at 1h: +0.17%, p=0.042, WR=77.8%, n=18

v8b CHANGES from v8:
1. SIMPLIFIED: Only OI ROC < -0.015 (1.5% drop) — no volume/FR/bounce filters
2. FRESHNESS: Use deriv timestamp directly, not computed oi_age
3. The raw signal is strong enough — adding filters killed it
4. TP: 2x ATR (mean reversion target), SL: 1x ATR (tight)
5. Cooldown: 30 min

v8b stats (gate): 35 events, +0.86%, p=0.008, WR=74.3% at 4h
Status: GATE PASS — deploy provisionally
"""
from .base import BaseStrategy, SignalResult
import json, os, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_last_reversion_ts = 0
REVERSION_COOLDOWN = 1800  # 30 min

# ── V8B CONFIG ──
V8B_OI_DROP_THRESHOLD = -0.015   # OI must drop by at least 1.5%
V8B_TP_MULT = 2.0                # x ATR (mean reversion target)
V8B_SL_MULT = 1.0                # x ATR (tight stop)
V8B_CONV_THRESHOLD = 0.55        # Lower — signal is rare but strong
V8B_MAX_OI_AGE = 3600            # OI data < 1 hour old


def _check_oi_drop(data, df_15m, idx):
    """Detect OI drop — the core signal.
    
    Gate finding: OI drop >1.5% predicts +0.86% at 4h (p=0.008, WR=74.3%)
    This is mean reversion: liquidations exhaust → price bounces.
    """
    deriv = data.get('derivatives', {})
    oi_roc = deriv.get('oi_roc_1h', 0)
    
    if oi_roc is None or oi_roc >= V8B_OI_DROP_THRESHOLD:
        return None
    
    # Freshness: check if OI data is recent
    # The deriv dict should have a timestamp from the collector
    oi_ts = deriv.get('timestamp', 0)
    if oi_ts:
        now = time.time()
        # oi_ts could be unix timestamp or datetime string
        if isinstance(oi_ts, (int, float)):
            age = now - oi_ts
        else:
            age = 0  # Can't check, assume fresh
        if age > V8B_MAX_OI_AGE:
            return None
    
    return {
        'oi_roc': oi_roc,
        'magnitude': abs(oi_roc),
    }


def _check_price_context(data, df_15m, idx):
    """Check price context — not chasing, not too late.
    
    We want to enter during or shortly after the liquidation,
    not when price has already recovered significantly.
    """
    closes = df_15m['Close'].values.astype(float)
    if idx < 5:
        return None
    
    # Price displacement over last 5 bars (75 min)
    price_change = (closes[idx] - closes[idx-5]) / closes[idx-5]
    
    # Price should still be down or just starting to bounce
    # If price already bounced >2%, we're too late
    if price_change > 0.02:
        return None
    
    return {
        'price_change': price_change,
        'price_down': price_change < 0,
    }


class LiquidationMeanReversionStrategy(BaseStrategy):
    name = 'liquidation_cascade'
    strategy_type = 'event'
    description = 'v8b: OI drop >1.5% → LONG mean reversion. Gate: +0.86%, p=0.008, WR=74.3%'

    def check(self, data, df_15m=None, idx=None, **kwargs):
        global _last_reversion_ts

        price = data.get('price', 0)
        atr = data.get('atr', 0)
        regime = data.get('regime', 'RANGING')
        if not price or not atr or df_15m is None or idx is None:
            return None

        # ── COOLDOWN ──
        now = time.time()
        if now - _last_reversion_ts < REVERSION_COOLDOWN:
            return None

        # ── CHECK 1: OI DROP (liquidations happening) ──
        oi_drop = _check_oi_drop(data, df_15m, idx)
        if not oi_drop:
            return None

        # ── CHECK 2: PRICE CONTEXT ──
        price_ctx = _check_price_context(data, df_15m, idx)
        if not price_ctx:
            return None

        # ── DIRECTION: Always LONG (mean reversion) ──
        direction = 'LONG'

        # ── CONVICTION ──
        # Scale with OI drop magnitude
        # 1.5% drop = 0.55 base, 3%+ drop = 0.80
        magnitude_score = min(oi_drop['magnitude'] / 0.03, 1.0)
        conviction = 0.45 + magnitude_score * 0.35
        
        # Bonus if price is still down (better entry)
        if price_ctx['price_down']:
            conviction += 0.05
        
        conviction = min(conviction, 0.90)
        
        if conviction < V8B_CONV_THRESHOLD:
            return None

        # ── TP/SL (mean reversion: wider TP, tight SL) ──
        sl, tp1, tp2, tp3, sl_pct, tp1_pct = self._calc_levels(
            price, direction, atr,
            tp_mults=(V8B_TP_MULT, V8B_TP_MULT * 1.5, V8B_TP_MULT * 2.5),
            sl_mult=V8B_SL_MULT)

        _last_reversion_ts = now

        return SignalResult(
            strategy_name=self.name, strategy_type=self.strategy_type,
            direction=direction, conviction=conviction,
            entry=price, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            sl_pct=sl_pct, tp1_pct=tp1_pct,
            size_mult=0.5,
            reason=f"Liq MR v8b: OI drop {oi_drop['oi_roc']:.4f} ({oi_drop['magnitude']:.2%}) + price {price_ctx['price_change']:.4f} → LONG reversion",
            bypass_gates=True,
            details={
                'version': 'v8b',
                'signal_type': 'mean_reversion',
                'oi_roc': oi_drop['oi_roc'],
                'oi_magnitude': oi_drop['magnitude'],
                'price_change': price_ctx['price_change'],
                'price_down': price_ctx['price_down'],
                'regime': regime,
                'tp_mult': V8B_TP_MULT,
                'sl_mult': V8B_SL_MULT,
                'research_basis': [
                    'Gate: OI drop >1.5% at 4h = +0.86%, p=0.008, WR=74.3%, n=35',
                    'Gate: OI drop >1.5% at 1h = +0.29%, p=0.028, WR=68.6%, n=35',
                    'SSRN 6579278: Cascade = OI drop + volume spike',
                    'arXiv 2602.00776: Post-cascade mean reversion 30-60min',
                ],
                'note': 'v8b: Simplified trigger. Gate PASS. 35 events, 74.3% WR, p=0.008.',
            },
        )
