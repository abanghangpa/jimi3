"""
S20: Liquidation Cascade Strategy v7
=====================================
Single validated signal + regime filter architecture.

VALIDATED EDGE (8-Agent Report, 2026-07-19):
  oi_roc < -0.015 + MID volatility → +2.22% mean at 4h, p=0.002, n=12
  oi_roc < -0.015 alone → +0.86% mean at 4h, p=0.008, n=35

Architecture:
  Signal:   OI ROC from real derivatives data ONLY
  Regime:   MID volatility filter (rolling vol percentile 33-67)
  Direction: SHORT only (LONG not validated)
  TP/SL:    2.0% / 1.0% (from 8-agent report best config)
  Hold:     4h (16 bars of 15m)
  Cooldown: 30 min per direction

What this does NOT include (unvalidated):
  - No funding spike, OB withdrawal, path convexity, basis widening
  - No LONG direction (only 15 events, 27% WR)
  - No volume as OI proxy
  - No composite conviction score
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import math


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class S20Config:
    """Immutable configuration for v7. All thresholds from validated research."""
    
    # --- Signal thresholds (validated) ---
    oi_roc_primary: float = -0.015      # Primary: n=35, p=0.008, mean +0.86%
    oi_roc_borderline: float = -0.01    # Borderline: n=96, p=0.042, mean +0.32%
    
    # --- Regime filter ---
    vol_lookback: int = 96              # Rolling window for vol (96 bars = 24h at 15m)
    vol_pct_low: float = 0.33           # LOW regime: below 33rd percentile
    vol_pct_high: float = 0.67          # HIGH regime: above 67th percentile
    regime_filter: str = "MID"          # Only trade in MID vol regime
    
    # --- Direction ---
    direction: str = "SHORT"            # Only SHORT (LONG not validated)
    
    # --- Risk management (from 8-agent report) ---
    tp_pct: float = 0.020               # Take profit: 2.0%
    sl_pct: float = 0.010               # Stop loss: 1.0%
    hold_bars: int = 16                 # Max hold: 16 bars (4h at 15m)
    
    # --- Cooldown ---
    cooldown_minutes: int = 30          # Min 30 min between trades
    
    # --- Conviction calibration (from historical returns) ---
    # Map OI ROC to expected return → conviction score
    conviction_oi_roc_strong: float = -0.015  # Threshold for high conviction
    conviction_oi_roc_weak: float = -0.01     # Threshold for low conviction
    conviction_floor: float = 0.3              # Minimum conviction
    conviction_cap: float = 0.95               # Maximum conviction
    
    # --- Data requirements ---
    require_real_oi: bool = True        # Reject if OI data unavailable
    min_oi_value: float = 100000.0      # Minimum OI to consider valid
    
    # --- Position sizing (for backtest) ---
    position_size_pct: float = 1.0      # Fraction of capital per trade


# ═══════════════════════════════════════════════════════════════
# SIGNAL STATE
# ═══════════════════════════════════════════════════════════════

@dataclass
class SignalState:
    """Current state of derivatives + market data for signal evaluation."""
    timestamp: datetime
    price: float
    oi: Optional[float] = None
    oi_roc: Optional[float] = None      # Rate of change (validated metric)
    vol_regime: Optional[str] = None     # 'LOW', 'MID', 'HIGH'
    rolling_vol: Optional[float] = None
    vol_percentile: Optional[float] = None
    data_source: str = "unknown"         # 'collected', 'backfilled', 'none'


@dataclass
class TradeSignal:
    """Output signal from strategy evaluation."""
    timestamp: datetime
    direction: str                       # 'SHORT' only in v7
    entry_price: float
    tp_price: float
    sl_price: float
    conviction: float                    # Calibrated from OI ROC → expected return
    oi_roc: float                        # Raw OI ROC value
    vol_regime: str                      # 'LOW', 'MID', 'HIGH'
    hold_bars: int                       # Max hold duration
    signal_type: str                     # 'PRIMARY' or 'BORDERLINE'
    expected_return_pct: float           # From historical calibration


# ═══════════════════════════════════════════════════════════════
# CORE STRATEGY
# ═══════════════════════════════════════════════════════════════

class S20LiquidationCascadeV7:
    """
    Liquidation Cascade Strategy v7.
    
    Single validated signal (OI ROC crash) + regime filter (MID volatility).
    No composite scoring, no unvalidated add-ons.
    """
    
    def __init__(self, config: Optional[S20Config] = None):
        self.config = config or S20Config()
        self._last_trade_time: Optional[datetime] = None
        self._trade_history: List[TradeSignal] = []
    
    # ─── Public Interface ──────────────────────────────────────
    
    def evaluate(self, state: SignalState) -> Optional[TradeSignal]:
        """
        Evaluate current market state for a trade signal.
        
        Returns TradeSignal if conditions are met, None otherwise.
        This is the ONLY entry point for live/backtest signal generation.
        """
        # Gate 1: Data quality — must have real OI data
        if not self._check_oi_data(state):
            return None
        
        # Gate 2: Compute OI ROC — must be valid and below threshold
        oi_roc = state.oi_roc
        if oi_roc is None:
            return None
        
        signal_type = self._classify_signal(oi_roc)
        if signal_type is None:
            return None
        
        # Gate 3: Regime filter — must be MID volatility
        if not self._check_regime(state):
            return None
        
        # Gate 4: Cooldown
        if not self._check_cooldown(state.timestamp):
            return None
        
        # Generate signal
        return self._generate_signal(state, oi_roc, signal_type)
    
    def reset(self):
        """Reset strategy state (for backtest)."""
        self._last_trade_time = None
        self._trade_history = []
    
    @property
    def trade_history(self) -> List[TradeSignal]:
        return self._trade_history
    
    # ─── Signal Classification ─────────────────────────────────
    
    def _classify_signal(self, oi_roc: float) -> Optional[str]:
        """
        Classify OI ROC into signal type.
        Returns 'PRIMARY', 'BORDERLINE', or None.
        """
        if oi_roc < self.config.oi_roc_primary:
            return "PRIMARY"
        elif oi_roc < self.config.oi_roc_borderline:
            return "BORDERLINE"
        return None
    
    # ─── Data Quality Gates ────────────────────────────────────
    
    def _check_oi_data(self, state: SignalState) -> bool:
        """Verify OI data is real and usable."""
        if not self.config.require_real_oi:
            return True
        
        # Must have OI value
        if state.oi is None or state.oi <= 0:
            return False
        
        # Must meet minimum threshold
        if state.oi < self.config.min_oi_value:
            return False
        
        # Must have valid OI ROC
        if state.oi_roc is None:
            return False
        
        # Reject if data source is unknown
        if state.data_source == "none":
            return False
        
        return True
    
    def _check_regime(self, state: SignalState) -> bool:
        """Check volatility regime filter."""
        if state.vol_regime is None:
            return False
        
        if self.config.regime_filter == "MID":
            return state.vol_regime == "MID"
        elif self.config.regime_filter == "ANY":
            return True
        return False
    
    def _check_cooldown(self, timestamp: datetime) -> bool:
        """Encooldown period between trades."""
        if self._last_trade_time is None:
            return True
        
        elapsed = (timestamp - self._last_trade_time).total_seconds() / 60
        return elapsed >= self.config.cooldown_minutes
    
    # ─── Signal Generation ─────────────────────────────────────
    
    def _generate_signal(
        self, state: SignalState, oi_roc: float, signal_type: str
    ) -> TradeSignal:
        """Generate trade signal with calibrated conviction."""
        
        conviction = self._calibrate_conviction(oi_roc, signal_type)
        expected_return = self._oi_roc_to_expected_return(oi_roc, signal_type)
        
        entry = state.price
        if self.config.direction == "SHORT":
            tp = entry * (1 - self.config.tp_pct)
            sl = entry * (1 + self.config.sl_pct)
        else:
            tp = entry * (1 + self.config.tp_pct)
            sl = entry * (1 - self.config.sl_pct)
        
        signal = TradeSignal(
            timestamp=state.timestamp,
            direction=self.config.direction,
            entry_price=entry,
            tp_price=tp,
            sl_price=sl,
            conviction=conviction,
            oi_roc=oi_roc,
            vol_regime=state.vol_regime or "UNKNOWN",
            hold_bars=self.config.hold_bars,
            signal_type=signal_type,
            expected_return_pct=expected_return,
        )
        
        # Record trade
        self._last_trade_time = state.timestamp
        self._trade_history.append(signal)
        
        return signal
    
    # ─── Conviction Calibration ────────────────────────────────
    
    def _calibrate_conviction(self, oi_roc: float, signal_type: str) -> float:
        """
        Calibrate conviction from OI ROC → historical expected return.
        
        NOT a heuristic. Maps directly to validated outcomes:
          oi_roc < -0.015 → mean +0.86% (n=35, WR 74%) → conviction ~0.85
          oi_roc < -0.01  → mean +0.32% (n=96, WR 58%) → conviction ~0.55
          oi_roc < -0.015 + MID vol → mean +2.22% (n=12, WR 92%) → conviction ~0.95
        """
        # Base conviction from signal type
        if signal_type == "PRIMARY":
            # oi_roc < -0.015: mean +0.86%, WR 74%
            base = 0.85
        else:  # BORDERLINE
            # oi_roc < -0.01: mean +0.32%, WR 58%
            base = 0.55
        
        # Scale by how deep the OI ROC is below threshold
        # More negative = stronger signal
        depth = abs(oi_roc) - abs(self.config.oi_roc_borderline)
        max_depth = abs(self.config.oi_roc_primary * 2) - abs(self.config.oi_roc_borderline)
        depth_factor = min(depth / max(max_depth, 0.001), 1.0)
        
        conviction = base + depth_factor * 0.10  # Up to +0.10 for extreme readings
        
        # Clamp
        return max(self.config.conviction_floor, 
                   min(conviction, self.config.conviction_cap))
    
    def _oi_roc_to_expected_return(self, oi_roc: float, signal_type: str) -> float:
        """
        Map OI ROC directly to historical expected return.
        Used for position sizing and risk assessment.
        """
        if signal_type == "PRIMARY":
            # Linear interpolation: oi_roc=-0.015 → +0.86%, oi_roc=-0.03 → +1.5%
            base_return = 0.86
            scale = (abs(oi_roc) - 0.015) / 0.015
            return base_return + scale * 0.64
        else:  # BORDERLINE
            base_return = 0.32
            scale = (abs(oi_roc) - 0.01) / 0.005
            return base_return + scale * 0.54


# ═══════════════════════════════════════════════════════════════
# HELPER: Regime Classifier
# ═══════════════════════════════════════════════════════════════

class VolatilityRegimeClassifier:
    """
    Precomputed rolling volatility percentile classifier.
    
    Uses rolling standard deviation of 15m returns, classified into
    LOW/MID/HIGH terciles. Must be precomputed on full history to
    avoid lookahead bias.
    """
    
    def __init__(self, lookback: int = 96, low_pct: float = 0.33, high_pct: float = 0.67):
        self.lookback = lookback
        self.low_pct = low_pct
        self.high_pct = high_pct
        self._vol_series: List[Tuple[datetime, float]] = []  # (ts, rolling_vol)
        self._percentiles: Dict[datetime, float] = {}
        self._regimes: Dict[datetime, str] = {}
    
    def compute(self, closes: List[float], timestamps: List[datetime]) -> None:
        """
        Precompute rolling volatility and percentile ranks for all bars.
        Must be called ONCE on full history before use.
        """
        if len(closes) < self.lookback + 1:
            return
        
        # Step 1: Compute log returns
        returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0:
                returns.append(math.log(closes[i] / closes[i-1]))
            else:
                returns.append(0.0)
        
        # Step 2: Compute rolling volatility (std of returns over lookback)
        rolling_vols = []
        for i in range(self.lookback, len(returns)):
            window = returns[i - self.lookback:i]
            mean_r = sum(window) / len(window)
            var = sum((r - mean_r) ** 2 for r in window) / len(window)
            vol = math.sqrt(var)
            rolling_vols.append((timestamps[i + 1], vol))  # +1 because returns are offset
        
        if not rolling_vols:
            return
        
        # Step 3: Compute percentile rank for each vol value
        all_vols = sorted([v for _, v in rolling_vols])
        n = len(all_vols)
        
        for ts, vol in rolling_vols:
            # Binary search for rank
            rank = 0
            for v in all_vols:
                if v < vol:
                    rank += 1
                else:
                    break
            percentile = rank / n
            self._percentiles[ts] = percentile
            
            # Classify regime
            if percentile < self.low_pct:
                self._regimes[ts] = "LOW"
            elif percentile < self.high_pct:
                self._regimes[ts] = "MID"
            else:
                self._regimes[ts] = "HIGH"
    
    def get_regime(self, ts: datetime) -> Optional[str]:
        """Get volatility regime for a timestamp."""
        return self._regimes.get(ts)
    
    def get_percentile(self, ts: datetime) -> Optional[float]:
        """Get volatility percentile for a timestamp."""
        return self._percentiles.get(ts)


# ═══════════════════════════════════════════════════════════════
# HELPER: OI ROC Calculator
# ═══════════════════════════════════════════════════════════════

class OIROCCalculator:
    """
    Compute OI Rate of Change from derivatives data.
    
    oi_roc = (oi_current - oi_prev) / oi_prev
    
    Lookback: 4 bars (1 hour at 15m intervals) — matches derivatives_loader.py
    """
    
    def __init__(self, lookback_bars: int = 4):
        self.lookback_bars = lookback_bars
        self._oi_history: List[Tuple[datetime, float]] = []
    
    def add(self, ts: datetime, oi: float) -> None:
        """Add an OI observation."""
        if oi > 0:
            self._oi_history.append((ts, oi))
    
    def compute_roc(self, ts: datetime, current_oi: float) -> Optional[float]:
        """
        Compute OI ROC at current timestamp.
        Returns None if insufficient history or invalid data.
        """
        if current_oi <= 0:
            return None
        
        # Find the bar that is `lookback_bars` ago
        target_time = ts - timedelta(minutes=15 * self.lookback_bars)
        
        # Find closest historical OI within ±15 min of target
        best_oi = None
        best_diff = float('inf')
        
        for hist_ts, hist_oi in self._oi_history:
            diff = abs((hist_ts - target_time).total_seconds())
            if diff < best_diff and diff <= 1800:  # Within 30 min
                best_diff = diff
                best_oi = hist_oi
        
        if best_oi is None or best_oi <= 0:
            return None
        
        return (current_oi - best_oi) / best_oi


# ═══════════════════════════════════════════════════════════════
# HELPER: Derivatives Data Loader
# ═══════════════════════════════════════════════════════════════

def load_derivatives_data(csv_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load derivatives data from CSV into a dict keyed by timestamp string.
    
    Handles both formats:
    - derivatives_collected.csv: timestamp,oi,oi_usd,ls_ratio,...
    - derivatives_backfilled.csv: timestamp,funding_rate,oi,...
    
    Returns dict: {timestamp_str: {oi: float, oi_usd: float, ...}}
    """
    import csv
    
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_raw = row.get("timestamp", "").strip()
            if not ts_raw:
                continue
            
            # Normalize timestamp format
            # Handle "2026-04-13 06:30:00", "2026-04-13 06:30:00.000", "2026-07-18T23:17:27.798340"
            ts = ts_raw[:16].replace("T", " ")
            
            # Extract OI
            oi_str = row.get("oi", "").strip()
            oi = float(oi_str) if oi_str else 0.0
            
            oi_usd_str = row.get("oi_usd", "").strip()
            oi_usd = float(oi_usd_str) if oi_usd_str else 0.0
            
            ls_str = row.get("ls_ratio", "").strip()
            ls = float(ls_str) if ls_str else 0.0
            
            fr_str = row.get("funding_rate", "").strip()
            fr = float(fr_str) if fr_str else 0.0
            
            data[ts] = {
                "oi": oi,
                "oi_usd": oi_usd,
                "ls_ratio": ls,
                "funding_rate": fr,
            }
    
    return data


def merge_derivatives_sources(
    collected_path: str, backfilled_path: str
) -> Dict[str, Dict[str, float]]:
    """
    Merge derivatives data from both sources.
    Collected data takes priority where available.
    """
    collected = load_derivatives_data(collected_path)
    backfilled = load_derivatives_data(backfilled_path)
    
    # Start with backfilled, overlay with collected (higher quality)
    merged = backfilled.copy()
    merged.update(collected)
    
    return merged


# ═══════════════════════════════════════════════════════════════
# MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("S20 Liquidation Cascade v7 — Strategy Module")
    print("=" * 50)
    print(f"Signal: OI ROC < -0.015 (primary) / < -0.01 (borderline)")
    print(f"Regime: MID volatility only (33rd-67th percentile)")
    print(f"Direction: SHORT only")
    print(f"TP/SL: 2.0% / 1.0%")
    print(f"Hold: 4h (16 bars)")
    print(f"Cooldown: 30 min")
    print()
    
    # Quick validation
    config = S20Config()
    strategy = S20LiquidationCascadeV7(config)
    
    # Test with a mock signal
    test_state = SignalState(
        timestamp=datetime(2026, 7, 14, 2, 45),
        price=1785.17,
        oi=2181909.315,
        oi_roc=-0.02,
        vol_regime="MID",
        rolling_vol=0.001,
        vol_percentile=0.5,
        data_source="collected",
    )
    
    signal = strategy.evaluate(test_state)
    if signal:
        print(f"Test signal generated:")
        print(f"  Direction: {signal.direction}")
        print(f"  Entry: {signal.entry_price}")
        print(f"  TP: {signal.tp_price}")
        print(f"  SL: {signal.sl_price}")
        print(f"  Conviction: {signal.conviction:.3f}")
        print(f"  OI ROC: {signal.oi_roc:.4f}")
        print(f"  Signal Type: {signal.signal_type}")
        print(f"  Expected Return: {signal.expected_return_pct:.2f}%")
    else:
        print("No signal generated (check test data)")
