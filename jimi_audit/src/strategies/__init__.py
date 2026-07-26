"""JIMI Multi-Strategy System — optimized strategies with targeted fixes."""
from .base import BaseStrategy, SignalResult
from .runner import StrategyRunner
from .s01_failed_breakout import FailedBreakoutStrategy
from .s02_squeeze_breakout import SqueezeBreakoutStrategy
from .s03_cascade import CascadeStrategy
from .s04_positioning_fade import PositioningFadeStrategy
from .s05_kill_zone import KillZoneStrategy
from .s06_liquidity_grab import LiquidityGrabStrategy
from .s07_taker_flow import TakerFlowStrategy
from .s08_regime_switch import RegimeSwitchStrategy
from .s09_power_of_3 import PowerOf3Strategy
from .s10_structural_break import StructuralBreakStrategy
from .s11_cross_asset import CrossAssetStrategy
from .s12_macro_surprise import MacroSurpriseStrategy
from .s13_funding_arb import FundingArbStrategy
from .s14_whale_watch import WhaleWatchStrategy
from .s15_vol_rotation import VolRotationStrategy
from .s16_mtf_confluence import MTFConfluenceStrategy
from .s17_scalp_v2 import ScalpV2Strategy
from .s18_momentum_v3 import MomentumV3Strategy
from .s19_orderbook_imbalance import OrderBookImbalanceStrategy
from .s20_liquidation_cascade import LiquidationMeanReversionStrategy as LiquidationCascadeStrategy
from .s21_trade_flow import TradeFlowStrategy
from .s22_judas_sweep import JudasSweepStrategy
from .s24_forced_movement import ForcedMovementStrategy
from .s25_funding_squeeze import FundingSqueezeStrategy
from .s23_bb_mom6 import BBMom6Strategy

# DISABLED strategies:
# - RegimeSwitchStrategy: 33.3% WR, LONG at tops
# - ScalpV2Strategy: 25% WR, ALL LONG in downtrend
# - MTFConfluenceStrategy: too few signals

ALL_STRATEGIES = [
    BBMom6Strategy,
    TradeFlowStrategy, CrossAssetStrategy, OrderBookImbalanceStrategy,
    SqueezeBreakoutStrategy, CascadeStrategy, PositioningFadeStrategy,
    KillZoneStrategy, LiquidityGrabStrategy, TakerFlowStrategy,
    PowerOf3Strategy, StructuralBreakStrategy, MacroSurpriseStrategy,
    FundingArbStrategy, WhaleWatchStrategy, VolRotationStrategy,
    MomentumV3Strategy, LiquidationCascadeStrategy,
    FailedBreakoutStrategy, JudasSweepStrategy,
]

def create_runner(config=None) -> StrategyRunner:
    runner = StrategyRunner(config=config)
    for StratClass in ALL_STRATEGIES:
        runner.register(StratClass(config=config))
    return runner
