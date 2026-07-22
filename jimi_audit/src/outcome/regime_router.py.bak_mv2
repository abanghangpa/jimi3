"""
Regime-Strategy Router — config-driven strategy filtering by market regime.

Instead of running all strategies in all regimes and filtering after,
this router prevents incompatible strategies from even executing.
"""
import json
import os
from typing import Dict, List, Optional, Set

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                           'config', 'regime_strategy_matrix.json')

# Default matrix — data-driven, based on verified performance
DEFAULT_MATRIX = {
    "STRONG_DOWN": {
        "allowed": [
            "failed_breakout",      # 77.6% WR — best in downtrends
            "trade_flow",           # 59.2% WR — solid
            "orderbook_imbalance",  # 56.7% WR — solid
            "judas_sweep",          # structural, works in all regimes
            "taker_flow",           # flow-based, regime-agnostic
            "funding_arb",          # contrarian, works in downtrends
            "whale_watch",          # sentiment-based
            "liquidity_grab",       # structural
            "momentum_v2",          # momentum, can catch reversals
        ],
        "blocked": [
            "structural_break",     # 29.2% WR — actively harmful
            "regime_switch",        # 37.1% WR — already disabled
            "mtf_confluence",       # trend-following, poor in strong downtrends
        ],
        "notes": "Strong downtrend: favor flow/structural strategies, block trend-following"
    },
    "DOWN": {
        "allowed": [
            "failed_breakout",
            "trade_flow",
            "orderbook_imbalance",
            "judas_sweep",
            "taker_flow",
            "funding_arb",
            "whale_watch",
            "liquidity_grab",
            "momentum_v2",
            "squeeze_breakout",     # can work in mild downtrends
            "kill_zone",            # time-based, regime-flexible
            "power_of_3",           # structural
            "cascade",              # event-driven
        ],
        "blocked": [
            "structural_break",     # still problematic
            "regime_switch",
        ],
        "notes": "Downtrend: broader strategy set, still block weakest performers"
    },
    "RANGING": {
        "allowed": [
            "failed_breakout",
            "trade_flow",
            "orderbook_imbalance",
            "judas_sweep",
            "taker_flow",
            "funding_arb",
            "whale_watch",
            "liquidity_grab",
            "momentum_v2",
            "squeeze_breakout",
            "kill_zone",
            "power_of_3",
            "cascade",
            "structural_break",     # can work in ranging (no trend to fight)
            "cross_asset",
            "scalp_v2",
            "positioning_fade",
            "liquidation_cascade",
        ],
        "blocked": [
            "regime_switch",
        ],
        "notes": "Ranging: most strategies allowed, block only weakest"
    },
    "UP": {
        "allowed": [
            "failed_breakout",
            "trade_flow",
            "orderbook_imbalance",
            "judas_sweep",
            "taker_flow",
            "funding_arb",
            "whale_watch",
            "liquidity_grab",
            "momentum_v2",
            "squeeze_breakout",
            "kill_zone",
            "power_of_3",
            "cascade",
            "structural_break",
            "cross_asset",
            "scalp_v2",
            "positioning_fade",
            "liquidation_cascade",
            "mtf_confluence",
            "regime_switch",
            "macro_surprise",
            "vol_rotation",
        ],
        "blocked": [],
        "notes": "Uptrend: all strategies allowed"
    },
    "STRONG_UP": {
        "allowed": [
            "failed_breakout",
            "trade_flow",
            "orderbook_imbalance",
            "judas_sweep",
            "taker_flow",
            "funding_arb",
            "whale_watch",
            "liquidity_grab",
            "momentum_v2",
            "squeeze_breakout",
            "kill_zone",
            "power_of_3",
            "cascade",
            "structural_break",
            "cross_asset",
            "scalp_v2",
            "liquidation_cascade",
            "mtf_confluence",
            "regime_switch",
            "macro_surprise",
            "vol_rotation",
        ],
        "blocked": [
            "positioning_fade",     # contrarian, poor in strong uptrends
        ],
        "notes": "Strong uptrend: all except contrarian strategies"
    },
}


class RegimeStrategyRouter:
    """
    Config-driven regime-strategy router.
    
    Checks if a strategy is allowed to run in the current regime.
    Matrix is loaded from config file, with fallback to defaults.
    """

    def __init__(self, config_path: str = None):
        self.config_path = config_path or CONFIG_PATH
        self.matrix = self._load_matrix()

    def _load_matrix(self) -> Dict:
        """Load matrix from config file, fallback to defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Save defaults for future editing
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(DEFAULT_MATRIX, f, indent=2)

        return DEFAULT_MATRIX

    def is_allowed(self, strategy: str, regime: str) -> bool:
        """Check if a strategy is allowed in the given regime."""
        regime_config = self.matrix.get(regime, self.matrix.get('RANGING', {}))

        # If explicit allowed list exists, use it
        if 'allowed' in regime_config:
            return strategy in regime_config['allowed']

        # If explicit blocked list exists, use it
        if 'blocked' in regime_config:
            return strategy not in regime_config['blocked']

        # Default: allow
        return True

    def get_allowed_strategies(self, regime: str) -> List[str]:
        """Get list of allowed strategies for a regime."""
        regime_config = self.matrix.get(regime, self.matrix.get('RANGING', {}))
        return regime_config.get('allowed', [])

    def get_blocked_strategies(self, regime: str) -> List[str]:
        """Get list of blocked strategies for a regime."""
        regime_config = self.matrix.get(regime, self.matrix.get('RANGING', {}))
        return regime_config.get('blocked', [])

    def filter_signals(self, signals: List[Dict], regime: str) -> Dict:
        """
        Filter strategy signals based on regime compatibility.
        
        Returns:
            {
                'allowed': [...],
                'blocked': [...],
                'regime': regime,
            }
        """
        allowed = []
        blocked = []

        for sig in signals:
            strategy = sig.get('strategy', '')
            if self.is_allowed(strategy, regime):
                allowed.append(sig)
            else:
                blocked.append({
                    'strategy': strategy,
                    'direction': sig.get('direction'),
                    'conviction': sig.get('conviction'),
                    'reason': f'blocked in {regime} regime',
                })

        return {
            'allowed': allowed,
            'blocked': blocked,
            'regime': regime,
        }

    def get_regime_notes(self, regime: str) -> str:
        """Get notes for a regime."""
        regime_config = self.matrix.get(regime, {})
        return regime_config.get('notes', '')

    def update_matrix(self, strategy: str, regime: str, action: str):
        """Add or remove a strategy from a regime's allowed/blocked list."""
        if regime not in self.matrix:
            self.matrix[regime] = {'allowed': [], 'blocked': [], 'notes': ''}

        if action == 'allow':
            if strategy not in self.matrix[regime].get('allowed', []):
                self.matrix[regime].setdefault('allowed', []).append(strategy)
            if strategy in self.matrix[regime].get('blocked', []):
                self.matrix[regime]['blocked'].remove(strategy)
        elif action == 'block':
            if strategy not in self.matrix[regime].get('blocked', []):
                self.matrix[regime].setdefault('blocked', []).append(strategy)
            if strategy in self.matrix[regime].get('allowed', []):
                self.matrix[regime]['allowed'].remove(strategy)

        # Save to file
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.matrix, f, indent=2)

    def generate_report(self) -> str:
        """Generate a human-readable router report."""
        lines = []
        lines.append("=" * 80)
        lines.append("  REGIME-STRATEGY ROUTER MATRIX")
        lines.append("=" * 80)

        for regime in sorted(self.matrix.keys()):
            config = self.matrix[regime]
            allowed = config.get('allowed', [])
            blocked = config.get('blocked', [])
            notes = config.get('notes', '')

            lines.append(f"\n  {regime}")
            if notes:
                lines.append(f"    Note: {notes}")
            lines.append(f"    ✅ Allowed ({len(allowed)}): {', '.join(sorted(allowed))}")
            if blocked:
                lines.append(f"    ❌ Blocked ({len(blocked)}): {', '.join(sorted(blocked))}")

        return "\n".join(lines)
