"""
Confirmation Layer — waits for market to confirm signal direction before entry.
Transforms JIMI from indicator-led to market-confirmed.
"""
import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, List

PENDING_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pending_signals.json')

# Strategy-specific hold windows (hours)
# Based on historical optimal windows from data analysis
STRATEGY_HOLD_WINDOWS = {
    'main_pipeline': 4,           # was 2h, 46.4% WR — give more room to develop
    'failed_breakout': 8,         # was 12h, 77.6% WR @ 8h — sweet spot
    'funding_arb': 4,             # was 6h, 51.9% WR @ 4h — tighten
    'orderbook_imbalance': 4,     # was 8h, 56.7% WR @ 2h — 8h too long
    'trade_flow': 4,              # was 24h, 59.2% WR @ 2h — 24h way too long
    'cross_asset': 4,
    'mtf_confluence': 4,
    'structural_break': 8,        # DISABLED (29.2% WR) — kept for pending signals only
    'scalp_v2': 2,
    'momentum_v2': 4,
    'squeeze_breakout': 4,
    'positioning_fade': 2,
    'kill_zone': 4,
    'liquidity_grab': 4,
    'taker_flow': 2,
    'regime_switch': 4,
    'power_of_3': 4,
    'cascade': 4,
    'macro_surprise': 8,
    'whale_watch': 4,
    'vol_rotation': 4,
    'liquidation_cascade': 4,
    'judas_sweep': 4,
}

# Confirmation parameters
CONFIRM_BARS = 3       # Wait 3 bars (45 min) for confirmation
CONFIRM_BARS_1 = 1     # Quick confirm (1 bar = 15 min)


def _load_pending() -> List[Dict]:
    """Load pending signals from disk."""
    if os.path.exists(PENDING_FILE):
        try:
            with open(PENDING_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def _save_pending(signals: List[Dict]):
    """Save pending signals to disk."""
    os.makedirs(os.path.dirname(PENDING_FILE), exist_ok=True)
    with open(PENDING_FILE, 'w') as f:
        json.dump(signals, f, indent=2, default=str)


def add_pending(signal: Dict, source: str = 'main_pipeline', ensemble: Dict = None) -> Dict:
    """
    Add a signal to the pending queue for confirmation.
    Returns the pending entry with confirmation metadata.
    """
    pending = _load_pending()
    
    # Get hold window for this strategy
    hold_window = STRATEGY_HOLD_WINDOWS.get(source, 2)
    
    entry = {
        'signal_id': f"{signal.get('timestamp', '')}_{source}_{signal.get('direction', '')}",
        'timestamp': signal.get('timestamp', ''),
        'price': signal.get('price', 0),
        'direction': signal.get('direction', ''),
        'source': source,
        'ics': signal.get('ics', 0),
        'entry': signal.get('entry', 0),
        'sl': signal.get('sl', 0),
        'tp1': signal.get('tp1', 0),
        'tp2': signal.get('tp2', 0),
        'tp3': signal.get('tp3', 0),
        'sl_pct': signal.get('sl_pct', 0),
        'tp1_pct': signal.get('tp1_pct', 0),
        'hold_window_hours': hold_window,
        'confirm_bars': CONFIRM_BARS,
        'added_at': datetime.now(timezone.utc).isoformat(),
        'status': 'PENDING',
        'bars_waited': 0,
        'confirmation_prices': [],
        'ensemble': ensemble if ensemble else None,
    }
    
    # Avoid duplicates
    existing_ids = {s['signal_id'] for s in pending}
    if entry['signal_id'] not in existing_ids:
        pending.append(entry)
        _save_pending(pending)
    
    return entry


def check_confirmations(current_price: float, current_timestamp: str) -> Dict:
    """
    Check all pending signals against current price.
    Returns dict with confirmed, expired, and still-pending signals.
    """
    pending = _load_pending()
    
    confirmed = []
    expired = []
    still_pending = []
    
    for sig in pending:
        sig['bars_waited'] = sig.get('bars_waited', 0) + 1
        sig['confirmation_prices'].append(current_price)
        
        entry_price = sig.get('entry', sig.get('price', 0))
        direction = sig.get('direction', '')
        
        # Check confirmation: did price move in signal direction?
        if direction == 'LONG':
            confirmed_move = current_price > entry_price
        elif direction == 'SHORT':
            confirmed_move = current_price < entry_price
        else:
            confirmed_move = False
        
        # Check 3-bar average confirmation
        prices = sig['confirmation_prices']
        three_bar_confirmed = False
        if len(prices) >= 3:
            avg3 = sum(prices[-3:]) / 3
            if direction == 'LONG':
                three_bar_confirmed = avg3 > entry_price
            elif direction == 'SHORT':
                three_bar_confirmed = avg3 < entry_price
        
        if three_bar_confirmed:
            # 3-bar confirmation — high confidence
            sig['status'] = 'CONFIRMED_3BAR'
            sig['confirmed_at'] = current_timestamp
            sig['confirmation_price'] = current_price
            sig['confirmation_avg3'] = avg3
            sig['price_vs_entry_pct'] = (current_price - entry_price) / entry_price * 100
            confirmed.append(sig)
        elif sig['bars_waited'] >= CONFIRM_BARS and not confirmed_move:
            # Expired — price didn't confirm within window
            sig['status'] = 'EXPIRED'
            sig['expired_at'] = current_timestamp
            sig['expired_price'] = current_price
            expired.append(sig)
        elif sig['bars_waited'] >= CONFIRM_BARS * 2:
            # Hard expiry — waited too long
            sig['status'] = 'HARD_EXPIRED'
            sig['expired_at'] = current_timestamp
            expired.append(sig)
        else:
            still_pending.append(sig)
    
    # Save remaining pending
    _save_pending(still_pending)
    
    return {
        'confirmed': confirmed,
        'expired': expired,
        'pending': still_pending,
    }


def get_hold_window(source: str) -> int:
    """Get optimal hold window for a strategy."""
    return STRATEGY_HOLD_WINDOWS.get(source, 2)


def format_confirmation_report(result: Dict) -> str:
    """Format confirmation status for report output."""
    lines = []
    
    confirmed = result.get('confirmed', [])
    expired = result.get('expired', [])
    pending = result.get('pending', [])
    
    if confirmed:
        lines.append('### ✅ CONFIRMED SIGNALS')
        for sig in confirmed:
            conf_type = '3-BAR' if 'CONFIRMED_3BAR' in sig.get('status', '') else '1-BAR'
            lines.append(f"  {sig['source']}: {sig['direction']} @ ${sig['price']:.2f}")
            lines.append(f"    Confirmation: {conf_type} | Entry: ${sig.get('entry',0):.2f} | Price moved: {sig.get('price_vs_entry_pct',0):+.2f}%")
            lines.append(f"    Hold window: {sig.get('hold_window_hours',2)}h | SL: ${sig.get('sl',0):.2f} | TP1: ${sig.get('tp1',0):.2f}")
    
    if expired:
        lines.append('### ❌ EXPIRED (unconfirmed)')
        for sig in expired:
            lines.append(f"  {sig['source']}: {sig['direction']} @ ${sig['price']:.2f} — market did not confirm")
    
    if pending:
        lines.append(f'### ⏳ PENDING CONFIRMATION ({len(pending)} signals)')
        for sig in pending:
            lines.append(f"  {sig['source']}: {sig['direction']} @ ${sig['price']:.2f} — {sig.get('bars_waited',0)}/{sig.get('confirm_bars',3)} bars")
    
    return chr(10).join(lines) if lines else ''
