"""M12: Order Book Imbalance (live only)."""

import requests


def fetch_order_book_imbalance(symbol='ETHUSDT', depth=20):
    """Fetch order book and compute imbalance metrics."""
    try:
        r = requests.get(
            'https://api.binance.com/api/v3/depth',
            params={'symbol': symbol, 'limit': depth},
            timeout=5
        )
        r.raise_for_status()
        data = r.json()

        bids = [(float(p), float(q)) for p, q in data['bids']]
        asks = [(float(p), float(q)) for p, q in data['asks']]

        if not bids or not asks:
            return 1.0, {}, 0.5

        mid = (bids[0][0] + asks[0][0]) / 2
        pct_range = 0.01

        bid_vol = sum(q for p, q in bids if p >= mid * (1 - pct_range))
        ask_vol = sum(q for p, q in asks if p <= mid * (1 + pct_range))

        ba_ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0

        avg_bid = bid_vol / len(bids) if bids else 0
        avg_ask = ask_vol / len(asks) if asks else 0

        bid_walls = [(p, q) for p, q in bids if q > avg_bid * 3 and p >= mid * (1 - pct_range)]
        ask_walls = [(p, q) for p, q in asks if q > avg_ask * 3 and p <= mid * (1 + pct_range)]

        wall_info = {
            'bid_walls': len(bid_walls),
            'ask_walls': len(ask_walls),
            'largest_bid': max((q for _, q in bid_walls), default=0),
            'largest_ask': max((q for _, q in ask_walls), default=0),
        }

        return ba_ratio, wall_info, ba_ratio

    except Exception:
        return 1.0, {}, 0.5


def score_m12_orderbook(direction, live=True):
    """Score order book imbalance for trade direction."""
    details = {}

    if not live:
        return 'SKIP', 0.5, {'mode': 'backtest_neutral'}

    ba_ratio, wall_info, imbalance = fetch_order_book_imbalance()
    details['bid_ask_ratio'] = round(ba_ratio, 4)
    details.update(wall_info)

    score = 0.5

    # LONG-only contrarian: mild imbalance = best entry
    # Extreme imbalances already priced in (high conv returns LESS)
    if direction == 'LONG':
        if 1.0 < ba_ratio <= 1.2:
            score = 0.80  # mild bid dominance — best
        elif 1.2 < ba_ratio <= 1.5:
            score = 0.65  # moderate
        elif ba_ratio > 1.5:
            score = 0.50  # extreme — already priced in
        elif ba_ratio < 1.0:
            score = 0.30  # asks dominate — skip
    else:
        score = 0.30  # SHORT disabled
        if ba_ratio > 1.5:
            score = 0.30
        elif ba_ratio > 1.2:
            score = 0.40
        if wall_info.get('ask_walls', 0) > wall_info.get('bid_walls', 0):
            score = min(score + 0.05, 1.0)
        elif wall_info.get('bid_walls', 0) > wall_info.get('ask_walls', 0) + 1:
            score = max(score - 0.05, 0.0)

    score = max(0.0, min(1.0, score))
    status = 'PASS' if score >= 0.50 else 'FAIL'
    details['ob_score'] = round(score, 3)
    return status, score, details
