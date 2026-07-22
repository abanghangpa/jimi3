#!/usr/bin/env python3
"""
Whale Alert Integration — Direct On-Chain Measurement
Per "Buy the Paint" principle: measure whale activity directly, don't infer from derivatives.

Data sources:
1. Etherscan API (free, 5/sec) — whale wallet tracking + large tx monitoring
2. Whale Alert WebSocket (free trial / $29.95/mo) — real-time large transfers
3. Fallback: Binance large trade stream (free) — whale trades on exchange

This module:
- Monitors whale wallets for large ETH transfers
- Detects accumulation (exchange outflows) vs distribution (exchange inflows)
- Produces signals for the trading executor

Usage:
    python3 whale_tracker.py [--mode collect|signal|daemon] [--threshold 100]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import requests

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "whale")
STATE_FILE = os.path.join(DATA_DIR, "whale_state.json")
SIGNALS_FILE = os.path.join(DATA_DIR, "whale_signals.jsonl")
CONFIG_FILE = os.path.join(DATA_DIR, "whale_config.json")

# Known whale wallets (ETH) — top holders, exchanges, funds
# These are well-known addresses, NOT private info
WHALE_WALLETS = {
    # Major exchanges (cold wallets)
    "0x28C6c06298d514Db089934071355E5743bf21d60": "Binance_Hot",
    "0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549": "Binance_Cold",
    "0xDFd5293D8e347dFe59E90eFd55b2956a1343963d": "Binance_Cold2",
    "0x56Eddb7aa87536c09CCc2793473599fD21A8b17F": "Coinbase_Hot",
    "0xA7efAe728D2936e78BDA97dc267687568dD593f3": "Coinbase_Cold",
    "0x71660c4005BA85c37ccec55d0C4493E66Fe775d3": "Coinbase_Cold2",
    "0x1151314c646Ce4E0eFD76d1aF4760aE66a9Fe30F": "Bitfinex_Hot",
    "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD0E": "Bitfinex_Cold",
    "0x267be1C1D684F78cb4F6a176C4911b741E4Ffdc0": "Kraken_Hot",
    "0xFA2C9e01661a86Ca22CE7F238e8bA77C3fBB93E4": "Kraken_Cold",
    # Funds / whales
    "0x1B3cB81E51011b549d78bf720b0d924ac763A7C2": "Grayscale_ETHE",
    "0x5f65f7b609678448494De4C87521CdF6cEf1e932": "Jump_Trading",
    "0x176F3DAb24a159341c0509bB36B833E7fdd0a132": "Wintermute",
}

# Exchange addresses for inflow/outflow detection
EXCHANGE_ADDRESSES = set()
for addr, label in WHALE_WALLETS.items():
    if "Exchange" in label or "Hot" in label or "Binance" in label or "Coinbase" in label or "Bitfinex" in label or "Kraken" in label:
        EXCHANGE_ADDRESSES.add(addr.lower())


def load_config():
    """Load or create whale tracker config."""
    default = {
        "etherscan_api_key": os.environ.get("ETHERSCAN_API_KEY", ""),
        "whale_alert_api_key": os.environ.get("WHALE_ALERT_API_KEY", ""),
        "min_value_eth": 100,  # minimum ETH transfer to track
        "min_value_usd": 250000,  # minimum USD value
        "check_interval_sec": 300,  # 5 minutes
        "accumulation_threshold": 500,  # ETH net outflow from exchange = bullish
        "distribution_threshold": 500,  # ETH net inflow to exchange = bearish
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            saved = json.load(f)
            default.update(saved)
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(default, f, indent=2)
    return default


def load_state():
    """Load whale tracker state."""
    default = {
        "last_check": None,
        "total_transfers": 0,
        "net_exchange_flow_24h": 0,  # positive = inflow to exchange (bearish)
        "accumulation_score": 0,  # -1.0 to 1.0
        "recent_transfers": [],
        "whale_positions": {},  # wallet -> last known balance change
    }
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            saved = json.load(f)
            default.update(saved)
    return default


def save_state(state):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def fetch_etherscan_large_txs(api_key, min_value_eth=100, pages=2):
    """Fetch recent large ETH transactions from Etherscan."""
    if not api_key:
        print("WARN: No Etherscan API key. Set ETHERSCAN_API_KEY env var or edit whale_config.json")
        return []

    transfers = []
    for page in range(1, pages + 1):
        url = "https://api.etherscan.io/v2/api"
        params = {
            "chainid": 1,
            "module": "account",
            "action": "txlist",
            "address": "0x0000000000000000000000000000000000000000",  # placeholder
            "startblock": 0,
            "endblock": 99999999,
            "page": page,
            "offset": 100,
            "sort": "desc",
            "apikey": api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if data.get("status") == "1":
                for tx in data.get("result", []):
                    value_eth = int(tx.get("value", 0)) / 1e18
                    if value_eth >= min_value_eth:
                        transfers.append({
                            "hash": tx.get("hash"),
                            "from": tx.get("from", "").lower(),
                            "to": tx.get("to", "").lower(),
                            "value_eth": value_eth,
                            "timestamp": int(tx.get("timeStamp", 0)),
                            "block": int(tx.get("blockNumber", 0)),
                        })
        except Exception as e:
            print(f"ERROR fetching Etherscan: {e}")
            break

        time.sleep(0.25)  # rate limit

    return transfers


def fetch_whale_wallet_transfers(api_key, wallets=None, min_value_eth=100):
    """Fetch recent transfers for known whale wallets."""
    if not api_key:
        return []

    if wallets is None:
        wallets = WHALE_WALLETS

    transfers = []
    for addr, label in wallets.items():
        url = "https://api.etherscan.io/v2/api"
        params = {
            "chainid": 1,
            "module": "account",
            "action": "txlist",
            "address": addr,
            "startblock": 0,
            "endblock": 99999999,
            "page": 1,
            "offset": 50,
            "sort": "desc",
            "apikey": api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if data.get("status") == "1":
                for tx in data.get("result", []):
                    value_eth = int(tx.get("value", 0)) / 1e18
                    if value_eth >= min_value_eth:
                        transfers.append({
                            "hash": tx.get("hash"),
                            "from": tx.get("from", "").lower(),
                            "to": tx.get("to", "").lower(),
                            "value_eth": value_eth,
                            "timestamp": int(tx.get("timeStamp", 0)),
                            "block": int(tx.get("blockNumber", 0)),
                            "wallet_label": label,
                            "wallet_address": addr.lower(),
                        })
        except Exception as e:
            print(f"ERROR fetching {label}: {e}")

        time.sleep(0.25)  # rate limit

    return transfers


def analyze_exchange_flow(transfers):
    """Analyze exchange inflows/outflows from transfer data."""
    inflow_eth = 0  # TO exchange = bearish
    outflow_eth = 0  # FROM exchange = bullish
    details = []

    for tx in transfers:
        to_addr = tx.get("to", "").lower()
        from_addr = tx.get("from", "").lower()
        value = tx.get("value_eth", 0)

        to_exchange = to_addr in EXCHANGE_ADDRESSES
        from_exchange = from_addr in EXCHANGE_ADDRESSES

        if to_exchange and not from_exchange:
            inflow_eth += value
            details.append({"type": "INFLOW", "eth": value, "label": tx.get("wallet_label", "")})
        elif from_exchange and not to_exchange:
            outflow_eth += value
            details.append({"type": "OUTFLOW", "eth": value, "label": tx.get("wallet_label", "")})

    net_flow = inflow_eth - outflow_eth  # positive = bearish (more going to exchange)
    return {
        "inflow_eth": round(inflow_eth, 2),
        "outflow_eth": round(outflow_eth, 2),
        "net_flow_eth": round(net_flow, 2),
        "details": details,
    }


def compute_whale_signal(state, config):
    """
    Convert whale data into a trading signal.

    Signal logic:
    - Strong accumulation (large exchange outflows) -> LONG signal
    - Strong distribution (large exchange inflows) -> SHORT signal
    - Neutral -> no signal

    Returns: dict with direction, conviction, reason
    """
    net_flow = state.get("net_exchange_flow_24h", 0)
    accum_threshold = config.get("accumulation_threshold", 500)
    dist_threshold = config.get("distribution_threshold", 500)

    # Compute conviction based on magnitude
    if net_flow < -accum_threshold:
        # Accumulation: exchange outflows exceed threshold
        magnitude = min(abs(net_flow) / (accum_threshold * 3), 1.0)
        conviction = 0.3 + magnitude * 0.5  # 0.3 to 0.8
        return {
            "direction": "LONG",
            "conviction": round(conviction, 2),
            "reason": f"Whale accumulation: {abs(net_flow):.0f} ETH net outflow from exchanges",
            "net_flow": net_flow,
            "signal_type": "whale_accumulation",
        }
    elif net_flow > dist_threshold:
        # Distribution: exchange inflows exceed threshold
        magnitude = min(abs(net_flow) / (dist_threshold * 3), 1.0)
        conviction = 0.3 + magnitude * 0.5
        return {
            "direction": "SHORT",
            "conviction": round(conviction, 2),
            "reason": f"Whale distribution: {abs(net_flow):.0f} ETH net inflow to exchanges",
            "net_flow": net_flow,
            "signal_type": "whale_distribution",
        }
    else:
        return {
            "direction": None,
            "conviction": 0,
            "reason": f"Neutral: net flow {net_flow:.0f} ETH (thresholds: +-{accum_threshold})",
            "net_flow": net_flow,
            "signal_type": "neutral",
        }


def collect_whale_data(config):
    """Collect whale data from all sources."""
    api_key = config.get("etherscan_api_key", "")
    min_eth = config.get("min_value_eth", 100)

    print(f"Collecting whale data (min {min_eth} ETH)...")

    # Fetch whale wallet transfers
    transfers = fetch_whale_wallet_transfers(api_key, min_value_eth=min_eth)
    print(f"  Whale wallet transfers: {len(transfers)}")

    # Analyze exchange flow
    flow = analyze_exchange_flow(transfers)
    print(f"  Exchange inflow: {flow['inflow_eth']:.2f} ETH")
    print(f"  Exchange outflow: {flow['outflow_eth']:.2f} ETH")
    print(f"  Net flow: {flow['net_flow_eth']:.2f} ETH")

    return transfers, flow


def generate_signal(state, config):
    """Generate a whale signal from current state."""
    signal = compute_whale_signal(state, config)

    if signal["direction"]:
        print(f"\n  WHALE SIGNAL: {signal['direction']}")
        print(f"  Conviction: {signal['conviction']}")
        print(f"  Reason: {signal['reason']}")

        # Log signal
        signal_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": "whale_tracker",
            "direction": signal["direction"],
            "conviction": signal["conviction"],
            "reason": signal["reason"],
            "net_flow_eth": signal["net_flow"],
            "signal_type": signal["signal_type"],
            "price": None,  # to be filled by executor
        }

        os.makedirs(DATA_DIR, exist_ok=True)
        with open(SIGNALS_FILE, "a") as f:
            f.write(json.dumps(signal_entry, default=str) + "\n")

        print(f"  Signal logged: {SIGNALS_FILE}")
    else:
        print(f"\n  No signal: {signal['reason']}")

    return signal


def run_daemon(config):
    """Run whale tracker as a daemon, collecting data periodically."""
    interval = config.get("check_interval_sec", 300)
    print(f"Whale tracker daemon starting (interval: {interval}s)")
    print(f"Min value: {config.get('min_value_eth', 100)} ETH")
    print()

    while True:
        try:
            state = load_state()
            transfers, flow = collect_whale_data(config)

            # Update state
            state["last_check"] = datetime.now(timezone.utc).isoformat()
            state["total_transfers"] += len(transfers)
            state["net_exchange_flow_24h"] = flow["net_flow_eth"]
            state["recent_transfers"] = transfers[:20]  # keep last 20

            # Compute accumulation score (-1to 1)
            threshold = config.get("accumulation_threshold", 500)
            state["accumulation_score"] = max(-1, min(1, -flow["net_flow_eth"] / (threshold * 3)))

            save_state(state)

            # Generate signal
            generate_signal(state, config)

            print(f"\n  Next check in {interval}s...")
            time.sleep(interval)

        except KeyboardInterrupt:
            print("\nDaemon stopped.")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="Whale tracker — direct on-chain measurement")
    parser.add_argument("--mode", choices=["collect", "signal", "daemon"], default="collect")
    parser.add_argument("--threshold", type=int, default=100, help="Min ETH transfer size")
    args = parser.parse_args()

    config = load_config()
    if args.threshold:
        config["min_value_eth"] = args.threshold

    if args.mode == "collect":
        state = load_state()
        transfers, flow = collect_whale_data(config)
        state["last_check"] = datetime.now(timezone.utc).isoformat()
        state["net_exchange_flow_24h"] = flow["net_flow_eth"]
        state["recent_transfers"] = transfers[:20]
        threshold = config.get("accumulation_threshold", 500)
        state["accumulation_score"] = max(-1, min(1, -flow["net_flow_eth"] / (threshold * 3)))
        save_state(state)
        print(f"\nState saved: {STATE_FILE}")

    elif args.mode == "signal":
        state = load_state()
        generate_signal(state, config)

    elif args.mode == "daemon":
        run_daemon(config)


if __name__ == "__main__":
    main()
