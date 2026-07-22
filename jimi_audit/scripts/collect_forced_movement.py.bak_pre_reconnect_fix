#!/usr/bin/env python3
"""
Forced Movement Data Collector
Collects and stores the data needed for the forced-movement strategy:
1. Funding rate history (Binance + Bybit + OKX) → cumulative 72h cost
2. OI history with ROC computation
3. Spot-perp basis (Binance spot vs perp)
4. Liquidation events (Bybit free websocket)

Stores to: data/forced_movement/
Run modes:
    python3 collect_forced_movement.py              # single snapshot
    python3 collect_forced_movement.py --loop 300   # every 5 min
    python3 collect_forced_movement.py --liq-stream  # liquidation websocket
"""

import os, sys, json, time, argparse, asyncio
from datetime import datetime, timezone, timedelta
from collections import deque

import requests
import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "forced_movement")
os.makedirs(DATA_DIR, exist_ok=True)

# ── File paths ──
FUNDING_CSV = os.path.join(DATA_DIR, "funding_history.csv")
OI_CSV = os.path.join(DATA_DIR, "oi_history.csv")
BASIS_CSV = os.path.join(DATA_DIR, "basis_history.csv")
LIQ_LOG = os.path.join(DATA_DIR, "liquidation_events.jsonl")
STATE_FILE = os.path.join(DATA_DIR, "fm_state.json")

# ── Binance Futures API (free, no key needed for public endpoints) ──
BINANCE_FAPI = "https://fapi.binance.com"
BYBIT_REST = "https://api.bybit.com"
OKX_REST = "https://www.okx.com"

# Binance fapi endpoints (corrected paths)
def _binance_get(path, params=None):
    """Binance futures GET with correct /fapi/ prefix."""
    url = f"https://fapi.binance.com{path}"
    try:
        r = requests.get(url, params=params or {}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ============================================================
# 1. FUNDING RATE COLLECTOR
# ============================================================
def fetch_binance_funding(symbol="ETHUSDT", limit=100):
    """Get recent funding rate history from Binance (free, no key)."""
    try:
        r = requests.get(f"{BINANCE_FAPI}/futures/data/fundingRate",
                        params={"symbol": symbol, "limit": limit}, timeout=10)
        r.raise_for_status()
        return [{"exchange": "binance", "rate": float(d["fundingRate"]),
                 "ts": int(d["fundingTime"])} for d in r.json()]
    except Exception as e:
        print(f"  ⚠️ Binance funding failed: {e}")
        return []


def fetch_bybit_funding(symbol="ETHUSDT", limit=200):
    """Get recent funding rate history from Bybit (free, no key)."""
    try:
        r = requests.get(f"{BYBIT_REST}/v5/market/funding/history",
                        params={"category": "linear", "symbol": symbol, "limit": limit}, timeout=10)
        r.raise_for_status()
        data = r.json().get("result", {}).get("list", [])
        return [{"exchange": "bybit", "rate": float(d["fundingRate"]),
                 "ts": int(d["fundingRateTimestamp"])} for d in data]
    except Exception as e:
        print(f"  ⚠️ Bybit funding failed: {e}")
        return []


def fetch_okx_funding(symbol="ETH-USDT-SWAP", limit=100):
    """Get recent funding rate history from OKX (free, no key)."""
    try:
        r = requests.get(f"{OKX_REST}/api/v5/public/funding-history",
                        params={"instId": symbol, "limit": str(limit)}, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        return [{"exchange": "okx", "rate": float(d["fundingRate"]),
                 "ts": int(d["fundingTime"])} for d in data]
    except Exception as e:
        print(f"  ⚠️ OKX funding failed: {e}")
        return []


def compute_cumulative_funding(hours=72):
    """Read funding_history.csv and compute cumulative cost over N hours."""
    if not os.path.exists(FUNDING_CSV):
        return {"cumulative_72h": 0, "count": 0, "avg_rate": 0}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_ms = int(cutoff.timestamp() * 1000)
    total = 0.0
    count = 0

    try:
        with open(FUNDING_CSV) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                try:
                    ts = int(parts[1])
                    rate = float(parts[2])
                    if ts >= cutoff_ms:
                        total += rate
                        count += 1
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass

    avg = total / count if count > 0 else 0
    return {"cumulative_72h": round(total, 6), "count": count, "avg_rate": round(avg, 8)}


def collect_funding():
    """Collect funding rates from all 3 exchanges and append to CSV."""
    ts = datetime.now(timezone.utc)
    ts_ms = int(ts.timestamp() * 1000)
    rows = []

    for fetch_fn, sym in [(fetch_binance_funding, "ETHUSDT"),
                           (fetch_bybit_funding, "ETHUSDT"),
                           (fetch_okx_funding, "ETH-USDT-SWAP")]:
        rates = fetch_fn(sym)
        for r in rates:
            rows.append(f"{r['exchange']},{r['ts']},{r['rate']},{ts_ms}")

    # Append to CSV (deduplicate by exchange+ts)
    existing = set()
    if os.path.exists(FUNDING_CSV):
        with open(FUNDING_CSV) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    existing.add(f"{parts[0]}_{parts[1]}")

    new_count = 0
    with open(FUNDING_CSV, "a") as f:
        for row in rows:
            parts = row.split(",")
            key = f"{parts[0]}_{parts[1]}"
            if key not in existing:
                f.write(row + "\n")
                existing.add(key)
                new_count += 1

    return new_count


# ============================================================
# 2. OI HISTORY COLLECTOR
# ============================================================
def fetch_oi(symbol="ETHUSDT"):
    """Get current OI — try Bybit first, then Binance."""
    # Try Bybit first — ticker endpoint (simpler, always has OI)
    try:
        r = requests.get(f"{BYBIT_REST}/v5/market/tickers",
                        params={"category": "linear", "symbol": symbol}, timeout=10)
        r.raise_for_status()
        data = r.json().get("result", {}).get("list", [])
        if data:
            d = data[0]
            return {"oi": float(d["openInterest"]),
                    "oi_usd": float(d.get("openInterestValue", 0)),
                    "ts": int(time.time() * 1000)}
    except Exception:
        pass
    # Fallback: Binance
    try:
        r = requests.get(f"https://fapi.binance.com/futures/data/openInterestHist",
                        params={"symbol": symbol, "limit": 1}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data:
            d = data[-1]
            return {"oi": float(d["sumOpenInterest"]),
                    "oi_usd": float(d["sumOpenInterestValue"]),
                    "ts": int(d["timestamp"])}
    except Exception as e:
        print(f"  ⚠️ OI fetch failed: {e}")
    return None


def compute_oi_roc(hours=1):
    """Compute OI rate of change over N hours."""
    if not os.path.exists(OI_CSV):
        return {"oi_roc": 0, "current_oi": 0, "prev_oi": 0}

    lines = []
    try:
        with open(OI_CSV) as f:
            lines = f.readlines()
    except Exception:
        pass

    if len(lines) < 2:
        return {"oi_roc": 0, "current_oi": 0, "prev_oi": 0}

    # Latest entry
    latest = lines[-1].strip().split(",")
    current_oi = float(latest[1])
    current_ts = int(latest[0])

    # Find entry closest to N hours ago
    target_ts = current_ts - (hours * 3600 * 1000)
    prev_oi = current_oi
    for line in reversed(lines[:-1]):
        parts = line.strip().split(",")
        if len(parts) >= 2:
            ts = int(parts[0])
            if ts <= target_ts:
                prev_oi = float(parts[1])
                break

    roc = (current_oi - prev_oi) / prev_oi if prev_oi > 0 else 0
    return {"oi_roc": round(roc, 6), "current_oi": current_oi, "prev_oi": prev_oi}


def collect_oi():
    """Collect OI snapshot and append to CSV."""
    data = fetch_oi()
    if not data:
        return 0

    ts = data["ts"]
    row = f"{ts},{data['oi']},{data['oi_usd']}"

    # Check last entry to avoid duplicates
    if os.path.exists(OI_CSV):
        with open(OI_CSV) as f:
            lines = f.readlines()
            if lines:
                last_ts = lines[-1].strip().split(",")[0]
                if last_ts == str(ts):
                    return 0

    with open(OI_CSV, "a") as f:
        f.write(row + "\n")
    return 1


# ============================================================
# 3. SPOT-PERP BASIS COLLECTOR
# ============================================================
def fetch_basis(symbol="ETHUSDT"):
    """Compute spot-perp basis — try Bybit first, then Binance."""
    # Bybit
    try:
        r = requests.get(f"{BYBIT_REST}/v5/market/tickers",
                        params={"category": "linear", "symbol": symbol}, timeout=10)
        r.raise_for_status()
        perp_data = r.json().get("result", {}).get("list", [])
        if perp_data:
            perp_price = float(perp_data[0]["lastPrice"])
        else:
            raise Exception("No perp data")

        # Spot from Bybit
        r2 = requests.get(f"{BYBIT_REST}/v5/market/tickers",
                         params={"category": "spot", "symbol": "ETHUSDT"}, timeout=10)
        r2.raise_for_status()
        spot_data = r2.json().get("result", {}).get("list", [])
        if spot_data:
            spot_price = float(spot_data[0]["lastPrice"])
        else:
            raise Exception("No spot data")

        basis = (perp_price - spot_price) / spot_price
        return {"perp_price": perp_price, "spot_price": spot_price,
                "basis": round(basis, 6), "basis_pct": round(basis * 100, 4)}
    except Exception:
        pass
    # Fallback: Binance
    try:
        r = requests.get(f"https://fapi.binance.com/fapi/v1/ticker/24hr",
                        params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        perp_price = float(r.json()["lastPrice"])
        r2 = requests.get("https://api.binance.com/api/v3/ticker/price",
                         params={"symbol": "ETHUSDT"}, timeout=10)
        r2.raise_for_status()
        spot_price = float(r2.json()["price"])
        basis = (perp_price - spot_price) / spot_price
        return {"perp_price": perp_price, "spot_price": spot_price,
                "basis": round(basis, 6), "basis_pct": round(basis * 100, 4)}
    except Exception as e:
        print(f"  ⚠️ Basis fetch failed: {e}")
        return None


def collect_basis():
    """Collect spot-perp basis and append to CSV."""
    data = fetch_basis()
    if not data:
        return 0

    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    row = f"{ts},{data['perp_price']},{data['spot_price']},{data['basis']}"

    with open(BASIS_CSV, "a") as f:
        f.write(row + "\n")
    return 1


# ============================================================
# 4. LIQUIDATION STREAM (Bybit free websocket)
# ============================================================
async def liq_stream():
    """
    Connect to Bybit's free public liquidation websocket.
    No API key needed. Streams real-time liquidation events.
    """
    url = "wss://stream.bybit.com/v5/public/linear"
    subscribe_msg = {"op": "subscribe", "args": ["liquidation.ETHUSDT"]}

    print(f"[{datetime.now(timezone.utc).isoformat()}] Connecting to Bybit liquidation stream...")

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url, heartbeat=20) as ws:
            await ws.send_json(subscribe_msg)
            print("  ✅ Connected, subscribed to liquidation.ETHUSDT")

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("topic") == "liquidation.ETHUSDT":
                        for event in data.get("data", []):
                            liq = {
                                "ts": int(event.get("T", 0)),
                                "symbol": event.get("s", ""),
                                "side": event.get("S", ""),  # Buy=short liq, Sell=long liq
                                "qty": float(event.get("v", 0)),
                                "price": float(event.get("p", 0)),
                                "exchange": "bybit",
                                "collected_at": datetime.now(timezone.utc).isoformat()
                            }
                            # Append to JSONL
                            with open(LIQ_LOG, "a") as f:
                                f.write(json.dumps(liq) + "\n")

                            side_label = "LONG_LIQUIDATION" if liq["side"] == "Sell" else "SHORT_LIQUIDATION"
                            print(f"  💥 {side_label}: {liq['qty']:.1f} ETH @ ${liq['price']:.2f}")

                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    print(f"  ❌ WebSocket closed/error: {msg}")
                    break


def get_recent_liquidations(minutes=60):
    """Read recent liquidation events from JSONL."""
    if not os.path.exists(LIQ_LOG):
        return {"total_volume": 0, "long_liq_volume": 0, "short_liq_volume": 0,
                "count": 0, "events": []}

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    cutoff_ms = int(cutoff.timestamp() * 1000)
    events = []
    long_vol = 0
    short_vol = 0

    try:
        with open(LIQ_LOG) as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    if e.get("ts", 0) >= cutoff_ms:
                        events.append(e)
                        if e.get("side") == "Sell":
                            long_vol += e.get("qty", 0)
                        else:
                            short_vol += e.get("qty", 0)
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception:
        pass

    return {"total_volume": round(long_vol + short_vol, 2),
            "long_liq_volume": round(long_vol, 2),
            "short_liq_volume": round(short_vol, 2),
            "count": len(events), "events": events[-20:]}  # last 20 for display


# ============================================================
# 5. COMPOSITE: FORCED MOVEMENT SIGNAL
# ============================================================
def forced_movement_check():
    """
    Check all 3 forced-movement conditions.
    Returns dict with signal info or None.
    """
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signals": [],
        "direction": None,
        "conviction": 0,
    }

    # ── Signal 1: OI Divergence ──
    # Price making new high but OI declining → SHORT
    oi_data = compute_oi_roc(hours=1)
    basis_data = fetch_basis()
    funding_data = compute_cumulative_funding(hours=72)

    if basis_data:
        perp_price = basis_data["perp_price"]
    else:
        perp_price = 0

    # Get 20-bar high from price data (use derivatives history)
    price_high_20 = 0
    if os.path.exists(os.path.join(BASE_DIR, "data", "derivatives_history", "derivatives_collected.csv")):
        try:
            import csv
            prices = []
            with open(os.path.join(BASE_DIR, "data", "derivatives_history", "derivatives_collected.csv")) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        prices.append(float(row.get("price", 0)))
                    except (ValueError, KeyError):
                        continue
            if len(prices) >= 20:
                price_high_20 = max(prices[-20:])
        except Exception:
            pass

    # OI divergence: current price near 20-bar high BUT OI declining
    if perp_price > 0 and price_high_20 > 0 and oi_data["oi_roc"] < -0.005:
        if perp_price >= price_high_20 * 0.995:  # within 0.5% of 20-bar high
            result["signals"].append({
                "name": "oi_divergence",
                "direction": "SHORT",
                "detail": f"Price ${perp_price:.2f} near high ${price_high_20:.2f} but OI ROC={oi_data['oi_roc']:.4f}",
                "strength": min(abs(oi_data["oi_roc"]) * 100, 1.0)
            })

    # ── Signal 2: Funding Squeeze ──
    # Cumulative 72h funding > 0.3% AND L/S > 2.0 → longs being squeezed → SHORT
    if funding_data["cumulative_72h"] > 0.003:  # 0.3%
        # Read current L/S from latest derivatives snapshot
        ls_ratio = 0
        try:
            deriv_csv = os.path.join(BASE_DIR, "data", "derivatives_history", "derivatives_collected.csv")
            with open(deriv_csv) as f:
                lines = f.readlines()
                if lines:
                    header = lines[0].strip().split(",")
                    last = lines[-1].strip().split(",")
                    ls_idx = header.index("ls_ratio") if "ls_ratio" in header else -1
                    if ls_idx >= 0:
                        ls_ratio = float(last[ls_idx])
        except Exception:
            pass

        if ls_ratio > 2.0:
            result["signals"].append({
                "name": "funding_squeeze",
                "direction": "SHORT",
                "detail": f"Cumulative 72h FR={funding_data['cumulative_72h']:.4f} L/S={ls_ratio:.2f}",
                "strength": min(funding_data["cumulative_72h"] / 0.01, 1.0)
            })
        elif ls_ratio < 0.5:  # shorts paying longs
            result["signals"].append({
                "name": "funding_squeeze",
                "direction": "LONG",
                "detail": f"Cumulative 72h FR={funding_data['cumulative_72h']:.4f} L/S={ls_ratio:.2f} (shorts paying)",
                "strength": min(abs(funding_data["cumulative_72h"]) / 0.01, 1.0)
            })

    # ── Signal 3: Spot-Perp Basis Convergence ──
    # Strong backwardation (< -0.3%) → perps will bounce up
    if basis_data and basis_data["basis"] < -0.003:
        result["signals"].append({
            "name": "basis_convergence",
            "direction": "LONG",
            "detail": f"Basis={basis_data['basis_pct']:.3f}% (backwardation)",
            "strength": min(abs(basis_data["basis"]) / 0.01, 1.0)
        })
    # Strong contango (> 0.3%) → perps will drop
    elif basis_data and basis_data["basis"] > 0.003:
        result["signals"].append({
            "name": "basis_convergence",
            "direction": "SHORT",
            "detail": f"Basis={basis_data['basis_pct']:.3f}% (contango)",
            "strength": min(basis_data["basis"] / 0.01, 1.0)
        })

    # ── Liquidation cascade (if streaming) ──
    liq = get_recent_liquidations(minutes=15)
    if liq["count"] > 5:
        # More long liqs = bearish cascade in progress
        if liq["long_liq_volume"] > liq["short_liq_volume"] * 2:
            result["signals"].append({
                "name": "liquidation_cascade",
                "direction": "SHORT",
                "detail": f"Long liq cascade: {liq['long_liq_volume']:.1f} ETH liquidated (15m)",
                "strength": min(liq["long_liq_volume"] / 100, 1.0)
            })
        elif liq["short_liq_volume"] > liq["long_liq_volume"] * 2:
            result["signals"].append({
                "name": "liquidation_cascade",
                "direction": "LONG",
                "detail": f"Short liq cascade: {liq['short_liq_volume']:.1f} ETH liquidated (15m)",
                "strength": min(liq["short_liq_volume"] / 100, 1.0)
            })

    # ── Composite: need at least 1 signal ──
    if result["signals"]:
        # Majority vote on direction
        long_votes = sum(1 for s in result["signals"] if s["direction"] == "LONG")
        short_votes = sum(1 for s in result["signals"] if s["direction"] == "SHORT")

        if long_votes > short_votes:
            result["direction"] = "LONG"
        elif short_votes > long_votes:
            result["direction"] = "SHORT"
        else:
            # Tie: use strongest signal
            strongest = max(result["signals"], key=lambda x: x["strength"])
            result["direction"] = strongest["direction"]

        # Conviction: average strength * bonus for multiple signals
        avg_strength = sum(s["strength"] for s in result["signals"]) / len(result["signals"])
        multi_bonus = min((len(result["signals"]) - 1) * 0.15, 0.30)
        result["conviction"] = min(avg_strength * 0.7 + multi_bonus + 0.2, 0.90)

    return result


# ============================================================
# MAIN
# ============================================================
def single_snapshot():
    """Collect all data sources in one pass."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{ts}] Forced Movement Data Collection")

    # 1. Funding
    new = collect_funding()
    funding = compute_cumulative_funding(72)
    print(f"  Funding: {new} new rates | 72h cumulative={funding['cumulative_72h']:.6f} ({funding['count']} samples)")

    # 2. OI
    new = collect_oi()
    oi = compute_oi_roc(1)
    print(f"  OI: {new} new | ROC(1h)={oi['oi_roc']:.4f} | Current={oi['current_oi']:.0f}")

    # 3. Basis
    new = collect_basis()
    basis = fetch_basis()
    if basis:
        print(f"  Basis: {basis['basis_pct']:.3f}% (perp=${basis['perp_price']:.2f} spot=${basis['spot_price']:.2f})")

    # 4. Check signals
    signal = forced_movement_check()
    if signal["signals"]:
        print(f"\n  🚨 FORCED MOVEMENT SIGNAL: {signal['direction']} (conviction={signal['conviction']:.2f})")
        for s in signal["signals"]:
            print(f"    → {s['name']}: {s['direction']} | {s['detail']}")
    else:
        print(f"\n  ⏸️  No forced movement signals")

    # Save state
    state = {
        "last_collection": ts,
        "funding_72h": funding,
        "oi_roc_1h": oi,
        "basis": basis,
        "signal": signal,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forced Movement Data Collector")
    parser.add_argument("--loop", type=int, help="Loop every N seconds")
    parser.add_argument("--liq-stream", action="store_true", help="Run liquidation websocket stream")
    args = parser.parse_args()

    if args.liq_stream:
        print("Starting Bybit liquidation websocket stream...")
        asyncio.run(liq_stream())
    elif args.loop:
        print(f"Collecting every {args.loop}s. Ctrl+C to stop.")
        while True:
            try:
                single_snapshot()
                time.sleep(args.loop)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        single_snapshot()
