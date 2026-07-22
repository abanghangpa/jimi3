#!/usr/bin/env python3
"""
ETH Data Pipeline — Incremental candle collector for backtesting.

Fetches 15m candles from Binance, paginates beyond the 1000 limit,
stores locally as CSV for multi-year backtesting.

Usage:
    python3 data_pipeline.py                    # Fetch all available history
    python3 data_pipeline.py --update           # Fetch only new candles since last run
    python3 data_pipeline.py --symbol BTCUSDT   # Collect different pair
    python3 data_pipeline.py --interval 1h      # Different timeframe

Storage: data/history/{symbol}_{interval}.csv
"""

import os, sys, json, time, csv, argparse
from datetime import datetime, timezone, timedelta
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "history")
STATE_FILE = os.path.join(DATA_DIR, "collection_state.json")

BINANCE_API = "https://api.binance.com/api/v3/klines"
BATCH_SIZE = 1000  # Binance max per request


def load_state():
    """Load collection state (last timestamp per symbol/interval)."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_csv_path(symbol, interval):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")


def fetch_batch(symbol, interval, start_ms, end_ms=None, limit=BATCH_SIZE):
    """Fetch one batch of candles from Binance."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "limit": limit,
    }
    if end_ms:
        params["endTime"] = end_ms
    
    r = requests.get(BINANCE_API, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def kline_to_row(k):
    """Convert Binance kline to CSV row."""
    return {
        "ts": int(k[0]),
        "open": float(k[1]),
        "high": float(k[2]),
        "low": float(k[3]),
        "close": float(k[4]),
        "volume": float(k[5]),
        "close_ts": int(k[6]),
        "quote_volume": float(k[7]),
        "trades": int(k[8]),
        "taker_buy_vol": float(k[9]),
        "taker_buy_quote_vol": float(k[10]),
    }


def get_existing_timestamps(csv_path):
    """Load existing timestamps from CSV to avoid duplicates."""
    if not os.path.exists(csv_path):
        return set()
    timestamps = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timestamps.add(int(row["ts"]))
            except (ValueError, KeyError):
                pass
    return timestamps


def append_to_csv(csv_path, rows):
    """Append rows to CSV. Create file with headers if needed."""
    if not rows:
        return 0
    
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    
    with open(csv_path, "a", newline="") as f:
        fieldnames = ["ts", "open", "high", "low", "close", "volume",
                      "close_ts", "quote_volume", "trades", "taker_buy_vol", "taker_buy_quote_vol"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    
    return len(rows)


def collect_history(symbol="ETHUSDT", interval="15m", start_date="2024-01-01"):
    """
    Collect full history from start_date to now.
    Paginates through Binance API to get beyond 1000-candle limit.
    """
    csv_path = get_csv_path(symbol, interval)
    existing_ts = get_existing_timestamps(csv_path)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    # Interval to milliseconds
    interval_ms = {
        "1m": 60000, "3m": 180000, "5m": 300000, "15m": 900000,
        "30m": 1800000, "1h": 3600000, "2h": 7200000, "4h": 14400000,
        "6h": 21600000, "8h": 28800000, "12h": 43200000, "1d": 86400000,
    }.get(interval, 900000)
    
    total_fetched = 0
    total_skipped = 0
    current_ms = start_ms
    batch_num = 0
    
    print(f"📊 Collecting {symbol} {interval} from {start_date}")
    print(f"   Existing records: {len(existing_ts)}")
    print(f"   Target: {start_date} → now")
    
    while current_ms < now_ms:
        batch_num += 1
        try:
            klines = fetch_batch(symbol, interval, current_ms)
        except Exception as e:
            print(f"   ⚠️ Batch {batch_num} error: {e}")
            time.sleep(2)
            continue
        
        if not klines:
            break
        
        # Convert and filter duplicates
        rows = []
        for k in klines:
            row = kline_to_row(k)
            if row["ts"] not in existing_ts:
                rows.append(row)
                existing_ts.add(row["ts"])
            else:
                total_skipped += 1
        
        # Append to CSV
        added = append_to_csv(csv_path, rows)
        total_fetched += added
        
        # Move to next batch
        last_ts = int(klines[-1][0])
        current_ms = last_ts + interval_ms
        
        # Progress
        dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        print(f"   Batch {batch_num}: +{added} records → {dt.strftime('%Y-%m-%d %H:%M')} | Total: {total_fetched}")
        
        # Rate limit (Binance: 1200 requests/min)
        time.sleep(0.1)
    
    # Update state
    state = load_state()
    key = f"{symbol}_{interval}"
    state[key] = {
        "last_update": datetime.now(timezone.utc).isoformat(),
        "last_ts": current_ms,
        "total_records": len(existing_ts),
        "start_date": start_date,
    }
    save_state(state)
    
    print(f"\n✅ Done!")
    print(f"   Fetched: {total_fetched} new records")
    print(f"   Skipped: {total_skipped} duplicates")
    print(f"   Total in CSV: {len(existing_ts)}")
    print(f"   File: {csv_path}")
    
    return total_fetched


def update(symbol="ETHUSDT", interval="15m"):
    """Fetch only new candles since last run."""
    state = load_state()
    key = f"{symbol}_{interval}"
    
    csv_path = get_csv_path(symbol, interval)
    existing_ts = get_existing_timestamps(csv_path)
    
    if key in state:
        last_ts = state[key].get("last_ts", 0)
        # Start from 1 hour before last timestamp to catch any gaps
        start_ms = last_ts - 3600000
    elif existing_ts:
        start_ms = max(existing_ts) - 3600000
    else:
        # No previous data — fetch last 30 days
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)
    
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    interval_ms = {
        "1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000,
    }.get(interval, 900000)
    
    total_fetched = 0
    current_ms = start_ms
    batch_num = 0
    
    print(f"📊 Updating {symbol} {interval}")
    print(f"   From: {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    
    while current_ms < now_ms:
        batch_num += 1
        try:
            klines = fetch_batch(symbol, interval, current_ms)
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            time.sleep(2)
            continue
        
        if not klines:
            break
        
        rows = []
        for k in klines:
            row = kline_to_row(k)
            if row["ts"] not in existing_ts:
                rows.append(row)
                existing_ts.add(row["ts"])
        
        added = append_to_csv(csv_path, rows)
        total_fetched += added
        
        last_ts = int(klines[-1][0])
        current_ms = last_ts + interval_ms
        
        if added > 0:
            dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            print(f"   +{added} → {dt.strftime('%Y-%m-%d %H:%M')}")
        
        time.sleep(0.1)
    
    # Update state
    state[key] = {
        "last_update": datetime.now(timezone.utc).isoformat(),
        "last_ts": current_ms,
        "total_records": len(existing_ts),
    }
    save_state(state)
    
    print(f"\n✅ Updated: +{total_fetched} new records, {len(existing_ts)} total")


def main():
    parser = argparse.ArgumentParser(description="ETH Data Pipeline")
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading pair")
    parser.add_argument("--interval", default="15m", help="Candle interval")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--update", action="store_true", help="Only fetch new candles")
    parser.add_argument("--status", action="store_true", help="Show collection status")
    args = parser.parse_args()
    
    if args.status:
        state = load_state()
        if not state:
            print("No data collected yet.")
            return
        for key, info in state.items():
            print(f"\n📊 {key}:")
            print(f"   Last update: {info.get('last_update', '?')}")
            print(f"   Total records: {info.get('total_records', '?')}")
            csv_path = get_csv_path(*key.rsplit('_', 1))
            if os.path.exists(csv_path):
                size = os.path.getsize(csv_path)
                print(f"   File size: {size / 1024 / 1024:.1f} MB")
        return
    
    if args.update:
        update(args.symbol, args.interval)
    else:
        collect_history(args.symbol, args.interval, args.start)


if __name__ == "__main__":
    main()
