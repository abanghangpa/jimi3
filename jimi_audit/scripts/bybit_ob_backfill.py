#!/usr/bin/env python3
"""
Bybit Historical Orderbook Downloader v3 — Memory-efficient, resumable.
Processes one day at a time, writes to CSV immediately, clears memory.

Usage:
    python3 bybit_ob_backfill.py --resume  # continue from last downloaded date
    python3 bybit_ob_backfill.py --start-date 2026-04-01 --end-date 2026-06-30
"""

import os, sys, json, zipfile, csv, time, argparse, requests, gc
from datetime import datetime, timedelta, timezone

BASE_DIR = "/root/.openclaw/workspace/jimi_audit"
DATA_DIR = os.path.join(BASE_DIR, "data", "ob_history")
DERIVED_CSV = os.path.join(DATA_DIR, "ob_historical.csv")
RAW_DIR = os.path.join(DATA_DIR, "raw_downloads")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

SYMBOL = "ETHPERP"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0",
    "Accept": "*/*",
    "Referer": "https://www.bybit.com/derivatives/en/history-data",
}

DERIVED_COLS = [
    "timestamp", "ts_ms", "bid_total", "ask_total", "ob_ratio",
    "top5_bid_vol", "top5_ask_vol", "top5_ratio",
    "spread", "spread_pct", "best_bid", "best_ask",
    "max_bid_vol", "max_ask_vol", "bid_levels", "ask_levels",
]


def download_file(url, filepath, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120, stream=True)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"  Retry {attempt+1}/{max_retries}: {e}")
            time.sleep(3)
    return False


def get_file_list(start_str, end_str):
    url = (
        "https://www.bybit.com/x-api/quote/public/support/download/list-files"
        f"?bizType=contract&productId=orderbook&symbols={SYMBOL}&interval=daily"
        f"&periods=&startDay={start_str}&endDay={end_str}"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        result = data.get("result", {})
        if isinstance(result, dict):
            return result.get("list", [])
        return []
    except Exception as e:
        print(f"  Error getting file list: {e}")
        return []


def extract_metrics(data):
    try:
        ob = data.get("data", data)
        bids_raw = ob.get("b", ob.get("bids", []))
        asks_raw = ob.get("a", ob.get("asks", []))
        ts = data.get("ts", ob.get("ts", 0))
        if not bids_raw or not asks_raw:
            return None
        bids = [(float(p), float(q)) for p, q in bids_raw if float(q) > 0]
        asks = [(float(p), float(q)) for p, q in asks_raw if float(q) > 0]
        if not bids or not asks:
            return None
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        bid_total = sum(q for _, q in bids)
        ask_total = sum(q for _, q in asks)
        total = bid_total + ask_total
        if total == 0:
            return None
        ob_ratio = (bid_total - ask_total) / total
        top5_bid_vol = sum(q for _, q in bids[:5])
        top5_ask_vol = sum(q for _, q in asks[:5])
        top5_total = top5_bid_vol + top5_ask_vol
        top5_ratio = (top5_bid_vol - top5_ask_vol) / top5_total if top5_total > 0 else 0
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid
        spread_pct = spread / best_bid if best_bid > 0 else 0
        max_bid_vol = max(q for _, q in bids)
        max_ask_vol = max(q for _, q in asks)
        if isinstance(ts, (int, float)):
            ts_ms = int(ts) if ts > 1e12 else int(ts * 1000)
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)
            ts_ms = int(dt.timestamp() * 1000)
        return {
            "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "ts_ms": ts_ms,
            "bid_total": round(bid_total, 4), "ask_total": round(ask_total, 4),
            "ob_ratio": round(ob_ratio, 6),
            "top5_bid_vol": round(top5_bid_vol, 4), "top5_ask_vol": round(top5_ask_vol, 4),
            "top5_ratio": round(top5_ratio, 6),
            "spread": round(spread, 2), "spread_pct": round(spread_pct, 8),
            "best_bid": round(best_bid, 2), "best_ask": round(best_ask, 2),
            "max_bid_vol": round(max_bid_vol, 4), "max_ask_vol": round(max_ask_vol, 4),
            "bid_levels": len(bids), "ask_levels": len(asks),
        }
    except Exception:
        return None


def process_and_write(filepath, csv_path):
    """Process a zip file and write to CSV immediately (no memory accumulation)."""
    count = 0
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DERIVED_COLS)
        if not file_exists:
            writer.writeheader()
        try:
            with zipfile.ZipFile(filepath, "r") as zf:
                for name in zf.namelist():
                    if name.endswith("/"):
                        continue
                    with zf.open(name) as zf_file:
                        for line in zf_file:
                            try:
                                line_str = line.decode("utf-8").strip()
                                if not line_str:
                                    continue
                                data = json.loads(line_str)
                                rec = extract_metrics(data)
                                if rec:
                                    writer.writerow(rec)
                                    count += 1
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue
        except Exception as e:
            print(f"  Error processing: {e}")
    return count


def get_last_date_in_csv(csv_path):
    """Get the last timestamp in the CSV to know where to resume."""
    if not os.path.exists(csv_path):
        return None
    try:
        last_ts = None
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                last_ts = row.get("timestamp", "")
        if last_ts:
            return datetime.fromisoformat(last_ts).date()
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    if args.resume:
        last_date = get_last_date_in_csv(DERIVED_CSV)
        if last_date:
            start = datetime.combine(last_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=timezone.utc)
            print(f"Resuming from {start.strftime('%Y-%m-%d')}")
        else:
            start = datetime.now(timezone.utc) - timedelta(days=args.months * 30)
    elif args.months:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.months * 30)
    else:
        end = datetime.now(timezone.utc) if not args.end_date else datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start = end - timedelta(days=90) if not args.start_date else datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    end = datetime.now(timezone.utc)
    
    print(f"Bybit OB Backfill v3 (memory-efficient)")
    print(f"Symbol: {SYMBOL}")
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"CSV: {DERIVED_CSV}")
    print()
    
    total_records = 0
    total_files = 0
    current = start
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        next_date = current + timedelta(days=1)
        next_str = next_date.strftime("%Y-%m-%d")
        
        files = get_file_list(date_str, date_str)
        if not files:
            current = next_date
            continue
        
        for file_info in files:
            file_url = file_info.get("url", "")
            filename = file_info.get("filename", "")
            file_size = int(file_info.get("size", 0))
            
            if not file_url or not filename:
                continue
            
            filepath = os.path.join(RAW_DIR, filename)
            print(f"{date_str}: {filename} ({file_size/(1024*1024):.1f} MB)...", end=" ", flush=True)
            
            if not download_file(file_url, filepath):
                print("FAILED")
                continue
            
            total_files += 1
            records = process_and_write(filepath, DERIVED_CSV)
            total_records += records
            
            # Cleanup immediately
            try:
                os.remove(filepath)
            except Exception:
                pass
            
            print(f"{records} snapshots")
            
            # Force garbage collection to free memory
            gc.collect()
            
            time.sleep(0.5)
        
        current = next_date
    
    print()
    print(f"=== COMPLETE ===")
    print(f"Files: {total_files}, New snapshots: {total_records}")
    print(f"CSV: {DERIVED_CSV}")
    
    # Final stats
    if os.path.exists(DERIVED_CSV):
        total_lines = sum(1 for _ in open(DERIVED_CSV)) - 1
        size_mb = os.path.getsize(DERIVED_CSV) / (1024 * 1024)
        print(f"Total CSV: {total_lines:,} rows, {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
