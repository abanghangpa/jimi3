#!/usr/bin/env python3
"""Backfill derivatives history from Binance API.
Downloads: funding rates, OI history, taker volume ratio.
Outputs: derivatives_history/derivatives_collected.csv (extended)
"""
import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE_DIR, "data", "derivatives_history", "derivatives_backfilled.csv")
SYMBOL = "ETHUSDT"

def get_funding_history(start_ts, end_ts, limit=1000):
    """Get funding rate history from Binance."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_data = []
    current = start_ts
    while current < end_ts:
        params = {
            "symbol": SYMBOL,
            "startTime": int(current),
            "endTime": int(end_ts),
            "limit": limit,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            current = data[-1]["fundingTime"] + 1
            time.sleep(0.2)
        except Exception as e:
            print(f"Funding error: {e}")
            break
    return all_data

def get_oi_history(start_ts, end_ts, period="15m", limit=500):
    """Get OI history from Binance."""
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    all_data = []
    current = start_ts
    while current < end_ts:
        params = {
            "symbol": SYMBOL,
            "period": period,
            "startTime": int(current),
            "endTime": int(end_ts),
            "limit": limit,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            current = data[-1]["timestamp"] + 1
            time.sleep(0.2)
        except Exception as e:
            print(f"OI error: {e}")
            break
    return all_data

def get_taker_history(start_ts, end_ts, period="15m", limit=500):
    """Get taker buy/sell volume from Binance."""
    url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
    all_data = []
    current = start_ts
    while current < end_ts:
        params = {
            "symbol": SYMBOL,
            "period": period,
            "startTime": int(current),
            "endTime": int(end_ts),
            "limit": limit,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            if not data or "data" not in data:
                break
            all_data.extend(data["data"])
            current = int(data["data"][-1]["timestamp"]) + 1
            time.sleep(0.2)
        except Exception as e:
            print(f"Taker error: {e}")
            break
    return all_data

def main():
    # Backfill from 2024-01-01 to now
    start = datetime(2024, 1, 1)
    end = datetime.now()
    start_ts = int(start.timestamp() * 1000)
    end_ts = int(end.timestamp() * 1000)

    print(f"Backfilling {SYMBOL} from {start} to {end}")

    # 1. Funding rates
    print("Downloading funding rates...")
    funding = get_funding_history(start_ts, end_ts)
    print(f"  Got {len(funding)} funding records")

    # 2. OI history
    print("Downloading OI history...")
    oi = get_oi_history(start_ts, end_ts, period="15m")
    print(f"  Got {len(oi)} OI records")

    # 3. Taker ratio
    print("Downloading taker ratio...")
    taker = get_taker_history(start_ts, end_ts, period="15m")
    print(f"  Got {len(taker)} taker records")

    # Build merged dataset
    # Funding -> DataFrame
    if funding:
        fund_df = pd.DataFrame(funding)
        fund_df["fundingTime"] = pd.to_numeric(fund_df["fundingTime"])
        fund_df["fundingRate"] = pd.to_numeric(fund_df["fundingRate"])
        fund_df["timestamp"] = pd.to_datetime(fund_df["fundingTime"], unit="ms")
        fund_df = fund_df[["timestamp", "fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
    else:
        fund_df = pd.DataFrame(columns=["timestamp", "funding_rate"])

    # OI -> DataFrame
    if oi:
        oi_df = pd.DataFrame(oi)
        oi_df["timestamp"] = pd.to_datetime(pd.to_numeric(oi_df["timestamp"]), unit="ms")
        oi_df["sumOpenInterest"] = pd.to_numeric(oi_df["sumOpenInterest"])
        oi_df = oi_df[["timestamp", "sumOpenInterest"]].rename(columns={"sumOpenInterest": "oi"})
    else:
        oi_df = pd.DataFrame(columns=["timestamp", "oi"])

    # Taker -> DataFrame
    if taker:
        taker_df = pd.DataFrame(taker)
        taker_df["timestamp"] = pd.to_datetime(pd.to_numeric(taker_df["timestamp"]), unit="ms")
        taker_df["buySellRatio"] = pd.to_numeric(taker_df["buySellRatio"])
        taker_df = taker_df[["timestamp", "buySellRatio"]].rename(columns={"buySellRatio": "futures_taker_ratio"})
    else:
        taker_df = pd.DataFrame(columns=["timestamp", "futures_taker_ratio"])

    # Merge on timestamp (nearest)
    merged = fund_df.copy()
    if not oi_df.empty:
        merged = pd.merge_asof(merged.sort_values("timestamp"), oi_df.sort_values("timestamp"),
                               on="timestamp", direction="nearest", tolerance=pd.Timedelta(minutes=30))
    if not taker_df.empty:
        merged = pd.merge_asof(merged.sort_values("timestamp"), taker_df.sort_values("timestamp"),
                               on="timestamp", direction="nearest", tolerance=pd.Timedelta(minutes=30))

    # Add placeholder columns
    merged["ls_ratio"] = 2.0  # placeholder
    merged["long_pct"] = 0.67  # placeholder
    merged["short_pct"] = 0.33  # placeholder

    # Save
    merged.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(merged)} records to {OUTPUT}")
    print(f"Date range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")
    print(f"Columns: {list(merged.columns)}")

if __name__ == "__main__":
    main()
