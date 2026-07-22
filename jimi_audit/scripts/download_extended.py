#!/usr/bin/env python3
"""
Download extended ETH/USDT 15m data from Binance.
Merge with existing data to create a larger dataset for backtesting.

Binance API: 1000 bars per request, no auth needed for public data.
15m bars: 1000 bars = ~10.4 days
2 years = ~73,000 bars = ~74 requests
"""
import os, sys, time, json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

def download_binance_klines(symbol, interval, start_ms, end_ms, limit=1000):
    """Download klines from Binance API."""
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    current_start = start_ms
    
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }
        
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            print("  Rate limited, waiting 60s...")
            time.sleep(60)
            continue
        
        data = resp.json()
        if not data:
            break
        
        all_klines.extend(data)
        current_start = data[-1][0] + 1  # next ms after last kline
        
        # Rate limit: 1200 requests/min, we're well under
        time.sleep(0.1)
        
        if len(all_klines) % 10000 == 0:
            print(f"  Downloaded {len(all_klines)} bars...")
    
    return all_klines

def klines_to_df(klines):
    """Convert Binance klines to DataFrame matching existing format."""
    df = pd.DataFrame(klines, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])
    
    # Convert types
    for col in ["Open", "High", "Low", "Close", "Volume", "Quote asset volume",
                "Taker buy base asset volume", "Taker buy quote asset volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Number of trades"] = pd.to_numeric(df["Number of trades"], errors="coerce")
    df["Ignore"] = 0
    
    # Convert timestamps
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")
    
    return df

def main():
    symbol = "ETHUSDT"
    interval = "15m"
    
    # Download from 2024-01-01 to now
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    
    print(f"Downloading {symbol} {interval} data")
    print(f"  From: {start.isoformat()}")
    print(f"  To:   {end.isoformat()}")
    print(f"  Est. bars: ~{(end - start).days * 96:,}")
    print()
    
    klines = download_binance_klines(symbol, interval, start_ms, end_ms)
    print(f"\nDownloaded {len(klines)} bars")
    
    if not klines:
        print("ERROR: No data downloaded")
        return
    
    df = klines_to_df(klines)
    print(f"Date range: {df['Open time'].iloc[0]} -> {df['Open time'].iloc[-1]}")
    
    # Save as extended dataset
    output = "/root/.openclaw/workspace/jimi_audit/data/eth_15m_extended.csv"
    df.to_csv(output, index=False)
    print(f"Saved: {output} ({len(df)} bars)")
    
    # Also merge with existing data
    existing = "/root/.openclaw/workspace/jimi_audit/data/eth_15m_merged.csv"
    if os.path.exists(existing):
        df_old = pd.read_csv(existing)
        print(f"\nExisting data: {len(df_old)} bars ({df_old['Open time'].iloc[0]} -> {df_old['Open time'].iloc[-1]})")
        
        # Merge: use extended data (it's more complete), add any bars from old that aren't in extended
        df_old["Open time"] = pd.to_datetime(df_old["Open time"])
        df["Open time"] = pd.to_datetime(df["Open time"])
        
        # Merge on timestamp
        df_merged = pd.concat([df_old, df]).drop_duplicates(subset=["Open time"]).sort_values("Open time").reset_index(drop=True)
        
        merged_output = "/root/.openclaw/workspace/jimi_audit/data/eth_15m_merged_extended.csv"
        df_merged.to_csv(merged_output, index=False)
        print(f"Merged: {merged_output} ({len(df_merged)} bars)")
        print(f"  From: {df_merged['Open time'].iloc[0]}")
        print(f"  To:   {df_merged['Open time'].iloc[-1]}")

if __name__ == "__main__":
    main()
