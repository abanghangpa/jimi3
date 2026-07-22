"""Data loading, resampling, and exchange fetching utilities."""

import pandas as pd
import numpy as np
import os
import json
import time
import ccxt
from pathlib import Path


def load_data(filepath):
    """Load 15m OHLCV CSV and normalize columns."""
    df = pd.read_csv(filepath, low_memory=False)
    # Keep only first 11 columns (expected OHLCV fields)
    expected_cols = 11
    if df.shape[1] > expected_cols:
        df = df.iloc[:, :expected_cols]
    df.columns = df.columns.str.strip()
    df['Open time'] = pd.to_datetime(df['Open time'].astype(str).str.strip(), format='mixed')
    # Close time may be raw ms or formatted datetime
    close_raw = df['Close time'].astype(str).str.strip()
    try:
        df['Close time'] = pd.to_datetime(close_raw, format='mixed')
    except (ValueError, pd.errors.OutOfBoundsDatetime):
        # Raw millisecond timestamps
        df['Close time'] = pd.to_datetime(pd.to_numeric(close_raw, errors='coerce'), unit='ms')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume',
                'Quote asset volume', 'Number of trades',
                'Taker buy base asset volume', 'Taker buy quote asset volume']:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
    df = df.sort_values('Open time').reset_index(drop=True)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    return df


def resample_ohlcv(df_15m, timeframe):
    """Resample 15m data to higher timeframe."""
    df = df_15m.copy().set_index('Open time')
    agg = {
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum', 'Quote asset volume': 'sum',
        'Number of trades': 'sum',
        'Taker buy base asset volume': 'sum',
        'Taker buy quote asset volume': 'sum',
    }
    rule_map = {'30m': '30min', '1H': '1h', '2H': '2h', '4H': '4h', '1D': '1D'}
    return df.resample(rule_map[timeframe]).agg(agg).dropna(subset=['Open']).reset_index()


_binance_exchange = None


def _get_binance():
    global _binance_exchange
    if _binance_exchange is None:
        _binance_exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})  # 20 seconds
    return _binance_exchange


def fetch_recent(symbol='ETH/USDT', timeframe='15m', bars=1000):
    """Fetch recent OHLCV from Binance (live scanner).

    Supports multi-request fetching when bars > 1000 (Binance API limit).
    Makes multiple paginated requests and stitches data together.
    """
    ex = _get_binance()
    symbol_raw = symbol.replace('/', '')
    max_per_request = 1000

    all_rows = []
    remaining = bars
    end_time = None  # fetch backwards from now

    while remaining > 0:
        limit = min(remaining, max_per_request)
        params = {
            'symbol': symbol_raw, 'interval': timeframe, 'limit': limit,
        }
        if end_time is not None:
            params['endTime'] = end_time

        raw = ex.publicGetKlines(params)
        if not raw:
            break

        for c in raw:
            all_rows.append({
                'Open time': pd.to_datetime(int(c[0]), unit='ms'),
                'Open': float(c[1]), 'High': float(c[2]), 'Low': float(c[3]),
                'Close': float(c[4]), 'Volume': float(c[5]),
                'Close time': pd.to_datetime(int(c[6]), unit='ms'),
                'Quote asset volume': float(c[7]),
                'Number of trades': int(c[8]),
                'Taker buy base asset volume': float(c[9]),
                'Taker buy quote asset volume': float(c[10]),
            })

        if len(raw) < limit:
            break

        # Move end_time to before the earliest candle we got
        end_time = int(raw[0][0]) - 1
        remaining -= len(raw)

    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        df = df.drop_duplicates(subset='Open time').sort_values('Open time').reset_index(drop=True)
    return df


def fetch_daily_ohlcv_ccxt(symbol, start_date, end_date, exchange_id='binance',
                           cache_dir="/tmp/jimi_cache"):
    """Fetch daily OHLCV using ccxt with file caching."""
    os.makedirs(cache_dir, exist_ok=True)
    safe = symbol.replace("/", "_").replace(":", "_")
    cache_file = os.path.join(cache_dir, f"{safe}_daily.json")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if os.path.exists(cache_file):
        try:
            cached = pd.read_json(cache_file)
            if len(cached) > 0:
                cached['date'] = pd.to_datetime(cached['date']).dt.normalize()
                if cached['date'].min() <= start and cached['date'].max() >= end:
                    return cached[(cached['date'] >= start) & (cached['date'] <= end)].reset_index(drop=True)
        except Exception:
            pass

    exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
    since_ms = int(start.timestamp() * 1000)
    until_ms = int(end.timestamp() * 1000)

    candles = []
    cur = since_ms
    retries = 0
    while cur < until_ms and retries < 3:
        try:
            raw = exchange.fetch_ohlcv(symbol, '1d', since=cur, limit=1000)
            if not raw:
                break
            for c in raw:
                ts = int(c[0])
                if ts >= until_ms:
                    break
                candles.append({
                    'date': pd.to_datetime(ts, unit='ms').normalize().isoformat(),
                    'open': float(c[1]), 'high': float(c[2]),
                    'low': float(c[3]), 'close': float(c[4]),
                    'volume': float(c[5]),
                })
            last = raw[-1][0]
            if last <= cur:
                break
            cur = last + 1
            retries = 0
        except Exception:
            retries += 1
            time.sleep(5 * retries)

    df = pd.DataFrame(candles)
    if len(df) > 0:
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        try:
            df.to_json(cache_file, orient='records', date_format='iso')
        except Exception:
            pass
    return df


def fetch_btc_15m(start_time, end_time):
    """Fetch BTC/USDT 15m data for cross-asset correlation."""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        since_ms = int(pd.Timestamp(start_time).timestamp() * 1000)
        until_ms = int(pd.Timestamp(end_time).timestamp() * 1000)
        all_candles = []
        current = since_ms
        while current < until_ms:
            raw = exchange.fetch_ohlcv('BTC/USDT', '15m', since=current, limit=1000)
            if not raw:
                break
            for c in raw:
                ts = int(c[0])
                if ts >= until_ms:
                    break
                all_candles.append({
                    'Open time': pd.to_datetime(ts, unit='ms'),
                    'Open': float(c[1]), 'High': float(c[2]),
                    'Low': float(c[3]), 'Close': float(c[4]),
                    'Volume': float(c[5]),
                })
            last_ts = raw[-1][0]
            if last_ts <= current:
                break
            current = last_ts + 1
        return pd.DataFrame(all_candles) if all_candles else None
    except Exception as e:
        print(f"  BTC fetch error: {e}")
        return None


def load_daily_from_csv(csv_path):
    """Load 15m OHLCV CSV and resample to daily for EMA warmup."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    df_15m = load_data(csv_path)
    if df_15m is None or len(df_15m) == 0:
        return None
    df_1d = resample_ohlcv(df_15m, '1D')
    return df_1d