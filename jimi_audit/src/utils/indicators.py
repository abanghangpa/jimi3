"""Technical indicator calculations."""

import pandas as pd
import numpy as np


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_vwap(high, low, close, volume, lookback=96):
    typical = (high + low + close) / 3
    cum_tv = (typical * volume).rolling(lookback).sum()
    cum_vol = volume.rolling(lookback).sum()
    return cum_tv / cum_vol.replace(0, np.nan)


def calc_vol_ratio(volume_15m):
    vol_24h = volume_15m.rolling(96).sum()
    vol_7d = volume_15m.rolling(672).sum()
    return vol_24h / vol_7d.replace(0, np.nan)


def calc_swing_bias(df_1d):
    ema21 = calc_ema(df_1d['Close'], 21)
    ema55 = calc_ema(df_1d['Close'], 55)
    bias = pd.Series('NEUTRAL', index=df_1d.index)
    bias[ema21 > ema55] = 'BULLISH'
    bias[ema21 < ema55] = 'BEARISH'
    return bias


def calc_phase0(df_1d):
    rsi = calc_rsi(df_1d['Close'], 14)
    vol_ma20 = df_1d['Volume'].rolling(20).mean()
    vol_ratio = df_1d['Volume'] / vol_ma20.replace(0, np.nan)
    rsi_score = (rsi - 50).abs() / 50
    vol_score = vol_ratio.clip(0, 3) / 3
    return (rsi_score * 0.6 + vol_score * 0.4).clip(0, 1)


def calc_phase0_1h(df_1h):
    rsi = calc_rsi(df_1h['Close'], 14)
    vol_ma20 = df_1h['Volume'].rolling(20).mean()
    vol_ratio = df_1h['Volume'] / vol_ma20.replace(0, np.nan)
    rsi_score = (rsi - 50).abs() / 50
    vol_score = vol_ratio.clip(0, 3) / 3
    return (rsi_score * 0.6 + vol_score * 0.4).clip(0, 1)


def calc_trend_state(df_1d):
    """Compute daily trend state using multiple confirmations."""
    close = df_1d['Close']
    high = df_1d['High']
    low = df_1d['Low']

    ema21 = calc_ema(close, 21)
    ema55 = calc_ema(close, 55)
    rsi = calc_rsi(close, 14)
    roc_7d = close.pct_change(7)

    hh = (high > high.shift(1)) & (high.shift(1) > high.shift(2))
    ll = (low < low.shift(1)) & (low.shift(1) < low.shift(2))

    trend_score = pd.Series(0.0, index=df_1d.index)

    ema_diff = (ema21 - ema55) / ema55
    trend_score += ema_diff.clip(-0.10, 0.10) * 3.0

    price_vs_ema = (close - ema21) / ema21
    trend_score += price_vs_ema.clip(-0.05, 0.05) * 4.0

    trend_score += roc_7d.clip(-0.10, 0.10) * 2.5

    rsi_signal = (rsi - 50) / 50
    trend_score += rsi_signal.clip(-0.50, 0.50) * 0.30

    structure = hh.astype(float) - ll.astype(float)
    trend_score += structure.rolling(3).mean().clip(-0.10, 0.10)

    trend_score = trend_score.clip(-1.0, 1.0)

    trend = pd.Series('NEUTRAL', index=df_1d.index)
    trend[trend_score > 0.40] = 'STRONG_UP'
    trend[(trend_score > 0.15) & (trend_score <= 0.40)] = 'UP'
    trend[(trend_score < -0.15) & (trend_score >= -0.40)] = 'DOWN'
    trend[trend_score < -0.40] = 'STRONG_DOWN'

    return trend, trend_score


def compute_btc_correlation(df_15m, btc_df, lookback=48):
    """Compute rolling correlation between ETH and BTC returns."""
    if btc_df is None or len(btc_df) < lookback + 10:
        return pd.Series(0.5, index=df_15m.index)

    eth_ret = df_15m['Close'].pct_change()
    btc_ret = btc_df['Close'].pct_change()

    min_len = min(len(eth_ret), len(btc_ret))
    eth_ret = eth_ret.iloc[-min_len:].reset_index(drop=True)
    btc_ret = btc_ret.iloc[-min_len:].reset_index(drop=True)

    corr = eth_ret.rolling(lookback).corr(btc_ret)
    result = pd.Series(0.5, index=df_15m.index)
    result.iloc[-min_len:] = corr.fillna(0.5).values
    return result
