import pandas as pd
import numpy as np

DATA_DIR = '/root/.openclaw/workspace/jimi_audit/data'
ohlcv = pd.read_csv(f'{DATA_DIR}/eth_15m_extended.csv')
ohlcv['timestamp'] = pd.to_datetime(ohlcv['Open time'])
ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)
for c in ['Close','High','Low','Volume']: ohlcv[c] = ohlcv[c].astype(float)

print(f'Bars: {len(ohlcv)}')

highs = ohlcv['High'].values.astype(float)
lows = ohlcv['Low'].values.astype(float)
closes = ohlcv['Close'].values.astype(float)

# Check: how many bars break above swing_high?
above_count = 0
below_count = 0
for idx in range(48, min(2000, len(ohlcv))):
    lookback = min(48, idx)
    swing_high = float(np.max(highs[idx-lookback:idx]))
    swing_low = float(np.min(lows[idx-lookback:idx]))
    
    for lb in range(1, min(8, idx)):
        bar_idx = idx - lb
        if highs[bar_idx] > swing_high * 1.001:
            above_count += 1
            break
        if lows[bar_idx] < swing_low * 0.999:
            below_count += 1
            break

print(f'Bars breaking above swing_high (first 2000): {above_count}')
print(f'Bars breaking below swing_low (first 2000): {below_count}')

# Test detection
events = 0
for idx in range(48, len(ohlcv)):
    lookback = min(48, idx)
    swing_high = float(np.max(highs[idx-lookback:idx]))
    swing_low = float(np.min(lows[idx-lookback:idx]))
    
    for lb in range(1, min(8, idx)):
        bar_idx = idx - lb
        bar_high = highs[bar_idx]
        bar_close = closes[bar_idx]
        
        if bar_high > swing_high * 1.001:
            if bar_close < swing_high:
                bars_held = 0
                for j in range(bar_idx, idx + 1):
                    if highs[j] > swing_high:
                        bars_held += 1
                    else:
                        break
                if closes[idx] < swing_high and bars_held >= 1:
                    events += 1
                    if events <= 3:
                        print(f'SHORT FB at idx={idx}: level={swing_high:.2f}, bar_high={bar_high:.2f}, close={bar_close:.2f}, held={bars_held}')
                    break
    
    for lb in range(1, min(8, idx)):
        bar_idx = idx - lb
        bar_low = lows[bar_idx]
        bar_close = closes[bar_idx]
        
        if bar_low < swing_low * 0.999:
            if bar_close > swing_low:
                bars_held = 0
                for j in range(bar_idx, idx + 1):
                    if lows[j] < swing_low:
                        bars_held += 1
                    else:
                        break
                if closes[idx] > swing_low and bars_held >= 1:
                    events += 1
                    if events <= 3:
                        print(f'LONG FB at idx={idx}: level={swing_low:.2f}, bar_low={bar_low:.2f}, close={bar_close:.2f}, held={bars_held}')
                    break

print(f'Total failed breakout events: {events}')
