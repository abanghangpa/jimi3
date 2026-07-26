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

# FIX: compute swing levels from bars BEFORE the breakout bar
events = 0
for idx in range(48, len(ohlcv)):
    # For each potential breakout bar (up to 8 bars ago)
    for lb in range(1, min(8, idx)):
        bar_idx = idx - lb
        bar_high = highs[bar_idx]
        bar_low = lows[bar_idx]
        bar_close = closes[bar_idx]
        
        # Swing levels computed from bars BEFORE the breakout bar
        if bar_idx < 48:
            continue
        swing_high = float(np.max(highs[bar_idx-48:bar_idx]))
        swing_low = float(np.min(lows[bar_idx-48:bar_idx]))
        
        # Failed breakout ABOVE
        if bar_high > swing_high * 1.001:  # broke above
            if bar_close < swing_high:  # closed below (failed)
                # Count bars the breakout held
                bars_held = 0
                for j in range(bar_idx, idx + 1):
                    if highs[j] > swing_high:
                        bars_held += 1
                    else:
                        break
                if closes[idx] < swing_high and bars_held >= 1:
                    events += 1
                    if events <= 5:
                        print(f'SHORT FB at idx={idx}: level={swing_high:.2f}, bar_high={bar_high:.2f}, close={bar_close:.2f}, held={bars_held}')
                    break
        
        # Failed breakout BELOW
        if bar_low < swing_low * 0.999:  # broke below
            if bar_close > swing_low:  # closed above (failed)
                bars_held = 0
                for j in range(bar_idx, idx + 1):
                    if lows[j] < swing_low:
                        bars_held += 1
                    else:
                        break
                if closes[idx] > swing_low and bars_held >= 1:
                    events += 1
                    if events <= 5:
                        print(f'LONG FB at idx={idx}: level={swing_low:.2f}, bar_low={bar_low:.2f}, close={bar_close:.2f}, held={bars_held}')
                    break

print(f'Total failed breakout events: {events}')
