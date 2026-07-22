import sys
sys.path.insert(0, ".")
from src.utils.data_handler import fetch_recent
from src.utils.indicators import calc_atr

df = fetch_recent("ETH/USDT", 200)
atr = calc_atr(df["High"], df["Low"], df["Close"], 14)
last_atr = atr.iloc[-1]
price = df["Close"].iloc[-1]
print(f"Price: {price:.2f}")
print(f"ATR14: {last_atr:.2f}")
print(f"ATR%: {last_atr/price*100:.2f}%")
print(f"TP1 (1.5x ATR): {last_atr*1.5:.2f}")
print(f"SL (1.0x ATR): {last_atr*1.0:.2f}")
print(f"TP1 with min $15: {max(last_atr*1.5, 15):.2f}")
print(f"SL with min $30: {max(last_atr*1.0, 30):.2f}")
print(f"R:R without floors: {last_atr*1.5 / (last_atr*1.0):.2f}")
print(f"R:R with floors: {max(last_atr*1.5, 15) / max(last_atr*1.0, 30):.2f}")
