import json, sys
d = json.load(open(sys.argv[1]))
p = d.get('price', 0)
a = d.get('atr', 0)
print(f"Price: {p:.2f}  ATR: {a:.2f}  ATR%: {a/p*100:.2f}%")
print(f"TP1 1.5x ATR: {a*1.5:.2f}  SL 1.0x ATR: {a*1.0:.2f}")
print(f"TP1 min$15: {max(a*1.5,15):.2f}  SL min$30: {max(a*1.0,30):.2f}")
print(f"R:R raw: {a*1.5/a:.2f}  R:R floored: {max(a*1.5,15)/max(a,30):.2f}")
