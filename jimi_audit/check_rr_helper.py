
import sys, json
sys.path.insert(0, ".")
from src.strategies.base import BaseStrategy

class TestStrat(BaseStrategy):
    name = "test"
    strategy_type = "test"
    def check(self, data, **kwargs):
        return None

d = json.load(open(sys.argv[1]))
p = d.get("price", 0)
a = d.get("atr", 0)
s = TestStrat({})
sl, tp1, tp2, tp3, sl_pct, tp1_pct = s._calc_levels(p, "LONG", a, tp_mults=(1.5, 2.5, 4.0), sl_mult=1.0)
rr = (tp1 - p) / (p - sl) if p > sl else 0
print(f"Price: {p:.2f}  ATR: {a:.2f}")
print(f"TP1: {tp1:.2f} (dist={tp1-p:.2f})  SL: {sl:.2f} (dist={p-sl:.2f})")
print(f"R:R: {rr:.2f}")
