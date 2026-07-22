import json, sys
d = json.load(open(sys.argv[1]))
ss = d.get("strategy_signal", {})
if ss:
    st = ss.get("strategy", "?")
    di = ss.get("direction", "?")
    en = ss.get("entry", 0)
    sl = ss.get("sl", 0)
    tp = ss.get("tp1", 0)
    conv = ss.get("conviction", 0)
    if di == "LONG":
        rr = (tp - en) / (en - sl) if en > sl else 0
        sl_pct = (en - sl) / en * 100 if en > 0 else 0
        tp_pct = (tp - en) / en * 100 if en > 0 else 0
    else:
        rr = (en - tp) / (sl - en) if sl > en else 0
        sl_pct = (sl - en) / en * 100 if en > 0 else 0
        tp_pct = (en - tp) / en * 100 if en > 0 else 0
    print(f"Strategy: {st}")
    print(f"Direction: {di}")
    print(f"Conviction: {conv:.4f}")
    print(f"Entry: {en:.2f}")
    print(f"SL: {sl:.2f} ({sl_pct:.2f}%)")
    print(f"TP: {tp:.2f} ({tp_pct:.2f}%)")
    print(f"R:R: {rr:.2f}")
