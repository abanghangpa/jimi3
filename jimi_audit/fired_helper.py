import json, sys
d = json.load(open(sys.argv[1]))
for s in d.get("multi_strategy", {}).get("all_signals", []):
    if s.get("fired"):
        st = s.get("strategy", "?")
        di = s.get("direction", "?")
        en = s.get("entry", 0)
        sl = s.get("sl", 0)
        tp = s.get("tp1", 0)
        conv = s.get("conviction", 0)
        if di == "LONG":
            rr = (tp - en) / (en - sl) if en > sl else 0
            sl_pct = (en - sl) / en * 100
            tp_pct = (tp - en) / en * 100
        else:
            rr = (en - tp) / (sl - en) if sl > en else 0
            sl_pct = (sl - en) / en * 100
            tp_pct = (en - tp) / en * 100
        print(f"Strategy: {st}")
        print(f"Direction: {di}")
        print(f"Conviction: {conv:.4f}")
        print(f"Entry: {en:.2f}")
        print(f"SL: {sl:.2f} ({sl_pct:.2f}%)")
        print(f"TP: {tp:.2f} ({tp_pct:.2f}%)")
        print(f"R:R: {rr:.2f}")
        print()
