import json, sys
d = json.load(open(sys.argv[1]))
print(f"Keys: {list(d.keys())[:15]}")
ms = d.get("multi_strategy", {})
print(f"multi_strategy keys: {list(ms.keys())[:10]}")
sigs = ms.get("all_signals", [])
print(f"all_signals count: {len(sigs)}")
for s in sigs[:10]:
    print(f"  {s.get('strategy')}: fired={s.get('fired')} dir={s.get('direction')}")
# Find fired
for s in sigs:
    if s.get("fired"):
        print(f"\nFIRED: {s}")
        break
# Also check strategy_signal
ss = d.get("strategy_signal", {})
if ss:
    print(f"\nstrategy_signal: {ss.get('strategy')} fired={ss.get('direction') is not None}")
