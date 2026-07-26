"""Fix executor config for S20 v8b (LONG) and S01 v3.1 (LONG/SHORT)."""
import re

filepath = '/root/.openclaw/workspace/jimi_audit/scripts/scanner_executor.py'

with open(filepath) as f:
    content = f.read()

# Fix liquidation_cascade config: SHORT → LONG (v8b trades LONG mean reversion)
old_lc = '"liquidation_cascade": {"tp_pct": 2.0, "sl_pct": 1.0, "hold_hours": 4, "direction": "SHORT", "enabled": True, "group": "A", "min_conviction": 0.45, "notes": "v4: 8-Agent validated. PRIMARY: OI<-0.01+LS>1.5 (87evt, +0.375%, p=0.011). HIGH: OI<-0.015+LS>1.5 (29evt, +0.976%, p=0.0003). PREMIUM: OI<-0.015+MID (12evt, +2.22%, p=0.002)."},'
new_lc = '"liquidation_cascade": {"tp_pct": 2.0, "sl_pct": 1.0, "hold_hours": 4, "direction": None, "enabled": True, "group": "A", "min_conviction": 0.50, "notes": "v8b: OI drop >1.5pct mean reversion. Gate: 33 events, +0.642% 4h, p=0.030, WR=72.7%, MC p=0.007. LONG only."},'

if old_lc in content:
    content = content.replace(old_lc, new_lc)
    print("Fixed liquidation_cascade: SHORT → LONG (v8b)")
else:
    print("WARNING: liquidation_cascade config not found in expected format")

# Fix failed_breakout notes
old_fb = '"notes": "v9 merged config. Structural TP/SL with fallback. RANGING regime only."'
new_fb = '"notes": "v3.1: WEAK+ACCUM+LONG (MC p=0.033, WR=62.1%). Hybrid M14 detection. 95 events."'

if old_fb in content:
    content = content.replace(old_fb, new_fb)
    print("Fixed failed_breakout notes (v3.1)")
else:
    print("WARNING: failed_breakout notes not found in expected format")

with open(filepath, 'w') as f:
    f.write(content)

print("Done. Config updated.")
