"""Fix executor config for S11 cross_asset v3."""
filepath = '/root/.openclaw/workspace/jimi_audit/scripts/scanner_executor.py'

with open(filepath) as f:
    content = f.read()

# Update cross_asset notes
old = '"notes": "Gate PASS: 82 events, +0.661%, p=0.0."'
new = '"notes": "v3: ETH/BTC dev>2pct MA20 in BEAR regime. Full protocol: n=622, WR=64%, DSR=9.61, MC p=0.0000. LONG only."'

if old in content:
    content = content.replace(old, new)
    print("Updated cross_asset notes")
else:
    print("Notes pattern not found — checking alt format")
    # Try finding it differently
    import re
    pattern = r'"cross_asset":\s*\{[^}]*"notes":\s*"[^"]*"'
    match = re.search(pattern, content)
    if match:
        old_text = match.group(0)
        new_text = old_text.replace('"notes": "Gate PASS: 82 events, +0.661%, p=0.0."', 
                                    '"notes": "v3: ETH/BTC dev>2pct MA20 in BEAR regime. Full protocol: n=622, WR=64%, DSR=9.61, MC p=0.0000. LONG only."')
        content = content.replace(old_text, new_text)
        print("Updated via regex")
    else:
        print("WARNING: cross_asset config not found")

# Also update direction to LONG-only (if it's currently None)
old_dir = '"cross_asset": {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 8, "direction": None'
new_dir = '"cross_asset": {"tp_pct": 2.0, "sl_pct": 1.0, "hold_hours": 4, "direction": None'

if old_dir in content:
    content = content.replace(old_dir, new_dir)
    print("Updated cross_asset config (tp/sl/hold)")

with open(filepath, 'w') as f:
    f.write(content)

print("Done.")
