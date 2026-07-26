import json, os
scan_dir = '/root/.openclaw/workspace/jimi_audit/data/scans'
files = sorted(os.listdir(scan_dir))
print(f"Total scans: {len(files)}")
print(f"First: {files[0]}, Last: {files[-1]}")

# Check M14/M21 in recent scans
for f in files[-5:]:
    with open(os.path.join(scan_dir, f)) as fh:
        d = json.load(fh)
    m14 = d.get('m14', {})
    m21 = d.get('m21', {})
    print(f"{f}: M14 status={m14.get('status','?')} score={m14.get('score','?')} | M21 status={m21.get('status','?')} phase={m21.get('phase','?')} zone={m21.get('zone','?')} spring={m21.get('spring_upthrust','?')}")

# Count scans with M14 sweep
m14_sweeps = 0
m21_springs = 0
for f in files:
    with open(os.path.join(scan_dir, f)) as fh:
        d = json.load(fh)
    m14 = d.get('m14', {})
    m21 = d.get('m21', {})
    if m14.get('score', 0) > 0.5:
        m14_sweeps += 1
    if 'spring' in str(m21.get('spring_upthrust', '')).lower() or 'upthrust' in str(m21.get('spring_upthrust', '')).lower():
        m21_springs += 1

print(f"\nM14 score > 0.5: {m14_sweeps}/{len(files)}")
print(f"M21 spring/upthrust: {m21_springs}/{len(files)}")

# Sample M14 and M21 full dict
with open(os.path.join(scan_dir, files[-1])) as fh:
    d = json.load(fh)
print(f"\nM14 full: {json.dumps(d.get('m14', {}), indent=2)[:500]}")
print(f"\nM21 full: {json.dumps(d.get('m21', {}), indent=2)[:500]}")
