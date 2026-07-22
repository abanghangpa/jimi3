#!/usr/bin/env python3
"""Daily execution quality report — runs at 00:00 UTC."""
import json, os, sys
from datetime import datetime, timezone

sys.path.insert(0, "/root/.openclaw/workspace/jimi_audit/scripts")
from execution_tracker import get_stats

stats = get_stats()
report_path = "/root/.openclaw/workspace/jimi_audit/live/data/execution_report.json"

report = {
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "generated_at": datetime.now(timezone.utc).isoformat(),
    **stats,
}

with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"Execution report: {json.dumps(stats, indent=2)}")
