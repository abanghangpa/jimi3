#!/usr/bin/env python3
"""
Execution Quality Tracker — Shadow Trading Period
Logs every signal with expected vs actual metrics.

Track:
- Signal timestamp vs fill timestamp (latency)
- Expected entry vs actual entry (slippage)
- Spread at entry
- Signal strength (conviction, OBI magnitude)
- Regime, direction
- Whether fill was missed
"""
import json, os, time
from datetime import datetime, timezone

LOG_PATH = "/root/.openclaw/workspace/jimi_audit/live/data/execution_quality.jsonl"

def log_signal(signal, actual_fill=None):
    """
    Call this for every signal the executor produces.
    
    signal: dict with strategy, direction, entry, sl, tp1, conviction, regime, etc.
    actual_fill: dict with fill_price, fill_time, spread, slippage (None if shadow/dry-run)
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "signal_ts": signal.get("timestamp", ""),
        "strategy": signal.get("strategy", ""),
        "direction": signal.get("direction", ""),
        "regime": signal.get("regime", ""),
        "expected_entry": signal.get("entry", 0),
        "conviction": signal.get("conviction", 0),
        "obi_magnitude": signal.get("details", {}).get("obi_combined", 0),
        "trade_obi_z": signal.get("details", {}).get("trade_obi_z", 0),
        "vwap_dev": signal.get("details", {}).get("vwap_dev", 0),
        "vol_ratio": signal.get("details", {}).get("vol_ratio", 0),
        "bars_held": signal.get("hold_bars", 0),
        "sl_pct": signal.get("sl_pct", 0),
        "tp_pct": signal.get("tp_pct", 0),
    }
    
    if actual_fill:
        entry["actual_entry"] = actual_fill.get("fill_price", 0)
        entry["fill_latency_ms"] = actual_fill.get("latency_ms", 0)
        entry["spread_bps"] = actual_fill.get("spread_bps", 0)
        entry["slippage_bps"] = actual_fill.get("slippage_bps", 0)
        entry["total_cost_bps"] = actual_fill.get("total_cost_bps", 0)
        entry["filled"] = True
    else:
        # Shadow mode — no actual fill
        entry["actual_entry"] = None
        entry["filled"] = False
    
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def get_stats():
    """Analyze execution quality from logged signals."""
    if not os.path.exists(LOG_PATH):
        return {"error": "no data"}
    
    entries = []
    with open(LOG_PATH) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    if not entries:
        return {"error": "no entries"}
    
    filled = [e for e in entries if e.get("filled")]
    shadow = [e for e in entries if not e.get("filled")]
    
    stats = {
        "total_signals": len(entries),
        "filled": len(filled),
        "shadow": len(shadow),
    }
    
    if filled:
        slippages = [e.get("slippage_bps", 0) for e in filled]
        spreads = [e.get("spread_bps", 0) for e in filled]
        costs = [e.get("total_cost_bps", 0) for e in filled]
        latencies = [e.get("fill_latency_ms", 0) for e in filled]
        
        stats["avg_slippage_bps"] = round(sum(slippages) / len(slippages), 2)
        stats["avg_spread_bps"] = round(sum(spreads) / len(spreads), 2)
        stats["avg_total_cost_bps"] = round(sum(costs) / len(costs), 2)
        stats["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)
        stats["cost_vs_threshold"] = f"{stats['avg_total_cost_bps']:.1f} bps vs 15 bps threshold"
    
    # By strategy
    by_strat = {}
    for e in entries:
        s = e.get("strategy", "unknown")
        if s not in by_strat:
            by_strat[s] = {"count": 0, "convictions": []}
        by_strat[s]["count"] += 1
        by_strat[s]["convictions"].append(e.get("conviction", 0))
    
    for s in by_strat:
        convs = by_strat[s]["convictions"]
        by_strat[s]["avg_conviction"] = round(sum(convs) / len(convs), 3) if convs else 0
    
    stats["by_strategy"] = by_strat
    
    # Signal strength segmentation
    all_entries = sorted(entries, key=lambda e: e.get("conviction", 0))
    n = len(all_entries)
    if n >= 10:
        top_half = all_entries[n//2:]
        bottom_half = all_entries[:n//2]
        stats["top_50pct_avg_conviction"] = round(sum(e.get("conviction",0) for e in top_half) / len(top_half), 3)
        stats["bottom_50pct_avg_conviction"] = round(sum(e.get("conviction",0) for e in bottom_half) / len(bottom_half), 3)
    
    return stats

if __name__ == "__main__":
    stats = get_stats()
    print(json.dumps(stats, indent=2))
