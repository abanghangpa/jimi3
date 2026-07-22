#!/usr/bin/env python3
"""Monitor liquidity_grab v3.1 performance.
Reads executor trades, computes stats, logs to memory."""
import json, os
from datetime import datetime

TRADES_FILE = "/root/.openclaw/workspace/jimi_audit/live/data/executor_trades.json"
MONITOR_LOG = "/root/.openclaw/workspace/jimi_audit/data/lg_monitor.json"

def main():
    if not os.path.exists(TRADES_FILE):
        print("No trades file")
        return

    with open(TRADES_FILE) as f:
        trades = json.load(f)

    # Filter LG v3.1 trades
    lg_trades = [t for t in trades if t.get("strategy") == "liquidity_grab"]

    if not lg_trades:
        print("No LG trades yet")
        return

    wins = [t for t in lg_trades if t.get("pnl", 0) > 0]
    losses = [t for t in lg_trades if t.get("pnl", 0) <= 0]
    total_pnl = sum(t.get("pnl", 0) for t in lg_trades)
    wr = len(wins) / len(lg_trades) if lg_trades else 0

    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_trades": len(lg_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(wr, 3),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(lg_trades), 2) if lg_trades else 0,
        "trades": lg_trades[-10:],  # last 10
    }

    with open(MONITOR_LOG, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"LG v3.1: {len(lg_trades)} trades, WR={wr:.1%}, PnL=${total_pnl:.2f}")

if __name__ == "__main__":
    main()
