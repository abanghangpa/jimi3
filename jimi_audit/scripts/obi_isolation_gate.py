#!/usr/bin/env python3
"""OBI Isolation Gate — Historical L2 Orderbook Validation"""
import csv, json, os, sys
import numpy as np
from scipy import stats
from datetime import datetime

CSV_PATH = "/root/.openclaw/workspace/jimi_audit/data/ob_history/ob_historical.csv"
REPORT_PATH = "/root/.openclaw/workspace/jimi_audit/reports/ob_historical_validation.json"

print("Loading data...", flush=True)
# Read in chunks, compute forward returns on the fly
# We need: ob_ratio, top5_ratio, and price (best_bid) at each row
# Forward return = price[t+h] - price[t] / price[t]

ob_ratios = []
top5_ratios = []
prices = []

with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        try:
            ob_ratios.append(float(row["ob_ratio"]))
            top5_ratios.append(float(row["top5_ratio"]))
            prices.append(float(row["best_bid"]))
        except (ValueError, KeyError):
            continue
        if i % 5000000 == 0:
            print(f"  Loaded {i/1e6:.0f}M rows...", flush=True)

ob_ratios = np.array(ob_ratios)
top5_ratios = np.array(top5_ratios)
prices = np.array(prices)
n = len(prices)
print(f"Total rows: {n:,}", flush=True)

# Compute forward returns at different horizons
horizons = {"1m": 1, "4m": 4, "16m": 16, "64m": 64, "256m": 256}

results = {}

# Test 1: ob_ratio thresholds
print("\n=== ISOLATION GATE: ob_ratio ===", flush=True)
for thresh_name, thresh_fn in [
    ("bid_heavy_0.05", lambda x: x > 0.05),
    ("bid_heavy_0.10", lambda x: x > 0.10),
    ("ask_heavy_-0.05", lambda x: x < -0.05),
    ("ask_heavy_-0.10", lambda x: x < -0.10),
    ("extreme_bid_0.20", lambda x: x > 0.20),
    ("extreme_ask_-0.20", lambda x: x < -0.20),
]:
    mask = thresh_fn(ob_ratios)
    event_count = np.sum(mask)
    
    if event_count < 100:
        print(f"\n{thresh_name}: n={event_count} (TOO SMALL, skip)")
        continue
    
    print(f"\n{thresh_name}: n={event_count:,}", flush=True)
    results[thresh_name] = {"n": int(event_count), "horizons": {}}
    
    for h_name, h_bars in horizons.items():
        if h_bars >= n:
            continue
        fwd = (prices[h_bars:] - prices[:-h_bars]) / prices[:-h_bars] * 100
        mask_trimmed = mask[:-h_bars]
        
        event_returns = fwd[mask_trimmed]
        non_event_returns = fwd[~mask_trimmed]
        
        if len(event_returns) < 50:
            continue
        
        mean_ret = np.mean(event_returns)
        std_ret = np.std(event_returns)
        wr = np.mean(event_returns > 0) * 100
        
        # t-test: event vs non-event
        t_stat, p_value = stats.ttest_ind(event_returns, non_event_returns[:min(len(non_event_returns), len(event_returns)*10)])
        
        direction = "CORRECT" if mean_ret > 0 else "BACKWARDS"
        gate_pass = p_value < 0.1 and abs(mean_ret) > 0.04  # above round-trip cost
        
        print(f"  {h_name}: mean={mean_ret:+.4f}% WR={wr:.1f}% p={p_value:.4f} {direction} {'PASS' if gate_pass else 'FAIL'}", flush=True)
        
        results[thresh_name]["horizons"][h_name] = {
            "mean_return_pct": round(float(mean_ret), 6),
            "std_pct": round(float(std_ret), 6),
            "win_rate": round(float(wr), 2),
            "p_value": round(float(p_value), 6),
            "direction": direction,
            "gate_pass": gate_pass,
        }

# Test 2: top5_ratio thresholds
print("\n=== ISOLATION GATE: top5_ratio ===", flush=True)
for thresh_name, thresh_fn in [
    ("top5_bid_0.10", lambda x: x > 0.10),
    ("top5_bid_0.20", lambda x: x > 0.20),
    ("top5_ask_-0.10", lambda x: x < -0.10),
    ("top5_ask_-0.20", lambda x: x < -0.20),
    ("top5_extreme_bid_0.40", lambda x: x > 0.40),
    ("top5_extreme_ask_-0.40", lambda x: x < -0.40),
]:
    mask = thresh_fn(top5_ratios)
    event_count = np.sum(mask)
    
    if event_count < 100:
        print(f"\n{thresh_name}: n={event_count} (TOO SMALL, skip)")
        continue
    
    print(f"\n{thresh_name}: n={event_count:,}", flush=True)
    results[thresh_name] = {"n": int(event_count), "horizons": {}}
    
    for h_name, h_bars in horizons.items():
        if h_bars >= n:
            continue
        fwd = (prices[h_bars:] - prices[:-h_bars]) / prices[:-h_bars] * 100
        mask_trimmed = mask[:-h_bars]
        
        event_returns = fwd[mask_trimmed]
        non_event_returns = fwd[~mask_trimmed]
        
        if len(event_returns) < 50:
            continue
        
        mean_ret = np.mean(event_returns)
        std_ret = np.std(event_returns)
        wr = np.mean(event_returns > 0) * 100
        
        t_stat, p_value = stats.ttest_ind(event_returns, non_event_returns[:min(len(non_event_returns), len(event_returns)*10)])
        
        direction = "CORRECT" if mean_ret > 0 else "BACKWARDS"
        gate_pass = p_value < 0.1 and abs(mean_ret) > 0.04
        
        print(f"  {h_name}: mean={mean_ret:+.4f}% WR={wr:.1f}% p={p_value:.4f} {direction} {'PASS' if gate_pass else 'FAIL'}", flush=True)
        
        results[thresh_name]["horizons"][h_name] = {
            "mean_return_pct": round(float(mean_ret), 6),
            "std_pct": round(float(std_ret), 6),
            "win_rate": round(float(wr), 2),
            "p_value": round(float(p_value), 6),
            "direction": direction,
            "gate_pass": gate_pass,
        }

# Save report
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
report = {
    "timestamp": datetime.utcnow().isoformat(),
    "data_rows": n,
    "date_range": "2026-07-01 to 2026-07-20",
    "results": results,
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved to {REPORT_PATH}")
print("\n=== SUMMARY ===")
for name, data in results.items():
    best_h = None
    best_p = 1.0
    for h_name, h_data in data["horizons"].items():
        if h_data["p_value"] < best_p:
            best_p = h_data["p_value"]
            best_h = h_name
    if best_h:
        h = data["horizons"][best_h]
        status = "PASS" if h["gate_pass"] else "FAIL"
        print(f"  {name}: n={data['n']:,} best={best_h} mean={h['mean_return_pct']:+.4f}% p={h['p_value']:.4f} {h['direction']} {status}")
