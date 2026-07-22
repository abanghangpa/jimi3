#!/usr/bin/env python3
"""
VPS Analysis Toolkit — extend derivatives data + run isolation gates.
Run: python3 jimi_toolkit.py [command]

Commands:
  fetch-deriv    Fetch latest OI/funding/LS from Bybit + Binance
  extend-oi      Backfill OI history from Binance API
  gate           Run isolation gate on current data
  plot           Generate forward return distribution plot
  shell          Launch interactive IPython shell
"""
import os, sys, json, time, argparse
from datetime import datetime, timezone, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
DERIV_DIR = os.path.join(DATA_DIR, "derivatives_history")
REPORT_DIR = os.path.join(BASE, "reports")


def fetch_derivatives_snapshot():
    """Fetch current derivatives data from Bybit + Binance."""
    import requests

    result = {"timestamp": datetime.now(timezone.utc).isoformat()}

    # Binance: OI (primary — matches collector source, ~2.29M ETH)
    try:
        r = requests.get("https://fapi.binance.com/futures/data/openInterestHist",
                        params={"symbol": "ETHUSDT", "period": "5m", "limit": 1}, timeout=10)
        r.raise_for_status()
        d = r.json()[-1]
        result["oi"] = float(d["sumOpenInterest"])
        result["oi_usd"] = float(d["sumOpenInterestValue"])
    except Exception as e:
        print(f"  Binance OI: {e}")

    # Bybit: funding rate
    try:
        r = requests.get("https://api.bybit.com/v5/market/funding/history",
                        params={"category": "linear", "symbol": "ETHUSDT", "limit": 1}, timeout=10)
        d = r.json().get("result", {}).get("list", [{}])[0]
        result["funding_rate"] = float(d.get("fundingRate", 0))
    except Exception as e:
        print(f"  Bybit funding: {e}")

    # Bybit: L/S ratio (from account-ratio endpoint)
    try:
        r = requests.get("https://api.bybit.com/v5/market/account-ratio",
                        params={"category": "linear", "symbol": "ETHUSDT", "period": "5min", "limit": 1}, timeout=10)
        d = r.json().get("result", {}).get("list", [{}])[0]
        buy_ratio = float(d.get("buyRatio", 0))
        sell_ratio = float(d.get("sellRatio", 1))
        result["ls_ratio"] = round(buy_ratio / sell_ratio, 4) if sell_ratio > 0 else None
    except Exception as e:
        print(f"  Bybit LS: {e}")

    # Binance OI already fetched as primary above

    # Binance: L/S ratio (cross-check)
    try:
        r = requests.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                        params={"symbol": "ETHUSDT", "period": "5m", "limit": 1}, timeout=10)
        d = r.json()[-1]
        result["binance_ls"] = float(d["longShortRatio"])
    except Exception as e:
        print(f"  Binance LS: {e}")

    return result


def append_derivatives(snapshot):
    """Append snapshot to derivatives_collected.csv (14 columns)."""
    csv_path = os.path.join(DERIV_DIR, "derivatives_collected.csv")
    ts = snapshot["timestamp"]
    ts_ms = int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)

    # 14 columns: timestamp, timestamp_ms, oi, oi_usd, ls_ratio, long_pct, short_pct,
    #             top_ls_ratio, top_long_pct, top_short_pct, futures_taker_ratio,
    #             futures_buy_vol, futures_sell_vol, funding_rate
    row = f"{ts},{ts_ms},{snapshot.get('oi','')},{snapshot.get('oi_usd','')},{snapshot.get('ls_ratio','')},,,,,,,,{snapshot.get('funding_rate','')}"

    if os.path.exists(csv_path):
        with open(csv_path) as f:
            header = f.readline().strip()
    else:
        header = "timestamp,timestamp_ms,oi,oi_usd,ls_ratio,long_pct,short_pct,top_ls_ratio,top_long_pct,top_short_pct,futures_taker_ratio,futures_buy_vol,futures_sell_vol,funding_rate,oi_source"
        with open(csv_path, "w") as f:
            f.write(header + "\n")

    with open(csv_path, "a") as f:
        f.write(row + "\n")

    return csv_path


def fetch_and_store():
    """Fetch snapshot and store."""
    print("Fetching derivatives snapshot...")
    snap = fetch_derivatives_snapshot()
    print(f"  OI: {snap.get('oi', 'N/A')}")
    print(f"  LS: {snap.get('ls_ratio', 'N/A')}")
    print(f"  Funding: {snap.get('funding_rate', 'N/A')}")
    path = append_derivatives(snap)
    print(f"  Saved to: {path}")
    return snap


def extend_oi_history(days=14):
    """Backfill OI history from Binance API. Max 14 days (API limit)."""
    import requests

    days = min(days, 14)  # Binance openInterestHist max lookback ~14 days
    print(f"Fetching {days} days of OI history from Binance...")
    all_data = []
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 86400 * 1000)

    while start_time < end_time:
        r = requests.get("https://fapi.binance.com/futures/data/openInterestHist",
                        params={"symbol": "ETHUSDT", "period": "5m",
                                "startTime": start_time, "limit": 500}, timeout=15)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            print(f"  API returned: {str(data)[:200]}")
            break
        all_data.extend(data)
        start_time = int(data[-1]["timestamp"]) + 1
        time.sleep(0.2)

    print(f"  Fetched {len(all_data)} data points")

    out_path = os.path.join(DERIV_DIR, "oi_history_binance.csv")
    with open(out_path, "w") as f:
        f.write("timestamp,sumOpenInterest,sumOpenInterestValue\n")
        for d in all_data:
            f.write(f"{d['timestamp']},{d['sumOpenInterest']},{d['sumOpenInterestValue']}\n")

    print(f"  Saved to: {out_path}")
    return len(all_data)


def run_isolation_gate():
    """Run isolation gate with current data."""
    import pandas as pd
    import numpy as np
    from scipy import stats

    print("Running isolation gate...")

    ohlcv = pd.read_csv(f"{DATA_DIR}/eth_15m_extended.csv")
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["Open time"])
    ohlcv = ohlcv.sort_values("timestamp").reset_index(drop=True)

    deriv = pd.read_csv(f"{DERIV_DIR}/derivatives_collected.csv")
    deriv["timestamp"] = pd.to_datetime(deriv["timestamp"], format="mixed", utc=True).dt.tz_localize(None)
    deriv = deriv.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        ohlcv, deriv[["timestamp", "oi", "ls_ratio", "funding_rate"]],
        on="timestamp", direction="backward", tolerance=pd.Timedelta("2h")
    )
    merged["oi_roc"] = merged["oi"].pct_change(4, fill_method=None)
    merged["fwd_ret_16"] = merged["Close"].shift(-16) / merged["Close"] - 1

    configs = [
        ("OI<-0.01 + LS>1.5", (merged["oi_roc"] < -0.01) & (merged["ls_ratio"] > 1.5)),
        ("OI<-0.015 + LS>1.5", (merged["oi_roc"] < -0.015) & (merged["ls_ratio"] > 1.5)),
        ("OI<-0.015 only", merged["oi_roc"] < -0.015),
    ]

    results = {}
    for name, mask in configs:
        shifted = mask.shift(1).fillna(False)
        events = merged[shifted]
        rets = merged.loc[events.index, "fwd_ret_16"].dropna()
        if len(rets) < 5:
            print(f"  {name}: n={len(rets)} (too few)")
            continue
        mean_r = rets.mean()
        t, p = stats.ttest_1samp(rets, 0)
        wr = (rets > 0).mean()
        gate = "PASS" if p < 0.1 and mean_r > 0.001 else "FAIL"
        print(f"  {'+' if gate=='PASS' else '-'} {name:30s} n={len(rets):4d} mean={mean_r*100:+.4f}% p={p:.4f} WR={wr:.1%} [{gate}]")
        results[name] = {"passed": gate == "PASS", "events": len(rets), "mean_pct": round(mean_r*100, 4), "p": round(p, 4), "wr": round(wr, 4)}

    # Write to gate registry
    gate_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'isolation_gate_results.json')
    if os.path.exists(gate_path):
        try:
            with open(gate_path) as f:
                registry = json.load(f)
            for cfg_name, result in results.items():
                key = f'toolkit_gate_{cfg_name}'
                registry[key] = {'passed': result['passed'], 'events': result['events'], 'mean_return_pct': result['mean_pct'], 'p_value': result['p'], 'win_rate': result['wr'], 'date': datetime.now().strftime('%Y-%m-%d'), 'source': 'jimi_toolkit.py gate'}
            with open(gate_path, 'w') as f:
                json.dump(registry, f, indent=2)
            print(f'  Registry updated: {gate_path}')
        except Exception as e:
            print(f'  Registry write failed: {e}')

    return results


def plot_forward_returns():
    """Generate forward return distribution plot."""
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ohlcv = pd.read_csv(f"{DATA_DIR}/eth_15m_extended.csv")
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["Open time"])
    ohlcv = ohlcv.sort_values("timestamp").reset_index(drop=True)

    deriv = pd.read_csv(f"{DERIV_DIR}/derivatives_collected.csv")
    deriv["timestamp"] = pd.to_datetime(deriv["timestamp"], format="mixed", utc=True).dt.tz_localize(None)

    merged = pd.merge_asof(ohlcv, deriv[["timestamp","oi","ls_ratio"]], on="timestamp", direction="backward", tolerance=pd.Timedelta("2h"))
    merged["oi_roc"] = merged["oi"].pct_change(4, fill_method=None)
    merged["fwd_ret_16"] = merged["Close"].shift(-16) / merged["Close"] - 1

    mask = (merged["oi_roc"] < -0.01) & (merged["ls_ratio"] > 1.5)
    shifted = mask.shift(1).fillna(False)
    events = merged[shifted]
    rets = merged.loc[events.index, "fwd_ret_16"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(rets * 100, bins=30, edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--", label="Zero")
    axes[0].axvline(rets.mean() * 100, color="green", linestyle="-", label=f"Mean: {rets.mean()*100:.3f}%")
    axes[0].set_xlabel("Forward Return (%)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"OI<-0.01 + LS>1.5 (4h forward, n={len(rets)})")
    axes[0].legend()

    sorted_rets = np.sort(rets.values)
    cumprob = np.arange(1, len(sorted_rets) + 1) / len(sorted_rets)
    axes[1].plot(sorted_rets * 100, cumprob)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].axvline(0.10, color="orange", linestyle=":", label="Cost threshold (0.10%)")
    axes[1].set_xlabel("Forward Return (%)")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_title("CDF of Forward Returns")
    axes[1].legend()

    plt.tight_layout()
    out = f"{REPORT_DIR}/cascade_forward_returns.png"
    plt.savefig(out, dpi=150)
    print(f"Plot saved: {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JIMI Analysis Toolkit")
    parser.add_argument("command", choices=["fetch-deriv", "extend-oi", "gate", "plot", "shell"],
                       help="Command to run")
    parser.add_argument("--days", type=int, default=30, help="Days of history to fetch")
    args = parser.parse_args()

    if args.command == "fetch-deriv":
        fetch_and_store()
    elif args.command == "extend-oi":
        extend_oi_history(args.days)
    elif args.command == "gate":
        run_isolation_gate()
    elif args.command == "plot":
        plot_forward_returns()
    elif args.command == "shell":
        os.system("bash /root/.openclaw/workspace/jimi_audit/jimi_shell.sh")
