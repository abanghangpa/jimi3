#!/usr/bin/env python3
"""
SQUEEZE_BREAKOUT V5B — PROPER BACKTEST with 2.5 years of data.

Tests fade-the-failure strategy across:
- Bull markets (2024 Q1, 2025 Q4)
- Bear markets (2024 Q3, 2026 Q2)
- Ranging markets (2024 Q2, 2025 Q2)
- Crisis events (2024 Aug, 2025 Jan)
"""

import json, os, sys, csv
from datetime import datetime, timezone
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_csv_data(csv_path, max_rows=None):
    """Load candles from CSV."""
    candles = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            candles.append({
                "ts": int(row["ts"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "taker_buy_vol": float(row.get("taker_buy_vol", 0)),
            })
    return candles


def compute_atr(highs, lows, closes, period=14):
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(highs))]
    return sum(trs[-period:]) / period if len(trs) >= period else (sum(trs)/len(trs) if trs else 0)


def compute_ema(data, period):
    if len(data) < period: return data[-1] if data else 0
    ema = np.mean(data[:period])
    mult = 2 / (period + 1)
    for i in range(period, len(data)):
        ema = data[i] * mult + ema * (1 - mult)
    return ema


def compute_bb(closes, period=20, std_mult=2.0):
    if len(closes) < period: return closes[-1], closes[-1], closes[-1]
    sma = np.mean(closes[-period:]); std = np.std(closes[-period:])
    return sma, sma + std_mult * std, sma - std_mult * std


def compute_keltner(highs, lows, closes, period=20, mult=1.5):
    ema = compute_ema(closes, period)
    atr = compute_atr(highs, lows, closes, period)
    return ema, ema + mult * atr, ema - mult * atr


def run_backtest(candles, version="v3", initial_capital=200.0):
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]
    
    capital = initial_capital; peak = initial_capital
    positions = []; trades = []; fee_rate = 0.0004
    
    in_squeeze_prev = False
    breakout_bar = None; breakout_dir = None; breakout_extreme = None
    last_signal_bar = -999
    signals = 0; filtered = 0
    
    for i in range(60, len(candles)):
        price = closes[i]
        
        # Exits
        closed = []
        for pos in positions:
            pos["bars_held"] = pos.get("bars_held", 0) + 1
            exited = False; outcome = "TIMEOUT"; exit_p = price
            if pos["direction"] == "LONG":
                if lows[i] <= pos["sl"]: outcome = "LOSS"; exit_p = pos["sl"]; exited = True
                elif highs[i] >= pos["tp"]: outcome = "WIN"; exit_p = pos["tp"]; exited = True
            else:
                if highs[i] >= pos["sl"]: outcome = "LOSS"; exit_p = pos["sl"]; exited = True
                elif lows[i] <= pos["tp"]: outcome = "WIN"; exit_p = pos["tp"]; exited = True
            if not exited and pos.get("bars_held", 0) >= pos.get("hold_bars", 8):
                exited = True; exit_p = price
            if exited:
                pnl = ((exit_p - pos["entry"]) / pos["entry"] * 100) if pos["direction"] == "LONG" else ((pos["entry"] - exit_p) / pos["entry"] * 100)
                pnl -= fee_rate * 100
                pnl_dollar = pos["size"] * pnl / 100 * 10
                capital += pnl_dollar; peak = max(peak, capital)
                
                # Get date for per-period analysis
                ts_dt = datetime.fromtimestamp(candles[i]["ts"]/1000, tz=timezone.utc)
                
                trades.append({
                    "bar": pos["entry_bar"], "direction": pos["direction"],
                    "entry": pos["entry"], "exit": round(exit_p, 2),
                    "outcome": outcome, "pnl_pct": round(pnl, 4),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "bars_held": pos.get("bars_held", 0),
                    "mode": pos.get("mode", "?"),
                    "date": ts_dt.strftime("%Y-%m"),
                })
                closed.append(pos)
        for pos in closed: positions.remove(pos)
        
        if positions: continue
        
        atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1])
        if atr == 0: continue
        bb_mid, bb_upper, bb_lower = compute_bb(closes[:i+1])
        kc_mid, kc_upper, kc_lower = compute_keltner(highs[:i+1], lows[:i+1], closes[:i+1])
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
        
        avg_vol = np.mean(volumes[max(0,i-20):i])
        vol_ratio = volumes[i] / avg_vol if avg_vol > 0 else 1
        
        in_squeeze = kc_upper < bb_upper and kc_lower > bb_lower
        
        if in_squeeze_prev and not in_squeeze:
            if price > bb_upper:
                breakout_bar = i; breakout_dir = "LONG"; breakout_extreme = highs[i]
            elif price < bb_lower:
                breakout_bar = i; breakout_dir = "SHORT"; breakout_extreme = lows[i]
            else:
                breakout_bar = None; breakout_dir = None
        
        in_squeeze_prev = in_squeeze
        
        if breakout_bar is None: continue
        if in_squeeze: continue
        bars_since = i - breakout_bar
        if bars_since > 12: breakout_bar = None; continue
        if (i - last_signal_bar) < 4: continue
        
        if version == "v3":
            if bb_width > 0.02: continue
            if price > bb_upper: direction = "LONG"
            elif price < bb_lower: direction = "SHORT"
            else: continue
            if direction == "LONG": sl = price - atr; tp = price + atr * 2
            else: sl = price + atr; tp = price - atr * 2
            mode = "BREAKOUT"
        
        elif version == "v5b":
            if breakout_dir == "LONG":
                if price < bb_upper and closes[i-1] >= bb_upper:
                    direction = "SHORT"; entry = price
                    sl = breakout_extreme + atr * 0.3; tp = bb_mid
                    mode = "FADE"
                else: continue
            elif breakout_dir == "SHORT":
                if price > bb_lower and closes[i-1] <= bb_lower:
                    direction = "LONG"; entry = price
                    sl = breakout_extreme - atr * 0.3; tp = bb_mid
                    mode = "FADE"
                else: continue
            else: continue
            
            risk = abs(entry - sl); reward = abs(entry - tp)
            if risk == 0 or reward / risk < 1.0: filtered += 1; continue
            
            squeeze_quality = max(0, 1.0 - bb_width / 0.03)
            conviction = 0.5 + squeeze_quality * 0.2
            if vol_ratio > 1.5: conviction += 0.1
            conviction = min(conviction, 0.85)
            if conviction < 0.5: filtered += 1; continue
        else: continue
        
        signals += 1; last_signal_bar = i; breakout_bar = None
        positions.append({
            "direction": direction, "entry": entry if version == "v5b" else price,
            "sl": round(sl, 2), "tp": round(tp, 2),
            "size": 10, "leverage": 10, "hold_bars": 8,
            "bars_held": 0, "entry_bar": i, "mode": mode,
        })
    
    # Close remaining
    for pos in positions:
        price = closes[-1]
        pnl = ((price - pos["entry"]) / pos["entry"] * 100) if pos["direction"] == "LONG" else ((pos["entry"] - price) / pos["entry"] * 100)
        pnl -= fee_rate * 100; pnl_dollar = pos["size"] * pnl / 100 * 10
        capital += pnl_dollar
        trades.append({**pos, "exit": round(price, 2), "outcome": "TIMEOUT",
                       "pnl_pct": round(pnl, 4), "pnl_dollar": round(pnl_dollar, 2)})
    
    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    timeouts = [t for t in trades if t["outcome"] == "TIMEOUT"]
    gross_profit = sum(t["pnl_dollar"] for t in trades if t["pnl_dollar"] > 0)
    gross_loss = abs(sum(t["pnl_dollar"] for t in trades if t["pnl_dollar"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    dd = (peak - capital) / peak * 100 if peak > 0 else 0
    
    return {
        "version": version, "capital": round(capital, 2),
        "pnl": round(capital - initial_capital, 2),
        "pnl_pct": round((capital - initial_capital) / initial_capital * 100, 1),
        "max_dd": round(dd, 1), "trades": len(trades),
        "wins": len(wins), "losses": len(losses), "timeouts": len(timeouts),
        "wr": round(len(wins)/len(trades)*100, 1) if trades else 0,
        "pf": round(pf, 2), "signals": signals, "filtered": filtered,
        "avg_win": round(np.mean([t["pnl_pct"] for t in wins]), 4) if wins else 0,
        "avg_loss": round(np.mean([t["pnl_pct"] for t in losses]), 4) if losses else 0,
        "trades_detail": trades,
    }


def main():
    print("=" * 60)
    print("SQUEEZE BREAKOUT: V3 vs V5B — 2.5 YEAR BACKTEST")
    print("=" * 60)
    
    csv_path = os.path.join(BASE, "data", "history", "ETHUSDT_15m.csv")
    if not os.path.exists(csv_path):
        print("❌ No data file. Run data_pipeline.py first.")
        return
    
    candles = load_csv_data(csv_path)
    print(f"📊 Loaded {len(candles)} candles")
    start = datetime.fromtimestamp(candles[0]["ts"]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    end = datetime.fromtimestamp(candles[-1]["ts"]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"   Period: {start} → {end}")
    
    print("\n🔄 Running V3 (current breakout)...")
    v3 = run_backtest(candles, "v3")
    
    print("🔄 Running V5B (fade the failure)...")
    v5b = run_backtest(candles, "v5b")
    
    print(f"\n{'='*60}")
    print(f"{'Metric':<20} {'V3 (breakout)':<15} {'V5B (fade)':<15}")
    print(f"{'-'*50}")
    for m, a, b in [
        ("Capital", f"${v3['capital']}", f"${v5b['capital']}"),
        ("PnL", f"${v3['pnl']}", f"${v5b['pnl']}"),
        ("PnL %", f"{v3['pnl_pct']}%", f"{v5b['pnl_pct']}%"),
        ("Max DD", f"{v3['max_dd']}%", f"{v5b['max_dd']}%"),
        ("Trades", str(v3['trades']), str(v5b['trades'])),
        ("Wins", str(v3['wins']), str(v5b['wins'])),
        ("Losses", str(v3['losses']), str(v5b['losses'])),
        ("Win Rate", f"{v3['wr']}%", f"{v5b['wr']}%"),
        ("Profit Factor", str(v3['pf']), str(v5b['pf'])),
        ("Avg Win", f"{v3['avg_win']:+.2f}%", f"{v5b['avg_win']:+.2f}%"),
        ("Avg Loss", f"{v3['avg_loss']:+.2f}%", f"{v5b['avg_loss']:+.2f}%"),
    ]:
        print(f"  {m:<20} {a:<15} {b:<15}")
    
    # Per-period analysis
    if v5b['trades_detail']:
        print(f"\n📅 V5B Per Period:")
        periods = {}
        for t in v5b['trades_detail']:
            p = t.get("date", "?")
            if p not in periods: periods[p] = {"wins": 0, "losses": 0, "pnl": 0}
            if t["outcome"] == "WIN": periods[p]["wins"] += 1
            elif t["outcome"] == "LOSS": periods[p]["losses"] += 1
            periods[p]["pnl"] += t["pnl_dollar"]
        
        for p in sorted(periods.keys()):
            d = periods[p]
            total = d["wins"] + d["losses"]
            wr = d["wins"] / total * 100 if total > 0 else 0
            emoji = "✅" if d["pnl"] > 0 else "❌"
            print(f"  {emoji} {p}: {total}T {d['wins']}W/{d['losses']}L WR={wr:.0f}% PnL=${d['pnl']:.2f}")
    
    # Sample trades
    if v5b['trades_detail']:
        print(f"\n📋 V5B Sample Trades (first 20):")
        for t in v5b['trades_detail'][:20]:
            emoji = "✅" if t['outcome'] == 'WIN' else "❌" if t['outcome'] == 'LOSS' else "⏱️"
            print(f"  {emoji} {t['direction']:5} ${t['entry']:.0f} -> ${t['exit']:.0f} "
                  f"pnl={t['pnl_pct']:+.2f}% hold={t.get('bars_held',0)}")
    
    # Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candles": len(candles), "period": f"{start} → {end}",
        "v3": {k: v for k, v in v3.items() if k != "trades_detail"},
        "v5b": {k: v for k, v in v5b.items() if k != "trades_detail"},
        "v5b_trades": v5b["trades_detail"],
    }
    out_path = os.path.join(BASE, "data", "5agent_backtest", "squeeze_2year_backtest.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Saved: {out_path}")


if __name__ == "__main__":
    main()
