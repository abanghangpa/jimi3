#!/usr/bin/env python3
"""
SQUEEZE_BREAKOUT V5 — Trade the FAILURE, not the breakout.

Data shows:
- 70% of BB squeeze breakouts FAIL (mean-revert back into range)
- 30% of breakouts SUCCEED (continue into trend)
- Most markets range 70% of the time

Strategy:
- FADE breakouts in ranging markets (high WR, capture snapback)
- FOLLOW breakouts in trending markets (low WR, but big wins)
- Use regime detection to determine which mode

Target: 75% WR, PF 2.0
"""

import requests, json, os
from datetime import datetime, timezone
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fetch_candles(symbol="ETHUSDT", interval="15m", limit=1500):
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    r.raise_for_status()
    candles = []
    for c in r.json():
        candles.append({
            "ts": c[0], "open": float(c[1]), "high": float(c[2]),
            "low": float(c[3]), "close": float(c[4]), "volume": float(c[5]),
        })
    return candles


def compute_atr(highs, lows, closes, period=14):
    trs = []
    for i in range(1, len(highs)):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        trs.append(tr)
    if len(trs) < period:
        return sum(trs)/len(trs) if trs else 0
    return sum(trs[-period:]) / period


def compute_ema(data, period):
    if len(data) < period:
        return data[-1] if data else 0
    ema = np.mean(data[:period])
    mult = 2 / (period + 1)
    for i in range(period, len(data)):
        ema = data[i] * mult + ema * (1 - mult)
    return ema


def compute_bb(closes, period=20, std_mult=2.0):
    if len(closes) < period:
        return closes[-1], closes[-1], closes[-1]
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma, upper, lower


def compute_keltner(highs, lows, closes, period=20, mult=1.5):
    ema = compute_ema(closes, period)
    atr = compute_atr(highs, lows, closes, period)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return ema, upper, lower


def is_real_squeeze(bb_upper, bb_lower, kc_upper, kc_lower):
    return kc_upper < bb_upper and kc_lower > bb_lower


def detect_regime(closes, i, ema_fast=20, ema_slow=50):
    """
    Detect if market is trending or ranging.
    
    Trending: Fast EMA > Slow EMA (or vice versa) with clear separation
    Ranging: Fast EMA ≈ Slow EMA (converged)
    """
    if i < ema_slow + 10:
        return "RANGING", 0.5
    
    fast = compute_ema(closes[:i+1], ema_fast)
    slow = compute_ema(closes[:i+1], ema_slow)
    
    separation = (fast - slow) / slow * 100  # % difference
    
    # ADX-like: measure trend strength via directional movement
    # Simple version: use EMA separation as proxy
    if abs(separation) > 0.5:
        if separation > 0:
            return "BULL", min(abs(separation) / 1.0, 0.9)
        else:
            return "BEAR", min(abs(separation) / 1.0, 0.9)
    elif abs(separation) > 0.2:
        if separation > 0:
            return "MILDLY_BULL", 0.5
        else:
            return "MILDLY_BEAR", 0.5
    else:
        return "RANGING", 0.5


def run_backtest(candles, version="v5", initial_capital=200.0):
    """
    Run squeeze_breakout backtest.
    
    v3: Current live (fire on every breakout)
    v5: Trade the failure (fade in range, follow in trend)
    """
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]
    
    capital = initial_capital
    peak = initial_capital
    positions = []
    trades = []
    fee_rate = 0.0004
    
    last_signal_bar = -999
    signal_cooldown = 6  # bars between signals
    
    signals_generated = 0
    signals_filtered = 0
    
    # Track breakout state
    breakout_bar = None
    breakout_dir = None
    breakout_price = None
    in_squeeze = False
    
    for i in range(60, len(candles)):
        price = closes[i]
        
        # ── Check exits ──
        closed = []
        for pos in positions:
            pos["bars_held"] = pos.get("bars_held", 0) + 1
            exited = False
            outcome = "TIMEOUT"
            exit_p = price
            
            if pos["direction"] == "LONG":
                if lows[i] <= pos["sl"]:
                    outcome = "LOSS"; exit_p = pos["sl"]; exited = True
                elif highs[i] >= pos["tp"]:
                    outcome = "WIN"; exit_p = pos["tp"]; exited = True
            else:
                if highs[i] >= pos["sl"]:
                    outcome = "LOSS"; exit_p = pos["sl"]; exited = True
                elif lows[i] <= pos["tp"]:
                    outcome = "WIN"; exit_p = pos["tp"]; exited = True
            
            if not exited and pos.get("bars_held", 0) >= pos.get("hold_bars", 8):
                exited = True; outcome = "TIMEOUT"; exit_p = price
            
            if exited:
                if pos["direction"] == "LONG":
                    pnl = (exit_p - pos["entry"]) / pos["entry"] * 100
                else:
                    pnl = (pos["entry"] - exit_p) / pos["entry"] * 100
                pnl -= fee_rate * 100
                pnl_dollar = pos["size"] * pnl / 100 * pos.get("leverage", 10)
                capital += pnl_dollar
                peak = max(peak, capital)
                trades.append({
                    "bar": pos["entry_bar"], "direction": pos["direction"],
                    "entry": pos["entry"], "exit": round(exit_p, 2),
                    "outcome": outcome, "pnl_pct": round(pnl, 4),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "bars_held": pos.get("bars_held", 0),
                    "mode": pos.get("mode", "?"),
                    "regime": pos.get("regime", "?"),
                    "bb_width": pos.get("bb_width", 0),
                })
                closed.append(pos)
        
        for pos in closed:
            positions.remove(pos)
        
        if len(positions) >= 1:
            continue
        
        # ── Compute indicators ──
        atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1])
        if atr == 0:
            continue
        
        bb_mid, bb_upper, bb_lower = compute_bb(closes[:i+1])
        kc_mid, kc_upper, kc_lower = compute_keltner(highs[:i+1], lows[:i+1], closes[:i+1])
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
        
        # Volume
        avg_vol = np.mean(volumes[max(0,i-20):i])
        vol_ratio = volumes[i] / avg_vol if avg_vol > 0 else 1
        
        # Regime
        regime, regime_conf = detect_regime(closes, i)
        
        # EMA for trend
        ema_20 = compute_ema(closes[:i+1], 20)
        ema_50 = compute_ema(closes[:i+1], 50)
        
        # Momentum
        mom_1h = (closes[i] - closes[i-4]) / closes[i-4] if i >= 4 else 0
        mom_4h = (closes[i] - closes[i-16]) / closes[i-16] if i >= 16 else 0
        
        # ── Check squeeze state ──
        was_in_squeeze = in_squeeze
        in_squeeze = is_real_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)
        
        # ── Version-specific logic ──
        
        if version == "v3":
            # === V3: Current live (fire on every breakout) ===
            if bb_width > 0.02:
                continue
            
            direction = None
            if price > bb_upper:
                direction = "LONG"
            elif price < bb_lower:
                direction = "SHORT"
            else:
                continue
            
            # Cooldown
            if (i - last_signal_bar) < 2:
                continue
            
            conviction = 0.6
            if direction == "LONG":
                sl = price - atr * 1.0
                tp = price + atr * 2.0
            else:
                sl = price + atr * 1.0
                tp = price - atr * 2.0
            
            last_signal_bar = i
            mode = "BREAKOUT"
        
        elif version == "v5":
            # === V5: Trade the FAILURE ===
            
            # Track breakouts
            if not in_squeeze and was_in_squeeze:
                # Squeeze just ended — breakout happened
                if price > bb_upper:
                    breakout_bar = i
                    breakout_dir = "LONG"
                    breakout_price = price
                elif price < bb_lower:
                    breakout_bar = i
                    breakout_dir = "SHORT"
                    breakout_price = price
                continue
            
            if in_squeeze:
                # Still in squeeze — wait
                breakout_bar = None
                breakout_dir = None
                continue
            
            # We're outside squeeze. Was there a recent breakout?
            if breakout_bar is None:
                continue
            
            bars_since_breakout = i - breakout_bar
            if bars_since_breakout > 16:
                # Too old — reset
                breakout_bar = None
                breakout_dir = None
                continue
            
            # Cooldown
            if (i - last_signal_bar) < signal_cooldown:
                continue
            
            # ── MODE 1: FADE the breakout (ranging market) ──
            # If market is ranging and breakout happened, fade it
            if regime in ("RANGING",) and regime_conf < 0.6:
                # Check if price is starting to fail (coming back toward BB)
                if breakout_dir == "LONG":
                    # Price broke above BB, now coming back down
                    if price < bb_upper and price > bb_mid:
                        # Fading: SHORT the failure back to BB midline
                        direction = "SHORT"
                        entry = price
                        sl = breakout_price + atr * 0.5  # above the breakout high
                        tp = bb_mid  # target: BB midline (where it should revert)
                        
                        # Only if RR is good
                        risk = abs(entry - sl)
                        reward = abs(entry - tp)
                        if risk > 0 and reward / risk < 1.5:
                            signals_filtered += 1
                            continue
                        
                        conviction = 0.6 + regime_conf * 0.2
                        mode = "FADE"
                    else:
                        continue
                
                elif breakout_dir == "SHORT":
                    # Price broke below BB, now coming back up
                    if price > bb_lower and price < bb_mid:
                        direction = "LONG"
                        entry = price
                        sl = breakout_price - atr * 0.5
                        tp = bb_mid
                        
                        risk = abs(entry - sl)
                        reward = abs(entry - tp)
                        if risk > 0 and reward / risk < 1.5:
                            signals_filtered += 1
                            continue
                        
                        conviction = 0.6 + regime_conf * 0.2
                        mode = "FADE"
                    else:
                        continue
                else:
                    continue
            
            # ── MODE 2: FOLLOW the breakout (trending market) ──
            elif regime in ("BULL", "BEAR") and regime_conf > 0.5:
                if breakout_dir == "LONG" and regime == "BULL":
                    # Trend confirmed — enter on pullback to BB upper
                    if price > bb_upper * 0.998 and price < bb_upper * 1.01:
                        direction = "LONG"
                        entry = price
                        sl = bb_mid  # stop at BB midline
                        tp = price + atr * 3.0  # bigger target in trend
                        
                        conviction = 0.5 + regime_conf * 0.3
                        mode = "FOLLOW"
                    else:
                        continue
                
                elif breakout_dir == "SHORT" and regime == "BEAR":
                    if price < bb_lower * 1.002 and price > bb_lower * 0.99:
                        direction = "SHORT"
                        entry = price
                        sl = bb_mid
                        tp = price - atr * 3.0
                        
                        conviction = 0.5 + regime_conf * 0.3
                        mode = "FOLLOW"
                    else:
                        continue
                else:
                    signals_filtered += 1
                    continue
            
            else:
                # Mildly trending — skip
                signals_filtered += 1
                continue
            
            # Volume confirmation
            if vol_ratio < 1.2:
                conviction *= 0.8
            
            if conviction < 0.5:
                signals_filtered += 1
                continue
            
            last_signal_bar = i
            signals_generated += 1
        
        else:
            continue
        
        # ── Open position ──
        positions.append({
            "direction": direction, "entry": entry if version == "v5" else price,
            "sl": round(sl, 2), "tp": round(tp, 2),
            "size": 10, "leverage": 10,
            "hold_bars": 8 if mode == "FADE" else 12,
            "bars_held": 0, "entry_bar": i,
            "mode": mode, "regime": regime,
            "bb_width": round(bb_width*100, 3),
        })
    
    # Close remaining
    for pos in positions:
        price = closes[-1]
        if pos["direction"] == "LONG":
            pnl = (price - pos["entry"]) / pos["entry"] * 100
        else:
            pnl = (pos["entry"] - price) / pos["entry"] * 100
        pnl -= fee_rate * 100
        pnl_dollar = pos["size"] * pnl / 100 * pos.get("leverage", 10)
        capital += pnl_dollar
        trades.append({"direction": pos["direction"], "entry": pos["entry"],
                       "exit": round(price, 2), "outcome": "TIMEOUT",
                       "pnl_pct": round(pnl, 4), "pnl_dollar": round(pnl_dollar, 2),
                       "mode": pos.get("mode", "?"), "regime": pos.get("regime", "?")})
    
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
        "pf": round(pf, 2),
        "avg_win": round(np.mean([t["pnl_pct"] for t in wins]), 4) if wins else 0,
        "avg_loss": round(np.mean([t["pnl_pct"] for t in losses]), 4) if losses else 0,
        "signals_generated": signals_generated,
        "signals_filtered": signals_filtered,
        "trades_detail": trades,
    }


def main():
    print("=" * 60)
    print("SQUEEZE_BREAKOUT: V3 vs V5 (fade the failure)")
    print("=" * 60)
    
    print("\n📊 Fetching data...")
    candles = fetch_candles(limit=1500)
    print(f"  Candles: {len(candles)}")
    
    print("\n🔄 Running V3 (current)...")
    v3 = run_backtest(candles, version="v3")
    
    print("🔄 Running V5 (fade the failure)...")
    v5 = run_backtest(candles, version="v5")
    
    # Compare
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'V3 (current)':<15} {'V5 (fade)':<15}")
    print("-" * 55)
    for metric, v3v, v5v in [
        ("Capital", f"${v3['capital']}", f"${v5['capital']}"),
        ("PnL", f"${v3['pnl']}", f"${v5['pnl']}"),
        ("PnL %", f"{v3['pnl_pct']}%", f"{v5['pnl_pct']}%"),
        ("Max DD", f"{v3['max_dd']}%", f"{v5['max_dd']}%"),
        ("Trades", str(v3['trades']), str(v5['trades'])),
        ("Wins", str(v3['wins']), str(v5['wins'])),
        ("Losses", str(v3['losses']), str(v5['losses'])),
        ("Win Rate", f"{v3['wr']}%", f"{v5['wr']}%"),
        ("Profit Factor", str(v3['pf']), str(v5['pf'])),
        ("Avg Win", f"{v3['avg_win']:+.2f}%", f"{v5['avg_win']:+.2f}%"),
        ("Avg Loss", f"{v3['avg_loss']:+.2f}%", f"{v5['avg_loss']:+.2f}%"),
    ]:
        print(f"  {metric:<25} {v3v:<15} {v5v:<15}")
    
    # V5 trade log
    if v5['trades_detail']:
        print(f"\n📋 V5 Trade Log:")
        for t in v5['trades_detail']:
            emoji = "✅" if t['outcome'] == 'WIN' else "❌" if t['outcome'] == 'LOSS' else "⏱️"
            print(f"  {emoji} {t['direction']:5} ${t['entry']:.0f} -> ${t['exit']:.0f} "
                  f"pnl={t['pnl_pct']:+.2f}% mode={t.get('mode','?'):6} "
                  f"regime={t.get('regime','?'):12} hold={t.get('bars_held',0)}")
    
    # Per-mode analysis
    if v5['trades_detail']:
        print(f"\n📊 Per Mode (V5):")
        for mode in sorted(set(t.get("mode", "?") for t in v5['trades_detail'])):
            mt = [t for t in v5['trades_detail'] if t.get("mode") == mode]
            mw = sum(1 for t in mt if t['outcome'] == 'WIN')
            mp = sum(t['pnl_dollar'] for t in mt)
            mwr = mw / len(mt) * 100 if mt else 0
            print(f"  {mode}: {len(mt)}T {mw}W WR={mwr:.0f}% PnL=${mp:.2f}")
    
    # Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candles": len(candles),
        "v3": {k: v for k, v in v3.items() if k != "trades_detail"},
        "v5": {k: v for k, v in v5.items() if k != "trades_detail"},
        "v3_trades": v3["trades_detail"],
        "v5_trades": v5["trades_detail"],
    }
    out_path = os.path.join(BASE, "data", "5agent_backtest", "squeeze_v5_fade.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Saved: {out_path}")


if __name__ == "__main__":
    main()
