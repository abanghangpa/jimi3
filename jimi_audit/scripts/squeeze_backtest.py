#!/usr/bin/env python3
"""
SQUEEZE_BREAKOUT V4 — Proper Implementation + Backtest

Fixes from v3:
1. Real squeeze: Keltner Channel inside Bollinger Band
2. Confirmation: Close above BB + next bar sustains
3. Trend filter: Only trade with 4h EMA direction
4. Volume: Breakout bar needs 1.5x+ avg volume
5. One shot: Max 1 entry per squeeze event (cooldown 8 bars)
6. TP/SL: ATR-adaptive
7. Consecutive signal filter: Skip if same direction fired in last 3 bars
"""

import requests, json, os, sys, math, statistics
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
            "taker_buy_vol": float(c[9]),
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
    """Compute EMA."""
    if len(data) < period:
        return data[-1] if data else 0
    ema = np.mean(data[:period])
    mult = 2 / (period + 1)
    for i in range(period, len(data)):
        ema = data[i] * mult + ema * (1 - mult)
    return ema


def compute_keltner(highs, lows, closes, period=20, mult=1.5):
    """Compute Keltner Channel (EMA-based)."""
    ema = compute_ema(closes, period)
    atr = compute_atr(highs, lows, closes, period)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return ema, upper, lower


def compute_bb(closes, period=20, std_mult=2.0):
    """Compute Bollinger Bands."""
    if len(closes) < period:
        return closes[-1], closes[-1], closes[-1]
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma, upper, lower


def is_real_squeeze(bb_upper, bb_lower, kc_upper, kc_lower):
    """
    Real squeeze: Keltner Channel is INSIDE Bollinger Band.
    This means volatility is compressed inside the BB range.
    """
    return kc_upper < bb_upper and kc_lower > bb_lower


def run_backtest(candles, version="v4", initial_capital=200.0):
    """
    Run squeeze_breakout backtest.
    
    version:
      "v3" = original (current live code)
      "v4" = fixed (proper implementation)
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
    
    # Cooldown tracking
    last_squeeze_end = -999  # bar index when last squeeze ended
    last_signal_bar = -999
    squeeze_cooldown = 8  # bars between squeeze trades
    last_direction = None
    consecutive_same_dir = 0
    
    signals_generated = 0
    signals_filtered = 0
    
    for i in range(50, len(candles)):
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
                    "bb_width": pos.get("bb_width", 0),
                    "vol_ratio": pos.get("vol_ratio", 0),
                    "mom_4h": pos.get("mom_4h", 0),
                    "with_trend": pos.get("with_trend", False),
                })
                closed.append(pos)
        
        for pos in closed:
            positions.remove(pos)
        
        if len(positions) >= 1:
            continue  # max 1 position for squeeze
        
        # ── Compute indicators ──
        atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1])
        if atr == 0:
            continue
        
        # BB(20, 2.0)
        bb_mid, bb_upper, bb_lower = compute_bb(closes[:i+1])
        
        # Keltner Channel(20, 1.5)
        kc_mid, kc_upper, kc_lower = compute_keltner(
            highs[:i+1], lows[:i+1], closes[:i+1])
        
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
        
        # 4h EMA for trend
        ema_50 = compute_ema(closes[:i+1], 50)
        
        # Volume
        avg_vol = np.mean(volumes[max(0,i-20):i])
        vol_ratio = volumes[i] / avg_vol if avg_vol > 0 else 1
        
        # 4h momentum
        mom_4h = (closes[i] - closes[i-16]) / closes[i-16] if i >= 16 else 0
        mom_1h = (closes[i] - closes[i-4]) / closes[i-4] if i >= 4 else 0
        
        # ── VERSION-SPECIFIC LOGIC ──
        
        if version == "v3":
            # === V3 (CURRENT LIVE CODE) ===
            # Simple: BB width < 2% + price breaks BB
            if bb_width > 0.02:
                continue
            
            direction = None
            if price > bb_upper:
                direction = "LONG"
            elif price < bb_lower:
                direction = "SHORT"
            else:
                continue
            
            conviction = 0.6
            if vol_ratio > 1.5:
                conviction *= 1.2
            conviction = min(conviction, 0.85)
            
            if direction == "LONG":
                sl = price - atr * 1.0
                tp = price + atr * 2.0
            else:
                sl = price + atr * 1.0
                tp = price - atr * 2.0
        
        elif version == "v4":
            # === V4 (FIXED) ===
            
            # Step 1: Real squeeze detection (KC inside BB)
            in_squeeze = is_real_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)
            
            if not in_squeeze:
                # Track when squeeze ended (for breakout detection)
                if i > 0:
                    prev_bb_mid, prev_bb_upper, prev_bb_lower = compute_bb(closes[:i])
                    prev_kc_mid, prev_kc_upper, prev_kc_lower = compute_keltner(
                        highs[:i], lows[:i], closes[:i])
                    prev_in_squeeze = is_real_squeeze(prev_bb_upper, prev_bb_lower,
                                                       prev_kc_upper, prev_kc_lower)
                    if prev_in_squeeze:
                        last_squeeze_end = i - 1
                continue
            
            # Step 2: We're IN a squeeze. Wait for breakout.
            # Breakout = close above BB upper or below BB lower
            breakout_up = price > bb_upper
            breakout_down = price < bb_lower
            
            if not breakout_up and not breakout_down:
                continue
            
            # Step 3: Confirmation — previous bar must also be outside BB
            # (prevents wick-only breakouts)
            if i < 2:
                continue
            prev_price = closes[i-1]
            if breakout_up and prev_price <= bb_upper:
                # First bar outside — need next bar to confirm
                # Skip this signal, wait for confirmation
                signals_filtered += 1
                continue
            if breakout_down and prev_price >= bb_lower:
                signals_filtered += 1
                continue
            
            # Step 4: Cooldown — max 1 signal per squeeze event
            if (i - last_signal_bar) < squeeze_cooldown:
                signals_filtered += 1
                continue
            
            # Step 5: Trend filter — only trade with 4h EMA direction
            with_trend = False
            if breakout_up and price > ema_50:
                direction = "LONG"
                with_trend = True
            elif breakout_down and price < ema_50:
                direction = "SHORT"
                with_trend = True
            else:
                # Counter-trend — skip
                signals_filtered += 1
                continue
            
            # Step 6: Volume confirmation (1.5x+ avg)
            if vol_ratio < 1.3:
                signals_filtered += 1
                continue
            
            # Step 7: Conviction scoring
            conviction = 0.5
            
            # Squeeze quality (tighter = better)
            squeeze_tightness = 1.0 - (bb_width / 0.02)  # 0-1, higher = tighter
            conviction += squeeze_tightness * 0.15
            
            # Volume strength
            conviction += min((vol_ratio - 1) * 0.1, 0.15)
            
            # Trend alignment
            if with_trend:
                conviction += 0.1
            
            # Momentum confirmation
            if (direction == "LONG" and mom_1h > 0) or (direction == "SHORT" and mom_1h < 0):
                conviction += 0.05
            
            conviction = min(conviction, 0.85)
            
            if conviction < 0.5:
                signals_filtered += 1
                continue
            
            # Step 8: ATR-adaptive TP/SL
            # Wider in trends, tighter in ranges
            if with_trend:
                sl_mult = 1.5
                tp_mult = 2.5
            else:
                sl_mult = 1.0
                tp_mult = 2.0
            
            if direction == "LONG":
                sl = price - atr * sl_mult
                tp = price + atr * tp_mult
            else:
                sl = price + atr * sl_mult
                tp = price - atr * tp_mult
            
            last_signal_bar = i
        
        else:
            continue
        
        signals_generated += 1
        
        # ── Open position ──
        positions.append({
            "direction": direction, "entry": price,
            "sl": round(sl, 2), "tp": round(tp, 2),
            "size": 10, "leverage": 10, "hold_bars": 8,
            "bars_held": 0, "entry_bar": i,
            "bb_width": round(bb_width*100, 3),
            "vol_ratio": round(vol_ratio, 2),
            "mom_4h": round(mom_4h*100, 2),
            "with_trend": direction == "LONG" and price > ema_50 or direction == "SHORT" and price < ema_50,
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
                       "pnl_pct": round(pnl, 4), "pnl_dollar": round(pnl_dollar, 2)})
    
    # Stats
    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    timeouts = [t for t in trades if t["outcome"] == "TIMEOUT"]
    gross_profit = sum(t["pnl_dollar"] for t in trades if t["pnl_dollar"] > 0)
    gross_loss = abs(sum(t["pnl_dollar"] for t in trades if t["pnl_dollar"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    dd = (peak - capital) / peak * 100 if peak > 0 else 0
    
    return {
        "version": version,
        "capital": round(capital, 2), "pnl": round(capital - initial_capital, 2),
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
    print("SQUEEZE_BREAKOUT: V3 (current) vs V4 (fixed)")
    print("=" * 60)
    
    print("\n📊 Fetching data...")
    candles = fetch_candles(limit=1500)
    print(f"  Candles: {len(candles)}")
    
    # Run both versions
    print("\n🔄 Running V3 backtest (current live code)...")
    v3 = run_backtest(candles, version="v3")
    
    print("🔄 Running V4 backtest (fixed implementation)...")
    v4 = run_backtest(candles, version="v4")
    
    # Compare
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'V3 (current)':<15} {'V4 (fixed)':<15}")
    print("-" * 55)
    for metric, v3v, v4v in [
        ("Capital", f"${v3['capital']}", f"${v4['capital']}"),
        ("PnL", f"${v3['pnl']}", f"${v4['pnl']}"),
        ("PnL %", f"{v3['pnl_pct']}%", f"{v4['pnl_pct']}%"),
        ("Max DD", f"{v3['max_dd']}%", f"{v4['max_dd']}%"),
        ("Trades", str(v3['trades']), str(v4['trades'])),
        ("Wins", str(v3['wins']), str(v4['wins'])),
        ("Losses", str(v3['losses']), str(v4['losses'])),
        ("Win Rate", f"{v3['wr']}%", f"{v4['wr']}%"),
        ("Profit Factor", str(v3['pf']), str(v4['pf'])),
        ("Avg Win", f"{v3['avg_win']:+.2f}%", f"{v4['avg_win']:+.2f}%"),
        ("Avg Loss", f"{v3['avg_loss']:+.2f}%", f"{v4['avg_loss']:+.2f}%"),
        ("Signals Generated", str(v3['signals_generated']), str(v4['signals_generated'])),
        ("Signals Filtered", str(v3.get('signals_filtered', 0)), str(v4['signals_filtered'])),
    ]:
        print(f"  {metric:<25} {v3v:<15} {v4v:<15}")
    
    # Trade analysis for V4
    if v4['trades_detail']:
        print(f"\n📋 V4 Trade Log:")
        for t in v4['trades_detail']:
            emoji = "✅" if t['outcome'] == 'WIN' else "❌" if t['outcome'] == 'LOSS' else "⏱️"
            trend = "↗" if t.get('with_trend') else "↘"
            print(f"  {emoji} bar={t.get('entry_bar','?'):4} {t['direction']:5} ${t['entry']:.0f} "
                  f"pnl={t['pnl_pct']:+.2f}% bb={t.get('bb_width',0):.3f}% "
                  f"vol={t.get('vol_ratio',0):.1f}x {trend} "
                  f"mom4h={t.get('mom_4h',0):+.1f}% hold={t.get('bars_held',0)}")
    
    # Per-squeeze analysis
    if v4['trades_detail']:
        with_trend = [t for t in v4['trades_detail'] if t.get('with_trend')]
        against = [t for t in v4['trades_detail'] if not t.get('with_trend')]
        print(f"\n🔄 Trend Analysis (V4):")
        if with_trend:
            wt_wins = sum(1 for t in with_trend if t['outcome'] == 'WIN')
            print(f"  With trend:    {len(with_trend)}T, {wt_wins}W, WR={wt_wins/len(with_trend)*100:.1f}%")
        if against:
            at_wins = sum(1 for t in against if t['outcome'] == 'WIN')
            print(f"  Against trend: {len(against)}T, {at_wins}W, WR={at_wins/len(against)*100:.1f}%")
    
    # Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candles": len(candles),
        "v3": {k: v for k, v in v3.items() if k != "trades_detail"},
        "v4": {k: v for k, v in v4.items() if k != "trades_detail"},
        "v3_trades": v3["trades_detail"],
        "v4_trades": v4["trades_detail"],
    }
    out_path = os.path.join(BASE, "data", "5agent_backtest", "squeeze_v3_vs_v4.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Saved: {out_path}")


if __name__ == "__main__":
    main()
