#!/usr/bin/env python3
"""
SQUEEZE V5B: Trade the FAILURE, not the breakout.

Core insight from data: 70% of BB squeeze breakouts FAIL.
Strategy: When breakout fails back inside BB, fade it.
"""
import requests, json, os
from datetime import datetime, timezone
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def fetch_candles(symbol="ETHUSDT", interval="15m", limit=1500):
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    r.raise_for_status()
    return [{"ts": c[0], "open": float(c[1]), "high": float(c[2]),
             "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])} for c in r.json()]

def compute_atr(highs, lows, closes, period=14):
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(highs))]
    return sum(trs[-period:]) / period if len(trs) >= period else (sum(trs)/len(trs) if trs else 0)

def compute_bb(closes, period=20, std_mult=2.0):
    if len(closes) < period: return closes[-1], closes[-1], closes[-1]
    sma = np.mean(closes[-period:]); std = np.std(closes[-period:])
    return sma, sma + std_mult * std, sma - std_mult * std

def compute_ema(data, period):
    if len(data) < period: return data[-1] if data else 0
    ema = np.mean(data[:period])
    mult = 2 / (period + 1)
    for i in range(period, len(data)):
        ema = data[i] * mult + ema * (1 - mult)
    return ema

def compute_keltner(highs, lows, closes, period=20, mult=1.5):
    ema = compute_ema(closes, period)
    atr = compute_atr(highs, lows, closes, period)
    return ema, ema + mult * atr, ema - mult * atr

def run_backtest(candles, version="v5b", initial_capital=200.0):
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
        
        # ── Exits ──
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
                trades.append({**pos, "exit": round(exit_p, 2), "outcome": outcome,
                               "pnl_pct": round(pnl, 4), "pnl_dollar": round(pnl_dollar, 2)})
                closed.append(pos)
        for pos in closed: positions.remove(pos)
        
        if positions: continue
        
        # ── Indicators ──
        atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1])
        if atr == 0: continue
        bb_mid, bb_upper, bb_lower = compute_bb(closes[:i+1])
        kc_mid, kc_upper, kc_lower = compute_keltner(highs[:i+1], lows[:i+1], closes[:i+1])
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
        
        avg_vol = np.mean(volumes[max(0,i-20):i])
        vol_ratio = volumes[i] / avg_vol if avg_vol > 0 else 1
        
        in_squeeze = kc_upper < bb_upper and kc_lower > bb_lower
        
        # ── Track breakout ──
        if in_squeeze_prev and not in_squeeze:
            if price > bb_upper:
                breakout_bar = i; breakout_dir = "LONG"
                breakout_extreme = highs[i]
            elif price < bb_lower:
                breakout_bar = i; breakout_dir = "SHORT"
                breakout_extreme = lows[i]
            else:
                breakout_bar = None; breakout_dir = None
        
        in_squeeze_prev = in_squeeze
        
        if breakout_bar is None: continue
        if in_squeeze: continue
        
        bars_since = i - breakout_bar
        if bars_since > 12: breakout_bar = None; continue
        if (i - last_signal_bar) < 4: continue
        
        # ── VERSION-SPECIFIC LOGIC ──
        
        if version == "v3":
            # V3: Current live — fire on breakout
            if bb_width > 0.02: continue
            if price > bb_upper: direction = "LONG"
            elif price < bb_lower: direction = "SHORT"
            else: continue
            if direction == "LONG":
                sl = price - atr; tp = price + atr * 2
            else:
                sl = price + atr; tp = price - atr * 2
            conviction = 0.6
            mode = "BREAKOUT"
        
        elif version == "v5b":
            # V5B: Fade the failure
            # After breakout, wait for price to FAIL back inside BB
            # Entry: close back below BB upper (for LONG breakout) or above BB lower (for SHORT breakout)
            
            if breakout_dir == "LONG":
                # LONG breakout — price was above BB upper, now failing back
                # Signal: close below BB upper AND previous close was above BB upper
                if price < bb_upper and closes[i-1] >= bb_upper:
                    direction = "SHORT"
                    entry = price
                    sl = breakout_extreme + atr * 0.3  # above breakout high
                    tp = bb_mid  # target: BB midline
                    mode = "FADE"
                else:
                    continue
            
            elif breakout_dir == "SHORT":
                # SHORT breakout — price was below BB lower, now failing back
                if price > bb_lower and closes[i-1] <= bb_lower:
                    direction = "LONG"
                    entry = price
                    sl = breakout_extreme - atr * 0.3  # below breakout low
                    tp = bb_mid
                    mode = "FADE"
                else:
                    continue
            else:
                continue
            
            # Check RR
            risk = abs(entry - sl)
            reward = abs(entry - tp)
            if risk == 0 or reward / risk < 1.0:
                filtered += 1; continue
            
            # Conviction
            squeeze_quality = max(0, 1.0 - bb_width / 0.03)
            conviction = 0.5 + squeeze_quality * 0.2
            if vol_ratio > 1.5: conviction += 0.1
            conviction = min(conviction, 0.85)
            
            if conviction < 0.5: filtered += 1; continue
        
        elif version == "v5c":
            # V5C: Fade + Follow hybrid
            # FADE in ranging, FOLLOW in trending
            
            ema_20 = np.mean(closes[max(0,i-19):i+1])
            ema_50 = np.mean(closes[max(0,i-49):i+1])
            trend_strength = (ema_20 - ema_50) / ema_50 * 100 if ema_50 > 0 else 0
            
            if breakout_dir == "LONG":
                if price < bb_upper and closes[i-1] >= bb_upper:
                    # Failure confirmed
                    if abs(trend_strength) < 0.3:
                        # RANGING — FADE
                        direction = "SHORT"
                        entry = price
                        sl = breakout_extreme + atr * 0.3
                        tp = bb_mid
                        mode = "FADE"
                        conviction = 0.6
                    elif trend_strength > 0.5:
                        # BULL TREND — FOLLOW (enter on pullback in trend direction)
                        if price > bb_mid and price < bb_upper * 1.005:
                            direction = "LONG"
                            entry = price
                            sl = bb_mid - atr * 0.5
                            tp = bb_upper + atr * 2
                            mode = "FOLLOW"
                            conviction = 0.55
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            
            elif breakout_dir == "SHORT":
                if price > bb_lower and closes[i-1] <= bb_lower:
                    if abs(trend_strength) < 0.3:
                        direction = "LONG"
                        entry = price
                        sl = breakout_extreme - atr * 0.3
                        tp = bb_mid
                        mode = "FADE"
                        conviction = 0.6
                    elif trend_strength < -0.5:
                        if price < bb_mid and price > bb_lower * 0.995:
                            direction = "SHORT"
                            entry = price
                            sl = bb_mid + atr * 0.5
                            tp = bb_lower - atr * 2
                            mode = "FOLLOW"
                            conviction = 0.55
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                continue
            
            risk = abs(entry - sl)
            reward = abs(entry - tp)
            if risk == 0 or reward / risk < 1.0:
                filtered += 1; continue
            
            if conviction < 0.5: filtered += 1; continue
        
        else:
            continue
        
        signals += 1
        last_signal_bar = i
        breakout_bar = None
        
        positions.append({
            "direction": direction, "entry": entry if version != "v3" else price,
            "sl": round(sl, 2), "tp": round(tp, 2),
            "size": 10, "leverage": 10, "hold_bars": 8,
            "bars_held": 0, "entry_bar": i,
            "mode": mode, "bb_width": round(bb_width*100, 3),
        })
    
    # Close remaining
    for pos in positions:
        price = closes[-1]
        pnl = ((price - pos["entry"]) / pos["entry"] * 100) if pos["direction"] == "LONG" else ((pos["entry"] - price) / pos["entry"] * 100)
        pnl -= fee_rate * 100
        pnl_dollar = pos["size"] * pnl / 100 * 10
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
    print("SQUEEZE BREAKOUT: V3 vs V5B vs V5C")
    print("=" * 60)
    
    candles = fetch_candles(limit=1500)
    print(f"📊 Candles: {len(candles)}")
    
    v3 = run_backtest(candles, "v3")
    v5b = run_backtest(candles, "v5b")
    v5c = run_backtest(candles, "v5c")
    
    print(f"\n{'Metric':<18} {'V3 (breakout)':<14} {'V5B (fade)':<14} {'V5C (hybrid)':<14}")
    print("-" * 60)
    for m, a, b, c in [
        ("Trades", str(v3['trades']), str(v5b['trades']), str(v5c['trades'])),
        ("Wins", str(v3['wins']), str(v5b['wins']), str(v5c['wins'])),
        ("Losses", str(v3['losses']), str(v5b['losses']), str(v5c['losses'])),
        ("Win Rate", f"{v3['wr']}%", f"{v5b['wr']}%", f"{v5c['wr']}%"),
        ("PF", str(v3['pf']), str(v5b['pf']), str(v5c['pf'])),
        ("PnL %", f"{v3['pnl_pct']}%", f"{v5b['pnl_pct']}%", f"{v5c['pnl_pct']}%"),
        ("Max DD", f"{v3['max_dd']}%", f"{v5b['max_dd']}%", f"{v5c['max_dd']}%"),
    ]:
        print(f"  {m:<18} {a:<14} {b:<14} {c:<14}")
    
    for ver in [v5b, v5c]:
        if ver['trades_detail']:
            print(f"\n📋 {ver['version'].upper()} Trades:")
            for t in ver['trades_detail']:
                emoji = "✅" if t['outcome'] == 'WIN' else "❌" if t['outcome'] == 'LOSS' else "⏱️"
                print(f"  {emoji} {t['direction']:5} ${t['entry']:.0f} -> ${t['exit']:.0f} "
                      f"pnl={t['pnl_pct']:+.2f}% mode={t.get('mode','?'):6} hold={t.get('bars_held',0)}")

if __name__ == "__main__":
    main()
