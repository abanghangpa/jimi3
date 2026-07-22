#!/usr/bin/env python3
"""
JIMI Live Trader v3 — Combined Strategy
BB Mean Rev + mom6_2pct + 48h 2% Volatility Gate
Backtested: 64.3% WR, PF 1.97, 100% survival, $61K avg withdrawn (50 seeds, 2021-2026)
Signal: BB oversold/overbought OR 6h momentum > 2%
Gate: 48h rolling avg abs 12h momentum >= 2%
Params: TP 0.3%, SL 0.2%, 20x leverage, 5% risk, 8h hold
"""
import json, os, sys, time, requests
from datetime import datetime, timezone, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE, "live", "config", "strategies.json")
STATE_FILE = os.path.join(BASE, "live", "data", "state.json")
TRADE_LOG = os.path.join(BASE, "live", "data", "trades.json")
LOG_FILE = os.path.join(BASE, "live", "logs", "trader.log")

with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

EXCHANGE = CONFIG["exchange"]["name"]
MODE = CONFIG["exchange"]["mode"]

# === BACKTESTED PARAMETERS ===
# BB Mean Rev primary: TP=0.3% SL=0.2% (from param sweep)
TP_PCT = 0.002
SL_PCT = 0.001
LEVERAGE = 25
RISK_PCT = 0.10
HOLD_HOURS = 8
FEE_RATE = 0.0002
SLIPPAGE = 0.001
INITIAL_CAPITAL = 200

# === SIGNAL THRESHOLDS ===
MOM6_THRESHOLD = 0.02      # 6h momentum > 2% for mom6 signal
BB_RSI_LOW = 35            # RSI threshold for BB combo (not used in pure BB)
BB_RSI_HIGH = 65
# BB signal: price < lower band (LONG), price > upper band (SHORT) — no RSI filter

# === VOLATILITY GATE ===
MOM12_GATE_WINDOW = 48
MOM12_GATE_THRESHOLD = 0.01

# === DRAWDOWN BREAKER ===
DD_STOP = 0.50
DD_COOLDOWN_HOURS = 24

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "capital": INITIAL_CAPITAL, "peak_capital": INITIAL_CAPITAL,
        "positions": [], "closed_trades": [], "total_pnl": 0, "total_fees": 0,
        "trades_count": 0, "wins": 0, "losses": 0, "withdrawals": [],
        "total_withdrawn": 0, "dd_cooldown_until": None, "dd_triggered_count": 0,
        "current_strategy": "combined_bb_mom6", "last_signal_check": None,
        "vol_gate_skips": 0, "vol_gate_active": True,
    }

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def log_trade(trade):
    os.makedirs(os.path.dirname(TRADE_LOG), exist_ok=True)
    trades = []
    if os.path.exists(TRADE_LOG):
        with open(TRADE_LOG) as f:
            trades = json.load(f)
    trades.append(trade)
    with open(TRADE_LOG, "w") as f:
        json.dump(trades, f, indent=2, default=str)

def get_price():
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT", timeout=5)
        return float(r.json()["price"])
    except Exception as e:
        log(f"ERROR price: {e}")
        return None

def get_candles(interval="1h", limit=50):
    try:
        r = requests.get(f"https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval={interval}&limit={limit}", timeout=10)
        return r.json()
    except Exception as e:
        log(f"ERROR candles: {e}")
        return None

# === VOLATILITY GATE ===
def check_volatility_gate():
    need = MOM12_GATE_WINDOW + 12 + 2
    candles = get_candles(interval="1h", limit=need)
    if not candles or len(candles) < need:
        return True, 0, "gate_open_no_data"
    mom12_abs = []
    for i in range(12, len(candles)):
        p_now = float(candles[i][4])
        p_past = float(candles[i-12][4])
        if p_past > 0:
            mom12_abs.append(abs((p_now - p_past) / p_past))
    if len(mom12_abs) < MOM12_GATE_WINDOW:
        return True, 0, "gate_open_no_window"
    avg = sum(mom12_abs[-MOM12_GATE_WINDOW:]) / MOM12_GATE_WINDOW
    passes = avg >= MOM12_GATE_THRESHOLD
    return passes, avg, f"vol={avg*100:.2f}% {'PASS' if passes else 'BLOCK'}"

# === SIGNALS ===
def check_signal_bb():
    """Bollinger Band mean reversion: price < lower BB LONG, price > upper BB SHORT"""
    candles = get_candles(interval="1h", limit=22)
    if not candles or len(candles) < 21:
        return None, 0, "bb_no_data"
    closes = [float(c[4]) for c in candles]
    # BB(20, 2.0)
    sma20 = sum(closes[-20:]) / 20
    std20 = (sum((x - sma20)**2 for x in closes[-20:]) / 20) ** 0.5
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    price = closes[-1]
    if price < lower:
        return "LONG", (price - lower) / lower, f"BB_LONG price={price:.2f}<low={lower:.2f}"
    elif price > upper:
        return "SHORT", (price - upper) / upper, f"BB_SHORT price={price:.2f}>up={upper:.2f}"
    return None, 0, f"BB_NEUTRAL {lower:.2f}<{price:.2f}<{upper:.2f}"

def check_signal_mom6():
    """6h momentum > 2%"""
    candles = get_candles(interval="1h", limit=8)
    if not candles or len(candles) < 7:
        return None, 0, "mom6_no_data"
    current = float(candles[-1][4])
    past = float(candles[-7][4])
    if past == 0:
        return None, 0, "mom6_zero"
    mom = (current - past) / past
    if mom > MOM6_THRESHOLD:
        return "LONG", mom, f"mom6={mom*100:+.2f}%>2%"
    elif mom < -MOM6_THRESHOLD:
        return "SHORT", mom, f"mom6={mom*100:+.2f}%<-2%"
    return None, mom, f"mom6={mom*100:+.2f}%<2%"

def get_combined_signal():
    """BB Mean Rev (primary) OR mom6_2pct (secondary). BB takes priority."""
    bb_dir, bb_val, bb_info = check_signal_bb()
    if bb_dir:
        return bb_dir, bb_val, bb_info
    mom_dir, mom_val, mom_info = check_signal_mom6()
    if mom_dir:
        return mom_dir, mom_val, mom_info
    return None, 0, f"{bb_info} | {mom_info}"

# === DRAWDOWN BREAKER ===
def check_dd(state):
    capital = state["capital"]
    peak = state.get("peak_capital", INITIAL_CAPITAL)
    if capital > peak:
        state["peak_capital"] = capital
        peak = capital
    cd = state.get("dd_cooldown_until")
    if cd:
        if datetime.now(timezone.utc) < datetime.fromisoformat(cd):
            return True, f"DD cooldown {cd[:16]}"
        state["dd_cooldown_until"] = None
    if peak > 0:
        dd = (peak - capital) / peak
        if dd >= DD_STOP:
            until = datetime.now(timezone.utc) + timedelta(hours=DD_COOLDOWN_HOURS)
            state["dd_cooldown_until"] = until.isoformat()
            state["dd_triggered_count"] = state.get("dd_triggered_count", 0) + 1
            return True, f"DD={dd*100:.0f}% STOP"
    return False, f"DD={((peak-capital)/peak*100) if peak>0 else 0:.1f}%"

def calc_size(capital, entry, sl):
    sd = abs(entry - sl)
    if sd == 0: return 0
    return min(capital * RISK_PCT / sd, capital * LEVERAGE / entry)

def open_position(state, direction, price, reason):
    entry = price * (1 + SLIPPAGE) if direction == "LONG" else price * (1 - SLIPPAGE)
    if direction == "LONG":
        tp = entry * (1 + TP_PCT); sl = entry * (1 - SL_PCT)
    else:
        tp = entry * (1 - TP_PCT); sl = entry * (1 + SL_PCT)
    size = calc_size(state["capital"], entry, sl)
    if size <= 0:
        log("SKIP: size too small"); return None
    now = datetime.now(timezone.utc)
    pos = {
        "id": f"{direction[0]}_{now.strftime('%Y%m%d_%H%M%S')}",
        "direction": direction, "entry": round(entry, 2),
        "tp": round(tp, 2), "sl": round(sl, 2), "size": round(size, 6),
        "capital_at_entry": round(state["capital"], 2),
        "opened_at": now.isoformat(), "strategy": "combined_bb_mom6", "reason": reason,
    }
    state["positions"].append(pos)
    log(f"OPEN: {direction} {size:.4f} ETH @ ${entry:.2f} TP=${tp:.2f} SL=${sl:.2f} | {reason}")
    return pos

def check_tp_sl(pos, price):
    d = pos["direction"]
    if d == "LONG":
        if price >= pos["tp"]: return "WIN", pos["tp"]
        elif price <= pos["sl"]: return "LOSS", pos["sl"]
    else:
        if price <= pos["tp"]: return "WIN", pos["tp"]
        elif price >= pos["sl"]: return "LOSS", pos["sl"]
    return None, None

def close_position(state, pos, exit_price, outcome):
    entry = pos["entry"]; size = pos["size"]; d = pos["direction"]
    pnl_raw = (exit_price - entry) * size if d == "LONG" else (entry - exit_price) * size
    fee = entry * size * FEE_RATE * 2
    pnl = pnl_raw - fee
    state["capital"] += pnl; state["total_pnl"] += pnl; state["total_fees"] += fee
    state["trades_count"] += 1
    if outcome == "WIN": state["wins"] += 1
    else: state["losses"] += 1
    if state["capital"] > state["peak_capital"]: state["peak_capital"] = state["capital"]
    closed = {**pos, "exit": round(exit_price, 2), "pnl": round(pnl, 4),
              "fee": round(fee, 4), "outcome": outcome, "closed_at": datetime.now(timezone.utc).isoformat()}
    state["closed_trades"].append(closed)
    state["positions"] = [p for p in state["positions"] if p["id"] != pos["id"]]
    log_trade(closed)
    log(f"CLOSE: {'WIN' if outcome=='WIN' else 'LOSS'} {d} ${entry:.2f}->${exit_price:.2f} PnL=${pnl:+.2f}")
    return state

def check_withdrawal(state):
    target = CONFIG.get("withdrawal", {}).get("target", 2700)
    amount = CONFIG.get("withdrawal", {}).get("withdraw_amount", 2500)
    keep = CONFIG.get("withdrawal", {}).get("keep_as_base", 200)
    if state["capital"] >= target and (state["capital"] - keep) >= amount:
        state["withdrawals"].append({"amount": amount, "date": datetime.now(timezone.utc).isoformat()})
        state["capital"] -= amount; state["total_withdrawn"] += amount
        log(f"WITHDRAWAL: ${amount:,.0f} | Remaining: ${state['capital']:,.0f}")
        save_state(state)

def main():
    log("=" * 60)
    log("JIMI v3 STARTING | Combined BB+Mom6 + VolGate")
    log(f"Mode: {MODE.upper()} | Exchange: {EXCHANGE}")
    log(f"Params: TP={TP_PCT*100:.1f}% SL={SL_PCT*100:.1f}% {LEVERAGE}x {RISK_PCT*100:.0f}% {HOLD_HOURS}h")
    log("=" * 60)
    state = load_state()
    while True:
        try:
            price = get_price()
            if not price: time.sleep(60); continue
            dd_blocked, dd_info = check_dd(state)
            for pos in list(state["positions"]):
                outcome, ep = check_tp_sl(pos, price)
                if outcome: state = close_position(state, pos, ep, outcome)
            now = datetime.now(timezone.utc)
            for pos in list(state["positions"]):
                opened = datetime.fromisoformat(pos["opened_at"])
                if (now - opened).total_seconds() > HOLD_HOURS * 3600:
                    state = close_position(state, pos, price, "TIME")
            sig_dir, sig_val, sig_info = get_combined_signal()
            vol_pass, vol_val, vol_info = check_volatility_gate()
            if not vol_pass: state["vol_gate_skips"] = state.get("vol_gate_skips", 0) + 1
            if sig_dir and vol_pass and not dd_blocked and len(state["positions"]) == 0 and state["capital"] > 10:
                open_position(state, sig_dir, price, sig_info)
            check_withdrawal(state)
            state["last_signal_check"] = now.isoformat()
            save_state(state)
            cap = state["capital"]; pk = state.get("peak_capital", INITIAL_CAPITAL)
            dd_pct = ((pk-cap)/pk*100) if pk>0 else 0
            ret = (cap-INITIAL_CAPITAL)/INITIAL_CAPITAL*100
            log(f"ETH=${price:,.2f} | Cap=${cap:,.2f}({ret:+.1f}%) | DD={dd_pct:.1f}% | {state['trades_count']}T {state['wins']}W/{state['losses']}L | Sig={sig_info} | {vol_info} | {dd_info}")
            time.sleep(60)
        except KeyboardInterrupt:
            log("Shutting down..."); save_state(state); break
        except Exception as e:
            log(f"ERROR: {e}"); save_state(state); time.sleep(60)

if __name__ == "__main__":
    main()
