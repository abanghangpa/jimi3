#!/usr/bin/env python3
"""
Scanner Live Executor — Places trades via HTX API based on scanner signals.
OPTIMIZED configs from 2026-07-05 optimization (PF >= 2.0 strategies only).
"""
import json, os, sys, time, math, argparse
from datetime import datetime, timezone, timedelta
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

SCAN_DIR = os.path.join(BASE, "data", "scans")
SIGNALS_FILE = os.path.join(BASE, "data", "strategy_signals.jsonl")
STATE_FILE = os.path.join(BASE, "live", "data", "executor_state.json")
TRADE_LOG = os.path.join(BASE, "live", "data", "executor_trades.json")
LOG_FILE = os.path.join(BASE, "live", "logs", "executor.log")
KEYS_FILE = os.path.join(BASE, "config", "exchange_keys.json")

SYMBOL = "ETH/USDT:USDT"
INITIAL_CAPITAL = 200.0

# === OPTIMIZED STRATEGY CONFIGS (from 2026-07-05 optimization, PF >= 2.0) ===
STRATEGY_CONFIGS = {
    # === PROVEN STRATEGIES (PF >= 2.0, WR >= 70%) ===
    "whale_watch": {
        "tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 8,
        "direction": "LONG", "enabled": False,
        "group": "B",
        "min_conviction": 0.5,
    },
    "funding_arb": {
        "tp_pct": 1.0, "sl_pct": 1.0, "hold_hours": 16,
        "direction": None, "enabled": True,
        "group": "B",
        "min_conviction": 0.50,
    },
    "orderbook_imbalance": {
        "tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 12,
        "direction": "LONG", "enabled": True,
        "group": "A",
        "min_conviction": 0.5,
    },
    "failed_breakout": {
        "tp_pct": 2.5, "sl_pct": 1.0, "hold_hours": 32,
        "direction": None, "enabled": False,
        "min_conviction": 0.7,
        "group": "A",
    },
    "positioning_fade": {
        "tp_pct": 1.0, "sl_pct": 1.0, "hold_hours": 16,
        "direction": "LONG", "enabled": True,
        "group": "A",
        "min_conviction": 0.35,
    },
    "trade_flow": {
        "tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 12,
        "direction": "LONG", "enabled": True,
        "group": "A",
        "min_conviction": 0.5,
    },
    "structural_break": {
        "tp_pct": 0.5, "sl_pct": 0.5, "hold_hours": 8,
        "direction": "SHORT", "enabled": False,
        "group": "A",
        "min_conviction": 0.5,
    },
    "regime_switch": {
        "tp_pct": 1.0, "sl_pct": 1.0, "hold_hours": 8,
        "direction": "SHORT", "enabled": False,
        "group": "A",
        "min_conviction": 0.5,
    },
    # === DISABLED: PF < 2.0 or insufficient data ===
    "squeeze_breakout": {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 8, "direction": None, "enabled": True, "group": "B"},  # Co-occurrence filter: confirms OBI (80% WR, PF=3.16)
    "bb_mom6": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A", },  # PF 1.86
    "cross_asset": {"tp_pct": 2.0, "sl_pct": 1.5, "hold_hours": 4, "direction": None, "enabled": True,
        "group": "B", "min_conviction": 0.5},  # PF 1.61
    "scalp_v2": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A", },  # PF 1.11
    "power_of_3": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A", },  # No signals
    "macro_surprise": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A", },  # No data
    "liquidation_cascade": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A", },  # No data
    "judas_sweep": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A", },  # Rarely fires
    "taker_flow": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A", },  # Only 7 signals
    "vol_rotation": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "B", },  # Only 8 signals
    "kill_zone": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A", },  # PF 0.99
    "liquidity_grab": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A", },  # Limited signals
    "cascade": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A", },  # No data
    "mtf_confluence": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": False, "group": "A", },  # PF 1.15
    "momentum_v3": {"tp_pct": 0.5, "sl_pct": 1.0, "hold_hours": 8, "direction": None, "enabled": True, "group": "B"},  # State filter: exhaustion - confirms Group A signals
    "momentum_v2": {"tp_pct": 1.0, "sl_pct": 1.0, "hold_hours": 16, "direction": None, "enabled": True, "group": "B"},  # Co-occurrence filter: confirms OBI (62.5% WR, PF=3.14)
}

# === EXECUTION PARAMS ===
RISK_PCT = 0.10
LEVERAGE = 25
MAX_SLIPPAGE_PCT = 0.30
BLOCKED_HOURS = {19, 20, 21}
BLOCKED_DAYS = {"Sat"}
MAX_POSITIONS = 3
SIGNAL_MAX_AGE_SEC = 1200
FEE_PCT = 0.001
MIN_CONVICTION = 0.5
ORDER_TYPE = "limit"
LIMIT_OFFSET_PCT = 0.02

def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
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
        "open_positions": [], "closed_trades": [],
        "total_pnl": 0, "total_fees": 0,
        "trades_count": 0, "wins": 0, "losses": 0, "timeouts": 0,
        "last_signal_ts": None, "dd_cooldown_until": None,
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

def get_exchange(dry_run=False):
    import ccxt
    api_key = os.environ.get("HTX_API_KEY", "")
    api_secret = os.environ.get("HTX_API_SECRET", "")
    if not api_key and os.path.exists(KEYS_FILE):
        with open(KEYS_FILE) as f:
            keys = json.load(f)
            api_key = keys.get("api_key", "")
            api_secret = keys.get("api_secret", "")
    exchange = ccxt.htx({
        "apiKey": api_key, "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "defaultMarginMode": "isolated"},
    })
    if not dry_run and not api_key:
        log("NO API KEY — switching to dry-run", "WARN")
        dry_run = True
    return exchange, dry_run

def get_latest_signals():
    scan_files = sorted(
        [f for f in os.listdir(SCAN_DIR) if f.startswith("scan_") and f.endswith(".json")],
        reverse=True
    )
    if not scan_files:
        return []
    latest = os.path.join(SCAN_DIR, scan_files[0])
    with open(latest) as f:
        data = json.load(f)
    ts = data.get("timestamp", "")
    price = data.get("price", 0)
    status = data.get("status", "")
    multi = data.get("multi_strategy") or {}
    all_signals = multi.get("all_signals", [])
    single = data.get("strategy_signal", {})
    if single and single.get("direction"):
        single_strat = single.get("strategy", "")
        if not any(s.get("strategy") == single_strat for s in all_signals):
            all_signals.append(single)

    # === GROUP A/B VOTING ===
    # Collect all fired signals by group
    group_a_fired = []  # event strategies
    group_b_fired = []  # state filters
    for sig_data in all_signals:
        if not isinstance(sig_data, dict):
            continue
        strat_name = sig_data.get("strategy", "")
        cfg = STRATEGY_CONFIGS.get(strat_name)
        if not cfg:
            continue
        direction = sig_data.get("direction")
        conviction = sig_data.get("conviction", 0) or 0
        if not direction or conviction < 0.1:
            continue
        grp = cfg.get("group", "A")
        if grp == "A":
            group_a_fired.append(sig_data)
        elif grp == "B":
            group_b_fired.append(sig_data)

    # Build set of (direction) confirmed by Group B
    b_directions = set()
    for sig in group_b_fired:
        d = sig.get("direction")
        if d:
            b_directions.add(d)

    # === BUILD SIGNALS ===
    signals = []
    for sig_data in all_signals:
        if not isinstance(sig_data, dict):
            continue
        strat_name = sig_data.get("strategy", "")
        cfg = STRATEGY_CONFIGS.get(strat_name)
        if not cfg or not cfg["enabled"]:
            continue
        if cfg.get("group") == "B":
            continue  # Group B don't trade standalone
        direction = sig_data.get("direction")
        conviction = sig_data.get("conviction", 0) or 0
        entry = sig_data.get("entry", price)
        sl = sig_data.get("sl", 0)
        tp1 = sig_data.get("tp1", 0)
        min_conv = cfg.get("min_conviction", MIN_CONVICTION)
        if not direction or conviction < min_conv:
            continue
        if cfg["direction"] and direction != cfg["direction"]:
            continue

        # Check if Group B confirms this direction
        group_boost = 1.0
        confirmed_by = []
        if direction in b_directions:
            group_boost = 1.5  # 50% size boost when B confirms
            confirmed_by = [s.get("strategy") for s in group_b_fired if s.get("direction") == direction]

        signals.append({
            "strategy": strat_name, "timestamp": ts, "direction": direction,
            "conviction": conviction, "entry": entry or price,
            "sl": sl, "tp1": tp1, "price": price, "cfg": cfg,
            "scan_status": status,
            "group_boost": group_boost,
            "confirmed_by": confirmed_by,
        })
    return signals

def is_signal_fresh(signal_ts):
    try:
        sig_dt = datetime.strptime(signal_ts, "%Y-%m-%d %H:%M:%S")
        sig_dt = sig_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            sig_dt = datetime.fromisoformat(signal_ts.replace("Z", "+00:00"))
        except:
            return False
    now = datetime.now(timezone.utc)
    return (now - sig_dt).total_seconds() < SIGNAL_MAX_AGE_SEC

def check_tp_sl(pos, current_price):
    d = pos["direction"]
    tp = pos["tp"]; sl = pos["sl"]
    if d == "LONG":
        if current_price >= tp: return "WIN", tp
        elif current_price <= sl: return "LOSS", sl
    else:
        if current_price <= tp: return "WIN", tp
        elif current_price >= sl: return "LOSS", sl
    return None, None

def close_position(state, pos, exit_price, outcome):
    entry = pos["fill_price"]; size = pos["size"]; d = pos["direction"]
    lev = pos.get("leverage", LEVERAGE)
    pnl_raw = (exit_price - entry) * size * lev if d == "LONG" else (entry - exit_price) * size * lev
    fee = entry * size * FEE_PCT * 2
    pnl = pnl_raw - fee
    state["capital"] += pnl; state["total_pnl"] += pnl; state["total_fees"] += fee
    state["trades_count"] += 1
    if outcome == "WIN": state["wins"] += 1
    else: state["losses"] += 1
    if state["capital"] > state["peak_capital"]: state["peak_capital"] = state["capital"]
    closed = {**pos, "exit": round(exit_price, 2), "pnl": round(pnl, 4),
              "fee": round(fee, 4), "outcome": outcome,
              "closed_at": datetime.now(timezone.utc).isoformat()}
    state["closed_trades"].append(closed)
    state["open_positions"] = [p for p in state["open_positions"] if p.get("order_id") != pos.get("order_id")]
    log_trade(closed)
    log(f"CLOSE: {outcome} {d} ${entry:.2f}->${exit_price:.2f} PnL=${pnl:+.2f}")
    return state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    log("=" * 60)
    log(f"Scanner Executor Starting ({'DRY RUN' if args.dry_run else 'LIVE'})")
    enabled = [k for k,v in STRATEGY_CONFIGS.items() if v['enabled']]
    log(f"Strategies: {enabled}")
    log(f"Params: {LEVERAGE}x | {RISK_PCT*100:.0f}% risk | Fee: {FEE_PCT*100:.2f}%")
    log("=" * 60)

    exchange, dry_run = get_exchange(args.dry_run)
    state = load_state()
    log(f"Capital: ${state['capital']:.2f} | Positions: {len(state['open_positions'])} | Trades: {state['trades_count']} ({state['wins']}W/{state['losses']}L)")

    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.hour in BLOCKED_HOURS or now.strftime("%a") in BLOCKED_DAYS:
                time.sleep(args.interval); continue

            # Check existing positions
            for pos in list(state["open_positions"]):
                opened = datetime.fromisoformat(pos["opened_at"])
                hold_h = pos.get("hold_hours", 8)
                if (now - opened).total_seconds() > hold_h * 3600:
                    close_position(state, pos, pos["fill_price"], "TIMEOUT")
                    continue
                try:
                    ticker = exchange.fetch_ticker(SYMBOL)
                    price = ticker["last"]
                except:
                    continue
                outcome, ep = check_tp_sl(pos, price)
                if outcome:
                    close_position(state, pos, ep, outcome)

            # Get new signals
            signals = get_latest_signals()
            for sig in signals:
                if not is_signal_fresh(sig["timestamp"]):
                    continue
                if any(p["strategy"] == sig["strategy"] for p in state["open_positions"]):
                    continue
                if len(state["open_positions"]) >= MAX_POSITIONS:
                    continue

                entry = sig["entry"]; sl = sig["sl"]; tp1 = sig["tp1"]
                cfg = sig["cfg"]

                if not entry or not sl or not tp1:
                    continue

                if dry_run:
                    fill_price = entry * (1 + 0.001) if sig["direction"] == "LONG" else entry * (1 - 0.001)
                else:
                    fill_price = entry

                sl_pct = abs(fill_price - sl) / fill_price
                if sl_pct <= 0: continue
                # Apply group boost (A+B confirmation = larger size)
                group_boost = sig.get("group_boost", 1.0)
                size = (state["capital"] * RISK_PCT * group_boost) / (sl_pct * LEVERAGE)
                if size < 0.001: continue

                pos = {
                    "strategy": sig["strategy"], "direction": sig["direction"],
                    "fill_price": round(fill_price, 2), "tp": round(tp1, 2),
                    "sl": round(sl, 2), "size": round(size, 6),
                    "leverage": LEVERAGE, "hold_hours": cfg["hold_hours"],
                    "tp_pct": cfg["tp_pct"], "sl_pct": cfg["sl_pct"],
                    "signal_ts": sig["timestamp"],
                    "opened_at": now.isoformat(),
                    "order_id": f"dry_{int(now.timestamp())}" if dry_run else None,
                }
                state["open_positions"].append(pos)
                confirmed = sig.get("confirmed_by", [])
                conf_str = f" +B:{','.join(confirmed)}" if confirmed else " (solo)"
                log(f"{'DRY RUN: ' if dry_run else ''}OPEN {sig['direction']} {size:.4f} ETH @ ${fill_price:.2f} TP=${tp1:.2f} SL=${sl:.2f} [{sig['strategy']}]{conf_str} boost={group_boost:.1f}x")
                log(f"POSITION OPENED: {sig['strategy']} {sig['direction']} {size:.4f} ETH @ ${fill_price:.2f} TP=${tp1:.2f} SL=${sl:.2f}")

            save_state(state)
            log(f"Capital: ${state['capital']:.2f} | Positions: {len(state['open_positions'])} | Trades: {state['trades_count']} ({state['wins']}W/{state['losses']}L)")

            if args.once: break
            time.sleep(args.interval)

        except KeyboardInterrupt:
            log("Shutting down..."); save_state(state); break
        except Exception as e:
            log(f"ERROR: {e}", "ERROR"); save_state(state); time.sleep(args.interval)

if __name__ == "__main__":
    main()

