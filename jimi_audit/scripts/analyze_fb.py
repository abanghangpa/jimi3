import json
with open('/root/.openclaw/workspace/jimi_audit/reports/failed_breakout_backtest.json') as f:
    d = json.load(f)
b = d['baseline']
print("=== BASELINE ===")
print(f"Trades: {b['total_trades']}")
print(f"WR: {b['win_rate']}%")
print(f"PF: {b['profit_factor']}")
print(f"PnL: {b['total_pnl_pct']}%")
print(f"DD: {b['max_drawdown_pct']}%")
print(f"Avg Win: {b['avg_win_pnl']}%")
print(f"Avg Loss: {b['avg_loss_pnl']}%")
print(f"Timeouts: {b['timeouts']}")
print(f"Config: TP={b['config']['tp_pct']}% SL={b['config']['sl_pct']}% Hold={b['config']['hold_hours']}h Conv>={b['config']['min_conv']}")
print()
print("=== MONTHLY ===")
months = b.get('monthly', {})
for m, v in sorted(months.items()):
    total = v['wins'] + v['losses']
    wr = v['wins']/total*100 if total > 0 else 0
    print(f"  {m}: {v['wins']}W/{v['losses']}L ({wr:.0f}%WR) PnL={v['pnl']:+.1f}%")
print()
bad = sum(1 for m, v in months.items() if v['pnl'] < 0)
good = sum(1 for m, v in months.items() if v['pnl'] >= 0)
print(f"Months: {good} good, {bad} bad")

# Analyze trade distribution
trades = d.get('trades', [])
if trades:
    print(f"\n=== TRADE ANALYSIS ({len(trades)} trades) ===")
    wins = [t for t in trades if t['outcome'] == 'WIN']
    losses = [t for t in trades if t['outcome'] == 'LOSS']
    timeouts = [t for t in trades if t['outcome'] == 'TIMEOUT']
    
    print(f"Wins: {len(wins)}, Losses: {len(losses)}, Timeouts: {len(timeouts)}")
    
    if wins:
        avg_win_hold = sum(t['held_hours'] for t in wins) / len(wins)
        avg_win_pnl = sum(t['pnl_pct'] for t in wins) / len(wins)
        print(f"Avg win hold: {avg_win_hold:.1f}h, avg win pnl: {avg_win_pnl:.2f}%")
    
    if losses:
        avg_loss_hold = sum(t['held_hours'] for t in losses) / len(losses)
        avg_loss_pnl = sum(t['pnl_pct'] for t in losses) / len(losses)
        print(f"Avg loss hold: {avg_loss_hold:.1f}h, avg loss pnl: {avg_loss_pnl:.2f}%")
    
    if timeouts:
        timeout_pnl = sum(t['pnl_pct'] for t in timeouts) / len(timeouts)
        timeout_win = sum(1 for t in timeouts if t['pnl_pct'] > 0)
        print(f"Timeouts: {len(timeouts)}, avg pnl: {timeout_pnl:.2f}%, {timeout_win} profitable")
    
    # Conviction analysis
    print(f"\n=== CONVICTION ANALYSIS ===")
    for conv_range in [(0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        subset = [t for t in trades if conv_range[0] <= t.get('conviction', 0) < conv_range[1]]
        if subset:
            w = sum(1 for t in subset if t['outcome'] == 'WIN')
            wr = w/len(subset)*100
            pnl = sum(t['pnl_pct'] for t in subset)
            print(f"  Conv {conv_range[0]:.1f}-{conv_range[1]:.1f}: {len(subset)}T, {wr:.0f}%WR, PnL={pnl:+.1f}%")
    
    # Direction analysis
    print(f"\n=== DIRECTION ANALYSIS ===")
    for d in ['LONG', 'SHORT']:
        subset = [t for t in trades if t['direction'] == d]
        if subset:
            w = sum(1 for t in subset if t['outcome'] == 'WIN')
            wr = w/len(subset)*100
            pnl = sum(t['pnl_pct'] for t in subset)
            print(f"  {d}: {len(subset)}T, {wr:.0f}%WR, PnL={pnl:+.1f}%")
