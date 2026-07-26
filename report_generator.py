import json
import sys

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    data = load_json('/root/.openclaw/workspace/latest_scan.json')
    
    # Helper to get nested value safely
    def get(d, path, default=None):
        for p in path:
            if isinstance(d, dict) and p in d:
                d = d[p]
            else:
                return default
        return d
    
    # Basic info
    signal_status = data.get('status', 'UNKNOWN')
    directional_bias = data.get('direction', 'NEUTRAL')
    # Source reason? We can get from direction_resolver reason
    source_reason = get(data, ['direction_resolver', 'reason'], '')
    ics_score = get(data, ['ics'], 0.0)
    
    # Signal Filters
    # Signal Status from confirmation.status? Actually signal status maybe from data['status']? 
    # The spec: Signal Status: [status] — if NO_SIGNAL, explain the blocker (ICS too low? Sweep blocked? M20 blocked?)
    # We'll use data['status'] as signal status.
    signal_status_field = data.get('status', 'UNKNOWN')
    # Determine blocker explanation if NO_SIGNAL
    blocker_explanation = ""
    if signal_status_field == "NO_SIGNAL":
        # Check reasons
        if ics_score < 0.5:
            blocker_explanation = "ICS too low for signal"
        elif data.get('sweep_blocked'):
            blocker_explanation = "Sweep blocked"
        elif data.get('m20_blocked'):
            blocker_explanation = "M20 blocked"
        else:
            blocker_explanation = "Unknown blocker"
    else:
        blocker_explanation = ""  # not needed if signal
    
    sweep_filter = get(data, ['sweep_filter'], None)
    m20_filter = get(data, ['m20_filter'], None)
    m20_entry_level = get(data, ['m20', 'level'], None)
    sweep_blocked = data.get('sweep_blocked', False)
    m20_blocked = data.get('m20_blocked', False)
    ensemble_passes = data.get('ensemble_passes', False)
    ics = get(data, ['ics'], 0.0)
    
    # Ensemble Gate
    consensus = get(data, ['ensemble', 'consensus'], None)
    passes = get(data, ['ensemble', 'passes'], None)
    agree_count = get(data, ['ensemble', 'agree_count'], None)
    conviction = get(data, ['ensemble', 'ensemble_conviction'], None)
    regime = get(data, ['ensemble', 'regime', 'regime'], None)
    
    # Confirmation Status
    confirmation_status = get(data, ['confirmation', 'status'], None)
    bars_to_confirm = get(data, ['confirmation', 'bars_to_confirm'], None)
    hold_window_hours = get(data, ['confirmation', 'hold_window_hours'], None)
    
    # Exchange Activity
    price = data.get('price', 0.0)
    ema_200 = data.get('ema_200', 0.0)
    derivatives = data.get('derivatives', {})
    oi = derivatives.get('oi', 0)
    oi_usd = derivatives.get('oi_usd', 0)
    oi_roc_1h = derivatives.get('oi_roc_1h', 0.0)
    ls_ratio = derivatives.get('ls_ratio', 0.0)
    long_pct = derivatives.get('long_pct', 0.0)
    top_ls_ratio = derivatives.get('top_ls_ratio', 0.0)
    whale_signal = derivatives.get('whale_signal', '')
    whale_retail_gap = derivatives.get('whale_retail_gap', 0.0)
    funding_rate = derivatives.get('funding_rate', 0.0)
    futures_taker_ratio = derivatives.get('futures_taker_ratio', 0.0)
    futures_flow = derivatives.get('futures_flow', '')
    oi_price_div = derivatives.get('oi_price_div', 'NONE')
    
    exchange_activity = data.get('exchange_activity', {})
    spot_details = exchange_activity.get('spot_details', {})
    basis_avg = spot_details.get('basis_avg', 0.0)
    basis_state = spot_details.get('basis_state', '')
    funding_spread = exchange_activity.get('signals', {}).get('funding_spread', 0.0)
    funding_spread_exchanges = exchange_activity.get('signals', {}).get('funding_spread_exchanges', [])
    exchange_score = exchange_activity.get('score', 0.0)
    spot_score = exchange_activity.get('spot_score', 0.0)
    
    # Macro & Regime
    regime_field = data.get('swing_bias', 'UNKNOWN')  # maybe swing_bias is regime? Actually swing_bias is from daily.
    # Also maybe from ensemble.regime.regime
    macro_indicators = {}
    # We can pull some macro data from m22 etc.
    m22 = data.get('m22', {})
    if m22:
        macro_indicators['m22_regime'] = m22.get('regime')
        macro_indicators['m22_score'] = m22.get('score')
        macro_indicators['m22_size_mult'] = m22.get('size_mult')
    # Add more if needed
    
    # Conflict & Resolution
    conflict_type = data.get('veto', 'NONE')
    # Extract severity? We'll just use veto string.
    # Key Level to Watch: maybe from limit_entry or what_if?
    key_level = get(data, ['limit_entry', 'entry_price'], None)
    if key_level is None:
        key_level = get(data, ['what_if', 'entry'], None)
    scenario = get(data, ['what_if', 'invalidation'], [])
    if isinstance(scenario, list):
        scenario = '; '.join(scenario[:2])  # first two
    
    # Strategy Signals
    strategies_fired = get(data, ['strategies_fired'], None)
    total_strategies = get(data, ['total_strategies'], None)
    # Best strategy? We can look at m1,m2,m3 etc. but maybe we have a field.
    best_strategy_name = get(data, ['best_strategy', 'name'], None)
    best_strategy_type = get(data, ['best_strategy', 'type'], None)
    best_direction = get(data, ['best_strategy', 'direction'], None)
    best_conviction = get(data, ['best_strategy', 'conviction'], None)
    entry_price = get(data, ['best_strategy', 'entry'], None)
    sl_price = get(data, ['best_strategy', 'sl'], None)
    tp1_price = get(data, ['best_strategy', 'tp1'], None)
    rr1 = get(data, ['best_strategy', 'rr1'], None)
    
    # Order Flow
    ob_imbalance = get(data, ['ob_imbalance'], None)
    ob_consensus = get(data, ['ob_consensus'], None)
    taker = get(data, ['taker'], None)
    net_flow = get(data, ['net_flow'], None)
    
    # Build the report
    report_lines = []
    report_lines.append(f"### 🚨 Status: {signal_status}")
    report_lines.append(f"*Directional Bias:* `{directional_bias}` ([{source_reason}: Reason])")
    report_lines.append(f"*Primary Blocker:* {blocker_explanation if blocker_explanation else 'N/A'}")
    report_lines.append(f"*ICS Score:* `{ics_score:.3f}`")
    report_lines.append("")
    report_lines.append("### 🛡️ Signal Filters (explain WHY if null — no signal = filters not applied)")
    report_lines.append(f"*Signal Status:* `{signal_status_field}` — {('explain blocker: ' + blocker_explanation) if signal_status_field == 'NO_SIGNAL' else 'Signal present'}")
    report_lines.append(f"*Sweep Filter:* `{sweep_filter if sweep_filter is not None else 'Not applied (no signal)'}`")
    report_lines.append(f"*M20 Filter:* `{m20_filter if m20_filter is not None else 'Not applied (no signal)'}`")
    report_lines.append(f"*M20 Entry Level:* `{m20_entry_level if m20_entry_level is not None else 'No M20 level available'}`")
    report_lines.append(f"*Sweep Blocked:* `{sweep_blocked}` | *M20 Blocked:* `{m20_blocked}` | *Ensemble Passes:* `{ensemble_passes}`")
    report_lines.append(f"*ICS Score:* `{ics:.3f}` vs threshold 0.50 — {'ICS too low for signal' if ics < 0.5 else 'OK'}")
    report_lines.append("")
    report_lines.append("### 🎯 Ensemble Gate")
    report_lines.append(f"*Consensus:* `{consensus}` | *Passes:* `{passes}`")
    report_lines.append(f"*Agree Count:* `{agree_count}` strategies | *Conviction:* `{conviction}`")
    report_lines.append(f"*Regime:* `{regime if regime is not None else 'N/A'}`")
    report_lines.append("")
    report_lines.append("### ⏳ Confirmation Status")
    report_lines.append(f"*Signal Status:* `{confirmation_status}`")
    report_lines.append(f"*Bars to Confirm:* `{bars_to_confirm}` | *Hold Window:* `{hold_window_hours}h`")
    report_lines.append("")
    report_lines.append("### 📈 Exchange Activity (MUST include ALL these from derivatives + exchange_activity)")
    report_lines.append(f"*Price:* `${price:.2f}` | *EMA200:* `{ema_200:.2f}`")
    report_lines.append(f"*OI:* `{oi}` (${oi_usd:,.0f}) | *OI ROC 1h:* `{oi_roc_1h*100:.2f}%` — {'flag' if abs(oi_roc_1h) > 0.001 else 'ok'}")
    report_lines.append(f"*L/S Ratio:* `{ls_ratio:.3f}` ({long_pct:.1f}% long) | *Top Traders:* `{top_ls_ratio:.3f}`")
    report_lines.append(f"*Whale Signal:* `{whale_signal}` | *Whale-Retail Gap:* `{whale_retail_gap:.3f}`")
    report_lines.append(f"*Funding Rate:* `{funding_rate:.5f}` | *Futures Taker:* `{futures_taker_ratio:.3f}` → `{futures_flow}`")
    report_lines.append(f"*OI-Price Divergence:* `{oi_price_div}` — {'DIVERGENCE, flag it! Price up + OI down = weak move' if oi_price_div == 'DIVERGENCE' else 'No divergence'}")
    report_lines.append(f"*Spot Basis:* `{basis_avg:.4f}` ({basis_state}) — {'backwardation = bearish' if basis_state == 'BACKWARDATION' else 'contango = bullish'}")
    report_lines.append(f"*Funding Spread:* `{funding_spread:.6f}` between `{', '.join(funding_spread_exchanges) if funding_spread_exchanges else 'N/A'}`")
    report_lines.append(f"*Exchange Score:* `{exchange_score}` | *Spot Score:* `{spot_score}`")
    report_lines.append("")
    report_lines.append("### 🌍 Macro & Regime")
    report_lines.append(f"*Regime:* `{regime_field}`")
    # Add macro indicators
    if macro_indicators:
        for k, v in macro_indicators.items():
            report_lines.append(f"*{k}:* {v}")
    report_lines.append("")
    report_lines.append("### ⚖️ Conflict & Resolution")
    report_lines.append(f"*Conflict:* `{conflict_type}` (Severity: N/A)")
    report_lines.append(f"**Key Level to Watch:** `${key_level if key_level is not None else 'N/A'}`")
    report_lines.append(f"*Scenario:* {scenario}")
    report_lines.append("")
    report_lines.append("### 🎯 Strategy Signals")
    report_lines.append(f"*Strategies Fired:* `{strategies_fired if strategies_fired is not None else 'N/A'}`/`{total_strategies if total_strategies is not None else 'N/A'}`")
    report_lines.append(f"*Best Strategy:* {best_strategy_name if best_strategy_name is not None else 'N/A'} ({best_strategy_type if best_strategy_type is not None else 'N/A'}) | *Direction:* {best_direction if best_direction is not None else 'N/A'} | *Conviction:* {best_conviction if best_conviction is not None else 'N/A'}%")
    report_lines.append(f"*Entry:* ${entry_price if entry_price is not None else 'N/A'} | *SL:* ${sl_price if sl_price is not None else 'N/A'} | *TP1:* ${tp1_price if tp1_price is not None else 'N/A'} | *R:R:* {rr1 if rr1 is not None else 'N/A'}")
    report_lines.append("")
    report_lines.append("### 📊 Order Flow")
    report_lines.append(f"*OB Imbalance:* `{ob_imbalance}` ({ob_consensus})")
    report_lines.append(f"*Trade Taker:* `{taker}` | *Net Flow:* `{net_flow}`")
    report_lines.append("")
    report_lines.append("### 📝 Narrative")
    # Narrative generation
    narrative = f"The market shows a {directional_bias} bias with {signal_status_field} signal. "
    narrative += f"ICS is {ics_score:.3f} ({'below' if ics_score < 0.5 else 'above'} threshold). "
    narrative += f"Price is ${price:.2f} vs EMA200 ${ema_200:.2f}. "
    narrative += f"OI is {oi} (${oi_usd:,.0f}) with {oi_roc_1h*100:.2f}% hourly change. "
    narrative += f"L/S ratio shows {long_pct:.1f}% long. "
    narrative += f"Funding rate is {funding_rate:.5f}. "
    narrative += f"Verdict: WATCH. Wait for confirmation."
    report_lines.append(narrative)
    report_lines.append("")
    report_lines.append("*Verdict:* [WATCH/TRADE/AVOID]. [Short instruction].")
    report_lines.append("Current time: Saturday, July 25th, 2026 - 12:45 AM (UTC)")
    report_lines.append("Reference UTC: 2026-07-25 00:45 UTC")
    
    print("\n".join(report_lines))

if __name__ == "__main__":
    main()