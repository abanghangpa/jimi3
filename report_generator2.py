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
    source_reason = get(data, ['direction_resolver', 'reason'], '')
    ics_score = get(data, ['ics'], 0.0)
    
    # Signal Filters
    signal_status_field = data.get('status', 'UNKNOWN')
    blocker_explanation = ""
    if signal_status_field == "NO_SIGNAL":
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
    
    sweep_filter = get(data, ['sweep_filter'])
    m20_filter = get(data, ['m20_filter'])
    m20_entry_level = get(data, ['m20', 'level'])
    sweep_blocked = data.get('sweep_blocked', False)
    m20_blocked = data.get('m20_blocked', False)
    ensemble_passes = data.get('ensemble_passes', False)
    ics = get(data, ['ics'], 0.0)
    
    # Ensemble Gate
    consensus = get(data, ['ensemble', 'consensus'])
    passes = get(data, ['ensemble', 'passes'])
    agree_count = get(data, ['ensemble', 'agree_count'])
    conviction = get(data, ['ensemble', 'ensemble_conviction'])
    regime = get(data, ['ensemble', 'regime', 'regime'])
    
    # Confirmation Status
    confirmation_status = get(data, ['confirmation', 'status'])
    bars_to_confirm = get(data, ['confirmation', 'bars_to_confirm'])
    hold_window_hours = get(data, ['confirmation', 'hold_window_hours'])
    
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
    # We'll also get regime from swing_bias (daily bias) and maybe from ensemble
    regime_field = data.get('swing_bias', 'UNKNOWN')
    macro_indicators = {}
    m22 = data.get('m22', {})
    if m22:
        macro_indicators['m22_regime'] = m22.get('regime')
        macro_indicators['m22_score'] = m22.get('score')
        macro_indicators['m22_size_mult'] = m22.get('size_mult')
    
    # Conflict & Resolution
    conflict_type = data.get('veto', 'NONE')
    # Key level from limit_entry or what_if
    key_level = get(data, ['limit_entry', 'entry_price'])
    if key_level is None:
        key_level = get(data, ['what_if', 'entry'])
    scenario_list = get(data, ['what_if', 'invalidation'], [])
    if isinstance(scenario_list, list):
        scenario = '; '.join(scenario_list[:2]) if scenario_list else 'None'
    else:
        scenario = str(scenario_list)
    
    # Strategy Signals: try to count strategies that fired
    # We'll define a list of strategy keys we know
    strategy_keys = ['m1', 'm2', 'm3', 'm4', 'm5', 'm7', 'm8', 'm10', 'm12', 'm13', 'm14', 'm17', 'm20', 'm21', 'm22', 'm23', 'm24', 'm25', 'm26', 'm27', 'm28', 'm29', 'm30', 'm31', 'm32', 'm33', 'm34', 'm35', 'm36', 'm37', 'm38', 'm39', 'm40', 'm41', 'm42', 'm43', 'm44', 'm45', 'm46', 'm47', 'm48', 'm49', 'm50', 'm51', 'm52', 'm53', 'm54', 'm55', 'm56', 'm57', 'm58', 'm59', 'm60', 'm61', 'm62']
    strategies_fired = 0
    total_strategies = len(strategy_keys)
    best_strategy_name = None
    best_strategy_type = None
    best_direction = None
    best_conviction = None
    best_score = -1.0
    best_entry = None
    best_sl = None
    best_tp1 = None
    best_rr1 = None
    
    for key in strategy_keys:
        val = data.get(key)
        if isinstance(val, dict):
            score = val.get('score')
            status = val.get('status')
            # Consider fired if status == 'PASS' and score > 0.5? We'll just count PASS as fired.
            if status == 'PASS':
                strategies_fired += 1
                if score is not None and score > best_score:
                    best_score = score
                    best_strategy_name = key.upper()
                    # Map to a readable name
                    name_map = {
                        'M1': 'MACD',
                        'M2': 'EMA',
                        'M3': 'VWAP',
                        'M4': 'CVD',
                        'M5': 'LIQUIDATION',
                        'M7': 'MACRO',
                        'M8': 'FUNDING',
                        'M10': 'MACRO',
                        'M12': 'ORDERBOOK',
                        'M13': 'STRUCTURE',
                        'M14': 'SWEEP',
                        'M17': 'RESISTANCE_QUALITY',
                        'M20': 'FAILED_BREAKOUT',
                        'M21': 'WYCKOFF',
                        'M22': 'INFLATION_REGIME',
                        'M23': 'PPI_SESSION',
                        'M24': 'NBS_PMI',
                        'M25': 'CAIXIN_PMI',
                        'M26': 'EZ_PMI',
                        'M27': 'ISM_MFG',
                        'M28': 'ISM_SVC',
                        'M29': 'UNKNOWN',
                        'M30': 'UNKNOWN',
                        'M31': 'UK_CPI',
                        'M32': 'UK_WAGES',
                        'M33': 'RETAIL_SALES',
                        'M34': 'HOUSING_STARTS',
                        'M35': 'PBC_LPR',
                        'M36': 'ADP_EMPLOYMENT',
                        'M37': 'NFP',
                        'M38': 'IFO',
                        'M39': 'UMS',
                        'M40': 'GERMANY_CPI',
                        'M41': 'EZ_CPI',
                        'M42': 'EZ_GDP',
                        'M43': 'US_GDP',
                        'M44': 'DURABLES',
                        'M45': 'PCE',
                        'M46': 'JAPAN_CPI',
                        'M47': 'BOJ_RATE',
                        'M48': 'ECB_RATE',
                        'M49': 'BOE_RATE',
                        'M50': 'CB_CONFIDENCE',
                        'M51': 'UK_GDP',
                        'M52': 'RBA_RATE',
                        'M53': 'AU_CPI',
                        'M54': 'CHINA_GDP',
                        'M55': 'TREASURY_AUCTION',
                        'M56': 'US_CPI',
                        'M57': 'FOMC',
                        'M58': 'POWELL_PRESSER',
                        'M59': 'FOMC_MINUTES',
                        'M60': 'US_PPI',
                        'M61': 'US_CLAIMS',
                        'M62': 'US_UNEMPLOYMENT'
                    }
                    best_strategy_type = name_map.get(best_strategy_name, 'UNKNOWN')
                    # Direction from the strategy result if available
                    best_direction = val.get('direction', 'NEUTRAL')
                    # Conviction maybe as percentage? We'll use score*100
                    best_conviction = int(score * 100) if score is not None else None
                    # Entry, SL, TP1 from the strategy if present
                    best_entry = val.get('entry')
                    best_sl = val.get('sl')
                    best_tp1 = val.get('tp1')
                    # RR1 maybe compute?
                    if best_entry is not None and best_sl is not None and best_tp1 is not None:
                        if best_direction == 'LONG':
                            risk = abs(best_entry - best_sl)
                            reward = abs(best_tp1 - best_entry)
                        else:
                            risk = abs(best_sl - best_entry)
                            reward = abs(best_entry - best_tp1)
                        if risk > 0:
                            best_rr1 = reward / risk
                        else:
                            best_rr1 = None
                    else:
                        best_rr1 = None
    
    # If no best found, set defaults
    if best_strategy_name is None:
        best_strategy_name = 'NONE'
        best_strategy_type = 'NONE'
        best_direction = 'NEUTRAL'
        best_conviction = 0
        best_entry = None
        best_sl = None
        best_tp1 = None
        best_rr1 = None
    
    # Order Flow
    ob_imbalance = get(data, ['ob_imbalance'])
    ob_consensus = get(data, ['ob_consensus'])
    taker = get(data, ['taker'])
    net_flow = get(data, ['net_flow'])
    
    # Build the report
    report_lines = []
    report_lines.append(f"### 🚨 Status: {signal_status}")
    report_lines.append(f"*Directional Bias:* `{directional_bias}` ([{source_reason}: Reason])")
    # Primary Blocker: plain English, no module IDs
    if signal_status_field == "NO_SIGNAL":
        if blocker_explanation:
            primary_blocker = blocker_explanation
        else:
            primary_blocker = "Unknown blocker"
    else:
        # Determine blocker from status if not NO_SIGNAL but still blocked? e.g., M20_BLOCKED
        if signal_status_field == "M20_BLOCKED":
            primary_blocker = "Signal blocked by failed breakout condition"
        elif signal_status_field == "SWEEP_BLOCKED":
            primary_blocker = "Signal blocked by sweep condition"
        else:
            primary_blocker = "None"
    report_lines.append(f"*Primary Blocker:* {primary_blocker}")
    report_lines.append(f"*ICS Score:* `{ics_score:.3f}`")
    report_lines.append("")
    report_lines.append("### 🛡️ Signal Filters (explain WHY if null — no signal = filters not applied)")
    report_lines.append(f"*Signal Status:* `{signal_status_field}` — {'explain blocker: ' + blocker_explanation if signal_status_field == 'NO_SIGNAL' else 'Signal present'}")
    sweep_val = sweep_filter if sweep_filter is not None else ("Not applied (no signal)" if signal_status_field == "NO_SIGNAL" else "N/A")
    report_lines.append(f"*Sweep Filter:* `{sweep_val}`")
    m20_val = m20_filter if m20_filter is not None else ("Not applied (no signal)" if signal_status_field == "NO_SIGNAL" else "N/A")
    report_lines.append(f"*M20 Filter:* `{m20_val}`")
    m20_entry_val = m20_entry_level if m20_entry_level is not None else ("No M20 level available" if signal_status_field == "NO_SIGNAL" else "N/A")
    report_lines.append(f"*M20 Entry Level:* `{m20_entry_val}`")
    report_lines.append(f"*Sweep Blocked:* `{sweep_blocked}` | *M20 Blocked:* `{m20_blocked}` | *Ensemble Passes:* `{ensemble_passes}`")
    report_lines.append(f"*ICS Score:* `{ics:.3f}` vs threshold 0.50 — {'ICS too low for signal' if ics < 0.5 else 'OK'}")
    report_lines.append("")
    report_lines.append("### 🎯 Ensemble Gate")
    report_lines.append(f"*Consensus:* `{conclusion if (conclusion := consensus) is not None else 'N/A'}` | *Passes:* `{passes if passes is not None else 'N/A'}`")
    report_lines.append(f"*Agree Count:* `{agree_count if agree_count is not None else 'N/A'}` strategies | *Conviction:* `{conviction if conviction is not None else 'N/A'}`")
    report_lines.append(f"*Regime:* `{regime if regime is not None else 'N/A'}`")
    report_lines.append("")
    report_lines.append("### ⏳ Confirmation Status")
    report_lines.append(f"*Signal Status:* `{confirmation_status if confirmation_status is not None else 'N/A'}`")
    report_lines.append(f"*Bars to Confirm:* `{bars_to_confirm if bars_to_confirm is not None else 'N/A'}` | *Hold Window:* `{hold_window_hours if hold_window_hours is not None else 'N/A'}h`")
    report_lines.append("")
    report_lines.append("### 📈 Exchange Activity (MUST include ALL these from derivatives + exchange_activity)")
    report_lines.append(f"*Price:* `${price:.2f}` | *EMA200:* `{ema_200:.2f}`")
    oi_roc_pct = oi_roc_1h * 100
    oi_roc_flag = "flag" if abs(oi_roc_1h) > 0.001 else "ok"
    report_lines.append(f"*OI:* `{int(oi):,}` (${oi_usd:,.0f}) | *OI ROC 1h:* `{oi_roc_pct:.2f}%` — {oi_roc_flag}")
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
    report_lines.append(f"*Strategies Fired:* `{strategies_fired}`/`{total_strategies}`")
    report_lines.append(f"*Best Strategy:* {best_strategy_name} ({best_strategy_type}) | *Direction:* {best_direction} | *Conviction:* {best_conviction}%")
    entry_str = f"${best_entry:.2f}" if isinstance(best_entry, (int, float)) else "N/A"
    sl_str = f"${best_sl:.2f}" if isinstance(best_sl, (int, float)) else "N/A"
    tp1_str = f"${best_tp1:.2f}" if isinstance(best_tp1, (int, float)) else "N/A"
    rr1_str = f"{best_rr1:.2f}" if isinstance(best_rr1, (float, int)) else "N/A"
    report_lines.append(f"*Entry:* {entry_str} | *SL:* {sl_str} | *TP1:* {tp1_str} | *R:R:* {rr1_str}")
    report_lines.append("")
    report_lines.append("### 📊 Order Flow")
    ob_imp_str = f"`{ob_imbalance}`" if ob_imbalance is not None else "`N/A`"
    ob_con_str = f"({ob_consensus})" if ob_consensus is not None else "()"
    report_lines.append(f"*OB Imbalance:* {ob_imp_str} {ob_con_str}")
    taker_str = f"`{taker}`" if taker is not None else "`N/A`"
    net_str = f"`{net_flow}`" if net_flow is not None else "`N/A`"
    report_lines.append(f"*Trade Taker:* {taker_str} | *Net Flow:* {net_str}")
    report_lines.append("")
    report_lines.append("### 📝 Narrative")
    # Generate a simple narrative
    narrative = f"The market shows a {directional_bias} bias with {signal_status_field} signal. "
    narrative += f"ICS is {ics_score:.3f} ({'below' if ics_score < 0.5 else 'above'} threshold). "
    narrative += f"Price is ${price:.2f} vs EMA200 ${ema_200:.2f}. "
    narrative += f"OI is {int(oi):,} (${oi_usd:,.0f}) with {oi_roc_pct:.2f}% hourly change. "
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