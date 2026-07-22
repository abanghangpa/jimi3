
"""
Ensemble Gate - requires multiple strategies to agree before signaling.
Combined with confirmation layer: consensus + market confirmation = high WR.
"""
MIN_AGREE = 1
MIN_CONVICTION = 0.3

STRATEGY_WEIGHTS = {
    "mtf_confluence": 1.5, "structural_break": 1.3,"orderbook_imbalance": 1.2,
    "trade_flow": 1.2, "cross_asset": 1.2, "funding_arb": 1.1,
    "failed_breakout": 0.8, "regime_switch": 0.9, "scalp_v2": 0.7,
    "squeeze_breakout": 0.5, "positioning_fade": 0.5, "kill_zone": 1.0,
    "liquidity_grab": 1.0, "taker_flow": 0.8, "power_of_3": 1.0,
    "cascade": 1.0, "macro_surprise": 1.0, "whale_watch": 1.0,
    "vol_rotation": 1.0, "liquidation_cascade": 1.0, "judas_sweep": 1.0,
    "momentum_v2": 1.0,
}

def evaluate_ensemble(strategy_signals, m10_details=None):
    if not strategy_signals:
        return _empty()
    # Apply regime-based weight adjustments if m10 data available
    regime_result = None
    if m10_details:
        try:
            regime_result = apply_regime_weights(strategy_signals, m10_details)
            strategy_signals = regime_result['adjusted_signals']
        except Exception:
            pass
    
    fired = [s for s in strategy_signals if s.get("direction") and s.get("conviction",0) >= MIN_CONVICTION]
    if not fired:
        return _empty()
    long_score = short_score = 0.0
    long_strats = []
    short_strats = []
    for s in fired:
        strat = s.get("strategy","")
        d = s.get("direction","")
        c = s.get("conviction",0)
        w = STRATEGY_WEIGHTS.get(strat, 1.0)
        wc = c * w
        if d == "LONG":
            long_score += wc
            long_strats.append({"strategy":strat,"conviction":c,"weight":w,"weighted":round(wc,4)})
        elif d == "SHORT":
            short_score += wc
            short_strats.append({"strategy":strat,"conviction":c,"weight":w,"weighted":round(wc,4)})
    lc = len(long_strats)
    sc = len(short_strats)
    if lc >= MIN_AGREE and lc > sc:
        consensus,agree_count,ws,agreeing,disagreeing = "LONG",lc,long_score,long_strats,short_strats
    elif sc >= MIN_AGREE and sc > lc:
        consensus,agree_count,ws,agreeing,disagreeing = "SHORT",sc,short_score,short_strats,long_strats
    else:
        return {"consensus":"NONE","agree_count":max(lc,sc),"total_fired":len(fired),
                "weighted_score":max(long_score,short_score),"agreeing_strategies":long_strats if lc>sc else short_strats,
                "disagreeing_strategies":short_strats if lc>sc else long_strats,
                "ensemble_conviction":0.0,"passes":False,
                "reason":"No consensus: %d LONG vs %d SHORT (need %d+)" % (lc, sc, MIN_AGREE)}
    avg = ws/agree_count if agree_count else 0
    ec = min(avg, 1.0)
    result = {"consensus":consensus,"agree_count":agree_count,"total_fired":len(fired),
            "weighted_score":round(ws,4),"agreeing_strategies":agreeing,"disagreeing_strategies":disagreeing,
            "ensemble_conviction":round(ec,4),"passes":True,
            "reason":"Ensemble: %d strategies agree %s (weighted=%.2f)" % (agree_count, consensus, ws)}
    if regime_result:
        result["regime"] = regime_result.get("regime")
        result["regime_blocked"] = regime_result.get("blocked", [])
        result["regime_boosted"] = regime_result.get("boosted", [])
    return result

def _empty():
    return {"consensus":"NONE","agree_count":0,"total_fired":0,"weighted_score":0,
            "agreeing_strategies":[],"disagreeing_strategies":[],"ensemble_conviction":0.0,
            "passes":False,"reason":"No strategies fired"}

def format_ensemble(e):
    if not e.get("passes"):
        return "  ! Ensemble: NO CONSENSUS - %s" % e.get("reason","")
    lines = []
    lines.append("  >> Ensemble: %d strategies agree %s" % (e["agree_count"], e["consensus"]))
    lines.append("     Conviction: %.2f | Weighted: %.2f" % (e["ensemble_conviction"], e["weighted_score"]))
    for s in e.get("agreeing_strategies",[]):
        lines.append("     + %s: %.2f (weight=%.1f)" % (s["strategy"], s["conviction"], s["weight"]))
    for s in e.get("disagreeing_strategies",[]):
        lines.append("     - %s: %.2f (dissenting)" % (s["strategy"], s["conviction"]))
    return chr(10).join(lines)
