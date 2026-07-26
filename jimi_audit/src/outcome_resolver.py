"""
JIMI Outcome Resolver — Phase 2 of Optimization Framework
==========================================================

Matches fired signals against actual OHLCV price data to determine
whether each trade hit TP or SL first, computing real PnL.

Algorithm:
  1. Load OHLCV candles sorted by timestamp
  2. For each fired signal, binary-search for the entry candle
  3. Walk forward through candles checking if SL or TP was hit
  4. Same-candle collision: use Open proximity heuristic
  5. Write resolved outcomes to JSONL
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from bisect import bisect_left
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("jimi.outcome")

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    """Single OHLCV candle."""
    timestamp: float       # epoch seconds
    open: float
    high: float
    low: float
    close: float
    volume: float

    @staticmethod
    def from_csv_row(row: dict) -> Candle:
        """Parse from CSV row with 'Open time','Open','High','Low','Close','Volume'."""
        ts_str = row.get("Open time", row.get("open_time", ""))
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            ts = dt.timestamp()
        except ValueError:
            # Try ISO format
            ts = float(ts_str)
        return Candle(
            timestamp=ts,
            open=float(row.get("Open", row.get("open", 0))),
            high=float(row.get("High", row.get("high", 0))),
            low=float(row.get("Low", row.get("low", 0))),
            close=float(row.get("Close", row.get("close", 0))),
            volume=float(row.get("Volume", row.get("volume", 0))),
        )


@dataclass
class Signal:
    """A fired strategy signal to resolve."""
    line_idx: int
    timestamp: str          # original string
    entry_time: float       # epoch seconds
    strategy: str
    direction: str          # LONG | SHORT
    entry: float
    sl: float
    tp: float
    rr: float               # risk:reward ratio
    conviction: float
    price: float            # price at signal time
    symbol: str = "ETH/USDT"


@dataclass
class ResolvedOutcome:
    """Result of resolving a single signal against price data."""
    line_idx: int
    strategy: str
    direction: str
    entry: float
    sl: float
    tp: float
    rr: float
    conviction: float
    signal_time: str
    exit_time: str
    outcome: str            # "tp_hit" | "sl_hit" | "timeout" | "no_data"
    pnl_r: float            # PnL in R-multiples
    bars_held: int
    exit_price: float
    max_favorable: float    # max excursion in favorable direction (R)
    max_adverse: float      # max excursion against (R)


# ---------------------------------------------------------------------------
# OHLCV Loader
# ---------------------------------------------------------------------------

def load_ohlcv(path: str | Path) -> list[Candle]:
    """Load OHLCV candles from CSV, sorted by timestamp."""
    candles = []
    path = Path(path)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                c = Candle.from_csv_row(row)
                candles.append(c)
            except (ValueError, KeyError) as e:
                continue

    candles.sort(key=lambda c: c.timestamp)
    logger.info("Loaded %d candles from %s (%s to %s)",
                len(candles), path,
                datetime.fromtimestamp(candles[0].timestamp).strftime("%Y-%m-%d"),
                datetime.fromtimestamp(candles[-1].timestamp).strftime("%Y-%m-%d"))
    return candles


def load_signals_jsonl(
    path: str | Path,
    strategy_filter: Optional[str] = None,
) -> list[Signal]:
    """Load fired signals from JSONL."""
    signals = []
    path = Path(path)
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not rec.get("fired", False):
                continue

            strategy = rec.get("strategy", "")
            if strategy_filter and strategy != strategy_filter:
                continue

            direction = (rec.get("direction") or "").upper()
            if direction not in ("LONG", "SHORT"):
                continue

            entry = rec.get("entry")
            sl = rec.get("sl")
            tp = rec.get("tp1")
            rr = rec.get("rr1")
            conviction = rec.get("conviction")
            price = rec.get("price")

            if entry is None or sl is None:
                continue

            # Parse timestamp
            ts_str = rec.get("timestamp", "")
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                entry_time = dt.timestamp()
            except ValueError:
                continue

            # Derive TP if missing
            if tp is None and rr and rr > 0:
                risk = abs(entry - sl)
                if direction == "LONG":
                    tp = entry + risk * rr
                else:
                    tp = entry - risk * rr

            if tp is None:
                continue

            signals.append(Signal(
                line_idx=i,
                timestamp=ts_str,
                entry_time=entry_time,
                strategy=strategy,
                direction=direction,
                entry=float(entry),
                sl=float(sl),
                tp=float(tp),
                rr=float(rr) if rr else 0,
                conviction=float(conviction) if conviction else 0,
                price=float(price) if price else float(entry),
            ))

    signals.sort(key=lambda s: s.entry_time)
    logger.info("Loaded %d fired signals from %s (filter=%s)",
                len(signals), path, strategy_filter)
    return signals


# ---------------------------------------------------------------------------
# Outcome Resolution Engine
# ---------------------------------------------------------------------------

class OutcomeResolver:
    """
    Resolves signal outcomes against OHLCV price data.

    For each signal, walks forward through candles to determine
    whether TP or SL was hit first.
    """

    def __init__(
        self,
        candles: list[Candle],
        max_bars: int = 96,         # max bars to hold (96 = 24h for 15m)
        timeout_action: str = "close_at_last",  # close_at_last | skip
    ):
        self.candles = candles
        self.timestamps = [c.timestamp for c in candles]
        self.max_bars = max_bars
        self.timeout_action = timeout_action

    def resolve_all(self, signals: list[Signal]) -> list[ResolvedOutcome]:
        """Resolve all signals against price data."""
        results = []
        for sig in signals:
            result = self.resolve_one(sig)
            results.append(result)

        # Summary stats
        n = len(results)
        tp_hits = sum(1 for r in results if r.outcome == "tp_hit")
        sl_hits = sum(1 for r in results if r.outcome == "sl_hit")
        timeouts = sum(1 for r in results if r.outcome == "timeout")
        no_data = sum(1 for r in results if r.outcome == "no_data")
        logger.info(
            "Resolved %d signals: TP=%d (%.1f%%), SL=%d (%.1f%%), "
            "timeout=%d (%.1f%%), no_data=%d (%.1f%%)",
            n, tp_hits, tp_hits/n*100 if n else 0,
            sl_hits, sl_hits/n*100 if n else 0,
            timeouts, timeouts/n*100 if n else 0,
            no_data, no_data/n*100 if n else 0,
        )
        return results

    def resolve_one(self, sig: Signal) -> ResolvedOutcome:
        """Resolve a single signal."""
        # Find entry candle
        idx = bisect_left(self.timestamps, sig.entry_time)
        if idx >= len(self.candles):
            return self._no_data(sig)

        # Check if we're within a reasonable range of the signal
        candle = self.candles[idx]
        gap = abs(candle.timestamp - sig.entry_time)
        if gap > 3600 * 4:  # > 4 hours from signal
            return self._no_data(sig)

        # Walk forward through candles
        risk = abs(sig.entry - sig.sl)
        if risk <= 0:
            return self._no_data(sig)

        max_favorable = 0.0
        max_adverse = 0.0

        for bar_offset in range(self.max_bars):
            candle_idx = idx + bar_offset
            if candle_idx >= len(self.candles):
                break

            c = self.candles[candle_idx]

            # Calculate excursions
            if sig.direction == "LONG":
                favorable = (c.high - sig.entry) / risk
                adverse = (sig.entry - c.low) / risk
            else:
                favorable = (sig.entry - c.low) / risk
                adverse = (c.high - sig.entry) / risk

            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

            # Check TP and SL
            tp_hit = self._check_level(c, sig.direction, sig.tp, is_tp=True)
            sl_hit = self._check_level(c, sig.direction, sig.sl, is_tp=False)

            if tp_hit and sl_hit:
                # Same-candle collision: use open proximity heuristic
                outcome, exit_price = self._resolve_collision(c, sig)
                exit_dt = datetime.fromtimestamp(c.timestamp)
                return ResolvedOutcome(
                    line_idx=sig.line_idx,
                    strategy=sig.strategy,
                    direction=sig.direction,
                    entry=sig.entry,
                    sl=sig.sl,
                    tp=sig.tp,
                    rr=sig.rr,
                    conviction=sig.conviction,
                    signal_time=sig.timestamp,
                    exit_time=exit_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    outcome=outcome,
                    pnl_r=sig.rr if outcome == "tp_hit" else -1.0,
                    bars_held=bar_offset + 1,
                    exit_price=exit_price,
                    max_favorable=round(max_favorable, 4),
                    max_adverse=round(max_adverse, 4),
                )
            elif tp_hit:
                exit_dt = datetime.fromtimestamp(c.timestamp)
                return ResolvedOutcome(
                    line_idx=sig.line_idx,
                    strategy=sig.strategy,
                    direction=sig.direction,
                    entry=sig.entry,
                    sl=sig.sl,
                    tp=sig.tp,
                    rr=sig.rr,
                    conviction=sig.conviction,
                    signal_time=sig.timestamp,
                    exit_time=exit_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    outcome="tp_hit",
                    pnl_r=sig.rr,
                    bars_held=bar_offset + 1,
                    exit_price=sig.tp,
                    max_favorable=round(max_favorable, 4),
                    max_adverse=round(max_adverse, 4),
                )
            elif sl_hit:
                exit_dt = datetime.fromtimestamp(c.timestamp)
                return ResolvedOutcome(
                    line_idx=sig.line_idx,
                    strategy=sig.strategy,
                    direction=sig.direction,
                    entry=sig.entry,
                    sl=sig.sl,
                    tp=sig.tp,
                    rr=sig.rr,
                    conviction=sig.conviction,
                    signal_time=sig.timestamp,
                    exit_time=exit_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    outcome="sl_hit",
                    pnl_r=-1.0,
                    bars_held=bar_offset + 1,
                    exit_price=sig.sl,
                    max_favorable=round(max_favorable, 4),
                    max_adverse=round(max_adverse, 4),
                )

        # Timeout — max bars reached
        last_c = self.candles[min(idx + self.max_bars - 1, len(self.candles) - 1)]
        exit_dt = datetime.fromtimestamp(last_c.timestamp)
        if sig.direction == "LONG":
            pnl_r = (last_c.close - sig.entry) / risk
        else:
            pnl_r = (sig.entry - last_c.close) / risk

        return ResolvedOutcome(
            line_idx=sig.line_idx,
            strategy=sig.strategy,
            direction=sig.direction,
            entry=sig.entry,
            sl=sig.sl,
            tp=sig.tp,
            rr=sig.rr,
            conviction=sig.conviction,
            signal_time=sig.timestamp,
            exit_time=exit_dt.strftime("%Y-%m-%d %H:%M:%S"),
            outcome="timeout",
            pnl_r=round(pnl_r, 4),
            bars_held=self.max_bars,
            exit_price=last_c.close,
            max_favorable=round(max_favorable, 4),
            max_adverse=round(max_adverse, 4),
        )

    def _check_level(self, candle: Candle, direction: str, level: float, is_tp: bool) -> bool:
        """Check if a candle touched the given price level."""
        if direction == "LONG":
            if is_tp:
                return candle.high >= level
            else:
                return candle.low <= level
        else:  # SHORT
            if is_tp:
                return candle.low <= level
            else:
                return candle.high >= level

    def _resolve_collision(self, candle: Candle, sig: Signal) -> tuple[str, float]:
        """
        Same candle touched both TP and SL. Use open proximity heuristic:
        - If Open is closer to SL → SL hit first → loss
        - If Open is closer to TP → TP hit first → win
        """
        dist_to_sl = abs(candle.open - sig.sl)
        dist_to_tp = abs(candle.open - sig.tp)

        if dist_to_sl < dist_to_tp:
            return "sl_hit", sig.sl
        else:
            return "tp_hit", sig.tp

    def _no_data(self, sig: Signal) -> ResolvedOutcome:
        return ResolvedOutcome(
            line_idx=sig.line_idx,
            strategy=sig.strategy,
            direction=sig.direction,
            entry=sig.entry,
            sl=sig.sl,
            tp=sig.tp,
            rr=sig.rr,
            conviction=sig.conviction,
            signal_time=sig.timestamp,
            exit_time="",
            outcome="no_data",
            pnl_r=0.0,
            bars_held=0,
            exit_price=0.0,
            max_favorable=0.0,
            max_adverse=0.0,
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outcomes(outcomes: list[ResolvedOutcome], output_path: str | Path) -> Path:
    """Save resolved outcomes as JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for o in outcomes:
            f.write(json.dumps(asdict(o)) + "\n")

    logger.info("Saved %d outcomes to %s", len(outcomes), output_path)
    return output_path


def save_summary(outcomes: list[ResolvedOutcome], output_path: str | Path) -> Path:
    """Save human-readable summary report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate by strategy
    strats: dict[str, list[ResolvedOutcome]] = {}
    for o in outcomes:
        strats.setdefault(o.strategy, []).append(o)

    lines = []
    lines.append("=" * 70)
    lines.append("  JIMI OUTCOME RESOLUTION REPORT")
    lines.append(f"  Generated: {datetime.utcnow().isoformat()}")
    lines.append("=" * 70)
    lines.append("")

    for strat, results in sorted(strats.items()):
        n = len(results)
        if n == 0:
            continue

        tp = sum(1 for r in results if r.outcome == "tp_hit")
        sl = sum(1 for r in results if r.outcome == "sl_hit")
        to = sum(1 for r in results if r.outcome == "timeout")
        nd = sum(1 for r in results if r.outcome == "no_data")
        resolved = [r for r in results if r.outcome in ("tp_hit", "sl_hit")]
        wr = tp / len(resolved) * 100 if resolved else 0

        pnls = [r.pnl_r for r in results if r.outcome != "no_data"]
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        gp = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        pf = gp / gl if gl > 0 else float("inf")

        avg_bars = sum(r.bars_held for r in resolved) / len(resolved) if resolved else 0

        lines.append(f"  📊 {strat}")
        lines.append(f"     Total signals: {n}")
        lines.append(f"     TP hit: {tp} ({tp/n*100:.1f}%)")
        lines.append(f"     SL hit: {sl} ({sl/n*100:.1f}%)")
        lines.append(f"     Timeout: {to} ({to/n*100:.1f}%)")
        lines.append(f"     No data: {nd} ({nd/n*100:.1f}%)")
        lines.append(f"     Win rate (resolved): {wr:.1f}%")
        lines.append(f"     Avg PnL (R): {avg_pnl:.3f}")
        lines.append(f"     Profit factor: {pf:.2f}")
        lines.append(f"     Avg bars held: {avg_bars:.1f}")
        lines.append("")

        # Conviction breakdown
        high_conv = [r for r in resolved if r.conviction >= 0.7]
        low_conv = [r for r in resolved if r.conviction < 0.7]
        if high_conv:
            hc_wr = sum(1 for r in high_conv if r.outcome == "tp_hit") / len(high_conv) * 100
            lines.append(f"     High conviction (≥0.7): {len(high_conv)} trades, WR={hc_wr:.1f}%")
        if low_conv:
            lc_wr = sum(1 for r in low_conv if r.outcome == "tp_hit") / len(low_conv) * 100
            lines.append(f"     Low conviction (<0.7):  {len(low_conv)} trades, WR={lc_wr:.1f}%")
        lines.append("")

    # Overall
    all_resolved = [r for r in outcomes if r.outcome in ("tp_hit", "sl_hit")]
    all_tp = sum(1 for r in outcomes if r.outcome == "tp_hit")
    overall_wr = all_tp / len(all_resolved) * 100 if all_resolved else 0
    all_pnls = [r.pnl_r for r in outcomes if r.outcome != "no_data"]

    lines.append(f"  {'=' * 50}")
    lines.append(f"  OVERALL: {len(outcomes)} signals, {len(all_resolved)} resolved")
    lines.append(f"  Win rate: {overall_wr:.1f}%")
    lines.append(f"  Avg PnL: {sum(all_pnls)/len(all_pnls):.3f} R" if all_pnls else "  Avg PnL: N/A")
    lines.append("=" * 70)

    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)

    logger.info("Summary saved to %s", output_path)
    return output_path
