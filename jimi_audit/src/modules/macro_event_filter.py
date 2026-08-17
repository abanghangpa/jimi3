"""
Macro Event Pre-Filter — gates scanner entries around macro data releases.

Uses backtested session transmission data to:
  1. Detect if a macro event just fired (within session cascade window)
  2. Apply regime filters (Phase0, 30d trend, surprise classification)
  3. Adjust size multiplier or block entries entirely

Usage:
    from src.modules.macro_event_filter import MacroEventFilter
    mef = MacroEventFilter(config=cfg)
    result = mef.check(current_time, df_15m, idx, direction)
    if result['blocked']:
        print(f"Blocked: {result['reason']}")
    elif result['size_mult'] < 1.0:
        print(f"Reduced size: {result['size_mult']}")
"""

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

UTC = timezone.utc

# ══════════════════════════════════════════════════════════════
# MACRO EVENT CALENDAR — key release dates and times
# ══════════════════════════════════════════════════════════════

# Approximate release schedule (day of month, hour UTC)
# These are used to detect proximity to macro events
MACRO_EVENTS = [
    {'id': 'caixin_mfg', 'name': 'Caixin Mfg PMI', 'day': 1, 'hour': 1, 'minute': 45,
     'country': 'CN', 'impact': 'MEDIUM', 'cascade_hours': 30},
    {'id': 'nbs_pmi', 'name': 'NBS PMI', 'day': 1, 'hour': 1, 'minute': 0,
     'country': 'CN', 'impact': 'MEDIUM', 'cascade_hours': 6},
    {'id': 'us_nfp', 'name': 'NFP', 'day': 'first_friday', 'hour': 13, 'minute': 30,
     'country': 'US', 'impact': 'HIGH', 'cascade_hours': 48},
    {'id': 'us_cpi', 'name': 'US CPI', 'day': 12, 'hour': 13, 'minute': 30,
     'country': 'US', 'impact': 'HIGHEST', 'cascade_hours': 48},
    {'id': 'us_ppi', 'name': 'US PPI', 'day': 13, 'hour': 13, 'minute': 30,
     'country': 'US', 'impact': 'MEDIUM', 'cascade_hours': 24},
    {'id': 'us_fomc', 'name': 'FOMC', 'day': 'fomc_schedule', 'hour': 19, 'minute': 0,
     'country': 'US', 'impact': 'HIGH', 'cascade_hours': 48},
    {'id': 'cn_pboc_lpr', 'name': 'PBoC LPR', 'day': 20, 'hour': 1, 'minute': 30,
     'country': 'CN', 'impact': 'HIGH', 'cascade_hours': 48},
    {'id': 'eu_ecb', 'name': 'ECB', 'day': 'ecb_schedule', 'hour': 13, 'minute': 15,
     'country': 'EU', 'impact': 'HIGH', 'cascade_hours': 48},
    {'id': 'jp_boj', 'name': 'BoJ', 'day': 'boj_schedule', 'hour': 3, 'minute': 0,
     'country': 'JP', 'impact': 'HIGH', 'cascade_hours': 48},
    {'id': 'us_claims', 'name': 'Jobless Claims', 'day': 'thursday', 'hour': 13, 'minute': 30,
     'country': 'US', 'impact': 'MEDIUM', 'cascade_hours': 8},
    {'id': 'us_pce', 'name': 'Core PCE', 'day': 28, 'hour': 13, 'minute': 30,
     'country': 'US', 'impact': 'HIGH', 'cascade_hours': 48},
]


def _first_friday(year, month):
    d = datetime(year, month, 1, tzinfo=UTC)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def _fomc_dates_2026():
    """Known FOMC dates for 2026."""
    return [
        datetime(2026, 1, 29, 19, 0, tzinfo=UTC),
        datetime(2026, 3, 19, 19, 0, tzinfo=UTC),
        datetime(2026, 5, 7, 19, 0, tzinfo=UTC),
        datetime(2026, 6, 18, 19, 0, tzinfo=UTC),
        datetime(2026, 7, 30, 19, 0, tzinfo=UTC),
        datetime(2026, 9, 18, 19, 0, tzinfo=UTC),
        datetime(2026, 10, 29, 19, 0, tzinfo=UTC),
        datetime(2026, 12, 17, 19, 0, tzinfo=UTC),
    ]


def _ecb_dates_2026():
    """Known ECB dates for 2026."""
    return [
        datetime(2026, 2, 5, 13, 15, tzinfo=UTC),
        datetime(2026, 4, 16, 13, 15, tzinfo=UTC),
        datetime(2026, 6, 4, 13, 15, tzinfo=UTC),
        datetime(2026, 8, 6, 13, 15, tzinfo=UTC),
        datetime(2026, 10, 29, 13, 15, tzinfo=UTC),
        datetime(2026, 12, 17, 13, 15, tzinfo=UTC),
    ]


def _boj_dates_2026():
    """Known BoJ dates for 2026."""
    return [
        datetime(2026, 2, 19, 3, 0, tzinfo=UTC),
        datetime(2026, 4, 17, 3, 0, tzinfo=UTC),
        datetime(2026, 6, 18, 3, 0, tzinfo=UTC),
        datetime(2026, 8, 6, 3, 0, tzinfo=UTC),
        datetime(2026, 10, 30, 3, 0, tzinfo=UTC),
        datetime(2026, 12, 18, 3, 0, tzinfo=UTC),
    ]


def get_next_macro_events(reference_time=None, lookback_hours=4, lookahead_hours=2):
    """Find macro events within a window around the current time.

    Returns list of events that recently fired or are about to fire,
    with their cascade status.
    """
    now = reference_time or datetime.now(UTC)
    if hasattr(now, 'tzinfo') and now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    elif hasattr(now, 'tzlocalize'):
        now = now.tz_localize(UTC) if now.tzinfo is None else now.tz_convert(UTC)
    
    year, month = now.year, now.month

    window_start = now - timedelta(hours=lookback_hours)
    window_end = now + timedelta(hours=lookahead_hours)

    nearby = []

    for evt in MACRO_EVENTS:
        release_dt = None
        day = evt['day']

        if isinstance(day, int):
            release_dt = datetime(year, month, day, evt['hour'], evt['minute'], tzinfo=UTC)
            # Check next month if past
            if release_dt < now - timedelta(days=2):
                nm = month + 1 if month < 12 else 1
                ny = year if month < 12 else year + 1
                release_dt = datetime(ny, nm, day, evt['hour'], evt['minute'], tzinfo=UTC)
        elif day == 'first_friday':
            release_dt = _first_friday(year, month).replace(hour=evt['hour'], minute=evt['minute'], tzinfo=UTC)
            if release_dt < now - timedelta(days=2):
                nm = month + 1 if month < 12 else 1
                ny = year if month < 12 else year + 1
                release_dt = _first_friday(ny, nm).replace(hour=evt['hour'], minute=evt['minute'], tzinfo=UTC)
        elif day == 'thursday':

            # Find next Thursday
            d = now
            while d.weekday() != 3:
                d += timedelta(days=1)
            release_dt = d.replace(hour=evt['hour'], minute=evt['minute'], second=0, microsecond=0, tzinfo=UTC)
        elif day == 'fomc_schedule':
            for dt in _fomc_dates_2026():
                if window_start <= dt <= window_end:
                    release_dt = dt
                    break
        elif day == 'ecb_schedule':
            for dt in _ecb_dates_2026():
                if window_start <= dt <= window_end:
                    release_dt = dt
                    break
        elif day == 'boj_schedule':
            for dt in _boj_dates_2026():
                if window_start <= dt <= window_end:
                    release_dt = dt
                    break

        if release_dt is None:
            continue

        # Check if event is within our window
        if window_start <= release_dt <= window_end:
            delta = now - release_dt
            hours_since = delta.total_seconds() / 3600
            cascade_hours = evt.get('cascade_hours', 24)

            if hours_since >= 0:
                # Event already fired
                phase = 'ACTIVE' if hours_since < cascade_hours else 'RESOLVED'
                cascade_pct = min(hours_since / cascade_hours, 1.0)
            else:
                # Event upcoming
                phase = 'UPCOMING'
                cascade_pct = 0

            nearby.append({
                'id': evt['id'],
                'name': evt['name'],
                'country': evt['country'],
                'impact': evt['impact'],
                'release_dt': release_dt,
                'hours_since': round(hours_since, 1),
                'hours_until': round(-hours_since, 1) if hours_since < 0 else 0,
                'cascade_hours': cascade_hours,
                'cascade_pct': round(cascade_pct, 2),
                'phase': phase,
            })

    nearby.sort(key=lambda e: abs(e['hours_since']))
    return nearby


class MacroEventFilter:
    """Pre-filter for scanner entries around macro data releases.

    Backtested rules:
      - Phase0 < 0.15 (DEATH_ZONE): block entries
      - Phase0 0.15-0.30 (LOW): allow, reduced size
      - Phase0 > 0.70 (STRONG): allow, normal/boosted size
      - 30d trend STRONG_DOWN entering: caution
      - 30d trend SLIGHT_DOWN entering: best setup
      - Active macro cascade: adjust size based on session phase
    """

    def __init__(self, config=None):
        cfg = config or {}
        self.enabled = cfg.get('MACRO_EVENT_FILTER_ENABLED', True)
        self.phase0_block_threshold = cfg.get('PHASE0_MAX_BLOCK', 0.30)  # 1h Phase0 > 0.5 = 59.4% WR
        self.phase0_caution_threshold = cfg.get('PHASE0_CAUTION', 0.15)  # Match block threshold
        self.cascade_size_mult = cfg.get('MACRO_CASCADE_SIZE_MULT', 0.70)
        self.lookback_hours = cfg.get('MACRO_LOOKBACK_HOURS', 4)
        self.lookahead_hours = cfg.get('MACRO_LOOKAHEAD_HOURS', 2)

    def check(self, current_time, df_15m, idx, direction, phase0=None, trend_30d=None):
        """Check if entry should be blocked or size-adjusted around macro events.

        Args:
            current_time: Current datetime (UTC)
            df_15m: 15m DataFrame
            idx: Current bar index
            direction: 'LONG' or 'SHORT'
            phase0: Phase0 value (optional, computed from df_1d if not provided)
            trend_30d: 30-day price trend % (optional)

        Returns:
            dict with keys: blocked, reason, size_mult, active_events, regime_notes
        """
        result = {
            'blocked': False,
            'reason': '',
            'size_mult': 1.0,
            'active_events': [],
            'regime_notes': [],
        }

        if not self.enabled:
            return result

        # ── 1. Phase0 filter (backtested: DEATH_ZONE = 43% WR) ──
        if phase0 is not None:
            if phase0 > self.phase0_block_threshold:
                result['blocked'] = True
                result['reason'] = f'Phase0_1h={phase0:.3f} CROWDED (>{self.phase0_block_threshold}) — high positioning noise, reduced edge'
                result['regime_notes'].append(f'Phase0 CROWDED_ZONE: {phase0:.3f}')
                return result
            elif phase0 > self.phase0_caution_threshold:
                result['size_mult'] *= 0.70
                result['regime_notes'].append(f'Phase0 ACTIVE: {phase0:.3f} — reduced size')

        # ── 2. 30-day trend filter (backtested: SLIGHT_DOWN = best, STRONG_DOWN = avoid) ──
        if trend_30d is not None:
            if trend_30d < -10:
                result['size_mult'] *= 0.80
                result['regime_notes'].append(f'30d trend STRONG_DOWN ({trend_30d:+.1f}%) — caution')
            elif -5 < trend_30d < -2:
                # SLIGHT_DOWN = best setup (+1.62% avg, 71% WR)
                result['regime_notes'].append(f'30d trend SLIGHT_DOWN ({trend_30d:+.1f}%) — best setup')
            elif trend_30d > 10:
                result['regime_notes'].append(f'30d trend STRONG_UP ({trend_30d:+.1f}%) — momentum')

        # ── 3. Active macro cascade filter ──
        active_events = get_next_macro_events(
            reference_time=current_time,
            lookback_hours=self.lookback_hours,
            lookahead_hours=self.lookahead_hours,
        )

        for evt in active_events:
            if evt['phase'] == 'ACTIVE':
                # Event recently fired, still in cascade window
                impact = evt['impact']
                cascade_pct = evt['cascade_pct']

                if impact == 'HIGHEST':
                    # CPI/FOMC/Powell — regime breakers
                    if cascade_pct < 0.25:
                        result['blocked'] = True
                        result['reason'] = f'{evt["name"]} just fired ({evt["hours_since"]:+.1f}h) — regime breaker, wait for dust to settle'
                        result['active_events'].append(evt)
                        return result
                    else:
                        result['size_mult'] *= self.cascade_size_mult
                        result['regime_notes'].append(f'{evt["name"]} cascade {cascade_pct:.0%} — reduced size')

                elif impact == 'HIGH':
                    # NFP/PBoC/ECB/BoJ — directional triggers
                    if cascade_pct < 0.15:
                        result['size_mult'] *= 0.50
                        result['regime_notes'].append(f'{evt["name"]} just fired ({evt["hours_since"]:+.1f}h) — high impact, reduced size')
                    else:
                        result['size_mult'] *= 0.80
                        result['regime_notes'].append(f'{evt["name"]} cascade {cascade_pct:.0%}')

                elif impact == 'MEDIUM':
                    # PMI/Claims/PPI — micro catalysts
                    result['regime_notes'].append(f'{evt["name"]} active ({evt["hours_since"]:+.1f}h)')

            elif evt['phase'] == 'UPCOMING':
                # Event about to fire
                hours_until = evt['hours_until']
                if hours_until < 1 and evt['impact'] in ('HIGHEST', 'HIGH'):
                    result['size_mult'] *= 0.60
                    result['regime_notes'].append(f'{evt["name"]} in {hours_until:.1f}h — pre-event caution')

        # ── 4. BoJ special rule (carry unwind = #1 tail risk) ──
        for evt in active_events:
            if evt['id'] == 'jp_boj' and evt['phase'] == 'ACTIVE' and evt['hours_since'] < 4:
                result['size_mult'] *= 0.30
                result['regime_notes'].append('BoJ just fired — carry unwind risk, minimal size')

        return result


def get_phase0_from_df(df_1d, idx_1d):
    """Extract Phase0 value from daily DataFrame."""
    if 'phase0' in df_1d.columns and idx_1d < len(df_1d):
        val = df_1d['phase0'].iloc[idx_1d]
        if not pd.isna(val):
            return float(val)
    return None


def get_trend_30d(df_15m, idx):
    """Compute 30-day price trend from 15m DataFrame."""
    lookback = min(idx, 2880)  # 30 days of 15m bars
    if lookback < 100:
        return None
    price_now = float(df_15m['Close'].iloc[idx])
    price_30d = float(df_15m['Close'].iloc[idx - lookback])
    return (price_now - price_30d) / price_30d * 100
