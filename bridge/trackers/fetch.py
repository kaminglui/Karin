"""Per-source fetchers.

Each fetcher is a function `(config, client) -> TrackerReading` that either
returns one observation or raises FetchError. The dispatch registry
`FETCHERS` maps a `TrackerConfig.source` string to the function; adding
a new source is: implement the function, add it to FETCHERS, reference
it from the config. No code in service.py or store.py needs to change.

Swap strategy (per user note about Stooq):
The abstraction is deliberately thin — fetchers don't share state or
talk to the store. To swap gold from Stooq to e.g. LBMA, implement a
new fetch_lbma(), register it, and change `source: "stooq"` to
`source: "lbma"` in trackers.json. Tests for the old fetcher can stay
in place since its function still exists.
"""
from __future__ import annotations

import calendar
import csv
import logging
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Callable

import httpx

from bridge.trackers.models import TrackerConfig, TrackerReading

log = logging.getLogger("bridge.trackers.fetch")

# How many days back to pull on first-time history backfill (daily series).
# 90 covers 1m/1w/1d deltas with buffer and fully populates Phase 5.2's
# shock/movement windows (~26 days max) on the first call.
_BACKFILL_DAILY_DAYS = 90

# Tunable via config/tuning.yaml → trackers.fetch.fetch_timeout_s.
from bridge import tuning as _tuning
FETCH_TIMEOUT = _tuning.get("trackers.fetch.fetch_timeout_s", 10.0)


# State (two-letter US code) → EIA PADD "duoarea" code for weekly
# retail gasoline. Groupings come from EIA's official PADD allocation:
# https://www.eia.gov/tools/faqs/faq.php?id=78
# All 50 states + DC covered. Territories (PR, GU, ...) fall through
# to R10 (US total).
STATE_TO_PADD: dict[str, str] = {
    # PADD 1A — New England
    "CT": "R1X", "ME": "R1X", "MA": "R1X", "NH": "R1X", "RI": "R1X", "VT": "R1X",
    # PADD 1B — Central Atlantic
    "DE": "R1Y", "DC": "R1Y", "MD": "R1Y", "NJ": "R1Y", "NY": "R1Y", "PA": "R1Y",
    # PADD 1C — Lower Atlantic
    "FL": "R1Z", "GA": "R1Z", "NC": "R1Z", "SC": "R1Z", "VA": "R1Z", "WV": "R1Z",
    # PADD 2 — Midwest
    "IL": "R20", "IN": "R20", "IA": "R20", "KS": "R20", "KY": "R20", "MI": "R20",
    "MN": "R20", "MO": "R20", "ND": "R20", "NE": "R20", "OH": "R20", "OK": "R20",
    "SD": "R20", "TN": "R20", "WI": "R20",
    # PADD 3 — Gulf Coast
    "AL": "R30", "AR": "R30", "LA": "R30", "MS": "R30", "NM": "R30", "TX": "R30",
    # PADD 4 — Rocky Mountain
    "CO": "R40", "ID": "R40", "MT": "R40", "UT": "R40", "WY": "R40",
    # PADD 5 — West Coast
    "AK": "R50", "AZ": "R50", "CA": "R50", "HI": "R50", "NV": "R50",
    "OR": "R50", "WA": "R50",
}

# Fallback when no home_state is configured or it doesn't map. R1Y
# (Central Atlantic, PA/NJ/NY/MD/DE/DC) is a reasonable default for a
# developer-in-PA baseline — chosen to match the original config.
DEFAULT_PADD = "R1Y"


def state_to_padd(state_code: str | None) -> str:
    """Resolve a two-letter US state code to its EIA PADD duoarea.

    Unknown or empty input returns :data:`DEFAULT_PADD` rather than
    raising — this is a UI-default helper, not a validator.
    """
    if not state_code:
        return DEFAULT_PADD
    return STATE_TO_PADD.get(state_code.strip().upper(), DEFAULT_PADD)


class FetchError(Exception):
    """Fetcher-layer failure. Raised for HTTP, parse, or schema errors.

    The service catches this and records the message in
    TrackerRecord.last_fetch_error, so users can see why a tracker is
    empty or not updating without reading logs.
    """


# --- Frankfurter (FX) ------------------------------------------------------

def fetch_frankfurter(config: TrackerConfig, client: httpx.Client) -> TrackerReading:
    """USD -> target FX rate from Frankfurter (ECB-based, no API key).

    Endpoint: https://api.frankfurter.dev/v1/latest?base=USD&symbols=CNY

    Params expected in config:
        {"from": "USD", "to": "CNY"}

    Response shape:
        {"amount": 1.0, "base": "USD", "date": "YYYY-MM-DD",
         "rates": {"CNY": 7.23}}

    Note: ECB publishes a date only (no intraday time). We stamp the
    reading at 15:00 UTC of that date — approximately when ECB's ~16:00
    CET reference fix is posted. Good enough for daily-cadence comparison;
    don't treat it as tick-level.
    """
    from_ccy = config.params.get("from", "USD")
    to_ccy = config.params.get("to")
    if not to_ccy:
        raise FetchError(f"frankfurter: missing 'to' param in tracker {config.id}")
    url = f"https://api.frankfurter.dev/v1/latest?base={from_ccy}&symbols={to_ccy}"
    try:
        resp = client.get(url, timeout=FETCH_TIMEOUT)
    except httpx.HTTPError as e:
        raise FetchError(f"frankfurter: http error: {e}") from e
    if resp.status_code != 200:
        raise FetchError(f"frankfurter: status {resp.status_code}: {resp.text[:200]}")
    try:
        data = resp.json()
    except ValueError as e:
        raise FetchError(f"frankfurter: invalid JSON: {e}") from e
    rates = data.get("rates") or {}
    rate = rates.get(to_ccy)
    if rate is None:
        raise FetchError(f"frankfurter: no rate for {to_ccy} in response: {data!r}")
    date_str = data.get("date")
    if not date_str:
        raise FetchError(f"frankfurter: no date in response: {data!r}")
    try:
        base_dt = datetime.fromisoformat(date_str)
    except ValueError as e:
        raise FetchError(f"frankfurter: bad date {date_str!r}: {e}") from e
    ts = base_dt.replace(hour=15, minute=0, second=0, tzinfo=timezone.utc)
    return TrackerReading(timestamp=ts, value=float(rate))


# --- Stooq (gold spot proxy) -----------------------------------------------

def fetch_stooq(config: TrackerConfig, client: httpx.Client) -> TrackerReading:
    """Daily close from Stooq's CSV endpoint (no API key).

    Endpoint: https://stooq.com/q/l/?s=<symbol>&f=sd2t2ohlcv&h&e=csv
    Params:  {"symbol": "xauusd"}

    Stooq is a convenient no-key source for spot gold in USD, but it's
    a daily-close / spot-proxy — NOT an official benchmark feed (LBMA
    PM fixing etc.). Treat values as tracking-grade. If higher fidelity
    is ever needed, implement a fetch_lbma() and swap the source key.

    CSV columns: Symbol,Date,Time,Open,High,Low,Close,Volume
    Stooq returns "N/D" for either field when data isn't available —
    we raise so the service records the error and leaves history alone.
    """
    symbol = config.params.get("symbol")
    if not symbol:
        raise FetchError(f"stooq: missing 'symbol' param in tracker {config.id}")
    url = f"https://stooq.com/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv"
    try:
        resp = client.get(url, timeout=FETCH_TIMEOUT)
    except httpx.HTTPError as e:
        raise FetchError(f"stooq: http error: {e}") from e
    if resp.status_code != 200:
        raise FetchError(f"stooq: status {resp.status_code}")
    rows = list(csv.reader(StringIO(resp.text)))
    if len(rows) < 2:
        raise FetchError(f"stooq: expected header+row, got {len(rows)} rows")
    row = rows[1]
    if len(row) < 7:
        raise FetchError(f"stooq: short row: {row!r}")
    date_str, time_str, close_str = row[1], row[2], row[6]
    if "N/D" in (date_str, time_str, close_str):
        raise FetchError(f"stooq: no data available for {symbol}")
    try:
        ts = datetime.fromisoformat(f"{date_str}T{time_str}").replace(tzinfo=timezone.utc)
        value = float(close_str)
    except ValueError as e:
        raise FetchError(f"stooq: parse failed for {symbol}: {e}") from e
    return TrackerReading(timestamp=ts, value=value)


# --- BLS (US food CPI) -----------------------------------------------------

def fetch_bls(config: TrackerConfig, client: httpx.Client) -> TrackerReading:
    """Latest monthly observation from the BLS public API (no key required
    for light traffic; ≤25 requests/day is free).

    Endpoint: POST https://api.bls.gov/publicAPI/v2/timeseries/data/
    Body:     {"seriesid": ["CUUR0000SAF1"]}
    Params:   {"series_id": "CUUR0000SAF1"}

    BLS returns observations newest-first. We stamp the timestamp at
    noon UTC on the LAST day of the reported month. Rationale:
    first-of-month stamping made the freshest-published CPI data read
    as >30 days old the moment it arrived (BLS releases month N data
    in mid-month N+1). End-of-month stamping ties the reading to the
    period it describes, so the default stale_after_hours threshold
    only trips when a release is actually overdue.

    calendar.monthrange handles leap years; no manual February logic needed.
    """
    series_id = config.params.get("series_id")
    if not series_id:
        raise FetchError(f"bls: missing 'series_id' param in tracker {config.id}")
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    body = {"seriesid": [series_id]}
    try:
        resp = client.post(url, json=body, timeout=FETCH_TIMEOUT)
    except httpx.HTTPError as e:
        raise FetchError(f"bls: http error: {e}") from e
    if resp.status_code != 200:
        raise FetchError(f"bls: status {resp.status_code}")
    try:
        data = resp.json()
    except ValueError as e:
        raise FetchError(f"bls: invalid JSON: {e}") from e
    if data.get("status") != "REQUEST_SUCCEEDED":
        msgs = data.get("message") or [str(data)[:200]]
        raise FetchError(f"bls: {msgs[0]}")
    results = (data.get("Results") or {}).get("series") or []
    if not results:
        raise FetchError(f"bls: no series in response for {series_id}")
    observations = results[0].get("data") or []
    if not observations:
        raise FetchError(f"bls: empty data for {series_id}")
    obs = observations[0]
    try:
        year = int(obs["year"])
        period = obs["period"]
    except (KeyError, ValueError) as e:
        raise FetchError(f"bls: bad observation shape {obs!r}: {e}") from e
    if not period.startswith("M") or period == "M13":
        # M13 is BLS's annual average — skip it for monthly tracking.
        raise FetchError(f"bls: unexpected period {period} in {series_id}")
    month = int(period[1:])
    raw_value = obs.get("value")
    if raw_value in (None, "", "-", "."):
        raise FetchError(f"bls: missing value in observation: {obs!r}")
    try:
        value = float(raw_value)
    except ValueError as e:
        raise FetchError(f"bls: non-numeric value {raw_value!r}: {e}") from e
    _, last_day = calendar.monthrange(year, month)
    ts = datetime(year, month, last_day, 12, 0, 0, tzinfo=timezone.utc)
    return TrackerReading(timestamp=ts, value=value)


# --- history fetchers -----------------------------------------------------
#
# First-time backfill. Called by TrackerService when a record has <=1
# stored readings; pulls enough history to immediately satisfy the 1d/1w/1m
# deltas and the Phase 5.2 label windows. Subsequent calls use the normal
# single-reading fetchers above.

def fetch_frankfurter_history(
    config: TrackerConfig, client: httpx.Client,
    lookback_days: int = _BACKFILL_DAILY_DAYS,
) -> list[TrackerReading]:
    """Historical daily rates from Frankfurter's date-range endpoint.

    Endpoint: /v1/YYYY-MM-DD..latest?base=USD&symbols=CNY
    Returns a dict keyed by date; we stamp each reading at 15:00 UTC
    (matches the single-reading fetcher's convention).
    """
    from_ccy = config.params.get("from", "USD")
    to_ccy = config.params.get("to")
    if not to_ccy:
        raise FetchError(f"frankfurter: missing 'to' param in tracker {config.id}")
    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=lookback_days)).isoformat()
    # Use explicit end-date rather than the `..latest` alias: current
    # Frankfurter deployment silently returns an empty `rates` map for
    # the ..latest form, so we pin to today's date to get actual rows.
    end = today.isoformat()
    url = f"https://api.frankfurter.dev/v1/{start}..{end}?base={from_ccy}&symbols={to_ccy}"
    try:
        resp = client.get(url, timeout=FETCH_TIMEOUT)
    except httpx.HTTPError as e:
        raise FetchError(f"frankfurter history: http error: {e}") from e
    if resp.status_code != 200:
        raise FetchError(f"frankfurter history: status {resp.status_code}: {resp.text[:200]}")
    try:
        data = resp.json()
    except ValueError as e:
        raise FetchError(f"frankfurter history: invalid JSON: {e}") from e
    rates = data.get("rates") or {}
    out: list[TrackerReading] = []
    for date_str, day_rates in rates.items():
        rate = day_rates.get(to_ccy) if isinstance(day_rates, dict) else None
        if rate is None:
            continue
        try:
            ts = datetime.fromisoformat(date_str).replace(
                hour=15, minute=0, second=0, tzinfo=timezone.utc,
            )
        except ValueError:
            continue
        out.append(TrackerReading(timestamp=ts, value=float(rate)))
    out.sort(key=lambda r: r.timestamp)
    return out


def fetch_stooq_history(
    config: TrackerConfig, client: httpx.Client,
    lookback_days: int = _BACKFILL_DAILY_DAYS,
) -> list[TrackerReading]:
    """Historical daily closes from Stooq's CSV endpoint.

    Endpoint: https://stooq.com/q/d/l/?s=<symbol>&i=d&d1=YYYYMMDD&d2=YYYYMMDD
    CSV columns: Date,Open,High,Low,Close,Volume
    We stamp each reading at noon UTC (precise time isn't available in
    the daily historical feed; noon sits between session opens/closes
    across major time zones and won't drift the sort order).
    """
    symbol = config.params.get("symbol")
    if not symbol:
        raise FetchError(f"stooq history: missing 'symbol' param in tracker {config.id}")
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=lookback_days)
    d1 = start.strftime("%Y%m%d")
    d2 = today.strftime("%Y%m%d")
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d&d1={d1}&d2={d2}"
    try:
        resp = client.get(url, timeout=FETCH_TIMEOUT)
    except httpx.HTTPError as e:
        raise FetchError(f"stooq history: http error: {e}") from e
    if resp.status_code != 200:
        raise FetchError(f"stooq history: status {resp.status_code}")
    # Stooq now gates historical ranges behind an apikey. The non-key
    # response is a plain-text instructional page starting with
    # "Get your apikey:". Detect that and return no history so the
    # service falls back to the normal single-reading fetcher (which
    # still works on the non-historical endpoint). History will simply
    # accumulate naturally over time instead of backfilling.
    body = resp.text
    if body.lstrip().lower().startswith("get your apikey"):
        log.info(
            "stooq history requires apikey now; returning [] so the "
            "service falls back to single-reading fetch. Register at "
            "stooq.com to unlock backfill."
        )
        return []
    rows = list(csv.reader(StringIO(body)))
    if len(rows) < 2:
        raise FetchError(f"stooq history: expected header+rows, got {len(rows)}")
    header = [h.strip().lower() for h in rows[0]]
    # Expect at minimum Date and Close. Column order has been stable but
    # we look up by name to stay forward-compatible if Stooq reorders.
    try:
        date_idx = header.index("date")
        close_idx = header.index("close")
    except ValueError as e:
        raise FetchError(f"stooq history: unexpected header {header}: {e}") from e
    out: list[TrackerReading] = []
    for row in rows[1:]:
        if len(row) <= max(date_idx, close_idx):
            continue
        date_str = row[date_idx].strip()
        close_str = row[close_idx].strip()
        if not date_str or not close_str or "N/D" in (date_str, close_str):
            continue
        try:
            ts = datetime.fromisoformat(date_str).replace(
                hour=12, minute=0, second=0, tzinfo=timezone.utc,
            )
            value = float(close_str)
        except ValueError:
            continue
        out.append(TrackerReading(timestamp=ts, value=value))
    out.sort(key=lambda r: r.timestamp)
    return out


def fetch_bls_history(
    config: TrackerConfig, client: httpx.Client,
    lookback_days: int | None = None,  # unused; BLS returns ~3y by default
) -> list[TrackerReading]:
    """All monthly observations BLS returns for the series (usually ~3y).

    Same endpoint + request body as the single-reading fetcher; difference
    is we iterate every observation in the response instead of taking
    only the newest. End-of-month stamping (matches fetch_bls).
    """
    series_id = config.params.get("series_id")
    if not series_id:
        raise FetchError(f"bls history: missing 'series_id' param in tracker {config.id}")
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    body = {"seriesid": [series_id]}
    try:
        resp = client.post(url, json=body, timeout=FETCH_TIMEOUT)
    except httpx.HTTPError as e:
        raise FetchError(f"bls history: http error: {e}") from e
    if resp.status_code != 200:
        raise FetchError(f"bls history: status {resp.status_code}")
    try:
        data = resp.json()
    except ValueError as e:
        raise FetchError(f"bls history: invalid JSON: {e}") from e
    if data.get("status") != "REQUEST_SUCCEEDED":
        msgs = data.get("message") or [str(data)[:200]]
        raise FetchError(f"bls history: {msgs[0]}")
    results = (data.get("Results") or {}).get("series") or []
    if not results:
        raise FetchError(f"bls history: no series in response for {series_id}")
    observations = results[0].get("data") or []
    out: list[TrackerReading] = []
    for obs in observations:
        try:
            year = int(obs["year"])
            period = obs["period"]
        except (KeyError, ValueError):
            continue
        if not period.startswith("M") or period == "M13":
            continue
        try:
            month = int(period[1:])
        except ValueError:
            continue
        raw_value = obs.get("value")
        if raw_value in (None, "", "-", "."):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        _, last_day = calendar.monthrange(year, month)
        ts = datetime(year, month, last_day, 12, 0, 0, tzinfo=timezone.utc)
        out.append(TrackerReading(timestamp=ts, value=value))
    out.sort(key=lambda r: r.timestamp)
    return out


# --- EIA (US Energy Information Administration) -------------------------
#
# Weekly retail gasoline prices by region/state. Requires a free API key
# registered at https://www.eia.gov/opendata/register.php, passed via
# KARIN_EIA_API_KEY env var. The single-reading fetcher returns the most
# recent weekly observation; the history fetcher pulls the configured
# retention window of weekly data.
#
# Trackers config params expected:
#   {"duoarea": "<code>", "product": "EPMR"}
#   duoarea: "NUS" (US avg), "R10"-"R5X" (PADD regions), "S<ST>" (state
#     where available). For PA/NJ/NY/MD use "R1Y" (PADD 1B, Central
#     Atlantic).
#   product: "EPMR" (regular), "EPMP" (premium), "EPMM" (midgrade),
#     "EPD2D" (diesel).

def _eia_api_key() -> str:
    """Read the EIA API key from env. Raises FetchError when unset so
    the tracker subsystem marks the record with a clear message."""
    import os
    key = (os.environ.get("KARIN_EIA_API_KEY") or "").strip()
    if not key:
        raise FetchError(
            "eia: KARIN_EIA_API_KEY not set — register a free key at "
            "eia.gov/opendata/register.php and add it to .env"
        )
    return key


def _eia_fetch_rows(config: TrackerConfig, client: httpx.Client, length: int) -> list[dict]:
    """Shared EIA v2 weekly retail gas fetch. Returns data rows (dicts)
    sorted newest-first. Common implementation used by both the
    single-reading and history fetchers."""
    duoarea = config.params.get("duoarea")
    product = config.params.get("product", "EPMR")
    if not duoarea:
        raise FetchError(f"eia: missing 'duoarea' param in tracker {config.id}")
    params = [
        ("api_key", _eia_api_key()),
        ("frequency", "weekly"),
        ("data[]", "value"),
        ("facets[duoarea][]", str(duoarea)),
        ("facets[product][]", str(product)),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "desc"),
        ("length", str(length)),
    ]
    try:
        resp = client.get(
            "https://api.eia.gov/v2/petroleum/pri/gnd/data/",
            params=params, timeout=FETCH_TIMEOUT,
        )
    except httpx.HTTPError as e:
        raise FetchError(f"eia: http error: {e}") from e
    if resp.status_code != 200:
        raise FetchError(f"eia: status {resp.status_code}: {resp.text[:200]}")
    try:
        data = resp.json()
    except ValueError as e:
        raise FetchError(f"eia: invalid JSON: {e}") from e
    rows = ((data.get("response") or {}).get("data") or [])
    if not rows:
        raise FetchError(
            f"eia: no rows for duoarea={duoarea} product={product}"
        )
    return rows


def _eia_row_to_reading(row: dict) -> TrackerReading:
    """Parse one EIA data row into a TrackerReading. Stamps timestamp
    at 12:00 UTC on the period date — EIA publishes weekly averages,
    not intraday ticks, so an approximate midday is sufficient for
    delta comparisons."""
    period = row.get("period")
    value = row.get("value")
    if period is None or value is None:
        raise FetchError(f"eia: row missing period/value: {row!r}")
    try:
        base_dt = datetime.fromisoformat(str(period))
    except ValueError as e:
        raise FetchError(f"eia: bad period {period!r}: {e}") from e
    ts = base_dt.replace(hour=12, minute=0, second=0, tzinfo=timezone.utc)
    try:
        return TrackerReading(timestamp=ts, value=float(value))
    except (TypeError, ValueError) as e:
        raise FetchError(f"eia: bad value {value!r}: {e}") from e


def fetch_eia(config: TrackerConfig, client: httpx.Client) -> TrackerReading:
    """Most recent weekly retail-gas price from EIA."""
    rows = _eia_fetch_rows(config, client, length=1)
    return _eia_row_to_reading(rows[0])


def fetch_eia_history(
    config: TrackerConfig, client: httpx.Client,
) -> list[TrackerReading]:
    """Fill `history_days` worth of weekly EIA data (newest-first from
    the API, returned oldest-first for the store)."""
    # Weekly cadence; ceil(days/7) plus buffer for initial backfill.
    weeks = max(4, (config.history_days // 7) + 2)
    rows = _eia_fetch_rows(config, client, length=weeks)
    readings = [_eia_row_to_reading(r) for r in rows]
    readings.sort(key=lambda r: r.timestamp)
    return readings


# --- dispatch --------------------------------------------------------------

# Adding a new source: define a fetch_<x>() above, add the entry here,
# reference it from trackers.json. No changes needed elsewhere.
FETCHERS: dict[str, Callable[..., TrackerReading]] = {
    "frankfurter": fetch_frankfurter,
    "stooq": fetch_stooq,
    "bls": fetch_bls,
    "eia": fetch_eia,
}

# History fetchers keyed by the same `source` string as single-reading
# fetchers. A source that doesn't yet support history backfill simply
# omits itself here; the service falls back to appending one reading at
# a time (original behavior).
HISTORY_FETCHERS: dict[str, Callable[..., list[TrackerReading]]] = {
    "frankfurter": fetch_frankfurter_history,
    "stooq":       fetch_stooq_history,
    "bls":         fetch_bls_history,
    "eia":         fetch_eia_history,
}


def fetch(config: TrackerConfig, client: httpx.Client) -> TrackerReading:
    """Dispatch to the per-source fetcher. Raises FetchError on failure
    or on an unknown source."""
    fn = FETCHERS.get(config.source)
    if fn is None:
        raise FetchError(
            f"unknown source {config.source!r} for tracker {config.id}"
        )
    return fn(config, client)


def fetch_history(
    config: TrackerConfig, client: httpx.Client,
) -> list[TrackerReading]:
    """Dispatch to the per-source history fetcher. Returns [] if the
    source doesn't support history backfill — callers treat that as a
    non-fatal "just use the single-reading fetcher" signal."""
    fn = HISTORY_FETCHERS.get(config.source)
    if fn is None:
        return []
    return fn(config, client)
