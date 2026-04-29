"""Holiday lookup for the UI banner.

Pulls official public holidays from date.nager.at (free, no API key),
caches per-year JSON to ``data/holidays/{year}-{cc}.json`` so we hit
the network at most once per year per country. Supplements nager's
official list with a small hand-curated set of cultural days that
aren't federal holidays but are widely observed (Valentine's,
Halloween, Qixi, Lantern Festival, etc.).

API exposed via :func:`today_holiday` — returns the holiday record
the frontend banner needs (name, country, emoji, greeting, date) or
``None``. Fail-soft: a missing cache + unreachable nager.date just
means the supplemental list is consulted alone.
"""
from __future__ import annotations

import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from bridge._http import make_client
from bridge.utils import REPO_ROOT, atomic_write_text

log = logging.getLogger("bridge.holidays")

_CACHE_DIR: Path = REPO_ROOT / "data" / "holidays"
_AUDIO_DIR: Path = _CACHE_DIR / "audio"
_NAGER_BASE = "https://date.nager.at/api/v3/PublicHolidays"
_COUNTRIES = ("CN", "US")    # preference order — CN matches win over US on same date
_FETCH_TIMEOUT_S = 8.0

# Emoji + greeting mapping keyed by common holiday names nager returns.
# Unknown names get a generic festive fallback so the banner still
# renders — we'd rather show "🎉 Ascension Day" than swallow it.
_NAME_META: dict[str, tuple[str, str]] = {
    "New Year's Day":                ("🎉", "Happy New Year!"),
    "Martin Luther King, Jr. Day":   ("🕊️", "Honoring Dr. King today."),
    "Memorial Day":                  ("🇺🇸", "Remembering those who served."),
    "Juneteenth National Independence Day": ("🇺🇸", "Happy Juneteenth!"),
    "Independence Day":              ("🇺🇸", "Happy 4th of July!"),
    "Labor Day":                     ("🛠️", "Happy Labor Day!"),
    "Columbus Day":                  ("⛵",  "Happy Columbus Day!"),
    "Veterans Day":                  ("🇺🇸", "Thank you to all who served."),
    "Thanksgiving Day":              ("🦃",  "Happy Thanksgiving!"),
    "Washington's Birthday":         ("🇺🇸", "Happy Presidents' Day!"),
    "Presidents' Day":               ("🇺🇸", "Happy Presidents' Day!"),
    "Christmas Day":                 ("🎄",  "Merry Christmas!"),
    "Chinese New Year":              ("🧧",  "Happy Chinese New Year! 新年快乐!"),
    "Spring Festival":               ("🧧",  "Happy Chinese New Year! 新年快乐!"),
    "Tomb-Sweeping Day":             ("🌿",  "清明节安康。"),
    "Qingming Festival":             ("🌿",  "清明节安康。"),
    "Labour Day":                    ("🛠️", "Happy Labour Day!"),
    "Dragon Boat Festival":          ("🚣",  "端午节快乐!"),
    "Mid-Autumn Festival":           ("🥮",  "中秋节快乐!"),
    "National Day":                  ("🇨🇳", "国庆节快乐!"),
}

# Cultural / observed days that aren't on nager's public-holiday list.
# Two key shapes:
#   "MM-DD"      — fixed every year
#   "YYYY-MM-DD" — specific year (used for lunar cultural days whose
#                  Gregorian date shifts: Qixi, Lantern, etc.)
# YYYY-MM-DD keys take precedence over MM-DD keys for the same date.
_SUPPLEMENTAL: dict[str, dict[str, str]] = {
    # --- fixed-date cultural ---
    "02-14": {"name": "Valentine's Day",   "country": "US", "emoji": "💝",  "greeting": "Happy Valentine's Day!"},
    "03-17": {"name": "St. Patrick's Day", "country": "US", "emoji": "🍀",  "greeting": "Happy St. Patrick's Day!"},
    "10-31": {"name": "Halloween",         "country": "US", "emoji": "🎃",  "greeting": "Happy Halloween!"},
    "12-24": {"name": "Christmas Eve",     "country": "US", "emoji": "🎄",  "greeting": "Merry Christmas Eve!"},
    "12-31": {"name": "New Year's Eve",    "country": "US", "emoji": "🥂",  "greeting": "Happy New Year's Eve!"},

    # --- Chinese cultural lunar (per-year Gregorian dates) ---
    "2026-03-03": {"name": "Lantern Festival", "country": "CN", "emoji": "🏮", "greeting": "元宵节快乐!"},
    "2026-08-19": {"name": "Qixi",             "country": "CN", "emoji": "💞", "greeting": "七夕快乐!"},
    "2027-02-20": {"name": "Lantern Festival", "country": "CN", "emoji": "🏮", "greeting": "元宵节快乐!"},
    "2027-08-08": {"name": "Qixi",             "country": "CN", "emoji": "💞", "greeting": "七夕快乐!"},
    "2028-02-09": {"name": "Lantern Festival", "country": "CN", "emoji": "🏮", "greeting": "元宵节快乐!"},
    "2028-08-26": {"name": "Qixi",             "country": "CN", "emoji": "💞", "greeting": "七夕快乐!"},

    # --- Mother's / Father's Day (US variable, per-year) ---
    "2026-05-10": {"name": "Mother's Day", "country": "US", "emoji": "💐", "greeting": "Happy Mother's Day!"},
    "2026-06-21": {"name": "Father's Day", "country": "US", "emoji": "👔", "greeting": "Happy Father's Day!"},
    "2027-05-09": {"name": "Mother's Day", "country": "US", "emoji": "💐", "greeting": "Happy Mother's Day!"},
    "2027-06-20": {"name": "Father's Day", "country": "US", "emoji": "👔", "greeting": "Happy Father's Day!"},
    "2028-05-14": {"name": "Mother's Day", "country": "US", "emoji": "💐", "greeting": "Happy Mother's Day!"},
    "2028-06-18": {"name": "Father's Day", "country": "US", "emoji": "👔", "greeting": "Happy Father's Day!"},
}


def _cache_path(year: int, country: str) -> Path:
    return _CACHE_DIR / f"{year}-{country}.json"


def _fetch_from_nager(year: int, country: str) -> list[dict] | None:
    """Return the raw nager.date response or None on any failure.
    Response shape per entry: ``{"date": "YYYY-MM-DD", "name": ..., "localName": ..., ...}``.
    """
    url = f"{_NAGER_BASE}/{year}/{country}"
    try:
        with make_client(timeout=_FETCH_TIMEOUT_S) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
    except Exception as e:
        log.warning("nager %s/%s failed: %s", year, country, e)
    return None


def _load_holidays(year: int, country: str) -> list[dict]:
    """Load (or fetch-then-cache) a year's official holidays for ``country``."""
    cache = _cache_path(year, country)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("cached holidays %s unreadable: %s", cache, e)
    data = _fetch_from_nager(year, country)
    if data is None:
        return []
    try:
        atomic_write_text(
            cache,
            json.dumps(data, ensure_ascii=False, indent=2),
        )
    except Exception as e:
        log.warning("failed to write holiday cache %s: %s", cache, e)
    return data


def _make_record(name: str, country: str, date_str: str) -> dict[str, Any]:
    emoji, greeting = _NAME_META.get(name, ("🎉", f"Happy {name}!"))
    return {
        "name": name,
        "country": country,
        "emoji": emoji,
        "greeting": greeting,
        "date": date_str,
    }


def today_holiday(today: datetime.date | None = None) -> dict[str, Any] | None:
    """Return the best-match holiday for ``today`` or None.

    Lookup order (first match wins):
      1. Dated supplemental entry (``YYYY-MM-DD``) — catches cultural
         CN lunar dates and year-specific US holidays.
      2. nager.date official public-holiday list for each country
         in ``_COUNTRIES`` order.
      3. Fixed-date supplemental entry (``MM-DD``) — cultural days
         observed every year.
    """
    today = today or datetime.date.today()
    yyyymmdd = today.isoformat()
    mmdd = f"{today.month:02d}-{today.day:02d}"

    # 1. Dated supplemental (most specific)
    sup = _SUPPLEMENTAL.get(yyyymmdd)
    if sup is not None:
        rec = {**sup, "date": yyyymmdd}
        rec.setdefault("country", "US")
        return rec

    # 2. Official nager sources
    for cc in _COUNTRIES:
        for h in _load_holidays(today.year, cc):
            if h.get("date") != yyyymmdd:
                continue
            name = (h.get("name") or h.get("localName") or "").strip() or "Holiday"
            return _make_record(name, cc, yyyymmdd)

    # 3. Fixed-date supplemental (fallback every-year cultural days)
    sup = _SUPPLEMENTAL.get(mmdd)
    if sup is not None:
        rec = {**sup, "date": yyyymmdd}
        rec.setdefault("country", "US")
        return rec

    return None


# ---- Voiced-greeting audio cache -----------------------------------------
#
# When today is a holiday, we let the frontend play a one-time Karin-voiced
# greeting (e.g. "Happy Chinese New Year! 新年快乐!") via the cloned voice.
# Audio is synthesized lazily on the first request and cached to disk so
# repeat visits don't reseynthesize. Files older than "today" get purged
# automatically so the cache never holds yesterday's holiday.


def _audio_slug(name: str) -> str:
    """Filesystem-friendly slug for a holiday name. Keeps ASCII letters
    and digits, collapses everything else to a single dash."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "holiday"


def _audio_cache_path(holiday: dict) -> Path:
    return _AUDIO_DIR / f"{holiday['date']}-{_audio_slug(holiday['name'])}.wav"


def _purge_stale_audio(today_prefix: str) -> int:
    """Delete cached greeting audio whose filename doesn't start with
    ``today_prefix`` (a ``YYYY-MM-DD`` string). Returns the count deleted.
    Called on every audio-fetch so cleanup rides along the normal flow
    — no separate cron job needed.
    """
    if not _AUDIO_DIR.exists():
        return 0
    removed = 0
    for p in _AUDIO_DIR.iterdir():
        if not p.is_file():
            continue
        if not p.name.startswith(today_prefix):
            try:
                p.unlink()
                removed += 1
                log.info("purged stale holiday audio: %s", p.name)
            except Exception as e:
                log.warning("couldn't purge %s: %s", p, e)
    return removed


def get_or_synth_greeting_audio(
    tts_client,
    holiday: dict | None = None,
) -> Path | None:
    """Return a path to today's greeting WAV, synthesizing if needed.

    ``tts_client`` must expose ``synthesize(text) -> (pcm, sample_rate)``
    — in practice ``bridge.tts.SoVITSTTS``. Returns None when:
      - No holiday today.
      - sovits is unreachable or synthesis fails (caller treats as
        "no audio for you" and skips playback — banner still shows).

    Always runs the stale-audio purge first so deleted-holiday files
    get cleaned up without a separate job.
    """
    if holiday is None:
        holiday = today_holiday()

    today_prefix = holiday["date"] if holiday else "__no_holiday__"
    _purge_stale_audio(today_prefix)
    if holiday is None:
        return None

    try:
        _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.warning("couldn't create audio cache dir: %s", e)
        return None

    cache = _audio_cache_path(holiday)
    if cache.exists() and cache.stat().st_size > 0:
        return cache

    try:
        pcm, sr = tts_client.synthesize(holiday["greeting"])
    except Exception as e:
        log.warning("holiday greeting TTS failed (%s): %s", holiday["name"], e)
        return None

    try:
        import soundfile as sf
        sf.write(str(cache), pcm, sr, format="WAV", subtype="PCM_16")
    except Exception as e:
        log.warning("couldn't write greeting WAV %s: %s", cache, e)
        return None

    log.info("synthesized holiday greeting for %s → %s", holiday["name"], cache.name)
    return cache
