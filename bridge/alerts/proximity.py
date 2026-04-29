"""Rule-based threat scoring for the alerts pipeline.

Pure functions — no I/O, no LLM, no state. Plugged inline into
AlertService's signal collection so every signal gets a
``threat_score`` in its payload without a separate pass over the
data.

Score meaning (used by rules to decide whether to fire):

  0  No threat — skip entirely
  1  Background awareness — not panel-worthy
  2  Watch — borderline, surface dimmed (LLM re-check in G.b)
  3  Advisory — surface with standard styling
  4  Critical — surface prominently, sort to top

Scoring axes (multiplied, capped at 4):

  * Proximity — does the event affect the user's location?
  * Category — does it carry life-safety weight? (weather > markets)
  * Certainty — confirmed > provisional > speculative

Numbers are hand-tuned starting points. The user-visible threshold
(what's shown vs hidden) lives in tuning.yaml so operators can
tighten or loosen without editing code.
"""
from __future__ import annotations

import math

from bridge.alerts.models import SignalKind
from bridge.alerts.user_context import UserContext


# --- constants -----------------------------------------------------------

# Scoring caps.
_MAX_SCORE = 4

# Category weights. Values are multipliers applied to the proximity
# base. Higher weight => the same location hit produces a bigger final
# threat score.
#
# Reasoning:
#   - event watchlists (disasters, conflicts, elections) are where
#     life-impact news lives; weight > 1.
#   - regions are pure geography; weight = 1 baseline.
#   - topics (tech, AI, economy) are interesting but rarely a threat;
#     weight < 1 so they need strong location + confirmation signals
#     to cross the firing threshold.
_WATCHLIST_TYPE_WEIGHT: dict[str, float] = {
    "event":  1.5,
    "region": 1.0,
    "topic":  0.5,
}

# Tracker categories by life-relevance. Gas and food move money that
# every person spends weekly; FX / metal / crypto only hit the user
# through indirect channels. So price shocks on the former register
# as threats, shocks on the latter as financial awareness only.
_LIFE_RELEVANT_TRACKER_CATEGORIES = frozenset({"energy", "food_index"})

# NWS severity levels map to threat scores. "Severe/Extreme" are
# NWS's own "imminent action required" tier. Anything below
# "Moderate" is informational.
_NWS_SEVERITY_TO_SCORE: dict[str, int] = {
    "extreme":  4,
    "severe":   3,
    "moderate": 2,
    "minor":    1,
    "unknown":  1,
}

# Radius (km) around user's lat/lon below which a geolocated news
# event counts as "physically near". Rough rule-of-thumb: commuting /
# regional news radius. Anything inside = proximity 3 bonus on top of
# the text-match score.
_NEAR_RADIUS_KM = 100.0


# --- pure helpers --------------------------------------------------------

def _contains_ci(needle: str, haystack: str) -> bool:
    """Case-insensitive substring match guarding against empty inputs."""
    if not needle or not haystack:
        return False
    return needle.lower() in haystack.lower()


def location_match_score(text: str, ctx: UserContext) -> int:
    """How closely does ``text`` mention the user's location?

    Returns 0-3:
      3  city match (or region + city in same text)
      2  region / state match
      1  country match
      0  no hit

    City takes priority because it's most specific; region falls
    through when city isn't mentioned. Counter-intuitive edge: a
    global story that happens to mention the user's country
    everywhere still only scores 1 — location match alone doesn't
    make something a threat (category + certainty multipliers
    temper it).
    """
    if not text:
        return 0
    if ctx.city and _contains_ci(ctx.city, text):
        return 3
    if ctx.region and _contains_ci(ctx.region, text):
        return 2
    if ctx.country and _contains_ci(ctx.country, text):
        return 1
    return 0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in km.

    Used by the NWS-alert scorer when the alert carries geographic
    coordinates for its affected area — we check whether the user's
    configured lat/lon sits inside the alert's radius. Accurate
    enough for the "within 100 km" bucket decision we make; not for
    sub-km precision.
    """
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _certainty_multiplier(state: str) -> float:
    """Confidence from news cluster state. Confirmed news is a
    stronger safety signal than a developing story."""
    s = (state or "").lower()
    if s == "confirmed":
        return 1.2
    if s == "provisionally_confirmed":
        return 1.0
    if s == "developing":
        return 0.7
    return 0.5


# --- dispatcher ----------------------------------------------------------

def compute_threat_score(payload: dict, kind, ctx: UserContext) -> int:
    """Compute 0-4 threat score for one signal.

    Dispatches on SignalKind. Pure function — callers own the signal
    construction and can annotate `payload["threat_score"]` with this
    return value during collection.

    Design: returning 0 is the "filter me out" sentinel; rules that
    care about life-safety pre-check ``threat_score > 0`` before firing.
    Rules that DON'T care (existing behaviour where every signal
    becomes an alert) simply ignore the field.
    """
    if kind == SignalKind.NWS_WEATHER_ALERT:
        return _score_nws(payload, ctx)
    if kind == SignalKind.TRAVEL_ADVISORY_CHANGED:
        return _score_travel(payload, ctx)
    if kind == SignalKind.NEWS_WATCHLIST_MATCH:
        return _score_news(payload, ctx)
    if kind == SignalKind.TRACKER_SHOCK:
        return _score_tracker(payload, ctx)
    return 0


# --- per-kind scorers ---------------------------------------------------

def _score_nws(payload: dict, ctx: UserContext) -> int:
    """NWS alerts are already life-safety graded — trust the severity
    field, then gate on whether the user is actually inside the
    affected area. If the alert publishes lat/lon and we have the
    user's, prefer haversine distance; otherwise fall back to
    region-name containment in the affected-zones string."""
    severity = str(payload.get("severity", "")).lower()
    base = _NWS_SEVERITY_TO_SCORE.get(severity, 1)

    # Geofence — drop if the user isn't in-range. NWS events often
    # span multiple counties or states; conservatively keep the
    # alert if we can't prove the user is out of range.
    affected = str(payload.get("affected_zones", "") or payload.get("areaDesc", ""))
    if ctx.region and affected and not _contains_ci(ctx.region, affected):
        # We know the user's region AND the alert's area — and they
        # don't overlap by name. Demote by one tier.
        base = max(0, base - 1)
    # If lat/lon available on both sides, use that instead of names.
    alat = payload.get("lat")
    alon = payload.get("lon")
    if (
        isinstance(alat, (int, float)) and isinstance(alon, (int, float))
        and ctx.latitude is not None and ctx.longitude is not None
    ):
        if haversine_km(ctx.latitude, ctx.longitude, float(alat), float(alon)) > _NEAR_RADIUS_KM:
            base = max(0, base - 2)
    return min(_MAX_SCORE, base)


def _score_travel(payload: dict, ctx: UserContext) -> int:
    """Travel advisories only fire as threats if the user IS in (or
    named as being near) the destination country. State Dept levels:
    1 exercise precautions, 2 increased caution, 3 reconsider
    travel, 4 do not travel."""
    new_level = int(payload.get("new_level", 0) or 0)
    country = str(payload.get("country", "") or "")
    if not country:
        return max(0, new_level - 1)
    if ctx.country and ctx.country.lower() == country.lower():
        # User is IN this country — treat at face value.
        return min(_MAX_SCORE, new_level)
    # Elsewhere advisory — demote. Still visible at 4 (user might be
    # about to travel there) but no longer at 1-2.
    return max(0, new_level - 2)


def _score_news(payload: dict, ctx: UserContext) -> int:
    """News cluster threat scoring. Proximity (from headline + label)
    multiplied by watchlist-type weight and cluster certainty. This
    is where most of the filtering noise reduction comes from — the
    previous rule engine fired R4/R5 on every confirmed match; here
    we additionally require the story to be *about* somewhere the
    user is physically near."""
    headline = str(payload.get("headline", "") or "")
    # Combine headline + watchlist label so "ASML earnings" alone
    # doesn't gain a location bonus, but "Pennsylvania" as the label
    # does — covers cases where the geography lives in the label
    # rather than the headline text.
    label = str(payload.get("watchlist_label", "") or "")
    loc_score = max(
        location_match_score(headline, ctx),
        location_match_score(label, ctx),
    )
    if loc_score == 0:
        # No proximity => it's news, not a threat. Even a confirmed
        # CONFLICT story in a distant country doesn't warrant an
        # alert — it belongs on the News panel, not here.
        return 0

    wl_type = str(payload.get("watchlist_type", "") or "").lower()
    weight = _WATCHLIST_TYPE_WEIGHT.get(wl_type, 1.0)

    state = str(payload.get("cluster_state", "") or "")
    certainty = _certainty_multiplier(state)

    score = loc_score * weight * certainty
    return max(0, min(_MAX_SCORE, int(round(score))))


def _score_tracker(payload: dict, ctx: UserContext) -> int:
    """Tracker shocks. Gas + food are tangible life-impact signals
    (every-day wallet pain). FX + metal + crypto are markets — they
    shock loudly but don't change whether you can fill your tank or
    buy groceries. Separate the two so the alerts panel stops
    flashing on every crypto move."""
    category = str(payload.get("category", "") or "").lower()
    magnitude = float(payload.get("change_pct_abs", 0.0) or 0.0)
    if category in _LIFE_RELEVANT_TRACKER_CATEGORIES:
        # Gas jumping 5% matters to everyone; 15% is painful.
        if magnitude >= 10:
            return 3
        if magnitude >= 5:
            return 2
        return 1
    # Financial category — keep as awareness, never critical unless
    # the ShockPlusGeopolitical cross-channel rule combines it with
    # real-world impact (that stays in R7 as today).
    return 1
