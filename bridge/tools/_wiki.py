"""Wikipedia tools + biographical/event date extraction."""
from __future__ import annotations

import datetime
import logging
import re

import httpx

log = logging.getLogger("bridge.tools")


# ---- On This Day in history -----------------------------------------------

def _today_in_history() -> str:
    """LLM tool: a random historical event from today's calendar date.

    Uses Wikipedia's "On This Day" REST endpoint, which returns ~50
    actual historical events that happened on this day in past years.
    Beats wiki_random for history because every entry is real history,
    not a random article about a basketball club or a bridge.
    """
    import random
    today = datetime.datetime.now().astimezone()
    headers = {
        "User-Agent": (
            "karin/0.1 (https://github.com/kaminglui/Karin; "
            "personal voice assistant; en) httpx"
        ),
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=8.0, headers=headers) as client:
            url = (
                f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/"
                f"events/{today.month:02d}/{today.day:02d}"
            )
            resp = client.get(url, follow_redirects=True)
            if resp.status_code != 200:
                return f"Error: on-this-day fetch failed [{resp.status_code}]"
            events = (resp.json().get("events") or [])
            if not events:
                return f"No historical events found for {today.strftime('%B %d')}."
            ev = random.choice(events)
            year = ev.get("year")
            text = (ev.get("text") or "").strip()
            return f"On this day in {year}: {text}"
    except httpx.HTTPError as e:
        return f"Error fetching on-this-day: {e}"
    except Exception as e:
        return f"Error: {e}"


# ---- Wikipedia random ----------------------------------------------------

def _wiki_random() -> str:
    """LLM tool: a random Wikipedia article summary.

    Used for vague requests like "tell me some history" or "tell me
    something interesting" so the model gives a concrete answer
    instead of bouncing the question back to the user.
    """
    headers = {
        "User-Agent": (
            "karin/0.1 (https://github.com/kaminglui/Karin; "
            "personal voice assistant; en) httpx"
        ),
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=8.0, headers=headers) as client:
            resp = client.get(
                "https://en.wikipedia.org/api/rest_v1/page/random/summary",
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return f"Error: random article fetch failed [{resp.status_code}]"
            s = resp.json()
            title = s.get("title") or "Unknown"
            extract = (s.get("extract") or "").strip()
            if not extract:
                return f"{title}: no summary available."
            sentences = extract.split(". ")
            short = ". ".join(sentences[:3])
            if not short.endswith("."):
                short += "."
            return f"{title}: {short}"
    except httpx.HTTPError as e:
        return f"Error fetching random article: {e}"
    except Exception as e:
        return f"Error: {e}"


# ---- Biographical date / age extraction ------------------------------------

# Patterns cover the most common Wikipedia biography lead formats:
#   "Name (born Month Day, Year)"
#   "Name (Month Day, Year – Month Day, Year)"
#   "Name (c. Year – Year)" (approximate dates)
# The regex uses a non-greedy scan of the first ~500 chars of the extract.

_BORN_RE = re.compile(
    r"\((?:born|b\.)\s+"
    r"(?:c\.\s*)?"
    r"(\w+ \d{1,2},? \d{4}|\d{4})"
    r"\)",
    re.IGNORECASE,
)

# Full lifespan: "(Month Day, Year – Month Day, Year)" or "(Year–Year)"
_LIFESPAN_RE = re.compile(
    r"\("
    r"(?:c\.\s*)?"
    r"(\w+ \d{1,2},? \d{4}|\d{4})"
    r"\s*[\u2013\u2014\-–—]+\s*"
    r"(?:c\.\s*)?"
    r"(\w+ \d{1,2},? \d{4}|\d{4})"
    r"\)",
    re.IGNORECASE,
)


def _parse_bio_date(s: str) -> "datetime.date | None":
    """Best-effort parse of a date fragment like 'August 4, 1961' or '1961'."""
    s = s.strip().lstrip("c. ")
    for fmt in ("%B %d, %Y", "%B %d %Y", "%Y"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _bio_age_suffix(extract: str) -> str:
    """If ``extract`` looks like a biography, return an age annotation.

    Returns a string like:
      " [Born: August 4, 1961 — age 64 as of April 2026]"
      " [Born: January 15, 1929 — Died: April 4, 1968 (age 39). Dead for ~58 years.]"
    or empty string if no dates found.
    """
    # Only scan the first 500 chars — dates are always in the lead.
    lead = extract[:500]
    today = datetime.date.today()

    # Try lifespan first (dead person).
    m = _LIFESPAN_RE.search(lead)
    if m:
        birth = _parse_bio_date(m.group(1))
        death = _parse_bio_date(m.group(2))
        if birth and death:
            age_at_death = (
                death.year - birth.year
                - ((death.month, death.day) < (birth.month, birth.day))
            )
            years_dead = (
                today.year - death.year
                - ((today.month, today.day) < (death.month, death.day))
            )
            return (
                f" [Born: {m.group(1)} — Died: {m.group(2)} "
                f"(age {age_at_death}). Dead for ~{years_dead} years.]"
            )

    # Try born-only (living person).
    m = _BORN_RE.search(lead)
    if m:
        birth = _parse_bio_date(m.group(1))
        if birth:
            age = (
                today.year - birth.year
                - ((today.month, today.day) < (birth.month, birth.day))
            )
            return (
                f" [Born: {m.group(1)} — age {age} as of "
                f"{today.strftime('%B %Y')}]"
            )

    return ""


# ---- Event-date "X years ago" extraction ----------------------------------

# Matches dates in common Wikipedia prose like:
#   "ended on September 2, 1945"
#   "was founded in 1998"
#   "occurred on March 11, 2011"
#   "signed on July 4, 1776"
#   "opened in June 2007"
#   "released on June 29, 2007"
# We only pick up the first match to keep the suffix short. Biographical
# dates (already handled by _bio_age_suffix) are parenthesized, so they
# won't match these prose-level patterns.

_EVENT_DATE_FULL_RE = re.compile(
    r"\b(?:on|in|since|from|after|before|during|until)\s+"
    r"(\w+ \d{1,2},? \d{4})",
    re.IGNORECASE,
)

_EVENT_DATE_MONTH_YEAR_RE = re.compile(
    r"\b(?:on|in|since|from|after|before|during|until)\s+"
    r"((?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4})",
    re.IGNORECASE,
)

_EVENT_DATE_YEAR_RE = re.compile(
    r"\b(?:in|since|from|after|founded|established|created|"
    r"built|opened|started|began|ended|signed|occurred)\s+"
    r"(?:in\s+)?(\d{4})\b",
    re.IGNORECASE,
)


def _event_date_suffix(extract: str) -> str:
    """If the extract mentions a notable event date, compute how long ago.

    Scans for the first date mentioned in prose (not parenthesized bio
    dates — those are covered by ``_bio_age_suffix``). Returns a bracket
    annotation like ``[September 2, 1945 — ~81 years ago]`` or empty
    string if nothing found.

    Precision:
      - Full date (Month Day, Year): accurate to the day.
      - Month + Year: assumes the 1st of that month.
      - Year only: assumes January 1st of that year.
    """
    lead = extract[:500]
    today = datetime.date.today()

    # Skip if this looks like a biography (bio suffix already handles it).
    if _BORN_RE.search(lead) or _LIFESPAN_RE.search(lead):
        return ""

    # Try full date first (most precise).
    m = _EVENT_DATE_FULL_RE.search(lead)
    if m:
        d = _parse_bio_date(m.group(1))
        if d and d < today:
            delta = _years_months_ago(d, today)
            return f" [{m.group(1)} — {delta}]"

    # Month + year.
    m = _EVENT_DATE_MONTH_YEAR_RE.search(lead)
    if m:
        d = _parse_bio_date(m.group(1) + " 1")
        if d is None:
            # Try "June 2007" -> "June 1, 2007"
            parts = m.group(1).split()
            if len(parts) == 2:
                d = _parse_bio_date(f"{parts[0]} 1, {parts[1]}")
        if d and d < today:
            delta = _years_months_ago(d, today)
            return f" [{m.group(1)} — {delta}]"

    # Year only (least precise).
    m = _EVENT_DATE_YEAR_RE.search(lead)
    if m:
        try:
            year = int(m.group(1))
        except ValueError:
            year = None
        if year and 1 <= year <= today.year:
            years_ago = today.year - year
            if years_ago == 0:
                return f" [{year} — this year]"
            return f" [{year} — ~{years_ago} year{'s' if years_ago != 1 else ''} ago]"

    return ""


def _years_months_ago(past: "datetime.date", today: "datetime.date") -> str:
    """Human-friendly 'X years, Y months ago' string, day-accurate."""
    total_months = (today.year - past.year) * 12 + (today.month - past.month)
    if today.day < past.day:
        total_months -= 1
    years = total_months // 12
    months = total_months % 12
    if years == 0 and months == 0:
        days = (today - past).days
        return f"~{days} days ago"
    parts: list[str] = []
    if years > 0:
        parts.append(f"~{years} year{'s' if years != 1 else ''}")
    if months > 0:
        parts.append(f"{months} month{'s' if months != 1 else ''}")
    return ", ".join(parts) + " ago"


# ---- Wikipedia search -----------------------------------------------------

def _wiki_search(query: str) -> str:
    """LLM tool: top-result Wikipedia summary in 1-3 sentences.

    Two-step lookup so we get the canonical article even when the model
    sends a sloppy query:
      1. opensearch -> resolve to the best matching page title
      2. summary endpoint -> short extract for that title
    """
    q = (query or "").strip()
    if not q:
        return "Error: empty query"
    if len(q) > 200:
        return "Error: query too long"
    # Wikipedia's API policy requires a descriptive User-Agent that
    # identifies the app + a contact. Generic UAs get 403'd. See
    # https://meta.wikimedia.org/wiki/User-Agent_policy
    headers = {
        "User-Agent": (
            "karin/0.1 (https://github.com/kaminglui/Karin; "
            "personal voice assistant; en) httpx"
        ),
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=8.0, headers=headers) as client:
            # 1. Resolve to a real article title.
            resp = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": q,
                    "limit": 1,
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                return f"Error: Wikipedia search failed [{resp.status_code}]"
            data = resp.json()
            titles = data[1] if len(data) > 1 else []
            if not titles:
                return f"No Wikipedia article found for '{q}'."
            title = titles[0]

            # 2. Fetch the summary extract.
            sum_resp = client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
            )
            if sum_resp.status_code != 200:
                return f"No summary available for '{title}'."
            s = sum_resp.json()
            extract = (s.get("extract") or "").strip()
            if not extract:
                return f"No summary text for '{title}'."
            # Cap to 3 sentences so the model doesn't dump a wall of text.
            sentences = extract.split(". ")
            short = ". ".join(sentences[:3])
            if not short.endswith("."):
                short += "."
            # Append computed age for biographical articles so the LLM
            # can answer "how old is X" without guessing.
            # Append computed date context: bio ages for people,
            # event dates for historical articles.
            date_info = _bio_age_suffix(extract) or _event_date_suffix(extract)
            return f"{title}: {short}{date_info}"
    except httpx.HTTPError as e:
        return f"Error fetching Wikipedia: {e}"
    except Exception as e:
        return f"Error: {e}"


_DISAMBIG_MARKERS = ("may refer to:", "may also refer to:", "can refer to:")



def _wiki(query: str | None = None) -> str:
    """Merged wiki tool — query → summary, empty → random article.

    Consolidates wiki_search + wiki_random into one schema so the LLM
    sees fewer nearly-identical tools and wastes fewer tokens picking
    between them.
    """
    q = (query or "").strip()
    if q:
        return _wiki_search(q)
    return _wiki_random()

