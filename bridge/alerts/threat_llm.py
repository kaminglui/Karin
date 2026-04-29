"""LLM verifier for borderline threat scores (Phase G.b).

Rule-based scoring (bridge.alerts.proximity) handles the clear cases:
a weather warning in the user's county is a threat; a story about
a country the user isn't in isn't. The interesting middle — scores
2 and 3, "is this actually dangerous or not?" — benefits from the
LLM reading the cluster text in context of the user.

Hallucination guardrails (flagged + promised earlier):

  1. Temperature 0 + strict JSON schema. Makes outputs repeatable.
  2. LLM can only adjust the rule-based score by ±1 tier. It can
     NEVER jump the score by 2+ in either direction. Hard clamp.
  3. Model must cite specific text from the cluster. No citation =
     no adjustment (return rule_score unchanged).
  4. On any parse error / HTTP error / bad JSON, fall back to the
     rule-based score. The LLM can only make a confident case
     stronger or reject an iffy one — it can't invent threats.
  5. Decisions are cached per (cluster_id, user_context_hash) so a
     single wrong call doesn't repeat every scan cycle. TTL 7 days
     keeps the cache from holding stale judgments across a news
     lifecycle.

Public API:
    ThreatVerifier(base_url, model, cache_path, request_timeout)
    verifier.verify(payload, rule_score, user_context) -> int
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from bridge.alerts.user_context import UserContext

log = logging.getLogger("bridge.alerts.threat_llm")


_SYSTEM_PROMPT = (
    "You are a safety assessor for news events. Given an event and a "
    "user's location, decide whether the event poses a physical or "
    "life-safety threat to the user. Output ONLY a JSON object with "
    "two keys: \"score\" (integer 0-4) and \"citation\" (a short "
    "quotation from the event that supports your call). No prose, "
    "no markdown fences, no extra keys. "
    "Score rubric: 0=no threat, 1=awareness, 2=watch, 3=advisory, "
    "4=critical/imminent. If you cannot find supporting text, echo "
    "the rule-based score unchanged."
)


def _build_messages(
    payload: dict, rule_score: int, ctx: UserContext,
) -> list[dict]:
    headline = payload.get("headline", "")
    wl_label = payload.get("watchlist_label", "")
    wl_type = payload.get("watchlist_type", "")
    state = payload.get("cluster_state", "")
    place = ", ".join(p for p in (ctx.city, ctx.region, ctx.country) if p) or "unknown"
    user_block = (
        f"USER LOCATION: {place}\n"
        f"RULE-BASED SCORE: {rule_score}\n\n"
        f"EVENT:\n"
        f"- Headline: {headline}\n"
        f"- Watchlist match: {wl_type}/{wl_label}\n"
        f"- Confidence: {state}\n\n"
        f"Respond with a JSON object containing score (0-4) and citation. "
        f"The score may be at most ±1 from the rule-based value."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)


def _extract_json(raw: str) -> dict | None:
    """Defensive JSON extraction. Strict parse first, then greedy
    find-first-brace. Returns None on any failure — caller falls back
    to the rule score, never raises."""
    if not raw:
        return None
    text = _CODE_FENCE_RE.sub("", raw.strip())
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            out = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return out if isinstance(out, dict) else None


def _context_hash(ctx: UserContext) -> str:
    """Stable 8-char hash of the fields the verifier sees. Same
    location + language -> same hash -> cache hit. Changing city
    invalidates — intended: a move to a new place should re-verify
    everything."""
    parts = f"{ctx.city}|{ctx.region}|{ctx.country}|{ctx.latitude}|{ctx.longitude}"
    return hashlib.sha1(parts.encode("utf-8")).hexdigest()[:8]


# --- cache record --------------------------------------------------------

@dataclass(frozen=True)
class _CacheEntry:
    """One cached LLM decision. Keys are (cluster_id, context_hash)."""
    score: int
    saved_at: datetime


# --- verifier ------------------------------------------------------------

class ThreatVerifier:
    """Ollama-backed borderline-score verifier with caching."""

    # Decisions expire after a week. News lifecycles shorter than
    # this, but re-asking more often wastes LLM time on stable calls.
    CACHE_TTL_DAYS = 7

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        cache_path: Path,
        request_timeout: float = 60.0,
        num_predict: int = 128,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache_path = Path(cache_path)
        self.request_timeout = float(request_timeout)
        self.num_predict = int(num_predict)
        self._cache: dict[str, _CacheEntry] | None = None
        self._lock = threading.Lock()

    # --- public --------------------------------------------------------

    def verify(
        self,
        payload: dict,
        rule_score: int,
        ctx: UserContext,
    ) -> int:
        """Return the final 0-4 threat score.

        Clamped to ``[max(0, rule_score-1), min(4, rule_score+1)]``
        per the hallucination guardrail. Serves from cache when a
        prior decision for (cluster_id, user-context-hash) exists
        and is fresh. Falls back to rule_score on any error path.
        """
        cluster_id = str(payload.get("cluster_id", ""))
        if not cluster_id:
            # Without a stable id we can't cache, so skip — let the
            # rule score stand. Verifier adds latency; run only when
            # its decisions can be cached.
            return rule_score

        key = f"{cluster_id}:{_context_hash(ctx)}"
        cached = self._read_cache(key)
        if cached is not None:
            return self._clamp(cached, rule_score)

        try:
            raw = self._call_ollama(_build_messages(payload, rule_score, ctx))
        except (httpx.HTTPError, ValueError) as e:
            log.debug("threat_llm: ollama error %s (using rule score %d)", e, rule_score)
            return rule_score

        parsed = _extract_json(raw)
        if not parsed:
            log.debug("threat_llm: unparseable LLM output; using rule score")
            return rule_score

        try:
            llm_score = int(parsed.get("score"))
        except (TypeError, ValueError):
            return rule_score
        citation = str(parsed.get("citation", "")).strip()
        if not citation:
            # Guardrail: no citation = no confidence. Fall back
            # rather than let an uncited judgement move the score.
            return rule_score

        final = self._clamp(llm_score, rule_score)
        self._write_cache(key, final)
        return final

    # --- internals -----------------------------------------------------

    @staticmethod
    def _clamp(llm_score: int, rule_score: int) -> int:
        """±1 tier hard clamp. Out-of-range llm_score (e.g. 9 from
        a hallucinating model) still collapses safely to within
        band of the rule-based value."""
        low = max(0, rule_score - 1)
        high = min(4, rule_score + 1)
        if llm_score < low:
            return low
        if llm_score > high:
            return high
        return llm_score

    def _call_ollama(self, messages: list[dict]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # `think` is a top-level field in Ollama's /api/chat
            # (memory: Ollama think field is top-level, not an
            # option). Qwen3 otherwise emits 700+ thinking tokens.
            "think": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 1024,
                "num_predict": self.num_predict,
            },
        }
        with httpx.Client(base_url=self.base_url, timeout=self.request_timeout) as client:
            resp = client.post("/api/chat", json=body)
            resp.raise_for_status()
            payload = resp.json()
        msg = payload.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"empty Ollama response: {payload!r}")
        return content

    # --- cache I/O -----------------------------------------------------

    def _load_cache(self) -> dict[str, _CacheEntry]:
        if self._cache is not None:
            return self._cache
        with self._lock:
            if self._cache is not None:
                return self._cache
            out: dict[str, _CacheEntry] = {}
            if self.cache_path.exists():
                try:
                    raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as e:
                    log.warning(
                        "threat-verifier cache at %s unreadable (%s); starting fresh",
                        self.cache_path, e,
                    )
                    raw = {}
                for k, v in (raw or {}).items():
                    try:
                        score = int(v.get("score"))
                        saved = datetime.fromisoformat(v.get("saved_at"))
                        if saved.tzinfo is None:
                            saved = saved.replace(tzinfo=timezone.utc)
                    except (TypeError, ValueError, AttributeError):
                        continue
                    out[str(k)] = _CacheEntry(score=score, saved_at=saved)
            self._cache = out
            return self._cache

    def _read_cache(self, key: str) -> int | None:
        cache = self._load_cache()
        entry = cache.get(key)
        if entry is None:
            return None
        now = datetime.now(timezone.utc)
        if (now - entry.saved_at) > timedelta(days=self.CACHE_TTL_DAYS):
            return None  # stale; let verify() re-ask
        return entry.score

    def _write_cache(self, key: str, score: int) -> None:
        cache = self._load_cache()
        with self._lock:
            cache[key] = _CacheEntry(
                score=int(score),
                saved_at=datetime.now(timezone.utc),
            )
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            out = {
                k: {
                    "score": e.score,
                    "saved_at": e.saved_at.isoformat(),
                }
                for k, e in cache.items()
            }
            self.cache_path.write_text(
                json.dumps(out, indent=2) + "\n",
                encoding="utf-8",
            )
