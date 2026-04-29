"""Tests for bridge.tools.

Focus areas (UI-7 additions):
  - _is_vague_location classifies "here" / "my location" / empty / None
    as vague, and explicit city names as non-vague.
  - _looks_like_coords recognises "lat,lon" pairs and rejects normal names.
  - _get_weather path coverage:
      * explicit city name -> Open-Meteo geocoder + forecast
      * vague location -> IP geolocate + forecast
      * lat,lon string -> forecast directly (no geocoder)
      * unresolved city -> clear one-line error
      * IP lookup failure when vague -> clear one-line error
      * common nicknames pass through (tests the schema / prompt
        contract; Open-Meteo handles the actual alias resolution).
  - _resolve_tracker_alias covers the new gasoline / rbob aliases and
    confirms existing ones still work.

All network I/O is patched via httpx.Client; no live calls happen.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from bridge import tools


@pytest.fixture(autouse=True)
def _clear_weather_cache():
    """Reset the in-memory weather cache between tests so a successful
    lookup in one test doesn't leak into another that expects the
    network path to actually run."""
    tools._WEATHER_CACHE.clear()
    yield
    tools._WEATHER_CACHE.clear()


# --- helpers --------------------------------------------------------------

def _mock_response(status: int = 200, payload: dict | None = None,
                   text: str | None = None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    r.text = text if text is not None else json.dumps(payload or {})
    r.raise_for_status = MagicMock(
        side_effect=None if status == 200 else Exception(f"HTTP {status}"),
    )
    return r


class _ClientStub:
    """Minimal httpx.Client substitute. Routes GET requests by URL
    substring to canned MagicMock responses. Usable as a context manager
    (entering returns itself, exit is a no-op)."""

    def __init__(self, routes: dict[str, MagicMock]):
        self._routes = routes
        self.get = MagicMock(side_effect=self._route_get)

    def _route_get(self, url, **_kwargs):
        for needle, resp in self._routes.items():
            if needle in url:
                return resp
        raise RuntimeError(f"unexpected URL in test: {url}")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_httpx_client(routes: dict[str, MagicMock]):
    """Return a patch context that replaces httpx.Client inside bridge.tools
    with a stub routed by URL substring."""
    return patch.object(tools.httpx, "Client", return_value=_ClientStub(routes))


# --- _is_vague_location + _looks_like_coords ------------------------------

class TestVagueLocation:
    @pytest.mark.parametrize("raw", [
        None, "", "   ", "here", "Here", "here.", "HERE!",
        "my location", "current location", "where i am",
        "my area", "nearby", "local", "local weather",
    ])
    def test_vague(self, raw):
        assert tools._is_vague_location(raw) is True

    @pytest.mark.parametrize("raw", [
        "Philadelphia", "Tokyo", "Philly", "NYC", "London",
        "40.7,-74.0", "State College PA",
    ])
    def test_specific(self, raw):
        assert tools._is_vague_location(raw) is False


class TestLooksLikeCoords:
    @pytest.mark.parametrize("raw,expected", [
        ("40.7,-74.0", True),
        ("40.7, -74.0", True),
        ("-33.86,151.21", True),
        ("0,0", True),
        ("Philadelphia", False),
        ("Philadelphia, PA", False),
        ("40.7", False),
        ("abc,def", False),
        ("40.7,def", False),
    ])
    def test(self, raw, expected):
        assert tools._looks_like_coords(raw) is expected


# --- _get_weather happy paths --------------------------------------------

# Canned Open-Meteo forecast payload shared across tests.
_FORECAST_OK = {
    "current": {
        "temperature_2m": 12.3,
        "apparent_temperature": 11.0,
        "precipitation": 0.0,
        "wind_speed_10m": 8.5,
        "weather_code": 2,            # "partly cloudy"
        "relative_humidity_2m": 60,
    },
}


class TestGetWeatherExplicit:
    def test_named_city_hits_geocoder_then_forecast(self):
        geo_resp = _mock_response(payload={
            "results": [{
                "name": "Philadelphia",
                "latitude": 39.95, "longitude": -75.17,
                "admin1": "Pennsylvania", "country": "United States",
            }],
        })
        wx_resp = _mock_response(payload=_FORECAST_OK)
        with _patch_httpx_client({
            "geocoding-api.open-meteo.com": geo_resp,
            "api.open-meteo.com": wx_resp,
        }):
            out = tools._get_weather("Philadelphia")
        assert "Philadelphia, Pennsylvania, United States" in out
        assert "partly cloudy" in out
        assert "12.3" in out

    def test_nickname_passed_through_to_geocoder(self):
        # "Philly" is passed straight to Open-Meteo. Simulate that
        # Open-Meteo's geocoder knows it maps to Philadelphia by
        # returning that as the first result — i.e. we're verifying
        # we don't short-circuit or transform the name client-side.
        geo_resp = _mock_response(payload={
            "results": [{
                "name": "Philadelphia",
                "latitude": 39.95, "longitude": -75.17,
                "admin1": "Pennsylvania", "country": "United States",
            }],
        })
        wx_resp = _mock_response(payload=_FORECAST_OK)
        captured_params = {}

        def geo_side_effect(url, **kwargs):
            captured_params.update(kwargs.get("params", {}))
            return geo_resp

        client = _ClientStub({
            "api.open-meteo.com": wx_resp,
        })
        # Custom route for the geocoder so we can inspect the params
        # the tool actually sends.
        def route(url, **kw):
            if "geocoding-api.open-meteo.com" in url:
                return geo_side_effect(url, **kw)
            if "api.open-meteo.com" in url:
                return wx_resp
            raise RuntimeError(f"unexpected URL: {url}")
        client.get = MagicMock(side_effect=route)
        with patch.object(tools.httpx, "Client", return_value=client):
            out = tools._get_weather("Philly")

        # The tool forwarded the user's word verbatim; no local alias
        # rewriting happened.
        assert captured_params.get("name") == "Philly"
        # And it rendered the geocoder's canonical name, not "Philly".
        assert "Philadelphia" in out


class TestGetWeatherCoords:
    def test_lat_lon_bypasses_geocoder(self):
        wx_resp = _mock_response(payload=_FORECAST_OK)
        client = _ClientStub({"api.open-meteo.com": wx_resp})
        with patch.object(tools.httpx, "Client", return_value=client):
            out = tools._get_weather("40.7,-74.0")
        # No geocoder call; only the forecast URL was hit.
        urls_hit = [call.args[0] for call in client.get.call_args_list]
        assert all("geocoding-api" not in u for u in urls_hit)
        assert "40.70,-74.00" in out


# --- _get_weather IP fallback --------------------------------------------

_IPAPI_OK = {
    # ipwho.is response shape: success flag + flat city/region/country
    # fields (country is the full name, not country_code). Matches what
    # bridge.tools._ip_geolocate parses.
    "success": True,
    "ip": "1.2.3.4",
    "city": "Philadelphia",
    "region": "Pennsylvania",
    "country": "United States",
    "country_code": "US",
    "latitude": 39.9526,
    "longitude": -75.1652,
}


class TestGetWeatherVagueUsesIP:
    @pytest.mark.parametrize("vague", ["", "here", "my location", None])
    def test_vague_uses_ipapi(self, vague, monkeypatch):
        # _ip_geolocate now checks a user_location override first.
        # Stub it out so the test exercises the HTTP-fallback branch.
        monkeypatch.setattr(
            "bridge.location._load_user_override",
            lambda: {"city": "", "region": "", "country": "", "timezone": "", "latitude": None, "longitude": None},
        )
        wx_resp = _mock_response(payload=_FORECAST_OK)
        ip_resp = _mock_response(payload=_IPAPI_OK)
        client = _ClientStub({
            "ipwho.is": ip_resp,
            "api.open-meteo.com": wx_resp,
        })
        with patch.object(tools.httpx, "Client", return_value=client):
            out = tools._get_weather(vague)
        # IP endpoint was called; geocoder was NOT.
        urls = [c.args[0] for c in client.get.call_args_list]
        assert any("ipwho.is" in u for u in urls)
        assert all("geocoding-api.open-meteo.com" not in u for u in urls)
        # Place name from the IP lookup shows up in the output.
        assert "Philadelphia, Pennsylvania, United States" in out

    def test_ip_lookup_failure_surfaces_clear_error(self, monkeypatch):
        # IP provider returns an unusable response. The tool should NOT
        # hit the forecast endpoint; it should return a neutral
        # "couldn't detect" message the LLM can paraphrase. Override
        # stubbed to empty so we exercise the HTTP-failure branch.
        monkeypatch.setattr(
            "bridge.location._load_user_override",
            lambda: {"city": "", "region": "", "country": "", "timezone": "", "latitude": None, "longitude": None},
        )
        broken_ip_resp = _mock_response(payload={"success": False, "message": "rate limited"})
        client = _ClientStub({"ipwho.is": broken_ip_resp})
        with patch.object(tools.httpx, "Client", return_value=client):
            out = tools._get_weather("")
        assert "couldn't detect" in out.lower()
        urls = [c.args[0] for c in client.get.call_args_list]
        assert all("api.open-meteo.com" not in u for u in urls)


# --- _get_weather unresolved location ------------------------------------

class TestGetWeatherUnknownCity:
    def test_no_geocoder_results_returns_clear_error(self):
        empty_geo = _mock_response(payload={"results": []})
        client = _ClientStub({"geocoding-api.open-meteo.com": empty_geo})
        with patch.object(tools.httpx, "Client", return_value=client):
            out = tools._get_weather("Atlantisville")
        assert "couldn't find" in out.lower()
        assert "Atlantisville" in out
        # Didn't attempt the forecast call. Check for the specific
        # forecast path — geocoding-api.open-meteo.com is a substring
        # collision with api.open-meteo.com, so match on the full path.
        urls = [c.args[0] for c in client.get.call_args_list]
        assert all("/v1/forecast" not in u for u in urls)


# --- tracker alias cleanup (new aliases shipped this session) ------------

class TestTrackerAliases:
    @pytest.mark.parametrize("alias,expected", [
        # Casual "gas" / "gasoline" now point at EIA retail, not RBOB
        # futures — most users mean pump price when they ask.
        ("gasoline", "gas_retail"),
        ("GASOLINE", "gas_retail"),
        ("  gasoline  ", "gas_retail"),
        ("gas", "gas_retail"),
        ("pump", "gas_retail"),
        # Crypto aliases (placeholders; disabled by default in config).
        ("btc", "btc_usd"),
        ("bitcoin", "btc_usd"),
        ("eth", "eth_usd"),
        ("ethereum", "eth_usd"),
        # Existing aliases still resolve.
        ("gold", "gold_usd"),
        ("usd/cny", "usd_cny"),
        ("food", "us_cpi_food"),
        # Canonical ids pass through.
        ("gas_retail", "gas_retail"),
        ("usd_cny", "usd_cny"),
    ])
    def test_resolves(self, alias, expected):
        assert tools._resolve_tracker_alias(alias) == expected

    def test_unknown_passes_through_lowercased(self):
        # Per the design note in tools.py, unknown inputs pass through
        # unchanged so the service layer produces the 'not found' msg.
        assert tools._resolve_tracker_alias("not_a_tracker") == "not_a_tracker"
        assert tools._resolve_tracker_alias("Not-A-Tracker") == "not-a-tracker"

    def test_philly_variants_are_not_aliased(self):
        # "philly_gasoline" / "philly_gas" should pass through so the
        # service returns 'not found' instead of silently resolving to
        # the wrong tracker.
        assert tools._resolve_tracker_alias("philly_gasoline") == "philly_gasoline"
        assert tools._resolve_tracker_alias("philly_gas") == "philly_gas"


class TestDispatcherKwargFiltering:
    """Unknown kwargs should be dropped, not raise. Keeps small LLMs
    from parroting Python tracebacks when they hallucinate args."""

    def test_drops_unknown_kwarg_silently(self, monkeypatch):
        calls: list[dict] = []
        def fake(topic: str | None = None) -> str:
            calls.append({"topic": topic})
            return "ok"
        monkeypatch.setitem(tools._DISPATCH, "__test_tool", fake)
        # 'location' isn't in the signature — should be filtered out,
        # the function should still run with the valid kwargs.
        result = tools.execute("__test_tool", {"topic": "x", "location": "NYC"})
        assert result == "ok"
        assert calls == [{"topic": "x"}]

    def test_drops_all_unknown_kwargs_no_call_still_runs(self, monkeypatch):
        # Even if ALL kwargs are unknown, the function runs with no args
        # (its defaults handle the nothing-supplied case).
        calls: list[str] = []
        def fake(topic: str | None = None) -> str:
            calls.append(topic or "no topic")
            return "ran"
        monkeypatch.setitem(tools._DISPATCH, "__test_tool", fake)
        result = tools.execute("__test_tool", {"bogus": 1, "also_bogus": 2})
        assert result == "ran"
        assert calls == ["no topic"]

    def test_var_keyword_functions_pass_through_all_args(self, monkeypatch):
        # If a tool explicitly takes **kwargs, don't filter.
        seen: dict = {}
        def fake(**kwargs) -> str:
            seen.update(kwargs)
            return "ok"
        monkeypatch.setitem(tools._DISPATCH, "__test_tool", fake)
        tools.execute("__test_tool", {"a": 1, "b": 2})
        assert seen == {"a": 1, "b": 2}


class TestConsolidatedTools:
    """The tool schemas were slimmed 16 → 12 by merging wiki_search +
    wiki_random into a single `wiki` and get_tracker + get_trackers
    into a single `tracker`. Verify both merged entries route correctly
    at dispatch time, and that legacy names still resolve (for
    restored conversation widget replay)."""

    def test_wiki_with_query_routes_to_search(self, monkeypatch):
        # Patch at the submodule level so the local call inside `_wiki`
        # resolves to the fake — patching `tools._wiki_search` doesn't
        # intercept module-local references. Use importlib because the
        # `bridge.tools._wiki` attribute is the function, not the module
        # (the package shim deliberately pins same-name functions to
        # avoid masking them with the submodule).
        import importlib
        wiki_mod = importlib.import_module("bridge.tools._wiki")
        called: dict = {}
        def fake_search(q: str) -> str:
            called["search"] = q
            return "search-ok"
        def fake_random() -> str:
            called["random"] = True
            return "random-ok"
        monkeypatch.setattr(wiki_mod, "_wiki_search", fake_search)
        monkeypatch.setattr(wiki_mod, "_wiki_random", fake_random)
        result = tools._wiki(query="Ada Lovelace")
        assert result == "search-ok"
        assert called == {"search": "Ada Lovelace"}

    def test_wiki_empty_query_routes_to_random(self, monkeypatch):
        import importlib
        wiki_mod = importlib.import_module("bridge.tools._wiki")
        monkeypatch.setattr(wiki_mod, "_wiki_search", lambda q: "should-not-run")
        monkeypatch.setattr(wiki_mod, "_wiki_random", lambda: "random-ok")
        assert tools._wiki() == "random-ok"
        assert tools._wiki(query="") == "random-ok"
        assert tools._wiki(query="   ") == "random-ok"

    def test_tracker_with_id_routes_to_single(self, monkeypatch):
        import importlib
        tracker_mod = importlib.import_module("bridge.tools._tracker")
        called: dict = {}
        def fake_single(tid: str) -> str:
            called["id"] = tid
            return "single"
        monkeypatch.setattr(tracker_mod, "_get_tracker", fake_single)
        monkeypatch.setattr(tracker_mod, "_get_trackers", lambda: "grid")
        assert tools._tracker(id="gold") == "single"
        assert called == {"id": "gold"}

    def test_tracker_empty_id_shows_grid(self, monkeypatch):
        import importlib
        tracker_mod = importlib.import_module("bridge.tools._tracker")
        monkeypatch.setattr(tracker_mod, "_get_tracker", lambda tid: "should-not-run")
        monkeypatch.setattr(tracker_mod, "_get_trackers", lambda: "grid")
        assert tools._tracker() == "grid"
        assert tools._tracker(id="") == "grid"

    def test_tool_schemas_count_and_expected_names(self):
        """Lock in the current set so a future accidental re-addition or
        removal is caught immediately. 21 tools."""
        names = {
            (s.get("function") or {}).get("name")
            for s in tools.TOOL_SCHEMAS
        }
        expected = {
            "get_time", "get_weather", "get_news", "get_alerts",
            "get_digest",
            "tracker", "math", "find_places", "web_search",
            "wiki", "convert", "graph", "circuit",
            "schedule_reminder", "update_memory", "say",
            "inflation", "population", "facts", "analyze", "alice",
        }
        assert names == expected, f"unexpected tool set: {names ^ expected}"

    def test_facts_aggregates_world_population_and_inflation(self):
        from bridge.tools._facts import _facts
        result = json.loads(_facts(year=1985))
        assert "error" not in result
        assert result["year"] == 1985
        assert "world_population" in result["sections"]
        assert "inflation_baseline" in result["sections"]
        # World population in 1985 was ~4.85B.
        assert 4_500_000_000 < result["sections"]["world_population"]["population"] < 5_200_000_000
        # US inflation baseline is the default region for facts.
        infl = result["sections"]["inflation_baseline"]
        assert infl["region"] == "us"
        assert infl["currency"] == "USD"
        # Aggregated interpretation joins both sentences.
        assert "billion" in result["interpretation"]
        assert "1985" in result["interpretation"]

    def test_facts_with_region_includes_region_population(self):
        from bridge.tools._facts import _facts
        result = json.loads(_facts(year=1985, region="japan"))
        assert "region_population" in result["sections"]
        assert result["sections"]["region_population"]["country"] == "Japan"
        # Japan inflation baseline takes over (currency = JPY).
        assert result["sections"]["inflation_baseline"]["currency"] == "JPY"

    def test_population_rank_returns_top_n_with_region_position(self):
        """v1 rank: top-N most populous + region's position within global list."""
        from bridge.tools._population import _population
        out = json.loads(_population(metric="rank", year=2024, top=5, region="us"))
        assert "error" not in out
        assert len(out["top"]) == 5
        # Top 2 in 2024 are India + China (India passed China ~2023).
        top_iso3 = {r["iso3"] for r in out["top"][:2]}
        assert "IND" in top_iso3 and "CHN" in top_iso3
        # US should be in top-5 in 2024.
        assert any(r["iso3"] == "USA" for r in out["top"])
        # region_position should resolve to USA.
        assert out["region_position"]["iso3"] == "USA"
        assert 1 <= out["region_position"]["rank"] <= 10
        assert out["total_countries"] >= 200  # ~217 sovereign

    def test_population_rank_top_clamped_and_default(self):
        from bridge.tools._population import _population
        # top=100 should clamp to 50.
        out = json.loads(_population(metric="rank", year=2024, top=100))
        assert len(out["top"]) <= 50
        # default top=10
        out = json.loads(_population(metric="rank", year=2024))
        assert len(out["top"]) == 10

    def test_population_rank_extractor_detects_phrasings(self):
        from bridge.tools._population import extract_population_args
        assert extract_population_args(
            "top 5 most populous countries in 2024"
        ) == {"year": 2024, "metric": "rank", "top": 5}
        assert extract_population_args(
            "biggest country by population in 1990"
        ).get("metric") == "rank"
        # "world population in 2020" should NOT trigger rank.
        assert "metric" not in extract_population_args("world population in 2020")

    def test_facts_extractor_pulls_year_and_optional_region(self):
        from bridge.tools._facts import extract_facts_args
        assert extract_facts_args("Tell me about 1985") == {"year": 1985}
        assert extract_facts_args("fun facts about Japan in 1985") == {
            "year": 1985, "region": "japan"
        }
        assert extract_facts_args("hello") == {}

    def test_population_world_at_year(self):
        """v1 population: world aggregate at 1985 should be ~4.85B."""
        from bridge.tools._population import _population
        result = json.loads(_population(year=1985))
        assert "error" not in result
        assert result["region"] == "world"
        assert result["country"] == "World"
        # 1985 world pop ~4.85B; allow generous band so future World
        # Bank revisions don't break the test.
        assert 4_500_000_000 < result["population"] < 5_200_000_000
        assert "billion" in result["interpretation"]

    def test_population_region_at_year(self):
        from bridge.tools._population import _population
        result = json.loads(_population(region="us", year=2020))
        # US 2020 ~330M.
        assert 320_000_000 < result["population"] < 345_000_000
        assert result["country"] == "United States"

    def test_population_range_query_returns_change(self):
        from bridge.tools._population import _population
        result = json.loads(_population(region="cn_mainland", from_year=1980, to_year=2020))
        assert "error" not in result
        assert "change_pct" in result
        assert "change_abs" in result
        # China 1980→2020 grew ~40% — 430M absolute.
        assert result["change_pct"] > 30
        assert result["change_abs"] > 300_000_000
        assert "grew" in result["interpretation"].lower()

    def test_population_out_of_range_errors(self):
        from bridge.tools._population import _population
        result = json.loads(_population(year=1500))
        assert "error" in result
        assert "1960" in result["error"]

    def test_population_extractor_pulls_region_and_year(self):
        from bridge.tools._population import extract_population_args
        assert extract_population_args(
            "How many people lived in Japan in 1970?"
        ) == {"region": "japan", "year": 1970}
        assert extract_population_args(
            "world population in 1985"
        ) == {"region": "world", "year": 1985}
        assert extract_population_args(
            "China population from 1980 to 2020"
        ) == {"region": "cn_mainland", "from_year": 1980, "to_year": 2020}
        # Decoy
        assert extract_population_args("When was the Berlin Wall built?") == {}

    def test_inflation_basic_cpi_calc(self):
        """Spot-check $1 in 1970 → $7.92 in 2026 (within rounding).
        Uses the shipped CPI data; won't drift unless data is updated."""
        from bridge.tools._inflation import _inflation
        result = json.loads(_inflation(amount=1.0, from_year=1970, to_year=2026))
        assert "cpi" in result
        assert result["cpi"]["amount_output"] > 7.0
        assert result["cpi"]["amount_output"] < 9.0
        assert result["cpi"]["confidence"] == "high"
        assert "BLS" in result["cpi"]["source"]

    def test_inflation_rejects_pre_data(self):
        """v2 extends CPI floor to 1860 (NBER spliced). 1800 still rejected."""
        from bridge.tools._inflation import _inflation
        result = json.loads(_inflation(amount=1.0, from_year=1800))
        assert "error" in result
        assert "1860" in result["error"]

    def test_inflation_pre_1913_uses_nber_with_low_confidence(self):
        """v2 pre-1913: $1 in 1900 should compute (NBER spliced) and
        carry confidence='low' to flag the historical methodology."""
        from bridge.tools._inflation import _inflation
        result = json.loads(_inflation(amount=1.0, from_year=1900, to_year=2026))
        assert "error" not in result
        # 1900 NBER ≈ 7.87 rebased to CPI-U; 2026 ≈ 330.2 → ratio ~42×
        assert 30 < result["cpi"]["ratio"] < 60
        assert result["cpi"]["confidence"] == "low"

    def test_inflation_wiki_redirect_pattern_catches_money_queries(self):
        """The wiki→inflation redirect regex must catch the iter-3 LoRA's
        common wiki-misroutes (the ones that drove this fix): dollar-in-
        YYYY, worth today, purchasing power. Decoy wiki queries (Berlin
        Wall in 1989, Berlin Wall built) must NOT match."""
        from bridge.tools._inflation import _WIKI_TO_INFLATION_REDIRECT_RE
        # Should redirect (LoRA picked wiki on these, we want inflation):
        for q in [
            "dollar in 1865 worth today",
            "dollar in 1900",
            "1865 dollars",
            "in 1970 money",
            "purchasing power of 1970 dollar",
            "X adjusted for inflation",
        ]:
            assert _WIKI_TO_INFLATION_REDIRECT_RE.search(q), q
        # Should NOT redirect (legitimate wiki queries that happen to
        # mention years or money in non-inflation contexts):
        for q in [
            "Berlin Wall built in 1989",
            "Berlin Wall",
            "Statue of Liberty",
            "history of the dollar bill",
            "George Washington biography",
        ]:
            assert not _WIKI_TO_INFLATION_REDIRECT_RE.search(q), q

    def test_inflation_hk_sar_returns_hkd_with_correct_label(self):
        """v3 Hong Kong SAR, China — World Bank CPI 1981+ in HKD."""
        from bridge.tools._inflation import _inflation
        out = json.loads(_inflation(amount=1.0, from_year=1990, region="hk_sar"))
        assert "error" not in out
        assert out["region"] == "hk_sar"
        assert out["currency"] == "HKD"
        assert out["country"] == "Hong Kong SAR, China"
        # 1990 HK CPI ≈ 50.4, 2024 ≈ 145 → ratio ~2.5-3x. Wide band for safety.
        assert 1.5 < out["cpi"]["ratio"] < 4.0
        # Currency symbol in interpretation must NOT be USD-style "$1.00".
        assert "HK$" in out["interpretation"]

    def test_inflation_hk_wages_and_items_return_friendly_errors(self):
        """v3 regions other than US don't ship wages/items datasets —
        the tool must surface a clear error block, not crash or fabricate."""
        from bridge.tools._inflation import _inflation
        out = json.loads(_inflation(amount=1.0, from_year=1990, region="hk_sar", measure="wages"))
        assert "wages" in out
        assert "error" in out["wages"]
        assert "us" in out["wages"]["error"].lower()
        out = json.loads(_inflation(amount=1.0, from_year=1990, region="hk_sar", item="gas"))
        assert "item" in out
        assert "error" in out["item"]

    def test_inflation_unknown_region_errors_cleanly(self):
        from bridge.tools._inflation import _inflation
        out = json.loads(_inflation(amount=1.0, from_year=1990, region="atlantis"))
        assert "error" in out
        assert "Unknown region" in out["error"]

    def test_inflation_comparison_returns_per_region_blocks(self):
        """v3 comparison mode: regions='us,japan' returns a comparison
        map with one CPI block per region + a spread interpretation."""
        from bridge.tools._inflation import _inflation
        out = json.loads(_inflation(from_year=1990, regions="us,japan"))
        assert "error" not in out
        assert "comparison" in out
        assert set(out["comparison"].keys()) == {"us", "japan"}
        # US 1990→2024 ratio is ~2.5×; Japan ~1.2× (deflationary era).
        us_ratio = out["comparison"]["us"]["ratio"]
        jp_ratio = out["comparison"]["japan"]["ratio"]
        assert us_ratio > jp_ratio  # US inflated more than Japan
        assert "interpretation" in out
        assert "spread" in out["interpretation"].lower() or "highest" in out["interpretation"].lower()

    def test_inflation_extractor_detects_comparison_phrasings(self):
        from bridge.tools._inflation import extract_inflation_args
        e = extract_inflation_args("Compare inflation in the US and Japan since 1990")
        assert "regions" in e
        assert "us" in e["regions"] and "japan" in e["regions"]
        # "vs" cue
        e = extract_inflation_args("US vs Japan inflation")
        assert "regions" in e
        # Decoy: "Tell us about" is the pronoun, not the country.
        e = extract_inflation_args("Tell us about bread")
        assert "regions" not in e

    def test_inflation_extractor_detects_hk_keywords(self):
        from bridge.tools._inflation import extract_inflation_args
        assert extract_inflation_args("How much was a dollar in 1990 in Hong Kong?")["region"] == "hk_sar"
        assert extract_inflation_args("Hong Kong CPI in 1985")["region"] == "hk_sar"
        assert extract_inflation_args("HKD in 2000")["region"] == "hk_sar"
        # US queries don't get region in extractor (default "us" applies).
        assert "region" not in extract_inflation_args("How much is $1 in 1970 worth today?")

    def test_inflation_default_region_us_unchanged(self):
        """Backwards compat: omitting region behaves exactly like v2 US."""
        from bridge.tools._inflation import _inflation
        a = json.loads(_inflation(amount=1.0, from_year=1970))
        b = json.loads(_inflation(amount=1.0, from_year=1970, region="us"))
        assert a["cpi"]["amount_output"] == b["cpi"]["amount_output"]
        assert b["region"] == "us"

    def test_inflation_pre_1913_splice_continuity_at_overlap(self):
        """The rebase point: 1912 NBER value should be within ~1% of
        1913 BLS-CPI-U value, so the spliced series is continuous."""
        from bridge.tools._inflation import _load_cpi
        cpi = _load_cpi()
        v_1912 = float(cpi["annual"]["1912"])
        v_1913 = float(cpi["annual"]["1913"])
        # Within 5% — splice quality check (both should be ~9.9).
        assert abs(v_1912 - v_1913) / v_1913 < 0.05

    def test_inflation_rejects_negative_amount(self):
        from bridge.tools._inflation import _inflation
        result = json.loads(_inflation(amount=-5, from_year=1970))
        assert "error" in result

    def test_inflation_wages_computes_real_ratio(self):
        """v1.5: wages dataset is shipped — measure='wages' returns
        non-zero wage_from/wage_to/wage_ratio with the BLS source cited."""
        from bridge.tools._inflation import _inflation
        result = json.loads(_inflation(amount=1.0, from_year=1970, measure="wages"))
        assert "wages" in result
        wages = result["wages"]
        assert "error" not in wages
        # 1970 AHETPI ~$3.40, latest ~$32 — ratio comfortably 8-12×.
        assert 5.0 < wages["wage_ratio"] < 15.0
        assert "BLS" in wages["source"]

    def test_inflation_measure_both_includes_real_wage_delta(self):
        """measure='both' should expose `real_wage_delta` = wages/cpi
        ratio so the LLM can say "real wages outpaced prices by X%"
        without inventing the math."""
        from bridge.tools._inflation import _inflation
        result = json.loads(_inflation(amount=1.0, from_year=1970, measure="both"))
        assert "real_wage_delta" in result
        # Production-worker real wages have grown modestly since 1970,
        # so the delta should be near 1 (0.8 < x < 1.5 is a wide band
        # that won't trip on small BLS revisions).
        assert 0.8 < result["real_wage_delta"] < 1.5

    def test_inflation_wages_pre_1964_returns_friendly_error(self):
        """Wages start 1964; earlier queries should error cleanly inside
        the wages block, not crash or fabricate. CPI block stays intact."""
        from bridge.tools._inflation import _inflation
        result = json.loads(_inflation(amount=1.0, from_year=1950, measure="wages"))
        assert "wages" in result
        assert "error" in result["wages"]
        assert "1964" in result["wages"]["error"]
        # CPI block is still computed (1950 is in the CPI range).
        assert result["cpi"]["amount_output"] > 1.0

    def test_inflation_pre_1947_is_medium_confidence(self):
        from bridge.tools._inflation import _inflation
        result = json.loads(_inflation(amount=1.0, from_year=1920))
        assert result["cpi"]["confidence"] == "medium"

    def test_inflation_defaults_amount_to_one_when_omitted(self):
        """LoRA frequently calls inflation without amount on phrasings
        like 'in 1970 money?'. v1 used to error with 'amount must be
        positive' — now it treats missing/zero as $1."""
        from bridge.tools._inflation import _inflation
        result = json.loads(_inflation(from_year=1970, to_year=2026))
        assert result.get("amount_input") == 1.0
        assert "error" not in result

    def test_inflation_extractor_pulls_amount_and_year(self):
        from bridge.tools._inflation import extract_inflation_args
        assert extract_inflation_args("How much is a dollar in 1970 money?") == {
            "amount": 1.0, "from_year": 1970,
        }
        assert extract_inflation_args("What is $50 in 1980 worth today?") == {
            "amount": 50.0, "from_year": 1980,
        }
        assert extract_inflation_args("100 dollars in 1950") == {
            "amount": 100.0, "from_year": 1950,
        }
        # 4-digit "amount" that's actually a year — treat as period
        # dollars, not a $1925 amount.
        assert extract_inflation_args("1925 dollars in 2026") == {
            "from_year": 1925, "to_year": 2026,
        }
        assert extract_inflation_args("5 cents in 1925")["amount"] == 0.05
        assert extract_inflation_args("nothing relevant here") == {}

    def test_inflation_extractor_infers_measure_from_keywords(self):
        """Wage-question phrasings promote measure→'wages'/'both' so the
        LoRA's default 'cpi' doesn't strip wage data out of the response."""
        from bridge.tools._inflation import extract_inflation_args
        # Comparison phrasings → 'both' (so the JSON includes
        # real_wage_delta and a no-fabrication-needed direction line).
        assert extract_inflation_args(
            "How have wages kept up with inflation since 1970?"
        )["measure"] == "both"
        assert extract_inflation_args(
            "Real wages from 1970 to 2020"
        )["measure"] == "both"
        # Plain wage queries → 'wages'.
        assert extract_inflation_args(
            "What was the average wage in 1980 compared to today?"
        )["measure"] == "wages"
        assert extract_inflation_args("salary in 1980")["measure"] == "wages"
        # Pure CPI queries → no measure key (defaults to 'cpi').
        assert "measure" not in extract_inflation_args(
            "How much is $1 in 1970 worth today?"
        )

    def test_inflation_wages_interpretation_states_direction(self):
        """The wages-only interpretation must contain explicit
        'outpaced'/'lagged' wording — without it the LoRA inverted
        '9.4× wages vs 8.5× prices' on live test."""
        from bridge.tools._inflation import _inflation
        out = json.loads(_inflation(amount=1.0, from_year=1970, measure="wages"))
        text = out["interpretation"].lower()
        assert "outpaced" in text or "lagged" in text
        assert "real_wage_delta" in out

    def test_inflation_widget_data_returns_series(self):
        """v2 widget endpoint helper: same calculation block + year-by-year
        series for plotting. Series start at from_year going forward."""
        from bridge.tools._inflation import inflation_widget_data
        d = inflation_widget_data(amount=1.0, from_year=1970, measure="both")
        assert "error" not in d
        assert "series" in d
        assert "cpi" in d["series"]
        cpi = d["series"]["cpi"]
        assert cpi["years"][0] == 1970
        # Value at from_year should equal $amount (purchasing-power
        # baseline).
        assert abs(cpi["values"][0] - 1.0) < 0.01
        # Forward-plotted value at the latest year matches the calc's
        # amount_output (sanity check: chart and headline agree).
        assert abs(cpi["values"][-1] - d["cpi"]["amount_output"]) < 0.05
        # With wages dataset shipped, both wage series should be present.
        assert "wages_nominal" in d["series"]
        assert "wages_real" in d["series"]
        assert d["series"]["wages_real"]["values"][0] > 0

    def test_inflation_item_lookup_returns_real_bls_price(self):
        """v2 item-prices: the tool should return the BLS AP price at
        from_year and (when available) at to_year, plus a today's-dollar
        equivalent via CPI. Concrete spot-check: gas in 1985 was ~$1.20."""
        from bridge.tools._inflation import _inflation
        out = json.loads(_inflation(item="gasoline", from_year=1985))
        assert "item" in out
        item = out["item"]
        assert "error" not in item
        # 1985 regular gasoline annual avg ~$1.20 (BLS AP). Allow a wide
        # band so BLS revisions don't break the test.
        assert 1.0 < item["price_from"] < 1.5
        assert "BLS" in item["source"]
        # Interpretation should mention the item.
        assert "gasoline" in out["interpretation"].lower()

    def test_inflation_item_pre_1980_returns_friendly_error(self):
        """Pre-1980 item queries should error in the item block (data
        floor) without crashing or computing a CPI-extrapolated estimate
        — items inflate at different rates, extrapolation misleads."""
        from bridge.tools._inflation import _inflation
        out = json.loads(_inflation(item="gasoline", from_year=1965))
        assert "item" in out
        assert "error" in out["item"]
        assert "1980" in out["item"]["error"]
        # CPI block is unaffected.
        assert out["cpi"]["amount_output"] > 1.0

    def test_inflation_item_unknown_returns_available_list(self):
        """Unknown items should error with a list the LLM can paraphrase
        ('I don't have X but I do have eggs/bread/...')."""
        from bridge.tools._inflation import _inflation
        out = json.loads(_inflation(item="caviar", from_year=2000))
        assert "item" in out
        assert "error" in out["item"]
        assert "available" in out["item"]
        assert "eggs" in out["item"]["available"]

    def test_inflation_extractor_pulls_item_from_user_text(self):
        from bridge.tools._inflation import extract_inflation_args
        # Multi-word items should beat shorter overlapping ones.
        assert extract_inflation_args(
            "natural gas cost in 2010"
        )["item"] == "natural_gas"
        assert extract_inflation_args(
            "ground beef in 1985"
        )["item"] == "ground_beef"
        # Bare "gas" defaults to gasoline.
        assert extract_inflation_args(
            "how much was gas in 1985?"
        )["item"] == "gasoline"
        # Common single-word items.
        assert extract_inflation_args("eggs in 1990")["item"] == "eggs"
        assert extract_inflation_args("price of bread in 2000")["item"] == "bread"
        # No item keyword → no item key in output.
        assert "item" not in extract_inflation_args(
            "how much is $1 in 1970 worth today?"
        )

    def test_say_echoes_input_verbatim(self):
        from bridge.tools._say import _say
        assert _say("hello world") == "hello world"
        assert _say("  trailing space   ") == "trailing space"

    def test_say_empty_input_returns_friendly_marker(self):
        from bridge.tools._say import _say
        out = _say("")
        assert out and "no text" in out.lower()
        assert _say(None) and "no text" in _say(None).lower()  # type: ignore[arg-type]

    def test_say_truncates_overlong_input(self):
        from bridge.tools._say import _MAX_LEN, _say
        long = "x" * (_MAX_LEN + 50)
        out = _say(long)
        assert out.startswith("x" * _MAX_LEN)
        assert out.endswith("…")
        assert len(out) == _MAX_LEN + 1   # trimmed body + ellipsis char

    def test_retired_tools_not_in_schemas(self):
        """tell_story, today_in_history, wiki_search, wiki_random,
        get_tracker, get_trackers must NOT be in the LLM-facing list."""
        names = {
            (s.get("function") or {}).get("name")
            for s in tools.TOOL_SCHEMAS
        }
        for retired in ("tell_story", "today_in_history",
                        "wiki_search", "wiki_random",
                        "get_tracker", "get_trackers"):
            assert retired not in names, f"{retired} leaked back into TOOL_SCHEMAS"

    def test_legacy_wiki_search_dispatch_still_resolves(self):
        """Conversation histories written before the merge contain
        'wiki_search' (audit 2026-04-15: 10 references in persisted
        conversations). Dispatch must still accept it so widget replay
        works. Other retired names were dropped after the audit found
        zero references."""
        assert "wiki_search" in tools._DISPATCH

    def test_retired_legacy_names_are_not_dispatched(self):
        """After the audit-driven cleanup, these names should NOT be
        in _DISPATCH. Their implementations are still callable as
        module-level functions if needed, but the dispatcher will
        return the friendly 'unknown tool' message."""
        for dropped in ("tell_story", "today_in_history",
                        "wiki_random", "get_tracker",
                        "get_trackers", "fill_format"):
            assert dropped not in tools._DISPATCH, (
                f"{dropped} should have been removed from dispatch"
            )


class TestDispatcherNestedArgUnwrap:
    """Llama 3.1 sometimes emits {function: X, args: {...}} instead of
    flat kwargs. The dispatcher should unwrap that before the signature
    filter, so real args reach the function."""

    def test_unwraps_function_plus_args_wrapper(self, monkeypatch):
        seen: dict = {}
        def fake(location: str | None = None) -> str:
            seen["location"] = location
            return "ok"
        monkeypatch.setitem(tools._DISPATCH, "__test_tool", fake)
        # The wrapper shape the dispatcher is supposed to peel
        wrapped = {"function": "__test_tool", "args": {"location": "Tokyo"}}
        result = tools.execute("__test_tool", wrapped)
        assert result == "ok"
        assert seen["location"] == "Tokyo"

    def test_unwraps_wrapper_with_name_field(self, monkeypatch):
        seen: dict = {}
        def fake(query: str | None = None) -> str:
            seen["query"] = query
            return "ok"
        monkeypatch.setitem(tools._DISPATCH, "__test_tool", fake)
        # Some variants include a 'name' alongside 'function' + 'args'
        wrapped = {"function": "__test_tool", "name": "__test_tool",
                   "args": {"query": "Tokyo"}}
        result = tools.execute("__test_tool", wrapped)
        assert seen["query"] == "Tokyo"

    def test_flat_args_pass_through_untouched(self, monkeypatch):
        """Wrapper detection must not misfire on legitimate flat args."""
        seen: dict = {}
        def fake(location: str | None = None) -> str:
            seen["location"] = location
            return "ok"
        monkeypatch.setitem(tools._DISPATCH, "__test_tool", fake)
        tools.execute("__test_tool", {"location": "Tokyo"})
        assert seen["location"] == "Tokyo"

    def test_kwargs_fn_never_unwrapped(self, monkeypatch):
        """A tool that takes **kwargs accepts everything, so unwrapping
        would just hide a real key. Leave it alone."""
        seen: dict = {}
        def fake(**kwargs) -> str:
            seen.update(kwargs)
            return "ok"
        monkeypatch.setitem(tools._DISPATCH, "__test_tool", fake)
        tools.execute("__test_tool",
                      {"function": "x", "args": {"a": 1}, "extra_field": "y"})
        # All three outer keys survive (no unwrap on **kwargs fns)
        assert "extra_field" in seen
        assert "function" in seen
        assert "args" in seen

    def test_picks_inner_when_it_matches_signature_better(self, monkeypatch):
        """Real Llama 3.1 leak: {op: 'math', args: {op: 'evaluate', expression: '15*23'}}.
        The outer 'op' is a tool-name echo (not a valid math op value
        per se for our enum). The inner dict has 2 valid keys (op +
        expression) vs outer's 1 (op alone — 'args' is unknown). Unwrap
        picks the inner."""
        seen: dict = {}
        def fake(op: str | None = None,
                 expression: str | None = None,
                 variable: str | None = None) -> str:
            seen["op"] = op
            seen["expression"] = expression
            return "ok"
        monkeypatch.setitem(tools._DISPATCH, "__test_tool", fake)
        # The outer + inner shape we saw live
        tools.execute("__test_tool",
                      {"op": "math",
                       "args": {"op": "evaluate", "expression": "15*23"}})
        # Inner wins — more signature-valid keys
        assert seen["op"] == "evaluate"
        assert seen["expression"] == "15*23"

    def test_prefers_outer_when_inner_has_fewer_matches(self, monkeypatch):
        """If the inner dict is LESS useful (fewer valid keys) than the
        outer, stick with outer. Ties break outer too."""
        seen: dict = {}
        def fake(location: str | None = None, query: str | None = None) -> str:
            seen["location"] = location
            seen["query"] = query
            return "ok"
        monkeypatch.setitem(tools._DISPATCH, "__test_tool", fake)
        # Outer has 2 valid keys (location, query). Inner has 0.
        tools.execute("__test_tool",
                      {"location": "NYC", "query": "pizza",
                       "args": {"nonsense": "x"}})
        assert seen["location"] == "NYC"
        assert seen["query"] == "pizza"


class TestDispatcherFriendlyErrors:
    """Exceptions from tools get wrapped into short, LLM-friendly strings
    so the model can paraphrase gracefully instead of parroting tracebacks."""

    def test_timeout_becomes_unavailable_msg(self, monkeypatch):
        def fake() -> str:
            raise TimeoutError("httpx: request timed out after 30s")
        monkeypatch.setitem(tools._DISPATCH, "__flaky", fake)
        result = tools.execute("__flaky", {})
        assert "temporarily unavailable" in result
        assert "try again" in result
        # Critically: no Python-internal jargon leaks
        assert "TimeoutError" not in result
        assert "httpx" not in result

    def test_rate_limit_becomes_clear_msg(self, monkeypatch):
        def fake() -> str:
            raise RuntimeError("429 Too Many Requests: rate limit exceeded")
        monkeypatch.setitem(tools._DISPATCH, "__flaky", fake)
        result = tools.execute("__flaky", {})
        assert "rate-limited" in result.lower() or "rate limit" in result.lower()

    def test_generic_exception_truncates_and_wraps(self, monkeypatch):
        def fake() -> str:
            raise ValueError("something went wrong" + "x" * 500)
        monkeypatch.setitem(tools._DISPATCH, "__flaky", fake)
        result = tools.execute("__flaky", {})
        # Wrapped in brackets, short, no ValueError jargon, no 500-char spam
        assert result.startswith("[__flaky couldn't complete:")
        assert result.endswith("]")
        assert len(result) < 250

    def test_unknown_tool_name_is_friendly(self):
        result = tools.execute("__does_not_exist", {})
        assert "[unknown tool" in result
        assert "answer without it" in result

    def test_malformed_json_string_args(self):
        result = tools.execute("get_weather", "{not valid json")
        assert "malformed arguments" in result
        assert result.startswith("[")


class TestAnalyzeTool:
    """Statistical operations over cached economic series. Pure-numpy
    so these tests run without scipy/statsmodels and exercise real
    cached data (CPI, wages, items) — no mocks."""

    def _call(self, **kwargs) -> dict:
        from bridge.tools._analyze import _analyze
        return json.loads(_analyze(**kwargs))

    def test_peak_inflation_year_in_us_post_war(self):
        """1980 is the headline-CPI peak after WWII — Volcker era."""
        out = self._call(series="cpi", op="peak",
                         year_from=1947, year_to=2024)
        assert "error" not in out
        # The 1980 peak inflation reading should still be visible in
        # the level series (CPI keeps rising; "peak" of LEVELS is
        # the latest year). Use trend instead for inflation rates.
        # For the LEVEL test, the peak is always the latest year:
        assert out["op"] == "peak"
        assert out["year"] == 2024
        assert out["value"] > 200  # CPI 1982-84 = 100; 2024 ≈ 313

    def test_trend_us_wages_post_1980(self):
        out = self._call(series="wages", op="trend",
                         year_from=1980, year_to=2024)
        assert "error" not in out
        assert out["op"] == "trend"
        assert out["slope_per_year"] > 0  # wages climb in nominal $
        assert out["r_squared"] > 0.9     # wages rise smoothly nominally

    def test_volatility_returns_yoy_stats(self):
        out = self._call(series="cpi", op="volatility",
                         year_from=1970, year_to=2024)
        assert "error" not in out
        assert out["op"] == "volatility"
        assert out["yoy_mean_pct"] > 0
        assert out["yoy_std_pct"] > 0

    def test_percentile_rank_recent_year_high(self):
        """2022 had unusually high inflation — should rank in upper
        percentile of YoY CPI levels post-1980."""
        out = self._call(series="cpi", op="percentile_rank", year=2022,
                         year_from=1980, year_to=2024)
        assert "error" not in out
        assert out["op"] == "percentile_rank"
        # 2022 CPI level is near the top of the post-1980 distribution
        # (just below 2023 + 2024)
        assert out["percentile"] >= 90

    def test_zscore_returns_signed_distance(self):
        out = self._call(series="wages", op="zscore", year=1985,
                         year_from=1964, year_to=2024)
        assert "error" not in out
        assert "zscore" in out
        # 1985 wage is below the long-run mean → negative z
        assert out["zscore"] < 0

    def test_deflate_real_wages_smaller_recent_growth(self):
        """Deflated wages should still rise but much less than nominal —
        real-wage stagnation is the canonical economics finding."""
        out = self._call(series="wages", op="deflate", base_year=2020)
        assert "error" not in out
        assert out["op"] == "deflate"
        assert out["base_year"] == 2020
        # Tail values are real wages in 2020 dollars — should be in a
        # plausible band for production-worker hourly real wage
        latest_real = list(out["tail"].values())[-1]
        assert 20 < latest_real < 40  # rough band

    def test_correlate_wages_vs_cpi_strong_positive(self):
        """Nominal wages and CPI both grow over time → strong positive r."""
        out = self._call(series="wages", op="correlate", series_b="cpi",
                         year_from=1964, year_to=2024)
        assert "error" not in out
        assert out["op"] == "correlate"
        assert out["pearson_r"] > 0.95  # nominally tracked tight

    def test_correlate_with_lag_still_works(self):
        out = self._call(series="wages", op="correlate", series_b="cpi",
                         lag=2, year_from=1970, year_to=2024)
        assert "error" not in out
        assert out["lag_years"] == 2

    def test_unknown_op_returns_error(self):
        out = self._call(series="cpi", op="bogus")
        assert "error" in out
        assert "valid" in out["error"]

    def test_unknown_series_returns_error(self):
        out = self._call(series="zzz", op="peak")
        assert "error" in out

    def test_percentile_rank_requires_year(self):
        out = self._call(series="cpi", op="percentile_rank")
        assert "error" in out
        assert "year" in out["error"]

    def test_correlate_requires_series_b(self):
        out = self._call(series="cpi", op="correlate")
        assert "error" in out
        assert "series_b" in out["error"]

    def test_item_series_works(self):
        """item:gasoline is a real BLS series; peak should be a real year."""
        out = self._call(series="item:gasoline", op="peak")
        assert "error" not in out
        assert "year" in out
        assert out["value"] > 0  # gas is positive $/gal

    def test_extract_analyze_args_pulls_op_and_year(self):
        from bridge.tools._analyze import extract_analyze_args
        out = extract_analyze_args("how unusual was inflation in 2022?")
        assert out.get("op") == "percentile_rank"
        assert out.get("year") == 2022


class TestAliceTool:
    """ALICE estimator. Cross-validates against United for ALICE
    published thresholds — exact match isn't expected but the deltas
    should stay bounded."""

    def _call(self, **kwargs) -> dict:
        from bridge.tools._alice import _alice
        return json.loads(_alice(**kwargs))

    def test_default_returns_latest_year(self):
        out = self._call()
        assert "error" not in out
        assert out["year"] in (2022, 2023, 2024)
        # Default composition is the canonical 4-person family.
        assert out["composition"] == "2A2K"
        assert out["household_size"] == 4

    def test_2022_threshold_within_25pct_of_published(self):
        """Our methodology + United for ALICE's headline number for 2022
        should match within 25%. Larger gap means a baseline constant
        drifted or the methodology diverged."""
        out = self._call(year=2022, composition="2A2K")
        assert "error" not in out
        ours = out["alice_threshold_4person_us"]
        theirs = out["reference_united_for_alice"]["alice_threshold_4person_us"]
        delta_pct = abs(ours - theirs) / theirs
        assert delta_pct < 0.25, (
            f"4-person threshold drift too large: ours=${ours:,.0f}, "
            f"theirs=${theirs:,.0f}, delta={delta_pct*100:.1f}%"
        )

    def test_pct_alice_within_10pp_of_published(self):
        """Estimated % ALICE should be within 10 percentage points of
        the published figure — wider gap suggests the bracket
        interpolation or composition weighting drifted."""
        out = self._call(year=2022, household_size=4)
        ours = out["population_shares"]["pct_alice"]
        theirs = out["reference_united_for_alice"]["pct_alice"]
        assert abs(ours - theirs) < 0.10, (
            f"% ALICE drift: ours={ours*100:.1f}%, theirs={theirs*100:.1f}%"
        )

    def test_budget_components_all_positive(self):
        out = self._call(year=2024, composition="2A2K")
        sb = out["survival_budget"]
        for line in ("housing", "food", "healthcare", "childcare",
                     "transport", "technology"):
            assert sb[line]["annual"] > 0, f"{line} should be positive"
            assert sb[line].get("explanation"), \
                f"{line} missing explanation"
        assert sb["total_for_size"] > sb["subtotal_pretax"]  # taxes add

    def test_childless_composition_has_no_childcare_line(self):
        """Bug fix from v1: a 1-person budget shouldn't include $14K
        of childcare (which the v1 scaled-from-4p model implicitly did)."""
        out = self._call(year=2024, composition="1A0K")
        sb = out["survival_budget"]
        assert sb["childcare"]["annual"] == 0
        assert sb["taxes"]["num_kids"] == 0

    def test_single_parent_uses_HoH_filing_status(self):
        out = self._call(year=2024, composition="1A1K")
        sb = out["survival_budget"]
        assert sb["taxes"]["filing_status"] == "HoH"
        assert sb["taxes"]["num_kids"] == 1
        assert sb["taxes"]["ctc"] == 2000  # 1 kid × $2,000 CTC

    def test_couple_uses_MFJ_filing_status(self):
        out = self._call(year=2024, composition="2A0K")
        sb = out["survival_budget"]
        assert sb["taxes"]["filing_status"] == "MFJ"
        assert sb["taxes"]["num_kids"] == 0
        assert sb["taxes"]["ctc"] == 0  # no kids

    def test_single_adult_higher_effective_tax_than_family(self):
        """A 1-person childless adult pays a higher effective tax rate
        than a 4-person family with 2 kids — no CTC, no EITC, half-size
        std deduction. This is the bug the v1 flat 10.5% rate masked."""
        single = self._call(year=2024, composition="1A0K")
        family = self._call(year=2024, composition="2A2K")
        single_rate = single["survival_budget"]["taxes"]["effective_rate_pct"]
        family_rate = family["survival_budget"]["taxes"]["effective_rate_pct"]
        assert single_rate > family_rate, (
            f"Single rate {single_rate}% should exceed family rate "
            f"{family_rate}% (no CTC, no EITC for the single)"
        )

    def test_compositions_increase_with_size(self):
        """1-adult < couple < couple+kids — survival cost should rise."""
        ordered = []
        for c in ["1A0K", "2A0K", "1A1K", "2A2K", "2A3K"]:
            out = self._call(year=2024, composition=c)
            ordered.append((c, out["survival_budget"]["total_for_size"]))
        # 1A0K (single) < 2A0K (couple)
        assert ordered[0][1] < ordered[1][1]
        # 2A2K (canonical) < 2A3K (3 kids — more food, bigger housing)
        assert ordered[3][1] < ordered[4][1]

    def test_interpretation_reflects_selected_composition(self):
        """When the user picks a non-canonical composition, the
        interpretation should name it explicitly, not say '4-person'."""
        out_4p = self._call(year=2022, composition="2A2K")
        out_1p = self._call(year=2022, composition="1A0K")
        assert "single adult" in out_4p["interpretation"].lower() \
            or "4-person" in out_4p["interpretation"].lower() \
            or "couple + 2" in out_4p["interpretation"].lower()
        assert "single adult" in out_1p["interpretation"].lower()

    def test_household_size_backward_compat(self):
        """household_size=N still works, mapped via _SIZE_TO_COMPOSITION."""
        out = self._call(year=2024, household_size=1)
        assert out["composition"] == "1A0K"
        out = self._call(year=2024, household_size=4)
        assert out["composition"] == "2A2K"

    def test_canonical_4p_budget_provided_when_composition_isnt_2A2K(self):
        """When composition != 2A2K, the response also returns a
        canonical_4person_budget so cross-validation against UFA's
        published 4-person threshold still works."""
        out = self._call(year=2022, composition="1A0K")
        assert out["canonical_4person_budget"] is not None
        out2 = self._call(year=2022, composition="2A2K")
        # When canonical IS the picked composition, no need to duplicate
        assert out2["canonical_4person_budget"] is None

    def test_available_years_lists_baseline_coverage(self):
        out = self._call()
        assert "available_years" in out
        assert isinstance(out["available_years"], list)
        # We seeded 2018, 2020, 2022, 2023, 2024 in the baseline.
        assert 2018 in out["available_years"]
        assert 2024 in out["available_years"]

    def test_extract_alice_args_pulls_year(self):
        from bridge.tools._alice import extract_alice_args
        out = extract_alice_args("what was the ALICE rate in 2020?")
        assert out.get("year") == 2020

    def test_extract_alice_args_pulls_household_size_numeric(self):
        from bridge.tools._alice import extract_alice_args
        out = extract_alice_args("ALICE for a 3-person family")
        assert out.get("household_size") == 3

    def test_extract_alice_args_pulls_household_size_word(self):
        from bridge.tools._alice import extract_alice_args
        out = extract_alice_args("survival budget for a two-person household")
        assert out.get("household_size") == 2

    def test_extract_alice_args_pulls_family_of_phrasing(self):
        from bridge.tools._alice import extract_alice_args
        out = extract_alice_args("ALICE threshold for a family of 5 in 2022")
        assert out.get("household_size") == 5
        assert out.get("year") == 2022

    def test_extract_alice_args_returns_empty_on_no_match(self):
        from bridge.tools._alice import extract_alice_args
        out = extract_alice_args("hello world")
        assert out == {}

    def test_force_rescue_declines_unrelated_text(self):
        """Force-rescue should refuse to invoke ALICE on text that
        doesn't reference ALICE/poverty/survival-budget — otherwise
        random LoRA under-fires would land on the wrong tool."""
        from bridge.routing.force_fire import default_args
        result = default_args("alice", "tell me a joke")
        assert result is None

    def test_force_rescue_accepts_alice_keyword(self):
        """A bare 'ALICE' keyword without year/size still rescues —
        the tool defaults handle the rest."""
        from bridge.routing.force_fire import default_args
        result = default_args("alice", "what's the ALICE estimate?")
        assert result is not None  # may be {} but still valid

    def test_dispatch_executes_alice(self):
        """End-to-end: tools.execute('alice', ...) returns a JSON
        string with the survival_budget key. Mirrors how the bridge
        invokes it after the LLM emits a tool_call."""
        from bridge import tools
        result = tools.execute("alice", {"year": 2022, "household_size": 4})
        data = json.loads(result)
        assert "survival_budget" in data
        assert "interpretation" in data
        assert data["year"] == 2022

    def test_unknown_year_falls_back(self):
        """Asking for a year before the baseline window picks the latest
        available; asking for a year after picks the latest too. Both
        modes return a valid response, not an error."""
        out_old = self._call(year=2010, household_size=4)
        assert "error" not in out_old
        # Year is the closest available year ≤ 2010, but our baseline
        # starts at 2022 — so it should clip to the latest year available.
        assert out_old["year"] >= 2022

    def test_invalid_household_size_clamps_to_largest_composition(self):
        """The composition resolver caps size at 7 → 2A3K. No error,
        just clamping (a household_size of 99 is silly but not worth
        an error response when the user clearly wants 'big family')."""
        out = self._call(year=2024, household_size=99)
        assert "error" not in out
        assert out["composition"] == "2A3K"

    def test_invalid_composition_string_returns_error(self):
        """An unknown composition key should error explicitly so the
        caller knows their string didn't match the registry."""
        out = self._call(year=2024, composition="5A0K")
        assert "error" in out
        assert "Unknown composition" in out["error"]

    def test_data_cross_check_present(self):
        """BLS food basket + EIA gas cross-checks should be populated
        when those datasets are cached locally — they are in the test
        env."""
        out = self._call(year=2022)
        cc = out["data_cross_check"]
        assert cc is not None
        assert cc["bls_food_basket"] is not None
        assert cc["bls_food_basket"]["subtotal_annual"] > 0
        assert cc["eia_gasoline"] is not None
        assert cc["eia_gasoline"]["annual_gas_cost"] > 0


class TestCountyMetricsTool:
    """County-overlay Phase A scaffolding. Tests use synthetic JSON
    files written into tmp_path so the pipeline is verified without
    needing a real Census API key."""

    def _seed(self, monkeypatch, tmp_path):
        from bridge.tools import _county_metrics as cm
        county_dir = tmp_path / "county"
        county_dir.mkdir()
        monkeypatch.setattr(cm, "_DATA_DIR", county_dir)
        gini_data = {
            "_metadata": {"metric": "gini", "label": "Gini index",
                          "unit": "ratio", "source": "test", "vintage": "test"},
            "by_county": {
                "06037": {"2018": 0.50, "2022": 0.52},
                "36061": {"2018": 0.55, "2022": 0.57},
                "48201": {"2018": 0.48, "2022": 0.49},
            },
        }
        (county_dir / "gini.json").write_text(json.dumps(gini_data))
        rent_data = {
            "_metadata": {"metric": "rent", "label": "Median gross rent",
                          "unit": "USD/month", "source": "test", "vintage": "test"},
            "by_county": {
                "06037": {"2018": 1500, "2022": 1850},
                "36061": {"2018": 2000, "2022": 2400},
                "48201": {"2018": 1100, "2022": 1300},
            },
        }
        (county_dir / "rent.json").write_text(json.dumps(rent_data))
        names = {
            "_metadata": {"vintage": "test"},
            "by_county": {
                "06037": "Los Angeles County, CA",
                "36061": "New York County, NY",
                "48201": "Harris County, TX",
            },
        }
        (county_dir / "county_names.json").write_text(json.dumps(names))
        return cm

    def test_list_metrics_distinguishes_loaded_from_missing(self, monkeypatch, tmp_path):
        cm = self._seed(monkeypatch, tmp_path)
        out = cm.list_available_metrics()
        keys_avail = [m["key"] for m in out["available"]]
        keys_missing = [m["key"] for m in out["missing"]]
        assert "gini" in keys_avail
        assert "rent" in keys_avail
        assert "mortality_all_cause" in keys_missing
        assert "violent_crime_rate" in keys_missing

    def test_metric_returns_values_by_county(self, monkeypatch, tmp_path):
        cm = self._seed(monkeypatch, tmp_path)
        out = cm.county_metric("rent", year=2022)
        assert out["data_status"] == "loaded"
        assert out["year"] == 2022
        assert out["values_by_county"]["06037"] == 1850
        assert sorted(out["available_years"]) == [2018, 2022]
        stats = out["stats"]
        assert stats["count"] == 3
        assert stats["min"] == 1300
        assert stats["max"] == 2400

    def test_metric_year_falls_back_when_unavailable(self, monkeypatch, tmp_path):
        cm = self._seed(monkeypatch, tmp_path)
        out = cm.county_metric("rent", year=2099)
        assert out["year"] == 2022

    def test_unfetched_metric_returns_clear_status(self, monkeypatch, tmp_path):
        cm = self._seed(monkeypatch, tmp_path)
        out = cm.county_metric("violent_crime_rate")
        assert out["data_status"] == "not_fetched_yet"
        assert out["values_by_county"] == {}
        assert "fetched yet" in out["interpretation"]

    def test_unknown_metric_errors(self, monkeypatch, tmp_path):
        cm = self._seed(monkeypatch, tmp_path)
        out = cm.county_metric("zzz")
        assert "error" in out

    def test_compare_returns_pearson_r(self, monkeypatch, tmp_path):
        cm = self._seed(monkeypatch, tmp_path)
        out = cm.county_compare("gini", "rent", year=2022)
        assert out["data_status"] == "loaded"
        assert out["n"] == 3
        # Synthetic data: higher Gini → higher rent at 2022.
        assert out["pearson_r"] > 0.9

    def test_compare_flags_missing_data(self, monkeypatch, tmp_path):
        cm = self._seed(monkeypatch, tmp_path)
        out = cm.county_compare("gini", "violent_crime_rate")
        assert out["data_status"] == "incomplete"
        assert "violent_crime_rate" in out["missing"]

    def test_drill_returns_all_metrics_for_a_county(self, monkeypatch, tmp_path):
        cm = self._seed(monkeypatch, tmp_path)
        out = cm.county_drill("06037")
        assert out["data_status"] == "loaded"
        assert out["name"] == "Los Angeles County, CA"
        assert "gini" in out["metrics"]
        assert "rent" in out["metrics"]
        assert out["metrics"]["rent"]["latest_value"] == 1850
        assert 30 <= out["metrics"]["rent"]["percentile_among_counties"] <= 50

    def test_drill_pads_short_fips(self, monkeypatch, tmp_path):
        cm = self._seed(monkeypatch, tmp_path)
        out = cm.county_drill("6037")
        assert out["fips"] == "06037"
        assert out["data_status"] == "loaded"

    def test_drill_invalid_fips_errors(self, monkeypatch, tmp_path):
        cm = self._seed(monkeypatch, tmp_path)
        out = cm.county_drill("not-a-fips")
        assert "error" in out
