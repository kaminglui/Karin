/* weather.js — inline weather widget.
 *
 * Fetches /api/weather (same resolution logic as the get_weather tool)
 * and renders: location title, big temperature + emoji, a row of chip
 * stats (feels-like / wind / humidity / precip).
 *
 * Mount from chat.js when the LLM calls get_weather.
 */
(function () {
  "use strict";

  const DEFAULT_API_BASE = "";

  function mountWeatherPanel(container, options = {}) {
    const opts = {
      location: null,
      apiBase: DEFAULT_API_BASE,
      onSubtitle: null,
      ...options,
    };

    let aborted = false;

    async function load() {
      container.innerHTML = Panels.stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");

      let data;
      try {
        const q = opts.location ? `?location=${encodeURIComponent(opts.location)}` : "";
        const res = await Panels.safeFetch(`${opts.apiBase}/api/weather${q}`);
        data = await res.json();
      } catch (e) {
        if (aborted) return;
        container.innerHTML = Panels.stateCard({
          kind: "error",
          message: `Failed to load weather: ${e.message}`,
        });
        if (opts.onSubtitle) opts.onSubtitle("error");
        return;
      }
      if (aborted) return;

      if (data.error) {
        container.innerHTML = Panels.stateCard({
          kind: "empty",
          label: "weather",
          message: data.error,
        });
        if (opts.onSubtitle) opts.onSubtitle("unavailable");
        return;
      }

      render(data);
    }

    function render(d) {
      const tempC = d.temp_c != null ? `${Math.round(d.temp_c)}°C` : "—";
      const feelsC = d.feels_like_c != null ? `${Math.round(d.feels_like_c)}°C` : "—";
      const windKmh = d.wind_kmh != null ? `${Math.round(d.wind_kmh)} km/h` : "—";
      const humidity = d.humidity != null ? `${Math.round(d.humidity)}%` : "—";
      const precip = d.precipitation_mm != null ? `${d.precipitation_mm} mm` : "—";

      container.innerHTML = `
        <div class="weather-card">
          <div class="weather-hero">
            <span class="weather-emoji" aria-hidden="true">${escapeHtml(d.emoji || "🌤")}</span>
            <div class="weather-hero-text">
              <div class="weather-temp">${escapeHtml(tempC)}</div>
              <div class="weather-condition">${escapeHtml(d.condition || "")}</div>
            </div>
          </div>
          <div class="weather-chips">
            <span class="weather-chip"><span class="chip-label">Feels</span>${escapeHtml(feelsC)}</span>
            <span class="weather-chip"><span class="chip-label">Wind</span>${escapeHtml(windKmh)}</span>
            <span class="weather-chip"><span class="chip-label">Humidity</span>${escapeHtml(humidity)}</span>
            <span class="weather-chip"><span class="chip-label">Precip</span>${escapeHtml(precip)}</span>
          </div>
        </div>
      `;

      if (opts.onSubtitle) opts.onSubtitle(d.place_name || "");
    }

    function escapeHtml(s) {
      return String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    load();

    return {
      unmount: () => {
        aborted = true;
        container.innerHTML = "";
      },
      reload: load,
    };
  }

  if (typeof window !== "undefined") {
    window.Weather = { mountWeatherPanel };
  }
})();
