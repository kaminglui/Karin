/* holidays.js — dismissible banner on US / China holidays.
 *
 * On page load, calls /api/holiday/today. If the backend returns a
 * holiday, a banner fades in above the transcript with the holiday's
 * emoji, name, and greeting. A 🔊 button fetches a Karin-voiced
 * rendition via /api/holiday/today/audio (lazily synthesized by
 * sovits, cached on disk for the day, auto-purged when the holiday
 * passes). Dismissal is stored in localStorage with today's date,
 * so clicking × suppresses it for the rest of the day.
 *
 * The heavy lifting (nager.date fetch, per-year cache, supplemental
 * cultural days like Qixi/Lantern, audio cleanup) lives server-side
 * in bridge/holidays.py. This file is just display + playback.
 */
(function () {
    "use strict";

    const BANNER_ID = "holiday-banner";

    // ---- dismiss / replay state -----------------------------------------

    function _dismissKey(date) { return `karin.holiday.dismissed.${date}`; }

    function isDismissed(date) {
        try { return !!localStorage.getItem(_dismissKey(date)); }
        catch { return false; }
    }

    function markDismissed(date) {
        try { localStorage.setItem(_dismissKey(date), "1"); } catch { /* ignore */ }
    }

    // ---- banner render ---------------------------------------------------

    // Normalize a holiday name into a CSS-friendly slug so stylesheets
    // can target specific festivals (e.g. body[data-holiday="chinese-new-year"]).
    function holidaySlug(name) {
        return (name || "")
            .toLowerCase()
            .replace(/['’`]/g, "")
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/^-+|-+$/g, "");
    }

    function applyTheme(holiday) {
        // Body data-attributes drive the "something different" theming
        // in style.css (accent color tint, giant corner glyph, etc.).
        // Specific-holiday rules stack on top of country-level rules,
        // so Chinese New Year gets both the generic CN red tint AND
        // its own lantern-gold accents.
        const body = document.body;
        body.dataset.holiday = holidaySlug(holiday.name);
        body.dataset.holidayCountry = holiday.country || "";
        if (holiday.emoji) {
            body.style.setProperty("--holiday-emoji", `"${holiday.emoji}"`);
        }
    }

    function clearTheme() {
        const body = document.body;
        delete body.dataset.holiday;
        delete body.dataset.holidayCountry;
        body.style.removeProperty("--holiday-emoji");
    }

    function mountBanner(holiday) {
        const banner = document.getElementById(BANNER_ID);
        if (!banner) return;
        banner.querySelector(".holiday-emoji").textContent = holiday.emoji || "🎉";
        applyTheme(holiday);

        const text = banner.querySelector(".holiday-text");
        text.innerHTML = "";
        const name = document.createElement("span");
        name.className = "holiday-name";
        name.textContent = holiday.name;
        const greet = document.createElement("span");
        greet.className = "holiday-greeting";
        greet.textContent = " — " + (holiday.greeting || "");
        text.appendChild(name);
        text.appendChild(greet);

        // 🔊 play button — only wire up if one exists in the markup.
        // Hide entirely when TTS is disabled on the server, since the
        // synth endpoint would just 503.
        const play = banner.querySelector(".holiday-play");
        if (play) {
            if (window.KARIN_CAPS && window.KARIN_CAPS.tts === false) {
                play.hidden = true;
            } else {
                play.onclick = () => playGreeting(play);
            }
        }

        const close = banner.querySelector(".holiday-close");
        close.onclick = () => {
            markDismissed(holiday.date);
            banner.hidden = true;
            // Banner dismissal also clears the main-screen theme so
            // the user isn't stuck staring at Chinese New Year red
            // after they've waved it off. They can still re-enable
            // via a hard refresh until end-of-day.
            clearTheme();
        };

        banner.hidden = false;
    }

    // ---- voiced greeting playback ---------------------------------------

    let _greetingObjectURL = null;

    async function playGreeting(btn) {
        // Simple UX: disable while fetching/playing; re-enable on
        // completion or error. Caches the object URL so repeat plays
        // don't re-download.
        btn.disabled = true;
        const prevText = btn.textContent;
        btn.textContent = "…";
        try {
            if (!_greetingObjectURL) {
                const r = await fetch("/api/holiday/today/audio");
                if (r.status === 503) {
                    btn.textContent = "voice off";
                    btn.title = "Turn Voice on in the sidebar to hear Karin's greeting.";
                    setTimeout(() => { btn.textContent = prevText; btn.disabled = false; }, 2500);
                    return;
                }
                if (!r.ok || r.status === 204) {
                    btn.textContent = prevText;
                    btn.disabled = false;
                    return;
                }
                const blob = await r.blob();
                _greetingObjectURL = URL.createObjectURL(blob);
            }
            const audio = new Audio(_greetingObjectURL);
            audio.onended = () => { btn.textContent = prevText; btn.disabled = false; };
            audio.onerror = () => { btn.textContent = prevText; btn.disabled = false; };
            await audio.play();
        } catch (e) {
            console.warn("holiday greeting playback failed:", e);
            btn.textContent = prevText;
            btn.disabled = false;
        }
    }

    // ---- init ------------------------------------------------------------

    async function init() {
        const banner = document.getElementById(BANNER_ID);
        if (!banner) return;

        // Test override: ?testHoliday=YYYY-MM-DD bypasses the server
        // and renders a fixed demo banner so you can preview styling.
        // Optional extras for theme preview:
        //   ?testHolidayName=Chinese+New+Year
        //   ?testHolidayCountry=CN
        //   ?testHolidayEmoji=%F0%9F%A7%A7
        const params = new URLSearchParams(window.location.search);
        const test = params.get("testHoliday");
        if (test && /^\d{4}-\d{2}-\d{2}$/.test(test)) {
            mountBanner({
                date: test,
                name: params.get("testHolidayName") || "Preview holiday",
                emoji: params.get("testHolidayEmoji") || "🎉",
                greeting: "Styling preview — dismiss to close.",
                country: params.get("testHolidayCountry") || "US",
            });
            return;
        }

        let holiday = null;
        try {
            const r = await fetch("/api/holiday/today");
            if (r.ok) {
                const data = await r.json();
                holiday = data && data.holiday;
            }
        } catch (e) {
            console.warn("holiday fetch failed:", e);
            return;
        }
        if (!holiday) return;
        if (isDismissed(holiday.date)) return;
        mountBanner(holiday);
    }

    window.Holidays = { init, playGreeting };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
