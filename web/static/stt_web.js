/* stt_web.js — browser Web Speech API wrapper.
 *
 * Chrome / Edge / Safari expose a native SpeechRecognition object that
 * streams audio to the browser vendor's cloud STT (Google on Chrome,
 * Apple on Safari) and returns transcripts as events. No API key,
 * no paid tier, no server roundtrip.
 *
 * This wraps the verbose spec into a small event-callback class the
 * rest of app.js can consume without thinking about vendor prefixes,
 * interim-vs-final gymnastics, or `onend` vs `onerror` races.
 *
 * Availability test: call ``WebSpeechRecognizer.isAvailable()``.
 * Returns true only when the API exists AND ``navigator.onLine`` is
 * truthy. Firefox doesn't implement the API and will always report
 * false — callers fall back to local faster-whisper in that case.
 *
 * Privacy note: when this is the active STT path, your voice goes
 * to Google / Apple over TLS. TLS encrypts in transit; the endpoint
 * decrypts to recognize. Only enable for sessions where that's OK.
 */
(function () {
    "use strict";

    class WebSpeechRecognizer {
        constructor({ lang, continuous, interimResults } = {}) {
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.SR = SR;
            this.lang = lang || (navigator.language || "en-US");
            this.continuous = continuous !== undefined ? continuous : false;
            this.interimResults = interimResults !== undefined ? interimResults : true;
            this._r = null;
            this._t0 = 0;
            this._gotInterim = false;
            // Consumer-supplied callbacks.
            this.onInterim = null;   // (text: string, dtMs: number)
            this.onFinal   = null;   // (text: string, dtMs: number)
            this.onError   = null;   // (code: string)
            this.onEnd     = null;   // ()
        }

        static isAvailable() {
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            return !!SR && !!navigator.onLine;
        }

        /** Start listening. Returns true if the API was actually engaged. */
        start() {
            if (!this.SR) return false;
            const r = new this.SR();
            r.continuous = this.continuous;
            r.interimResults = this.interimResults;
            r.lang = this.lang;
            this._t0 = performance.now();
            this._gotInterim = false;

            r.onresult = (e) => {
                let interim = "";
                let final = "";
                for (let i = e.resultIndex; i < e.results.length; i++) {
                    const alt = e.results[i][0];
                    if (!alt) continue;
                    if (e.results[i].isFinal) final += alt.transcript;
                    else                      interim += alt.transcript;
                }
                const dt = performance.now() - this._t0;
                if (interim && this.onInterim) {
                    if (!this._gotInterim) {
                        this._gotInterim = true;
                        console.log(`[web-speech] first-interim ${dt.toFixed(0)} ms`);
                    }
                    this.onInterim(interim, dt);
                }
                if (final && this.onFinal) {
                    console.log(`[web-speech] final ${dt.toFixed(0)} ms: ${final}`);
                    this.onFinal(final.trim(), dt);
                }
            };

            r.onerror = (e) => {
                // `no-speech` / `aborted` are routine termination
                // events, not real errors. Caller differentiates.
                console.warn(`[web-speech] error: ${e.error}`);
                if (this.onError) this.onError(e.error || "unknown");
            };

            r.onend = () => {
                if (this.onEnd) this.onEnd();
            };

            try {
                r.start();
            } catch (e) {
                console.warn("[web-speech] start() threw:", e);
                return false;
            }
            this._r = r;
            return true;
        }

        stop() {
            if (this._r) {
                try { this._r.stop(); } catch { /* ignore */ }
            }
        }

        abort() {
            if (this._r) {
                try { this._r.abort(); } catch { /* ignore */ }
            }
        }

        get active() { return this._r !== null; }
    }

    if (typeof window !== "undefined") {
        window.WebSpeechRecognizer = WebSpeechRecognizer;
    }
})();
