/* face.js — vowel-driven expression animator for the top PTT button.
 *
 * Two rendering modes, picked at startup by the face picker in
 * index.html:
 *
 *   - BITMAP mode: the element is an <img>. Legacy PNG-per-vowel
 *     swap, one frame per vowel (a/e/i/o/u + default), loaded from
 *     /api/characters/<name>/expressions/. Used for characters that
 *     ship with expression art (karin and any other named skin).
 *
 *   - PROCEDURAL mode: the element is an inline <svg> sun. The
 *     mouth ellipse's rx / ry / cy are mutated directly so the
 *     shape morphs between "closed" / "aah" / "oo" / etc. Used for
 *     the "general" character — no bitmaps needed.
 *
 * Same public API in both modes so app.js doesn't need to branch:
 *
 *   - stepText(delta)    scans a streaming text delta and shows the
 *                        matching frame on each vowel char.
 *   - speak(text, ms)    used during TTS playback — walks the whole
 *                        reply's vowel sequence paced over `ms`.
 *   - buildSchedule / play  precompute once, play many.
 *   - idle()             snap back to default.
 */
(function () {
  "use strict";

  // Dynamic base for BITMAP mode; set by the face picker inline
  // script in index.html. Falls back to legacy /static/faces/ so a
  // hand-embedded <img> still works.
  let BASE = window.KARIN_FACE_BASE || "/static/faces/";
  const VOWELS = new Set(["a", "e", "i", "o", "u"]);
  // How long each vowel holds visually before we fall back to default
  // if no further text arrives. Short enough that the mouth feels
  // responsive; long enough that individual frames are readable.
  const HOLD_MS = 90;
  // Per-char step delay when animating a multi-char delta. Roughly
  // tracks natural typing speed (~16 chars/sec) so the face moves at
  // the same pace the text appears in the bubble.
  const STEP_MS = 55;

  // Mouth geometry per frame. Values are in the SVG viewBox space
  // (0–100). `cy` shifts slightly for round shapes so they sit
  // centered rather than dropping below the eye-line; `ry` near zero
  // is a near-flat closed line.
  //
  // Source of truth is `characters/<voice>/face.json` — the server
  // injects that into window.KARIN_FACE_CONFIG.mouth_shapes. The
  // fallback defaults below match the general character's values so
  // nothing breaks if the config is absent or malformed.
  const FALLBACK_MOUTH_SHAPES = {
    default: { rx: 12, ry: 1.5, cy: 65 },
    a:       { rx: 14, ry: 9,   cy: 65 },
    e:       { rx: 13, ry: 4,   cy: 65 },
    i:       { rx: 10, ry: 2,   cy: 65 },
    o:       { rx:  7, ry: 7,   cy: 64 },
    u:       { rx:  5, ry: 5,   cy: 64 },
  };
  const MOUTH_SHAPES = (function () {
    var cfg = (typeof window !== "undefined" && window.KARIN_FACE_CONFIG) || null;
    if (cfg && cfg.mouth_shapes && typeof cfg.mouth_shapes === "object") {
      // Shallow merge: config overrides fallback per-frame so a
      // partial config (e.g. tweaking just "a") still works.
      var out = {};
      Object.keys(FALLBACK_MOUTH_SHAPES).forEach(function (k) {
        out[k] = Object.assign({}, FALLBACK_MOUTH_SHAPES[k], cfg.mouth_shapes[k] || {});
      });
      return out;
    }
    return FALLBACK_MOUTH_SHAPES;
  })();

  function vowelOf(ch) {
    if (!ch) return null;
    const c = ch.toLowerCase();
    return VOWELS.has(c) ? c : null;
  }

  class Face {
    constructor(rootEl) {
      // rootEl is whatever the face picker assigned id="face-img" to.
      // Detect its mode by tag name — SVG → procedural mouth; IMG →
      // bitmap src swap. Any other element type makes _set a no-op.
      this.root = rootEl;
      const tag = (rootEl && rootEl.tagName || "").toLowerCase();
      this.mode = tag === "svg" ? "svg" : (tag === "img" ? "img" : "none");

      if (this.mode === "svg") {
        this.mouth = rootEl.querySelector
          ? rootEl.querySelector("#face-mouth")
          : null;
      } else if (this.mode === "img") {
        this.img = rootEl;
        // Preload all frames so the first swap doesn't flash.
        for (const k of ["a", "e", "i", "o", "u", "default"]) {
          const p = new Image();
          p.src = `${BASE}${k}.png`;
        }
      }

      this._idleTimer = null;
      this._speakAbort = null;  // cancels in-flight speak() if a new turn starts
      this._scheduledTimers = null;
      this._currentFrame = "default";
    }

    _set(key) {
      if (this.mode === "svg") {
        if (!this.mouth) return;
        const shape = MOUTH_SHAPES[key] || MOUTH_SHAPES.default;
        // setAttribute on rx/ry/cy is what CSS transitions animate —
        // direct property access works on evergreen browsers but is
        // inconsistent on older Safari, so we stick with the
        // attribute form the spec guarantees.
        this.mouth.setAttribute("rx", shape.rx);
        this.mouth.setAttribute("ry", shape.ry);
        this.mouth.setAttribute("cy", shape.cy);
      } else if (this.mode === "img") {
        if (!this.img) return;
        this.img.src = `${BASE}${key}.png`;
      }
      this._currentFrame = key;
    }

    /** Instant one-shot: show this vowel, then fall back after HOLD_MS. */
    step(char) {
      const v = vowelOf(char);
      if (!v) return;
      this._set(v);
      if (this._idleTimer) clearTimeout(this._idleTimer);
      this._idleTimer = setTimeout(() => this.idle(), HOLD_MS);
    }

    /**
     * Animate across a multi-char delta, one vowel at a time, paced
     * at ~STEP_MS. Non-vowels are still consumed (they just don't
     * reshape the mouth), which keeps timing aligned with the
     * underlying text rate.
     */
    stepText(delta) {
      if (!delta) return;
      // Cancel any in-flight speak() pacing — streaming tokens own
      // the face now.
      if (this._speakAbort) { this._speakAbort.aborted = true; this._speakAbort = null; }
      for (let i = 0; i < delta.length; i++) {
        const c = delta[i];
        const v = vowelOf(c);
        if (!v) continue;
        setTimeout(() => this._set(v), i * STEP_MS);
      }
      // After the whole delta finishes animating, schedule the
      // default fallback (replaces any earlier timer).
      if (this._idleTimer) clearTimeout(this._idleTimer);
      this._idleTimer = setTimeout(
        () => this.idle(),
        delta.length * STEP_MS + HOLD_MS,
      );
    }

    /**
     * Precompute a mouth-sync schedule from the reply text + the TTS
     * audio duration. Output is ``[{t_ms, frame}, ...]`` with absolute
     * offsets from playback start.
     *
     * Compute-once, play-many: callers stash this on the turn object
     * and hand it to ``play()`` on every replay click. Recomputing the
     * same plan on each playback was why the first vs. second replay
     * could drift differently relative to audio startup latency.
     *
     * The algorithm is still a uniform walk (every character gets
     * ``per = durationMs/len`` ms), the win here is *stability across
     * plays* — one plan binds the animation to the same absolute
     * timeline the audio will follow.
     */
    static buildSchedule(text, durationMs) {
      if (!text || !durationMs || durationMs <= 0) return [];
      const per = Math.max(18, durationMs / Math.max(text.length, 1));
      const schedule = [];
      let lastFrame = null;
      for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        const v = vowelOf(ch);
        let frame = null;
        if (v) {
          frame = v;
        } else if (/\s|[.,;:!?]/.test(ch)) {
          // Word-boundary / punctuation: briefly snap to default so
          // the mouth "rests" between words.
          frame = "default";
        }
        // Drop no-op events (consonants that'd keep the last frame)
        // AND dedupe adjacent same-frame events to keep the schedule
        // compact without affecting what the user sees.
        if (frame !== null && frame !== lastFrame) {
          schedule.push({ t_ms: i * per, frame });
          lastFrame = frame;
        }
      }
      // Always terminate on default so we don't freeze on the last
      // vowel when the audio ends.
      schedule.push({ t_ms: text.length * per, frame: "default" });
      return schedule;
    }

    /**
     * Play a precomputed schedule. Each frame is timed as
     * ``startTime + evt.t_ms`` via ``setTimeout`` against a baseline
     * captured at play() entry — so the mouth and audio share the
     * same wall clock from the moment both start.
     *
     * Returns a cancel token that can be flipped to abort.
     */
    play(schedule) {
      // Cancel any prior animation first so rapid replays don't stack.
      this._cancelAll();
      if (!schedule || schedule.length === 0) {
        this.idle();
        return { aborted: false, cancel: () => {} };
      }
      const token = { aborted: false };
      this._speakAbort = token;
      const startTime = performance.now();
      const timers = [];
      for (const evt of schedule) {
        const delay = Math.max(0, (startTime + evt.t_ms) - performance.now());
        const t = setTimeout(() => {
          if (token.aborted) return;
          this._set(evt.frame);
        }, delay);
        timers.push(t);
      }
      this._scheduledTimers = timers;
      token.cancel = () => { token.aborted = true; this._cancelAll(); this.idle(); };
      return token;
    }

    _cancelAll() {
      if (this._speakAbort) { this._speakAbort.aborted = true; this._speakAbort = null; }
      if (this._scheduledTimers) {
        for (const t of this._scheduledTimers) clearTimeout(t);
        this._scheduledTimers = null;
      }
      if (this._idleTimer) { clearTimeout(this._idleTimer); this._idleTimer = null; }
    }

    /**
     * Backward-compat wrapper. Builds + plays in one call; prefer
     * ``buildSchedule`` + ``play`` for precomputed caching.
     */
    speak(text, durationMs) {
      this.play(Face.buildSchedule(text, durationMs));
    }

    idle() {
      this._cancelAll();
      this._set("default");
    }
  }

  if (typeof window !== "undefined") {
    window.Face = Face;
  }
})();
