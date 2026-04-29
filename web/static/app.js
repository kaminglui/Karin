"use strict";

// ---- Global async-error surfacing ----------------------------------------
// Panels fire-and-forget a lot of fetches. If one forgets to .catch(),
// the error used to vanish silently; now it lands in the console with
// enough context to trace which promise chain rejected. Safe in prod —
// the handler only logs, never throws.
window.addEventListener("unhandledrejection", (event) => {
    const reason = event.reason;
    const msg = reason && reason.message ? reason.message : String(reason);
    console.warn("[unhandled promise rejection]", msg, reason);
});
window.addEventListener("error", (event) => {
    console.warn("[uncaught error]", event.message, event.error);
});

// ---- Mobile pinch-zoom + double-tap zoom block --------------------------
//
// iOS Safari ignores `user-scalable=no` in viewport meta since iOS 10,
// so we have to intercept `gesturestart` (the pinch trigger) and the
// double-tap-to-zoom pattern explicitly. Desktop browsers ignore both.
document.addEventListener("gesturestart", (e) => e.preventDefault(), { passive: false });
document.addEventListener("gesturechange", (e) => e.preventDefault(), { passive: false });
document.addEventListener("gestureend", (e) => e.preventDefault(), { passive: false });
let _lastTap = 0;
document.addEventListener("touchend", (e) => {
    const now = Date.now();
    if (now - _lastTap < 300 && e.touches.length === 0) {
        // Don't preventDefault on form inputs — that breaks tapping into them.
        const t = e.target;
        if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
        e.preventDefault();
    }
    _lastTap = now;
}, { passive: false });

const pttButton = document.getElementById("ptt");
const pttWrap = document.getElementById("ptt-wrap");
const waveRing = document.getElementById("wave-ring");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const chatArea = document.getElementById("chat-area");
const textInput = document.getElementById("text-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");

// Vowel-expression animator for the top PTT button. Driven from two
// event streams below: token_delta (streaming text while the LLM
// composes) and audio (TTS playback). `window.Face` is loaded by
// face.js before app.js so this binding is safe at module load.
const faceImgEl = document.getElementById("face-img");
const face = (window.Face && faceImgEl) ? new window.Face(faceImgEl) : null;

// Panel integration. Panels now mount INLINE per turn, at the bottom
// of the assistant's response, inside a collapsible <details>. Each
// turn's panel is independent so scrolling back through history
// preserves context. Controllers live on the turn object so they can
// be unmounted when the user starts a new chat.
const mountedPanelControllers = new Set();

let mediaRecorder = null;
let mediaStream = null;
let chunks = [];
let isProcessing = false;

// When a voice recording starts we create the turn immediately so the
// listening animation appears in the user bubble. submitVoice() reuses
// the same turn on release instead of creating a second one.
let activeVoiceTurn = null;

// ---- Audio Context (shared across recording + playback) -------------------

let audioCtx = null;
let currentAnalyser = null;
let animFrameId = null;
let smoothScale = 1;

// Browser-side VAD: track whether speech has been heard yet and how
// long since the last loud frame. Drives auto-stop after silence so
// the user doesn't have to manually stop in toggle mode. Tuned for
// short voice-assistant prompts; raise SILENCE_MS if it cuts users off
// mid-thought.
const VAD_RMS_SPEECH = 0.06;       // RMS threshold to count as "talking"
const VAD_RMS_SILENCE = 0.025;     // RMS threshold for "definitely quiet"
const VAD_SILENCE_MS = 1500;       // contiguous silence after speech → stop
const VAD_MAX_PRESPEECH_MS = 5000; // give up if no speech detected in this window
let vadSpeechHeard = false;
let vadLastSpeechAt = 0;
let vadStartedAt = 0;

// Why the recording ended, for the onstop handler. "no_speech" means
// VAD fired early-silence cutoff — we throw the blob away and skip
// the LLM call instead of sending empty audio. "silence_after_speech"
// and null (manual tap-to-stop) both go to the LLM normally.
let _vadStopReason = null;

function ensureAudioCtx() {
    if (!audioCtx || audioCtx.state === "closed") {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === "suspended") audioCtx.resume();
    return audioCtx;
}

// ---- Wave animation (volume-responsive, shared) ---------------------------

function animateWave() {
    if (!currentAnalyser) return;
    const data = new Uint8Array(currentAnalyser.frequencyBinCount);
    currentAnalyser.getByteFrequencyData(data);

    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    const rms = Math.sqrt(sum / data.length) / 255;

    const targetScale = 1 + rms * 3;
    smoothScale += (targetScale - smoothScale) * 0.14;

    waveRing.style.transform = `translate(-50%,-50%) scale(${smoothScale.toFixed(3)})`;
    waveRing.style.opacity = Math.max(0, 0.65 - (smoothScale - 1) * 0.15).toFixed(3);
    // Button breathes subtly with volume — dropped coupling to 0.04
    // after the previous 0.125 still felt too large on TTS peaks.
    // The portrait grows at most 12% at max volume (smoothScale ~4),
    // which reads as alive without looking like a zoom.
    pttButton.style.transform = `scale(${(1 + (smoothScale - 1) * 0.04).toFixed(3)})`;

    // No drop-shadow filter on the face image: user preferred the
    // clean silhouette with just the aura rings as the state signal.
    // isRec is still used below by the VAD auto-stop path, so keep
    // the cheap classList check — just don't tie visuals to it.
    const isRec = pttWrap.classList.contains("recording");

    // VAD path — only when actively recording. Two stages:
    //  1) wait until we hear ANY speech (above VAD_RMS_SPEECH);
    //  2) then auto-stop after VAD_SILENCE_MS of contiguous quiet.
    // If no speech at all within VAD_MAX_PRESPEECH_MS we also stop —
    // the user probably tapped by accident.
    if (isRec && mediaRecorder && mediaRecorder.state === "recording") {
        const now = performance.now();
        if (rms > VAD_RMS_SPEECH) {
            vadSpeechHeard = true;
            vadLastSpeechAt = now;
        }
        if (vadSpeechHeard) {
            if (rms < VAD_RMS_SILENCE && now - vadLastSpeechAt > VAD_SILENCE_MS) {
                _vadStopReason = "silence_after_speech";
                stopRecording();
            }
        } else if (now - vadStartedAt > VAD_MAX_PRESPEECH_MS) {
            // No speech ever arrived within the grace window — user
            // probably tapped by accident. Drop the recording without
            // wasting an LLM turn.
            _vadStopReason = "no_speech";
            stopRecording();
        }
    }

    animFrameId = requestAnimationFrame(animateWave);
}

function stopWaveAnim() {
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    currentAnalyser = null;
    smoothScale = 1;
    waveRing.style.transform = "translate(-50%,-50%) scale(1)";
    waveRing.style.opacity = "0";
    pttButton.style.transform = "";
    pttButton.style.boxShadow = "";
    // Drop the inline drop-shadow we set per-frame so the CSS-class
    // animations (idle-glow, processing) take back over cleanly.
    if (faceImgEl) faceImgEl.style.filter = "";
}

// Recording wave (mic)
function startRecordingWave(stream) {
    try {
        const ctx = ensureAudioCtx();
        const src = ctx.createMediaStreamSource(stream);
        currentAnalyser = ctx.createAnalyser();
        currentAnalyser.fftSize = 256;
        currentAnalyser.smoothingTimeConstant = 0.7;
        src.connect(currentAnalyser);
        smoothScale = 1;
        pttWrap.classList.remove("no-audio-api");
        animateWave();
    } catch (e) {
        console.warn("Web Audio mic failed:", e);
        pttWrap.classList.add("no-audio-api");
    }
}

// ---- Voice auto-play preference (localStorage) ---------------------------
//
// When off: TTS audio is still fetched + the replay button is built,
// but playback doesn't start automatically. User taps the replay ▶
// button to hear the voice. Default: on (matches pre-toggle behavior).

function isVoiceAutoPlay() {
    try { return localStorage.getItem("karin.voice.autoplay") !== "false"; }
    catch { return true; }
}
function setVoiceAutoPlay(on) {
    try { localStorage.setItem("karin.voice.autoplay", on ? "true" : "false"); }
    catch { /* private browsing */ }
}

// ---- Streaming TTS playback (Web Audio API) -------------------------------

let streamAnalyser = null;
let nextChunkTime = 0;
let streamChunks = [];   // collected for replay
let streamSr = 32000;

function initStreamPlayback() {
    const ctx = ensureAudioCtx();
    streamAnalyser = ctx.createAnalyser();
    streamAnalyser.fftSize = 256;
    streamAnalyser.smoothingTimeConstant = 0.7;
    streamAnalyser.connect(ctx.destination);
    nextChunkTime = ctx.currentTime + 0.08;
    streamChunks = [];

    currentAnalyser = streamAnalyser;
    pttWrap.classList.add("speaking");
    smoothScale = 1;
    animateWave();
}

function scheduleChunk(int16, sampleRate) {
    if (!streamAnalyser) return;
    const ctx = ensureAudioCtx();
    streamSr = sampleRate;
    streamChunks.push(int16);

    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

    const buf = ctx.createBuffer(1, float32.length, sampleRate);
    buf.getChannelData(0).set(float32);

    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(streamAnalyser);

    const t = Math.max(ctx.currentTime + 0.01, nextChunkTime);
    src.start(t);
    nextChunkTime = t + buf.duration;
}

function endStreamPlayback() {
    const ctx = ensureAudioCtx();
    const remaining = Math.max(0, nextChunkTime - ctx.currentTime);
    setTimeout(() => {
        stopWaveAnim();
        pttWrap.classList.remove("speaking");
    }, remaining * 1000 + 150);
}

function stopStreamPlaybackNow() {
    // Hard-stop any in-flight TTS. Called when the user hits "new chat".
    // We close and null out the AudioContext — every scheduled BufferSource
    // is rooted in it, so the close() cancels their playback cleanly.
    streamAnalyser = null;
    streamChunks = [];
    nextChunkTime = 0;
    if (audioCtx && audioCtx.state !== "closed") {
        try { audioCtx.close(); } catch (_) { /* ignore */ }
    }
    audioCtx = null;
    stopWaveAnim();
    pttWrap.classList.remove("speaking");
}

// ---- Helpers: base64 / WAV ------------------------------------------------

function b64ToInt16(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Int16Array(bytes.buffer);
}

function writeStr(view, off, s) {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
}

function buildReplayAudio(chunks, sr) {
    let len = 0;
    for (const c of chunks) len += c.length;
    const ab = new ArrayBuffer(44 + len * 2);
    const v = new DataView(ab);
    writeStr(v, 0, "RIFF");
    v.setUint32(4, 36 + len * 2, true);
    writeStr(v, 8, "WAVE");
    writeStr(v, 12, "fmt ");
    v.setUint32(16, 16, true);
    v.setUint16(20, 1, true);
    v.setUint16(22, 1, true);
    v.setUint32(24, sr, true);
    v.setUint32(28, sr * 2, true);
    v.setUint16(32, 2, true);
    v.setUint16(34, 16, true);
    writeStr(v, 36, "data");
    v.setUint32(40, len * 2, true);
    const out = new Int16Array(ab, 44);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    const blob = new Blob([ab], { type: "audio/wav" });
    return new Audio(URL.createObjectURL(blob));
}

// ---- UI helpers -----------------------------------------------------------

function setStatus(text, isError) {
    stopThinking();
    statusEl.textContent = text || "";
    statusEl.classList.toggle("error", Boolean(isError));
}

function startThinking() {
    pttWrap.classList.add("thinking");
}

function stopThinking() {
    pttWrap.classList.remove("thinking");
}

function setButtonState(state) {
    pttWrap.classList.toggle("recording", state === "recording");
    pttWrap.classList.toggle("processing", state === "processing");
    // PTT stays disabled while processing (voice turns don't queue —
    // the pipeline already has one running). Text input + send stay
    // enabled so the user can type ahead; extra messages land in
    // pendingTexts and drain once the current turn finishes.
    pttButton.disabled = state === "processing" || state === "loading";
    sendBtn.disabled = state === "loading";
    textInput.disabled = state === "loading";
}

function scrollToBottom() {
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: "smooth" });
}

// ---- Transcript -----------------------------------------------------------
//
// Each chat exchange gets its own .turn container with four pieces:
//   1. user bubble         (inserted immediately so the user sees their
//                           text echo back; voice turns show a placeholder
//                           "…" that gets replaced when STT transcribes)
//   2. thinking indicator  (animated dots with "Karin is thinking…"; stays
//                           visible until the transcript event arrives)
//   3. tool trace          (collapsed <details> that holds one line per
//                           tool_call event — "Tools (N)" summary; the
//                           user can click to inspect what Karin did)
//   4. assistant bubble    (filled in when the transcript arrives)
//
// The turn object returned by startTurn() has references to each piece
// so the NDJSON event handler can update them as events flow in.

function startTurn(initialUserText, options) {
    const opts = options || {};
    const turn = document.createElement("div");
    turn.className = "turn thinking";

    const userBubble = document.createElement("div");
    userBubble.className = "bubble user";
    if (opts.listening) {
        // Voice turn kicked off on PTT press — show an animated
        // "listening…" indicator until the server's transcript arrives.
        userBubble.classList.add("listening");
        userBubble.innerHTML =
            '<span class="listening-indicator">' +
            '<span class="bar"></span><span class="bar"></span>' +
            '<span class="bar"></span><span class="bar"></span>' +
            '<span class="listening-label">listening\u2026</span>' +
            '</span>';
    } else {
        userBubble.textContent = initialUserText || "\u2026";
    }
    turn.appendChild(userBubble);

    // Collapsed tool trace. Hidden by default; becomes visible only if
    // at least one tool_call event arrives. Defaults to collapsed.
    const toolTrace = document.createElement("details");
    toolTrace.className = "tool-trace";
    toolTrace.hidden = true;
    const toolSummary = document.createElement("summary");
    toolSummary.textContent = "Tools (0)";
    toolTrace.appendChild(toolSummary);
    turn.appendChild(toolTrace);

    const asstBubble = document.createElement("div");
    asstBubble.className = "bubble assistant";
    const thinkingEl = document.createElement("span");
    thinkingEl.className = "thinking-indicator";
    thinkingEl.innerHTML =
        '<span class="dot"></span><span class="dot"></span><span class="dot"></span>' +
        '<span class="thinking-label">thinking\u2026</span>';
    asstBubble.appendChild(thinkingEl);
    turn.appendChild(asstBubble);

    // Container for inline panels mounted after the assistant bubble.
    // Hidden until at least one tool-call with a panel arrives.
    const panelsContainer = document.createElement("div");
    panelsContainer.className = "turn-panels";
    panelsContainer.hidden = true;
    turn.appendChild(panelsContainer);

    transcriptEl.appendChild(turn);
    requestAnimationFrame(() => scrollToBottom());

    return {
        turn,
        userBubble,
        // Remember the user's actual text (not the "…" placeholder). Used
        // by the retry-on-thumbs-down flow to replay the same prompt with
        // a different tool-routing nudge.
        userText: (opts.listening ? "" : (initialUserText || "")),
        toolTrace,
        toolSummary,
        toolCount: 0,
        asstBubble,
        thinkingEl,
        panelsContainer,
        jobId: null,         // set by _runTextTurn once the server returns
    };
}

function turnSetUserText(turnObj, userText) {
    turnObj.userBubble.classList.remove("listening");
    const t = (userText || "").trim();
    turnObj.userText = t;
    if (!t) {
        // Silence / unclear audio — hide the bubble entirely so the
        // turn reads as Karin volunteering a "didn't catch that" line.
        turnObj.userBubble.remove();
        turnObj.userBubble = null;
        return;
    }
    turnObj.userBubble.textContent = t;
}

/**
 * Render an inline "⏰ Reminder set" card under the user bubble.
 * Called when the server emits a `reminder_set` event (chat-detection
 * layer in bridge/reminders/detect.py fired). The card shows the
 * trigger time + parsed message and an [Undo] button. Tapping Undo
 * calls /api/reminders/:id/cancel; on success the card collapses.
 *
 * Kept minimal on purpose — no "Edit" button yet, no time-zone
 * localization beyond the browser's default formatter. The Phase-4
 * calendar card and the Phase-5 LLM-fallback reuse this slot.
 */
function mountReminderCard(turnObj, rem) {
    if (!turnObj || !turnObj.userBubble) return;
    const card = document.createElement("div");
    card.className = "reminder-card";
    card.setAttribute("role", "status");
    card.setAttribute("aria-live", "polite");
    card.dataset.reminderId = rem.id;

    const icon = document.createElement("span");
    icon.className = "reminder-card-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "\u23f0";   // ⏰

    const textWrap = document.createElement("div");
    textWrap.className = "reminder-card-text";
    const title = document.createElement("div");
    title.className = "reminder-card-title";
    title.textContent = "Reminder set";
    const detail = document.createElement("div");
    detail.className = "reminder-card-detail";
    const when = _formatReminderTime(rem.triggerAt);
    detail.textContent = `${rem.message || "(no details)"} \u2014 ${when}`;
    textWrap.appendChild(title);
    textWrap.appendChild(detail);

    const undoBtn = document.createElement("button");
    undoBtn.type = "button";
    undoBtn.className = "reminder-card-undo";
    undoBtn.textContent = "Undo";
    undoBtn.addEventListener("click", async () => {
        undoBtn.disabled = true;
        undoBtn.textContent = "\u2026";   // working
        try {
            const r = await fetch(
                `/api/reminders/${encodeURIComponent(rem.id)}/cancel`,
                { method: "POST" },
            );
            // 404 = already fired / cancelled elsewhere — still treat
            // as "card is done" so the UI never lies.
            if (!r.ok && r.status !== 404) throw new Error(`HTTP ${r.status}`);
            card.classList.add("cancelled");
            card.querySelector(".reminder-card-title").textContent = "Reminder cancelled";
            undoBtn.hidden = true;
        } catch (e) {
            undoBtn.disabled = false;
            undoBtn.textContent = "Undo";
            card.querySelector(".reminder-card-detail").textContent =
                `Cancel failed: ${e.message || e}`;
        }
    });

    card.appendChild(icon);
    card.appendChild(textWrap);
    card.appendChild(undoBtn);

    // Slot the card into the turn DOM right after the user bubble so
    // the reading order matches the intent flow: you asked for it,
    // here's confirmation, then Karin's reply follows below.
    turnObj.userBubble.insertAdjacentElement("afterend", card);
    turnObj.reminderCard = card;
}

/** Friendly "in X minutes" / "tomorrow at 5:00 PM" style formatter.
 *  Browser's Intl.DateTimeFormat for the absolute, small home-grown
 *  delta string for the relative — avoids pulling in moment.js for
 *  one widget. */
function _formatReminderTime(isoString) {
    let d;
    try { d = new Date(isoString); } catch { return String(isoString); }
    if (!(d instanceof Date) || isNaN(d)) return String(isoString);
    const now = new Date();
    const deltaMs = d - now;
    const deltaMin = Math.round(deltaMs / 60000);
    if (deltaMin <= 0) return "now";
    if (deltaMin < 60) return `in ${deltaMin} min`;
    const deltaHr = Math.round(deltaMin / 60);
    if (deltaHr < 24) {
        return `in ${deltaHr} hour${deltaHr === 1 ? "" : "s"}`;
    }
    // Over a day out — show the absolute date + time in locale format.
    return d.toLocaleString(undefined, {
        weekday: "short", month: "short", day: "numeric",
        hour: "numeric", minute: "2-digit",
    });
}


function turnAddToolCall(turnObj, name, args, result) {
    const line = document.createElement("div");
    line.className = "tool-call";

    const callRow = document.createElement("div");
    callRow.className = "tool-call-call";
    const argsStr = Object.keys(args || {}).length ? ` ${JSON.stringify(args)}` : "";
    callRow.textContent = `\u2192 ${name}${argsStr}`;
    line.appendChild(callRow);

    // Result (if the server provided one — older clients may not).
    // Rendered below the call line so the expanded trace reads like:
    //   → get_weather {"location":"Philly"}
    //     Philadelphia, Pennsylvania, United States — partly cloudy, 12.3°C…
    if (result && String(result).trim()) {
        const resRow = document.createElement("div");
        resRow.className = "tool-call-result";
        resRow.textContent = String(result).trim();
        line.appendChild(resRow);
    }

    turnObj.toolTrace.appendChild(line);
    turnObj.toolCount += 1;
    turnObj.toolSummary.textContent =
        turnObj.toolCount === 1 ? "Tools (1)" : `Tools (${turnObj.toolCount})`;
    turnObj.toolTrace.hidden = false;
}

function turnFinalizeAssistant(turnObj, assistantText) {
    if (turnObj.thinkingEl && turnObj.thinkingEl.parentNode) {
        turnObj.thinkingEl.remove();
    }
    turnObj.turn.classList.remove("thinking");
    let at = turnObj.asstBubble.querySelector(".text");
    if (!at) {
        at = document.createElement("span");
        at.className = "text";
        turnObj.asstBubble.appendChild(at);
    }
    at.textContent = assistantText;
    turnObj.asstTextNode = at;
    // Copy button belongs on every real reply bubble. It's idempotent
    // so we can safely call it on re-renders (history restore, etc.).
    addCopyButton(turnObj.asstBubble);
    requestAnimationFrame(() => scrollToBottom());
    return turnObj.asstBubble;
}

function turnAppendAssistantText(turnObj, delta) {
    // Append a streaming token chunk. Lazily creates the text node on
    // the first delta (turnFinalizeAssistant("") sets it up).
    if (!turnObj.asstTextNode) {
        turnFinalizeAssistant(turnObj, "");
    }
    turnObj.asstTextNode.textContent += delta;
    // Keep the chat scrolled near the bottom while the bubble grows.
    if (chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight < 80) {
        chatArea.scrollTop = chatArea.scrollHeight;
    }
}

function turnReplaceAssistantText(turnObj, fullText) {
    if (!turnObj.asstTextNode) {
        turnFinalizeAssistant(turnObj, fullText);
        return;
    }
    turnObj.asstTextNode.textContent = fullText;
}

function attachFeedbackButtons(turnObj) {
    // No rating possible without a jobId (e.g. restored history turns).
    if (!turnObj || !turnObj.jobId || !turnObj.asstBubble) return;
    // Don't attach twice on the same turn.
    if (turnObj.feedbackRow) return;
    // Feature-flag gate: hide the thumbs row when
    // `feedback_thumbs` is off in features.yaml. The backing
    // /api/feedback endpoint and the feedback store stay active so
    // historical rows aren't lost — we just hide the UI entry point.
    const subs = (window.KARIN_CAPS && window.KARIN_CAPS.subsystems) || {};
    if (subs.feedback_thumbs === false) return;

    const row = document.createElement("div");
    row.className = "feedback-row";
    const up = document.createElement("button");
    up.className = "feedback-btn up";
    up.textContent = "\u{1F44D}";   // 👍
    up.title = "Good reply";
    const down = document.createElement("button");
    down.className = "feedback-btn down";
    down.textContent = "\u{1F44E}";   // 👎
    down.title = "Bad reply — regenerate";
    row.appendChild(up);
    row.appendChild(down);
    turnObj.asstBubble.appendChild(row);
    turnObj.feedbackRow = row;

    async function sendRating(rating) {
        try {
            await fetch("/api/feedback", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ turn_id: turnObj.jobId, rating }),
            });
        } catch (e) {
            console.warn("feedback POST failed:", e);
        }
    }

    up.addEventListener("click", async () => {
        up.disabled = true;
        down.disabled = true;
        row.classList.add("rated", "rated-up");
        await sendRating(1);
    });

    down.addEventListener("click", async () => {
        up.disabled = true;
        down.disabled = true;
        row.classList.add("rated", "rated-down");
        await sendRating(-1);
        // Regenerate the same user message with a retry hint.
        retryTurn(turnObj);
    });
}


function retryTurn(turnObj) {
    // Wipe the old assistant bubble + any inline widgets and re-run
    // the same user message with a retry hint on the server side.
    if (!turnObj.userText) return;
    const prevJobId = turnObj.jobId;

    // Clear the reply area + panels (keep the user bubble intact).
    const txt = turnObj.asstBubble.querySelector(".text");
    if (txt) txt.textContent = "";
    if (turnObj.feedbackRow && turnObj.feedbackRow.parentNode) {
        turnObj.feedbackRow.parentNode.removeChild(turnObj.feedbackRow);
        turnObj.feedbackRow = null;
    }
    if (turnObj.panelsContainer) {
        turnObj.panelsContainer.innerHTML = "";
        turnObj.panelsContainer.hidden = true;
    }
    turnObj.panelsByTool = null;
    turnObj.toolCount = 0;
    if (turnObj.toolTrace) {
        turnObj.toolTrace.hidden = true;
        const inner = turnObj.toolTrace.querySelectorAll(".tool-call");
        inner.forEach((el) => el.remove());
        if (turnObj.toolSummary) turnObj.toolSummary.textContent = "Tools (0)";
    }
    // Restore thinking state.
    turnObj.turn.classList.add("thinking");
    if (!turnObj.thinkingEl || !turnObj.thinkingEl.parentNode) {
        const t = document.createElement("span");
        t.className = "thinking-indicator";
        t.innerHTML =
            '<span class="dot"></span><span class="dot"></span><span class="dot"></span>' +
            '<span class="thinking-label">retrying\u2026</span>';
        turnObj.asstBubble.prepend(t);
        turnObj.thinkingEl = t;
    }

    _runTextTurn(turnObj.userText, turnObj, prevJobId);
}


function turnShowError(turnObj, message) {
    turnObj.thinkingEl.remove();
    turnObj.turn.classList.remove("thinking");
    turnObj.turn.classList.add("error");
    const at = document.createElement("span");
    at.className = "text";
    at.textContent = message;
    turnObj.asstBubble.appendChild(at);
    requestAnimationFrame(() => scrollToBottom());
}

// Back-compat shim for any legacy callers (kept for safety; not used by
// streamTurn anymore).
function addBubbles(userText, assistantText) {
    const t = startTurn(userText);
    return turnFinalizeAssistant(t, assistantText);
}

// Track the currently-playing replay so tapping another ▶ stops the
// previous one (no overlapping audio). Also lets the SAME button act
// as play/pause toggle: tap to play, tap again to stop.
let _activeReplay = { audio: null, btn: null };

function _stopActiveReplay() {
    if (_activeReplay.audio) {
        _activeReplay.audio.pause();
        _activeReplay.audio.currentTime = 0;
    }
    if (_activeReplay.btn) {
        _activeReplay.btn.textContent = "\u25B6";  // ▶
        _activeReplay.btn.classList.remove("playing");
    }
    // Cut any lip-sync animation driving the face button — otherwise
    // the mouth keeps cycling after the audio stops.
    if (face) face.idle();
    _activeReplay = { audio: null, btn: null };
}

function addReplayButton(bubble, audio, mouthSchedule) {
    if (window.KARIN_CAPS && window.KARIN_CAPS.tts === false) return;
    const btn = document.createElement("button");
    btn.className = "replay";
    btn.type = "button";
    btn.textContent = "\u25B6";
    btn.setAttribute("aria-label", "Play reply");

    // When audio ends naturally, reset the button.
    audio.addEventListener("ended", () => {
        if (_activeReplay.btn === btn) _stopActiveReplay();
    });

    btn.addEventListener("click", () => {
        // Same button tapped again while playing → stop.
        if (_activeReplay.audio === audio && !audio.paused) {
            _stopActiveReplay();
            return;
        }
        // Stop any streaming Web Audio playback first — prevents the
        // "plays twice, slightly delayed" overlap when auto-play was on
        // and the user taps ▶ before it finishes.
        stopStreamPlaybackNow();
        // Stop any other replay that's already playing.
        _stopActiveReplay();
        _activeReplay = { audio, btn };
        btn.textContent = "\u23F9";  // ⏹
        btn.classList.add("playing");
        audio.currentTime = 0;
        audio.play().then(() => {
            // Kick the precomputed schedule the instant playback starts.
            // Reusing the same plan on every replay means mouth and
            // audio stay locked to a single timeline — no drift between
            // the first (auto-play) render and later replay clicks.
            if (face && mouthSchedule) face.play(mouthSchedule);
        }).catch((e) => {
            _stopActiveReplay();
            setStatus(`Playback: ${e.message}`, true);
        });
    });
    bubble.appendChild(btn);
}

function addCopyButton(bubble) {
    // One-tap copy of the bubble's final text. Placed next to the
    // replay button. Idempotent: only one copy button per bubble.
    if (bubble.querySelector(".copy-btn")) return;
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.type = "button";
    btn.title = "Copy reply";
    btn.setAttribute("aria-label", "Copy reply");
    // Paper / clipboard glyph — matches the widget-button weight we
    // already use elsewhere (no emoji inflation).
    btn.innerHTML =
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" ' +
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
        'stroke-linejoin="round" aria-hidden="true">' +
        '<rect x="9" y="9" width="11" height="11" rx="2"></rect>' +
        '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
        '</svg>';
    btn.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        // Read the FINAL text element specifically. Skips the thinking
        // indicator, copy button itself, and any other bubble chrome.
        const textEl = bubble.querySelector(".text");
        const text = textEl ? textEl.textContent : bubble.textContent;
        const payload = (text || "").trim();
        if (!payload) return;
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(payload);
            } else {
                // Legacy fallback — covers older mobile Safari and any
                // context where Clipboard API is unavailable (insecure
                // origins, old browsers).
                const ta = document.createElement("textarea");
                ta.value = payload;
                ta.style.position = "fixed";
                ta.style.opacity = "0";
                document.body.appendChild(ta);
                ta.select();
                document.execCommand("copy");
                document.body.removeChild(ta);
            }
            // Visual confirmation — temporarily swap to "copied" state.
            btn.classList.add("copied");
            setTimeout(() => btn.classList.remove("copied"), 1200);
        } catch (err) {
            console.warn("copy failed:", err);
            setStatus("Copy failed", true);
        }
    });
    bubble.appendChild(btn);
}

// ---- Per-turn panel widget ------------------------------------------------

function closePanel() {
    // Called on "new chat" / conversation switch — tear every mounted
    // panel down so their polling / fetches stop.
    for (const ctl of mountedPanelControllers) {
        try { ctl.unmount(); } catch (_) { /* ignore */ }
    }
    mountedPanelControllers.clear();
}

function mountPanelInTurn(turnObj, toolName, args, result) {
    // Build a collapsible <details> containing the panel mount. The
    // shared mounter from chat.js returns null for tools that don't
    // have a panel (get_time, get_weather) — we skip those silently.
    //
    // Dedup: if the same tool fires again in this turn (e.g. the model
    // retries wiki_random after an unhelpful result), we REPLACE the
    // prior widget instead of stacking N widgets. The turn keeps one
    // widget per tool name; the most recent tool_call wins.
    if (!turnObj || !turnObj.panelsContainer) return;

    if (!turnObj.panelsByTool) turnObj.panelsByTool = new Map();
    const prior = turnObj.panelsByTool.get(toolName);
    if (prior) {
        try { prior.controller && prior.controller.unmount && prior.controller.unmount(); }
        catch (_) { /* best-effort cleanup */ }
        mountedPanelControllers.delete(prior.controller);
        if (prior.wrap && prior.wrap.parentNode) {
            prior.wrap.parentNode.removeChild(prior.wrap);
        }
        turnObj.panelsByTool.delete(toolName);
    }

    const wrap = document.createElement("details");
    wrap.className = "turn-panel";
    wrap.open = true;  // default expanded; user collapses via the summary

    const summary = document.createElement("summary");
    summary.className = "turn-panel-summary";
    const title = document.createElement("span");
    title.className = "turn-panel-title";
    title.textContent = "Panel";
    const subtitle = document.createElement("span");
    subtitle.className = "turn-panel-subtitle";
    summary.appendChild(title);
    summary.appendChild(subtitle);
    wrap.appendChild(summary);

    const body = document.createElement("div");
    body.className = "turn-panel-body";
    wrap.appendChild(body);

    const callbacks = {
        onSubtitle: (text) => { subtitle.textContent = text || ""; },
    };
    const mounted = Chat.mountPanelForTool(toolName, args || {}, body, callbacks, result);
    if (!mounted) return;  // no panel for this tool

    title.textContent = mounted.title;
    mountedPanelControllers.add(mounted.controller);
    turnObj.panelsContainer.appendChild(wrap);
    turnObj.panelsContainer.hidden = false;
    turnObj.panelsByTool.set(toolName, { controller: mounted.controller, wrap });
    requestAnimationFrame(() => scrollToBottom());
}

// ---- NDJSON streaming handler ---------------------------------------------

function isVoiceBackendDown(detail) {
    const s = String(detail || "").toLowerCase();
    return (
        s.includes("tts backend unreachable") ||
        s.includes("connection refused") ||
        s.includes("winerror 10061") ||
        s.includes("connecttimeout") ||
        s.includes("connecterror")
    );
}

function isTTSStreamGlitch(detail) {
    // Non-fatal mid-stream hiccups from GPT-SoVITS (chunked response
    // cut short, etc.). The assistant text already rendered — no point
    // alarming the user with red text above the circle.
    const s = String(detail || "").toLowerCase();
    return (
        s.includes("peer closed connection") ||
        s.includes("incomplete chunked read") ||
        s.includes("incomplete read") ||
        s.includes("chunked") ||
        s.includes("readerror") ||
        s.includes("remoteprotocolerror")
    );
}

function turnShowVoiceOffline(turnObj) {
    if (turnObj._voiceOfflineShown) return;
    turnObj._voiceOfflineShown = true;
    const note = document.createElement("div");
    note.className = "voice-offline-note";
    note.textContent = "voice not available \u2014 TTS server is offline";
    turnObj.turn.appendChild(note);
}

async function streamTurn(resp, turnObj) {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let asstBubble = null;
    let streamStarted = false;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        while (buf.includes("\n")) {
            const nl = buf.indexOf("\n");
            const line = buf.substring(0, nl).trim();
            buf = buf.substring(nl + 1);
            if (!line) continue;

            let evt;
            try { evt = JSON.parse(line); } catch { continue; }

            if (evt.type === "tool_call") {
                // Record the call in the turn's collapsible trace immediately,
                // but DEFER panel mounting until the transcript event fires
                // (i.e. until the LLM has a final answer). Otherwise a turn
                // that fans out across wiki_search + wiki_random + ... would
                // stack N half-formed widgets mid-stream. The last tool call
                // wins — that's the one the final reply is paraphrasing.
                turnAddToolCall(turnObj, evt.name, evt.arguments, evt.result);
                turnObj.pendingPanelCall = {
                    name: evt.name, args: evt.arguments, result: evt.result,
                };
            } else if (evt.type === "user_text") {
                // Job queue's pre-stream event: lock in the user bubble
                // text BEFORE the assistant starts streaming tokens.
                turnSetUserText(turnObj, evt.text);
            } else if (evt.type === "reminder_set") {
                // Chat-detected reminder: render an inline "⏰ set"
                // card with an Undo button. Happens before the LLM
                // reply so the card slots in naturally above the
                // assistant bubble. Cancel takes the reminder out
                // of the scheduler + hides the card.
                mountReminderCard(turnObj, {
                    id: evt.id,
                    triggerAt: evt.trigger_at,
                    message: evt.message,
                    matchedPhrase: evt.matched_phrase,
                });
            } else if (evt.type === "token_delta") {
                // Streaming token from the LLM. First delta: replace the
                // thinking-dots with an empty assistant bubble; subsequent
                // deltas append text. The full final reply also arrives
                // as a transcript event — we use that to reconcile in
                // case any deltas were missed (reconnect mid-stream).
                stopThinking();
                if (asstBubble == null) {
                    asstBubble = turnFinalizeAssistant(turnObj, "");
                }
                turnAppendAssistantText(turnObj, evt.delta);
                // Drive the face's vowel animation off the same stream,
                // but ONLY when audio will actually play — otherwise a
                // silent lip-sync during streaming is confusing (user
                // has auto-play off, or TTS disabled backend-side).
                if (face && isVoiceAutoPlay()
                    && !(window.KARIN_CAPS && window.KARIN_CAPS.tts === false)) {
                    face.stepText(evt.delta);
                }
            } else if (evt.type === "transcript") {
                stopThinking();
                turnSetUserText(turnObj, evt.user);
                if (asstBubble == null) {
                    asstBubble = turnFinalizeAssistant(turnObj, evt.assistant);
                } else {
                    // Reconcile in case stream tokens dropped — replace
                    // bubble text with the canonical full reply.
                    turnReplaceAssistantText(turnObj, evt.assistant);
                }
                // Remember the finished reply so we can scrub the face
                // through it during TTS playback (driven from 'done').
                turnObj.finalReply = evt.assistant || "";
                // Mount the widget NOW — we have a final response, so the
                // panel shows up once with the "winning" tool's data.
                if (turnObj.pendingPanelCall) {
                    const pc = turnObj.pendingPanelCall;
                    mountPanelInTurn(turnObj, pc.name, pc.args, pc.result);
                    turnObj.pendingPanelCall = null;
                }
            } else if (evt.type === "audio") {
                // Always collect chunks for the replay button, but
                // only PLAY them if voice auto-play is on. When off,
                // streamStarted stays false so the done handler still
                // builds the replay audio object + shows the ▶ button.
                const pcm = b64ToInt16(evt.b64);
                if (!streamChunks) streamChunks = [];
                streamSr = evt.sr || 32000;
                streamChunks.push(pcm);
                if (isVoiceAutoPlay()) {
                    if (!streamStarted) { initStreamPlayback(); streamStarted = true; }
                    scheduleChunk(pcm, evt.sr);
                }
            } else if (evt.type === "error") {
                // Voice backend being down is a known, non-fatal state —
                // the transcript already rendered, so don't light up the
                // red status bar. Show a small muted note instead.
                if (isVoiceBackendDown(evt.detail)) {
                    turnShowVoiceOffline(turnObj);
                } else if (isTTSStreamGlitch(evt.detail)) {
                    // Mid-stream hiccup — swallow silently, the bulk of
                    // the audio already played.
                    console.warn("TTS stream glitch (suppressed):", evt.detail);
                } else {
                    setStatus(`TTS: ${evt.detail}`, true);
                }
            } else if (evt.type === "done") {
                // Build the replay audio + button regardless of whether
                // auto-play was on — the ▶ button is how users with
                // auto-play off (or mobile-blocked autoplay) hear the
                // voice.
                if (streamStarted) {
                    endStreamPlayback();
                }
                if (streamChunks && streamChunks.length > 0) {
                    // Duration is the canonical sync target. Compute it
                    // ONCE from the PCM sample count — this is the exact
                    // wall-clock the audio will occupy.
                    let totalSamples = 0;
                    for (const c of streamChunks) totalSamples += c.length;
                    const durationMs = (totalSamples / streamSr) * 1000;
                    const audio = buildReplayAudio(streamChunks, streamSr);

                    // Build the mouth schedule ONCE and stash on the turn.
                    // Replay click reuses this plan instead of recomputing
                    // — keeps every playback locked to the same timeline.
                    if (window.Face && turnObj.finalReply) {
                        turnObj.mouthSchedule =
                            window.Face.buildSchedule(turnObj.finalReply, durationMs);
                    }

                    if (asstBubble) {
                        addReplayButton(asstBubble, audio, turnObj.mouthSchedule || null);
                    }
                    // Auto-play lip-sync: only when the audio is ACTUALLY
                    // going to play in real time. When auto-play is off,
                    // the face stays idle until the user clicks ▶.
                    if (face && turnObj.mouthSchedule && isVoiceAutoPlay()) {
                        face.play(turnObj.mouthSchedule);
                    } else if (face) {
                        face.idle();
                    }
                } else if (face) {
                    // No TTS — make sure we end on the default frame.
                    face.idle();
                }
            }
        }
    }
}

// ---- Recording (push-to-talk) --------------------------------------------

// Hard cap on recording length — defensive against the case where a
// pointerup event is lost (some mobile browsers drop pointerup if the
// gesture is interrupted, leaving recording stuck on). Also keeps a
// runaway PTT press from filling memory with audio chunks.
const MAX_RECORDING_MS = 30000;
let recordingTimeoutId = null;

// Web Speech API handle for the current turn, or null. When active,
// stopRecording() stops it instead of the MediaRecorder. Created per
// session so we don't leak recognizer instances across taps.
let webSpeechSession = null;

async function startRecording() {
    if (isProcessing) return;

    // Prefer the browser's native Web Speech API when it's available
    // AND we're online. Zero server-side STT cost, typically faster
    // first-word latency than uploading to our local Whisper. Falls
    // through to the local recording path on unsupported browsers
    // (Firefox) or offline.
    if (window.WebSpeechRecognizer && WebSpeechRecognizer.isAvailable()) {
        if (startRecordingWebSpeech()) return;
        // If it failed to start (rare), fall through to local recording.
    }

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
        chunks = [];
        mediaRecorder = new MediaRecorder(mediaStream);
        mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) chunks.push(e.data);
        };
        mediaRecorder.onstop = () => {
            if (recordingTimeoutId) { clearTimeout(recordingTimeoutId); recordingTimeoutId = null; }
            stopWaveAnim();
            pttWrap.classList.remove("recording");
            if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
            const reason = _vadStopReason;
            _vadStopReason = null;
            if (reason === "no_speech") {
                // VAD never heard speech — user tapped by accident or
                // the mic was muted. Silently drop everything; don't
                // wake the LLM, don't leave a placeholder turn hanging.
                chunks = [];
                if (activeVoiceTurn) {
                    activeVoiceTurn.turn.remove();
                    activeVoiceTurn = null;
                }
                setButtonState("idle");
                return;
            }
            handleRecordingStop();
        };
        mediaRecorder.start();
        startRecordingWave(mediaStream);
        setButtonState("recording");
        setStatus("");
        // VAD: reset speech-detection state per session.
        vadSpeechHeard = false;
        vadStartedAt = performance.now();
        vadLastSpeechAt = vadStartedAt;
        // Safety auto-stop — VAD should fire long before this.
        recordingTimeoutId = setTimeout(() => {
            console.warn("recording auto-stopped after max duration");
            stopRecording();
        }, MAX_RECORDING_MS);

        // Pre-create the turn now so the listening indicator is visible
        // the instant the user presses PTT. submitVoice() reuses this
        // turn when recording stops; if the recording gets cancelled
        // (too short / aborted), handleRecordingStop() clears it.
        activeVoiceTurn = startTurn("", { listening: true });
    } catch (err) {
        setStatus(`Mic error: ${err.message}`, true);
        setButtonState("idle");
    }
}

function startRecordingWebSpeech() {
    // Returns true on successful engagement, false if the API refused
    // (permissions error, already running, etc.) so the caller can
    // fall back to local recording.
    webSpeechSession = new WebSpeechRecognizer({
        continuous: false,
        interimResults: true,
    });
    activeVoiceTurn = startTurn("", { listening: true });
    setButtonState("recording");
    pttWrap.classList.add("recording");
    setStatus("");

    let finalized = false;

    webSpeechSession.onInterim = (text) => {
        // Live transcription: as Google / Apple streams back words,
        // write them into the user bubble in real time. The face's
        // mouth animates through the deltas via stepText, so the
        // user sees BOTH the text appearing and Karin visibly
        // listening.
        if (!activeVoiceTurn || !activeVoiceTurn.userBubble) return;
        activeVoiceTurn.userBubble.classList.remove("listening");
        const prev = activeVoiceTurn.userText || "";
        activeVoiceTurn.userBubble.textContent = text;
        activeVoiceTurn.userText = text;
        // Mouth-sync the face to the new characters since the last
        // interim — cheap way to animate during web-speech sessions
        // that don't drive the RMS analyser.
        if (face && text.length > prev.length) {
            face.stepText(text.slice(prev.length));
        }
    };

    webSpeechSession.onFinal = async (text) => {
        finalized = true;
        pttWrap.classList.remove("recording");
        setButtonState("idle");
        if (!text.trim()) return;
        if (activeVoiceTurn && activeVoiceTurn.userBubble) {
            activeVoiceTurn.userBubble.textContent = text;
            activeVoiceTurn.userText = text;
        }
        // Hand off to the regular text-turn flow, reusing the turn
        // object so the bubble we just populated doesn't duplicate.
        const turn = activeVoiceTurn;
        activeVoiceTurn = null;
        webSpeechSession = null;
        await _runTextTurn(text, turn);
    };

    webSpeechSession.onError = (code) => {
        pttWrap.classList.remove("recording");
        setButtonState("idle");
        // Treat "no-speech" and "aborted" as silent drops — user
        // either didn't talk or tapped again to cancel.
        if (code === "no-speech" || code === "aborted") {
            if (activeVoiceTurn) {
                activeVoiceTurn.turn.remove();
                activeVoiceTurn = null;
            }
            webSpeechSession = null;
            return;
        }
        // Any other error: tear down + surface to status bar. Local
        // recording doesn't auto-retry mid-gesture; next tap gets a
        // fresh availability check (e.g. if network just dropped).
        if (activeVoiceTurn) {
            activeVoiceTurn.turn.remove();
            activeVoiceTurn = null;
        }
        webSpeechSession = null;
        setStatus(`STT: ${code}`, true);
    };

    webSpeechSession.onEnd = () => {
        // Belt-and-suspenders: if the session ended without emitting
        // a final (can happen on rapid stop), clean up now.
        if (finalized) return;
        pttWrap.classList.remove("recording");
        setButtonState("idle");
        if (activeVoiceTurn) {
            activeVoiceTurn.turn.remove();
            activeVoiceTurn = null;
        }
        webSpeechSession = null;
    };

    const ok = webSpeechSession.start();
    if (!ok) {
        // API refused — clean up the provisional turn and report
        // failure so the caller can fall back.
        if (activeVoiceTurn) { activeVoiceTurn.turn.remove(); activeVoiceTurn = null; }
        pttWrap.classList.remove("recording");
        setButtonState("idle");
        webSpeechSession = null;
        return false;
    }
    return true;
}

function stopRecording() {
    // Web Speech path — stop() triggers a final result (if any) then onend.
    if (webSpeechSession && webSpeechSession.active) {
        webSpeechSession.stop();
        return;
    }
    // Local Whisper path — MediaRecorder.stop() fires onstop which
    // ships the blob to /api/turn.
    if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
}

async function handleRecordingStop() {
    if (chunks.length === 0) {
        // No audio at all — drop the placeholder turn and bail.
        if (activeVoiceTurn) { activeVoiceTurn.turn.remove(); activeVoiceTurn = null; }
        setButtonState("idle");
        return;
    }
    const mime = (mediaRecorder && mediaRecorder.mimeType) || "audio/webm";
    const blob = new Blob(chunks, { type: mime });
    chunks = [];
    // Don't filter short blobs client-side — the server treats them as
    // silence and routes through the "sorry, didn't catch that" sentinel.
    await submitVoice(blob, mime);
}

async function submitVoice(blob, mime) {
    isProcessing = true;
    setButtonState("processing");
    startThinking();
    // Reuse the turn created when recording started (so the listening
    // animation transitions straight into the thinking dots). Fall back
    // to a fresh turn if something odd happened and none exists.
    const turnObj = activeVoiceTurn || startTurn("", { listening: true });
    activeVoiceTurn = null;
    // Flip the user bubble from "listening…" to a short ellipsis while
    // STT runs on the server. The final text lands on the transcript event.
    turnSetUserText(turnObj, "\u2026");
    try {
        const fd = new FormData();
        fd.append("audio", blob, `input.${mime.includes("mp4") ? "mp4" : "webm"}`);
        const resp = await fetch("/api/turn-stream", { method: "POST", body: fd });
        if (!resp.ok) {
            let d = `HTTP ${resp.status}`;
            try { const e = await resp.json(); if (e.detail) d = e.detail; } catch {}
            throw new Error(d);
        }
        await streamTurn(resp, turnObj);
    } catch (err) {
        // Chat-flow errors live inside the turn bubble only. The top
        // status line is reserved for connection-level problems.
        turnShowError(turnObj, `Error: ${err.message}`);
    } finally {
        stopThinking();
        isProcessing = false;
        setButtonState("idle");
        loadConversationList();
    }
}

// ---- Text prompt ----------------------------------------------------------

// ---- Text input queue ----------------------------------------------------
//
// The pipeline serializes turns server-side (pipeline_lock) so firing two
// simultaneously just makes the second wait inside the server. Client-side
// we keep typing enabled and queue extras locally so each queued message
// appears in the transcript immediately (as a "queued" bubble) and drains
// in order as soon as the current turn completes. This mirrors how ChatGPT
// handles typing-ahead while a response is streaming.

const pendingTexts = [];  // [{ text, turnObj }]

async function submitText(text) {
    const trimmed = text.trim();
    if (!trimmed) return;

    // Show the user bubble immediately, even if we have to queue.
    const turnObj = startTurn(trimmed);
    if (isProcessing) {
        turnObj.turn.classList.add("queued");
        // The thinking indicator keeps blinking so it's clear the reply
        // is pending. No explicit "queued" badge — the ordering in the
        // transcript makes the queue visible on its own.
        pendingTexts.push({ text: trimmed, turnObj });
        return;
    }

    await _runTextTurn(trimmed, turnObj);
    // After completion, drain any that arrived during processing.
    while (pendingTexts.length > 0) {
        const next = pendingTexts.shift();
        next.turnObj.turn.classList.remove("queued");
        await _runTextTurn(next.text, next.turnObj);
    }
}

async function _runTextTurn(trimmed, turnObj, retryOf) {
    // Two-step: POST /api/turn/start enqueues the work on the server's
    // background worker (returns instantly with a job_id), then we
    // open a streaming GET on /api/turn/<id>/stream. If the user closes
    // the browser mid-turn, the server keeps chewing — when they come
    // back we could even resume by replaying events from the job log.
    //
    // `retryOf` is the job_id of a prior thumbs-down'd turn; when set,
    // the server injects a retry hint naming the tools used last time
    // so the model tries a different path.
    isProcessing = true;
    setButtonState("processing");
    startThinking();
    try {
        const body = { text: trimmed };
        if (retryOf) body.retry_of = retryOf;
        const startResp = await fetch("/api/turn/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!startResp.ok) {
            let d = `HTTP ${startResp.status}`;
            try { const e = await startResp.json(); if (e.detail) d = e.detail; } catch {}
            throw new Error(d);
        }
        const { job_id } = await startResp.json();
        turnObj.jobId = job_id;

        const streamResp = await fetch(`/api/turn/${encodeURIComponent(job_id)}/stream`);
        if (!streamResp.ok) {
            throw new Error(`HTTP ${streamResp.status} reading job stream`);
        }
        await streamTurn(streamResp, turnObj);
        // Thumbs row only makes sense once the reply actually landed.
        attachFeedbackButtons(turnObj);
    } catch (err) {
        turnShowError(turnObj, `Error: ${err.message}`);
    } finally {
        stopThinking();
        isProcessing = false;
        setButtonState("idle");
        loadConversationList();
    }
}

// ---- Voice auto-play toggle button -----------------------------------------

const voiceAutoPlayToggle = document.getElementById("voice-autoplay-toggle");
function syncAutoPlayToggle() {
    if (!voiceAutoPlayToggle) return;
    voiceAutoPlayToggle.checked = isVoiceAutoPlay();
    const row = document.getElementById("voice-autoplay-row");
    if (row && window.KARIN_CAPS && window.KARIN_CAPS.tts === false) {
        row.parentElement.hidden = true;
    }
}
if (voiceAutoPlayToggle) {
    voiceAutoPlayToggle.addEventListener("change", () => {
        setVoiceAutoPlay(voiceAutoPlayToggle.checked);
    });
    syncAutoPlayToggle();
}

// ---- Two-phase compose toggle (quality vs speed) --------------------
// Flips the server-side `llm.two_phase_compose` flag. When on, every
// tool-using turn gets an extra "compose" LLM call with only the user
// prompt + tool output in context — cleaner replies, +5-10s/turn. No-
// tool turns also go through a casual-compose pass that strips the
// tool schema before the final chitchat reply. Off by default.
const twoPhaseToggle = document.getElementById("two-phase-toggle");
async function syncTwoPhaseToggle() {
    if (!twoPhaseToggle) return;
    try {
        const r = await fetch("/api/features", { cache: "no-store" });
        if (r.ok) {
            const s = await r.json();
            twoPhaseToggle.checked = !!s.two_phase_compose;
        }
    } catch (e) { /* best-effort; leave unchecked on failure */ }
}
if (twoPhaseToggle) {
    twoPhaseToggle.addEventListener("change", async () => {
        const want = twoPhaseToggle.checked;
        try {
            const r = await fetch("/api/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ two_phase_compose: want }),
            });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            const body = await r.json();
            // If the server came back with a different value (e.g.
            // validation flipped it), show that instead of what the
            // user clicked.
            if (body && body.applied && typeof body.applied.two_phase_compose === "boolean") {
                twoPhaseToggle.checked = body.applied.two_phase_compose;
            }
        } catch (e) {
            // Revert optimistic UI change on failure.
            twoPhaseToggle.checked = !want;
            console.warn("two-phase toggle POST failed:", e);
        }
    });
    syncTwoPhaseToggle();
}

// ---- Character face + voice dropdown -------------------------------
// Picks WHICH character drives (a) the face on the top PTT button and
// (b) the TTS voice. Scope is one character per conversation thread —
// this is a single-selection global picker, persisted to localStorage.
// Multi-character-per-thread is a future concept bounded by how much
// persona state an on-device LLM can juggle; not in scope today.
//
// On change, robustness requires:
//   1. Block during in-flight turns (isProcessing) so a mid-turn swap
//      doesn't corrupt TTS playback or history.
//   2. Resolve the target VOICE (for "auto", fall back to the config
//      default injected as window.KARIN_ACTIVE_VOICE).
//   3. POST /api/tts/voice BEFORE saving + reloading, so a failed
//      swap leaves localStorage + dropdown at the prior selection.
//   4. Surface failures via setStatus; recognise "TTS is disabled" as
//      a tolerable no-op (face still changes).
//   5. Debounce rapid toggles by disabling the <select> while the
//      swap is in flight.
//
// Options are populated from window.KARIN_AVAILABLE_FACES (server
// scans characters/ on every page load). Adding a folder under
// characters/<name>/ makes it appear in the dropdown with no bridge
// restart. Each entry carries `has_voice: bool` (server-side filesystem
// scan for ref.wav + *.ckpt + *.pth); entries without a voice bundle
// get a leading "○" indicator so the dropdown doesn't need the noisy
// "(text only)" suffix on every row when voices/ is gitignored.
const faceCharacterSelect = document.getElementById("face-character-select");
if (faceCharacterSelect) {
    const available = Array.isArray(window.KARIN_AVAILABLE_FACES)
        ? window.KARIN_AVAILABLE_FACES
        : [];

    // Quick lookup so swapVoice can ask "does this name have a voice?"
    // without re-walking the available list each time.
    const voicePresence = new Map();
    for (const entry of available) {
        voicePresence.set(entry.name, entry.has_voice !== false);
    }

    // Build options: one per available character. No "Auto" — it was
    // ambiguous (swapped the voice but the persona was invisibly tied
    // to config/assistant.yaml::character, which the dropdown didn't
    // update). Now every option is a concrete character name; picking
    // one swaps BOTH voice + persona via /api/tts/voice (see
    // web/panels_api.py::tts_voice_switch for the server side of the
    // pairing).
    const nameSet = new Set();
    for (const entry of available) {
        nameSet.add(entry.name);
        const opt = document.createElement("option");
        opt.value = entry.name;
        const faceSuffix = entry.type === "procedural-sun" ? " ☀" : "";
        // Hollow-circle indicator for voice-less characters — reads as
        // a status light inside an <option> (which can't host colored
        // widgets) and is shorter than the old "(text only)" suffix.
        const voicePrefix = entry.has_voice === false ? "○ " : "";
        opt.textContent = voicePrefix + (entry.label || entry.name) + faceSuffix;
        if (entry.has_voice === false) {
            opt.title = "No voice bundle on disk — face + theme will swap, but TTS stays silent until you train/drop in weights.";
        }
        faceCharacterSelect.appendChild(opt);
    }

    // Default selection: server's KARIN_ACTIVE_CHARACTER (injected by
    // the template), then saved localStorage override, then whatever
    // the browser picks (first entry).
    //
    // Saved names that are no longer in the character list (folder
    // deleted / renamed) fall back to the active character. "auto" is
    // a legacy localStorage value from the pre-removal dropdown —
    // treat it as unset and use the active character.
    const active = (window.KARIN_ACTIVE_CHARACTER || "").trim();
    let saved = "";
    try { saved = window.localStorage.getItem("karin.faceCharacter") || ""; }
    catch (e) { /* localStorage unavailable */ }
    if (saved === "auto" || !nameSet.has(saved)) {
        saved = nameSet.has(active) ? active : faceCharacterSelect.options[0]?.value || "";
    }
    faceCharacterSelect.value = saved;
    // Remember the settled value so a failed swap can revert to it.
    faceCharacterSelect.dataset.lastValue = faceCharacterSelect.value;

    async function swapVoice(targetName) {
        // Swap the character server-side (persona + face + voice-weights
        // if present) and return {ok, detail}. ok=true means "safe to
        // persist the dropdown choice"; ok=false with detail means
        // "show the user and revert".
        //
        // Voice-less characters (has_voice=false, e.g. the shipped
        // "default") still POST so the server swaps persona +
        // KARIN_CHARACTER env; the endpoint handles the missing voice
        // bundle gracefully. Previously this short-circuited, which
        // broke persona swap for the voice-less case.
        try {
            const r = await fetch("/api/tts/voice", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: targetName }),
            });
            if (r.ok) return { ok: true };
            // Try to pull the FastAPI detail string; fall back to
            // status text if the body isn't JSON.
            let detail = `HTTP ${r.status}`;
            try {
                const body = await r.json();
                if (body && body.detail) detail = String(body.detail);
            } catch (_) { /* non-JSON body */ }
            // "TTS is disabled" is tolerable — a deployment without the
            // PC TTS sidecar still wants face switching to work.
            if (/TTS is disabled/i.test(detail)) return { ok: true, skipped: true };
            return { ok: false, detail };
        } catch (e) {
            return { ok: false, detail: `network: ${e && e.message || e}` };
        }
    }

    // Re-sync voice on page load. If the PC-TTS process restarted
    // between sessions (sleep/wake, tray-icon reset), it reverts to
    // its config default — the Jetson side still thinks
    // "_weights_loaded=True" so it won't push again on the next
    // synthesize. Result: user had Karin selected, comes back, speaks,
    // hears General's voice. One silent POST on load fixes it with a
    // model-reload that's ~1 s and invisible if everything was already
    // in sync (same weights path → no-op on PC-TTS).
    async function resyncVoiceOnce() {
        const pref = saved;  // captured above when restoring selection
        if (!pref || pref === "auto") return;
        // Same short-circuit as swapVoice — don't POST for a character
        // the server already reported as voice-less.
        if (voicePresence.get(pref) === false) return;
        try {
            // Skip if the server already reports this voice active —
            // avoids a needless model reload.
            const r = await fetch("/api/tts/voices", { cache: "no-store" });
            if (r.ok) {
                const body = await r.json();
                if (body && body.active === pref) return;
            }
        } catch (e) { /* non-fatal; fall through and try the POST */ }
        try {
            await fetch("/api/tts/voice", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: pref }),
            });
        } catch (e) {
            console.warn("voice resync on load failed (non-fatal):", e);
        }
    }
    // Run async after the page settles so we don't block first render.
    setTimeout(resyncVoiceOnce, 300);

    faceCharacterSelect.addEventListener("change", async () => {
        const want = faceCharacterSelect.value || "auto";
        const prev = faceCharacterSelect.dataset.lastValue || "auto";
        if (want === prev) return;

        // Mid-turn swaps would cut off streaming audio / break history
        // hydration on reload. Block until the turn finishes.
        if (typeof isProcessing !== "undefined" && isProcessing) {
            setStatus("Wait for the current turn to finish before switching character.", true);
            faceCharacterSelect.value = prev;
            return;
        }

        // Every dropdown option is a concrete character name — the
        // swap target IS the selected value. No legacy "auto" branch.
        const targetVoice = want;

        faceCharacterSelect.disabled = true;
        setStatus(`Switching to ${want}…`, false);

        let result = { ok: true };
        if (targetVoice) {
            result = await swapVoice(targetVoice);
        }

        if (!result.ok) {
            // Truncate very long server details (e.g. full stack traces)
            // so the status bar stays readable. Full message is in the
            // console for debugging.
            const d = String(result.detail || "unknown");
            console.warn("Character switch failed:", d);
            setStatus(`Character switch failed: ${d.slice(0, 160)}`, true);
            faceCharacterSelect.value = prev;
            faceCharacterSelect.disabled = false;
            return;
        }

        try { window.localStorage.setItem("karin.faceCharacter", want); }
        catch (e) { /* localStorage unavailable; choice won't persist */ }
        // Reload so the face picker mounts the right renderer + theme.
        // (Voice + persona are already swapped server-side by the POST
        // above — reload is purely for the face/theme visuals.)
        window.location.reload();
    });
}

sendBtn.addEventListener("click", () => {
    // Resume AudioContext on user gesture — mobile browsers block
    // autoplay unless the context was created/resumed from a tap.
    // Do this BEFORE submitText so the streaming audio path has a
    // live context when chunks arrive.
    ensureAudioCtx();
    const t = textInput.value; textInput.value = ""; submitText(t);
});
textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        ensureAudioCtx();
        const t = textInput.value; textInput.value = ""; submitText(t);
    }
});

// ---- Push-to-talk wiring --------------------------------------------------

// Single mic mode: tap the face once to start, tap again to stop.
// Browser-side VAD (in animateWave) handles auto-stop too:
//   - silence after speech → stop + submit (normal flow)
//   - no speech at all within VAD_MAX_PRESPEECH_MS → stop + SKIP SUBMIT
//     (the no-op handling sits in mediaRecorder.onstop; see
//     _vadStopReason in the recording section).
pttButton.addEventListener("click", (e) => {
    e.preventDefault();
    if (isProcessing) return;
    if (window.KARIN_CAPS && window.KARIN_CAPS.stt === false) return;
    if (mediaRecorder && mediaRecorder.state === "recording") {
        stopRecording();
    } else {
        startRecording();
    }
});

pttButton.addEventListener("contextmenu", (e) => e.preventDefault());

// ---- Sidebar: conversation list -------------------------------------------

const conversationListEl = document.getElementById("conversation-list");
const sidebarToggleBtn = document.getElementById("sidebar-toggle");
const appShell = document.querySelector(".app-shell");

function formatConvTimestamp(iso) {
    if (!iso) return "";
    try {
        const d = new Date(iso);
        return d.toLocaleString(undefined, {
            month: "short", day: "numeric",
            hour: "numeric", minute: "2-digit",
        });
    } catch { return ""; }
}

async function loadConversationList() {
    if (!conversationListEl) return;
    try {
        const r = await fetch("/api/history/list");
        if (!r.ok) return;
        const data = await r.json();
        renderConversationList(data.conversations || [], data.current);
    } catch (e) {
        console.warn("conversation list failed:", e);
    }
}

function renderConversationList(convs, currentId) {
    conversationListEl.innerHTML = "";
    if (!convs.length) {
        const empty = document.createElement("div");
        empty.className = "conv-item";
        empty.innerHTML = '<span class="conv-preview empty">No conversations yet</span>';
        conversationListEl.appendChild(empty);
        return;
    }
    for (const c of convs) {
        const item = document.createElement("div");
        item.className = "conv-item" + (c.id === currentId ? " current" : "");
        item.setAttribute("role", "listitem");
        item.dataset.conversationId = c.id;

        const preview = document.createElement("div");
        preview.className = "conv-preview" + (c.preview ? "" : " empty");
        preview.textContent = c.preview || "(empty)";
        item.appendChild(preview);

        const meta = document.createElement("div");
        meta.className = "conv-meta";
        const when = document.createElement("span");
        when.textContent = formatConvTimestamp(c.updated_at || c.created_at);
        const count = document.createElement("span");
        const n = c.message_count || 0;
        count.textContent = n ? `${n} msg` : "";
        meta.appendChild(when);
        meta.appendChild(count);
        item.appendChild(meta);

        // Empty conversations (no user messages yet) are reused by the
        // "New chat" button on the server — deleting one would be
        // pointless and confusing, so we just omit the button.
        if (n > 0) {
            const del = document.createElement("button");
            del.className = "conv-delete";
            del.type = "button";
            del.setAttribute("aria-label", "Delete conversation");
            del.title = "Delete";
            del.textContent = "\u00D7";  // ×
            del.addEventListener("click", (e) => {
                e.stopPropagation();
                deleteConversation(c.id);
            });
            item.appendChild(del);
        }

        item.addEventListener("click", () => switchConversation(c.id));
        conversationListEl.appendChild(item);
    }
}

async function deleteConversation(cid) {
    if (isProcessing) return;
    try {
        const r = await fetch(`/api/history/${encodeURIComponent(cid)}`, { method: "DELETE" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        if (data.was_current) {
            // Server switched us to a different conversation — refresh the
            // transcript so the UI matches what the LLM now holds.
            stopStreamPlaybackNow();
            pendingTexts.length = 0;
            transcriptEl.innerHTML = "";
            closePanel();
            await restoreHistory();
        }
        await loadConversationList();
    } catch (e) {
        setStatus(`Delete failed: ${e.message}`, true);
    }
}

async function switchConversation(cid) {
    if (isProcessing) return;
    try {
        const r = await fetch("/api/history/switch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ conversation_id: cid }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
    } catch (e) {
        setStatus(`Switch failed: ${e.message}`, true);
        return;
    }
    stopStreamPlaybackNow();
    pendingTexts.length = 0;
    transcriptEl.innerHTML = "";
    closePanel();
    setStatus("");
    await restoreHistory();
    await loadConversationList();
}

// --- Universal tri-state toggle + pop-up controller -----------------------
//
// The toggle button on the right edge of the viewport does one of three
// things depending on what's currently open:
//
//   "idle"    → open the left sidebar
//   "sidebar" → close the left sidebar
//   "popup"   → close the panel pop-up (overlay)
//
// The button itself never moves; it just swaps icon + aria-label. The
// pop-up is a full-screen overlay hosting an iframe that loads one
// /ui/* page at a time, with bottom tabs for switching between them.
// Any element carrying `data-panel-target="/ui/…"` — sidebar shortcut
// buttons, in-page deep links — routes through openPopup().

const popupOverlayEl = document.getElementById("popup-overlay");
const popupFrameEl = document.getElementById("popup-overlay-frame");

const UIState = (() => {
    // Remembers whether the sidebar was open before a pop-up took over,
    // so closing the pop-up can restore the prior layout rather than
    // always collapsing.
    let sidebarWasOpenBeforePopup = false;

    function syncToggleMode(mode) {
        if (!sidebarToggleBtn) return;
        sidebarToggleBtn.setAttribute("data-mode", mode);
        if (mode === "idle") {
            sidebarToggleBtn.setAttribute("aria-label", "Open sidebar");
            sidebarToggleBtn.setAttribute("aria-expanded", "false");
        } else if (mode === "sidebar") {
            sidebarToggleBtn.setAttribute("aria-label", "Close sidebar");
            sidebarToggleBtn.setAttribute("aria-expanded", "true");
        } else if (mode === "popup") {
            sidebarToggleBtn.setAttribute("aria-label", "Close panel");
            sidebarToggleBtn.setAttribute("aria-expanded", "true");
        }
    }

    function isMobile() {
        return window.matchMedia("(max-width: 720px)").matches;
    }

    function isSidebarOpen() {
        if (!appShell) return false;
        return isMobile()
            ? appShell.classList.contains("sidebar-open")
            : !appShell.classList.contains("sidebar-collapsed");
    }

    function openSidebar() {
        if (!appShell) return;
        if (isMobile()) {
            appShell.classList.add("sidebar-open");
        } else {
            appShell.classList.remove("sidebar-collapsed");
        }
        syncToggleMode("sidebar");
    }

    function closeSidebar() {
        if (!appShell) return;
        if (isMobile()) {
            appShell.classList.remove("sidebar-open");
        } else {
            appShell.classList.add("sidebar-collapsed");
        }
        syncToggleMode("idle");
    }

    function isPopupOpen() {
        return popupOverlayEl && !popupOverlayEl.hidden;
    }

    function setActiveTab(url) {
        const tabs = popupOverlayEl.querySelectorAll(".popup-overlay-tab");
        tabs.forEach((t) => {
            const match = t.getAttribute("data-panel-target") === url;
            t.setAttribute("aria-selected", match ? "true" : "false");
        });
    }

    function openPopup(url, label) {
        if (!popupOverlayEl || !popupFrameEl) return;
        const target = url || "/ui/digest";
        // Load the /ui/* page fresh each open so the iframe reflects
        // current data (digests / news rescore between visits).
        popupFrameEl.src = target;
        setActiveTab(target);
        // Track whether the sidebar was open so closing the pop-up can
        // restore that state (otherwise the user gets dumped back to a
        // collapsed sidebar they didn't ask for).
        sidebarWasOpenBeforePopup = isSidebarOpen();
        popupOverlayEl.hidden = false;
        document.body.classList.add("popup-open");
        syncToggleMode("popup");
    }

    function navigatePopup(url) {
        if (!popupOverlayEl || popupOverlayEl.hidden) return;
        if (!url) return;
        // Swap iframe src + highlight the tab. Does not close the
        // pop-up; used by the bottom tab bar.
        if (popupFrameEl.src.endsWith(url) === false) {
            popupFrameEl.src = url;
        }
        setActiveTab(url);
    }

    function closePopup() {
        if (!popupOverlayEl) return;
        popupOverlayEl.hidden = true;
        popupFrameEl.src = "about:blank";
        document.body.classList.remove("popup-open");
        // Restore the prior sidebar state. If it was open, the mode
        // becomes "sidebar"; if it was closed, "idle".
        if (sidebarWasOpenBeforePopup) {
            syncToggleMode("sidebar");
        } else {
            syncToggleMode("idle");
        }
    }

    function handleToggleClick() {
        if (isPopupOpen()) {
            closePopup();
            return;
        }
        if (isSidebarOpen()) {
            closeSidebar();
        } else {
            openSidebar();
        }
    }

    // Init: set mode based on whatever state the DOM booted with.
    // Desktop default has sidebar open → mode "sidebar"; mobile
    // boots collapsed → mode "idle".
    syncToggleMode(isSidebarOpen() ? "sidebar" : "idle");

    return {
        openPopup, closePopup, navigatePopup,
        openSidebar, closeSidebar,
        handleToggleClick,
    };
})();

if (sidebarToggleBtn) {
    sidebarToggleBtn.addEventListener("click", UIState.handleToggleClick);
}

// Collapsible sidebar sections — any `<div class="sidebar-section">`
// that has `data-collapsible-key="<id>"` becomes click-to-collapse.
// State persists in localStorage under `karin.sidebar.<id>` so the
// user's preference survives reloads. Mobile (<= 720px) collapses
// "panels" by default to save vertical space, since the bottom-nav
// popup tabs already give one-tap access to those same panels.
(function setupCollapsibleSidebarSections() {
    const sections = document.querySelectorAll(
        ".sidebar-section[data-collapsible-key]"
    );
    if (!sections.length) return;

    const isMobile = () =>
        window.matchMedia("(max-width: 720px)").matches;
    const MOBILE_DEFAULT_COLLAPSED = new Set(["panels"]);

    sections.forEach((section) => {
        const key = section.getAttribute("data-collapsible-key");
        const heading = section.querySelector(":scope > .sidebar-heading");
        if (!key || !heading) return;

        // Initial state: stored value wins; otherwise mobile default
        // for "panels"; otherwise expanded.
        let collapsed = false;
        try {
            const stored = window.localStorage.getItem("karin.sidebar." + key);
            if (stored != null) {
                collapsed = stored === "true";
            } else if (isMobile() && MOBILE_DEFAULT_COLLAPSED.has(key)) {
                collapsed = true;
            }
        } catch (_) { /* localStorage unavailable; in-memory only */ }

        if (collapsed) section.classList.add("is-collapsed");

        heading.setAttribute("role", "button");
        heading.setAttribute("tabindex", "0");
        heading.setAttribute("aria-expanded", collapsed ? "false" : "true");

        const toggle = (e) => {
            // Don't fire when the click is inside an input/select/button
            // nested in the heading (e.g. some headings hold a count
            // badge or a small action). Anchor on the heading itself.
            if (e && e.target && e.target.closest("input, select, button")) {
                return;
            }
            const nowCollapsed = section.classList.toggle("is-collapsed");
            heading.setAttribute(
                "aria-expanded", nowCollapsed ? "false" : "true"
            );
            try {
                window.localStorage.setItem(
                    "karin.sidebar." + key,
                    nowCollapsed ? "true" : "false"
                );
            } catch (_) { /* in-memory only */ }
        };

        heading.addEventListener("click", toggle);
        heading.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                toggle();
            }
        });
    });
})();

// Global click delegation: anything with data-panel-target opens the
// pop-up. Tab buttons inside the overlay's own tab bar only navigate
// within the open overlay without closing it.
document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-panel-target]");
    if (!btn) return;
    e.preventDefault();
    const url = btn.getAttribute("data-panel-target");
    const label = btn.getAttribute("data-panel-label") || "";
    if (btn.closest(".popup-overlay-tabs")) {
        UIState.navigatePopup(url);
    } else {
        UIState.openPopup(url, label);
    }
});

// Cross-frame panel navigation: panel iframes can send
//   window.parent.postMessage({type: "karin:navigate-panel", target: "/ui/settings"}, "*")
// to ask the parent to swap the popup overlay to a different panel.
// Use case: in-panel deep links like the "Settings" hint on the alerts
// page. Without this, the link inside the iframe could only navigate
// the iframe itself — leaving the bottom-nav tab stuck on the original
// panel. Same-origin only (parent + iframe both served from the bridge),
// so we don't need the typical origin-allowlist guard.
window.addEventListener("message", (e) => {
    const data = e && e.data;
    if (!data) return;
    if (data.type === "karin:navigate-panel") {
        const target = String(data.target || "");
        if (!target.startsWith("/ui/")) return;  // refuse anything outside the panel namespace
        UIState.navigatePopup(target);
        return;
    }
    if (data.type === "karin:focus-region") {
        // Phase 1 stub: log + dispatch a window-level event so future
        // listeners (alerts proximity scoring, news location filter,
        // weather widget) can react to a region pick on the map.
        // Phase 2 will wire actual behavior — for now this is a hook,
        // not an action.
        const detail = data.detail || {};
        try {
            console.info("karin:focus-region", detail);
            window.dispatchEvent(new CustomEvent("karin:focus-region", { detail }));
        } catch (_) { /* non-fatal */ }
        return;
    }
});

// Escape closes pop-up when focus is inside it.
document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (popupOverlayEl && !popupOverlayEl.hidden) {
        e.preventDefault();
        UIState.closePopup();
    }
});

// ---- Sidebar: memory editor ----------------------------------------------

const memoryHeadingEl = document.getElementById("memory-heading");
const memoryEditorEl = document.getElementById("memory-editor");
const memoryUserEl = document.getElementById("memory-user");
const memoryAgentEl = document.getElementById("memory-agent");
const memoryUserCount = document.getElementById("memory-user-count");
const memoryAgentCount = document.getElementById("memory-agent-count");
const memorySaveBtn = document.getElementById("memory-save");
const memoryStatusEl = document.getElementById("memory-status");
let memoryMaxChars = 1000;

function renderMemoryCount(el, textarea) {
    if (!el || !textarea) return;
    const n = textarea.value.length;
    el.textContent = `${n} / ${memoryMaxChars}`;
    el.classList.toggle("over", n > memoryMaxChars);
}

async function loadMemory() {
    if (!memoryUserEl) return;
    try {
        const r = await fetch("/api/memory");
        if (!r.ok) return;
        const data = await r.json();
        memoryMaxChars = data.max_chars || 1000;
        memoryUserEl.value = data.user || "";
        memoryAgentEl.value = data.agent || "";
        renderMemoryCount(memoryUserCount, memoryUserEl);
        renderMemoryCount(memoryAgentCount, memoryAgentEl);
    } catch (e) {
        console.warn("memory load failed:", e);
    }
}

async function saveMemory() {
    if (!memoryUserEl) return;
    memorySaveBtn.disabled = true;
    memoryStatusEl.textContent = "saving…";
    memoryStatusEl.classList.remove("saved");
    try {
        const r = await fetch("/api/memory", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user: memoryUserEl.value,
                agent: memoryAgentEl.value,
            }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        memoryUserEl.value = data.user || "";
        memoryAgentEl.value = data.agent || "";
        renderMemoryCount(memoryUserCount, memoryUserEl);
        renderMemoryCount(memoryAgentCount, memoryAgentEl);
        const t = data.truncated || {};
        memoryStatusEl.textContent = (t.user || t.agent)
            ? "saved (truncated to cap)"
            : "saved";
        memoryStatusEl.classList.add("saved");
    } catch (e) {
        memoryStatusEl.textContent = `save failed: ${e.message}`;
    } finally {
        memorySaveBtn.disabled = false;
    }
}

if (memoryHeadingEl && memoryEditorEl) {
    const toggleMemory = async () => {
        const expanded = memoryHeadingEl.getAttribute("aria-expanded") === "true";
        const next = !expanded;
        memoryHeadingEl.setAttribute("aria-expanded", String(next));
        memoryEditorEl.hidden = !next;
        if (next) await loadMemory();
    };
    memoryHeadingEl.addEventListener("click", toggleMemory);
    memoryHeadingEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleMemory(); }
    });
}

if (memoryUserEl) {
    memoryUserEl.addEventListener("input", () => renderMemoryCount(memoryUserCount, memoryUserEl));
}
if (memoryAgentEl) {
    memoryAgentEl.addEventListener("input", () => renderMemoryCount(memoryAgentCount, memoryAgentEl));
}
if (memorySaveBtn) {
    memorySaveBtn.addEventListener("click", saveMemory);
}

// ---- History restore ------------------------------------------------------

function renderHistoricalTurn(entry, conversationId) {
    // Render a saved turn (no streaming, no replay audio). Tool events
    // are replayed as widgets so reopening a conversation shows the
    // same panels the user saw live. These widgets are UI-only — the
    // bridge strips tool messages from the LLM's context on new turns
    // (see _llm_visible_history in bridge/llm.py).
    if (entry.role === "summary") {
        const summary = document.createElement("div");
        summary.className = "summary-note";
        summary.textContent = entry.content;
        transcriptEl.appendChild(summary);
        return;
    }
    const turn = startTurn(entry.user || "");
    // Replay tool events BEFORE the assistant bubble so widgets sit above
    // the reply (matching the live streaming order).
    for (const te of (entry.tool_events || [])) {
        turnAddToolCall(turn, te.name, te.arguments || {}, te.result || "");
        mountPanelInTurn(turn, te.name, te.arguments || {}, te.result || "");
    }
    const asstBubble = turnFinalizeAssistant(turn, entry.assistant || "");
    // Rehydrate the replay button if this turn has cached audio.
    if (entry.audio_id && conversationId && asstBubble) {
        rehydrateReplay(asstBubble, entry.assistant || "", conversationId, entry.audio_id);
    }
}

/**
 * Wire a replay button for a restored turn. We let the <audio> element
 * itself probe: ``loadedmetadata`` fires on success, ``error`` fires
 * on 404 (cache evicted — typical after a container restart since
 * audio_cache is tmpfs). No separate HEAD request needed.
 *
 * The button is only appended on success, so restored turns whose
 * audio is gone just show no replay button — same UX as "never
 * synthesized" turns.
 */
function rehydrateReplay(bubble, replyText, conversationId, audioId) {
    const url = `/api/audio/${encodeURIComponent(conversationId)}/${encodeURIComponent(audioId)}`;
    const audio = new Audio();
    audio.preload = "metadata";
    audio.addEventListener("loadedmetadata", () => {
        const durationMs = (audio.duration || 0) * 1000;
        let schedule = null;
        if (window.Face && replyText && durationMs > 0) {
            schedule = window.Face.buildSchedule(replyText, durationMs);
        }
        addReplayButton(bubble, audio, schedule);
    }, { once: true });
    audio.addEventListener("error", () => {
        // 404 / decode failure — stay silent; the button just doesn't
        // appear for this turn. Nothing to recover.
    }, { once: true });
    audio.src = url;  // triggers the probe
}

async function restoreHistory() {
    try {
        const r = await fetch("/api/history");
        if (!r.ok) return;
        const data = await r.json();
        const cid = data.conversation_id;
        for (const t of (data.turns || [])) renderHistoricalTurn(t, cid);
        requestAnimationFrame(() => scrollToBottom());
    } catch (e) {
        console.warn("history restore failed:", e);
    }
}

// ---- New chat -------------------------------------------------------------

async function startNewChat() {
    if (isProcessing) return;
    let reused = false;
    try {
        const r = await fetch("/api/history/new", { method: "POST" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        try {
            const data = await r.json();
            reused = !!data.reused;
        } catch { /* old-style response — treat as new */ }
    } catch (e) {
        setStatus(`New chat failed: ${e.message}`, true);
        return;
    }
    if (reused) {
        // Current conversation was already empty — server handed the
        // same id back. Nothing to do; leaves the sidebar alone so
        // repeat clicks don't stack empty placeholders.
        return;
    }
    // Stop any in-flight TTS, drop any queued typing, clear the
    // transcript, close any open panel.
    stopStreamPlaybackNow();
    pendingTexts.length = 0;
    transcriptEl.innerHTML = "";
    closePanel();
    setStatus("");
    if (face) face.idle();
    await loadConversationList();
}

if (newChatBtn) {
    newChatBtn.addEventListener("click", startNewChat);
}

// ---- Voice (sovits) availability indicator -------------------------------
//
// Read-only. Polls /api/voice/status every 20 s and on page load to
// reflect the sovits container's actual state. To start/stop sovits,
// SSH in and run `docker compose start|stop sovits` (the toggle is
// intentionally not in the UI — on tight memory it's easy to hang
// sovits if the LLM is still resident, and SSH encourages thinking
// before flipping it).

const ttsStatusEl = document.getElementById("tts-status");
const sttStatusEl = document.getElementById("stt-status");
// Legacy compat: code that referenced voiceStatusEl still works.
const voiceStatusEl = ttsStatusEl;

function _setIndicator(el, state, label, title) {
    if (!el) return;
    el.classList.remove("off", "starting", "on", "error");
    el.classList.add(state);
    const labelEl = el.querySelector(".voice-label");
    if (labelEl) labelEl.textContent = label;
    el.title = title;
}

function _voiceSetState(state, healthOrErr) {
    const labels = {
        off:      "TTS off",
        starting: "TTS starting...",
        on:       "TTS on",
        error:    "TTS error",
    };
    const titles = {
        off:      "Text-to-speech is off. Enable via KARIN_TTS_ENABLED=true.",
        starting: "TTS server starting — loading models.",
        on:       "TTS ready — spoken replies will play.",
        error:    `TTS unavailable: ${healthOrErr || "check sovits / PC server"}`,
    };
    _setIndicator(ttsStatusEl, state, labels[state] || "TTS", titles[state] || "");
}

function _voiceApplyStatus(s) {
    // Capability flag wins: TTS_ENABLED on the backend is the source
    // of truth for whether spoken replies will be produced. The local
    // sovits container status is advisory (and irrelevant when the
    // PC-offload path is in use — there's no local container to poll).
    const capsTtsOn = !!(window.KARIN_CAPS && window.KARIN_CAPS.tts);
    if (!capsTtsOn) { _voiceSetState("off"); return; }
    if (!s) { _voiceSetState("on"); return; }  // caps say yes, no container data
    if (s.status === "disabled") { _voiceSetState("off"); return; }
    // Remote PC sidecar path: server probed /docs and reports whether it's
    // reachable. Show a dedicated error state so the user knows the mic
    // won't work BEFORE they try.
    if (s.backend === "remote") {
        if (s.remote_reachable === false) {
            _voiceSetState(
                "error",
                `PC sidecar offline (${s.remote_url || "?"})`,
            );
            return;
        }
        _voiceSetState("on", "remote");
        return;
    }
    if (s.error && !s.exists) { _voiceSetState("error", s.error); return; }
    if (s.running === false)   { _voiceSetState("on"); return; }  // remote TTS: no local container
    if (s.health === "starting") { _voiceSetState("starting"); return; }
    _voiceSetState("on", s.health);
}

async function _pollCapabilitiesAndStatus() {
    // Re-fetch BOTH capabilities and voice/status on every tick so the
    // indicator tracks backend state after env-flag flips + container
    // restarts. Without this the indicator is frozen at page-load time:
    // flipping KARIN_TTS_ENABLED on/off wouldn't update the dot until
    // the user reloads.
    try {
        const r = await fetch("/api/features");
        if (r.ok) {
            const caps = await r.json();
            applyCapabilities(caps);  // re-applies indicator state too
        }
    } catch (e) {
        // Keep whatever caps we had; next tick will retry.
    }
    try {
        const r = await fetch("/api/voice/status");
        if (r.ok) {
            const s = await r.json();
            window.KARIN_LAST_VOICE_STATUS = s;
            _voiceApplyStatus(s);
        }
    } catch (e) {
        console.warn("voice status fetch failed:", e);
    }
    // STT runtime-flag poll. applyCapabilities() already set the dot
    // based on boot capability; this overrides it if the operator
    // toggled the runtime flag OR if the remote sidecar is unreachable.
    try {
        const r = await fetch("/api/stt/status", { cache: "no-store" });
        if (r.ok) {
            const s = await r.json();
            window.KARIN_LAST_STT_STATUS = s;
            // Remote path + unreachable wins over everything else —
            // the user needs to know BEFORE they try to transcribe.
            if (s.backend === "remote" && s.remote_reachable === false) {
                _setIndicator(
                    sttStatusEl, "error", "STT PC offline",
                    `Sidecar at ${s.remote_url || "?"} isn't responding. ` +
                    "Start the PC-side voice server (deploy/pc-tts/start.bat) or " +
                    "check Tailscale on both ends.",
                );
            } else if (s.boot_enabled && !s.runtime_enabled) {
                _setIndicator(
                    sttStatusEl, "off", "STT paused",
                    "STT loaded but paused at runtime — click to re-enable.",
                );
            }
            // Otherwise applyCapabilities has already set the dot
            // correctly (on if boot+runtime enabled, off if not).
        }
    } catch (e) {
        // Non-fatal; next tick retries.
    }
}

if (voiceStatusEl) {
    // Start polling unconditionally — even if TTS was off at page load,
    // the user might flip it on from the Jetson, and we want the dot
    // to light up on the next tick without a refresh.
    setTimeout(() => {
        _pollCapabilitiesAndStatus();
        setInterval(_pollCapabilitiesAndStatus, 20000);
    }, 500);
}


// ---- STT toggle (sidebar #stt-status dot is clickable) -------------------
//
// The STT dot shows runtime state (on/off). Clicking toggles the runtime
// flag via /api/stt/{enable,disable}. Unlike TTS (whose Docker container
// is intentionally SSH-only because starting it while the LLM is warm
// can hang), STT runs in-process and flipping it is cheap — the Whisper
// model stays resident either way, the flag just gates transcription.
//
// boot-disabled STT (KARIN_STT_ENABLED=false at startup) can't be
// toggled on from the UI because the model was never loaded. The server
// returns 503; the click handler surfaces that as a status message.

async function _sttToggle() {
    if (!sttStatusEl) return;
    // Read current boot + runtime state from the server rather than
    // trusting the indicator class (which can lag by one poll tick).
    let status;
    try {
        const r = await fetch("/api/stt/status", { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        status = await r.json();
    } catch (e) {
        setStatus(`STT status check failed: ${e && e.message || e}`, true);
        return;
    }
    if (!status.boot_enabled) {
        setStatus(
            "STT is disabled at boot. Set KARIN_STT_ENABLED=true in deploy/.env and restart the service.",
            true,
        );
        return;
    }
    const action = status.runtime_enabled ? "disable" : "enable";
    sttStatusEl.style.pointerEvents = "none";
    try {
        const r = await fetch(`/api/stt/${action}`, { method: "POST" });
        if (!r.ok) {
            let detail = `HTTP ${r.status}`;
            try {
                const body = await r.json();
                if (body && body.detail) detail = String(body.detail);
            } catch (_) { /* non-JSON body */ }
            setStatus(`STT ${action} failed: ${detail}`, true);
        } else {
            // Force an immediate capabilities + status refresh so the
            // dot reflects the new state without waiting for the 20s tick.
            await _pollCapabilitiesAndStatus();
            setStatus(`STT ${action}d`, false);
        }
    } catch (e) {
        setStatus(`STT ${action} failed: ${e && e.message || e}`, true);
    } finally {
        sttStatusEl.style.pointerEvents = "";
    }
}

if (sttStatusEl) {
    sttStatusEl.addEventListener("click", _sttToggle);
    sttStatusEl.style.cursor = "pointer";
    // Update hover tooltip to hint that it's clickable (the off / on text
    // in applyCapabilities() already covers the *current* state; this is
    // an affordance hint).
    const origTitle = sttStatusEl.title || "";
    sttStatusEl.title = origTitle + (origTitle ? " — " : "") + "click to toggle";
}



// ---- Server feature flags -------------------------------------------------
//
// The server exposes /api/features (full registry) and /api/capabilities
// (legacy 2-bool subset). We prefer /api/features so new code can read
// bandit + holiday flags too, but fall back to /api/capabilities if
// talking to an older server. Flags are published on window.KARIN_CAPS
// for other modules (face.js, holidays.js, stt_web.js).

window.KARIN_CAPS = { stt: true, tts: true };  // optimistic default

function applyCapabilities(caps) {
    // Normalize the two endpoint shapes. /api/features returns
    // {subsystems: {stt: bool, tts: bool, bandit: bool, ...}}; the
    // legacy /api/capabilities returns {stt: bool, tts: bool}.
    const subs = caps.subsystems || {};
    const sttVal = subs.stt !== undefined ? subs.stt : caps.stt;
    const ttsVal = subs.tts !== undefined ? subs.tts : caps.tts;
    window.KARIN_CAPS = {
        stt: sttVal !== false,
        tts: ttsVal !== false,
        subsystems: subs,
        tools: caps.tools || { disabled: [] },
    };
    const ttsOn = window.KARIN_CAPS.tts;
    const sttOn = window.KARIN_CAPS.stt;

    // /api/features only knows the boot flag — it doesn't know if the
    // remote sidecar is actually reachable. Cross-reference the latest
    // status fetched by _pollCapabilitiesAndStatus so we don't show
    // "on" while the user's PC is offline. Without this peek the dot
    // flickers through "on" for ~one polling-cycle every 20 s.
    const lastVoice = window.KARIN_LAST_VOICE_STATUS;
    const lastStt = window.KARIN_LAST_STT_STATUS;
    const ttsRemoteDown = lastVoice && lastVoice.backend === "remote"
        && lastVoice.remote_reachable === false;
    const sttRemoteDown = lastStt && lastStt.backend === "remote"
        && lastStt.remote_reachable === false;

    // Set the two sidebar indicators — always visible so the user
    // can see at a glance what's on and what's off.
    if (ttsOn && ttsRemoteDown) {
        _setIndicator(
            ttsStatusEl, "error", "TTS PC offline",
            `Sidecar at ${lastVoice.remote_url || "?"} isn't responding. ` +
            "Start the PC voice server or disable TTS.",
        );
    } else {
        _setIndicator(
            ttsStatusEl,
            ttsOn ? "on" : "off",
            ttsOn ? "TTS on" : "TTS off",
            ttsOn
                ? "Text-to-speech enabled -- replies have voice."
                : "Text-to-speech off. Enable via KARIN_TTS_ENABLED=true.",
        );
    }
    if (sttOn && sttRemoteDown) {
        _setIndicator(
            sttStatusEl, "error", "STT PC offline",
            `Sidecar at ${lastStt.remote_url || "?"} isn't responding. ` +
            "Start the PC voice server or disable STT.",
        );
    } else {
        _setIndicator(
            sttStatusEl,
            sttOn ? "on" : "off",
            sttOn ? "STT on" : "STT off",
            sttOn
                ? "Speech-to-text enabled -- you can talk to Karin."
                : "Speech-to-text off. Enable via KARIN_STT_ENABLED=true.",
        );
    }

    // Hide the PTT button wrap when STT is off.
    if (!sttOn && pttWrap) pttWrap.hidden = true;

    // Status-line hint when both are off.
    if (!sttOn && !ttsOn) setStatus("Text-only mode", false);

    // Hide holiday banner 🔊 button when TTS is off.
    if (!ttsOn) {
        const hp = document.querySelector(".holiday-play");
        if (hp) hp.hidden = true;
    }

    // Sync the voice auto-play toggle visibility.
    if (typeof syncAutoPlayButton === "function") syncAutoPlayButton();
}

// ---- Startup --------------------------------------------------------------

setButtonState("loading");
setStatus("Connecting...");
Promise.all([
    fetch("/api/health").then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json();
    }),
    // Prefer the richer /api/features; fall back to the legacy
    // /api/capabilities if the server predates the registry. Either
    // way we apply through applyCapabilities, which normalizes shape.
    fetch("/api/features")
        .then((r) => r.ok ? r.json() : null)
        .catch(() => null)
        .then((f) => f || fetch("/api/capabilities")
            .then((r) => r.ok ? r.json() : { stt: true, tts: true })
            .catch(() => ({ stt: true, tts: true }))),
    // Pre-fetch the voice + STT status so applyCapabilities below
    // sees remote_reachable on its FIRST call. Without this the dot
    // flashes "on" on page load before the polling cycle gets around
    // to overriding it. Cheap (two GETs against the local bridge);
    // the polls 500ms later still pick up subsequent changes.
    fetch("/api/voice/status").then((r) => r.ok ? r.json() : null).catch(() => null),
    fetch("/api/stt/status", { cache: "no-store" }).then((r) => r.ok ? r.json() : null).catch(() => null),
])
    .then(async ([d, caps, voiceStatus, sttStatus]) => {
        if (!d.ok) throw new Error("not ready");
        if (voiceStatus) window.KARIN_LAST_VOICE_STATUS = voiceStatus;
        if (sttStatus) window.KARIN_LAST_STT_STATUS = sttStatus;
        applyCapabilities(caps);
        setButtonState("idle");
        if (!(caps.stt === false && caps.tts === false)) setStatus("");
        await Promise.all([restoreHistory(), loadConversationList()]);
    })
    .catch((e) => setStatus(`Server unreachable: ${e.message}`, true));
