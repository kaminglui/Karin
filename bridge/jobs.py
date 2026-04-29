"""Background turn-job queue.

Each LLM turn is run as a JOB on a worker thread, with state persisted
to disk. The browser fires "start a turn" and gets back a job_id; it
then polls or streams events from "/api/turn/<id>". Closing the browser
in the middle of a turn no longer kills the work — when the user comes
back, the job either completed (replay events from disk) or is still
running (resume the stream).

Crucially this also fixes the failure mode where a 60-90s turn on the
Jetson exceeds the browser's idle/sleep timeout: the server keeps
chewing on the turn even when nobody's listening.

Design:
- One Job per turn. Identified by a UUID.
- Events: tool_call, transcript, audio, error, done. Same NDJSON shape
  the existing streaming endpoints emit.
- Job state lives in `data/jobs/<id>.json` for crash-resume; events
  also accumulate in-memory for live streaming.
- Worker pool size = 1 (matches pipeline_lock — turns serialize
  through the LLM/STT/TTS pipeline anyway).
- Old jobs cleaned up after `JOB_TTL_SECONDS` since last touch.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from bridge.utils import REPO_ROOT, atomic_write_text

log = logging.getLogger("bridge.jobs")

JOBS_DIR: Path = REPO_ROOT / "data" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# How long a finished job's state stays on disk + in memory before GC.
JOB_TTL_SECONDS: int = 60 * 60  # 1 hour


@dataclass
class Job:
    id: str
    created_at: float
    status: str = "pending"   # pending | running | done | failed | cancelled
    events: list[dict] = field(default_factory=list)
    error: str | None = None
    finished_at: float | None = None
    # threading.Event-like — set when status flips to done/failed/cancelled.
    _done_evt: threading.Event = field(default_factory=threading.Event)
    # Each new event sets this so subscribers can wait for fresh data.
    _new_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append_event(self, evt: dict) -> None:
        with self._lock:
            self.events.append(evt)
            self._new_event.set()

    def mark_status(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            if error is not None:
                self.error = error
            if status in ("done", "failed", "cancelled"):
                self.finished_at = time.time()
                self._done_evt.set()
                self._new_event.set()  # wake any pollers

    def to_disk(self, path: Path) -> None:
        # Persist job state — drop the threading objects so the JSON
        # round-trips cleanly. Crash-recovery just reads the events.
        with self._lock:
            data = {
                "id": self.id,
                "created_at": self.created_at,
                "status": self.status,
                "events": list(self.events),
                "error": self.error,
                "finished_at": self.finished_at,
            }
        atomic_write_text(path, json.dumps(data, ensure_ascii=False))


class JobStore:
    """In-memory + disk-backed registry of active and recent jobs."""

    def __init__(self, root: Path = JOBS_DIR) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, created_at=time.time())
        with self._lock:
            self._jobs[job_id] = job
        log.info("job %s: created", job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            j = self._jobs.get(job_id)
        if j is not None:
            return j
        # Fallback: try loading a finished job from disk.
        path = self.root / f"{job_id}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rehydrated = Job(
                    id=data["id"],
                    created_at=data["created_at"],
                    status=data.get("status", "done"),
                    events=data.get("events", []),
                    error=data.get("error"),
                    finished_at=data.get("finished_at"),
                )
                rehydrated._done_evt.set()
                with self._lock:
                    self._jobs[job_id] = rehydrated
                return rehydrated
            except Exception as e:
                log.warning("job %s: failed to reload from disk: %s", job_id, e)
        return None

    def persist(self, job: Job) -> None:
        try:
            job.to_disk(self.root / f"{job.id}.json")
        except Exception as e:
            log.warning("job %s: persist failed: %s", job.id, e)

    def gc(self) -> None:
        """Drop in-memory jobs older than TTL. Disk files outlive memory
        so a returning client can still fetch results, but in-memory
        we don't keep them forever."""
        now = time.time()
        with self._lock:
            stale = [
                jid for jid, j in self._jobs.items()
                if j.finished_at and (now - j.finished_at) > JOB_TTL_SECONDS
            ]
            for jid in stale:
                del self._jobs[jid]
        if stale:
            log.info("gc'd %d stale jobs", len(stale))


# Worker thread that runs turns. Single-threaded by design — the LLM
# pipeline serializes anyway, and one thread keeps reasoning simple.
class TurnWorker:
    """Runs jobs one at a time off an internal queue."""

    def __init__(self, store: JobStore) -> None:
        self.store = store
        self._queue: list[tuple[Job, Callable[[Job], None]]] = []
        self._cv = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, job: Job, runner: Callable[[Job], None]) -> None:
        """Enqueue a job. ``runner`` is the synchronous function that
        executes the turn — it should call ``job.append_event(...)`` as
        events happen and ``job.mark_status('done')`` at the end."""
        with self._cv:
            self._queue.append((job, runner))
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                job, runner = self._queue.pop(0)
            job.mark_status("running")
            try:
                runner(job)
                if job.status == "running":
                    job.mark_status("done")
            except Exception as e:
                log.exception("job %s: runner raised", job.id)
                job.mark_status("failed", error=str(e))
            finally:
                self.store.persist(job)
                self.store.gc()
