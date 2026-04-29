"""Tests for bridge.jobs — background turn-job queue.

Covers the contract web/server.py relies on:
  - Job id uniqueness + disk persistence round-trip
  - append_event ordering + _new_event signalling for streaming clients
  - Worker runs jobs, marks them done, handles runner exceptions
  - Store.get falls back to disk for finished jobs no longer in memory
"""
from __future__ import annotations

import threading
import time

import pytest

from bridge.jobs import JobStore, TurnWorker


@pytest.fixture
def store(tmp_path):
    return JobStore(root=tmp_path)


# --- Job lifecycle ----------------------------------------------------------

class TestJobLifecycle:
    def test_create_assigns_unique_ids(self, store):
        j1 = store.create()
        j2 = store.create()
        assert j1.id != j2.id
        assert j1.status == "pending"

    def test_append_event_maintains_order(self, store):
        j = store.create()
        j.append_event({"type": "a"})
        j.append_event({"type": "b"})
        j.append_event({"type": "c"})
        assert [e["type"] for e in j.events] == ["a", "b", "c"]

    def test_mark_status_sets_done_event(self, store):
        j = store.create()
        assert not j._done_evt.is_set()
        j.mark_status("done")
        assert j._done_evt.is_set()
        assert j.finished_at is not None

    def test_mark_status_records_error(self, store):
        j = store.create()
        j.mark_status("failed", error="boom")
        assert j.status == "failed"
        assert j.error == "boom"


# --- Streaming signalling (HTTP handler uses _new_event) --------------------

class TestStreamSignal:
    def test_new_event_fires_on_append(self, store):
        j = store.create()
        # Clear initial state so the test captures append() setting it.
        j._new_event.clear()
        assert not j._new_event.is_set()
        j.append_event({"type": "tool_call", "name": "get_weather"})
        assert j._new_event.is_set()

    def test_new_event_also_fires_on_mark_done(self, store):
        j = store.create()
        j._new_event.clear()
        j.mark_status("done")
        # Pollers waiting on _new_event should wake so they can exit.
        assert j._new_event.is_set()

    def test_new_event_wakes_waiting_thread(self, store):
        # Simulates the HTTP streaming handler waiting for new events;
        # a producer on another thread appends one and the waiter wakes.
        j = store.create()
        j._new_event.clear()
        woke = []

        def waiter():
            if j._new_event.wait(timeout=2.0):
                woke.append("yes")

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)
        j.append_event({"type": "token_delta", "delta": "Hi"})
        t.join(timeout=3.0)
        assert woke == ["yes"], "waiting thread did not wake"


# --- Disk persistence -------------------------------------------------------

class TestPersistence:
    def test_persist_then_reload_via_get(self, store):
        j = store.create()
        j.append_event({"type": "transcript", "user": "hi", "assistant": "hey"})
        j.mark_status("done")
        store.persist(j)

        # Forget the in-memory copy to force disk reload.
        with store._lock:
            store._jobs.clear()

        rehydrated = store.get(j.id)
        assert rehydrated is not None
        assert rehydrated.id == j.id
        assert rehydrated.status == "done"
        assert rehydrated.events[0]["type"] == "transcript"
        # Rehydrated jobs have the _done_evt pre-set so streams don't block.
        assert rehydrated._done_evt.is_set()

    def test_get_unknown_returns_none(self, store):
        assert store.get("never_existed_abc123") is None

    def test_persist_does_not_crash_on_threading_objects(self, store):
        # Job has threading.Lock + Event attrs that JSON can't serialize;
        # to_disk must filter them.
        j = store.create()
        j.append_event({"type": "user_text", "text": "hi"})
        j.mark_status("done")
        store.persist(j)   # would raise if it tried to serialize the Lock
        assert (store.root / f"{j.id}.json").exists()


# --- Worker ------------------------------------------------------------------

class TestTurnWorker:
    def test_worker_runs_job(self, store):
        worker = TurnWorker(store)
        j = store.create()
        received_job = threading.Event()

        def runner(job):
            job.append_event({"type": "hello"})
            received_job.set()

        worker.submit(j, runner)
        assert received_job.wait(timeout=5.0), "worker never picked up the job"
        # Give the worker a moment to mark status done.
        j._done_evt.wait(timeout=3.0)
        assert j.status == "done"
        assert any(e["type"] == "hello" for e in j.events)

    def test_worker_marks_failed_on_exception(self, store):
        worker = TurnWorker(store)
        j = store.create()

        def bad_runner(job):
            raise RuntimeError("kaboom")

        worker.submit(j, bad_runner)
        j._done_evt.wait(timeout=5.0)
        assert j.status == "failed"
        assert "kaboom" in (j.error or "")

    def test_worker_runs_jobs_serially(self, store):
        worker = TurnWorker(store)
        order = []
        start_barriers = []

        def runner_factory(label):
            evt = threading.Event()
            start_barriers.append(evt)
            def runner(job):
                order.append(f"{label}-start")
                evt.wait(timeout=3.0)
                order.append(f"{label}-end")
            return runner

        r1 = runner_factory("a")
        r2 = runner_factory("b")
        j1 = store.create(); j2 = store.create()
        worker.submit(j1, r1)
        worker.submit(j2, r2)

        # Let the first job START but not finish yet.
        time.sleep(0.2)
        assert order == ["a-start"]
        # Release first → second begins only after first ends.
        start_barriers[0].set()
        j1._done_evt.wait(timeout=3.0)
        time.sleep(0.1)
        assert order[:3] == ["a-start", "a-end", "b-start"]
        start_barriers[1].set()
        j2._done_evt.wait(timeout=3.0)


# --- GC ---------------------------------------------------------------------

class TestGC:
    def test_gc_drops_old_finished_jobs(self, store, monkeypatch):
        # Shrink TTL so we can actually hit the threshold.
        from bridge import jobs as jobs_mod
        monkeypatch.setattr(jobs_mod, "JOB_TTL_SECONDS", 0)

        j = store.create()
        j.mark_status("done")
        j.finished_at = time.time() - 1   # finished 1s ago, TTL is 0
        store.gc()
        with store._lock:
            assert j.id not in store._jobs

    def test_gc_keeps_running_jobs(self, store, monkeypatch):
        from bridge import jobs as jobs_mod
        monkeypatch.setattr(jobs_mod, "JOB_TTL_SECONDS", 0)
        j = store.create()
        # Not marked done; should NOT be gc'd even with 0 TTL.
        store.gc()
        with store._lock:
            assert j.id in store._jobs
