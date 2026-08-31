"""
In-memory job store for the FastAPI orchestration layer.

Deliberately NOT Supabase/Postgres-backed yet. The build spec's full tech
stack calls for Supabase job metadata storage, but Day 4 scope is "a
working end-to-end pipeline" with FastAPI submit/status/result endpoints,
not the production storage layer - and adding a database dependency mid
build wasn't part of what Day 4 asked for. This is a documented,
intentional scope cut, not an oversight: swap this module for a
Supabase-backed store when persistence-across-restarts or multi-instance
deployment becomes a real requirement. The JobStore interface below is
the seam to do that at - callers only use create/get/update/to_dict.

Thread-safe because FastAPI BackgroundTasks run in the server's
threadpool, not necessarily the same thread that handled the initiating
request.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from typing import Any


@dataclass
class Job:
    job_id: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    result: dict[str, Any] | None = None
    error: str | None = None
    error_stage: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(job_id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> Job | None:
        """
        Raises TypeError immediately (before touching the job) if a kwarg
        doesn't name a real Job field - a typo'd field name used to be
        silently accepted by plain setattr and would create an unused
        attribute instead of erroring, since Job is a dataclass, not a
        slotted class. Caught in Day 4 code review, fixed here in Day 6.
        """
        valid_fields = {f.name for f in dataclass_fields(Job)}
        unknown = set(fields) - valid_fields
        if unknown:
            raise TypeError(f"JobStore.update got unknown field(s): {sorted(unknown)}")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = datetime.now(timezone.utc).isoformat()
            return job

    def delete(self, job_id: str) -> None:
        """
        Remove a job record outright - used only to clean up a job that
        never actually started (e.g. its upload was rejected before the
        background pipeline task was ever scheduled). Not part of the
        normal job lifecycle; a job that started running is never deleted,
        only updated to a terminal status.
        """
        with self._lock:
            self._jobs.pop(job_id, None)

    def to_dict(self, job: Job) -> dict[str, Any]:
        return asdict(job)


job_store = JobStore()
