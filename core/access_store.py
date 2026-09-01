"""
Invite-code access control for the public demo frontend (Phase 1).

Purpose: let known Upwork clients try the pipeline for free via a
one-time code, without opening job submission to anyone on the internet
with no gate at all. This is deliberately simple - single-use random
codes, in-memory, no accounts - matching the same scope philosophy as
JobStore (core/job_store.py): good enough for a single-operator demo,
documented as not being more than that.

Not yet wired to payment. Phase 2 adds a Stripe-verified path so a
public visitor without a code can pay for one run instead of needing an
invite.

Thread-safe for the same reason as JobStore - FastAPI request handling
can touch this from more than one thread.
"""
from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _generate_code() -> str:
    # 5 groups of 4 uppercase alnum chars, easy to read aloud/type,
    # e.g. "AB12-CD34-EF56-GH78-IJ90". Excludes 0/O/1/I to avoid
    # ambiguity when a client types it in by hand.
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    groups = ["".join(secrets.choice(alphabet) for _ in range(4)) for _ in range(3)]
    return "-".join(groups)


@dataclass
class InviteCode:
    code: str
    label: str | None = None  # e.g. "client: acme corp upwork proposal"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    used_at: str | None = None
    used_by_job_id: str | None = None

    @property
    def is_used(self) -> bool:
        return self.used_at is not None


class AccessStore:
    def __init__(self) -> None:
        self._codes: dict[str, InviteCode] = {}
        self._lock = threading.Lock()

    def generate(self, label: str | None = None) -> InviteCode:
        with self._lock:
            code = _generate_code()
            while code in self._codes:  # astronomically unlikely, but don't silently collide
                code = _generate_code()
            invite = InviteCode(code=code, label=label)
            self._codes[code] = invite
            return invite

    def validate(self, code: str) -> InviteCode | None:
        """
        Returns the InviteCode if it exists and is unused, else None.
        Does NOT mark it used - call redeem() only after the job it
        gates has actually been accepted, so a validation check alone
        never burns a client's one shot.
        """
        with self._lock:
            invite = self._codes.get(code.strip().upper())
            if invite is None or invite.is_used:
                return None
            return invite

    def redeem(self, code: str, job_id: str) -> bool:
        """Marks a code used, tied to the job it unlocked. Returns False if the code was invalid/already used - caller must check this before proceeding."""
        with self._lock:
            invite = self._codes.get(code.strip().upper())
            if invite is None or invite.is_used:
                return False
            invite.used_at = datetime.now(timezone.utc).isoformat()
            invite.used_by_job_id = job_id
            return True

    def set_job_id(self, code: str, job_id: str) -> None:
        """Updates the audit-trail job_id after the real job is created (redeem() is called with a "pending" placeholder before the job exists - see api/demo.py)."""
        with self._lock:
            invite = self._codes.get(code.strip().upper())
            if invite is not None:
                invite.used_by_job_id = job_id

    def release(self, code: str) -> None:
        """
        Un-redeems a code - used only when redeem() succeeded but the
        upload that followed was then rejected (bad file type, oversized
        file) before any real pipeline work started, so a client doesn't
        lose their one free run to a fixable mistake. Never called once
        a job has actually started, even if it later fails for other
        reasons.
        """
        with self._lock:
            invite = self._codes.get(code.strip().upper())
            if invite is not None:
                invite.used_at = None
                invite.used_by_job_id = None

    def list_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "code": c.code,
                    "label": c.label,
                    "created_at": c.created_at,
                    "used_at": c.used_at,
                    "used_by_job_id": c.used_by_job_id,
                }
                for c in self._codes.values()
            ]


access_store = AccessStore()
