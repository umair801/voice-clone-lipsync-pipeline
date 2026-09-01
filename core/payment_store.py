"""
Single-use enforcement for paid demo runs (Phase 2).

Stripe's own Checkout Session status (payment_status == "paid") is the
source of truth for whether a session was actually paid - this store
does NOT duplicate that. What it adds is something Stripe doesn't give
you for free: a paid Checkout Session, on its own, can be *checked* as
many times as you want (nothing stops a visitor from reloading the
success page and reusing the same session_id to submit another job for
the same one payment). This store tracks which paid sessions have
already been redeemed for a pipeline run, so one payment buys exactly
one run - same one-time-use shape as core/access_store.py's invite
codes, and deliberately kept as a separate, simpler store rather than
merged with it, since a paid session and an invite code have different
lifecycles (a session is verified against Stripe's API, a code is
verified against nothing but itself).

In-memory, thread-safe, same scope-cut reasoning as JobStore/AccessStore:
resets on restart, fine for the current single-operator deployment.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RedeemedSession:
    session_id: str
    redeemed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    job_id: str | None = None


class PaymentStore:
    def __init__(self) -> None:
        self._redeemed: dict[str, RedeemedSession] = {}
        self._lock = threading.Lock()

    def is_redeemed(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._redeemed

    def redeem(self, session_id: str, job_id: str) -> bool:
        """Returns False if this session was already redeemed (caller must treat that as a rejection, not overwrite it)."""
        with self._lock:
            if session_id in self._redeemed:
                return False
            self._redeemed[session_id] = RedeemedSession(session_id=session_id, job_id=job_id)
            return True

    def set_job_id(self, session_id: str, job_id: str) -> None:
        """Updates the audit-trail job_id after the real job is created (redeem() is called with a "pending" placeholder before the job exists - see api/demo.py)."""
        with self._lock:
            record = self._redeemed.get(session_id)
            if record is not None:
                record.job_id = job_id

    def release(self, session_id: str) -> None:
        """
        Un-redeems a session - used only when redeem() succeeded but the
        upload that followed was then rejected (bad file type, oversized
        file) before any real pipeline work started. Without this, a
        visitor who fat-fingers a wrong file type would burn their one
        paid run on a 400 they never got to fix. Never called after a
        job actually started running, even if that job later fails - a
        real pipeline attempt was made, so the payment was used for what
        it was for.
        """
        with self._lock:
            self._redeemed.pop(session_id, None)


payment_store = PaymentStore()
