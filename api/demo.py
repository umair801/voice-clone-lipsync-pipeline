"""
Public demo frontend API.

POST /demo/jobs               submit a job, gated by EITHER a one-time
                               invite code (X-Invite-Code) OR a paid
                               Stripe Checkout session (X-Payment-Session).
                               Exactly one access path is required per
                               request.
GET  /demo/admin/codes        list all issued codes (admin key required).
POST /demo/admin/codes        generate a new one-time invite code (admin
                               key required) - how Umair issues a free
                               code to an Upwork client.
POST /demo/checkout           starts a Stripe Checkout session for one
                               paid demo run (Phase 2) - returns the URL
                               the frontend redirects the browser to.
POST /demo/stripe/webhook     Stripe webhook receiver (Phase 2) - audit
                               logging only, does NOT gate access (see
                               core/payments.py for why).

Kept as a separate router from api/routes.py on purpose: the internal
/jobs endpoint's trust model (a shared API key, meant for Umair's own
tooling) is different from this one (single-use codes and one-time
payments handed to external, unauthenticated visitors). Mixing the two
gates on one endpoint would make it easy to accidentally weaken one
while changing the other.
"""
from __future__ import annotations

import secrets

from fastapi import APIRouter, BackgroundTasks, File, Header, HTTPException, Request, UploadFile

from api.schemas import (
    CheckoutSessionResponse,
    InviteCodeCreateRequest,
    InviteCodeListResponse,
    InviteCodeResponse,
    JobSubmitResponse,
)
from core.access_store import access_store
from core.config import settings
from core.job_runner import accept_upload_job
from core.logging import get_logger
from core.payment_store import payment_store
from core.payments import PaymentAPIError, PaymentConfigError, create_checkout_session, session_is_paid, verify_webhook_signature

logger = get_logger(__name__)
router = APIRouter(prefix="/demo")


def _require_admin_key(x_admin_key: str | None) -> None:
    """
    Refuses every request when DEMO_ADMIN_KEY isn't set, rather than
    treating "unset" as "open" - the opposite default from
    PIPELINE_API_KEY on /jobs. Code generation should never be reachable
    by accident on a fresh deploy, since a leaked code is a free pipeline
    run at Umair's expense (real ElevenLabs/Sync Labs cost).
    """
    if settings.demo_admin_key is None:
        raise HTTPException(503, "Demo admin key is not configured on this server")
    if x_admin_key is None or not secrets.compare_digest(x_admin_key, settings.demo_admin_key):
        raise HTTPException(401, "Missing or invalid X-Admin-Key header")


@router.post("/admin/codes", response_model=InviteCodeResponse)
async def create_invite_code(
    body: InviteCodeCreateRequest,
    x_admin_key: str | None = Header(default=None),
) -> InviteCodeResponse:
    _require_admin_key(x_admin_key)
    invite = access_store.generate(label=body.label)
    logger.info("Invite code issued: label=%r", body.label)
    return InviteCodeResponse(
        code=invite.code,
        label=invite.label,
        created_at=invite.created_at,
        used_at=invite.used_at,
        used_by_job_id=invite.used_by_job_id,
    )


@router.get("/admin/codes", response_model=InviteCodeListResponse)
async def list_invite_codes(x_admin_key: str | None = Header(default=None)) -> InviteCodeListResponse:
    _require_admin_key(x_admin_key)
    return InviteCodeListResponse(codes=[InviteCodeResponse(**c) for c in access_store.list_all()])


@router.post("/checkout", response_model=CheckoutSessionResponse)
async def start_checkout() -> CheckoutSessionResponse:
    """
    Creates a Stripe Checkout session for one paid demo run and hands
    back its URL. The frontend redirects the browser there; Stripe
    redirects back to public_base_url with ?payment_session=<id> on
    success, which the frontend then submits as X-Payment-Session on
    POST /demo/jobs.
    """
    try:
        session = create_checkout_session()
    except PaymentConfigError as exc:
        raise HTTPException(503, str(exc)) from exc
    except PaymentAPIError as exc:
        raise HTTPException(502, f"Stripe error creating checkout session: {exc}") from exc
    return CheckoutSessionResponse(checkout_url=session["url"], session_id=session["id"])


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request) -> dict[str, bool]:
    """
    Receives Stripe webhook events for audit logging only - the actual
    access decision is made at redemption time in POST /demo/jobs by
    calling session_is_paid() directly (see core/payments.py). Requires
    a real public URL to ever be reached, so it won't fire until this is
    deployed; safe to leave configured-or-not either way.
    """
    payload = await request.body()
    signature_header = request.headers.get("stripe-signature")

    if settings.stripe_webhook_secret is None:
        logger.warning("Received a Stripe webhook call but STRIPE_WEBHOOK_SECRET is unset - ignoring, not verified")
        return {"received": True}

    if not verify_webhook_signature(payload, signature_header, settings.stripe_webhook_secret):
        raise HTTPException(400, "Invalid webhook signature")

    logger.info("Stripe webhook received and signature-verified (audit only, not used to grant access)")
    return {"received": True}


@router.post("/jobs", response_model=JobSubmitResponse, status_code=202)
async def submit_demo_job(
    background_tasks: BackgroundTasks,
    source_audio: UploadFile = File(..., description="Creator's raw recorded read (mp3/wav/m4a)"),
    reference_video: UploadFile = File(..., description="Real reference footage of the speaker (mp4/mov)"),
    invite_code: str | None = Header(default=None, alias="X-Invite-Code"),
    payment_session: str | None = Header(default=None, alias="X-Payment-Session"),
) -> JobSubmitResponse:
    """
    Requires exactly one access path: an unused invite code, or a paid,
    not-yet-redeemed Stripe Checkout session.

    Redemption happens atomically BEFORE the upload is processed, not
    after - a real race was caught in code review: checking "is this
    code/session already used" and then redeeming it only after the job
    was accepted left a window where two concurrent requests for the
    same code/session could both pass the check and both start a real,
    billable pipeline run from a single free code or a single $X
    payment. `redeem()` on both stores is a single atomic
    check-and-mark, so redeeming it first is what actually closes that
    window - whichever request gets there first wins, the other is
    rejected before it ever touches an upload.

    The tradeoff: if the upload itself is then rejected (wrong file
    type, oversized file), the code/session has already been consumed.
    Handled by releasing it back (access_store.release /
    payment_store.release) specifically for that case - a client who
    picks the wrong file doesn't lose their one shot - but NOT once a
    job has actually started, even if it fails later for an unrelated
    reason (a third-party API failure, say); that's a real attempt, not
    a client-side mistake, and isn't refunded automatically.
    """
    if not invite_code and not payment_session:
        raise HTTPException(400, "Provide either X-Invite-Code or X-Payment-Session")
    if invite_code and payment_session:
        raise HTTPException(400, "Provide only one of X-Invite-Code or X-Payment-Session, not both")

    if invite_code:
        if access_store.validate(invite_code) is None:
            raise HTTPException(403, "Invalid or already-used invite code")
    else:
        if payment_store.is_redeemed(payment_session):
            raise HTTPException(403, "This payment has already been used for a run")
        try:
            paid = session_is_paid(payment_session)
        except PaymentConfigError as exc:
            raise HTTPException(503, str(exc)) from exc
        except PaymentAPIError as exc:
            # A genuine connection/API failure while checking payment
            # status must not be treated the same as "not paid" - that
            # would incorrectly deny someone who actually paid, just
            # because Stripe was briefly unreachable. Surfaced as 502 so
            # the frontend can tell the visitor to retry, not as a silent
            # rejection.
            raise HTTPException(502, f"Could not verify payment status: {exc}") from exc
        if not paid:
            raise HTTPException(402, "Payment not completed for this session")

    # Redeem now, atomically, before the upload - see docstring above for
    # why this has to happen before accept_upload_job rather than after.
    access_key = invite_code or payment_session
    store = access_store if invite_code else payment_store
    if not store.redeem(access_key, job_id="pending"):
        kind = "invite code" if invite_code else "payment session"
        raise HTTPException(403, f"This {kind} was just used by another request")

    try:
        job = await accept_upload_job(background_tasks, source_audio, reference_video)
    except HTTPException:
        # The upload itself was rejected (bad extension, oversized file) -
        # no pipeline work ever started, so give the code/session back
        # rather than burning it on a fixable client mistake.
        store.release(access_key)
        raise

    store.set_job_id(access_key, job.job_id)
    return JobSubmitResponse(job_id=job.job_id, status="queued")
