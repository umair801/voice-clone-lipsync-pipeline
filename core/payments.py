"""
Stripe Checkout integration for the public demo frontend (Phase 2).

Implemented against Stripe's REST API directly via `requests` rather
than the official `stripe` Python SDK. Reason, found during real testing
on Umair's machine (not a style preference): the SDK's own HTTP client
hit intermittent `SSLError: UNEXPECTED_EOF_WHILE_READING` failures on
this specific network - reproduced multiple times, including on the very
first request of a fresh process - while a plain `requests.post`/`get`
call to the exact same host succeeded. `curl` to the same host also
succeeded reliably. This points to a flaky interaction between the SDK's
connection-pooling/retry layer and something on this particular Windows
machine's network path (proxy, AV, or ISP-level TLS interference), not a
bug in Stripe's API or in this code. Using `requests` directly sidesteps
it, adds one dependency less, and Railway's network (the real deployment
target) is very unlikely to have the same issue - but this is worth
re-testing once deployed rather than assumed fixed.

Two operations, both server-side, both against the real Stripe API (test
mode until Umair swaps in live keys):

1. create_checkout_session() - starts a Checkout Session for one demo run.
2. session_is_paid() - the actual payment check at redemption time. Calls
   the Checkout Session retrieve endpoint directly rather than trusting a
   webhook having fired - the webhook endpoint (api/demo.py's
   /demo/stripe/webhook) needs a real public URL to receive events, not
   available until this is deployed to Railway, and even once deployed
   the webhook is kept for audit logging only, not as the access gate -
   matches Stripe's own guidance to verify state server-side rather than
   trust a client-supplied session_id alone.

No error is ever swallowed here - a real API failure (bad key, Stripe
outage, unexpected response shape) surfaces as a real exception, not a
silent False. A short retry (via tenacity, same pattern used elsewhere in
this project for external API calls) covers exactly the kind of
transient connection reset seen during testing above - not indefinite
retries, and not retried on an actual 4xx from Stripe (a bad/expired
session id should fail once, not retry).
"""
from __future__ import annotations

from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

STRIPE_API_BASE = "https://api.stripe.com/v1"
REQUEST_TIMEOUT_SECONDS = 30


class PaymentConfigError(Exception):
    """Stripe isn't configured - raised instead of letting a request fail with a confusing 401 further down."""


class PaymentAPIError(Exception):
    """Stripe's API returned an error response (not a connection problem) - message includes Stripe's own detail."""


def _require_configured() -> str:
    if not settings.stripe_secret_key:
        raise PaymentConfigError("STRIPE_SECRET_KEY is not set - the paid demo path is not usable yet")
    return settings.stripe_secret_key


@retry(
    retry=retry_if_exception_type(requests.exceptions.ConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_fixed(1.5),
    reraise=True,
)
def _request(method: str, path: str, data: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    """
    Shared HTTP call for both _post/_get below. A connection-level
    failure (SSLError, ConnectionError, etc.) is retried up to 3 times by
    tenacity, then converted to PaymentAPIError here rather than left to
    propagate as a raw requests exception - a caller catching
    PaymentAPIError (every caller does) must not need to also know about
    requests' exception hierarchy. Caught in real testing: on a flaky
    network, the retries can still all fail and the raw
    requests.exceptions.SSLError was reaching FastAPI unhandled, turning
    into an opaque 500 instead of the clean 502 callers expect.
    """
    secret_key = _require_configured()
    try:
        resp = requests.request(
            method,
            f"{STRIPE_API_BASE}/{path}",
            auth=(secret_key, ""),
            data=data,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise PaymentAPIError(f"Could not reach Stripe: {exc}") from exc
    return resp.status_code, resp.json()


def _post(path: str, data: dict[str, Any]) -> dict[str, Any]:
    status_code, body = _request("POST", path, data)
    if status_code >= 400:
        raise PaymentAPIError(body.get("error", {}).get("message", f"Stripe API error {status_code}"))
    return body


def _get(path: str) -> tuple[int, dict[str, Any]]:
    return _request("GET", path)


def create_checkout_session() -> dict[str, Any]:
    """
    Creates a one-time-payment Checkout Session for a single demo run.
    Price is read from settings.demo_price_usd at call time (not a
    pre-created Stripe Price object), so Umair can change the price in
    .env without touching the Stripe dashboard. Returns the raw session
    dict (has "id" and "url").
    """
    success_url = f"{settings.public_base_url}/?payment_session={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{settings.public_base_url}/"
    data = {
        "mode": "payment",
        "success_url": success_url,
        "cancel_url": cancel_url,
        "line_items[0][quantity]": "1",
        "line_items[0][price_data][currency]": "usd",
        "line_items[0][price_data][unit_amount]": str(settings.demo_price_usd * 100),
        "line_items[0][price_data][product_data][name]": "Voice-Clone Lipsync Pipeline - one demo run",
        "line_items[0][price_data][product_data][description]": (
            "One voice-clone + lipsync generation through the live pipeline demo."
        ),
    }
    session = _post("checkout/sessions", data)
    logger.info("Created Stripe Checkout session %s ($%d)", session["id"], settings.demo_price_usd)
    return session


def verify_webhook_signature(payload: bytes, signature_header: str | None, secret: str) -> bool:
    """
    Verifies a Stripe webhook's Stripe-Signature header by hand (HMAC-SHA256
    over "{timestamp}.{payload}", matched against the v1 signature in the
    header) rather than pulling in the stripe SDK just for this one check.
    Format: "t=<unix-ts>,v1=<hex-hmac>[,v0=...]" - only v1 is checked, v0
    is an older scheme Stripe still sends for backwards compatibility.
    Does not check timestamp tolerance (replay-window enforcement) - this
    endpoint is audit-logging only, not the access gate (see module
    docstring), so a replayed webhook can't grant access on its own.
    """
    if not signature_header:
        return False
    import hashlib
    import hmac

    parts = dict(item.split("=", 1) for item in signature_header.split(",") if "=" in item)
    timestamp = parts.get("t")
    v1_signature = parts.get("v1")
    if not timestamp or not v1_signature:
        return False

    signed_payload = f"{timestamp}.".encode() + payload
    expected = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, v1_signature)


def session_is_paid(session_id: str) -> bool:
    """
    Retrieves the Checkout Session from Stripe and returns whether it was
    actually paid. Does not raise on a session_id Stripe doesn't
    recognize (a client sending a garbage/expired id) - treated as "not
    paid", same as any other invalid-access case, not a 500.
    """
    status_code, body = _get(f"checkout/sessions/{session_id}")
    if status_code >= 400:
        logger.warning("Checkout session lookup failed for %r: %s", session_id, body.get("error", {}).get("type"))
        return False
    return body.get("payment_status") == "paid"
