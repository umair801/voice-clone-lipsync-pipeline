"""
Tests for the Phase 2 payment gate: core/payment_store.py (single-use
redemption tracking) and core/payments.py's webhook signature check.
Does NOT hit the real Stripe API - create_checkout_session()/
session_is_paid() are exercised at the api/demo.py layer with mocks
instead (see tests/test_demo_payment_api.py), consistent with how
tests/test_api_hardening.py and tests/test_demo_api.py mock the other
real external APIs (ElevenLabs, Sync Labs) rather than calling them.

Run:
    python -m pytest tests/test_payments.py -v
"""
from __future__ import annotations

import hashlib
import hmac
import time

from core.payment_store import PaymentStore
from core.payments import verify_webhook_signature


def test_payment_session_redeem_blocks_reuse():
    store = PaymentStore()
    assert store.is_redeemed("cs_test_abc") is False
    assert store.redeem("cs_test_abc", job_id="job-1") is True
    assert store.is_redeemed("cs_test_abc") is True
    assert store.redeem("cs_test_abc", job_id="job-2") is False  # can't redeem twice
    print("PASS: a payment session can be redeemed exactly once")


def test_payment_session_redeem_is_independent_per_session():
    store = PaymentStore()
    store.redeem("cs_test_one", job_id="job-1")
    assert store.is_redeemed("cs_test_two") is False
    print("PASS: redeeming one session does not affect another")


def _sign(secret: str, payload: bytes, timestamp: str) -> str:
    signed_payload = f"{timestamp}.".encode() + payload
    return hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()


def test_webhook_signature_accepts_correctly_signed_payload():
    secret = "whsec_test_secret"
    payload = b'{"type": "checkout.session.completed"}'
    ts = str(int(time.time()))
    sig = _sign(secret, payload, ts)
    header = f"t={ts},v1={sig}"
    assert verify_webhook_signature(payload, header, secret) is True
    print("PASS: a correctly-signed webhook payload verifies")


def test_webhook_signature_rejects_wrong_secret():
    payload = b'{"type": "checkout.session.completed"}'
    ts = str(int(time.time()))
    sig = _sign("whsec_correct", payload, ts)
    header = f"t={ts},v1={sig}"
    assert verify_webhook_signature(payload, header, "whsec_wrong") is False
    print("PASS: a webhook signed with the wrong secret is rejected")


def test_webhook_signature_rejects_tampered_payload():
    secret = "whsec_test_secret"
    ts = str(int(time.time()))
    sig = _sign(secret, b'{"amount": 500}', ts)
    header = f"t={ts},v1={sig}"
    tampered_payload = b'{"amount": 50000}'
    assert verify_webhook_signature(tampered_payload, header, secret) is False
    print("PASS: a tampered payload fails signature verification")


def test_webhook_signature_rejects_missing_header():
    assert verify_webhook_signature(b"{}", None, "whsec_test_secret") is False
    print("PASS: a missing Stripe-Signature header is rejected, not treated as valid")


if __name__ == "__main__":
    test_payment_session_redeem_blocks_reuse()
    test_payment_session_redeem_is_independent_per_session()
    test_webhook_signature_accepts_correctly_signed_payload()
    test_webhook_signature_rejects_wrong_secret()
    test_webhook_signature_rejects_tampered_payload()
    test_webhook_signature_rejects_missing_header()
