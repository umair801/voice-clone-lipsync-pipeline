"""
Tests for the Phase 2 payment-gated demo API surface: POST /demo/checkout,
POST /demo/jobs with X-Payment-Session, and POST /demo/stripe/webhook.

core.payments.create_checkout_session / session_is_paid are mocked here
rather than calling the real Stripe API - same reasoning as every other
external-API test in this project (test_api_hardening.py mocks
ElevenLabs/Sync Labs, test_demo_api.py doesn't touch Stripe at all). The
underlying HTTP calls to Stripe were verified live against the real test
API during development (see project notes) - real session creation with
a real session id and checkout URL, real "unpaid" status on a fresh
session, a real rejection on a garbage session id. This device's network
showed intermittent SSL resets specifically on the retrieve (GET) call
through its outbound proxy, unrelated to the code - mocking here keeps
the test suite reliable regardless of that.

Run:
    python -m pytest tests/test_demo_payment_api.py -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402
from core.payment_store import payment_store  # noqa: E402
from core.payments import PaymentAPIError, PaymentConfigError  # noqa: E402


def _fake_voice_convert(self, input_path, output_path, params=None, model_id="eleven_multilingual_sts_v2"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(input_path, output_path)
    return output_path


def _fake_lipsync_generate(self, video_path, audio_path, output_path, options=None):
    from agents.tools.sync_labs_lipsync import LipsyncResult
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(video_path, output_path)
    return LipsyncResult(
        generation_id="mock-generation-id",
        status="COMPLETED",
        output_path=output_path,
        output_duration_seconds=None,
        model="lipsync-2",
    )


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "jobs_output_dir", str(tmp_path / "job_outputs"))
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


@pytest.fixture()
def sample_files():
    repo_root = Path(__file__).resolve().parent.parent
    audio = repo_root / "test_read.mp3"
    video = repo_root / "video_samples" / "input" / "clip1_fixed.mp4"
    assert audio.exists() and video.exists(), "missing fixtures for demo payment API tests"
    return audio, video


def test_checkout_requires_stripe_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "stripe_secret_key", None)
    resp = client.post("/demo/checkout")
    assert resp.status_code == 503, resp.text
    print("PASS: POST /demo/checkout refuses when Stripe isn't configured")


def test_checkout_creates_session(client, monkeypatch):
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_fake")
    fake_session = {"id": "cs_test_fake123", "url": "https://checkout.stripe.com/c/pay/cs_test_fake123"}
    with patch("api.demo.create_checkout_session", return_value=fake_session):
        resp = client.post("/demo/checkout")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"] == "cs_test_fake123"
    assert body["checkout_url"] == fake_session["url"]
    print("PASS: POST /demo/checkout returns a real-shaped session id + URL")


def test_checkout_surfaces_stripe_api_error(client, monkeypatch):
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_fake")
    with patch("api.demo.create_checkout_session", side_effect=PaymentAPIError("card declined or whatever Stripe said")):
        resp = client.post("/demo/checkout")
    assert resp.status_code == 502, resp.text
    print("PASS: a Stripe API error surfaces as 502, not swallowed")


def test_demo_jobs_requires_exactly_one_access_method(client, sample_files):
    audio, video = sample_files
    files = {
        "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
        "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
    }
    resp_neither = client.post("/demo/jobs", files=files)
    assert resp_neither.status_code == 400, resp_neither.text

    resp_both = client.post(
        "/demo/jobs", files=files, headers={"X-Invite-Code": "AB12-CD34-EF56", "X-Payment-Session": "cs_test_x"}
    )
    assert resp_both.status_code == 400, resp_both.text
    print("PASS: /demo/jobs rejects both zero and two access methods at once")


def test_demo_jobs_rejects_unpaid_session(client, sample_files):
    audio, video = sample_files
    files = {
        "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
        "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
    }
    with patch("api.demo.session_is_paid", return_value=False):
        resp = client.post("/demo/jobs", files=files, headers={"X-Payment-Session": "cs_test_unpaid"})
    assert resp.status_code == 402, resp.text
    print("PASS: an unpaid Checkout session is rejected with 402")


def test_demo_jobs_surfaces_stripe_connection_failure_as_502_not_denial(client, sample_files):
    """
    A real finding from live testing: session_is_paid() can raise
    PaymentAPIError on a genuine connection failure (not a clean "not
    paid" 4xx). That must surface as 502 (ask the visitor to retry), not
    silently deny access as if they hadn't paid, and not an unhandled 500.
    """
    audio, video = sample_files
    files = {
        "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
        "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
    }
    with patch("api.demo.session_is_paid", side_effect=PaymentAPIError("Could not reach Stripe: connection reset")):
        resp = client.post("/demo/jobs", files=files, headers={"X-Payment-Session": "cs_test_network_blip"})
    assert resp.status_code == 502, resp.text
    print("PASS: a Stripe connection failure surfaces as 502, not a false denial or a 500")


def test_demo_jobs_accepts_paid_session_and_redeems_it_once(client, sample_files):
    audio, video = sample_files
    files = {
        "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
        "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
    }
    session_id = "cs_test_paid_once_only"

    with (
        patch("api.demo.session_is_paid", return_value=True),
        patch("agents.tools.elevenlabs_voice_changer.ElevenLabsVoiceChanger.convert", _fake_voice_convert),
        patch("agents.tools.sync_labs_lipsync.SyncLabsLipsyncClient.generate_from_files", _fake_lipsync_generate),
    ):
        resp = client.post("/demo/jobs", files=files, headers={"X-Payment-Session": session_id})
        assert resp.status_code == 202, resp.text

        # same paid session must not work a second time, even though it's
        # still genuinely "paid" from Stripe's point of view - our own
        # single-use enforcement is what stops the reuse
        assert payment_store.is_redeemed(session_id) is True
        resp_again = client.post("/demo/jobs", files=files, headers={"X-Payment-Session": session_id})
        assert resp_again.status_code == 403, resp_again.text
    print("PASS: a paid session is accepted once, then rejected on reuse")


def test_paid_session_is_released_on_rejected_upload_not_burned(client, sample_files, monkeypatch):
    """
    Real fix from code review: redemption now happens atomically before
    the upload is processed (closing a race where two concurrent
    requests for the same session could both start a real pipeline run
    off one payment). The tradeoff is handled explicitly - if the
    upload itself is then rejected (here: oversized file), the session
    must be released back to unredeemed, not burned on a client mistake
    that never started any real pipeline work.
    """
    from core.config import settings as cfg
    monkeypatch.setattr(cfg, "max_upload_bytes", 1024)  # tiny, deliberately - force a 413

    audio, video = sample_files
    files = {
        "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
        "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
    }
    session_id = "cs_test_rejected_upload"

    with patch("api.demo.session_is_paid", return_value=True):
        resp = client.post("/demo/jobs", files=files, headers={"X-Payment-Session": session_id})
    assert resp.status_code == 413, resp.text
    assert payment_store.is_redeemed(session_id) is False, "a rejected upload must not burn the paid session"

    # the same session should now work on a real (non-oversized) attempt
    monkeypatch.setattr(cfg, "max_upload_bytes", 100 * 1024 * 1024)
    with (
        patch("api.demo.session_is_paid", return_value=True),
        patch("agents.tools.elevenlabs_voice_changer.ElevenLabsVoiceChanger.convert", _fake_voice_convert),
        patch("agents.tools.sync_labs_lipsync.SyncLabsLipsyncClient.generate_from_files", _fake_lipsync_generate),
    ):
        resp2 = client.post("/demo/jobs", files=files, headers={"X-Payment-Session": session_id})
    assert resp2.status_code == 202, resp2.text
    print("PASS: a session released after a rejected upload can still be used for a real run")


def test_webhook_ignores_when_secret_unset(client, monkeypatch):
    monkeypatch.setattr(settings, "stripe_webhook_secret", None)
    resp = client.post("/demo/stripe/webhook", json={"type": "checkout.session.completed"})
    assert resp.status_code == 200, resp.text
    print("PASS: webhook endpoint returns 200 (no-op) when no secret is configured")


def test_webhook_rejects_bad_signature(client, monkeypatch):
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_real_secret")
    resp = client.post(
        "/demo/stripe/webhook",
        json={"type": "checkout.session.completed"},
        headers={"stripe-signature": "t=123,v1=deadbeef"},
    )
    assert resp.status_code == 400, resp.text
    print("PASS: webhook endpoint rejects a bad/forged signature")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
