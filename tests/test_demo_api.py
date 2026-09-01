"""
Tests for api/demo.py - the invite-code-gated public demo submit path
and the admin code-generation endpoints (Phase 1 of the public demo
frontend). Same TestClient + mocked-external-API pattern as
test_api_hardening.py; these tests are about the demo API layer's own
behavior, not a re-proof that the pipeline itself works.

Run:
    python -m pytest tests/test_demo_api.py -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.access_store import access_store  # noqa: E402
from core.config import settings  # noqa: E402


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
    assert audio.exists() and video.exists(), "missing fixtures for demo API tests"
    return audio, video


def test_demo_jobs_rejects_missing_or_invalid_code(client, sample_files):
    """
    X-Invite-Code is no longer FastAPI-required (Phase 2 made it optional
    since X-Payment-Session is now a valid alternative access method) -
    a request with neither header is rejected by application logic (400,
    see test_demo_jobs_requires_exactly_one_access_method in
    test_demo_payment_api.py for that case specifically), while an
    invite code that's present but wrong/used is still a 403.
    """
    audio, video = sample_files
    files = {
        "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
        "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
    }

    resp_bad_code = client.post("/demo/jobs", files=files, headers={"X-Invite-Code": "NOPE-NOPE-NOPE"})
    assert resp_bad_code.status_code == 403, resp_bad_code.text
    print("PASS: /demo/jobs rejects an invalid invite code")


def test_demo_jobs_accepts_valid_unused_code_and_redeems_it(client, sample_files):
    invite = access_store.generate(label="test-run")
    audio, video = sample_files
    files = {
        "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
        "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
    }

    with (
        patch("agents.tools.elevenlabs_voice_changer.ElevenLabsVoiceChanger.convert", _fake_voice_convert),
        patch("agents.tools.sync_labs_lipsync.SyncLabsLipsyncClient.generate_from_files", _fake_lipsync_generate),
    ):
        resp = client.post("/demo/jobs", files=files, headers={"X-Invite-Code": invite.code})
    assert resp.status_code == 202, resp.text

    # the same code must not work a second time
    reused = access_store.validate(invite.code)
    assert reused is None
    print("PASS: a valid code is accepted once and then redeemed")


def test_demo_jobs_rejects_already_used_code(client, sample_files):
    invite = access_store.generate()
    access_store.redeem(invite.code, job_id="already-used")
    audio, video = sample_files
    files = {
        "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
        "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
    }

    resp = client.post("/demo/jobs", files=files, headers={"X-Invite-Code": invite.code})
    assert resp.status_code == 403, resp.text
    print("PASS: an already-used code is rejected on a second submission")


def test_admin_codes_requires_configured_admin_key(client, monkeypatch):
    monkeypatch.setattr(settings, "demo_admin_key", None)
    resp = client.post("/demo/admin/codes", json={"label": "x"}, headers={"X-Admin-Key": "anything"})
    assert resp.status_code == 503, resp.text
    print("PASS: admin endpoint refuses every request when DEMO_ADMIN_KEY is unset")


def test_admin_codes_requires_correct_key_and_generates(client, monkeypatch):
    monkeypatch.setattr(settings, "demo_admin_key", "the-real-admin-key")

    resp_wrong = client.post("/demo/admin/codes", json={"label": "x"}, headers={"X-Admin-Key": "wrong"})
    assert resp_wrong.status_code == 401, resp_wrong.text

    resp_ok = client.post("/demo/admin/codes", json={"label": "acme corp"}, headers={"X-Admin-Key": "the-real-admin-key"})
    assert resp_ok.status_code == 200, resp_ok.text
    body = resp_ok.json()
    assert body["label"] == "acme corp"
    assert body["used_at"] is None
    assert access_store.validate(body["code"]) is not None
    print("PASS: admin endpoint generates a usable code with the correct key")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
