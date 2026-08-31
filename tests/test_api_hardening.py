"""
Day 6 QA hardening tests - the two real gaps the Day 4 code review flagged
(no auth on POST /jobs, no upload size cap) plus the JobStore field-name
validation fix. Uses FastAPI's TestClient (in-process, no real server/port
needed) and mocks the external voice/lipsync calls exactly like
test_orchestrator_mocked.py - these tests are about the API layer's own
behavior, not about re-proving the pipeline works end to end.

Run:
    python -m pytest tests/test_api_hardening.py -v
"""
from __future__ import annotations

import io
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402
from core.job_store import job_store  # noqa: E402


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
    assert audio.exists() and video.exists(), "missing fixtures for API hardening tests"
    return audio, video


def test_submit_job_requires_api_key_when_set(client, sample_files, monkeypatch):
    """
    With PIPELINE_API_KEY set, POST /jobs without a matching X-API-Key
    header must be rejected before a job is ever created or any external
    API is called.
    """
    monkeypatch.setattr(settings, "pipeline_api_key", "test-secret-key")
    audio, video = sample_files

    with (
        patch("agents.tools.elevenlabs_voice_changer.ElevenLabsVoiceChanger.convert", _fake_voice_convert),
        patch("agents.tools.sync_labs_lipsync.SyncLabsLipsyncClient.generate_from_files", _fake_lipsync_generate),
    ):
        resp_no_key = client.post(
            "/jobs",
            files={
                "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
                "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
            },
        )
        assert resp_no_key.status_code == 401, resp_no_key.text

        resp_wrong_key = client.post(
            "/jobs",
            files={
                "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
                "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
            },
            headers={"X-API-Key": "not-the-right-key"},
        )
        assert resp_wrong_key.status_code == 401, resp_wrong_key.text

        resp_ok = client.post(
            "/jobs",
            files={
                "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
                "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
            },
            headers={"X-API-Key": "test-secret-key"},
        )
        assert resp_ok.status_code == 202, resp_ok.text
    print("PASS: POST /jobs enforces X-API-Key when PIPELINE_API_KEY is set")


def test_submit_job_allows_no_key_when_unset(client, sample_files, monkeypatch):
    """
    Local-dev default: PIPELINE_API_KEY unset means no auth is enforced -
    must not regress and start requiring a header nobody configured.
    """
    monkeypatch.setattr(settings, "pipeline_api_key", None)
    audio, video = sample_files

    with (
        patch("agents.tools.elevenlabs_voice_changer.ElevenLabsVoiceChanger.convert", _fake_voice_convert),
        patch("agents.tools.sync_labs_lipsync.SyncLabsLipsyncClient.generate_from_files", _fake_lipsync_generate),
    ):
        resp = client.post(
            "/jobs",
            files={
                "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
                "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
            },
        )
    assert resp.status_code == 202, resp.text
    print("PASS: POST /jobs works with no header when PIPELINE_API_KEY is unset")


def test_submit_job_rejects_oversized_upload(client, sample_files, monkeypatch):
    """
    An upload whose body exceeds settings.max_upload_bytes must be
    rejected with 413, checked against actual bytes streamed to disk (not
    just trusting a Content-Length header).
    """
    monkeypatch.setattr(settings, "pipeline_api_key", None)
    monkeypatch.setattr(settings, "max_upload_bytes", 1024)  # tiny, deliberately
    audio, video = sample_files

    resp = client.post(
        "/jobs",
        files={
            "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
            "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
        },
    )
    assert resp.status_code == 413, resp.text
    print("PASS: oversized upload is rejected with 413")


def test_submit_job_cleans_up_on_rejected_upload(client, sample_files, monkeypatch):
    """
    Day 6 code-review fix: when the second file (reference_video) exceeds
    max_upload_bytes after the first file (source_audio) already saved
    successfully, the job created for this submission must not be left
    stuck at 'queued' forever, and the partially-written input directory
    must not be left behind on disk.
    """
    monkeypatch.setattr(settings, "pipeline_api_key", None)
    monkeypatch.setattr(settings, "max_upload_bytes", 1024)
    audio, video = sample_files

    jobs_before = set(job_store._jobs.keys())

    resp = client.post(
        "/jobs",
        files={
            "source_audio": ("test_read.mp3", audio.read_bytes(), "audio/mpeg"),
            "reference_video": ("clip1_fixed.mp4", video.read_bytes(), "video/mp4"),
        },
    )
    assert resp.status_code == 413, resp.text

    jobs_after = set(job_store._jobs.keys())
    assert jobs_after == jobs_before, "a job record was left behind after a rejected upload"

    jobs_output_dir = Path(settings.jobs_output_dir)
    leftover_input_dirs = list(jobs_output_dir.glob("*/input")) if jobs_output_dir.exists() else []
    assert leftover_input_dirs == [], f"orphaned input dir(s) left on disk: {leftover_input_dirs}"
    print("PASS: rejected upload leaves no orphaned job record or partial files")


def test_job_store_rejects_unknown_field():
    """
    JobStore.update must raise on a kwarg that isn't a real Job field,
    rather than silently creating an unused attribute via plain setattr -
    the Day 4 code-review gap fixed in Day 6.
    """
    job = job_store.create()
    with pytest.raises(TypeError):
        job_store.update(job.job_id, statuss="typo")  # deliberately misspelled
    # a real field still works fine
    updated = job_store.update(job.job_id, status="completed")
    assert updated.status == "completed"
    print("PASS: JobStore.update rejects an unknown field name")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
