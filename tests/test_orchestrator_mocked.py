"""
Mocked end-to-end orchestrator test - exercises the full LangGraph
pipeline without calling the live ElevenLabs or Sync Labs APIs. Day 2-3
spent all 3 free Sync Labs generations for this month; this test must not
depend on live quota to pass, or on ElevenLabs minutes either.

Real, NOT mocked: video normalization (ffmpeg) and the QA duration check
(ffprobe) - these are local subprocess calls, not paid API calls, so
there's no reason to fake them out and lose real coverage of that code.

Mocked: ElevenLabsVoiceChanger.convert, SyncLabsLipsyncClient.generate_from_files.

Run:
    python -m tests.test_orchestrator_mocked
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.orchestrator import run_pipeline  # noqa: E402
from agents.tools.sync_labs_lipsync import LipsyncResult  # noqa: E402


def _fake_voice_convert(self, input_path, output_path, params=None, model_id="eleven_multilingual_sts_v2"):
    # Real behavior would call ElevenLabs; here just copy the source audio
    # through unchanged so the real downstream ffmpeg/ffprobe steps have a
    # real, valid file to operate on.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(input_path, output_path)
    return output_path


def _fake_lipsync_generate(self, video_path, audio_path, output_path, options=None):
    # Real behavior would call Sync Labs; here just copy the (already
    # normalized) reference video through as the "lipsync output" so the
    # QA duration check downstream runs against a real, ffprobe-readable
    # file with a real duration.
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


def _find_source_audio(repo_root: Path) -> Path:
    for candidate in (
        repo_root / "audio_samples" / "input" / "test_read.mp3",
        repo_root / "test_read.mp3",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No source audio fixture found (expected test_read.mp3)")


def test_pipeline_happy_path():
    repo_root = Path(__file__).resolve().parent.parent
    source_audio = _find_source_audio(repo_root)
    reference_video = repo_root / "video_samples" / "input" / "clip1_fixed.mp4"
    assert reference_video.exists(), f"missing fixture: {reference_video}"

    with tempfile.TemporaryDirectory() as tmp_out:
        with (
            patch("agents.tools.elevenlabs_voice_changer.ElevenLabsVoiceChanger.convert", _fake_voice_convert),
            patch("agents.tools.sync_labs_lipsync.SyncLabsLipsyncClient.generate_from_files", _fake_lipsync_generate),
        ):
            final_state = run_pipeline(
                job_id="test-job-1",
                source_audio_path=str(source_audio),
                reference_video_path=str(reference_video),
                output_dir=tmp_out,
            )

        assert final_state["error"] is None, final_state
        assert final_state["status"] in ("completed", "needs_review"), final_state
        assert Path(final_state["converted_audio_path"]).exists()
        assert Path(final_state["normalized_video_path"]).exists()
        assert Path(final_state["lipsync_output_path"]).exists()
        assert final_state["qa"] is not None
        assert final_state["qa"]["duration_check"]["ok"] is True, final_state["qa"]
        print("PASS: happy path ->", final_state["status"])


def test_pipeline_voice_conversion_failure_short_circuits():
    repo_root = Path(__file__).resolve().parent.parent
    reference_video = repo_root / "video_samples" / "input" / "clip1_fixed.mp4"

    def _raise(*args, **kwargs):
        from agents.tools.elevenlabs_voice_changer import VoiceChangerError
        raise VoiceChangerError("simulated ElevenLabs outage")

    with tempfile.TemporaryDirectory() as tmp_out:
        with patch("agents.tools.elevenlabs_voice_changer.ElevenLabsVoiceChanger.convert", _raise):
            final_state = run_pipeline(
                job_id="test-job-2",
                source_audio_path="does-not-matter.mp3",
                reference_video_path=str(reference_video),
                output_dir=tmp_out,
            )

    assert final_state["status"] == "failed"
    assert final_state["error_stage"] == "voice_conversion"
    # normalize_video/lipsync/qa must NOT have run after the failure
    assert final_state["normalized_video_path"] is None
    assert final_state["lipsync_output_path"] is None
    print("PASS: voice_conversion failure short-circuits the rest of the pipeline")


def test_pipeline_qa_flags_duration_mismatch():
    """
    QA duration check should route to needs_review, not silently pass,
    when the lipsync output's duration doesn't match the reference video.
    """
    repo_root = Path(__file__).resolve().parent.parent
    source_audio = _find_source_audio(repo_root)
    reference_video = repo_root / "video_samples" / "input" / "clip1_fixed.mp4"

    def _fake_lipsync_wrong_duration(self, video_path, audio_path, output_path, options=None):
        # Deliberately return the much-shorter Day 1 test read's own audio
        # re-packaged as if it were a video output, to force a duration
        # mismatch against the ~15.8s reference video.
        import subprocess
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-t", "3", "-c", "copy", str(output_path)],
            check=True,
        )
        return LipsyncResult(
            generation_id="mock-generation-id-short",
            status="COMPLETED",
            output_path=output_path,
            output_duration_seconds=3.0,
            model="lipsync-2",
        )

    with tempfile.TemporaryDirectory() as tmp_out:
        with (
            patch("agents.tools.elevenlabs_voice_changer.ElevenLabsVoiceChanger.convert", _fake_voice_convert),
            patch("agents.tools.sync_labs_lipsync.SyncLabsLipsyncClient.generate_from_files", _fake_lipsync_wrong_duration),
        ):
            final_state = run_pipeline(
                job_id="test-job-3",
                source_audio_path=str(source_audio),
                reference_video_path=str(reference_video),
                output_dir=tmp_out,
            )

        assert final_state["status"] == "needs_review", final_state
        assert final_state["qa"]["needs_review"] is True
        assert final_state["qa"]["duration_check"]["ok"] is False
        print("PASS: duration mismatch correctly routes to needs_review")


if __name__ == "__main__":
    test_pipeline_happy_path()
    test_pipeline_voice_conversion_failure_short_circuits()
    test_pipeline_qa_flags_duration_mismatch()
    print("\nAll orchestrator tests passed.")
