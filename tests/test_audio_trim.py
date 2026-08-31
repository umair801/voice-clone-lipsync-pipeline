"""
Tests for the leading near-silence trim step (Day 6 addendum fix).

Uses synthetically generated audio (ffmpeg anullsrc + sine tone), not a
recorded fixture, so these tests are deterministic and don't depend on
any specific take being present in the repo.

Run:
    python -m tests.test_audio_trim
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.tools.audio_trim import AudioTrimError, trim_leading_silence  # noqa: E402


def _make_clip(path: Path, quiet_seconds: float, tone_seconds: float, quiet_amplitude: float = 0.001) -> None:
    """
    Build a synthetic mp3: `quiet_seconds` of very low-amplitude noise
    (stands in for the room-tone/breath the real ElevenLabs output
    carries), followed by `tone_seconds` of a clear 440Hz tone (stands in
    for real speech). quiet_amplitude=0.001 puts the lead-in around -60dB,
    comfortably under the -40dB detection threshold, matching the
    magnitude of the real defect's measured RMS gap (Day 6 addendum:
    quiet stretch ~-55dB, real speech ~-31dB).
    """
    if quiet_seconds <= 0.0:
        filter_complex = f"sine=frequency=440:duration={tone_seconds}[out]"
    else:
        filter_complex = (
            f"anoisesrc=d={quiet_seconds}:c=white:a={quiet_amplitude}[quiet];"
            f"sine=frequency=440:duration={tone_seconds}[tone];"
            f"[quiet][tone]concat=n=2:v=0:a=1[out]"
        )
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-t", str(quiet_seconds + tone_seconds),
        "-c:a", "libmp3lame", "-q:a", "2",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def test_trims_leading_quiet_audio():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src.mp3"
        out = tmp / "out.mp3"
        _make_clip(src, quiet_seconds=2.0, tone_seconds=1.0)

        trim_leading_silence(src, out)

        assert out.exists() and out.stat().st_size > 0

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
            capture_output=True, text=True, check=True,
        )
        out_duration = float(probe.stdout.strip())

        # Original clip is 3.0s (2.0s quiet + 1.0s tone). Trimmed output
        # should be much shorter than the source but still cover the full
        # tone plus the padding - not clipped down to near-zero.
        assert out_duration < 2.0, f"expected most of the quiet lead-in trimmed, got {out_duration:.2f}s"
        assert out_duration > 0.9, f"expected the tone to survive intact, got {out_duration:.2f}s"
        print(f"PASS: trims leading quiet audio (source 3.0s -> trimmed {out_duration:.2f}s)")


def test_no_trim_when_clip_opens_on_real_audio():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src.mp3"
        out = tmp / "out.mp3"
        _make_clip(src, quiet_seconds=0.0, tone_seconds=1.0)

        trim_leading_silence(src, out)

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
            capture_output=True, text=True, check=True,
        )
        out_duration = float(probe.stdout.strip())

        # No leading silence to find - output should be left essentially
        # untouched (same ~1.0s duration as the source).
        assert out_duration > 0.9, f"expected untrimmed passthrough, got {out_duration:.2f}s"
        print(f"PASS: no-op passthrough when clip opens on real audio ({out_duration:.2f}s)")


def test_missing_input_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            trim_leading_silence(tmp / "does-not-exist.mp3", tmp / "out.mp3")
        except AudioTrimError as exc:
            assert "not found" in str(exc)
            print("PASS: missing input raises AudioTrimError")
            return
        raise AssertionError("expected AudioTrimError for missing input")


if __name__ == "__main__":
    test_trims_leading_quiet_audio()
    test_no_trim_when_clip_opens_on_real_audio()
    test_missing_input_raises()
    print("\nAll audio_trim tests passed.")
