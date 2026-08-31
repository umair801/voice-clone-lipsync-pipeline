"""
Video pre-normalization: bake rotation metadata into pixels before a
video is sent to Sync Labs.

Real bug found in Day 2-3 testing (see project handoff doc): Sync Labs'
handling of a source video's rotation display-matrix metadata (the tag
phone cameras use instead of physically rotating pixels) was
inconsistent - one clip came back correctly oriented, an identically
encoded clip came back sideways. Root cause never isolated. The reliable
fix is to remove the ambiguity before upload: re-encode so the pixels are
already in the correct display orientation and no rotation metadata is
left for Sync Labs to interpret one way or another.

This runs on every reference video before it reaches SyncLabsLipsyncClient
in the orchestration pipeline - not optional, not conditional on "looks
like it might need it". A silently sideways output in an automated
pipeline with no human previewing every clip is a much worse failure than
paying the encode cost on every run.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from core.logging import get_logger

logger = get_logger(__name__)


class VideoNormalizeError(Exception):
    """Raised when ffmpeg fails to normalize a video."""


def normalize_video(input_path: str | Path, output_path: str | Path, timeout: int = 120) -> Path:
    """
    Re-encode input_path so any rotation display-matrix is baked into the
    actual pixels, and strip the rotation tag so nothing downstream can
    misinterpret it. Also normalizes to h264/aac, matching what Sync Labs
    expects and what the Day 2-3 test clips used successfully.

    Raises VideoNormalizeError if ffmpeg isn't on PATH, the input is
    missing, or the encode fails - never returns a path to a file that
    doesn't actually exist.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise VideoNormalizeError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(input_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac",
        "-metadata:s:v:0", "rotate=0",
        str(output_path),
    ]
    logger.info("Normalizing video: %s -> %s", input_path.name, output_path.name)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise VideoNormalizeError(
            "ffmpeg is not installed or not on PATH - required for video normalization"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoNormalizeError(
            f"ffmpeg normalization timed out after {timeout}s on {input_path.name}"
        ) from exc

    if result.returncode != 0:
        raise VideoNormalizeError(
            f"ffmpeg failed on {input_path.name} (exit {result.returncode}): {result.stderr.strip()}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise VideoNormalizeError(f"Normalization produced no output: {output_path}")

    logger.info("Normalized video written: %s (%d bytes)", output_path, output_path.stat().st_size)
    return output_path
