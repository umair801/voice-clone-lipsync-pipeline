"""
QA / validation checks run on a completed lipsync output before it's
marked ready to deliver.

Per the build spec, the full QA stage is: automated duration-match and
face-detection-confidence checks, plus artifact flagging, then a manual
review gate (human-in-the-loop) before publish.

- Duration check (Day 4): implemented for real - cheap, deterministic,
  catches a concrete failure mode (a corrupted/near-empty input producing
  a much shorter output than expected).
- Face-detection confidence (Day 6): implemented for real - see
  `_face_presence_rate`. It answers one honest, narrow question: across a
  sample of frames, was a face detected at all? It catches a gross
  failure (the model returned a blank, corrupted, or wildly wrong clip),
  nothing subtler. A Haar cascade gives presence, not a calibrated
  per-frame confidence score - "confidence" in the result is the fraction
  of sampled frames with a detected face, not a probability from the
  detector itself. Documented as exactly that, not oversold.

  Real bug caught while building this: checking only the frontal-face
  cascade produced false negatives on the Day 5 demo clip (clip2 in the
  Day 2-3 handoff) - it's a deliberately off-angle test clip, and a
  frontal-only cascade misses a genuinely fine, undistorted face that is
  simply turned. Confirmed by manually inspecting the flagged frames
  before trusting the check's own output. Fixed by also trying the
  profile-face cascade (both the frame and its horizontal flip, since the
  bundled profile cascade only reliably matches one facing direction) -
  a frame counts as having a face if either cascade finds one.
- Artifact flagging: still NOT implemented, and Day 6 confirmed why this
  is a real gap, not laziness. The Day 5 demo clip had a genuine
  mouth/jaw-boundary blending artifact from the lipsync provider - it was
  only caught by a human noticing it on a full-attention watch, then
  confirmed by manually diffing frames against the source footage at the
  same timestamp. No automated technique for that class of artifact
  (frame-to-frame boundary flicker distinct from legitimate motion blur)
  was built here; doing it reliably is a real computer-vision problem,
  not a stub that should pretend to be more than a placeholder.
  Stays `"not_implemented"` rather than a fake passing score.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from core.logging import get_logger

logger = get_logger(__name__)

# How far apart (seconds) the reference video's duration and the final
# lipsync output's duration can be before we flag it for manual review.
# sync_mode="cut_off" trims to the shorter of video/audio, so under normal
# operation output duration should closely track input video duration,
# not the often-much-longer converted audio track.
DURATION_TOLERANCE_SECONDS = 1.0

# Face-presence sampling: check this many evenly-spaced frames across the
# output. 10 is enough to catch "the model lost the face for a stretch of
# the clip" without re-decoding every frame of a longer video.
FACE_CHECK_SAMPLE_FRAMES = 10
# Below this fraction of sampled frames having a detected face, flag for
# manual review - a real lipsync output should have a detectable face in
# nearly every frame; a low rate means something is seriously wrong
# (wrong content, severe corruption, a face that left frame and didn't
# return), not a benign edge case.
FACE_PRESENCE_MIN_OK_RATE = 0.9


class QACheckError(Exception):
    """Raised when a QA check itself cannot run (not the same as a failed check)."""


@dataclass
class QAResult:
    passed: bool
    needs_review: bool
    duration_check: dict = field(default_factory=dict)
    face_detection_confidence: dict | str = "not_implemented"
    artifact_flagging: str = "not_implemented"
    notes: list[str] = field(default_factory=list)


def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise QACheckError(f"ffprobe failed on {path}: {exc}") from exc
    if result.returncode != 0:
        raise QACheckError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise QACheckError(f"Could not parse duration for {path}: {exc}") from exc


def _detect_face(cv2_module, gray_frame, frontal_cascade, profile_cascade) -> bool:
    """
    True if a face is found by the frontal cascade, the profile cascade,
    or the profile cascade on the horizontally-flipped frame (the
    bundled profile cascade is trained on one facing direction only, so
    checking the flip catches the other). Three cheap checks, not a
    guarantee - still just Haar cascades, not a robust pose-invariant
    detector - but this closes the specific false-negative this project
    actually hit on off-angle footage, not a hypothetical one.
    """
    if len(frontal_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(60, 60))) > 0:
        return True
    if len(profile_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(60, 60))) > 0:
        return True
    flipped = cv2_module.flip(gray_frame, 1)
    return len(profile_cascade.detectMultiScale(flipped, 1.1, 5, minSize=(60, 60))) > 0


def _face_presence_rate(path: Path, duration_s: float, sample_frames: int = FACE_CHECK_SAMPLE_FRAMES) -> dict:
    """
    Extract `sample_frames` evenly-spaced frames from the output and run
    face detection (frontal + profile cascades, see `_detect_face`) on
    each. Returns the fraction with a detected face, plus the raw counts
    - a coarse but real signal against gross failure, not a substitute
    for actually watching the clip.

    Raises QACheckError only if the extraction/detection machinery itself
    fails (ffmpeg missing, no frames extracted, cascade files missing) -
    a clip where faces are genuinely undetected in every frame is a valid
    (bad) *result*, not a check failure.
    """
    import cv2
    import numpy as np
    import tempfile

    if duration_s <= 0:
        raise QACheckError(f"Cannot sample frames from a {duration_s}s clip: {path}")

    frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    if frontal_cascade.empty() or profile_cascade.empty():
        raise QACheckError("Failed to load OpenCV's bundled face-detection cascades")

    detections = 0
    checked = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Evenly spaced timestamps, staying clear of the very first/last
        # frame where a fade or a slightly-off seek is more likely.
        margin = min(0.5, duration_s * 0.05)
        timestamps = np.linspace(margin, max(margin, duration_s - margin), sample_frames)
        for i, ts in enumerate(timestamps):
            frame_path = tmp_dir / f"f_{i:03d}.jpg"
            cmd = [
                "ffmpeg", "-y", "-v", "error", "-ss", f"{ts:.3f}",
                "-i", str(path), "-frames:v", "1", str(frame_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0 or not frame_path.exists():
                continue
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            checked += 1
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if _detect_face(cv2, gray, frontal_cascade, profile_cascade):
                detections += 1

    if checked == 0:
        raise QACheckError(f"Could not extract any sample frames from {path}")

    rate = detections / checked
    return {
        "frames_checked": checked,
        "frames_with_face": detections,
        "rate": round(rate, 2),
        "ok": rate >= FACE_PRESENCE_MIN_OK_RATE,
    }


def run_qa(reference_video_path: str | Path, lipsync_output_path: str | Path) -> QAResult:
    """
    Run the Day 6 QA checks on a completed lipsync output.

    Returns a QAResult - never raises for a failed check (a duration
    mismatch or low face-presence rate is a normal, expected outcome that
    routes to manual review, not an exceptional condition). Only raises
    QACheckError if the checks themselves can't execute (ffprobe/ffmpeg
    missing, files unreadable, cascade file missing).
    """
    reference_video_path = Path(reference_video_path)
    lipsync_output_path = Path(lipsync_output_path)

    ref_duration = _probe_duration(reference_video_path)
    out_duration = _probe_duration(lipsync_output_path)
    delta = abs(ref_duration - out_duration)
    duration_ok = delta <= DURATION_TOLERANCE_SECONDS

    face_result = _face_presence_rate(lipsync_output_path, out_duration)

    notes = []
    if not duration_ok:
        notes.append(
            f"Output duration ({out_duration:.2f}s) differs from reference video "
            f"duration ({ref_duration:.2f}s) by {delta:.2f}s, over the "
            f"{DURATION_TOLERANCE_SECONDS}s tolerance."
        )
    if not face_result["ok"]:
        notes.append(
            f"Face detected in only {face_result['frames_with_face']}/"
            f"{face_result['frames_checked']} sampled frames "
            f"({face_result['rate']:.0%}), below the "
            f"{FACE_PRESENCE_MIN_OK_RATE:.0%} threshold."
        )

    overall_ok = duration_ok and face_result["ok"]

    result = QAResult(
        passed=overall_ok,
        needs_review=not overall_ok,
        duration_check={
            "reference_duration_s": round(ref_duration, 2),
            "output_duration_s": round(out_duration, 2),
            "delta_s": round(delta, 2),
            "tolerance_s": DURATION_TOLERANCE_SECONDS,
            "ok": duration_ok,
        },
        face_detection_confidence=face_result,
        notes=notes,
    )
    logger.info(
        "QA result for %s: passed=%s needs_review=%s duration_delta=%.2fs face_rate=%.2f",
        lipsync_output_path.name, result.passed, result.needs_review, delta, face_result["rate"],
    )
    return result
