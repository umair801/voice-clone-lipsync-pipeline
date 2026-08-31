"""
Leading near-silence trim: cuts quiet lead-in from converted voice audio
before it reaches Sync Labs.

Real defect found in Day 6 QA (see project handoff doc, "Day 6 addendum"):
Sync Labs' lipsync model does not appear to distinguish true digital
silence from quiet-but-nonzero audio (room tone / breath carried through
the ElevenLabs Voice Changer output) at the start of a clip. When a
converted track opens with roughly 1-2s of such quiet audio before the
first real word, the model renders continuous mouth motion through that
entire stretch instead of a closed/resting mouth - visible on screen as a
mouth already parted or moving well before any sound a viewer would call
speech.

That root cause is a working theory, not confirmed against Sync Labs'
internals or support (see the project handoff doc's own caveat on this).
The fix here does not depend on the theory being right: removing the
quiet lead-in before the audio reaches Sync Labs removes the ambiguous
input regardless of why the model was actually misbehaving on it.

Runs after voice_conversion, before lipsync, on every converted audio
track - not optional, matching the pattern video_normalize.py uses for
the rotation-metadata fix (see that module's docstring for the reasoning
on "not conditional on looks like it needs it").

Known limitation, stated plainly rather than glossed over: this only
detects and trims a single contiguous quiet block at the very start of
the clip. If a brief loud blip (a click, an early breath spike) sits
inside what should be the lead-in, detection stops at that blip and only
the audio before it is trimmed - the rest of the quiet stretch survives
untrimmed. This has not been observed in this project's actual converted
audio (verified against the real Day 6 addendum clips, which all show one
continuous quiet block), but a future source read with a noisier opening
could hit it. Worth knowing before assuming this fix is bulletproof on
audio it hasn't been tested against.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from core.logging import get_logger

logger = get_logger(__name__)

_SILENCE_END_RE = re.compile(r"silence_end:\s*([\d.]+)")


class AudioTrimError(Exception):
    """Raised when ffmpeg fails to analyze or trim the leading silence."""


def _detect_leading_silence_end(input_path: Path, noise_threshold_db: float, min_silence_duration: float, timeout: int) -> float:
    """
    Run ffmpeg's silencedetect over input_path and return the timestamp
    (seconds) where the first detected quiet block ends, or 0.0 if no
    quiet block is found at all.

    silencedetect writes its findings to stderr regardless of returncode
    (it's a filter, not a failure condition) - a missing/unparseable
    match just means "no leading silence found", not an error.
    """
    detect_cmd = [
        "ffmpeg", "-v", "info", "-nostats",
        "-i", str(input_path),
        "-af", f"silencedetect=noise={noise_threshold_db}dB:d={min_silence_duration}",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(detect_cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise AudioTrimError(
            "ffmpeg is not installed or not on PATH - required for silence trimming"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioTrimError(
            f"ffmpeg silence detection timed out after {timeout}s on {input_path.name}"
        ) from exc

    match = _SILENCE_END_RE.search(result.stderr)
    return float(match.group(1)) if match else 0.0


def _trim_from(input_path: Path, output_path: Path, start_seconds: float, timeout: int) -> None:
    """Re-encode input_path starting at start_seconds into output_path."""
    trim_cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start_seconds:.3f}",
        "-i", str(input_path),
        "-c:a", "libmp3lame", "-q:a", "2",
        str(output_path),
    ]
    try:
        result = subprocess.run(trim_cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise AudioTrimError(
            f"ffmpeg trim timed out after {timeout}s on {input_path.name}"
        ) from exc

    if result.returncode != 0:
        raise AudioTrimError(
            f"ffmpeg failed trimming {input_path.name} (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise AudioTrimError(f"Trim produced no output: {output_path}")


def trim_leading_silence(
    input_path: str | Path,
    output_path: str | Path,
    noise_threshold_db: float = -40.0,
    min_silence_duration: float = 0.15,
    pad_seconds: float = 0.12,
    max_trim_seconds: float = 6.0,
    timeout: int = 60,
) -> Path:
    """
    Detect quiet lead-in at the start of input_path and cut it, leaving a
    small pad_seconds buffer before the first detected sound so the
    attack of the first real word isn't clipped.

    noise_threshold_db: ffmpeg silencedetect threshold. -40dB was picked
        from the Day 6 addendum's measured RMS values on the affected
        clips - the quiet non-speech stretch measured roughly -55dB,
        real speech measured roughly -31dB, so -40dB sits in the gap
        between them. This is a tuned constant against the audio this
        pipeline has actually produced so far, not a law of physics -
        re-derive it if a future clip's noise floor looks different.
    min_silence_duration: a quiet stretch must sustain this long before
        it counts as "leading silence" - avoids trimming a normal brief
        pause at the very start of a clip.
    pad_seconds: kept immediately before the detected silence_end, so
        the cut doesn't land on top of the first phoneme.
    max_trim_seconds: safety ceiling. If ffmpeg reports more leading
        silence than this, something is probably wrong with the input
        (e.g. a genuinely broken conversion) - trimming that much
        automatically is more likely to eat real content than fix a
        lead-in, so this is treated as "nothing to trim" rather than
        trusted blindly.

    If no leading silence is detected (clip opens on real speech, or the
    detected stretch exceeds max_trim_seconds), input_path is copied to
    output_path unchanged - this is a normal outcome, not a failure.

    Raises AudioTrimError if ffmpeg isn't on PATH, the input is missing,
    or either ffmpeg pass fails. See the module docstring for a known
    limitation around blips interrupting the leading quiet block.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise AudioTrimError(f"Input audio not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    silence_end = _detect_leading_silence_end(
        input_path, noise_threshold_db, min_silence_duration, timeout
    )

    if silence_end <= 0.0 or silence_end > max_trim_seconds:
        if silence_end > max_trim_seconds:
            logger.info(
                "Leading silence in %s measured %.2fs, over the %.2fs safety ceiling - "
                "leaving audio untrimmed rather than guessing",
                input_path.name, silence_end, max_trim_seconds,
            )
        else:
            logger.info(
                "No leading silence detected in %s - leaving audio untrimmed", input_path.name
            )
        shutil.copyfile(input_path, output_path)
        return output_path

    trim_start = max(0.0, silence_end - pad_seconds)
    logger.info(
        "Trimming %s: detected %.2fs leading quiet audio, cutting to %.2fs (padded)",
        input_path.name, silence_end, trim_start,
    )
    _trim_from(input_path, output_path, trim_start, timeout)

    logger.info("Wrote trimmed audio: %s (%d bytes)", output_path, output_path.stat().st_size)
    return output_path
