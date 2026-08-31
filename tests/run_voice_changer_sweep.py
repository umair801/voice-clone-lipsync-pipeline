"""
Day 1 deliverable script: run a single recorded test read through
ElevenLabs Voice Changer across several stability/clarity/style
combinations so you can A/B listen and pick a setting.

Usage:
    1. Record a 60-90s test read of your own voice.
    2. Save it as audio_samples/input/test_read.mp3 (or .wav).
    3. Fill in .env with ELEVENLABS_API_KEY and ELEVENLABS_TARGET_VOICE_ID.
    4. Run:  python -m tests.run_voice_changer_sweep

Outputs land in audio_samples/output/, one file per parameter
combination, named so you can tell them apart by ear.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.tools.elevenlabs_voice_changer import (  # noqa: E402
    ElevenLabsVoiceChanger,
    VoiceChangerError,
    VoiceChangerParams,
)
from core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

INPUT_DIR = Path("audio_samples/input")
OUTPUT_DIR = Path("audio_samples/output")

# Sweep grid: covers the range worth listening to before committing to a
# default. Keep this small on purpose - each combo costs API minutes.
SWEEP_PARAMS = [
    VoiceChangerParams(stability=0.3, similarity_boost=0.75, style=0.0),   # expressive, close to original
    VoiceChangerParams(stability=0.5, similarity_boost=0.75, style=0.0),   # balanced (ElevenLabs default-ish)
    VoiceChangerParams(stability=0.7, similarity_boost=0.85, style=0.0),   # more corrected, more consistent
    VoiceChangerParams(stability=0.5, similarity_boost=0.75, style=0.15),  # slight style push
]


def find_input_file() -> Path:
    candidates = sorted(INPUT_DIR.glob("test_read.*"))
    if not candidates:
        raise FileNotFoundError(
            f"No test read found in {INPUT_DIR}/. "
            f"Save your 60-90s recording as {INPUT_DIR}/test_read.mp3 (or .wav) and re-run."
        )
    return candidates[0]


def main() -> None:
    input_file = find_input_file()
    logger.info("Using input read: %s", input_file)

    changer = ElevenLabsVoiceChanger()
    results = []

    for params in SWEEP_PARAMS:
        out_path = OUTPUT_DIR / f"{input_file.stem}__{params.label()}.mp3"
        try:
            changer.convert(input_file, out_path, params=params)
            results.append((params.label(), out_path, "ok"))
        except VoiceChangerError as exc:
            logger.error("Skipping %s due to error: %s", params.label(), exc)
            results.append((params.label(), out_path, f"FAILED: {exc}"))

    print("\n=== Voice Changer sweep complete ===")
    for label, path, status in results:
        print(f"  [{status}] {label} -> {path}")
    print(
        "\nListen to each file in audio_samples/output/ and pick the one that best "
        "preserves your natural delivery while correcting what you wanted corrected. "
        "That combination becomes the pipeline default in core/config.py or a new "
        "constant in elevenlabs_voice_changer.py."
    )


if __name__ == "__main__":
    main()
