"""
Day 2-3 deliverable script: run the Day 1 converted audio against each
reference video clip in video_samples/input/ through Sync Labs lipsync-2,
and report honestly on what comes back.

Usage:
    1. Record reference video clips per the Day 2-3 recording brief and
       save them as video_samples/input/*.mp4 (or .mov). Keep each clip
       under ~20s - the free tier caps generations at 20s (15s for sync-3).
    2. Fill in .env with SYNC_LABS_API_KEY.
    3. By default this is a DRY RUN - it lists what it would submit and
       the free-tier budget impact, but does not call the API. This is
       deliberate: the free tier is 3 generations/month total, an
       accidental full run burns a third of a month's quota in one command.
       Review the list, then re-run with --confirm to actually submit.

         python -m tests.run_lipsync_test              # dry run
         python -m tests.run_lipsync_test --confirm     # submits for real
         python -m tests.run_lipsync_test --confirm --only front_facing.mp4

    4. Outputs land in video_samples/output/, one file per input clip.

This script does NOT declare success from API status alone. A generation
coming back COMPLETED means Sync Labs finished the job - it does not mean
the lipsync is usable. Every COMPLETED result is flagged for manual
watch-through before being called done, per the project's standing rule:
don't claim it survived scrutiny until it actually has.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.tools.sync_labs_lipsync import (  # noqa: E402
    LipsyncOptions,
    SyncLabsConfigError,
    SyncLabsError,
    SyncLabsGenerationFailedError,
    SyncLabsLipsyncClient,
    SyncLabsTimeoutError,
)
from core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

VIDEO_INPUT_DIR = Path("video_samples/input")
VIDEO_OUTPUT_DIR = Path("video_samples/output")

# Default audio: the Day 1 chosen setting (stab0.70/sim0.85). Override with
# --audio if testing a different converted track.
DEFAULT_AUDIO = Path("audio_samples/output/test_read__stab0.70_sim0.85_style0.00.mp3")

FREE_TIER_GENERATIONS_PER_MONTH = 3
FREE_TIER_MAX_DURATION_SECONDS = 20


def find_video_clips(only: str | None) -> list[Path]:
    clips = sorted(VIDEO_INPUT_DIR.glob("*.mp4")) + sorted(VIDEO_INPUT_DIR.glob("*.mov"))
    if only:
        clips = [c for c in clips if c.name == only]
    if not clips:
        raise FileNotFoundError(
            f"No video clips found in {VIDEO_INPUT_DIR}/ "
            f"(looked for *.mp4 and *.mov{f', filtered to {only}' if only else ''}). "
            f"Record the reference clips from the Day 2-3 brief first."
        )
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm", action="store_true",
        help="Actually submit to the Sync Labs API. Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--audio", type=Path, default=DEFAULT_AUDIO,
        help=f"Converted audio to lipsync onto every clip (default: {DEFAULT_AUDIO})",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Filename (not path) of a single clip in video_samples/input/ to run, "
        "instead of every clip found.",
    )
    parser.add_argument(
        "--model", type=str, default="lipsync-2",
        help="lipsync-2 (default, iteration) or lipsync-2-pro (demo/QA-critical).",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Audio input not found: {args.audio}")
        print("Run tests/run_voice_changer_sweep.py first, or pass --audio explicitly.")
        sys.exit(1)

    clips = find_video_clips(args.only)

    print("\n=== Sync Labs lipsync run plan ===")
    print(f"Audio track: {args.audio}")
    print(f"Model:       {args.model}")
    print(f"Clips found: {len(clips)}")
    for c in clips:
        print(f"  - {c}")
    print(
        f"\nFree tier budget: {FREE_TIER_GENERATIONS_PER_MONTH} generations/month, "
        f"{FREE_TIER_MAX_DURATION_SECONDS}s max duration each. "
        f"This run would use {len(clips)} of that {FREE_TIER_GENERATIONS_PER_MONTH}."
    )

    if not args.confirm:
        print("\nDRY RUN - nothing submitted. Re-run with --confirm to actually call the API.")
        return

    if len(clips) > FREE_TIER_GENERATIONS_PER_MONTH:
        print(
            f"\nERROR: {len(clips)} clips exceeds the free tier's "
            f"{FREE_TIER_GENERATIONS_PER_MONTH}/month cap. Use --only to run one clip at a "
            f"time, or upgrade the Sync Labs plan before proceeding."
        )
        sys.exit(1)

    try:
        client = SyncLabsLipsyncClient()
    except SyncLabsConfigError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    options = LipsyncOptions(model=args.model)
    results = []

    for clip in clips:
        out_path = VIDEO_OUTPUT_DIR / f"{clip.stem}__{options.model}.mp4"
        print(f"\nSubmitting {clip.name} ...")
        try:
            result = client.generate_from_files(clip, args.audio, out_path, options=options)
            results.append((clip.name, "COMPLETED (UNREVIEWED)", str(result.output_path)))
        except SyncLabsGenerationFailedError as exc:
            logger.error("%s: generation failed - %s", clip.name, exc)
            results.append((clip.name, f"FAILED: {exc}", "-"))
        except SyncLabsTimeoutError as exc:
            logger.error("%s: %s", clip.name, exc)
            results.append((clip.name, f"TIMEOUT: {exc}", "-"))
        except SyncLabsError as exc:
            logger.error("%s: %s", clip.name, exc)
            results.append((clip.name, f"ERROR: {exc}", "-"))

    print("\n=== Sync Labs lipsync run complete ===")
    for name, status, path in results:
        print(f"  [{status}] {name} -> {path}")

    print(
        "\nCOMPLETED (UNREVIEWED) means the Sync Labs API finished the job, nothing more. "
        "Watch each output file at full attention before calling any of it done - check for "
        "mouth/jaw artifacts, audio drift, and whether the known failure mode (side angle / "
        "low light) shows up on the clips that were meant to test it. Report findings "
        "honestly, including on clips that look bad."
    )


if __name__ == "__main__":
    main()
