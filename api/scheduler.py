"""
APScheduler hook: polls an "incoming" folder for new script/audio +
reference video pairs and auto-submits them as pipeline jobs, per the
build spec's "Trigger: schedule (cron via APScheduler) or webhook"
requirement.

Convention: a pending pair is two files sharing a basename before the
"_audio"/"_video" suffix - <name>_audio.<ext> and <name>_video.<ext> -
dropped into settings.incoming_dir together. Once both exist, the pair is
submitted as a job and moved into a "processed" subfolder (never deleted -
matches this project's practice of not running destructive operations
without being asked, and leaves an audit trail of what's already run).

Reuses the exact same _run_job function the API endpoint calls, not a
second implementation of "how a job runs" - api/routes.py owns that
function since it's the primary entry point; the scheduler just calls it.

Note: poll_incoming_folder blocks for the full duration of any pipeline
run it kicks off (voice conversion + lipsync can take minutes). APScheduler's
BackgroundScheduler does not run overlapping instances of the same job id
by default, so a long-running pipeline naturally serializes the next poll
tick rather than piling up concurrent runs - this is intentional, not a
missed edge case: two pipeline runs racing over the same job_id/output_dir
would be worse.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from core.config import settings
from core.job_store import job_store
from core.logging import get_logger

logger = get_logger(__name__)

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a"}
VIDEO_SUFFIXES = {".mp4", ".mov"}


def _find_pending_pairs(incoming_dir: Path) -> list[tuple[str, Path, Path]]:
    audio_files = {
        f.stem[: -len("_audio")]: f
        for f in incoming_dir.glob("*_audio.*")
        if f.suffix.lower() in AUDIO_SUFFIXES
    }
    video_files = {
        f.stem[: -len("_video")]: f
        for f in incoming_dir.glob("*_video.*")
        if f.suffix.lower() in VIDEO_SUFFIXES
    }
    common = sorted(set(audio_files) & set(video_files))
    return [(name, audio_files[name], video_files[name]) for name in common]


def poll_incoming_folder() -> None:
    # Imported here, not at module load, to avoid a circular import with
    # api.routes (both need _run_job; routes owns it since it's the
    # primary entry point, this module just reuses it).
    from api.routes import _run_job

    incoming_dir = Path(settings.incoming_dir)
    incoming_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = incoming_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    pairs = _find_pending_pairs(incoming_dir)
    if not pairs:
        return

    for name, audio_path, video_path in pairs:
        job = job_store.create()
        logger.info("Scheduler found pending pair %r -> job %s", name, job.job_id)
        try:
            _run_job(job.job_id, str(audio_path), str(video_path))
        finally:
            # Move source files out of the watch folder regardless of
            # outcome, so a failed job doesn't get resubmitted forever on
            # every poll tick.
            shutil.move(str(audio_path), str(processed_dir / audio_path.name))
            shutil.move(str(video_path), str(processed_dir / video_path.name))


def start_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        poll_incoming_folder,
        "interval",
        seconds=settings.incoming_poll_interval_seconds,
        id="incoming_folder_poll",
    )
    scheduler.start()
    logger.info(
        "Scheduler started: polling %s every %ss",
        settings.incoming_dir, settings.incoming_poll_interval_seconds,
    )
    return scheduler
