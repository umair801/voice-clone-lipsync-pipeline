"""
Shared job-execution helpers used by both the internal API (api/routes.py)
and the public demo API (api/demo.py).

Pulled out during the Phase 1 demo-frontend build so the invite-code-gated
submit path and the original API-key-gated submit path run the exact same
upload-handling and pipeline-execution code, rather than two near-duplicate
copies that could quietly drift apart.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import BackgroundTasks, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from agents.orchestrator import run_pipeline
from core.config import settings
from core.job_store import Job, job_store
from core.logging import get_logger

logger = get_logger(__name__)

ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".m4a"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov"}


def save_upload(upload: UploadFile, dest: Path) -> Path:
    """
    Stream the upload to disk in bounded chunks, aborting once the body
    exceeds settings.max_upload_bytes. Checked against actual bytes
    written, not the client-supplied Content-Length header, which a
    client can omit or lie about - this is the real backstop.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    chunk_size = 1024 * 1024
    with open(dest, "wb") as f:
        while True:
            chunk = upload.file.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > settings.max_upload_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413,
                    f"{upload.filename!r} exceeds the "
                    f"{settings.max_upload_bytes / 1024 / 1024:.0f}MB upload limit",
                )
            f.write(chunk)
    return dest


def validate_extensions(audio_filename: str | None, video_filename: str | None) -> tuple[str, str]:
    audio_ext = Path(audio_filename or "").suffix.lower()
    video_ext = Path(video_filename or "").suffix.lower()
    if audio_ext not in ALLOWED_AUDIO_EXT:
        raise HTTPException(400, f"Unsupported audio format {audio_ext!r}, expected one of {sorted(ALLOWED_AUDIO_EXT)}")
    if video_ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(400, f"Unsupported video format {video_ext!r}, expected one of {sorted(ALLOWED_VIDEO_EXT)}")
    return audio_ext, video_ext


def run_job(job_id: str, source_audio_path: str, reference_video_path: str) -> None:
    """
    Runs the full pipeline for one job and writes the result into
    job_store. Shared by the internal submit endpoint, the demo submit
    endpoint, and the APScheduler folder-watch hook.
    """
    output_dir = Path(settings.jobs_output_dir) / job_id
    try:
        final_state = run_pipeline(
            job_id=job_id,
            source_audio_path=source_audio_path,
            reference_video_path=reference_video_path,
            output_dir=str(output_dir),
        )
    except Exception as exc:  # noqa: BLE001 - last-resort guard, a background task must never die silently
        logger.error("[%s] Pipeline crashed outside expected error handling: %s", job_id, exc)
        job_store.update(job_id, status="failed", error=str(exc), error_stage="unexpected")
        return

    job_store.update(
        job_id,
        status=final_state["status"],
        error=final_state.get("error"),
        error_stage=final_state.get("error_stage"),
        result={
            "lipsync_output_path": final_state.get("lipsync_output_path"),
            "converted_audio_path": final_state.get("converted_audio_path"),
            "normalized_video_path": final_state.get("normalized_video_path"),
            "qa": final_state.get("qa"),
        },
    )


async def accept_upload_job(
    background_tasks: BackgroundTasks,
    source_audio: UploadFile,
    reference_video: UploadFile,
) -> Job:
    """
    Shared submit flow: validate extensions, create the job record,
    stream both uploads to disk off the event loop, schedule the
    background pipeline run, and clean up on a rejected upload. Returns
    the created Job (job_store.Job) so callers can build their own
    response shape (internal vs demo responses differ slightly).
    """
    audio_ext, video_ext = validate_extensions(source_audio.filename, reference_video.filename)

    job = job_store.create()
    input_dir = Path(settings.jobs_output_dir) / job.job_id / "input"
    try:
        audio_path = await run_in_threadpool(save_upload, source_audio, input_dir / f"source_audio{audio_ext}")
        video_path = await run_in_threadpool(save_upload, reference_video, input_dir / f"reference_video{video_ext}")
    except HTTPException:
        job_store.delete(job.job_id)
        shutil.rmtree(input_dir, ignore_errors=True)
        raise

    background_tasks.add_task(run_job, job.job_id, str(audio_path), str(video_path))
    logger.info("Job %s submitted: audio=%s video=%s", job.job_id, audio_path.name, video_path.name)
    return job
