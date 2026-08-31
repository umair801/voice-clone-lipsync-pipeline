"""
FastAPI routes for the AI Avatar Video Pipeline orchestration layer.

POST /jobs           submit a source audio read + reference video, runs the
                      full voice-conversion -> normalize -> lipsync -> QA
                      pipeline in the background, returns a job_id
                      immediately (202 Accepted).
GET  /jobs/{job_id}   poll job status/result.
GET  /health          liveness check.
"""
from __future__ import annotations

import secrets
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from agents.orchestrator import run_pipeline
from api.schemas import HealthResponse, JobStatusResponse, JobSubmitResponse
from core.config import settings
from core.job_store import job_store
from core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".m4a"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov"}


def _save_upload(upload: UploadFile, dest: Path) -> Path:
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


def _require_api_key(x_api_key: str | None) -> None:
    """
    Shared-secret check for POST /jobs. A no-op when PIPELINE_API_KEY is
    unset (local dev default) - set it in .env before this server is
    reachable from anywhere but localhost. This is intentionally simple
    (one static key, no per-caller identity or rotation) - adequate for
    a single-operator tool, not a substitute for real auth if this ever
    serves multiple clients.
    """
    if settings.pipeline_api_key is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.pipeline_api_key):
        raise HTTPException(401, "Missing or invalid X-API-Key header")


def _run_job(job_id: str, source_audio_path: str, reference_video_path: str) -> None:
    """
    Runs the full pipeline for one job and writes the result into
    job_store. Shared by both the API submit endpoint (via
    BackgroundTasks) and the APScheduler folder-watch hook - one place
    that decides how a submitted job actually executes.
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


@router.post("/jobs", response_model=JobSubmitResponse, status_code=202)
async def submit_job(
    background_tasks: BackgroundTasks,
    source_audio: UploadFile = File(..., description="Creator's raw recorded read (mp3/wav/m4a)"),
    reference_video: UploadFile = File(..., description="Real reference footage of the speaker (mp4/mov)"),
    x_api_key: str | None = Header(default=None),
) -> JobSubmitResponse:
    _require_api_key(x_api_key)
    audio_ext = Path(source_audio.filename or "").suffix.lower()
    video_ext = Path(reference_video.filename or "").suffix.lower()
    if audio_ext not in ALLOWED_AUDIO_EXT:
        raise HTTPException(400, f"Unsupported audio format {audio_ext!r}, expected one of {sorted(ALLOWED_AUDIO_EXT)}")
    if video_ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(400, f"Unsupported video format {video_ext!r}, expected one of {sorted(ALLOWED_VIDEO_EXT)}")

    job = job_store.create()
    input_dir = Path(settings.jobs_output_dir) / job.job_id / "input"
    try:
        # Streamed to disk in bounded chunks (see _save_upload) - genuinely
        # blocking I/O, so it's pushed off the event loop rather than run
        # inline in this async route, which would otherwise stall every
        # other in-flight request (including /health) for the duration of
        # each upload.
        audio_path = await run_in_threadpool(_save_upload, source_audio, input_dir / f"source_audio{audio_ext}")
        video_path = await run_in_threadpool(_save_upload, reference_video, input_dir / f"reference_video{video_ext}")
    except HTTPException:
        # A rejected (e.g. oversized) upload must not leave a dead job
        # record sitting in the store forever, or a partial input dir on
        # disk from whichever file did save successfully.
        job_store.delete(job.job_id)
        shutil.rmtree(input_dir, ignore_errors=True)
        raise

    background_tasks.add_task(_run_job, job.job_id, str(audio_path), str(video_path))
    logger.info("Job %s submitted: audio=%s video=%s", job.job_id, audio_path.name, video_path.name)

    return JobSubmitResponse(job_id=job.job_id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job found with id {job_id}")
    return JobStatusResponse(**job_store.to_dict(job))


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()
