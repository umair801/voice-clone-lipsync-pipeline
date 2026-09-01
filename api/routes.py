"""
FastAPI routes for the AI Avatar Video Pipeline orchestration layer.

POST /jobs           submit a source audio read + reference video, runs the
                      full voice-conversion -> normalize -> lipsync -> QA
                      pipeline in the background, returns a job_id
                      immediately (202 Accepted). Gated by X-API-Key when
                      PIPELINE_API_KEY is set - this is the internal /
                      programmatic entry point, distinct from the
                      invite-code-gated public demo path in api/demo.py.
GET  /jobs/{job_id}   poll job status/result. Shared by both entry points -
                      a demo-submitted job and an internal one are polled
                      the same way, since job_store doesn't distinguish
                      how a job was created.
GET  /health          liveness check.
"""
from __future__ import annotations

import secrets

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.schemas import HealthResponse, JobStatusResponse, JobSubmitResponse
from core.config import settings
from core.job_runner import accept_upload_job
from core.job_store import job_store
from core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


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


@router.post("/jobs", response_model=JobSubmitResponse, status_code=202)
async def submit_job(
    background_tasks: BackgroundTasks,
    source_audio: UploadFile = File(..., description="Creator's raw recorded read (mp3/wav/m4a)"),
    reference_video: UploadFile = File(..., description="Real reference footage of the speaker (mp4/mov)"),
    x_api_key: str | None = Header(default=None),
) -> JobSubmitResponse:
    _require_api_key(x_api_key)
    job = await accept_upload_job(background_tasks, source_audio, reference_video)
    return JobSubmitResponse(job_id=job.job_id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job found with id {job_id}")
    return JobStatusResponse(**job_store.to_dict(job))


@router.get("/jobs/{job_id}/output")
async def get_job_output(job_id: str) -> FileResponse:
    """
    Streams the finished lipsync output video for a completed job, so a
    frontend can play/download it directly rather than needing
    filesystem access to the server. 404s until the job is actually
    completed and has a result - a queued/running/failed job has nothing
    to serve here.
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job found with id {job_id}")
    if job.status != "completed" or not job.result or not job.result.get("lipsync_output_path"):
        raise HTTPException(409, f"Job {job_id} has no output yet (status={job.status!r})")
    output_path = Path(job.result["lipsync_output_path"])
    if not output_path.exists():
        raise HTTPException(410, f"Output file for job {job_id} is no longer available on disk")
    return FileResponse(output_path, media_type="video/mp4", filename=output_path.name)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()
