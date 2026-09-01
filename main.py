"""
FastAPI application entrypoint for the AI Avatar Video Pipeline.

Run locally:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /jobs           submit a job (multipart: source_audio, reference_video)
    GET  /jobs/{job_id}   poll job status/result
    GET  /health          liveness check
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.demo import router as demo_router
from api.routes import router
from api.scheduler import start_scheduler
from core.logging import get_logger
from core.config import settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = start_scheduler()
    logger.info("AI Avatar Video Pipeline API starting up")
    yield
    scheduler.shutdown(wait=False)
    logger.info("AI Avatar Video Pipeline API shutting down")


app = FastAPI(
    title="AI Avatar Video Pipeline",
    description="Voice-clone + lipsync content automation pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

# Restrictive by default (no origins allowed) - override CORS_ALLOWED_ORIGINS
# in .env as a JSON array (e.g. '["https://app.example.com"]') once a
# frontend actually needs to call this API cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(demo_router)

# Public demo frontend (Phase 1) - served from the same origin as the
# API, so no CORS configuration is needed for it specifically. html=True
# serves frontend/index.html at "/".
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
