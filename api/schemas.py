"""Pydantic request/response models for the orchestration API."""
from __future__ import annotations

from pydantic import BaseModel


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    result: dict | None = None
    error: str | None = None
    error_stage: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"


class InviteCodeCreateRequest(BaseModel):
    label: str | None = None


class InviteCodeResponse(BaseModel):
    code: str
    label: str | None = None
    created_at: str
    used_at: str | None = None
    used_by_job_id: str | None = None


class InviteCodeListResponse(BaseModel):
    codes: list[InviteCodeResponse]


class CheckoutSessionResponse(BaseModel):
    checkout_url: str
    session_id: str
