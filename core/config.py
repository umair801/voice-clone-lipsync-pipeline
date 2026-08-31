"""
Centralized settings for the AI Avatar Video Pipeline.

Loads from environment variables / .env via pydantic-settings so no API
keys are ever hardcoded in source.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ElevenLabs
    elevenlabs_api_key: str
    elevenlabs_target_voice_id: str

    # Sync Labs (added Day 2-3, declared here now so config stays centralized)
    sync_labs_api_key: str | None = None

    # General
    log_level: str = "INFO"

    # Orchestration (Day 4)
    jobs_output_dir: str = "job_outputs"
    incoming_dir: str = "incoming"
    incoming_poll_interval_seconds: int = 30
    # Empty by default (no cross-origin frontend yet) - restrictive per
    # enterprise standard. Override in .env as a JSON array once a
    # frontend needs to call this API cross-origin.
    cors_allowed_origins: list[str] = []

    # API hardening (Day 6)
    # If set, POST /jobs requires this exact value in the X-API-Key
    # header. None by default so local dev/testing keeps working without
    # extra setup - set it in .env before exposing this server beyond
    # localhost. This is a shared-secret check, not real auth (no
    # per-user identity, no rotation) - adequate for a single-operator
    # tool, not a substitute for real auth on a multi-tenant deployment.
    pipeline_api_key: str | None = None
    # Reject an upload once its body exceeds this many bytes, checked
    # while streaming to disk (not just Content-Length, which a client
    # can lie about). 100MB covers real short-form video/audio with
    # headroom without letting an unbounded upload fill the disk.
    max_upload_bytes: int = 100 * 1024 * 1024


settings = Settings()  # type: ignore[call-arg]
