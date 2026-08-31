"""
Sync Labs (sync.so) lipsync client.

Wraps the Sync Labs Generations API to fuse a corrected audio track (Day 1's
ElevenLabs Voice Changer output) onto real reference video footage of the
speaker. Model default is `lipsync-2` (balanced quality/cost); pass
`lipsync-2-pro` in LipsyncOptions for the client demo or any QA-critical
delivery run, per the build spec.

Known failure mode (documented in the build spec, NOT a bug in this
client): side-angle or low-light reference footage produces poor lipsync
quality. This client cannot correct for that with retries or parameter
tuning - it is a property of the input footage. A generation can come back
`COMPLETED` from the API and still be unusable output. Callers MUST review
the actual video before reporting success; this client only reports API
status, never visual/perceptual quality.

Docs: https://sync.so/docs/quickstart ,
      https://sync.so/docs/api-reference/api-overview
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from sync import Sync
from sync.common import Audio, GenerationOptions, Video
from sync.core.api_error import ApiError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# create_with_files (direct multipart upload) is capped by Sync Labs at
# 20MB per file. Larger inputs need the presigned asset-upload flow
# (POST /v2/assets/upload -> PUT bytes -> POST /v2/assets -> reference by
# assetId) which this client does not implement yet - it isn't needed for
# short test/demo clips. If a future clip needs it, that's the extension
# point, not a reason to silently fail here.
MAX_DIRECT_UPLOAD_BYTES = 20 * 1024 * 1024

# Sync Labs generation objects settle into one of these; only COMPLETED
# means an output_url exists to download.
_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "REJECTED"}


class SyncLabsError(Exception):
    """Base error for all Sync Labs lipsync failures."""


class SyncLabsConfigError(SyncLabsError):
    """Raised when the client is missing required configuration (API key)."""


class SyncLabsInputError(SyncLabsError):
    """Raised when the input files fail local validation before any API call."""


class SyncLabsGenerationFailedError(SyncLabsError):
    """Raised when a generation reaches FAILED or REJECTED status."""

    def __init__(self, generation_id: str, status: str, error: str | None, error_code: str | None):
        self.generation_id = generation_id
        self.status = status
        self.error = error
        self.error_code = error_code
        super().__init__(
            f"Generation {generation_id} ended in {status}"
            f"{f' ({error_code})' if error_code else ''}: {error or 'no error detail returned'}"
        )


class SyncLabsTimeoutError(SyncLabsError):
    """Raised when a generation does not reach a terminal status within max_wait."""

    def __init__(self, generation_id: str, waited_seconds: float):
        self.generation_id = generation_id
        self.waited_seconds = waited_seconds
        super().__init__(
            f"Generation {generation_id} did not finish within {waited_seconds:.0f}s "
            f"(still not COMPLETED/FAILED/REJECTED - job may still be running server-side, "
            f"check the Sync Labs dashboard before resubmitting)"
        )


def _is_retryable_api_error(exc: BaseException) -> bool:
    """
    Retry on transient/server-side failures only.

    A validation error (bad request, no face detected in the reference
    frame, unsupported format) is a 4xx ApiError from Sync Labs and will
    not succeed on retry - retrying it just burns free-tier generation
    quota for nothing. Only retry on 5xx ApiErrors.

    Any other exception type reaching here (DNS failure, connection
    reset, read timeout) is a transport-level failure from the SDK's
    underlying HTTP client, not an application-level rejection - always
    worth a retry. We don't assume it's `requests` specifically since the
    Sync SDK may use a different HTTP client internally (the ElevenLabs
    SDK, for comparison, uses httpx under the hood - see Day 1 notes).
    """
    if isinstance(exc, ApiError):
        status = getattr(exc, "status_code", None)
        return status is not None and status >= 500
    return True


@dataclass(frozen=True)
class LipsyncOptions:
    """
    Tunable parameters for a single lipsync generation.

    model: "lipsync-2" (default, ~$0.05/sec) for iteration/testing,
        "lipsync-2-pro" (~$0.067-0.083/sec) for the client demo or any
        QA-critical delivery. "sync-3" is Sync Labs' newest/most powerful
        model but the free tier caps it at 1 generation/15s max - use only
        deliberately, not as the default during iteration.
    sync_mode: how Sync Labs reconciles mismatched audio/video length.
        "cut_off" trims to the shorter of the two - the safe default for
        short test clips where you've kept audio and video close in
        length already.
    output_file_name: label shown in the Sync Labs dashboard; purely
        cosmetic, does not affect the local output path. Only applies to
        generate_from_urls() - the underlying create_with_files() call used
        by generate_from_files() doesn't accept this field, so it's a
        no-op there.
    """

    model: str = "lipsync-2"
    sync_mode: str = "cut_off"
    output_file_name: str | None = None


@dataclass(frozen=True)
class LipsyncResult:
    """What actually happened, for honest downstream reporting."""

    generation_id: str
    status: str
    output_path: Path
    output_duration_seconds: float | None
    model: str


class SyncLabsLipsyncClient:
    """Thin, retry-hardened wrapper around the Sync Labs Generations API."""

    # This is the ceiling for the polling loop (a lipsync generation is an
    # async server-side job, typically tens of seconds to a few minutes for
    # a short clip), not a per-HTTP-call timeout like ElevenLabsVoiceChanger's
    # `timeout`. Named consistently with that client for a familiar
    # constructor signature; documented here so it isn't misread.
    DEFAULT_MAX_WAIT_SECONDS = 600
    DEFAULT_POLL_INTERVAL_SECONDS = 10

    def __init__(
        self,
        api_key: str | None = None,
        max_wait: int = DEFAULT_MAX_WAIT_SECONDS,
        poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._api_key = api_key or settings.sync_labs_api_key
        if not self._api_key:
            raise SyncLabsConfigError(
                "SYNC_LABS_API_KEY is not set. Get one from "
                "https://sync.so/settings/api-keys and add it to .env as "
                "SYNC_LABS_API_KEY=..."
            )
        self._max_wait = max_wait
        self._poll_interval = poll_interval
        self._client = Sync(api_key=self._api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception(_is_retryable_api_error),
        reraise=True,
    )
    def _create_with_files(self, video_bytes: bytes, audio_bytes: bytes, options: LipsyncOptions):
        # Fresh BytesIO per call, not the caller's open file handle: if a
        # prior attempt read the handle to EOF before failing, a retry
        # against the same handle would upload an empty/truncated file
        # instead of actually retrying. Rebuilding from bytes we already
        # hold in memory (files are capped at 20MB, so this is cheap) makes
        # every attempt independent.
        #
        # Note: create_with_files (the multipart/local-file endpoint) has
        # no output_file_name parameter - that's only accepted by the
        # URL-based create() call below. Confirmed against the installed
        # syncsdk's actual method signature, not assumed from docs.
        #
        # Second real bug, caught during the live Day 2-3 test run (not
        # by static inspection): create_with_files builds a multipart
        # request (httpx `data=` + `files=`), and the installed SDK does
        # NOT JSON-encode nested fields for that path - passing a
        # GenerationOptions object reaches httpx's multipart encoder as a
        # raw dict and throws `TypeError: Invalid type for value.
        # Expected primitive type, got <class 'dict'>`. The URL-based
        # create() call below is unaffected - it sends a pure JSON body
        # (httpx `json=`), where nested objects serialize fine. This is
        # the same class of bug Day 1 hit with ElevenLabs' voice_settings
        # needing manual json.dumps() for its multipart field. Verified
        # against the live API: a manually JSON-encoded string in the
        # `options` field is accepted and round-trips correctly (the
        # created generation's `options.sync_mode` comes back as
        # "cut_off" as requested).
        return self._client.generations.create_with_files(
            video=io.BytesIO(video_bytes),
            audio=io.BytesIO(audio_bytes),
            model=options.model,
            options=json.dumps({"sync_mode": options.sync_mode}),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception(_is_retryable_api_error),
        reraise=True,
    )
    def _create_from_urls(self, video_url: str, audio_url: str, options: LipsyncOptions):
        return self._client.generations.create(
            input=[Video(url=video_url), Audio(url=audio_url)],
            model=options.model,
            options=GenerationOptions(sync_mode=options.sync_mode),
            output_file_name=options.output_file_name,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception(_is_retryable_api_error),
        reraise=True,
    )
    def _get_generation(self, generation_id: str):
        return self._client.generations.get(generation_id)

    def _poll_until_terminal(self, generation_id: str):
        elapsed = 0.0
        generation = self._get_generation(generation_id)
        while generation.status not in _TERMINAL_STATUSES:
            if elapsed >= self._max_wait:
                raise SyncLabsTimeoutError(generation_id, elapsed)
            logger.info(
                "Generation %s status=%s, waiting %ss (elapsed %.0fs/%ss)",
                generation_id, generation.status, self._poll_interval, elapsed, self._max_wait,
            )
            time.sleep(self._poll_interval)
            elapsed += self._poll_interval
            generation = self._get_generation(generation_id)
        return generation

    def _download_output(self, output_url: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(output_url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise SyncLabsError(f"Downloaded output is empty: {output_path}")

    def _finish(self, generation_id: str, output_path: Path) -> LipsyncResult:
        try:
            generation = self._poll_until_terminal(generation_id)
        except ApiError as exc:
            logger.error("Polling generation %s failed: %s", generation_id, exc)
            raise SyncLabsError(
                f"Sync Labs API rejected a status check for generation {generation_id}: {exc}"
            ) from exc

        if generation.status != "COMPLETED":
            raise SyncLabsGenerationFailedError(
                generation_id,
                generation.status,
                getattr(generation, "error", None),
                getattr(generation, "error_code", None),
            )

        output_url = getattr(generation, "output_url", None)
        if not output_url:
            raise SyncLabsError(
                f"Generation {generation_id} reported COMPLETED but returned no output_url"
            )

        self._download_output(output_url, output_path)

        return LipsyncResult(
            generation_id=generation_id,
            status=generation.status,
            output_path=output_path,
            output_duration_seconds=getattr(generation, "output_duration", None),
            model=getattr(generation, "model", "unknown"),
        )

    def generate_from_files(
        self,
        video_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
        options: LipsyncOptions | None = None,
    ) -> LipsyncResult:
        """
        Submit a local reference video + local converted audio for
        lipsync, poll until terminal, download the result.

        Raises SyncLabsInputError before any API call if inputs are
        missing/empty/oversized. Raises SyncLabsGenerationFailedError or
        SyncLabsTimeoutError if the job doesn't complete cleanly. A
        successful return means the API says COMPLETED and a file landed
        on disk - it does NOT mean the lipsync is good. Watch the output.
        """
        options = options or LipsyncOptions()
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        for label, path in (("video", video_path), ("audio", audio_path)):
            if not path.exists():
                raise SyncLabsInputError(f"Input {label} not found: {path}")
            size = path.stat().st_size
            if size == 0:
                raise SyncLabsInputError(f"Input {label} is empty: {path}")
            if size > MAX_DIRECT_UPLOAD_BYTES:
                raise SyncLabsInputError(
                    f"Input {label} is {size / 1024 / 1024:.1f}MB, over the "
                    f"{MAX_DIRECT_UPLOAD_BYTES / 1024 / 1024:.0f}MB direct-upload limit. "
                    f"Trim the clip or add the presigned asset-upload flow "
                    f"(see module docstring) - do not silently truncate it."
                )

        logger.info(
            "Submitting lipsync: video=%s audio=%s model=%s",
            video_path.name, audio_path.name, options.model,
        )

        video_bytes = video_path.read_bytes()
        audio_bytes = audio_path.read_bytes()

        try:
            response = self._create_with_files(video_bytes, audio_bytes, options)
        except ApiError as exc:
            logger.error("Sync Labs generation request failed: %s", exc)
            raise SyncLabsError(f"Sync Labs API rejected the request: {exc}") from exc

        logger.info("Generation submitted: id=%s status=%s", response.id, response.status)
        return self._finish(response.id, output_path)

    def generate_from_urls(
        self,
        video_url: str,
        audio_url: str,
        output_path: str | Path,
        options: LipsyncOptions | None = None,
    ) -> LipsyncResult:
        """
        Same as generate_from_files, for when the reference video/audio
        already live at a public URL (e.g. Supabase Storage in the Day 4
        orchestration pipeline) instead of on local disk.
        """
        options = options or LipsyncOptions()
        output_path = Path(output_path)

        logger.info("Submitting lipsync (by URL): model=%s", options.model)

        try:
            response = self._create_from_urls(video_url, audio_url, options)
        except ApiError as exc:
            logger.error("Sync Labs generation request failed: %s", exc)
            raise SyncLabsError(f"Sync Labs API rejected the request: {exc}") from exc

        logger.info("Generation submitted: id=%s status=%s", response.id, response.status)
        return self._finish(response.id, output_path)
