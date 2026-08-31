"""
ElevenLabs Voice Changer (speech-to-speech) client.

This wraps ElevenLabs' speech-to-speech endpoint, NOT text-to-speech.
Input is a recorded read (the creator's own voice); output is that same
performance re-rendered through a target voice, with accent/delivery
corrected according to the stability/clarity/style parameters below.

Docs: https://elevenlabs.io/docs/api-reference/speech-to-speech
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from elevenlabs.client import ElevenLabs
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class VoiceChangerError(Exception):
    """Raised when the ElevenLabs Voice Changer call fails after retries."""


@dataclass(frozen=True)
class VoiceChangerParams:
    """
    Tunable parameters for a single conversion pass.

    stability: 0.0-1.0. Lower = more expressive/variable delivery (closer
        to the original read's emotional inflection). Higher = more
        consistent/flat, but more "corrected."
    similarity_boost: 0.0-1.0 (ElevenLabs calls this "clarity + similarity
        enhancement" in the UI). Higher = output hews closer to the
        target voice's exact timbre; can introduce artifacts if pushed
        too high on a source read with background noise or heavy accent.
    style: 0.0-1.0. Exaggerates the target voice's native style. Usually
        keep low (0.0-0.2) for corrective use cases; high style values
        fight against preserving the original read's pacing.
    remove_background_noise: strips noise from the source before
        conversion. Turn on if the read was recorded outside a treated
        room.
    """

    # Defaults chosen from the Day 1 parameter sweep (2026-08-25): stab0.70
    # /sim0.85 gave the most consistent, least-drifting output across a
    # sample read, judged by ear against the other 3 sweep variants. Prefer
    # consistency here since this pipeline generates many clips over time,
    # not a single one-off. If a future script needs more expressiveness
    # (excited delivery, sarcasm, emphasis) and this default sounds flat or
    # robotic on it, drop stability back toward ~0.5 for that job rather
    # than treating this as broken - it's a per-job tunable, not a fixed
    # constant.
    stability: float = 0.70
    similarity_boost: float = 0.85
    style: float = 0.0
    remove_background_noise: bool = False

    def label(self) -> str:
        return f"stab{self.stability:.2f}_sim{self.similarity_boost:.2f}_style{self.style:.2f}"


class ElevenLabsVoiceChanger:
    """Thin, retry-hardened wrapper around ElevenLabs speech-to-speech."""

    DEFAULT_TIMEOUT_SECONDS = 60

    def __init__(
        self,
        api_key: str | None = None,
        target_voice_id: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = api_key or settings.elevenlabs_api_key
        self._target_voice_id = target_voice_id or settings.elevenlabs_target_voice_id
        self._client = ElevenLabs(api_key=self._api_key, timeout=timeout)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_api(self, audio_bytes: bytes, params: VoiceChangerParams, model_id: str):
        # The installed SDK's speech_to_speech.convert expects voice_settings
        # as a JSON-encoded string (it goes into a multipart form field), not
        # a raw dict - passing a dict raises TypeError deep in httpx's
        # multipart encoder.
        voice_settings_json = json.dumps(
            {
                "stability": params.stability,
                "similarity_boost": params.similarity_boost,
                "style": params.style,
                "use_speaker_boost": True,
            }
        )
        return self._client.speech_to_speech.convert(
            voice_id=self._target_voice_id,
            audio=audio_bytes,
            model_id=model_id,
            voice_settings=voice_settings_json,
            remove_background_noise=params.remove_background_noise,
        )

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        params: VoiceChangerParams | None = None,
        model_id: str = "eleven_multilingual_sts_v2",
    ) -> Path:
        """
        Run one source audio file through Voice Changer and write the
        converted result to output_path.

        Raises VoiceChangerError if the input file is missing/empty or
        the API call fails after retries.
        """
        params = params or VoiceChangerParams()
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise VoiceChangerError(f"Input audio not found: {input_path}")
        audio_bytes = input_path.read_bytes()
        if not audio_bytes:
            raise VoiceChangerError(f"Input audio is empty: {input_path}")

        logger.info(
            "Converting %s -> %s [%s]", input_path.name, output_path.name, params.label()
        )

        try:
            audio_stream = self._call_api(audio_bytes, params, model_id)
        except Exception as exc:  # noqa: BLE001 - surfaced as VoiceChangerError
            logger.error("Voice Changer API call failed for %s: %s", input_path.name, exc)
            raise VoiceChangerError(f"ElevenLabs conversion failed: {exc}") from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                if chunk:
                    f.write(chunk)

        if output_path.stat().st_size == 0:
            raise VoiceChangerError(f"Conversion produced an empty file: {output_path}")

        logger.info("Wrote converted audio: %s (%d bytes)", output_path, output_path.stat().st_size)
        return output_path