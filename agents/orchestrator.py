"""
LangGraph orchestration for the AI Avatar Video Pipeline.

Pipeline:
  voice_conversion -> trim_lead_silence -> normalize_video -> lipsync -> qa -> deliver

trim_lead_silence was added in the Day 6 addendum (see project handoff
doc) after real testing showed Sync Labs rendering mouth motion during
the quiet lead-in of converted audio, before any real speech. It runs
on every job, not conditionally - same reasoning as normalize_video's
own docstring on why its fix isn't optional either.

Each stage's own client (ElevenLabsVoiceChanger, SyncLabsLipsyncClient)
already retries transient/5xx errors internally via tenacity. This graph
does NOT add another layer of automatic retry on top of an
application-level failure (e.g. a Sync Labs REJECTED/FAILED status) -
resubmitting an identical job to Sync Labs on a semantic rejection is
unlikely to succeed differently and burns real, scarce free-tier quota (3
generations/month). Instead, stage failures route straight to a terminal
"failed" state with the error attached, for a human to look at and decide
whether/how to retry - matching the human-in-the-loop pattern the build
spec calls for at the QA gate anyway.

Each node (after the first) checks state["status"] == "failed" at the top
and short-circuits without doing its own work if an earlier stage already
failed - this is what makes the plain linear edge chain below safe,
without needing conditional-edge branching for the failure path.
"""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents.tools.audio_trim import AudioTrimError, trim_leading_silence
from agents.tools.elevenlabs_voice_changer import (
    ElevenLabsVoiceChanger,
    VoiceChangerError,
    VoiceChangerParams,
)
from agents.tools.qa_checks import QACheckError, run_qa
from agents.tools.sync_labs_lipsync import LipsyncOptions, SyncLabsError, SyncLabsLipsyncClient
from agents.tools.video_normalize import VideoNormalizeError, normalize_video
from core.logging import get_logger

logger = get_logger(__name__)


class PipelineState(TypedDict):
    job_id: str
    source_audio_path: str
    reference_video_path: str
    output_dir: str

    voice_params: dict | None
    lipsync_model: str

    converted_audio_path: str | None
    trimmed_audio_path: str | None
    normalized_video_path: str | None
    lipsync_output_path: str | None
    qa: dict | None

    # queued -> converting_voice_done -> trimming_audio_done ->
    # normalizing_video_done -> lipsyncing_done -> qa_passed|needs_review
    # -> completed|needs_review
    # or, at any stage: failed
    status: str
    error: str | None
    error_stage: str | None


def voice_conversion_node(state: PipelineState) -> PipelineState:
    out_dir = Path(state["output_dir"])
    converted_path = out_dir / f"{state['job_id']}_converted_audio.mp3"
    params = VoiceChangerParams(**state["voice_params"]) if state.get("voice_params") else None
    try:
        changer = ElevenLabsVoiceChanger()
        changer.convert(state["source_audio_path"], converted_path, params=params)
    except VoiceChangerError as exc:
        logger.error("[%s] voice_conversion failed: %s", state["job_id"], exc)
        return {**state, "status": "failed", "error": str(exc), "error_stage": "voice_conversion"}
    return {**state, "converted_audio_path": str(converted_path), "status": "converting_voice_done"}


def trim_lead_silence_node(state: PipelineState) -> PipelineState:
    if state["status"] == "failed":
        return state
    out_dir = Path(state["output_dir"])
    trimmed_path = out_dir / f"{state['job_id']}_trimmed_audio.mp3"
    try:
        trim_leading_silence(state["converted_audio_path"], trimmed_path)
    except AudioTrimError as exc:
        logger.error("[%s] trim_lead_silence failed: %s", state["job_id"], exc)
        return {**state, "status": "failed", "error": str(exc), "error_stage": "trim_lead_silence"}
    return {**state, "trimmed_audio_path": str(trimmed_path), "status": "trimming_audio_done"}


def normalize_video_node(state: PipelineState) -> PipelineState:
    if state["status"] == "failed":
        return state
    out_dir = Path(state["output_dir"])
    normalized_path = out_dir / f"{state['job_id']}_normalized_video.mp4"
    try:
        normalize_video(state["reference_video_path"], normalized_path)
    except VideoNormalizeError as exc:
        logger.error("[%s] normalize_video failed: %s", state["job_id"], exc)
        return {**state, "status": "failed", "error": str(exc), "error_stage": "normalize_video"}
    return {**state, "normalized_video_path": str(normalized_path), "status": "normalizing_video_done"}


def lipsync_node(state: PipelineState) -> PipelineState:
    if state["status"] == "failed":
        return state
    out_dir = Path(state["output_dir"])
    output_path = out_dir / f"{state['job_id']}_lipsync.mp4"
    try:
        client = SyncLabsLipsyncClient()
        result = client.generate_from_files(
            state["normalized_video_path"],
            state["trimmed_audio_path"],
            output_path,
            options=LipsyncOptions(model=state.get("lipsync_model", "lipsync-2")),
        )
    except SyncLabsError as exc:
        logger.error("[%s] lipsync failed: %s", state["job_id"], exc)
        return {**state, "status": "failed", "error": str(exc), "error_stage": "lipsync"}
    return {**state, "lipsync_output_path": str(result.output_path), "status": "lipsyncing_done"}


def qa_node(state: PipelineState) -> PipelineState:
    if state["status"] == "failed":
        return state
    try:
        qa_result = run_qa(state["normalized_video_path"], state["lipsync_output_path"])
    except QACheckError as exc:
        logger.error("[%s] QA check itself failed to run: %s", state["job_id"], exc)
        return {**state, "status": "failed", "error": str(exc), "error_stage": "qa"}

    qa_dict = {
        "passed": qa_result.passed,
        "needs_review": qa_result.needs_review,
        "duration_check": qa_result.duration_check,
        "face_detection_confidence": qa_result.face_detection_confidence,
        "artifact_flagging": qa_result.artifact_flagging,
        "notes": qa_result.notes,
    }
    new_status = "needs_review" if qa_result.needs_review else "qa_passed"
    return {**state, "qa": qa_dict, "status": new_status}


def deliver_node(state: PipelineState) -> PipelineState:
    if state["status"] == "failed":
        return state
    final_status = "needs_review" if state["status"] == "needs_review" else "completed"
    logger.info(
        "[%s] Pipeline finished: status=%s output=%s",
        state["job_id"], final_status, state.get("lipsync_output_path"),
    )
    return {**state, "status": final_status}


def build_graph():
    builder = StateGraph(PipelineState)
    builder.add_node("voice_conversion", voice_conversion_node)
    builder.add_node("trim_lead_silence", trim_lead_silence_node)
    builder.add_node("normalize_video", normalize_video_node)
    builder.add_node("lipsync", lipsync_node)
    builder.add_node("qa", qa_node)
    builder.add_node("deliver", deliver_node)

    builder.set_entry_point("voice_conversion")
    builder.add_edge("voice_conversion", "trim_lead_silence")
    builder.add_edge("trim_lead_silence", "normalize_video")
    builder.add_edge("normalize_video", "lipsync")
    builder.add_edge("lipsync", "qa")
    builder.add_edge("qa", "deliver")
    builder.add_edge("deliver", END)

    return builder.compile()


def run_pipeline(
    job_id: str,
    source_audio_path: str,
    reference_video_path: str,
    output_dir: str,
    voice_params: dict | None = None,
    lipsync_model: str = "lipsync-2",
) -> PipelineState:
    """
    Run one job through the full pipeline synchronously (blocking) and
    return the final state dict. Callers that need this to not block
    (e.g. a FastAPI request handler) should invoke this from a background
    task/thread, not directly in an async route.
    """
    graph = build_graph()
    initial_state: PipelineState = {
        "job_id": job_id,
        "source_audio_path": source_audio_path,
        "reference_video_path": reference_video_path,
        "output_dir": output_dir,
        "voice_params": voice_params,
        "lipsync_model": lipsync_model,
        "converted_audio_path": None,
        "trimmed_audio_path": None,
        "normalized_video_path": None,
        "lipsync_output_path": None,
        "qa": None,
        "status": "queued",
        "error": None,
        "error_stage": None,
    }
    return graph.invoke(initial_state)
