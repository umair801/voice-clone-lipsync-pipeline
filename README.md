# AI Avatar Video Pipeline

Voice-clone + lipsync content automation. Takes a creator's own recorded
read, corrects its delivery through a voice-clone conversion, and fuses
the corrected audio onto real reference video footage of the speaker —
producing a finished, lip-synced clip with no manual editing between
steps.

This is not a face-generation or deepfake-from-photo tool. It requires
real reference footage of the person speaking. That's a deliberate design
choice: it's what gives the output legitimacy over photo-based avatar
tools, at the cost of needing a short video recording up front rather than
a single still image.

## How it works

```
raw audio read  ──┐
                   ├─► voice conversion ──► trim lead-in ──► normalize video ──► lipsync ──► QA check ──► finished clip
reference video ──┘
```

1. **Voice conversion** — the creator's own raw recorded read is run
   through a speech-to-speech voice-clone conversion (same words, same
   delivery, corrected voice characteristics). This is speech-to-speech,
   not text-to-speech — it starts from the creator's real performance.
2. **Trim lead-in** — any quiet, non-speech audio at the very start of
   the converted track (room tone, breath, near-silence) is detected and
   cut before the audio reaches the lipsync stage. This closes a real
   failure mode found during testing: the lipsync provider rendered
   continuous mouth motion through a quiet lead-in as if it were speech,
   producing a visible mouth/audio mismatch in the first ~1-2 seconds of
   a clip. See Known Constraints below for what this does and doesn't
   guarantee.
3. **Video normalization** — the reference video is re-encoded so any
   rotation metadata is baked into the actual pixels before it reaches the
   lipsync stage. This closes a real, non-obvious failure mode found
   during testing: a video's rotation tag was sometimes misread by the
   lipsync provider, producing a correctly-processed but sideways output.
   Normalizing removes the ambiguity up front rather than hoping it's
   read correctly.
4. **Lipsync** — the corrected audio is fused onto the normalized
   reference video via a commercial lipsync API.
5. **QA check** — the finished output's duration is automatically
   compared against the reference video's duration, and a sample of
   frames is checked for a detectable face (catching a gross failure —
   the provider returning a blank, corrupted, or wildly wrong clip). A
   failure on either check flags the job for manual review instead of
   silently shipping a bad result. Neither check judges lip-sync accuracy
   or subtle visual quality — see Known Constraints below.
6. **Delivery** — the finished clip and a full job record (what
   succeeded, what failed, at which stage, and why) are written to disk
   and available via the API.

Every stage's failure is caught, logged, and attached to the job's status
— a failed voice conversion, a lipsync provider rejection, or a QA flag
all produce a specific, inspectable reason rather than a silent bad
output or a generic error.

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) and `ffprobe` on `PATH` (used for video
  normalization and QA duration checks)
- An API key for a speech-to-speech voice-clone provider, with a cloned
  or chosen target voice already set up on that account
- An API key for a commercial lipsync API

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in your API keys and target voice ID in .env
```

See `.env.example` for every configuration value, what it does, and its
default.

## Running it

### As a one-off script

```python
from agents.orchestrator import run_pipeline

final_state = run_pipeline(
    job_id="my-first-job",
    source_audio_path="my_read.mp3",
    reference_video_path="my_reference_clip.mp4",
    output_dir="job_outputs/my-first-job",
)
print(final_state["status"])       # "completed" | "needs_review" | "failed"
print(final_state.get("error"))    # populated only if status == "failed"
```

### As an API server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Submit a job:

```bash
curl -X POST http://localhost:8000/jobs \
  -F "source_audio=@my_read.mp3;type=audio/mpeg" \
  -F "reference_video=@my_reference_clip.mp4;type=video/mp4"
# -> {"job_id": "...", "status": "queued"}
```

Check its status:

```bash
curl http://localhost:8000/jobs/<job_id>
```

A job's `status` moves through `queued → converting_voice_done →
normalizing_video_done → lipsyncing_done → qa_passed → completed`, or
lands on `needs_review` (finished, but the QA check wants a human look) or
`failed` (with `error` and `error_stage` telling you exactly what broke
and where).

### Automatically, via a watched folder

Drop a matching pair of files into the folder set by `INCOMING_DIR` (e.g.
`incoming/myjob_audio.mp3` and `incoming/myjob_video.mp4` — the shared
name before `_audio`/`_video` is what pairs them). A background poller
picks up the pair, submits it as a job automatically, and moves the
source files into `incoming/processed/` once picked up. No manual
submission step needed for a recurring/scheduled content pipeline.

## Known constraints (read before promising a client anything)

- **Lipsync quality depends heavily on the reference footage.**
  Front-facing, well-lit, minimal head movement performs best. This
  pipeline was explicitly tested against a deliberately compromised
  clip (off-angle, busier background) and did *not* clearly show worse
  results than the clean baseline in that one test — that is one data
  point, not a guarantee. Don't promise universal reliability from a
  single test; budget for review and, if needed, a reshoot on
  difficult footage.
- **A "COMPLETED" status from the lipsync provider is not the same as a
  good result.** The provider can return success while the output has a
  subtle sync or blending issue. This was found for real during testing:
  the budget lipsync model (`lipsync-2`) produced a small, intermittent
  mouth/jaw-boundary artifact on one clip, confirmed by comparing the
  output against the source footage at the same timestamp — not visible
  in a single still frame, only on close inspection. Re-running the same
  clip on the higher-quality model (`lipsync-2-pro`) resolved it, and
  that was confirmed on a second, different clip. Use `lipsync-2` for
  iteration and `lipsync-2-pro` for anything a client will watch closely.
  The automated QA checks (duration match, face-presence sampling) do not
  catch this class of artifact — they catch gross failures, not subtle
  blending issues. A human should review the output before it ships to a
  client or goes live, especially for the first several runs against new
  footage.
- **A quiet lead-in was found causing visible mouth motion before any real
  speech.** Found during Day 6 QA on real test clips (three different
  clips, all confirmed affected): the lipsync provider does not appear to
  distinguish true digital silence from quiet-but-nonzero audio (room
  tone, breath) at the start of a track, and renders continuous mouth
  motion through it as if it were speech. Fixed with an automated
  pre-processing step (`agents/tools/audio_trim.py`) that detects and
  trims a leading quiet block from the converted audio before it reaches
  the lipsync API, run on every job. Verified two ways: on the input side,
  the trimmed audio opens directly on the real speech onset with no
  clipping; and against a real re-generation through the actual lipsync
  API with the trimmed audio, where the output's own mouth motion now
  tracks the ~100ms-scale timing of the real audio instead of the
  original ~1.9s mismatch. A full-attention watch with sound on that
  re-generated clip is still the final human confirmation step, same as
  every other clip this pipeline produces. One documented limitation:
  the detector only trims a
  single contiguous quiet block — a brief loud blip (a click, an early
  breath spike) inside what should be the lead-in would stop detection
  early and leave some residual quiet audio untrimmed. Not observed in
  this project's actual audio so far, but worth knowing before assuming
  this is bulletproof against every possible source read. As with the
  jaw/mouth-boundary artifact above, review the first several seconds of
  new footage before it ships.
- **This pipeline does not have audio-listening capability built in.**
  The automated checks are visual/structural (duration, orientation,
  frame-level sanity). Final confirmation that the lip movement actually
  matches the words requires a person watching the clip with sound.
- **Multiple takes are normal.** First-pass perfection isn't realistic;
  budget 2-3 iteration passes per clip as a fair standard.
- **Job storage is in-memory only in this version** — job status/results
  are lost on server restart. This is a documented, intentional scope
  choice for the current build, not an oversight; swapping in a
  persistent store (e.g. Supabase/Postgres) is a contained change behind
  the `JobStore` interface in `core/job_store.py`.
- **API authentication is optional, off by default.** Set
  `PIPELINE_API_KEY` in `.env` and every `POST /jobs` request must
  include a matching `X-API-Key` header (checked with a constant-time
  comparison), or it's rejected with 401 before a job is created or any
  external API is called. Unset by default so local dev keeps working
  without extra setup, set it before exposing this server beyond
  localhost. It's a single shared secret, not per-user auth; adequate for
  a single-operator tool, not a multi-tenant deployment.
- **Uploads are capped at `MAX_UPLOAD_BYTES`** (100MB by default),
  enforced against actual bytes streamed to disk, not a client-supplied
  header. An oversized upload is rejected with 413, nothing is written to
  disk beyond the cap, and a rejected upload cleans up after itself, no
  orphaned job record or partial input directory left behind. The upload
  streaming itself runs off the event loop (via `run_in_threadpool`), so
  one large upload in progress doesn't stall other in-flight requests,
  including `/health`.

## Project layout

```
agents/
  orchestrator.py        # LangGraph pipeline: state graph + node functions
  tools/
    elevenlabs_voice_changer.py   # speech-to-speech voice conversion client
    sync_labs_lipsync.py          # lipsync API client
    video_normalize.py            # rotation-baking video normalization
    qa_checks.py                  # automated duration + face-presence QA
api/
  routes.py               # POST /jobs, GET /jobs/{id}, GET /health
  scheduler.py             # watched-folder auto-submit trigger
  schemas.py                # request/response models
core/
  config.py                 # centralized settings (.env-backed)
  logging.py                 # structured logging setup
  job_store.py                 # in-memory job status/result store
main.py                        # FastAPI app entrypoint
tests/
  test_orchestrator_mocked.py     # pipeline tests (real ffmpeg/ffprobe,
                                    # mocked external voice/lipsync APIs)
  test_api_hardening.py            # API auth, upload cap, cleanup-on-reject,
                                    # job-store tests
```

## Testing

```bash
python -m pytest tests/ -v
```

The orchestrator tests exercise the real video-normalization and QA logic
against real files; only the external voice-clone and lipsync API calls
are mocked, so a passing test suite confirms the pipeline's own logic
without spending API quota on every run.
