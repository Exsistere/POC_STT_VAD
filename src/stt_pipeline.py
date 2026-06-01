import asyncio
import logging
import sys
import time
from datetime import datetime
from typing import AsyncGenerator

from dotenv import load_dotenv
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    JobProcess,
    inference,
    room_io,
    TurnHandlingOptions,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    WorkerOptions,
)
from livekit.plugins import silero, ai_coustics

load_dotenv(override=True)

# ─── Logging ──────────────────────────────────────────────────────────────────


logger = logging.getLogger("speech_renderer")
logger.setLevel(logging.INFO)
logger.propagate = False

handler = logging.StreamHandler(sys.stdout)
handler.flush = sys.stdout.flush
formatter = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.getLogger("livekit.agents").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# After VAD silence, wait this long for the final transcript before flushing anyway
FINAL_WAIT = 1.5


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log_transcript(message: str) -> None:
    """
    Write transcript message atomically to stdout with logger prefix.
    Ensures entire message is written as one atomic operation with proper formatting.
    Uses ANSI color codes to match LiveKit's log style.
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    BLUE  = "\033[34m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    lines = message.split("\n")
    for line in lines:
        formatted = f"{BLUE}{timestamp}{RESET} {GREEN}INFO{RESET} speech_renderer {line}"
        sys.stdout.write(f"{formatted}\n")
    sys.stdout.flush()


# ─── Silent Agent ─────────────────────────────────────────────────────────────

class SilentAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="")


# ─── Prewarm ──────────────────────────────────────────────────────────────────
# Called by main.py's prewarm 

def prewarm(proc: JobProcess) -> None:
    try:
        proc.userdata["vad"] = silero.VAD.load(
            activation_threshold=0.65,
        )
    except Exception as e:
        logger.error(f"Error loading VAD: {e}")


# ─── Module-level state set by init_pipeline() ────────────────────────────────

_utterance_queue: asyncio.Queue[str] | None = None
_interrupt_fn = None   # callable injected by main.py → tts_pipeline.interrupt()


def init_pipeline(ctx: JobContext, interrupt_fn=None) -> None:
    """
    Initialise the STT pipeline module state.
    Must be called once from main.py before stt_stream() is consumed.

    Args:
        ctx:          The shared JobContext (room already connected).
        interrupt_fn: Optional callable — called when VAD detects speech
                      while TTS is playing (barge-in / interruption).
                      Pass tts_pipeline.get_pipeline().interrupt here.
    """
    global _utterance_queue, _interrupt_fn
    _utterance_queue = asyncio.Queue()
    _interrupt_fn    = interrupt_fn


# ─── Public callable ──────────────────────────────────────────────────────────

async def stt_stream(ctx: JobContext) -> AsyncGenerator[str, None]:
    """
    Public callable — runs the STT pipeline and yields finalised utterances.

    Takes mic audio (via the LiveKit room the ctx is connected to) as input.
    Yields each complete utterance as a plain string once VAD + Deepgram
    have confirmed end-of-utterance.

    Usage (from main.py):
        async for text in stt_pipeline.stt_stream(ctx):
            print(text)   # or route to LLM, TTS, etc.
    """
    assert _utterance_queue is not None, "call init_pipeline(ctx) first"

    # Start the internal session as a background task so this generator
    # can yield results while the session runs concurrently.
    session_task = asyncio.ensure_future(_run_session(ctx))

    try:
        while True:
            utterance = await _utterance_queue.get()
            if utterance is None:          # sentinel — session ended
                break
            yield utterance
    finally:
        session_task.cancel()
        try:
            await session_task
        except asyncio.CancelledError:
            pass


# ─── Internal session runner ──────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════
# ORIGINAL PIPELINE CODE BELOW — unchanged except:
#   1. ctx.connect() moved to BEFORE session.start() (required for real room)
#   2. on_utterance() pushes to _utterance_queue instead of being a no-op stub
#   3. interrupt_fn called on VAD speaking event if TTS is active
#   4. Wrapped in _run_session() coroutine so it can be task-cancelled cleanly
# ════════════════════════════════════════════════════════════════════════════════

async def _run_session(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
        ),
    )

    _speech_active              = False
    _speech_index               = 0
    _last_interim               = ""
    _utterance_parts: list[str] = []
    _speech_start_time: float   = 0.0
    _first_final_time: float    = 0.0
    _vad_ended: bool            = False
    _flushed: bool              = False
    _flush_task: asyncio.Task | None = None

    def _do_flush():
        nonlocal _utterance_parts, _last_interim, _flushed
        if _flushed:
            return
        if not _utterance_parts and not _last_interim:
            return
        _flushed = True

        full = " ".join(_utterance_parts).strip() or _last_interim.strip()
        if full:
            e2e_ms = (_first_final_time - _speech_start_time) * 1000
            block = (
                f"│  [{ts()}] EOU        VAD\n"
                f"│  [{ts()}] UTTERANCE  {full}\n"
                f"│  [{ts()}] E2E        {e2e_ms:.0f}ms (speech start → first final)\n"
                f"└─── SPEECH #{_speech_index} END   [{ts()}] {'─' * 30}"
            )
            log_transcript(block)

            # ── Push finalised utterance to the async generator queue ─────
            if _utterance_queue is not None:
                _utterance_queue.put_nowait(full)

        _utterance_parts.clear()
        _last_interim = ""

    async def _wait_for_final_then_flush():
        await asyncio.sleep(FINAL_WAIT)
        _do_flush()

    @session.on("user_state_changed")
    def on_user_state(event: UserStateChangedEvent):
        nonlocal _speech_active, _speech_index, _last_interim, _utterance_parts
        nonlocal _speech_start_time, _flushed, _vad_ended, _flush_task
        nonlocal _first_final_time

        if event.new_state == "speaking":
            # ── Barge-in: interrupt TTS if it is currently speaking ───────
            if _interrupt_fn is not None:
                try:
                    _interrupt_fn()
                except Exception:
                    pass

            if _flush_task and not _flush_task.done():
                _flush_task.cancel()
            _do_flush()

            _speech_active     = True
            _speech_index     += 1
            _speech_start_time = time.monotonic()
            _first_final_time  = 0.0
            _flushed           = False
            _vad_ended         = False
            _last_interim      = ""
            _utterance_parts   = []
            log_transcript(f"┌─── SPEECH #{_speech_index} START [{ts()}] {'─' * 30}")

        elif event.new_state != "speaking" and _speech_active:
            _speech_active = False
            _vad_ended     = True
            _flush_task    = asyncio.ensure_future(_wait_for_final_then_flush())

    @session.on("user_input_transcribed")
    def on_transcript(event: UserInputTranscribedEvent):
        nonlocal _last_interim, _utterance_parts, _first_final_time
        nonlocal _flush_task

        lang = getattr(event, "language", "?")
        if not event.is_final:
            _last_interim = event.transcript
            log_transcript(f"│  [{ts()}] INTERIM    [{lang}]  {event.transcript}")
            return

        if not _utterance_parts:
            _first_final_time = time.monotonic()
        _utterance_parts.append(event.transcript.strip())

        if _vad_ended:
            if _flush_task and not _flush_task.done():
                _flush_task.cancel()
            _do_flush()

    # ── session.start() — unchanged args from original ────────────────────
    # ctx.connect() is called in main.py BEFORE this function runs,
    # so the room is already live when we start the session.
    await session.start(
        agent=SilentAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                )
            ),
            audio_output=False,
            text_output=False,
        ),
    )

    # Keep the session alive until cancelled
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        if _utterance_queue is not None:
            _utterance_queue.put_nowait(None)   # signal generator to stop


# ─── Standalone entry point (unchanged from original) ─────────────────────────
# Kept so you can still run this file directly for STT-only testing:
#   uv run python stt_pipeline.py connect --room my-room

async def my_agent(ctx: JobContext):
    init_pipeline(ctx)
    await ctx.connect()
    async for utterance in stt_stream(ctx):
        log_transcript(f"[STANDALONE] utterance → {utterance!r}")


if __name__ == "__main__":
    from livekit.agents import cli
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=my_agent,
            prewarm_fnc=prewarm,
        )
    )
