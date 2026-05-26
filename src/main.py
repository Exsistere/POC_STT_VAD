import asyncio
import sys
import logging
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, WorkerOptions

import tts_pipeline_v1 as tts_pipeline
import stt_pipeline

load_dotenv(override=True)

# ─── Logging ──────────────────────────────────────────────────────────────────

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logger.propagate = False

handler = logging.StreamHandler(sys.stdout)
handler.flush = sys.stdout.flush
formatter = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.getLogger("livekit.agents").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# ─── ANSI colours ─────────────────────────────────────────────────────────────

YELLOW = "\033[33m"
GREEN  = "\033[32m"
RESET  = "\033[0m"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    for line in message.split("\n"):
        sys.stdout.write(f"{YELLOW}{timestamp}{RESET} {GREEN}INFO{RESET} main          {line}\n")
    sys.stdout.flush()


# ─── Prewarm ──────────────────────────────────────────────────────────────────

def prewarm(proc: JobProcess) -> None:
    """
    Prewarm is shared — STT pipeline needs the VAD loaded before the job starts.
    Stored in proc.userdata so both pipelines can access it via ctx.
    """
    stt_pipeline.prewarm(proc)


# ─── Terminal stdin → TTS loop ────────────────────────────────────────────────

async def _stdin_tts_loop() -> None:
    """
    Reads lines from terminal stdin and sends them to the TTS pipeline.

    Type text and hit Enter — the TTS pipeline will speak it.
    Type 'quit' or press Ctrl-C to stop.
    """
    loop = asyncio.get_event_loop()
    _log(
        f"[{_ts()}] STDIN LOOP  ready\n"
        f"[{_ts()}] STDIN LOOP  type text and press Enter to speak via TTS\n"
        f"[{_ts()}] STDIN LOOP  type 'quit' to exit"
    )

    while True:
        try:
            # run_in_executor keeps the event loop unblocked while waiting
            text = await loop.run_in_executor(None, sys.stdin.readline)
        except (EOFError, KeyboardInterrupt):
            break

        text = text.strip()
        if not text:
            continue
        if text.lower() == "quit":
            _log(f"[{_ts()}] STDIN LOOP  quit received — shutting down")
            break

        _log(f"[{_ts()}] STDIN LOOP  sending to TTS → {text!r}")
        # Fire-and-forget so stdin stays responsive during playback
        asyncio.ensure_future(tts_pipeline.speak(text))


# ─── STT result handler ───────────────────────────────────────────────────────

async def _stt_consumer(ctx: JobContext) -> None:
    """
    Consumes the async generator from stt_pipeline.stt_stream().

    Each yielded value is a finalised utterance string.
    Wire your own logic here (LLM call, routing, logging, etc.).
    Currently just logs — no automatic STT→TTS.
    """
    async for utterance in stt_pipeline.stt_stream(ctx):
        # ── Plug your downstream logic here ──────────────────────────────
        # Examples:
        #   response = await llm.complete(utterance)
        #   await tts_pipeline.speak(response)
        _log(f"[{_ts()}] STT RESULT  utterance received → {utterance!r}")


# ─── Agent entrypoint ─────────────────────────────────────────────────────────

async def my_agent(ctx: JobContext) -> None:
    """
    Single entrypoint — both pipelines share this JobContext and its room.

    Flow:
        1. Connect to the LiveKit room
        2. Initialise and start TTS pipeline (publishes audio track)
        3. Initialise STT pipeline (subscribes to participant mic track)
        4. Run concurrently:
               - STT consumer (mic → VAD → Deepgram → yields text)
               - stdin TTS loop (terminal input → speak())
    """
    _log(
        f"[{_ts()}] ══════════════════════════════════════════\n"
        f"[{_ts()}] AGENT START  room='{ctx.room.name}'\n"
        f"[{_ts()}] ══════════════════════════════════════════"
    )

    # ── 1. Connect to room first ───────────────────────────────────────────
    await ctx.connect()
    _log(f"[{_ts()}] ROOM        connected → '{ctx.room.name}'")

    # ── 2. TTS pipeline: init → publish track → start local speaker ───────
    tts_pl = tts_pipeline.init_pipeline(ctx)
    await tts_pipeline.start_pipeline()
    _log(f"[{_ts()}] TTS         pipeline ready")

    # ── 3. STT pipeline: init (AgentSession, VAD, Deepgram) ───────────────
    stt_pipeline.init_pipeline(ctx, interrupt_fn=tts_pl.interrupt)
    _log(f"[{_ts()}] STT         pipeline ready")

    _log(
        f"[{_ts()}] BOTH PIPELINES RUNNING\n"
        f"[{_ts()}] ── Speak into your mic  → STT transcript appears above\n"
        f"[{_ts()}] ── Type text + Enter    → TTS speaks it into the room"
    )

    # ── 4. Run both concurrently ───────────────────────────────────────────
    await asyncio.gather(
        _stt_consumer(ctx),
        _stdin_tts_loop(),
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from livekit.agents import cli
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=my_agent,
            prewarm_fnc=prewarm,
        )
    )
