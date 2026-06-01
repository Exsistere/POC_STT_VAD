import asyncio
import sys
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, WorkerOptions

import tts_pipeline_v1 as tts_pipeline
import stt_pipeline
from livekit.agents import inference
from livekit.agents import llm as agents_llm
from livekit.plugins import groq, azure, openai
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


# llm = groq.LLM(
#     model = "openai/gpt-oss-120b",
#     api_key = os.getenv("GROQ_API_KEY"),
#     max_completion_tokens=250,
#     temperature=0.3
# )

llm = openai.LLM.with_azure(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    model="openai/gpt-oss-120b",
    temperature=0.3,
    max_completion_tokens=250,
)

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

_current_llm_task: asyncio.Task | None = None

async def _stt_consumer(ctx: JobContext) -> None:
    global _current_llm_task
    chat_ctx = agents_llm.ChatContext()
    chat_ctx.add_message(role="system", content="You are a helpful assistant. Keep sentences short and conversational. "
            "Use natural pauses with commas and short sentences.")
    
    async for utterance in stt_pipeline.stt_stream(ctx):
        _log(f"[{_ts()}] STT RESULT  utterance received → {utterance!r}")
        
        #cancel previous LLM task if still running
        # Scenario: user says something, LLM still generating previous response, while tts stopped speaking,leading to is_speaking = false and interrupt() not called
        ## TODO seems redundant but just to be safe, we can cancel the previous LLM task here to avoid the half-spoken response when barge-in happens
        if _current_llm_task and not _current_llm_task.done():
            _current_llm_task.cancel()
            _log(f"[{_ts()}] LLM         previous task cancelled")
        #Reset tts interupt flag so new response can be spoken
        tts_pipeline.get_pipeline().reset_interrupt() ##TODO: to create the method
        chat_ctx.add_message(role="user", content=utterance)
        _current_llm_task = asyncio.ensure_future(_generate_and_speak(chat_ctx))

async def _generate_and_speak(chat_ctx) -> None:
    """
    Generates a response from the LLM and sends it
    to the TTS pipeline to speak in a single task.
    Aggressive streaming: flushes at clause boundaries with minimal
    buffering to exploit low TTFT (~200ms) and high TPS (~40).
    """
    # ── Tuning knobs ──────────────────────────────────────────────────────
    FIRST_CHUNK_MIN  = 20    # slightly longer first chunk for natural prosody
    CLAUSE_MIN_CHARS =  8    # subsequent chunks can be short
    HARD_CAP_CHARS   = 120   # never hold more than this — flush at word boundary
    SENTENCE_MARKERS = (". ", "? ", "! ", ".\n", "?\n", "!\n")
    CLAUSE_MARKERS   = (", ", "; ", ": ", " - ", "\n")
    # ─────────────────────────────────────────────────────────────────────

    sentence_buffer  = ""
    full_response    = ""
    first_chunk_sent = False

    def _should_flush() -> bool:
        buf      = sentence_buffer
        size     = len(buf)
        min_chars = FIRST_CHUNK_MIN if not first_chunk_sent else CLAUSE_MIN_CHARS

        if size >= HARD_CAP_CHARS:
            # only flush at a word boundary, never mid-word
            return buf[-1] == " " or buf.endswith(CLAUSE_MARKERS) or buf.endswith(SENTENCE_MARKERS)
        if buf.endswith(SENTENCE_MARKERS):
            return size >= min_chars
        if buf.endswith(CLAUSE_MARKERS):
            return size >= min_chars
        return False

    def _flush() -> None:
        nonlocal sentence_buffer, first_chunk_sent
        # strip leading punctuation from orphan chunks e.g. ", feel free..."
        text = sentence_buffer.strip().lstrip(", ;:-")
        if not text:
            sentence_buffer = ""
            return
        pl = tts_pipeline.get_pipeline()
        if not pl._interrupted:
            _log(f"[{_ts()}] LLM STREAM  → {text!r}")
            asyncio.ensure_future(tts_pipeline.speak(text))
        sentence_buffer  = ""
        first_chunk_sent = True

    try:
        stream = llm.chat(chat_ctx=chat_ctx)
        async with stream:
            async for chunk in stream:
                token = (chunk.delta and chunk.delta.content) or ""
                sentence_buffer += token
                full_response   += token

                if _should_flush():
                    _flush()

        # flush any remaining text
        if sentence_buffer.strip():
            pl = tts_pipeline.get_pipeline()
            if not pl._interrupted:
                text = sentence_buffer.strip().lstrip(", ;:-")
                if text:
                    _log(f"[{_ts()}] LLM STREAM  → {text!r}")
                    asyncio.ensure_future(tts_pipeline.speak(text))

        _log(f"[{_ts()}] LLM REPLY   → {full_response[:80]!r}")
        chat_ctx.add_message(role="assistant", content=full_response)

    except asyncio.CancelledError:
        if full_response.strip():
            chat_ctx.add_message(role="assistant", content=full_response)
        _log(f"[{_ts()}] LLM STREAM  cancelled mid-generation")
        raise
# async def _generate_and_speak(chat_ctx) -> None:
#     """
#     Generates a response from the LLM and sends it 
#     to the TTS pipeline to speak in a single task.
#     This allows us to cancel the entire generation+speaking 
#     process if a new utterance comes in, 
#     rather than just cancelling the generation and leaving a response half-spoken.
#     """
#     sentence_buffer = ""
#     full_response = ""

#     try:
#         stream = llm.chat(chat_ctx=chat_ctx)
#         async with stream:
#             async for chunk in stream:
#                 token = (chunk.delta and chunk.delta.content) or ""
#                 sentence_buffer += token
#                 full_response += token

#                 if sentence_buffer.endswith((". ", "? ", "! ", "\n")):
#                     if sentence_buffer.strip():
#                         pl = tts_pipeline.get_pipeline()
#                         if not pl._interrupted:          # ← don't enqueue if interrupted
#                             _log(f"[{_ts()}] LLM STREAM  → {sentence_buffer.strip()!r}")
#                             asyncio.ensure_future(tts_pipeline.speak(sentence_buffer.strip()))
#                     sentence_buffer = ""

#         if sentence_buffer.strip():
#             pl = tts_pipeline.get_pipeline()
#             if not pl._interrupted:                      # ← same check for remainder
#                 asyncio.ensure_future(tts_pipeline.speak(sentence_buffer.strip()))

#         _log(f"[{_ts()}] LLM REPLY   → {full_response[:80]!r}")
#         chat_ctx.add_message(role="assistant", content=full_response)

#     except asyncio.CancelledError:
#         if full_response.strip():
#             chat_ctx.add_message(role="assistant", content=full_response)
#         _log(f"[{_ts()}] LLM STREAM  cancelled mid-generation")
#         raise

# async def _stt_consumer(ctx: JobContext) -> None:
#     """
#     Consumes the async generator from stt_pipeline.stt_stream().

#     Each yielded value is a finalised utterance string.
#     Wire your own logic here (LLM call, routing, logging, etc.).
#     Currently just logs — no automatic STT→TTS.
#     """
#     chat_ctx = agents_llm.ChatContext()
#     chat_ctx.add_message(role="system", content="You are a helpful assistant that answers questions.")
#     _log(f"[{_ts()}] STT         listening for mic input...")
#     async for utterance in stt_pipeline.stt_stream(ctx):
#         # ── Plug your downstream logic here ──────────────────────────────
#         # Examples:
#         #   response = await llm.complete(utterance)
#         #   await tts_pipeline.speak(response)
#         _log(f"[{_ts()}] STT RESULT  utterance received → {utterance!r}")
        
#         #Add the user's message to the chat context
#         chat_ctx.add_message(role="user", content=utterance)
        
#         #Generate a response from the LLM based on the chat context
#         stream = llm.chat(chat_ctx = chat_ctx)
#         # collected = await stream.collect()
#         # response = collected.text
#         sentence_buffer = ""
#         full_response = ""
#         async with stream:
#             async for chunk in stream:
#                 token = (chunk.delta and chunk.delta.content) or ""
#                 sentence_buffer += token
#                 full_response += token
                
#                 if sentence_buffer.endswith((". ", "? ", "! ", "\n")):
#                     if sentence_buffer.strip():
#                         _log(f"[{_ts()}] LLM STREAM   → {sentence_buffer.strip()!r}")
#                         asyncio.ensure_future(tts_pipeline.speak(sentence_buffer.strip()))
#                     sentence_buffer = ""
#         if sentence_buffer.strip():
#             _log(f"[{_ts()}] LLM STREAM   → {sentence_buffer.strip()!r}")
#             asyncio.ensure_future(tts_pipeline.speak(sentence_buffer.strip()))
#         _log(f"[{_ts()}] LLM REPLY   → {full_response[:20]!r}")
        
#         # asyncio.ensure_future(tts_pipeline.speak(full_response))

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

    #
    def _cancel_llm():
        global _current_llm_task
        if _current_llm_task and not _current_llm_task.done():
            _current_llm_task.cancel()
            
    tts_pl.set_llm_cancel_fn(_cancel_llm) ##register the cancel function to tts pipeline so it can call it when barge-in happens
    
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
