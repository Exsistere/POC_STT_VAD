import asyncio
import sys
import logging
from datetime import datetime
import os
import json
from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, WorkerOptions

import tts_pipeline_v2 as tts_pipeline
import stt_pipeline as stt_pipeline
import tools
from tools import execute_tool, TOOLS
from livekit.agents import inference
from livekit.agents import llm as agents_llm
from livekit.agents.llm import FunctionCall, FunctionCallOutput
from livekit.plugins import groq, azure, openai
import asyncpg
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

# ─── Global state ─────────────────────────────────────────────────────────────

_db_pool: asyncpg.Pool | None = None
PERSONA_ID = os.getenv("PERSONA_ID", "979425e3-927b-4ffb-8be6-84145075c425")

# ─── ANSI colours ─────────────────────────────────────────────────────────────

YELLOW = "\033[33m"
GREEN  = "\033[32m"
RESET  = "\033[0m"


llm = groq.LLM(
    model = "openai/gpt-oss-120b",
    api_key = os.getenv("GROQ_API_KEY"),
    max_completion_tokens=250,
    temperature=0.3
)

# llm = openai.LLM.with_azure(
#     azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("OPENAI_API_VERSION"),
#     model="openai/gpt-oss-120b",
#     temperature=0.3,
#     max_completion_tokens=250,
# )

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
    Also initializes database connection for RAG tool.
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
    chat_ctx.add_message(
        role="system",
        content=(
            "You are a helpful assistant with access to a company knowledge base. "
            "Keep sentences short and conversational. Use natural pauses with commas and short sentences. "
            "When the user asks questions about products, services, pricing, policies, or technical details, "
            "ALWAYS use the search_knowledge_base tool to find accurate information from the knowledge base before answering. "
            "If the knowledge base has relevant information, cite it in your response."
        )
    )
    
    async for utterance in stt_pipeline.stt_stream(ctx):
        _log(f"[{_ts()}] STT RESULT  utterance received → {utterance!r}")
        
        #cancel previous LLM task if still running
        # Scenario: user says something, LLM still generating previous response, while tts stopped speaking,leading to is_speaking = false and interrupt() not called
        ## TODO seems redundant but just to be safe, we can cancel the previous LLM task here to avoid the half-spoken response when barge-in happens
        if _current_llm_task and not _current_llm_task.done():
            _current_llm_task.cancel()
            _log(f"[{_ts()}] LLM         previous task cancelled")
        #Reset tts interupt flag so new response can be spoken
        tts_pipeline.get_pipeline().reset_interrupt() 
        chat_ctx.add_message(role="user", content=utterance)
        _current_llm_task = asyncio.ensure_future(_generate_and_speak(chat_ctx))

async def _generate_and_speak(chat_ctx) -> None:
    """
    Streams the LLM response to TTS, handling tool calls in a loop:
    
      1. Stream one LLM turn — collect tool-call deltas, flush text to TTS.
      2. If tool calls came back: execute them, append results to chat_ctx, loop.
      3. If plain text came back: flush remainder to TTS and return.
    
    Aggressive streaming: flushes at clause boundaries to minimise
    time-to-first-audio while avoiding mid-word cuts.
    """
    # ── Tuning knobs ──────────────────────────────────────────────────────
    FIRST_CHUNK_MIN  = 20    # chars before the very first TTS flush
    CLAUSE_MIN_CHARS =  8    # min chars for subsequent clause flushes
    HARD_CAP_CHARS   = 120   # flush at word boundary if buffer exceeds this
    SENTENCE_MARKERS = (". ", "? ", "! ", ".\n", "?\n", "!\n")
    CLAUSE_MARKERS   = (", ", "; ", ": ", " - ", "\n")
    MAX_TOOL_ROUNDS  =  5    # cap on tool→LLM loops to prevent infinite cycles
    # ─────────────────────────────────────────────────────────────────────

    def _should_flush(buf: str, first_sent: bool) -> bool:
        size      = len(buf)
        min_chars = FIRST_CHUNK_MIN if not first_sent else CLAUSE_MIN_CHARS

        if size >= HARD_CAP_CHARS:
            return buf[-1] == " " or buf.endswith(CLAUSE_MARKERS) or buf.endswith(SENTENCE_MARKERS)
        if buf.endswith(SENTENCE_MARKERS):
            return size >= min_chars
        if buf.endswith(CLAUSE_MARKERS):
            return size >= min_chars
        return False

    def _flush_buffer(buf: str) -> str:
        """Sends buf to TTS if not interrupted; always returns '' to reset caller."""
        text = buf.strip().lstrip(", ;:-")
        if text:
            pl = tts_pipeline.get_pipeline()
            if not pl._interrupted:
                _log(f"[{_ts()}] LLM STREAM  → {text!r}")
                asyncio.ensure_future(tts_pipeline.speak(text))
        return ""

    full_response = ""  # kept in scope for CancelledError handler

    try:
        for _round in range(MAX_TOOL_ROUNDS):
            sentence_buffer  = ""
            full_response    = ""
            first_chunk_sent = False
            tool_calls_map: dict[str, dict] = {}  # call_id → {id, name, arguments_str}

            # ── Stream one LLM turn ───────────────────────────────────────
            stream = llm.chat(chat_ctx=chat_ctx, tools=TOOLS)
            async with stream:
                async for chunk in stream:
                    # Tool-call delta — merge by call_id to handle streaming fragments
                    if chunk.delta and chunk.delta.tool_calls:
                        for tc in chunk.delta.tool_calls:
                            # Initialize or merge into existing tool call by call_id
                            if tc.call_id not in tool_calls_map:
                                tool_calls_map[tc.call_id] = {
                                    "id": tc.call_id,
                                    "name": tc.name or "",
                                    "arguments_str": "",
                                }
                            else:
                                # Update name if this chunk has it (usually only first chunk)
                                if tc.name:
                                    tool_calls_map[tc.call_id]["name"] = tc.name
                            # Accumulate arguments across all chunks for this call_id
                            tool_calls_map[tc.call_id]["arguments_str"] += tc.arguments or ""
                        continue

                    # Text delta — stream to TTS
                    token = (chunk.delta and chunk.delta.content) or ""
                    sentence_buffer += token
                    full_response   += token

                    if _should_flush(sentence_buffer, first_chunk_sent):
                        sentence_buffer  = _flush_buffer(sentence_buffer)
                        first_chunk_sent = True

            # ── Plain text response — flush remainder and finish ──────────
            if not tool_calls_map:
                if sentence_buffer.strip():
                    _flush_buffer(sentence_buffer)
                _log(f"[{_ts()}] LLM REPLY   → {full_response[:80]!r}")
                chat_ctx.add_message(role="assistant", content=full_response)
                return

            # ── Tool calls — execute, append results, loop ────────────────
            _log(f"[{_ts()}] TOOL CALLS  {[tc['name'] for tc in tool_calls_map.values()]}")

            # 1. Append assistant's text response (if any)
            if full_response:
                chat_ctx.add_message(
                    role="assistant",
                    content=full_response,
                )

            # 2. Execute each tool and append its result using proper FunctionCall/FunctionCallOutput API
            for call_id, tc in tool_calls_map.items():
                try:
                    args = json.loads(tc["arguments_str"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                _log(f"[{_ts()}] TOOL EXEC   {tc['name']}({args})")
                result = await execute_tool(tc["name"], args)
                _log(f"[{_ts()}] TOOL RESULT {tc['name']} → {result[:80]!r}")

                # Insert FunctionCall and FunctionCallOutput using proper LiveKit API
                chat_ctx.insert(FunctionCall(
                    call_id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments_str"] or "{}",
                ))
                chat_ctx.insert(FunctionCallOutput(
                    call_id=tc["id"],
                    name=tc["name"],
                    output=result,
                    is_error=False,
                ))

            # 3. Loop — LLM will now generate a reply informed by the tool results

        # Exhausted MAX_TOOL_ROUNDS without a plain text reply
        _log(f"[{_ts()}] TOOL LOOP   max rounds reached — aborting")
        asyncio.ensure_future(
            tts_pipeline.speak("Sorry, I'm having trouble completing that request.")
        )

    except asyncio.CancelledError:
        if full_response.strip():
            chat_ctx.add_message(role="assistant", content=full_response)
        _log(f"[{_ts()}] LLM STREAM  cancelled mid-generation")
        raise



async def _init_rag_database() -> None:
    """
    Initialize database connection and ToolRegistry for RAG tool.
    Called once at agent startup.
    """
    global _db_pool
    
    if _db_pool is not None:
        _log(f"[{_ts()}] DB INIT     already initialized")
        return
    
    try:
        dsn = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/voice_db")
        _log(f"[{_ts()}] DB INIT     connecting to database...")
        _db_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        
        tools.ToolRegistry.init(db_connection=_db_pool, persona_id=PERSONA_ID)
        _log(f"[{_ts()}] DB INIT     ✅ database and ToolRegistry initialized")
    except Exception as exc:
        _log(f"[{_ts()}] DB ERROR    failed to initialize database: {exc}")
        logger.error("Database initialization failed: %s", exc)

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

    # ── 1. Initialize RAG database ────────────────────────────────────────
    await _init_rag_database()
    
    # ── 2. Connect to room ───────────────────────────────────────────────
    await ctx.connect()
    _log(f"[{_ts()}] ROOM        connected → '{ctx.room.name}'")

    # ── 3. TTS pipeline: init → publish track → start local speaker ───────
    tts_pl = tts_pipeline.init_pipeline(ctx)
    await tts_pipeline.start_pipeline()
    _log(f"[{_ts()}] TTS         pipeline ready")

    #
    def _cancel_llm():
        global _current_llm_task
        if _current_llm_task and not _current_llm_task.done():
            _current_llm_task.cancel()
            
    tts_pl.set_llm_cancel_fn(_cancel_llm) ##register the cancel function to tts pipeline so it can call it when barge-in happens
    
    # ── 4. STT pipeline: init (AgentSession, VAD, Deepgram) ───────────────
    stt_pipeline.init_pipeline(ctx, interrupt_fn=tts_pl.interrupt)
    _log(f"[{_ts()}] STT         pipeline ready")

    _log(
        f"[{_ts()}] BOTH PIPELINES RUNNING\n"
        f"[{_ts()}] ── Speak into your mic  → STT transcript appears above\n"
        f"[{_ts()}] ── Type text + Enter    → TTS speaks it into the room"
    )

    # ── 5. Run both concurrently ───────────────────────────────────────────
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
