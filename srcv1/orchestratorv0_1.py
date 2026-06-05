"""
orchestrator.py
───────────────
PipelineOrchestrator — the per-call coordinator for the WebRTC STT→LLM→TTS pipeline.

Drop-in replacement for VoiceAgentManager in main.py:

    orchestrator = PipelineOrchestrator(
        system_prompt=system_prompt,
        agent_audio_track=agent_audio,
        db_connection=db,
        persona_id=request.persona_id,
    )
    asyncio.create_task(orchestrator.start(user_audio_track))

What it owns per call
─────────────────────
• TTSPipeline (WebRTC sink → AgentAudioTrack)
• STTPipeline (WebRTC source ← browser mic track)
• LLM client  (LiveKit groq.LLM — abstracted behind _LLMClient for future swaps)
• ChatContext  (message history for the duration of the call)
• ToolRegistry (per-call instance with db_connection + persona_id)
• call_transcript list (mirrors VoiceAgentManager.call_transcript)
• call_tools_used list (mirrors VoiceAgentManager.call_tools_used)

VAD loading
───────────
Lazy — loaded on first call via get_vad(), cached module-level.
No app.state dependency.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional
import time
from dotenv import load_dotenv
from livekit.agents import llm as agents_llm
from livekit.agents.llm import FunctionCall, FunctionCallOutput
from livekit.plugins import groq, openai, google, cerebras

from tts import TTSPipeline
from stt import STTPipeline
from tools import execute_tool, TOOLS

load_dotenv(override=True)

# ─── Logging ──────────────────────────────────────────────────────────────────

logger = logging.getLogger("orchestrator")
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

YELLOW = "\033[33m"
GREEN  = "\033[32m"
RESET  = "\033[0m"

# ─── Latency tracking (use monotonic clock to avoid clock skew) ────────────────
# Global dict: (key) → monotonic_time
_latency_tracking: dict[str, float] = {}


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _record_start(key: str) -> None:
    """Record the start time of a stage using monotonic clock."""
    _latency_tracking[key] = time.monotonic()


def _compute_delta_ms(key: str) -> float | None:
    """
    Compute delta in ms if start time exists, otherwise return None.
    Removes the start time after computing to avoid memory leaks.
    """
    if key not in _latency_tracking:
        return None
    start_time = _latency_tracking.pop(key)
    return (time.monotonic() - start_time) * 1000


def _delta_str(name: str, ms: float | None) -> str:
    """Format a single delta as ' Δname=Xms', or '' if None."""
    if ms is None:
        return ""
    return f"  Δ{name}={ms:.0f}ms"


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    for line in message.split("\n"):
        sys.stdout.write(
            f"{YELLOW}{timestamp}{RESET} {GREEN}INFO{RESET} orchestrator  {line}\n"
        )
    sys.stdout.flush()


# ─── Lazy VAD loader ──────────────────────────────────────────────────────────
# Loaded on first call, cached for all subsequent calls.
# Uses an asyncio.Lock to prevent concurrent loads on simultaneous first calls.

_vad = None
_vad_lock = asyncio.Lock()


async def get_vad():
    global _vad
    if _vad is not None:
        return _vad
    async with _vad_lock:
        if _vad is not None:   # double-check after acquiring lock
            return _vad
        from livekit.plugins import silero
        _log(f"[{_ts()}] VAD         loading Silero (first call)...")
        _vad = silero.VAD.load(
            activation_threshold=0.4,
            min_silence_duration=0.2,
        )
        _log(f"[{_ts()}] VAD         loaded")
    return _vad


# ─── LLM abstraction ──────────────────────────────────────────────────────────
# Thin wrapper around the LiveKit groq.LLM plugin.
# To switch to direct Groq SDK or Azure: replace _LLMClient internals only.
# _generate_and_speak calls self._llm.chat(chat_ctx, tools) — that contract
# must be preserved by any future implementation.

class _LLMClient:
    """
    Abstracts the LLM provider.
    Currently wraps livekit.plugins.groq.LLM.

    Future swap point: replace __init__ and chat() internals without touching
    _generate_and_speak. The returned object must support:
        async with client.chat(chat_ctx=..., tools=...) as stream:
            async for chunk in stream:
                chunk.delta.content     # str | None
                chunk.delta.tool_calls  # list | None
    """

    def __init__(self) -> None:
        self._llm = groq.LLM(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            api_key=os.getenv("GROQ_API_KEY"),
            max_completion_tokens=int(os.getenv("LLM_MAX_TOKENS", "250")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        )
        # self._llm = cerebras.LLM(
        #     model=os.getenv("CEREBRAS_MODEL", "gpt-oss-120b"),
        #     api_key=os.getenv("CEREBRAS_API_KEY"),
        #     # max_completion_tokens=int(os.getenv("LLM_MAX_TOKENS", "250")),
        #     temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        # )
        # self._llm = openai.LLM.with_azure(
        #         azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
        #         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        #         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        #         api_version=os.getenv("OPENAI_API_VERSION"),
        #         model="openai/gpt-oss-120b",
        #         temperature=0.3,
        #         max_completion_tokens=250,
        #     )
        # self._llm = google.LLM(
        #     model = "gemini-3.5-flash",
        #     api_key = os.getenv("GEMINI_API_KEY"),
        #     temperature = 0.2,
        #     max_output_tokens = 250,
        #     thinking_config={
        #         "thinking_level": "low",  # disable thinking
        #     },
        #     automatic_function_calling_config = {
        #         "disable": True,  # disable automatic function calling — we handle it ourselves in _generate_and_speak
        #     }
        # )
        
    def chat(self, chat_ctx, tools):
        """Returns an async context manager that yields LLM chunks."""
        return self._llm.chat(chat_ctx=chat_ctx, tools=tools)
        # return self._llm.chat(chat_ctx=chat_ctx,)

# ─── PipelineOrchestrator ─────────────────────────────────────────────────────

class PipelineOrchestrator:
    """
    Per-call coordinator. One instance per WebRTC connection.
    """

    def __init__(
        self,
        system_prompt: str,
        agent_audio_track,
        persona_id: str,
        db_connection,
        language_code: str = "en",   # kept for API compatibility with VoiceAgentManager
    ) -> None:
        self.system_prompt    = system_prompt
        self.agent_audio_track = agent_audio_track
        self.persona_id       = persona_id
        self.db_pool          = db_connection
        self.language_code    = language_code

        # Call record — mirrors VoiceAgentManager
        self.call_transcript: list[dict] = []
        self.call_tools_used: list[dict] = []

        # Internals — created in start()
        self._tts: Optional[TTSPipeline]  = None
        self._stt: Optional[STTPipeline]  = None
        self._llm = _LLMClient()
        self._chat_ctx: Optional[agents_llm.ChatContext] = None
        self._current_llm_task: Optional[asyncio.Task] = None
        
        # Latency tracking — per-orchestrator instance
        self._turn_index = 0                      # increments on each utterance
        self._round_index = 0                     # increments per LLM round within turn
        self._tts_chunk_counter = 0               # increments per speak() call; resets each turn
        self._tool_call_counter = 0               # increments per tool call
        
        # Timestamps for cross-file coordination (monotonic, in seconds)
        self._vad_start_time: float = 0.0         # set by STT on UTTERANCE
        self._vad_end_time: float = 0.0           # set by STT on UTTERANCE (captures latest VAD END)
        self._utterance_time: float = 0.0         # set when UTTERANCE logged
        self._llm_reply_time: float = 0.0         # set when LLM REPLY logged (for Δllm_to_ttfa)

    # ── Public entrypoint ─────────────────────────────────────────────────────

    async def start(self, user_audio_track) -> None:
        """
        Boot all pipelines and run the STT consumer loop until the call ends.
        Called as an asyncio.Task from main.py's on_track handler.
        """
        _log(
            f"[{_ts()}] ══════════════════════════════════════════\n"
            f"[{_ts()}] CALL START   persona={self.persona_id}\n"
            f"[{_ts()}] ══════════════════════════════════════════"
        )

        try:
            # 1. VAD — lazy load (no-op if already cached)
            vad = await get_vad()

            # 2. TTS pipeline
            self._tts = TTSPipeline.for_webrtc(self.agent_audio_track)
            await self._tts.start()
            _log(f"[{_ts()}] TTS         pipeline ready")

            # 3. Wire LLM cancel → TTS barge-in so VAD speech start cancels generation
            def _cancel_llm():
                if self._current_llm_task and not self._current_llm_task.done():
                    self._current_llm_task.cancel()
                    _log(f"[{_ts()}] LLM         cancelled by barge-in")

            self._tts.set_llm_cancel_fn(_cancel_llm)

            # 4. STT pipeline — interrupt_fn fires on VAD speech start
            self._stt = STTPipeline.for_webrtc(
                track=user_audio_track,
                vad=vad,
                interrupt_fn=self._tts.interrupt,
            )
            _log(f"[{_ts()}] STT         pipeline ready")

            # 5. Chat context — system prompt seeded once per call
            self._chat_ctx = agents_llm.ChatContext()
            self._chat_ctx.add_message(role="system", content=self.system_prompt)

            # 6. ToolRegistry — per-call so persona_id is scoped correctly
            from tools import ToolRegistry
            ToolRegistry.init(db_connection=self.db_pool, persona_id=self.persona_id)
            _log(f"[{_ts()}] TOOLS       ToolRegistry initialized for persona={self.persona_id}")

            _log(f"[{_ts()}] ORCHESTRATOR running — waiting for utterances")

            # 7. Run STT consumer — blocks until call ends
            await self._stt_consumer()

        except asyncio.CancelledError:
            _log(f"[{_ts()}] ORCHESTRATOR cancelled")
        except Exception as exc:
            logger.error(f"Orchestrator error: {exc}", exc_info=True)
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """Tear down all resources and save call logs."""
        _log(f"[{_ts()}] ORCHESTRATOR closing")

        # Cancel any in-flight LLM generation
        if self._current_llm_task and not self._current_llm_task.done():
            self._current_llm_task.cancel()
            try:
                await self._current_llm_task
            except asyncio.CancelledError:
                pass

        # Close TTS
        if self._tts:
            await self._tts.aclose()

        # Close STT
        if self._stt:
            await self._stt.aclose()

        # Save transcript + tool events to DB — mirrors VoiceAgentManager.save_call_logs()
        await self._save_call_logs()

        _log(f"[{_ts()}] ORCHESTRATOR closed")

    # ── STT consumer ──────────────────────────────────────────────────────────

    async def _stt_consumer(self) -> None:
        """
        Drains utterances from STTPipeline.stream() and fires _generate_and_speak
        for each one. Mirrors _stt_consumer() from main.py exactly, translated
        from global state to instance state.
        """
        async for utterance in self._stt.stream():
            self._turn_index += 1
            turn_idx = self._turn_index
            _log(f"[{_ts()}] STT RESULT  utterance received → {utterance!r}")
            
            # Record start time for utt_to_llm latency tracking
            _record_start(f"utt_to_llm_{turn_idx}")
            # Store for e2e tracking (will be passed to TTS)
            self._vad_start_time = _latency_tracking.get(("vad_to_utt", turn_idx), time.monotonic())

            # Append to transcript
            self.call_transcript.append({"speaker": "User", "text": utterance})

            # Cancel previous LLM task if still running
            # (covers the case where TTS finished but LLM is still generating)
            if self._current_llm_task and not self._current_llm_task.done():
                self._current_llm_task.cancel()
                _log(f"[{_ts()}] LLM         previous task cancelled (new utterance)")

            # Reset TTS interrupt flag so next response can be spoken
            self._tts.reset_interrupt()

            self._chat_ctx.add_message(role="user", content=utterance)
            self._current_llm_task = asyncio.ensure_future(
                self._generate_and_speak(self._chat_ctx, turn_idx)
            )

    # ── LLM streaming + TTS flush ─────────────────────────────────────────────

    async def _generate_and_speak(self, chat_ctx, turn_idx: int) -> None:
        """
        Streams the LLM response to TTS, handling tool calls in a loop.

        Copied verbatim from main.py's _generate_and_speak and translated
        from global references to instance references:
            tts_pipeline.speak(text)          → self._tts.speak(text)
            tts_pipeline.get_pipeline()       → self._tts
            llm.chat(...)                     → self._llm.chat(...)
            execute_tool(...)                 → execute_tool(...) (unchanged)
            chat_ctx.insert(FunctionCall...)  → unchanged (LiveKit API)

        Tuning knobs, _should_flush, _flush_buffer, tool loop, CancelledError
        handler — all identical to main.py.
        """
        # ── Tuning knobs ──────────────────────────────────────────────────
        FIRST_CHUNK_MIN  = 20
        CLAUSE_MIN_CHARS =  8
        HARD_CAP_CHARS   = 120
        SENTENCE_MARKERS = (". ", "? ", "! ", ".\n", "?\n", "!\n")
        CLAUSE_MARKERS   = (", ", "; ", ": ", " - ", "\n")
        MAX_TOOL_ROUNDS  =  5
        # ──────────────────────────────────────────────────────────────────

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
            if text and not self._tts._interrupted:
                _log(f"[{_ts()}] LLM STREAM  → {text!r}")
                asyncio.ensure_future(self._tts.speak(text, vad_start_time=self._vad_start_time, llm_start_time=self._llm_start_time))
            return ""

        full_response = ""   # kept in scope for CancelledError handler

        try:
            for _round in range(MAX_TOOL_ROUNDS):
                self._round_index = _round + 1
                _log(f"[{_ts()}] LLM STREAM  starting round {self._round_index}")
                sentence_buffer  = ""
                full_response    = ""
                first_chunk_sent = False
                tool_calls_map: dict[str, dict] = {}

                # ── Stream one LLM turn ───────────────────────────────────
                stream = self._llm.chat(chat_ctx, TOOLS)
                async with stream:
                    first_chunk = True
                    first_chunk_time = time.perf_counter()
                    async for chunk in stream:
                        if first_chunk:
                            latency = time.perf_counter() - first_chunk_time
                            # Record start time for llm_to_ttfa tracking (use monotonic)
                            self._llm_start_time = time.monotonic()
                            utt_to_llm_delta = _compute_delta_ms(f"utt_to_llm_{turn_idx}")
                            _log(f"[{_ts()}] LLM STREAM  first chunk received after {latency:.2f}s")
                            first_chunk = False
                        # _log(f"[{_ts()}] LLM STREAM  chunk received → content={chunk.delta.content if chunk.delta and chunk.delta.content else ''}")
                        # Tool-call delta — merge fragments by call_id
                        if chunk.delta and chunk.delta.tool_calls:
                            for tc in chunk.delta.tool_calls:
                                if tc.call_id not in tool_calls_map:
                                    tool_calls_map[tc.call_id] = {
                                        "id":            tc.call_id,
                                        "name":          tc.name or "",
                                        "arguments_str": "",
                                    }
                                else:
                                    if tc.name:
                                        tool_calls_map[tc.call_id]["name"] = tc.name
                                tool_calls_map[tc.call_id]["arguments_str"] += tc.arguments or ""
                            continue

                        # Text delta — stream to TTS
                        token           = (chunk.delta and chunk.delta.content) or ""
                        sentence_buffer += token
                        full_response   += token

                        if _should_flush(sentence_buffer, first_chunk_sent):
                            sentence_buffer  = _flush_buffer(sentence_buffer)
                            first_chunk_sent = True

                # ── Plain text response — flush remainder and finish ──────
                if not tool_calls_map:
                    if sentence_buffer.strip():
                        _flush_buffer(sentence_buffer)
                    _log(f"[{_ts()}] LLM REPLY   → {full_response[:80]!r}")
                    chat_ctx.add_message(role="assistant", content=full_response)
                    # Append to transcript
                    self.call_transcript.append(
                        {"speaker": "Agent", "text": full_response}
                    )
                    return

                # ── Tool calls — execute, append results, loop ────────────
                _log(f"[{_ts()}] TOOL CALLS  {[tc['name'] for tc in tool_calls_map.values()]}")

                if full_response:
                    chat_ctx.add_message(role="assistant", content=full_response)

                for call_id, tc in tool_calls_map.items():
                    try:
                        args = json.loads(tc["arguments_str"] or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    self._tool_call_counter += 1
                    tool_idx = self._tool_call_counter
                    tool_name = tc["name"]
                    _record_start(f"tool_{tool_name}_{tool_idx}")
                    _log(f"[{_ts()}] TOOL EXEC   {tool_name}({args})")
                    result = await execute_tool(tool_name, args)
                    tool_delta = _compute_delta_ms(f"tool_{tool_name}_{tool_idx}")
                    _log(f"[{_ts()}] TOOL RESULT {tool_name} → {result[:80]!r}{tool_delta}")

                    # Record tool event — mirrors VoiceAgentManager.call_tools_used
                    self.call_tools_used.append({
                        "type": tool_name,
                        "data": {"args": args, "result": result},
                    })

                    # LiveKit ChatContext tool result API — unchanged from main.py
                    chat_ctx.insert(FunctionCall(
                        call_id=tc["id"],
                        name=tool_name,
                        arguments=tc["arguments_str"] or "{}",
                    ))
                    chat_ctx.insert(FunctionCallOutput(
                        call_id=tc["id"],
                        name=tool_name,
                        output=result,
                        is_error=False,
                    ))

                # Loop — LLM generates reply using tool results

            # Exhausted MAX_TOOL_ROUNDS
            _log(f"[{_ts()}] TOOL LOOP   max rounds reached — aborting")
            asyncio.ensure_future(
                self._tts.speak("Sorry, I'm having trouble completing that request.", vad_start_time=self._vad_start_time, llm_start_time=self._llm_start_time)
            )

        except asyncio.CancelledError:
            # Save whatever was generated before cancellation
            if full_response.strip():
                chat_ctx.add_message(role="assistant", content=full_response)
            _log(f"[{_ts()}] LLM STREAM  cancelled mid-generation")
            raise

    # ── Call log persistence ───────────────────────────────────────────────────

    async def _save_call_logs(self) -> None:
        """
        Saves transcript and tool events to Postgres.
        Mirrors VoiceAgentManager.save_call_logs() exactly.
        Acquires a connection from the pool for all DB operations, then releases it.
        No-op if db_pool is None or transcript is empty.
        """
        if not self.db_pool:
            return
        if not self.call_transcript and not self.call_tools_used:
            return

        try:
            async with self.db_pool.acquire() as conn:
                call_id = await conn.fetchval(
                    """
                    INSERT INTO calls (persona_id, duration_seconds)
                    VALUES ($1::uuid, 0) RETURNING id;
                    """,
                    self.persona_id,
                )

                if self.call_transcript:
                    transcript_data = [
                        (call_id, t["speaker"], t["text"])
                        for t in self.call_transcript
                    ]
                    await conn.executemany(
                        """
                        INSERT INTO call_transcripts (call_id, speaker, text)
                        VALUES ($1, $2, $3)
                        """,
                        transcript_data,
                    )

                if self.call_tools_used:
                    events_data = [
                        (call_id, e["type"], json.dumps(e["data"]))
                        for e in self.call_tools_used
                    ]
                    await conn.executemany(
                        """
                        INSERT INTO call_events (call_id, event_type, event_data)
                        VALUES ($1, $2, $3::jsonb)
                        """,
                        events_data,
                    )

                _log(f"[{_ts()}] DB          call logs saved — call_id={call_id}")

        except Exception as exc:
            logger.error(f"Failed to save call logs: {exc}")
