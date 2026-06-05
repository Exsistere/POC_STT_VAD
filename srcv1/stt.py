"""
stt_pipeline.py
───────────────
Dual-mode STT pipeline — works identically from:

  • CLI / LiveKit entrypoint  (my_agent)         → _LiveKitSource
  • WebRTC / PipelineOrchestrator                → _WebRTCSource

Public API (unchanged from v1 — CLI main.py needs no edits):
    prewarm(proc)                           (CLI only — loads VAD into proc.userdata)
    init_pipeline(ctx, interrupt_fn)        CLI mode
    init_pipeline_webrtc(track,             WebRTC mode
                         vad,
                         interrupt_fn)
    stt_stream(ctx=None)  → AsyncGenerator[str, None]

Both modes yield finalised utterance strings through the same stt_stream() interface.
"""

import asyncio
import logging
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncGenerator, Callable, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

# ─── Logging ──────────────────────────────────────────────────────────────────

logger = logging.getLogger("stt_pipeline")
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

# ─── Shared constants ─────────────────────────────────────────────────────────

# After VAD silence, wait this long for the final Deepgram transcript
FINAL_WAIT = 0.05

BLUE  = "\033[34m"
GREEN = "\033[32m"
RESET = "\033[0m"

# ─── Latency tracking (use monotonic clock to avoid clock skew) ────────────────
_latency_tracking: dict[tuple[str, int], float] = {}  # (stage, index) → start_time


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _record_start(stage: str, index: int) -> None:
    """Record the start time of a stage using monotonic clock."""
    _latency_tracking[(stage, index)] = time.monotonic()


def _compute_delta_ms(stage: str, index: int) -> str:
    """
    Compute delta in ms if start time exists, otherwise return empty string.
    Removes the start time after computing to avoid memory leaks.
    """
    key = (stage, index)
    if key not in _latency_tracking:
        return ""
    start_time = _latency_tracking.pop(key)
    delta_ms = (time.monotonic() - start_time) * 1000
    return f"  Δ{stage}={delta_ms:.0f}ms"


def log_transcript(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    for line in message.split("\n"):
        sys.stdout.write(
            f"{BLUE}{timestamp}{RESET} {GREEN}INFO{RESET} stt_pipeline {line}\n"
        )
    sys.stdout.flush()


# ─── Utterance state machine (shared by both modes) ───────────────────────────

class _UtteranceStateMachine:
    """
    Accumulates VAD + transcript events into finalised utterance strings.

    Identical logic regardless of audio source — the source calls:
        .on_speech_start()
        .on_speech_end()
        .on_interim(text)
        .on_final(text)

    Finalised utterances are pushed to utterance_queue.
    interrupt_fn is called on speech_start if barge-in is active.
    """

    def __init__(
        self,
        utterance_queue: asyncio.Queue,
        interrupt_fn: Optional[Callable] = None,
    ) -> None:
        self._queue        = utterance_queue
        self._interrupt_fn = interrupt_fn

        self._speech_active      = False
        self._speech_index       = 0
        self._last_interim       = ""
        self._utterance_parts: list[str] = []
        self._speech_start_time  = 0.0
        self._first_final_time   = 0.0
        self._vad_ended          = False
        self._flushed            = False
        self._flush_task: asyncio.Task | None = None

    def on_speech_start(self) -> None:
        # Barge-in: cancel TTS if speaking
        log_transcript(f"│  [{_ts()}] USER STATE  speaking")
        if self._interrupt_fn is not None:
            try:
                self._interrupt_fn()
            except Exception:
                pass

        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._do_flush()

        self._speech_active     = True
        self._speech_index     += 1
        self._speech_start_time = time.monotonic()
        self._first_final_time  = 0.0
        self._flushed           = False
        self._vad_ended         = False
        self._last_interim      = ""
        self._utterance_parts   = []
        
        # Record start time for vad_to_utt latency tracking and e2e tracking
        _record_start("vad_to_utt", self._speech_index)
        _record_start("e2e", self._speech_index)
        
        log_transcript(
            f"┌─── SPEECH #{self._speech_index} START [{_ts()}] {'─' * 30}"
        )

    def on_speech_end(self) -> None:
        self._speech_active = False
        self._vad_ended     = True
        self._flush_task    = asyncio.ensure_future(self._wait_and_flush())

    def on_interim(self, text: str, lang: str = "?") -> None:
        self._last_interim = text
        log_transcript(f"│  [{_ts()}] INTERIM    [{lang}]  {text}")

    def on_final(self, text: str) -> None:
        if not self._utterance_parts:
            self._first_final_time = time.monotonic()
        self._utterance_parts.append(text.strip())

        if self._vad_ended:
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
            self._do_flush()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _do_flush(self) -> None:
        if self._flushed:
            return
        if not self._utterance_parts and not self._last_interim:
            return
        self._flushed = True

        full = " ".join(self._utterance_parts).strip() or self._last_interim.strip()
        if full:
            e2e_ms = (self._first_final_time - self._speech_start_time) * 1000
            vad_to_utt_delta = _compute_delta_ms("vad_to_utt", self._speech_index)
            log_transcript(
                f"│  [{_ts()}] EOU        VAD\n"
                f"│  [{_ts()}] UTTERANCE  {full}{vad_to_utt_delta}\n"
                f"│  [{_ts()}] E2E        {e2e_ms:.0f}ms (speech start → first final)\n"
                f"└─── SPEECH #{self._speech_index} END   [{_ts()}] {'─' * 30}"
            )
            self._queue.put_nowait(full)

        self._utterance_parts.clear()
        self._last_interim = ""

    async def _wait_and_flush(self) -> None:
        await asyncio.sleep(FINAL_WAIT)
        self._do_flush()


# ─── Input source abstraction ─────────────────────────────────────────────────

class _InputSource(ABC):
    """
    Strategy interface — encapsulates where audio comes from and how
    VAD + STT are driven. The state machine is injected so both sources
    use identical utterance accumulation logic.
    """

    @abstractmethod
    async def run(self, utterance_queue: asyncio.Queue, interrupt_fn) -> None:
        """
        Drive audio → VAD → STT until cancelled.
        Push finalised utterance strings into utterance_queue.
        Push None sentinel when done.
        Call interrupt_fn() on VAD speech start (barge-in).
        Must handle asyncio.CancelledError cleanly.
        """


# ─── LiveKit source (CLI mode) ────────────────────────────────────────────────

class _LiveKitSource(_InputSource):
    """
    Subscribes to a LiveKit room's mic track via AgentSession + room_io.
    Noise cancellation via ai_coustics. VAD via Silero (loaded in prewarm).
    Deepgram via LiveKit's inference wrapper.
    """

    def __init__(self, ctx) -> None:
        self._ctx = ctx

    async def run(self, utterance_queue: asyncio.Queue, interrupt_fn) -> None:
        # Late imports — only needed in CLI mode
        from livekit.agents import (
            AgentSession, Agent, inference, room_io, TurnHandlingOptions,
            UserInputTranscribedEvent, UserStateChangedEvent,
        )
        from livekit.plugins import ai_coustics

        class _SilentAgent(Agent):
            def __init__(self):
                super().__init__(instructions="")

        session = AgentSession(
            stt=inference.STT(model="deepgram/nova-3", language="en-US"),
            vad=self._ctx.proc.userdata["vad"],
            turn_handling=TurnHandlingOptions(turn_detection="vad"),
        )

        # Inline utterance state machine — same logic as _WebRTCSource
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
                vad_to_utt_delta = _compute_delta_ms("vad_to_utt", _speech_index)
                log_transcript(
                    f"│  [{_ts()}] EOU        VAD\n"
                    f"│  [{_ts()}] UTTERANCE  {full}{vad_to_utt_delta}\n"
                    f"│  [{_ts()}] E2E        {e2e_ms:.0f}ms (speech start → first final)\n"
                    f"└─── SPEECH #{_speech_index} END   [{_ts()}] {'─' * 30}"
                )
                utterance_queue.put_nowait(full)
            _utterance_parts.clear()
            _last_interim = ""

        async def _wait_for_final_then_flush():
            await asyncio.sleep(FINAL_WAIT)
            _do_flush()

        @session.on("user_state_changed")
        def on_user_state(event: UserStateChangedEvent):
            nonlocal _speech_active, _speech_index, _last_interim, _utterance_parts
            nonlocal _speech_start_time, _flushed, _vad_ended, _flush_task, _first_final_time

            if event.new_state == "speaking":
                log_transcript(f"│  [{_ts()}] USER STATE  speaking")
                if interrupt_fn is not None:
                    try:
                        interrupt_fn()
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
                # Record start time for vad_to_utt and e2e latency tracking
                _record_start("vad_to_utt", _speech_index)
                _record_start("e2e", _speech_index)
                log_transcript(f"┌─── SPEECH #{_speech_index} START [{_ts()}] {'─' * 30}")

            elif event.new_state != "speaking" and _speech_active:
                _speech_active = False
                _vad_ended     = True
                _flush_task    = asyncio.ensure_future(_wait_for_final_then_flush())

        @session.on("user_input_transcribed")
        def on_transcript(event: UserInputTranscribedEvent):
            nonlocal _last_interim, _utterance_parts, _first_final_time, _flush_task
            lang = getattr(event, "language", "?")
            if not event.is_final:
                _last_interim = event.transcript
                log_transcript(f"│  [{_ts()}] INTERIM    [{lang}]  {event.transcript}")
                return
            if not _utterance_parts:
                _first_final_time = time.monotonic()
            _utterance_parts.append(event.transcript.strip())
            if _vad_ended:
                if _flush_task and not _flush_task.done():
                    _flush_task.cancel()
                _do_flush()

        await session.start(
            agent=_SilentAgent(),
            room=self._ctx.room,
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

        try:
            await asyncio.Future()   # keep alive until cancelled
        except asyncio.CancelledError:
            pass
        finally:
            utterance_queue.put_nowait(None)   # sentinel


# ─── WebRTC source (PipelineOrchestrator mode) ────────────────────────────────

class _WebRTCSource(_InputSource):
    """
    Receives mic audio from an aiortc MediaStreamTrack (the browser).

    STT:  Raw websockets to wss://api.deepgram.com/v1/listen — same approach
          as GeminiDeepgramVoiceAgent, no deepgram-sdk wrapper needed.
    VAD:  Silero driven from a concurrent frame-feeder task via livekit VAD stream.
          VAD controls the utterance state machine (speech start/end).
          Audio is sent to Deepgram continuously (not gated by VAD) so Deepgram
          always has context — VAD only drives the state machine.
    Resampling: av.AudioResampler (same tool used in GeminiDeepgramVoiceAgent's
          _stream_mic_to_deepgram) — resamples browser 48kHz → 16kHz for both
          Deepgram and Silero.
    KeepAlive: periodic JSON ping every 5s prevents Deepgram from timing out
          during silence gaps.

    Args:
        mic_track:  aiortc RemoteStreamTrack from on_track().
        vad:        silero.VAD instance — loaded once in FastAPI lifespan startup.
    """

    DEEPGRAM_URL = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-3"
        "&encoding=linear16"
        "&sample_rate=16000"
        "&channels=1"
        "&language=en-US"
        "&interim_results=true"
        "&endpointing=300"
        "&utterance_end_ms=1000"
    )
    KEEPALIVE_INTERVAL = 5   # seconds between KeepAlive pings

    def __init__(self, mic_track, vad) -> None:
        self._mic_track = mic_track
        self._vad       = vad

    async def run(
        self,
        utterance_queue: asyncio.Queue,
        interrupt_fn,
    ) -> None:
        import os
        import json
        import numpy as np
        import websockets
        import av
        from livekit.agents.vad import VADEventType
        from livekit import rtc as lk_rtc

        VAD_SAMPLE_RATE = 16000

        deepgram_headers = {
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY', '')}"
        }

        # ── Utterance state machine ────────────────────────────────────────
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
                vad_to_utt_delta = _compute_delta_ms("vad_to_utt", _speech_index)
                log_transcript(
                    f"│  [{_ts()}] EOU        VAD\n"
                    f"│  [{_ts()}] UTTERANCE  {full}{vad_to_utt_delta}\n"
                    f"│  [{_ts()}] E2E        {e2e_ms:.0f}ms (speech start → first final)\n"
                    f"└─── SPEECH #{_speech_index} END   [{_ts()}] {'─' * 30}"
                )
                utterance_queue.put_nowait(full)
            _utterance_parts.clear()
            _last_interim = ""

        async def _wait_for_final_then_flush():
            await asyncio.sleep(FINAL_WAIT)
            _do_flush()

        # ── VAD callbacks ──────────────────────────────────────────────────

        def _on_vad_speech_start():
            nonlocal _speech_active, _speech_index, _last_interim, _utterance_parts
            nonlocal _speech_start_time, _flushed, _vad_ended, _flush_task, _first_final_time

            log_transcript(f"│  [{_ts()}] USER STATE  speaking ← VAD START")
            if interrupt_fn is not None:
                try:
                    interrupt_fn()
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
            # Record start time for vad_to_utt and e2e latency tracking
            _record_start("vad_to_utt", _speech_index)
            _record_start("e2e", _speech_index)
            log_transcript(
                f"┌─── SPEECH #{_speech_index} START [{_ts()}] {'─' * 30}"
            )

        def _on_vad_speech_end():
            nonlocal _speech_active, _vad_ended, _flush_task
            log_transcript(f"│  [{_ts()}] USER STATE  silence ← VAD END")
            if not _speech_active:
                return
            _speech_active = False
            _vad_ended     = True
            _flush_task    = asyncio.ensure_future(_wait_for_final_then_flush())

        # ── Deepgram transcript handler (raw JSON from websocket) ──────────
        # Mirrors GeminiDeepgramVoiceAgent._listen_to_deepgram_and_trigger
        # but feeds the utterance state machine instead of triggering LLM.

        async def _listen_deepgram(dg_ws):
            nonlocal _last_interim, _utterance_parts, _first_final_time, _flush_task
            try:
                async for message in dg_ws:
                    try:
                        data = json.loads(message)
                    except Exception:
                        continue

                    if data.get("type") != "Results":
                        continue

                    channel      = data.get("channel", {})
                    if isinstance(channel, list):
                        channel  = channel[0] if channel else {}
                    alternatives = channel.get("alternatives", [{}])
                    alt          = alternatives[0] if alternatives else {}
                    transcript   = alt.get("transcript", "").strip() if isinstance(alt, dict) else ""
                    confidence   = alt.get("confidence", 0.0) if isinstance(alt, dict) else 0.0

                    if not transcript:
                        continue

                    # Ignore very low confidence noise
                    if confidence < 0.3:
                        continue

                    is_final = data.get("is_final", False)
                    lang     = data.get("channel", {}).get("detected_language", "?") if isinstance(data.get("channel"), dict) else "?"

                    if not is_final:
                        _last_interim = transcript
                        log_transcript(f"│  [{_ts()}] INTERIM    [{lang}]  {transcript}")
                        continue

                    # Final transcript
                    log_transcript(f"[{_ts()}] WEBRTC      final transcript (conf={confidence:.2f}): {transcript[:60]}")
                    if not _utterance_parts:
                        _first_final_time = time.monotonic()
                    _utterance_parts.append(transcript)

                    if _vad_ended:
                        if _flush_task and not _flush_task.done():
                            _flush_task.cancel()
                        _do_flush()

            except websockets.exceptions.ConnectionClosedOK:
                log_transcript(f"[{_ts()}] WEBRTC      Deepgram connection closed cleanly")
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error(f"[WEBRTC] Deepgram listen error: {exc}")

        # ── KeepAlive ping task ────────────────────────────────────────────
        # Mirrors GeminiDeepgramVoiceAgent._send_deepgram_keepalive

        async def _send_keepalive(dg_ws):
            try:
                while True:
                    await asyncio.sleep(self.KEEPALIVE_INTERVAL)
                    await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                    log_transcript(f"[{_ts()}] WEBRTC      KeepAlive sent")
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(f"[WEBRTC] KeepAlive error: {exc}")

        # ── VAD frame-feeder task ──────────────────────────────────────────
        # VAD stream is an async generator — must run in its own task.
        # Mirrors the _vad_pump + _frame_feeder pattern from stt.py _WebRTCSource.

        frame_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        vad_stream = self._vad.stream()

        async def _vad_pump():
            try:
                async for vad_event in vad_stream:
                    evt_type = getattr(vad_event, "type", None)
                    if evt_type == VADEventType.START_OF_SPEECH:
                        _on_vad_speech_start()
                    elif evt_type == VADEventType.END_OF_SPEECH:
                        _on_vad_speech_end()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error(f"[WEBRTC] VAD pump error: {exc}")

        async def _frame_feeder():
            try:
                while True:
                    item = await frame_queue.get()
                    if item is None:
                        break
                    pcm_16k_bytes, samples_16k = item
                    audio_frame = lk_rtc.AudioFrame(
                        data=pcm_16k_bytes,
                        sample_rate=VAD_SAMPLE_RATE,
                        num_channels=1,
                        samples_per_channel=samples_16k,
                    )
                    vad_stream.push_frame(audio_frame)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error(f"[WEBRTC] Frame feeder error: {exc}")
            finally:
                vad_stream.flush()

        # ── Mic recv loop ──────────────────────────────────────────────────
        # Mirrors GeminiDeepgramVoiceAgent._stream_mic_to_deepgram:
        # uses av.AudioResampler to resample browser 48kHz → 16kHz.
        # Sends resampled PCM to Deepgram always (not gated by VAD).

        async def _mic_to_deepgram(dg_ws):
            resampler  = av.AudioResampler(format="s16", layout="mono", rate=VAD_SAMPLE_RATE)
            frame_count = 0
            try:
                while True:
                    try:
                        frame = await self._mic_track.recv()
                    except Exception as exc:
                        log_transcript(f"[{_ts()}] WEBRTC      track recv error: {exc}")
                        break

                    frame_count += 1
                    resampled_frames = resampler.resample(frame)
                    for r_frame in resampled_frames:
                        pcm_bytes = r_frame.to_ndarray().tobytes()
                        samples   = r_frame.samples

                        # Send to Deepgram ##TODO: send only if VAD detects speech? But then we lose context for Deepgram's endpointing logic. Maybe send all but mark non-speech frames with a flag?
                        try:
                            await dg_ws.send(pcm_bytes)
                        except Exception as exc:
                            log_transcript(f"[{_ts()}] WEBRTC      Deepgram send error: {exc}")

                        # Enqueue for VAD
                        try:
                            frame_queue.put_nowait((pcm_bytes, samples))
                        except asyncio.QueueFull:
                            pass   # drop frame — VAD catches up on next tick

                    if frame_count % 200 == 0:
                        log_transcript(f"[{_ts()}] WEBRTC      {frame_count} frames sent to Deepgram + VAD")

            except asyncio.CancelledError:
                pass
            finally:
                # Signal frame feeder + close Deepgram stream
                await frame_queue.put(None)
                try:
                    await dg_ws.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass

        # ── Connect and run all tasks concurrently ─────────────────────────

        log_transcript(f"[{_ts()}] WEBRTC      opening Deepgram WebSocket connection")
        try:
            async with websockets.connect(
                self.DEEPGRAM_URL,
                additional_headers=deepgram_headers,
            ) as dg_ws:
                log_transcript(f"[{_ts()}] WEBRTC      ✓ Deepgram connection open")

                vad_pump_task     = asyncio.ensure_future(_vad_pump())
                frame_feeder_task = asyncio.ensure_future(_frame_feeder())
                keepalive_task    = asyncio.ensure_future(_send_keepalive(dg_ws))
                listen_task       = asyncio.ensure_future(_listen_deepgram(dg_ws))

                try:
                    await _mic_to_deepgram(dg_ws)
                except asyncio.CancelledError:
                    pass
                finally:
                    keepalive_task.cancel()
                    vad_pump_task.cancel()
                    await asyncio.gather(
                        frame_feeder_task,
                        vad_pump_task,
                        keepalive_task,
                        listen_task,
                        return_exceptions=True,
                    )

        except Exception as exc:
            logger.error(f"[WEBRTC] Failed to connect to Deepgram: {exc}")
        finally:
            log_transcript(f"[{_ts()}] WEBRTC      STT source closed")
            utterance_queue.put_nowait(None)   # sentinel — unblocks stt_stream()

    async def aclose(self) -> None:
        pass   # all resources torn down inside run() finally block


# ─── STTPipeline ──────────────────────────────────────────────────────────────

class STTPipeline:
    """
    Mode-agnostic STT pipeline.

    Utterance accumulation logic lives in _UtteranceStateMachine (shared).
    Audio ingestion is delegated to the injected _InputSource.

    Construct via class methods:
        STTPipeline.for_livekit(ctx, interrupt_fn)
        STTPipeline.for_webrtc(track, vad, interrupt_fn)
    """

    def __init__(self, source: _InputSource, interrupt_fn=None) -> None:
        self._source       = source
        self._interrupt_fn = interrupt_fn
        self._utterance_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._session_task: asyncio.Task | None = None

    @classmethod
    def for_livekit(cls, ctx, interrupt_fn=None) -> "STTPipeline":
        """CLI / LiveKit entrypoint mode."""
        return cls(_LiveKitSource(ctx), interrupt_fn)

    @classmethod
    def for_webrtc(cls, track, vad, interrupt_fn=None) -> "STTPipeline":
        """WebRTC / PipelineOrchestrator mode."""
        return cls(_WebRTCSource(track, vad), interrupt_fn)

    async def stream(self) -> AsyncGenerator[str, None]:
        """
        Public generator — yields finalised utterance strings.

        Starts the source session as a background task so this generator
        can yield while audio processing runs concurrently.

        Usage:
            async for utterance in pipeline.stream():
                print(utterance)
        """
        self._session_task = asyncio.ensure_future(
            self._source.run(self._utterance_queue, self._interrupt_fn)
        )

        try:
            while True:
                utterance = await self._utterance_queue.get()
                if utterance is None:   # sentinel — source ended
                    break
                yield utterance
        finally:
            if self._session_task and not self._session_task.done():
                self._session_task.cancel()
                try:
                    await self._session_task
                except asyncio.CancelledError:
                    pass

    async def aclose(self) -> None:
        """Tear down the source session."""
        if self._session_task and not self._session_task.done():
            self._session_task.cancel()
            try:
                await self._session_task
            except asyncio.CancelledError:
                pass
        # Push sentinel so any awaiting stream() generator unblocks
        self._utterance_queue.put_nowait(None)


# ─── Module-level API (CLI main.py calls these — no changes needed there) ─────

_pipeline: STTPipeline | None = None


def prewarm(proc) -> None:
    """
    CLI only — loads Silero VAD into proc.userdata before the job starts.
    Called by LiveKit worker prewarm hook. Unchanged from v1.
    """
    try:
        from livekit.plugins import silero
        proc.userdata["vad"] = silero.VAD.load(
            activation_threshold=0.4,
            min_silence_duration=0.3,
        )
    except Exception as e:
        logger.error(f"Error loading VAD: {e}")


def init_pipeline(ctx, interrupt_fn=None) -> None:
    """
    CLI / LiveKit mode initialiser.
    Called from my_agent() exactly as before.
    """
    global _pipeline
    _pipeline = STTPipeline.for_livekit(ctx, interrupt_fn)


def init_pipeline_webrtc(track, vad, interrupt_fn=None) -> STTPipeline:
    """
    WebRTC mode initialiser.
    Called from PipelineOrchestrator — one instance per call.
    Does NOT set module-level _pipeline so CLI and WebRTC don't stomp each other.
    Returns the instance directly for the orchestrator to hold.
    """
    return STTPipeline.for_webrtc(track, vad, interrupt_fn)


async def stt_stream(ctx=None) -> AsyncGenerator[str, None]:
    """
    Module-level stream — used by CLI main.py. Unchanged call signature.
    ctx is accepted for API compatibility but not used (pipeline already holds ctx).
    """
    assert _pipeline is not None, "call init_pipeline(ctx) first"
    async for utterance in _pipeline.stream():
        yield utterance


# ─── VAD loader for FastAPI lifespan (WebRTC mode) ────────────────────────────

_webrtc_vad = None


async def load_vad() -> None:
    """
    Load Silero VAD for WebRTC mode.
    Call once from FastAPI lifespan startup — equivalent of prewarm() for CLI.
    Returns the VAD instance; also stored module-level for convenience.

    Usage in FastAPI lifespan:
        await stt_pipeline.load_vad()
        # later:
        vad = stt_pipeline.get_vad()
    """
    global _webrtc_vad
    try:
        from livekit.plugins import silero
        _webrtc_vad = silero.VAD.load(
            activation_threshold=0.4,
            min_silence_duration=0.3,
        )
        logger.info("Silero VAD loaded for WebRTC mode")
    except Exception as e:
        logger.error(f"Failed to load VAD: {e}")


def get_vad():
    """Return the module-level VAD instance loaded by load_vad()."""
    assert _webrtc_vad is not None, "call await load_vad() first"
    return _webrtc_vad


# ─── Standalone CLI entry point (unchanged) ───────────────────────────────────

async def _standalone_agent(ctx):
    from livekit.agents import JobContext
    init_pipeline(ctx)
    await ctx.connect()
    async for utterance in stt_stream():
        log_transcript(f"[STANDALONE] utterance → {utterance!r}")


if __name__ == "__main__":
    from livekit.agents import cli, WorkerOptions
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=_standalone_agent,
            prewarm_fnc=prewarm,
        )
    )