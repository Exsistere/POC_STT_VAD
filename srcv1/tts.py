"""
tts_pipeline.py
───────────────
Dual-mode TTS pipeline — works identically from:

  • CLI / LiveKit entrypoint  (my_agent)         → LiveKitSink
  • WebRTC / PipelineOrchestrator                → WebRTCSink

Public API (unchanged from v1 — CLI main.py needs no edits):
    init_pipeline(ctx)              → TTSPipeline   (CLI mode)
    init_pipeline_webrtc(track)     → TTSPipeline   (WebRTC mode)
    start_pipeline()                → coroutine
    speak(text)                     → coroutine
    get_pipeline()                  → TTSPipeline

TTSPipeline instance API (same in both modes):
    await pipeline.start()
    await pipeline.speak(text)
    pipeline.interrupt()
    pipeline.reset_interrupt()
    pipeline.set_llm_cancel_fn(fn)
    pipeline.is_speaking            → bool
    await pipeline.aclose()
"""

import asyncio
import logging
import queue
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import os

import aiohttp
import numpy as np
from dotenv import load_dotenv
from livekit.plugins import cartesia

load_dotenv(override=True)

# ─── Shared aiohttp session (WebRTC / FastAPI mode only) ──────────────────────
# In CLI/LiveKit mode, Cartesia manages its own session via http_context.
# In FastAPI mode, we create one session at startup and pass it into Cartesia.
# Call create_http_session() in FastAPI lifespan startup,
# and close_http_session() in lifespan shutdown.

_http_session: aiohttp.ClientSession | None = None


async def create_http_session() -> None:
    """Call once from FastAPI lifespan startup."""
    global _http_session
    _http_session = aiohttp.ClientSession()
    _log(f"[{_ts()}] HTTP        aiohttp session created")


async def close_http_session() -> None:
    """Call once from FastAPI lifespan shutdown."""
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()
        _log(f"[{_ts()}] HTTP        aiohttp session closed")

# ─── Logging ──────────────────────────────────────────────────────────────────

logger = logging.getLogger("tts_pipeline")
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

CYAN  = "\033[36m"
GREEN = "\033[32m"
RESET = "\033[0m"

# ─── TTS config ───────────────────────────────────────────────────────────────

TTS_MODEL = "sonic-2"
TTS_VOICE = "248be419-c632-4f23-adf1-5324ed7dbf1d"   # British Lady — neutral
SD_QUEUE_MAXSIZE = 100


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    for line in message.split("\n"):
        sys.stdout.write(
            f"{CYAN}{timestamp}{RESET} {GREEN}INFO{RESET} tts_pipeline {line}\n"
        )
    sys.stdout.flush()


# ─── Output sink abstraction ──────────────────────────────────────────────────

class _OutputSink(ABC):
    """
    Strategy interface — encapsulates where synthesised PCM goes.
    TTSPipeline calls these methods; it never imports livekit.rtc or
    AgentAudioTrack directly.
    """

    @abstractmethod
    async def start(self, sample_rate: int, num_channels: int) -> None:
        """Called once after sample-rate detection. Set up output resources here."""

    @abstractmethod
    async def send_frame(self, frame) -> None:
        """Called per Cartesia AudioFrame during synthesis."""

    @abstractmethod
    async def send_silence(self, sample_rate: int, num_channels: int, duration_s: float = 0.04) -> None:
        """Called on barge-in CancelledError to flush/pad the output cleanly."""

    @abstractmethod
    def on_utterance_start(self) -> None:
        """Called before the first send_frame() of a new utterance."""

    @abstractmethod
    def on_utterance_done(self) -> None:
        """Called after the synthesis loop completes normally."""

    @abstractmethod
    def on_interrupt(self) -> None:
        """Called when interrupt() fires — drain stale audio immediately."""

    @abstractmethod
    async def aclose(self) -> None:
        """Tear down output resources."""


# ─── LiveKit sink (CLI mode) ──────────────────────────────────────────────────

class _LiveKitSink(_OutputSink):
    """
    Publishes audio to a LiveKit room and plays locally via sounddevice.
    Requires livekit.rtc and sounddevice — only imported in this class so the
    WebRTC path never needs those dependencies.
    """

    def __init__(self, ctx) -> None:
        # Late imports — only needed in CLI mode
        from livekit import rtc as _rtc
        import sounddevice as _sd
        self._rtc = _rtc
        self._sd  = _sd

        self._room         = ctx.room
        self._audio_source = None
        self._track        = None
        self._sd_queue: queue.Queue | None     = None
        self._sd_stream                        = None
        self._playback_task: asyncio.Task | None = None

    async def start(self, sample_rate: int, num_channels: int) -> None:
        rtc = self._rtc

        # AudioSource + LocalAudioTrack
        self._audio_source = rtc.AudioSource(sample_rate, num_channels)
        self._track = rtc.LocalAudioTrack.create_audio_track(
            "tts-output", self._audio_source
        )

        # Publish to room
        pub_opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self._room.local_participant.publish_track(self._track, pub_opts)
        _log(f"[{_ts()}] LIVEKIT     track published  room='{self._room.name}'")

        # Local sounddevice playback
        self._sd_queue = queue.Queue(maxsize=SD_QUEUE_MAXSIZE)
        self._sd_stream = self._open_speaker(sample_rate, num_channels)
        self._playback_task = asyncio.ensure_future(
            self._local_playback_loop(sample_rate, num_channels)
        )
        _log(f"[{_ts()}] LIVEKIT     local speaker started  sr={sample_rate}  ch={num_channels}")

    async def send_frame(self, frame) -> None:
        await self._audio_source.capture_frame(frame)

    async def send_silence(self, sample_rate: int, num_channels: int, duration_s: float = 0.04) -> None:
        if self._audio_source is None:
            return
        n = int(sample_rate * duration_s)
        silence = self._rtc.AudioFrame(
            data=bytes(n * num_channels * 2),
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=n,
        )
        try:
            await self._audio_source.capture_frame(silence)
        except Exception:
            pass

    def on_utterance_start(self) -> None:
        pass   # LiveKit room handles pacing natively

    def on_utterance_done(self) -> None:
        pass

    def on_interrupt(self) -> None:
        pass   # silence frame sent via send_silence() in _synthesise_and_play

    async def aclose(self) -> None:
        if self._playback_task:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
        if self._sd_stream:
            self._sd_stream.stop()
            self._sd_stream.close()

    # ── sounddevice helpers ───────────────────────────────────────────────

    def _open_speaker(self, sample_rate: int, num_channels: int):
        sd_q      = self._sd_queue
        remainder = [np.empty(0, dtype=np.int16)]

        def _callback(outdata, frames, time_info, status):
            needed = frames * num_channels
            buf    = remainder[0]
            while buf.size < needed:
                try:
                    chunk = sd_q.get_nowait()
                    buf   = np.concatenate((buf, chunk.flatten()))
                except queue.Empty:
                    break
            if buf.size >= needed:
                outdata[:]   = buf[:needed].reshape(frames, num_channels)
                remainder[0] = buf[needed:]
            elif buf.size > 0:
                out_flat           = outdata.reshape(-1)
                out_flat[:buf.size] = buf
                out_flat[buf.size:] = 0
                remainder[0]       = np.empty(0, dtype=np.int16)
            else:
                outdata.fill(0)

        import sounddevice as sd
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=num_channels,
            dtype="int16",
            callback=_callback,
            blocksize=int(sample_rate * 0.02),
        )
        stream.start()
        return stream

    async def _local_playback_loop(self, sample_rate: int, num_channels: int) -> None:
        assert self._track is not None
        audio_stream = self._rtc.AudioStream(
            track=self._track,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        try:
            async for event in audio_stream:
                frame = event.frame
                pcm   = np.frombuffer(frame.data, dtype=np.int16)
                try:
                    self._sd_queue.put_nowait(pcm)
                except queue.Full:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            await audio_stream.aclose()


# ─── WebRTC sink (PipelineOrchestrator mode) ──────────────────────────────────

class _WebRTCSink(_OutputSink):
    """
    Writes raw PCM bytes into an AgentAudioTrack for delivery via aiortc.
    No LiveKit dependencies.
    """

    def __init__(self, agent_audio_track) -> None:
        self._track = agent_audio_track

    async def start(self, sample_rate: int, num_channels: int) -> None:
        # AgentAudioTrack is already running — nothing to set up
        _log(f"[{_ts()}] WEBRTC      sink ready  sr={sample_rate}  ch={num_channels}")

    async def send_frame(self, frame) -> None:
        # frame.data is raw PCM16 bytes from Cartesia — exactly what AgentAudioTrack expects
        self._track.add_audio_bytes(frame.data)

    async def send_silence(self, sample_rate: int, num_channels: int, duration_s: float = 0.04) -> None:
        # AgentAudioTrack.clear() already handles stale audio on barge-in.
        # No need to inject a silence frame — clear() is called from on_interrupt().
        pass

    def on_utterance_start(self) -> None:
        self._track.mark_utterance_start()

    def on_utterance_done(self) -> None:
        self._track.mark_utterance_done()

    def on_interrupt(self) -> None:
        self._track.clear()   # drain stale frames immediately

    async def aclose(self) -> None:
        pass   # AgentAudioTrack lifetime is owned by main.py / orchestrator


# ─── TTSPipeline ──────────────────────────────────────────────────────────────

class TTSPipeline:
    """
    Mode-agnostic TTS pipeline.

    All speak/interrupt/barge-in logic lives here.
    Audio delivery is delegated entirely to the injected _OutputSink.

    Construct via class methods, not __init__ directly:
        TTSPipeline.for_livekit(ctx)
        TTSPipeline.for_webrtc(agent_audio_track)
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, sink: _OutputSink, http_session: "aiohttp.ClientSession | None" = None) -> None:
        self._sink = sink
        # In FastAPI/WebRTC mode, pass the shared aiohttp session so Cartesia
        # doesn't try to use LiveKit's http_context (only exists inside a job worker).
        # In CLI/LiveKit mode, http_session=None and Cartesia manages its own session.
        tts_kwargs: dict = dict(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            api_key=os.getenv("CARTESIA_API_KEY"),
        )
        if http_session is not None:
            tts_kwargs["http_session"] = http_session
        self._tts = cartesia.TTS(**tts_kwargs)

        self._sample_rate:  int | None = None
        self._num_channels: int | None = None

        self._speak_lock  = asyncio.Lock()
        self._speak_tasks: set[asyncio.Task] = set()
        self._tts_index   = 0

        self._is_speaking    = False
        self._interrupted    = False
        self._llm_cancel_fn: Optional[callable] = None

    @classmethod
    def for_livekit(cls, ctx) -> "TTSPipeline":
        """CLI / LiveKit entrypoint mode."""
        return cls(_LiveKitSink(ctx))


    @classmethod
    def for_webrtc(cls, agent_audio_track) -> "TTSPipeline":
        """WebRTC / PipelineOrchestrator mode. Uses module-level aiohttp session."""
        session = _http_session or aiohttp.ClientSession()
        return cls(_WebRTCSink(agent_audio_track), http_session=session)

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Detect sample rate from a test synthesis, then hand off to the sink
        to publish/open whatever output channel it manages.
        """
        sr, ch = await self._detect_sample_rate()
        self._sample_rate  = sr
        self._num_channels = ch
        _log(f"[{_ts()}] INIT       sample_rate={sr}  num_channels={ch}")
        await self._sink.start(sr, ch)

    async def speak(self, text: str) -> None:
        """
        Synthesise text and deliver audio through the active sink.
        Concurrent calls are serialised via _speak_lock.
        Barge-in cancels in-flight tasks immediately.
        """
        if self._interrupted:
            return
        task = asyncio.current_task()
        self._speak_tasks.add(task)
        try:
            if self._interrupted:
                return
            async with self._speak_lock:
                if self._interrupted:
                    return
                self._speak_task = task
                await self._synthesise_and_play(text)
        finally:
            self._speak_tasks.discard(task)

    def set_llm_cancel_fn(self, cancel_fn: callable) -> None:
        """Register the LLM cancellation hook for barge-in."""
        self._llm_cancel_fn = cancel_fn

    def interrupt(self) -> None:
        """
        Barge-in: cancel LLM generation + all in-flight TTS tasks + flush sink.
        Called by stt_pipeline's VAD speaking event.
        """
        if self._llm_cancel_fn:
            self._llm_cancel_fn()
        self._interrupted = True
        for t in list(self._speak_tasks):
            if not t.done():
                t.cancel()
        self._speak_tasks.clear()
        self._sink.on_interrupt()   # drain AgentAudioTrack queue OR no-op for LiveKit
        _log(f"[{_ts()}] INTERRUPTED  (barge-in detected)")

    def reset_interrupt(self) -> None:
        """Clear interrupt flag before processing a new utterance."""
        self._interrupted = False

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    async def aclose(self) -> None:
        self.interrupt()
        await self._sink.aclose()

    # ── Sample rate detection ─────────────────────────────────────────────────

    async def _detect_sample_rate(self) -> tuple[int, int]:
        try:
            stream = self._tts.stream()
            stream.push_text(" ")
            stream.end_input()
            async for synth in stream:
                frame = synth.frame
                _log(
                    f"[{_ts()}] TEST SYNTH  sample_rate={frame.sample_rate}  "
                    f"num_channels={frame.num_channels}  "
                    f"samples_per_channel={frame.samples_per_channel}"
                )
                return frame.sample_rate, frame.num_channels
        except Exception as exc:
            _log(f"[{_ts()}] TEST SYNTH  failed ({exc}) — falling back to 24000/1")
        return 24000, 1

    # ── Core synthesis ────────────────────────────────────────────────────────

    async def _synthesise_and_play(self, text: str) -> None:
        self._tts_index += 1
        idx           = self._tts_index
        start         = time.monotonic()
        first_frame_t = 0.0
        frames_sent   = 0

        _log(
            f"┌─── TTS #{idx} START [{_ts()}] {'─' * 32}\n"
            f"│  [{_ts()}] TEXT       {text!r}"
        )
        self._is_speaking = True
        self._sink.on_utterance_start()   # → mark_utterance_start() or no-op

        try:
            tts_stream = self._tts.stream()
            tts_stream.push_text(text)
            tts_stream.end_input()

            async for synthesised in tts_stream:
                if first_frame_t == 0.0:
                    first_frame_t = time.monotonic()
                    ttfa_ms       = (first_frame_t - start) * 1000
                    _log(f"│  [{_ts()}] TTFA       {ttfa_ms:.0f}ms (text → first audio frame)")

                await self._sink.send_frame(synthesised.frame)
                frames_sent += 1

            elapsed_ms = (time.monotonic() - start) * 1000
            _log(
                f"│  [{_ts()}] DONE       frames={frames_sent}  total={elapsed_ms:.0f}ms\n"
                f"└─── TTS #{idx} END   [{_ts()}] {'─' * 32}"
            )

        except asyncio.CancelledError:
            # Barge-in — flush/silence the output cleanly
            await self._sink.send_silence(
                self._sample_rate or 24000,
                self._num_channels or 1,
            )
            _log(
                f"│  [{_ts()}] CANCELLED  frames_before_interrupt={frames_sent}\n"
                f"└─── TTS #{idx} INTERRUPTED [{_ts()}] {'─' * 24}"
            )

        finally:
            self._is_speaking = False
            self._sink.on_utterance_done()   # → mark_utterance_done() or no-op


# ─── Module-level API (CLI main.py calls these — no changes needed there) ─────

_pipeline: TTSPipeline | None = None


def init_pipeline(ctx) -> TTSPipeline:
    """
    CLI / LiveKit mode initialiser.
    Called from my_agent() exactly as before.
    """
    global _pipeline
    _pipeline = TTSPipeline.for_livekit(ctx)
    return _pipeline


def init_pipeline_webrtc(agent_audio_track) -> TTSPipeline:
    """
    WebRTC mode initialiser.
    Called from PipelineOrchestrator — one instance per call.
    Does NOT touch the module-level _pipeline so CLI and WebRTC
    can coexist without stomping each other's state.
    """
    return TTSPipeline.for_webrtc(agent_audio_track)


async def start_pipeline() -> None:
    """Called from my_agent() after init_pipeline(). Unchanged."""
    assert _pipeline is not None, "call init_pipeline(ctx) first"
    await _pipeline.start()


async def speak(text: str) -> None:
    """Module-level speak — used by CLI main.py. Unchanged."""
    assert _pipeline is not None, "call init_pipeline(ctx) first"
    await _pipeline.speak(text)


def get_pipeline() -> TTSPipeline:
    """Return active module-level pipeline. Used by _generate_and_speak. Unchanged."""
    assert _pipeline is not None, "call init_pipeline(ctx) first"
    return _pipeline