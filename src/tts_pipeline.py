import asyncio
import logging
import sys
import time
from datetime import datetime

from livekit import rtc
from livekit.agents import JobContext, inference

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

# ─── ANSI colours (matching STT pipeline style) ───────────────────────────────

CYAN  = "\033[36m"
GREEN = "\033[32m"
RESET = "\033[0m"

# ─── TTS config ───────────────────────────────────────────────────────────────

TTS_MODEL = "cartesia/sonic-2"
TTS_VOICE = "248be419-c632-4f23-adf1-5324ed7dbf1d"  # British Lady — neutral
TTS_SR    = 44100   # sample rate emitted by cartesia/sonic-2
TTS_CH    = 1       # mono


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _log(message: str) -> None:
    """
    Atomic structured log to stdout.
    Matches the visual style of the STT pipeline log_transcript().
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    for line in message.split("\n"):
        sys.stdout.write(f"{CYAN}{timestamp}{RESET} {GREEN}INFO{RESET} tts_pipeline {line}\n")
    sys.stdout.flush()


# ─── TTSPipeline ──────────────────────────────────────────────────────────────

class TTSPipeline:
    """
    Manages a single TTS audio track published to a LiveKit room.

    Lifecycle:
        pipeline = TTSPipeline(ctx)
        await pipeline.start()          # publishes track, sets up local playback
        await pipeline.speak("hello")   # synthesise and play
        pipeline.interrupt()            # stop mid-speech (VAD barge-in)
        await pipeline.aclose()         # cleanup
    """

    def __init__(self, ctx: JobContext) -> None:
        self._ctx          = ctx
        self._room         = ctx.room
        self._tts          = inference.TTS(model=TTS_MODEL, voice=TTS_VOICE)
        self._audio_source = rtc.AudioSource(TTS_SR, TTS_CH)
        self._track        = rtc.LocalAudioTrack.create_audio_track(
                                 "tts-output", self._audio_source
                             )
        self._speak_task: asyncio.Task | None = None
        self._is_speaking  = False
        # Serialise concurrent speak() calls — one utterance at a time
        self._speak_lock   = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Publish the TTS audio track to the room and start local speaker playback.
        Must be called after ctx.connect().
        """
        pub_opts = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE
        )
        await self._room.local_participant.publish_track(self._track, pub_opts)
        _log(f"[{_ts()}] TTS track published  room='{self._room.name}'")

        # ── Local speaker playback via MediaDevices ────────────────────────
        # Subscribe to the room's audio tracks (including our own published
        # track when reflected back, or any other participant's audio) and
        # play through system speakers — no frontend required.
        try:
            devices = rtc.MediaDevices()
            player  = self._devices_player = devices.open_output()

            @self._room.on("track_subscribed")
            def _on_track_subscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ) -> None:
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    player.add_track(track)
                    _log(
                        f"[{_ts()}] LOCAL PLAYBACK  subscribed to audio from "
                        f"participant='{participant.identity}'"
                    )

            await player.start()
            _log(f"[{_ts()}] LOCAL PLAYBACK  speaker started")

        except Exception as exc:
            _log(
                f"[{_ts()}] WARNING  local speaker unavailable ({exc})\n"
                f"[{_ts()}]          TTS audio is still published to the room."
            )
            self._devices_player = None

    async def speak(self, text: str) -> None:
        """
        Callable entry point.

        Synthesises `text` via LiveKit Inference (Cartesia sonic-2),
        streams AudioFrames into the published room track, and plays
        them through the local speaker.

        Blocks until playback completes or is interrupted.
        Concurrent calls are serialised by _speak_lock.

        Usage (from main.py):
            await tts.speak("Hello, world!")
        """
        async with self._speak_lock:
            self._speak_task = asyncio.current_task()
            await self._synthesise_and_play(text)

    def interrupt(self) -> None:
        """
        Stop TTS immediately.
        Called by the STT pipeline the moment VAD detects user speech.
        """
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()
            _log(f"[{_ts()}] INTERRUPTED  (barge-in detected)")

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    async def aclose(self) -> None:
        self.interrupt()
        if getattr(self, "_devices_player", None):
            try:
                await self._devices_player.aclose()
            except Exception:
                pass

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _synthesise_and_play(self, text: str) -> None:
        start           = time.monotonic()
        first_frame_t   = 0.0
        frames_sent     = 0
        _tts_index      = getattr(self, "_tts_index", 0) + 1
        self._tts_index = _tts_index

        _log(
            f"┌─── TTS #{_tts_index} START [{_ts()}] {'─' * 32}\n"
            f"│  [{_ts()}] TEXT       {text!r}"
        )
        self._is_speaking = True

        try:
            tts_stream = self._tts.stream()
            tts_stream.push_text(text)
            tts_stream.end_input()

            async for synthesised in tts_stream:
                if first_frame_t == 0.0:
                    first_frame_t = time.monotonic()
                    ttfa_ms = (first_frame_t - start) * 1000
                    _log(f"│  [{_ts()}] TTFA       {ttfa_ms:.0f}ms  (text → first audio frame)")

                await self._audio_source.capture_frame(synthesised.frame)
                frames_sent += 1

            elapsed_ms = (time.monotonic() - start) * 1000
            _log(
                f"│  [{_ts()}] DONE       frames={frames_sent}  total={elapsed_ms:.0f}ms\n"
                f"└─── TTS #{_tts_index} END   [{_ts()}] {'─' * 32}"
            )

        except asyncio.CancelledError:
            # Barge-in: drain with a short silence to avoid audio glitch
            silence = rtc.AudioFrame(
                data=bytes(int(TTS_SR * 0.04) * TTS_CH * 2),
                sample_rate=TTS_SR,
                num_channels=TTS_CH,
                samples_per_channel=int(TTS_SR * 0.04),
            )
            try:
                await self._audio_source.capture_frame(silence)
            except Exception:
                pass
            _log(
                f"│  [{_ts()}] CANCELLED  frames_sent_before_interrupt={frames_sent}\n"
                f"└─── TTS #{_tts_index} INTERRUPTED [{_ts()}] {'─' * 24}"
            )

        finally:
            self._is_speaking = False


# ─── Module-level pipeline instance (set by main.py after ctx is available) ───

_pipeline: TTSPipeline | None = None


def init_pipeline(ctx: JobContext) -> TTSPipeline:
    """
    Create and store the module-level TTSPipeline.
    Called once from main.py inside the agent entrypoint.

    Returns the pipeline so main.py can also hold a reference.
    """
    global _pipeline
    _pipeline = TTSPipeline(ctx)
    return _pipeline


async def start_pipeline() -> None:
    """Publish the TTS track. Called after ctx.connect() in main.py."""
    assert _pipeline is not None, "call init_pipeline(ctx) first"
    await _pipeline.start()


async def speak(text: str) -> None:
    """
    Public callable — takes text, plays audio through the room + speakers.

    This is the function imported by main.py:
        from tts_pipeline import speak
        await speak("Hello!")
    """
    assert _pipeline is not None, "call init_pipeline(ctx) first"
    await _pipeline.speak(text)


def get_pipeline() -> TTSPipeline:
    """Return the active pipeline (used by STT pipeline to call interrupt())."""
    assert _pipeline is not None, "call init_pipeline(ctx) first"
    return _pipeline
