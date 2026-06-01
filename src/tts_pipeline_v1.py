import asyncio
import logging
import queue
import sys
import time
from datetime import datetime
import os
from dotenv import load_dotenv
import numpy as np
import sounddevice as sd

from livekit import rtc
from livekit.agents import JobContext, inference
from livekit.plugins import cartesia

load_dotenv(override=True)

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

TTS_MODEL = "sonic-2"
TTS_VOICE = "248be419-c632-4f23-adf1-5324ed7dbf1d"  # British Lady — neutral

# How many PCM frames to buffer in the sounddevice playback queue before
# the callback starts draining. Small = lower latency, too small = glitches.
SD_QUEUE_MAXSIZE = 100


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _log(message: str) -> None:
    """Atomic structured log — matches STT pipeline visual style."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    for line in message.split("\n"):
        sys.stdout.write(
            f"{CYAN}{timestamp}{RESET} {GREEN}INFO{RESET} tts_pipeline {line}\n"
        )
    sys.stdout.flush()


# ─── TTSPipeline ──────────────────────────────────────────────────────────────

class TTSPipeline:
    """
    Manages TTS synthesis, room audio publishing, and local speaker playback.

    Two bugs fixed vs previous version:
      1. Sample rate — detected from a test synthesis at start() so AudioSource
         is always created with the exact rate the TTS plugin emits.
      2. Local playback — uses rtc.AudioStream(local_track) → sounddevice
         OutputStream with a callback queue. MediaDevices.open_output() only
         fires on remote track_subscribed events; local tracks never trigger it.

    Lifecycle:
        pipeline = TTSPipeline(ctx)
        await pipeline.start()          # test synth → detect SR → publish track
        await pipeline.speak("hello")   # synthesise + play
        pipeline.interrupt()            # barge-in cancel
        await pipeline.aclose()
    """

    def __init__(self, ctx: JobContext) -> None:
        self._ctx           = ctx
        self._room          = ctx.room
        self._tts           = cartesia.TTS(model=TTS_MODEL, voice=TTS_VOICE, api_key=os.getenv("CARTESIA_API_KEY"))

        # These are set in start() after sample-rate detection
        self._sample_rate: int | None   = None
        self._num_channels: int | None  = None
        self._audio_source: rtc.AudioSource | None   = None
        self._track: rtc.LocalAudioTrack | None      = None

        self._speak_task: asyncio.Task | None = None
        self._is_speaking = False
        self._speak_lock  = asyncio.Lock()   # serialise concurrent speak() calls
        self._tts_index   = 0

        # sounddevice playback state
        self._sd_queue: queue.Queue | None    = None
        self._sd_stream: sd.OutputStream | None = None
        self._playback_task: asyncio.Task | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        1. Run a silent test synthesis to detect the real sample rate + channels.
        2. Create AudioSource and LocalAudioTrack with the detected values.
        3. Publish the track to the room.
        4. Start local sounddevice playback by reading from the local track
           via rtc.AudioStream.
        """
        # ── Step 1: detect sample rate from TTS ───────────────────────────
        sr, ch = await self._detect_sample_rate()
        self._sample_rate  = sr
        self._num_channels = ch
        _log(
            f"[{_ts()}] INIT       detected sample_rate={sr}  num_channels={ch}"
        )

        # ── Step 2: create AudioSource + track with correct values ────────
        self._audio_source = rtc.AudioSource(sr, ch)
        self._track = rtc.LocalAudioTrack.create_audio_track(
            "tts-output", self._audio_source
        )

        # ── Step 3: publish track to room ─────────────────────────────────
        pub_opts = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE
        )
        await self._room.local_participant.publish_track(self._track, pub_opts)
        _log(f"[{_ts()}] TTS track published  room='{self._room.name}'")

        # ── Step 4: local speaker playback via sounddevice ────────────────
        # AudioStream can wrap a LocalAudioTrack directly — no track_subscribed
        # event needed. We read frames from it in a background task and push
        # them into a sounddevice OutputStream callback queue.
        self._sd_queue = queue.Queue(maxsize=SD_QUEUE_MAXSIZE)
        self._sd_stream = self._open_speaker(sr, ch)
        self._playback_task = asyncio.ensure_future(
            self._local_playback_loop()
        )
        _log(f"[{_ts()}] LOCAL PLAYBACK  speaker started  sr={sr}  ch={ch}")

    async def speak(self, text: str) -> None:
        """
        Public callable.

        Synthesises text → streams AudioFrames → room track + local speakers.
        Blocks until done or interrupted. Concurrent calls are serialised.

        Usage (from main.py):
            await tts_pipeline.speak("Hello!")
        """
        async with self._speak_lock:
            self._speak_task = asyncio.current_task()
            await self._synthesise_and_play(text)

    def interrupt(self) -> None:
        """
        Cancel TTS immediately (barge-in from VAD).
        Called by stt_pipeline when user starts speaking.
        """
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()
            _log(f"[{_ts()}] INTERRUPTED  (barge-in detected)")

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    async def aclose(self) -> None:
        self.interrupt()
        if self._playback_task:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
        if self._sd_stream:
            self._sd_stream.stop()
            self._sd_stream.close()

    # ── Sample rate detection ─────────────────────────────────────────────────

    async def _detect_sample_rate(self) -> tuple[int, int]:
        """
        Synthesise a single silent/short string and read sample_rate +
        num_channels off the first AudioFrame that comes back.
        Falls back to 24000/1 (Cartesia default) if synthesis fails.
        """
        try:
            stream = self._tts.stream()
            stream.push_text(" ")   # minimal text — just enough to get a frame
            stream.end_input()
            async for synth in stream:
                frame = synth.frame
                _log(
                    f"[{_ts()}] TEST SYNTH  frame: "
                    f"sample_rate={frame.sample_rate}  "
                    f"num_channels={frame.num_channels}  "
                    f"samples_per_channel={frame.samples_per_channel}"
                )
                return frame.sample_rate, frame.num_channels
        except Exception as exc:
            _log(f"[{_ts()}] TEST SYNTH  failed ({exc}) — falling back to 24000/1")
        return 24000, 1

    # ── sounddevice speaker setup ─────────────────────────────────────────────

    def _open_speaker(self, sample_rate: int, num_channels: int) -> sd.OutputStream:
        """
        Open a sounddevice OutputStream with a callback that drains
        self._sd_queue. Silence is played when the queue is empty.

        Uses a remainder buffer so that AudioFrames of any size (e.g. 240
        samples at 10ms) are correctly assembled into whatever blocksize
        sounddevice requests (e.g. 480 samples at 20ms), avoiding the
        "could not broadcast input array from shape (X,) into shape (Y,)"
        ValueError when the two sizes don't match.
        """
        sd_q = self._sd_queue
        # Mutable container so the closure can update the buffer reference
        remainder: list[np.ndarray] = [np.empty(0, dtype=np.int16)]

        def _callback(outdata: np.ndarray, frames: int, time_info, status):
            needed = frames * num_channels  # total int16 samples required
            buf = remainder[0]

            # Pull chunks from the queue until we have enough samples or run dry
            while buf.size < needed:
                try:
                    chunk = sd_q.get_nowait()
                    buf = np.concatenate((buf, chunk.flatten()))
                except queue.Empty:
                    break

            if buf.size >= needed:
                outdata[:] = buf[:needed].reshape(frames, num_channels)
                remainder[0] = buf[needed:]
            elif buf.size > 0:
                # Partial data — fill what we have, silence the rest
                out_flat = outdata.reshape(-1)
                out_flat[:buf.size] = buf
                out_flat[buf.size:] = 0
                remainder[0] = np.empty(0, dtype=np.int16)
            else:
                outdata.fill(0)   # silence when nothing queued

        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=num_channels,
            dtype="int16",
            callback=_callback,
            blocksize=int(sample_rate * 0.02),  # 20ms blocks
        )
        stream.start()
        return stream

    # ── Local playback loop ───────────────────────────────────────────────────

    async def _local_playback_loop(self) -> None:
        """
        Reads AudioFrames from the local track via rtc.AudioStream and
        pushes raw int16 PCM data into the sounddevice callback queue.

        rtc.AudioStream accepts a LocalAudioTrack directly — no need for
        a remote participant or track_subscribed event.
        """
        assert self._track is not None
        assert self._sample_rate is not None
        assert self._num_channels is not None

        audio_stream = rtc.AudioStream(
            track=self._track,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
        )
        try:
            async for event in audio_stream:
                frame: rtc.AudioFrame = event.frame
                pcm = np.frombuffer(frame.data, dtype=np.int16)
                try:
                    self._sd_queue.put_nowait(pcm)
                except queue.Full:
                    pass   # drop oldest-equivalent: queue drains faster than fill
        except asyncio.CancelledError:
            pass
        finally:
            await audio_stream.aclose()

    # ── Core synthesis ────────────────────────────────────────────────────────

    async def _synthesise_and_play(self, text: str) -> None:
        self._tts_index += 1
        idx   = self._tts_index
        start = time.monotonic()
        first_frame_t = 0.0
        frames_sent   = 0

        _log(
            f"┌─── TTS #{idx} START [{_ts()}] {'─' * 32}\n"
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
                    _log(
                        f"│  [{_ts()}] TTFA       {ttfa_ms:.0f}ms "
                        f"(text → first audio frame)"
                    )

                await self._audio_source.capture_frame(synthesised.frame)
                frames_sent += 1

            elapsed_ms = (time.monotonic() - start) * 1000
            _log(
                f"│  [{_ts()}] DONE       frames={frames_sent}  "
                f"total={elapsed_ms:.0f}ms\n"
                f"└─── TTS #{idx} END   [{_ts()}] {'─' * 32}"
            )

        except asyncio.CancelledError:
            # Barge-in: push a short silence to avoid audio glitch
            if self._audio_source is not None:
                sr = self._sample_rate or 24000
                ch = self._num_channels or 1
                n  = int(sr * 0.04)   # 40ms
                silence = rtc.AudioFrame(
                    data=bytes(n * ch * 2),
                    sample_rate=sr,
                    num_channels=ch,
                    samples_per_channel=n,
                )
                try:
                    await self._audio_source.capture_frame(silence)
                except Exception:
                    pass
            _log(
                f"│  [{_ts()}] CANCELLED  frames_before_interrupt={frames_sent}\n"
                f"└─── TTS #{idx} INTERRUPTED [{_ts()}] {'─' * 24}"
            )

        finally:
            self._is_speaking = False


# ─── Module-level pipeline (set by init_pipeline in main.py) ──────────────────

_pipeline: TTSPipeline | None = None


def init_pipeline(ctx: JobContext) -> TTSPipeline:
    """Create and store the module-level TTSPipeline. Called once from main.py."""
    global _pipeline
    _pipeline = TTSPipeline(ctx)
    return _pipeline


async def start_pipeline() -> None:
    """Detect sample rate, publish track, start speaker. Called after ctx.connect()."""
    assert _pipeline is not None, "call init_pipeline(ctx) first"
    await _pipeline.start()


async def speak(text: str) -> None:
    """
    Public callable — takes text, plays audio through room + speakers.

    Imported by main.py:
        from tts_pipeline import speak
        await speak("Hello!")
    """
    assert _pipeline is not None, "call init_pipeline(ctx) first"
    await _pipeline.speak(text)


def get_pipeline() -> TTSPipeline:
    """Return active pipeline so stt_pipeline can call .interrupt()."""
    assert _pipeline is not None, "call init_pipeline(ctx) first"
    return _pipeline
