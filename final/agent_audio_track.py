import asyncio
import fractions
import logging
import numpy as np
import av
from aiortc import MediaStreamTrack

logger = logging.getLogger("agent-audio-track")

SAMPLE_RATE       = 24000
SAMPLES_PER_FRAME = int(SAMPLE_RATE * 0.02)  # 480 samples @ 20ms
BYTES_PER_FRAME   = SAMPLES_PER_FRAME * 2    # int16 = 2 bytes per sample


class AgentAudioTrack(MediaStreamTrack):
    """
    WebRTC audio output track for the STT/LLM/TTS pipeline.

    Responsibilities
    ────────────────
    • Accepts raw PCM16 mono 24 kHz bytes from TTSPipeline via add_audio_bytes().
    • Chunks those bytes into 20ms frames and queues them.
    • Emits frames to the browser at exactly 50 frames/sec via wall-clock pacing.
    • Emits silence when nothing is queued.
    • Supports mid-utterance queue flush for barge-in (clear()).

    State machine (driven by TTSPipeline)
    ──────────────────────────────────────
    • mark_utterance_start() — called before first add_audio_bytes() of a response.
                               recv() will block on queue while _speaking=True,
                               preventing silence gaps during Cartesia TTFA delay.
    • add_audio_bytes(bytes) — called per Cartesia frame (frame.data).
    • mark_utterance_done()  — called after synthesis loop ends normally.
                               Flushes partial buffer, clears _speaking so recv()
                               drains remaining frames then falls to silence.
    • clear()                — called on barge-in; drains queue + buffer instantly.

    Pacing
    ──────
    Wall-clock anchored: recv() sleeps until the exact target time for each
    frame based on pts. This guarantees 50 frames/sec regardless of event loop
    scheduling jitter, which prevents timestamp drift that causes browser audio
    glitches over long calls.
    """

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue    = asyncio.Queue()
        self._buffer   = bytearray()
        self._speaking = True   # True between mark_utterance_start and mark_utterance_done

        self._sample_rate = SAMPLE_RATE
        self._time_base   = fractions.Fraction(1, SAMPLE_RATE)
        self._pts         = 0
        self._start_time: float | None = None   # wall-clock anchor, set on first recv()

    # ── Public API called by TTSPipeline ──────────────────────────────────────

    def mark_utterance_start(self) -> None:
        """
        Signal that audio bytes are about to arrive.
        recv() blocks on the queue while _speaking=True, absorbing the
        Cartesia TTFA delay without emitting silence clicks.
        """
        self._speaking = True
        logger.debug("[AAT] utterance start")

    def add_audio_bytes(self, audio_bytes: bytes) -> None:
        """
        Accept raw PCM16 bytes from Cartesia (frame.data) and enqueue 20ms chunks.
        Partial frames accumulate in _buffer until a full frame is available.
        """
        self._buffer.extend(audio_bytes)
        while len(self._buffer) >= BYTES_PER_FRAME:
            chunk = bytes(self._buffer[:BYTES_PER_FRAME])
            del self._buffer[:BYTES_PER_FRAME]
            self._queue.put_nowait(chunk)

    def mark_utterance_done(self) -> None:
        """
        Signal end of synthesis.
        Flushes any partial frame (zero-padded) so the last word isn't clipped,
        then clears _speaking so recv() drains remaining frames then silences.
        """
        if self._buffer:
            remaining = bytes(self._buffer)
            padded    = remaining + b'\x00' * (BYTES_PER_FRAME - len(remaining))
            self._queue.put_nowait(padded)
            self._buffer.clear()

        self._speaking = False
        logger.debug(f"[AAT] utterance done — {self._queue.qsize()} frames remain in queue")

    def clear(self) -> None:
        """
        Barge-in flush. Drains the queue and partial buffer immediately so the
        next utterance starts clean without playing stale audio.
        Called by TTSPipeline.interrupt() which is triggered by VAD speaking event.
        """
        drained = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        self._buffer.clear()
        self._speaking = False
        logger.debug(f"[AAT] barge-in clear — drained {drained} frames")

    # ── WebRTC clock (called by aiortc at 20ms intervals) ─────────────────────

    async def recv(self) -> av.AudioFrame:
        """
        Called by aiortc at a fixed 20ms cadence.

        Pacing: wall-clock anchored sleep so each frame fires at exactly
        pts/sample_rate seconds from the first recv() call. Prevents timestamp
        drift over long calls.

        Audio selection:
        • _speaking=True  → await queue.get() — blocks until Cartesia delivers,
                            no silence during TTFA delay or between synthesis chunks.
        • _speaking=False, queue non-empty → get_nowait() — drain post-done frames.
        • _speaking=False, queue empty     → silence frame.
        """
        loop = asyncio.get_event_loop()

        # Anchor the wall clock on first call
        if self._start_time is None:
            self._start_time = loop.time()

        # Sleep until this frame's target delivery time
        target = self._start_time + (self._pts / self._sample_rate)
        delay  = target - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)

        # Select audio data
        audio_data = np.zeros(SAMPLES_PER_FRAME, dtype=np.int16)  # default: silence

        if self._speaking:
            # Synthesis in progress — block until a frame arrives.
            # This handles the Cartesia TTFA gap cleanly: no silence click
            # between mark_utterance_start() and the first actual audio frame.
            try:
                chunk      = await self._queue.get()
                audio_data = np.frombuffer(chunk, dtype=np.int16)
            except Exception as exc:
                logger.warning(f"[AAT] recv error while speaking: {exc}")

        elif not self._queue.empty():
            # Not speaking but frames remain (draining after mark_utterance_done)
            try:
                chunk      = self._queue.get_nowait()
                audio_data = np.frombuffer(chunk, dtype=np.int16)
            except asyncio.QueueEmpty:
                pass  # race between empty-check and get — emit silence this tick

        # else: idle / between utterances — silence already set above

        # Build av.AudioFrame
        frame             = av.AudioFrame.from_ndarray(
            audio_data.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = self._sample_rate
        frame.pts         = self._pts
        frame.time_base   = self._time_base

        self._pts += SAMPLES_PER_FRAME
        return frame