"""
test_main.py
────────────
Two independent WebRTC pipeline tests:

  GET /          → TTS sink test page   (browser recvonly, server speaks)
  POST /offer    → TTS WebRTC handshake

  GET /stt       → STT source test page (browser sendonly mic, server transcribes)
  POST /stt-offer → STT WebRTC handshake
  GET /stt-transcript/{conn_id} → SSE stream of utterances

Run:
    uvicorn test_main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
import asyncio
import logging
import json
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription
from livekit.plugins import silero

from agent_audio_track import AgentAudioTrack
from tts import init_pipeline_webrtc as tts_init_webrtc
from tts import create_http_session, close_http_session
from stt import init_pipeline_webrtc as stt_init_webrtc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test-main")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_http_session()          # Cartesia needs aiohttp outside LiveKit
    app.state.vad = silero.VAD.load(     # Silero loaded once, shared across calls
        activation_threshold=0.4,
        min_silence_duration=0.2,
    )
    logger.info("Startup complete — aiohttp session + VAD ready")
    yield
    await close_http_session()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs: set[RTCPeerConnection] = set()

# conn_id → asyncio.Queue[str | None]  (STT test only)
_transcript_queues: dict[str, asyncio.Queue] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# TTS TEST  ─  GET /   POST /offer
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def tts_index():
    with open("tts_test.html") as f:
        return f.read()


class OfferRequest(BaseModel):
    sdp:  str
    type: str


@app.post("/offer")
async def tts_offer(request: OfferRequest):
    pc = RTCPeerConnection()
    pcs.add(pc)

    agent_audio = AgentAudioTrack()
    pc.addTrack(agent_audio)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"[TTS] connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"[TTS] ICE state: {pc.iceConnectionState}")

    @pc.on("track")
    def on_track(track):
        # recvonly test — browser sends nothing, but handle gracefully if it does
        logger.info(f"[TTS] unexpected inbound track: {track.kind}")

    await pc.setRemoteDescription(RTCSessionDescription(sdp=request.sdp, type=request.type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Boot TTS after handshake so ICE has time to connect
    asyncio.create_task(_tts_speak_test(agent_audio))

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


TTS_TEST_SENTENCES = [
    "Hello! The WebRTC audio sink is working correctly.",
    "You are hearing this through Cartesia, the AgentAudioTrack, and your browser.",
    "If audio sounds clean with no glitches, the pipeline is ready.",
]


async def _tts_speak_test(agent_audio: AgentAudioTrack) -> None:
    await asyncio.sleep(1.5)          # let ICE complete before pushing audio
    logger.info("[TTS] starting test synthesis")

    # init_pipeline_webrtc(agent_audio_track) → TTSPipeline
    pipeline = tts_init_webrtc(agent_audio)

    # pipeline.start() detects Cartesia sample rate and calls sink.start()
    await pipeline.start()

    for sentence in TTS_TEST_SENTENCES:
        if pipeline._interrupted:
            break
        logger.info(f"[TTS] speaking: {sentence!r}")
        await pipeline.speak(sentence)
        await asyncio.sleep(0.3)

    logger.info("[TTS] test complete")


# ═══════════════════════════════════════════════════════════════════════════════
# STT TEST  ─  GET /stt   POST /stt-offer   GET /stt-transcript/{conn_id}
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/stt", response_class=HTMLResponse)
async def stt_index():
    with open("stt_test.html") as f:
        return f.read()


class SttOfferRequest(BaseModel):
    sdp:     str
    type:    str
    conn_id: str   # browser-generated UUID to route SSE to the right queue


@app.post("/stt-offer")
async def stt_offer(request: SttOfferRequest):
    conn_id = request.conn_id

    # Register transcript queue before any background task can push to it
    q: asyncio.Queue[str | None] = asyncio.Queue()
    _transcript_queues[conn_id] = q

    pc = RTCPeerConnection()
    pcs.add(pc)

    # Future resolves when on_track fires with the browser's mic track
    loop = asyncio.get_running_loop()
    mic_track_future: asyncio.Future = loop.create_future()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"[STT] [{conn_id[:8]}] connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)
            _end_stt_session(conn_id)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"[STT] [{conn_id[:8]}] ICE state: {pc.iceConnectionState}")

    @pc.on("track")
    def on_track(track):
        logger.info(f"[STT] [{conn_id[:8]}] received {track.kind} track")
        if track.kind == "audio" and not mic_track_future.done():
            mic_track_future.set_result(track)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=request.sdp, type=request.type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    asyncio.create_task(_stt_run(conn_id, mic_track_future))

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


def _end_stt_session(conn_id: str) -> None:
    """Push sentinel to end SSE stream and remove queue."""
    if conn_id in _transcript_queues:
        _transcript_queues[conn_id].put_nowait(None)
        del _transcript_queues[conn_id]


async def _stt_run(conn_id: str, mic_track_future: asyncio.Future) -> None:
    """
    Wait for the browser's mic track then run STTPipeline for this connection.
    Utterances are forwarded to the per-connection SSE queue.
    """
    try:
        mic_track = await asyncio.wait_for(mic_track_future, timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning(f"[STT] [{conn_id[:8]}] timed out waiting for mic track")
        _end_stt_session(conn_id)
        return

    logger.info(f"[STT] [{conn_id[:8]}] mic track received — starting STT pipeline")

    # init_pipeline_webrtc(track, vad, interrupt_fn) → STTPipeline
    pipeline = stt_init_webrtc(
        track=mic_track,
        vad=app.state.vad,
        interrupt_fn=None,   # no TTS to interrupt in this isolated STT test
    )

    try:
        # pipeline.stream() → AsyncGenerator[str, None]
        async for utterance in pipeline.stream():
            logger.info(f"[STT] [{conn_id[:8]}] utterance: {utterance!r}")
            if conn_id in _transcript_queues:
                _transcript_queues[conn_id].put_nowait(
                    json.dumps({"utterance": utterance})
                )
    except Exception as exc:
        logger.error(f"[STT] [{conn_id[:8]}] pipeline error: {exc}")
    finally:
        # Do NOT call pipeline.aclose() here — stream()'s own finally block
        # already cancels the source task when the async for exits.
        # Calling aclose() concurrently causes "asynchronous generator is already running".
        _end_stt_session(conn_id)
        logger.info(f"[STT] [{conn_id[:8]}] pipeline closed")


@app.get("/stt-transcript/{conn_id}")
async def stt_transcript_sse(conn_id: str):
    """
    SSE endpoint — browser subscribes after /stt-offer returns.
    Streams JSON events:  {"utterance": "..."}  or  {"status": "..."}
    """
    async def event_generator():
        # Queue may not be registered yet if GET arrives before POST completes
        for _ in range(20):
            if conn_id in _transcript_queues:
                break
            await asyncio.sleep(0.1)

        q = _transcript_queues.get(conn_id)
        if q is None:
            yield f"data: {json.dumps({'status': 'no_session'})}\n\n"
            return

        yield f"data: {json.dumps({'status': 'connected'})}\n\n"

        while True:
            item = await q.get()
            if item is None:
                yield f"data: {json.dumps({'status': 'closed'})}\n\n"
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )