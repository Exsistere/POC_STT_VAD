import asyncio
import logging
import fractions
import numpy as np
import av
import asyncpg
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
import wave
import threading
from orchestrator import PipelineOrchestrator
from agent_audio_track import AgentAudioTrack
from routers import personas
from database import init_db_pool, close_db_pool, get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc-agent")

pcs = set()
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Voice AI server is starting up...")
    await init_db_pool()  # Connect to Postgres!
    yield
    logger.info("Server shutting down. Cleaning up connections...")
    await close_db_pool()  # Disconnect from Postgres cleanly
    coros = [pc.close() for pc in pcs]
    if coros:
        await asyncio.gather(*coros)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wire up the Persona CRUD APIs to your main app
app.include_router(personas.router)


# --- UPDATED WEBRTC REQUEST ---
class WebRTCRequest(BaseModel):
    sdp: str
    type: str
    persona_id: str  # <--- Replaced language_persona with the dynamic DB UUID


# --- DYNAMIC DATABASE PROMPT BUILDER ---
async def fetch_dynamic_prompt(persona_id: str, db: asyncpg.Connection) -> str:
    """Fetches the core prompt and appends the dynamic probing questions."""
    # 1. Get the base persona
    persona = await db.fetchrow("SELECT * FROM personas WHERE id = $1", persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found in database.")

    # 2. Get the probing questions
    questions = await db.fetch(
        "SELECT * FROM probing_questions WHERE persona_id = $1 ORDER BY sequence_order",
        persona_id,
    )

    # 3. Stitch them together for OpenAI
    final_prompt = persona["system_prompt"]

    if questions:
        final_prompt += "\n\nCRITICAL INSTRUCTIONS - PROBING QUESTIONS:\n"
        final_prompt += "During the call, you MUST naturally weave in the following questions based on their conditions:\n"
        for q in questions:
            final_prompt += f"- Question to ask: '{q['question_text']}' (When to ask: {q['trigger_condition']})\n"

    return final_prompt



@app.post("/offer")
async def offer(
    request: WebRTCRequest, db: asyncpg.Connection = Depends(get_db_connection)
):
    logger.info(
        f"Incoming call. Fetching DB config for Persona ID: {request.persona_id}"
    )

    # Inject the dynamic prompt generated from PostgreSQL
    system_prompt = await fetch_dynamic_prompt(request.persona_id, db)

    offer = RTCSessionDescription(sdp=request.sdp, type=request.type)
    pc = RTCPeerConnection()
    pcs.add(pc)

    recorder = MediaBlackhole()
    agent_audio = AgentAudioTrack()
    pc.addTrack(agent_audio)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            # agent_audio.close_diag()  # ← flush the diagnostic WAV
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        # if track.kind == "audio":
        #     recorder.addTrack(track)
        #     # Boot the agent with the custom DB prompt!
        #     agent_manager = VoiceAgentManager(
        #         system_prompt=system_prompt,
        #         agent_audio_track=agent_audio,
        #         db_connection=db,
        #         persona_id=request.persona_id,
        #     )
        #     asyncio.create_task(agent_manager.start(track))
        ### create a task to run pipeline orchestration here, passing the track and system_prompt
        if track.kind == "audio":
            recorder.addTrack(track)
            # Boot the agent with the custom DB prompt!
            agent_manager = PipelineOrchestrator(
                system_prompt=system_prompt,
                agent_audio_track=agent_audio,
                db_connection=db,
                persona_id=request.persona_id,
            )
            asyncio.create_task(agent_manager.start(track))
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
