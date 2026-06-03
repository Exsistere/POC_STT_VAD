import os
import uuid
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional

from database import get_db_connection
from services.kb import process_document_pipeline

logger = logging.getLogger("persona-api")
router = APIRouter(prefix="/api/v1/personas", tags=["Personas"])


# --- PYDANTIC MODELS ---
class ProbingQuestionCreate(BaseModel):
    question_text: str
    trigger_condition: Optional[str] = None
    sequence_order: int = 0


class PersonaCreate(BaseModel):
    tenant_id: str  # To separate your clients
    name: str
    system_prompt: str
    voice_id: str
    probing_questions: List[ProbingQuestionCreate] = []


class PersonaResponse(BaseModel):
    id: str
    name: str
    status: str = "success"


# --- 1. CREATE PERSONA API ---
@router.post("/", response_model=PersonaResponse)
async def create_persona(persona: PersonaCreate, db=Depends(get_db_connection)):
    """Creates a new AI agent persona and attaches its probing questions."""
    try:
        # Start DB Transaction
        async with db.transaction():
            # 1. Insert Persona
            persona_query = """
                INSERT INTO personas (tenant_id, name, system_prompt, voice_id)
                VALUES ($1, $2, $3, $4) RETURNING id;
            """
            persona_id = await db.fetchval(
                persona_query,
                persona.tenant_id,
                persona.name,
                persona.system_prompt,
                persona.voice_id,
            )

            # 2. Insert Probing Questions
            if persona.probing_questions:
                questions_data = [
                    (persona_id, q.question_text, q.trigger_condition, q.sequence_order)
                    for q in persona.probing_questions
                ]
                await db.executemany(
                    """
                    INSERT INTO probing_questions (persona_id, question_text, trigger_condition, sequence_order)
                    VALUES ($1, $2, $3, $4);
                """,
                    questions_data,
                )

        return {"id": str(persona_id), "name": persona.name}

    except Exception as e:
        logger.error(f"Failed to create persona: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred.")


# --- 2. UPLOAD KNOWLEDGE BASE API ---
@router.post("/{persona_id}/knowledge-base")
async def upload_knowledge_base(
    persona_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db=Depends(get_db_connection),
):
    """Accepts a document and processes it in the background using Azure Doc Intel."""

    # 1. Read file into memory (or save to temp storage if huge)
    file_bytes = await file.read()

    # 2. Create tracking record in DB so the UI can show "Processing..."
    doc_id = await db.fetchval(
        """
        INSERT INTO knowledge_base_docs (persona_id, filename, upload_status)
        VALUES ($1::uuid, $2, 'processing') RETURNING id;
    """,
        persona_id,
        file.filename,
    )

    # 3. Hand off the heavy lifting to a background task so the API returns instantly
    background_tasks.add_task(process_document_pipeline, doc_id, persona_id, file_bytes)

    return {
        "message": "Document upload accepted. Processing started.",
        "document_id": str(doc_id),
        "status": "processing",
    }


# --- 3. GET PERSONA CONFIGURATION (For WebRTC Engine) ---
@router.get("/{persona_id}/config")
async def get_persona_config(persona_id: str, db=Depends(get_db_connection)):
    """Called by your agent_manager.py right before the call starts to load the brain."""
    persona = await db.fetchrow(
        "SELECT * FROM personas WHERE id = $1::uuid", persona_id
    )
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    questions = await db.fetch(
        "SELECT * FROM probing_questions WHERE persona_id = $1::uuid ORDER BY sequence_order",
        persona_id,
    )

    return {
        "system_prompt": persona["system_prompt"],
        "voice_id": persona["voice_id"],
        "probing_questions": [dict(q) for q in questions],
    }


@router.get("/", response_model=List[dict])
async def get_all_personas(tenant_id: str, db=Depends(get_db_connection)):
    """Fetches all agents for the dashboard grid."""
    try:
        records = await db.fetch(
            """
            SELECT id, name, voice_id, system_prompt 
            FROM personas 
            WHERE tenant_id = $1
            ORDER BY created_at DESC
        """,
            tenant_id,
        )

        # Manually construct the dict to cast the UUID to a string
        result = []
        for record in records:
            result.append(
                {
                    "id": str(record["id"]),
                    "name": record["name"],
                    "voice_id": record["voice_id"],
                    "system_prompt": record["system_prompt"],
                }
            )
        return result

    except Exception as e:
        logger.error(f"Failed to fetch personas: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred.")

@router.get("/{persona_id}/logs")
async def get_persona_logs(persona_id: str, db = Depends(get_db_connection)):
    """Fetches historical call logs for the Next.js Analytics Dashboard."""
    
    # 1. Fetch Master Calls
    calls = await db.fetch("""
        SELECT id, caller_id, created_at, duration_seconds 
        FROM calls WHERE persona_id = $1::uuid ORDER BY created_at DESC LIMIT 50;
    """, persona_id)

    log_response = []
    
    for call in calls:
        call_id = call["id"]
        
        # 2. Fetch Transcript for this call
        transcripts = await db.fetch("""
            SELECT speaker, text, created_at FROM call_transcripts 
            WHERE call_id = $1 ORDER BY created_at ASC;
        """, call_id)
        
        # 3. Fetch Tool Events
        events = await db.fetch("""
            SELECT event_type, event_data, created_at FROM call_events 
            WHERE call_id = $1 ORDER BY created_at ASC;
        """, call_id)

        # Format datetime objects to strings
        log_response.append({
            "id": str(call_id),
            "date": call["created_at"].strftime("%b %d, %Y - %I:%M %p"),
            "duration": f"{call['duration_seconds']}s",
            "caller_id": call["caller_id"],
            "probing_results": [], # Placeholder for advanced probing extraction
            "tool_events": [{
                "time": e["created_at"].strftime("%I:%M %p"),
                "tool_name": e["event_type"],
                "status": "success",
                "details": json.loads(e["event_data"])
            } for e in events],
            "transcript": [{
                "speaker": t["speaker"],
                "text": t["text"],
                "time": t["created_at"].strftime("%I:%M:%S")
            } for t in transcripts]
        })

    return log_response

