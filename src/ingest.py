"""
ingest.py — Standalone document ingestion script.

Usage:
    python ingest.py --file path/to/doc.pdf --persona-id <uuid>

    # Optional overrides:
    python ingest.py --file doc.pdf --persona-id <uuid> --doc-id <uuid> --chunk-size 800

Pipeline:
    PDF/file  →  Azure Document Intelligence  →  chunk  →  Azure embed  →  pgvector
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path

import asyncpg
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


# ─── Step 1: Extract text via Azure Document Intelligence ─────────────────────

def _extract_text_sync(file_bytes: bytes) -> str:
    """Runs the blocking Azure SDK call in a thread (called via asyncio.to_thread)."""
    endpoint = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
    key      = os.getenv("AZURE_DOC_INTEL_KEY")

    if not endpoint or not key:
        raise EnvironmentError(
            "AZURE_DOC_INTEL_ENDPOINT and AZURE_DOC_INTEL_KEY must be set in your .env"
        )

    client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))
    poller  = client.begin_analyze_document("prebuilt-layout", document=file_bytes)
    result  = poller.result()
    return result.content


# ─── Step 2: Chunk ────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """
    Naive fixed-size chunking. Splits on the nearest whitespace boundary so
    chunks never cut mid-word. Skips empty chunks.
    """
    chunks = []
    start  = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Walk back to the nearest whitespace so we don't cut mid-word
        if end < length and not text[end].isspace():
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks


# ─── Step 3: Embed ────────────────────────────────────────────────────────────

async def _embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Calls Azure OpenAI embeddings in one batched request."""
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
        api_key=os.getenv("AZURE_EMBEDDING_KEY"),
        api_version="2023-05-15",
    )
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

    if not deployment:
        raise EnvironmentError("AZURE_EMBEDDING_DEPLOYMENT must be set in your .env")

    response = await client.embeddings.create(input=chunks, model=deployment)
    # response.data is ordered the same as the input list
    return [item.embedding for item in response.data]


# ─── Step 4: Save to pgvector ─────────────────────────────────────────────────

async def _save_to_db(
    db: asyncpg.Connection,
    doc_id: str,
    persona_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    """Inserts all chunks in a single transaction, then marks the doc complete."""
    async with db.transaction():
        for chunk_text, embedding_vector in zip(chunks, embeddings):
            vector_str = "[" + ",".join(map(str, embedding_vector)) + "]"
            await db.execute(
                """
                INSERT INTO kb_chunks (doc_id, persona_id, chunk_text, embedding)
                VALUES ($1, $2, $3, $4::vector)
                """,
                doc_id,
                persona_id,
                chunk_text,
                vector_str,
            )

    await db.execute(
        "UPDATE knowledge_base_docs SET upload_status = 'completed' WHERE id = $1",
        doc_id,
    )


# ─── Main pipeline ────────────────────────────────────────────────────────────

async def run_ingestion(
    file_path: Path,
    persona_id: str,
    doc_id: str,
    chunk_size: int,
) -> None:

    # ── Connect (no SSL for local Postgres) ───────────────────────────────
    dsn = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/voice_db")
    logger.info("Connecting to database...")
    db = await asyncpg.connect(dsn)
    logger.info("Connected.")

    try:
        # ── Register this doc (so the UPDATE in step 4 has a row to hit) ─
        await db.execute(
            """
            INSERT INTO knowledge_base_docs (id, persona_id, filename, upload_status)
            VALUES ($1, $2, $3, 'processing')
            ON CONFLICT (id) DO UPDATE SET upload_status = 'processing'
            """,
            doc_id,
            persona_id,
            file_path.name,
        )

        # ── Step 1: Extract ───────────────────────────────────────────────
        logger.info("Reading file: %s", file_path)
        file_bytes = file_path.read_bytes()

        logger.info("Extracting text via Azure Document Intelligence...")
        extracted_text = await asyncio.to_thread(_extract_text_sync, file_bytes)

        if not extracted_text or not extracted_text.strip():
            raise ValueError("No readable text extracted from the document.")
        logger.info("Extracted %d characters.", len(extracted_text))

        # ── Step 2: Chunk ─────────────────────────────────────────────────
        chunks = _chunk_text(extracted_text, chunk_size)
        logger.info("Created %d chunks (chunk_size=%d).", len(chunks), chunk_size)

        # ── Step 3: Embed ─────────────────────────────────────────────────
        logger.info("Generating embeddings for %d chunks...", len(chunks))
        embeddings = await _embed_chunks(chunks)
        logger.info("Embeddings ready.")

        # ── Step 4: Save ──────────────────────────────────────────────────
        logger.info("Saving to database...")
        await _save_to_db(db, doc_id, persona_id, chunks, embeddings)
        logger.info("✅ Done — doc_id=%s  persona_id=%s  chunks=%d", doc_id, persona_id, len(chunks))

    except Exception as exc:
        logger.error("❌ Ingestion failed: %s", exc)
        try:
            await db.execute(
                "UPDATE knowledge_base_docs SET upload_status = 'failed' WHERE id = $1",
                doc_id,
            )
        except Exception:
            pass  # don't mask the original error
        sys.exit(1)

    finally:
        await db.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a document into the pgvector knowledge base.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to the document to ingest (PDF, DOCX, image, etc.).",
    )
    parser.add_argument(
        "--persona-id",
        required=True,
        help="UUID of the persona this document belongs to.",
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="UUID for this document record. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Target character count per chunk.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.file.exists():
        logger.error("File not found: %s", args.file)
        sys.exit(1)

    doc_id = args.doc_id or str(uuid.uuid4())
    logger.info("doc_id = %s", doc_id)

    asyncio.run(
        run_ingestion(
            file_path=args.file,
            persona_id=args.persona_id,
            doc_id=doc_id,
            chunk_size=args.chunk_size,
        )
    )