"""
tools.py — Tool registry for the STT→LLM→TTS voice pipeline.

Design:
  - TOOLS          : list of OpenAI-format function schemas, passed to llm.chat()
  - ToolRegistry   : holds shared dependencies (db, embed client, persona_id)
                     injected once at startup via ToolRegistry.init()
  - execute_tool() : single dispatcher — call this from _generate_and_speak()
                     with the name + parsed args from the LLM stream

Adding a new tool:
  1. Append its JSON schema to TOOLS.
  2. Write an async _impl function below.
  3. Add one line to the dispatch table in execute_tool().

"""

import __main__
import asyncio
import logging
import os
from openai import AsyncAzureOpenAI
import asyncpg
# livekit.agents.llm import removed — TOOLS is now plain OpenAI JSON schema
from dotenv import load_dotenv
import time
load_dotenv(override=True) 
logger = logging.getLogger(__name__)


# ─── Tool implementations (will be wrapped in LiveKit Tool objects below) ──────


# ─── Shared dependency container ─────────────────────────────────────────────

class ToolRegistry:
    """
    Holds resources that tool implementations need (DB pool, embed client, etc.).
    Call ToolRegistry.init() once at agent startup before any tool is executed.
    """

    _db_pool     = None
    _persona_id  : str = "default-persona"
    _embed_client: AsyncAzureOpenAI | None = None
    _embed_model : str = ""

    @classmethod
    def init(cls, db_connection, persona_id: str) -> None:
        """
        Wire up shared dependencies.

        Args:
            db_connection : asyncpg pool or connection already open.
            persona_id    : the persona whose KB chunks to query.
        """
        cls._db_pool    = db_connection
        cls._persona_id = persona_id
        cls._embed_client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_EMBEDDING_KEY"),
            api_version="2023-05-15",
        )
        cls._embed_model = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "")
        asyncio.ensure_future(cls._warmup_embed())
        logger.info("ToolRegistry initialised (persona_id=%s)", persona_id)
    
    @classmethod
    async def _warmup_embed(cls) -> None:
        try:
            await cls._embed_client.embeddings.create(input="warmup", model=cls._embed_model)
            logger.info("ToolRegistry: embed connection warmed up")
        except Exception as exc:
            logger.warning("ToolRegistry: embed warmup failed: %s", exc)

# ─── Tool schemas (OpenAI format - kept for reference/validation) ─────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "book_calendar_appointment",
            "description": (
                "Book a car service pickup appointment for the user. "
                "Call this when the user agrees on a specific date and time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date of appointment, e.g. '2026-05-27'",
                    },
                    "time": {
                        "type": "string",
                        "description": "Time of appointment, e.g. '09:30 AM'",
                    },
                },
                "required": ["date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the company knowledge base. "
                "Call this for ANY question about the Company — products, features, pricing, "
                "integrations, use cases, company info, or anything the user wants to know. "
                "This is your PRIMARY source of truth. Always call this before answering "
                "factual questions, even if you think you already know the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The exact question the user asked, formulated as a search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ─── Individual tool implementations ─────────────────────────────────────────

async def book_calendar_appointment(date: str, time: str) -> str:
    """Simulated calendar booking. Replace with your real calendar API call."""
    await asyncio.sleep(1)  # simulate network round-trip
    return f"Success! Appointment booked for {date} at {time}."


async def search_knowledge_base(query: str) -> str:
    """
    RAG retrieval:
      1. Embed the query via Azure OpenAI.
      2. Run a pgvector cosine-similarity search scoped to the current persona.
      3. Return the top-3 chunks as context for the LLM to synthesise from.
    """
    reg = ToolRegistry  # shorthand

    if reg._db_pool is None or reg._embed_client is None:
        logger.error("ToolRegistry.init() was never called — cannot execute RAG tool.")
        return "The knowledge base is temporarily unavailable. Please let the user know."

    try:
        # 1. Embed
        embed_response = await reg._embed_client.embeddings.create(
            input=query,
            model=reg._embed_model,
        )
        query_vector = embed_response.data[0].embedding
        vector_str   = "[" + ",".join(map(str, query_vector)) + "]"

        # 2. Vector similarity search — acquire connection from pool, use, and release
        async with reg._db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT chunk_text
                FROM   kb_chunks
                WHERE  persona_id = $1
                ORDER  BY embedding <-> $2::vector
                LIMIT  3;
                """,
                reg._persona_id,
                vector_str,
            )

            # 3. Build context string
            if rows:
                extracted = "\n\n".join(row["chunk_text"] for row in rows)
                logger.info("RAG: found %d chunks for query %r", len(rows), query[:60])
                return (
                    "Here is the relevant context from the knowledge base:\n"
                    f"{extracted}\n\n"
                    "Use this to answer the user naturally."
                )
            else:
                return "No relevant information found in the knowledge base. Inform the user gracefully."

    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        return "The knowledge base is temporarily unavailable. Please let the user know."


# ─── Tool schemas for LLM (OpenAI / Groq wire format) ────────────────────────
# TOOLS is the list passed to the LLM's tools parameter.
# Each entry follows the OpenAI function-calling schema.
# TOOL_SCHEMAS (defined above) is already in the correct format.

TOOLS: list[dict] = TOOL_SCHEMAS


# ─── Dispatcher ───────────────────────────────────────────────────────────────

# Maps LLM function name → (impl_coroutine, required_arg_keys)
# Adding a new tool = one new entry here + one impl function above.
_DISPATCH: dict[str, tuple] = {
    "book_calendar_appointment": (book_calendar_appointment, ["date", "time"]),
    "search_knowledge_base":     (search_knowledge_base,     ["query"]),
}


async def execute_tool(name: str, args: dict) -> str:
    """
    Dispatch a tool call from the LLM.

    Args:
        name : function name exactly as it appears in TOOLS.
        args : parsed JSON arguments from the LLM stream.

    Returns:
        A plain string that will be fed back to the LLM as the tool result.
    """
    entry = _DISPATCH.get(name)
    if entry is None:
        logger.warning("execute_tool: unknown tool %r", name)
        return f"Tool '{name}' is not implemented."

    impl, required_keys = entry

    # Pull only the keys the impl expects, defaulting missing ones to ""
    kwargs = {k: args.get(k, "") for k in required_keys}
    return await impl(**kwargs)

# async def main():
#     db_pool = None
#     dsn = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/voice_db")
#     db_pool = await asyncpg.create_pool(dsn)
#     ToolRegistry.init(db_connection=db_pool, persona_id="979425e3-927b-4ffb-8be6-84145075c425")
#     query = "What are the products and services offered by the company"
#     start_time = time.perf_counter()
#     result = await search_knowledge_base(query)
#     end_time = time.perf_counter()
#     print(f"RAG result:\n", result[:200])
#     print(f"RAG execution time: {end_time - start_time:.2f} seconds")
    
async def main():
    for i in range(2):
        dsn = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/voice_db")
        db_pool = await asyncpg.create_pool(dsn)
        ToolRegistry.init(db_connection=db_pool, persona_id="979425e3-927b-4ffb-8be6-84145075c425")

        query = "What are the products and services offered by the company"

        # ── Step 1: Embed ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        embed_response = await ToolRegistry._embed_client.embeddings.create(
            input=query,
            model=ToolRegistry._embed_model,
        )
        t1 = time.perf_counter()

        # ── Step 2: pgvector search ───────────────────────────────────────────
        query_vector = embed_response.data[0].embedding
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"

        rows = await ToolRegistry._db.fetch(
            """
            SELECT chunk_text
            FROM   kb_chunks
            WHERE  persona_id = $1
            ORDER  BY embedding <-> $2::vector
            LIMIT  3;
            """,
            ToolRegistry._persona_id,
            vector_str,
        )
        t2 = time.perf_counter()

        # ── Results ───────────────────────────────────────────────────────────
        print(f"Embed:    {t1 - t0:.3f}s")
        print(f"pgvector: {t2 - t1:.3f}s")
        print(f"Total:    {t2 - t0:.3f}s")
        print(f"\nRAG result preview:\n")
        for i, row in enumerate(rows, 1):
            print(f"  Chunk {i}: {row['chunk_text'][:120]!r}")

        await db_pool.close()
        
    
if __name__ == "__main__":
    asyncio.run(main())
