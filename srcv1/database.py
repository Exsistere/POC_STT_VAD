# database.py
import os
import ssl
import asyncpg
from typing import AsyncGenerator

# We will store the global pool here
db_pool = None

async def init_db_pool():
    global db_pool
    dsn = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/voice_db")

    # Azure Postgres requires SSL — create a permissive SSL context
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    db_pool = await asyncpg.create_pool(
        dsn,
        # ssl=ssl_ctx,
        ssl = False,
        timeout=30,
        min_size=1,
        max_size=5,
    )

async def close_db_pool():
    global db_pool
    if db_pool:
        await db_pool.close()

async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Dependency injection for FastAPI routes"""
    global db_pool
    async with db_pool.acquire() as connection:
        yield connection

def get_db_pool():
    """Returns the module-level db_pool object for long-lived operations."""
    return db_pool