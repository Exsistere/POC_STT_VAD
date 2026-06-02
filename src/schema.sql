-- Enable the pgvector extension (only needed once per database)
CREATE EXTENSION IF NOT EXISTS vector;

-- Tracks each uploaded document and its processing status
CREATE TABLE IF NOT EXISTS knowledge_base_docs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    persona_id    UUID NOT NULL,
    filename      TEXT NOT NULL,
    upload_status TEXT NOT NULL DEFAULT 'processing',  -- processing | completed | failed
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Stores the actual chunks + embeddings
CREATE TABLE IF NOT EXISTS kb_chunks (
    id         BIGSERIAL PRIMARY KEY,
    doc_id     UUID NOT NULL REFERENCES knowledge_base_docs(id) ON DELETE CASCADE,
    persona_id UUID NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding  vector(1536),  -- 1536 for text-embedding-ada-002, change if using a different model
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Index for fast vector similarity search scoped by persona
CREATE INDEX IF NOT EXISTS kb_chunks_persona_embedding_idx
    ON kb_chunks
    USING hnsw (embedding vector_cosine_ops)
    WHERE persona_id IS NOT NULL;