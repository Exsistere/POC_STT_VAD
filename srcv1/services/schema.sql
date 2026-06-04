-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =====================================================
-- PERSONAS
-- =====================================================

CREATE TABLE IF NOT EXISTS personas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    tenant_id TEXT NOT NULL,
    name TEXT NOT NULL,

    system_prompt TEXT NOT NULL,
    voice_id TEXT NOT NULL,

    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_personas_tenant
ON personas(tenant_id);

-- =====================================================
-- PROBING QUESTIONS
-- =====================================================

CREATE TABLE IF NOT EXISTS probing_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    persona_id UUID NOT NULL
        REFERENCES personas(id)
        ON DELETE CASCADE,

    question_text TEXT NOT NULL,
    trigger_condition TEXT,
    sequence_order INTEGER DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_probing_questions_persona
ON probing_questions(persona_id);

-- =====================================================
-- KNOWLEDGE BASE DOCUMENTS
-- =====================================================

CREATE TABLE IF NOT EXISTS knowledge_base_docs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    persona_id UUID NOT NULL
        REFERENCES personas(id)
        ON DELETE CASCADE,

    filename TEXT NOT NULL,

    upload_status TEXT NOT NULL DEFAULT 'processing',

    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kb_docs_persona
ON knowledge_base_docs(persona_id);

-- =====================================================
-- VECTOR CHUNKS
-- =====================================================

CREATE TABLE IF NOT EXISTS kb_chunks (
    id BIGSERIAL PRIMARY KEY,

    doc_id UUID NOT NULL
        REFERENCES knowledge_base_docs(id)
        ON DELETE CASCADE,

    persona_id UUID NOT NULL
        REFERENCES personas(id)
        ON DELETE CASCADE,

    chunk_text TEXT NOT NULL,

    embedding vector(1536),

    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS kb_chunks_persona_embedding_idx
ON kb_chunks
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_kb_chunks_persona
ON kb_chunks(persona_id);

-- =====================================================
-- CALLS
-- =====================================================

CREATE TABLE IF NOT EXISTS calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    persona_id UUID NOT NULL
        REFERENCES personas(id)
        ON DELETE CASCADE,

    caller_id TEXT,

    duration_seconds INTEGER DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_calls_persona
ON calls(persona_id);

-- =====================================================
-- CALL TRANSCRIPTS
-- =====================================================

CREATE TABLE IF NOT EXISTS call_transcripts (
    id BIGSERIAL PRIMARY KEY,

    call_id UUID NOT NULL
        REFERENCES calls(id)
        ON DELETE CASCADE,

    speaker TEXT NOT NULL,

    text TEXT NOT NULL,

    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_call_transcripts_call
ON call_transcripts(call_id);

-- =====================================================
-- CALL EVENTS
-- =====================================================

CREATE TABLE IF NOT EXISTS call_events (
    id BIGSERIAL PRIMARY KEY,

    call_id UUID NOT NULL
        REFERENCES calls(id)
        ON DELETE CASCADE,

    event_type TEXT NOT NULL,

    event_data JSONB NOT NULL,

    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_call_events_call
ON call_events(call_id);