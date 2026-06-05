import os
import logging
import asyncio
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AsyncAzureOpenAI

# Import the global database pool we created in database.py
import database

load_dotenv()

logger = logging.getLogger("kb-pipeline")

def _analyze_document_sync(endpoint: str, key: str, file_bytes: bytes) -> str:
    """
    Synchronous wrapper for the Azure SDK. 
    This prevents the heavy network request from blocking the FastAPI event loop.
    """
    document_client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))
    poller = document_client.begin_analyze_document("prebuilt-layout", document=file_bytes)
    result = poller.result()
    return result.content

async def process_document_pipeline(doc_id: str, persona_id: str, file_bytes: bytes):
    """Extracts text, creates chunks, embeds them, and saves to pgvector."""
    
    if database.db_pool is None:
        logger.error("Database pool is not initialized!")
        return

    # Securely acquire a connection from the pool
    async with database.db_pool.acquire() as db:
        try:
            logger.info(f"📄 Starting Azure Document extraction for doc {doc_id}...")
            
            # Step 1: Azure Document Intelligence Extraction (Pushed to a background thread)
            endpoint = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
            key = os.getenv("AZURE_DOC_INTEL_KEY")
            
            extracted_text = await asyncio.to_thread(
                _analyze_document_sync, endpoint, key, file_bytes
            )

            if not extracted_text or not extracted_text.strip():
                raise ValueError("No readable text could be extracted from the document.")

            logger.info("✂️ Extraction complete. Chunking text...")

            # Step 2: Basic Chunking
            chunk_size = 1000
            chunks = [extracted_text[i:i+chunk_size] for i in range(0, len(extracted_text), chunk_size)]

            logger.info(f"🧠 Generating embeddings for {len(chunks)} chunks via Azure OpenAI...")

            # Step 3: Generate Vector Embeddings (Azure)
            embed_client = AsyncAzureOpenAI(
                azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
                api_key=os.getenv("AZURE_EMBEDDING_KEY"),
                api_version="2023-05-15"
            )
            embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

            response = await embed_client.embeddings.create(
                input=chunks,
                model=embedding_deployment
            )

            logger.info("💾 Saving chunks and embeddings to PostgreSQL...")

            # Step 4: Save to pgvector Database (Using a transaction for safety)
            async with db.transaction():
                for i, chunk_text in enumerate(chunks):
                    embedding_vector = response.data[i].embedding
                    
                    # Correct formatting for pgvector: "[0.1, 0.2, ...]"
                    vector_str = "[" + ",".join(map(str, embedding_vector)) + "]"
                    
                    await db.execute("""
                        INSERT INTO kb_chunks (doc_id, persona_id, chunk_text, embedding)
                        VALUES ($1, $2, $3, $4::vector)
                    """, doc_id, persona_id, chunk_text, vector_str)

            # Step 5: Mark as complete!
            await db.execute("UPDATE knowledge_base_docs SET upload_status = 'completed' WHERE id = $1", doc_id)
            logger.info(f"✅ Successfully processed document {doc_id}")

        except Exception as e:
            logger.error(f"❌ Pipeline failed for doc {doc_id}: {e}")
            # Mark as failed so UI can tell user to try again
            await db.execute("UPDATE knowledge_base_docs SET upload_status = 'failed' WHERE id = $1", doc_id)