from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
import asyncio # PGVectorStore might need an event loop for async operations

import config
from models.llm_setup import initialize_llama_index_settings # Ensure settings are up

# Placeholder for actual LlamaIndex Document type
from llama_index.core.schema import Document as LlamaDocument
from typing import List, Optional

# Global index instance
INDEX_INSTANCE: Optional[VectorStoreIndex] = None

async def get_vector_store() -> PGVectorStore:
    connection_string = f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
    db_url = make_url(connection_string)

    return PGVectorStore.from_params(
        database=config.DB_NAME,
        host=db_url.host,
        password=db_url.password,
        port=db_url.port,
        user=db_url.username,
        table_name=config.DB_TABLE_NAME,
        schema_name=config.DB_SCHEMA_NAME,
        embed_dim=config.DB_EMBED_DIM,
        # hybrid_search=True, # If you want hybrid search
        # text_search_config="english" # For hybrid search
    )

async def initialize_index():
    global INDEX_INSTANCE
    initialize_llama_index_settings() # Setup LLM and Embeddings first

    vector_store = await get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Attempt to load the index if it might exist, or prepare for new documents
    # PGVectorStore manages persistence directly, so loading is more about connecting.
    # If the table is empty, from_documents([]) with the store will work.
    # If documents exist, we can just point VectorStoreIndex to this store.
    INDEX_INSTANCE = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context, # Not strictly needed if vector_store is passed
    )
    print(f"Connected to PGVectorStore. Index instance is ready.")
    # You might want to create tables if they don't exist upon first run
    # vector_store.create_tables_if_not_exists() # Check PGVectorStore API for exact method

async def add_documents_to_index(documents: List[LlamaDocument]):
    if not INDEX_INSTANCE:
        await initialize_index() # Ensure index is initialized
    if not documents:
        print("No documents to add.")
        return

    print(f"Adding/updating {len(documents)} documents in the index...")
    # PGVectorStore and VectorStoreIndex handle upserts based on document_id if provided
    # Or use index.insert_nodes / update_ref_doc / delete_ref_doc for finer control
    INDEX_INSTANCE.insert_nodes(documents) # Assumes documents are Node objects
    print(f"{len(documents)} documents processed for index.")

# Example of how you might delete and re-insert for an update
# async def update_document_in_index(document: LlamaDocument):
#     if not INDEX_INSTANCE:
#         await initialize_index()
#     # Assuming document.id_ is set to a stable identifier for the source
#     INDEX_INSTANCE.update_ref_doc(document, update_kwargs={'doc_id': document.doc_id})