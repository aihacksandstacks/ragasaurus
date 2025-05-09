import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM & Embedding Models ---
# EMBED_MODEL_PROVIDER = "huggingface" # or "ollama" or "fastembed"
# For local HuggingFace embedding model (e.g., Nomic)
# EMBED_MODEL_PROVIDER = "huggingface" # or "ollama"
# EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5" # Or your preferred SentenceTransformer
EMBED_MODEL_PROVIDER = "fastembed" # <--- CHANGE THIS
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5" # Or "BAAI/bge-base-en-v1.5"
DB_EMBED_DIM = 768 # For nomic-embed-text-v1.5 or bge-base
# If using bge-small-en-v1.5, then DB_EMBED_DIM = 384# For Ollama embedding (if you prefer to use Ollama for embeddings too)
# EMBED_MODEL_PROVIDER = "ollama"
# EMBED_MODEL_NAME = "nomic-embed-text" # Or other model served by Ollama

LLM_PROVIDER = "ollama"
LLM_MODEL_NAME = "qwen:latest" # Your Ollama model for synthesis
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- PostgreSQL (PGVector) ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "your_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
DB_NAME = os.getenv("DB_NAME", "your_db_name")
DB_SCHEMA_NAME = "rag_schema" # Optional: specify a schema
DB_TABLE_NAME = "coding_assistant_docs" # LlamaIndex table for documents
DB_EMBED_DIM = 768 # Dimension for nomic-embed-text-v1.5. Adjust if using other model.

# --- Indexing & Crawling ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
DEFAULT_REFRESH_INTERVAL_DAYS = 7 # For sources without cache headers
CRAWL4AI_OUTPUT_DIR = "./crawl4ai_temp_output" # If Crawl4AI needs a temp dir

# --- Source Metadata DB (SQLite for simplicity, or use another Postgres table) ---
SOURCE_METADATA_DB_PATH = "./persistent_storage/source_metadata.sqlite"