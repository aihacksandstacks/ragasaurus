from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # Keep for comparison or fallback
from llama_index.embeddings.ollama import OllamaEmbedding       # Keep for comparison or fallback
from llama_index.embeddings.fastembed import FastEmbedEmbedding # <--- IMPORT THIS

import config

def initialize_llama_index_settings():
    """Initializes global LlamaIndex settings for LLM and embedding model."""
    if config.LLM_PROVIDER == "ollama":
        Settings.llm = Ollama(model=config.LLM_MODEL_NAME, base_url=config.OLLAMA_BASE_URL, request_timeout=120.0)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER}")

    if config.EMBED_MODEL_PROVIDER == "fastembed":
        print(f"Using FastEmbed with model: {config.EMBED_MODEL_NAME}")
        Settings.embed_model = FastEmbedEmbedding(
            model_name=config.EMBED_MODEL_NAME,
            # cache_dir="path/to/your/fastembed_cache" # Optional: Default is ~/.cache/fastembed
            # threads=None,  # Optional: Defaults to number of CPU cores
            # doc_embed_type="default", # "default" for asymmetrical, "passage" for symmetrical query/doc
            # query_embed_type="default" # "default" for asymmetrical, "query" for symmetrical query/doc
            # For BGE models, "default" (asymmetric) or "passage"/"query" (symmetric) is often relevant.
            # Nomic might be symmetric, check its model card. If so, use "passage" for doc, "query" for query.
            # However, FastEmbedEmbedding as of recent versions might simplify this.
            # The default behavior of FastEmbedEmbedding usually handles this well.
        )
        # Note: FastEmbed's `QdrantEmbedding` (which `FastEmbedEmbedding` wraps)
        # supports quantization, but it's usually enabled in the Qdrant client or specific model loading.
        # The LlamaIndex wrapper might not directly expose quantization flags as it relies on
        # the underlying FastEmbed library's model loading.
        # Check FastEmbed's own docs for how it handles quantization with specified models.
        # For many models, FastEmbed will use the ONNX version by default if available.

    elif config.EMBED_MODEL_PROVIDER == "huggingface":
        print(f"Using HuggingFaceEmbedding with model: {config.EMBED_MODEL_NAME}")
        Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL_NAME)
    elif config.EMBED_MODEL_PROVIDER == "ollama":
        print(f"Using OllamaEmbedding with model: {config.EMBED_MODEL_NAME}")
        Settings.embed_model = OllamaEmbedding(
            model_name=config.EMBED_MODEL_NAME,
            base_url=config.OLLAMA_BASE_URL,
            ollama_additional_kwargs={"mirostat": 0}
        )
    else:
        raise ValueError(f"Unsupported EMBED_MODEL_PROVIDER: {config.EMBED_MODEL_PROVIDER}")

    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP
    print(f"LlamaIndex LLM and Embedding models initialized (Embed Provider: {config.EMBED_MODEL_PROVIDER}).")