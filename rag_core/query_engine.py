# rag_core/query_engine.py
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from rag_core import index_manager # To access INDEX_INSTANCE
import json

async def execute_query(query_text: str, top_k: int = 5) -> dict:
    if not index_manager.INDEX_INSTANCE:
        await index_manager.initialize_index() # Should already be initialized by server startup

    retriever = index_manager.INDEX_INSTANCE.as_retriever(similarity_top_k=top_k)
    
    # Optional: Configure a custom response synthesizer if needed
    # response_synthesizer = get_response_synthesizer(response_mode="compact")
    # query_engine_instance = RetrieverQueryEngine(
    #     retriever=retriever,
    #     response_synthesizer=response_synthesizer,
    # )
    
    # Default query engine
    query_engine_instance = index_manager.INDEX_INSTANCE.as_query_engine(similarity_top_k=top_k)

    response = await query_engine_instance.aquery(query_text) # Use async query

    sources_info = []
    for node in response.source_nodes:
        sources_info.append({
            "id": node.node_id,
            "text_preview": node.get_content(metadata_mode="all")[:250] + "...", # Show some metadata in preview
            "score": float(node.score) if node.score is not None else 0.0,
            "metadata": node.metadata,
        })
    
    return {
        "answer": str(response),
        "source_nodes": sources_info
    }