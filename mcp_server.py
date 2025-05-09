from mcp.server.fastmcp import FastMCP
from typing import Optional, List, Dict, Any
import json
import os
import asyncio

# Project modules
import config
from models.llm_setup import initialize_llama_index_settings
from rag_core import index_manager, query_engine # query_engine needs to be created
from data_ingestion.local_file_ingester import load_local_directory
from data_ingestion.web_crawler_adapter import AdaptedWebCrawler, CrawlMethod
from persistence.source_metadata_db import create_tables as create_metadata_db_tables

# Initialize MCP application
mcp_app = FastMCP(
    name="coding_assistant_rag_server",
    version="0.1.0",
    description="RAG system for AI Coding Assistants, accessible via MCP."
    # publisher_info={"name": "YourName/Org", "contact": "your@email.com"}
)

# Global crawler instance
web_crawler_adapter_instance = None

# --- Helper to add docs to index ---
async def _add_llama_documents_to_index(llama_docs: List[index_manager.LlamaDocument], source_identifiers: List[str]):
    if not llama_docs:
        print(f"No new documents to add from sources: {source_identifiers}")
        return "No new documents to add."
    
    # Here, you might want to handle updates/deletions if docs for these source_identifiers already exist.
    # For simplicity now, we're just adding. DocumentUpdater module would handle this.
    # E.g., delete old docs associated with source_identifier from index_manager.INDEX_INSTANCE
    # and source_metadata_db before adding new ones.
    
    await index_manager.add_documents_to_index(llama_docs)
    return f"Successfully processed {len(llama_docs)} documents from {len(source_identifiers)} sources."

# --- MCP Tool Definitions ---
@mcp_app.tool()
async def ingest_local_fs_directory(directory_path: str, force_refresh: bool = False) -> str:
    """Ingests/updates local files from a specified directory into the RAG knowledge base.
    Args:
        directory_path: Absolute path to the directory.
        force_refresh: If True, re-ingests all files regardless of modification status.
    Returns:
        JSON string status message.
    """
    try:
        if not os.path.isdir(directory_path):
            return json.dumps({"status": "error", "message": f"Directory not found: {directory_path}"})
        
        llama_docs = load_local_directory(directory_path, force_refresh)
        result_message = await _add_llama_documents_to_index(llama_docs, [directory_path])
        return json.dumps({"status": "success", "message": result_message, "documents_loaded": len(llama_docs)})
    except Exception as e:
        print(f"Error in ingest_local_fs_directory: {e}")
        return json.dumps({"status": "error", "message": str(e)})

@mcp_app.tool()
async def ingest_web_content(
    url: str,
    crawl_method_str: str = "recursive", # "recursive", "sitemap", "single_page"
    max_depth: int = 2,
    max_pages: int = 10,
    force_refresh: bool = False
) -> str:
    """Crawls a website (or sitemap, or single page) and ingests content.
    Args:
        url: Starting URL.
        crawl_method_str: 'recursive', 'sitemap', or 'single_page'.
        max_depth: Max depth for recursive crawling.
        max_pages: Max pages to process.
        force_refresh: If True, re-crawls even if cache suggests content is fresh.
    Returns:
        JSON string status message.
    """
    global web_crawler_adapter_instance
    if web_crawler_adapter_instance is None: # Lazy initialization
        web_crawler_adapter_instance = AdaptedWebCrawler()
    try:
        method_enum = CrawlMethod[crawl_method_str.upper()]
        llama_docs, source_ids = await web_crawler_adapter_instance.crawl_and_get_documents(
            url=url, method=method_enum, max_depth=max_depth, max_pages=max_pages,
            force_refresh=force_refresh
        )
        result_message = await _add_llama_documents_to_index(llama_docs, source_ids)
        return json.dumps({"status": "success", "message": result_message, "documents_added": len(llama_docs), "sources_processed": source_ids})
    except Exception as e:
        print(f"Error in ingest_web_content: {e}")
        return json.dumps({"status": "error", "message": str(e)})

@mcp_app.tool()
async def ingest_url_list_from_remote_file(
    list_file_url: str,
    max_pages_per_url_in_list: int = 1, # Typically each URL in list is a single doc/page
    force_refresh: bool = False
) -> str:
    """Fetches a remote .txt file containing a list of URLs, then ingests each URL's content.
    Args:
        list_file_url: URL of the .txt file listing content URLs.
        max_pages_per_url_in_list: Max pages to crawl for each URL from the list (usually 1).
        force_refresh: If True, re-crawls listed URLs even if cache suggests freshness.
    Returns:
        JSON string status message.
    """
    global web_crawler_adapter_instance
    if web_crawler_adapter_instance is None:
        web_crawler_adapter_instance = AdaptedWebCrawler()
    try:
        llama_docs, source_ids = await web_crawler_adapter_instance.crawl_and_get_documents(
            url=list_file_url, # This URL is the list file itself
            treat_as_raw_text_list=True, # Adapter handles fetching list and then individual URLs
            max_pages=max_pages_per_url_in_list, # This will apply to each URL *within* the list
            force_refresh=force_refresh
        )
        result_message = await _add_llama_documents_to_index(llama_docs, source_ids)
        return json.dumps({"status": "success", "message": result_message, "documents_added": len(llama_docs), "sources_processed": len(source_ids)})
    except Exception as e:
        print(f"Error in ingest_url_list_from_remote_file: {e}")
        return json.dumps({"status": "error", "message": str(e)})

@mcp_app.tool()
async def ingest_raw_text_content_from_single_url(
    content_url: str,
    force_refresh: bool = False
) -> str:
    """Fetches a single URL (e.g. a .txt or .md file) and ingests its raw content as one document.
    Args:
        content_url: The URL of the file whose content is to be ingested directly.
        force_refresh: If True, re-fetches even if cache suggests freshness.
    Returns:
        JSON string status message.
    """
    global web_crawler_adapter_instance
    if web_crawler_adapter_instance is None:
        web_crawler_adapter_instance = AdaptedWebCrawler()
    try:
        # The crawler adapter's ingest_url_content_as_is will make Ragasaurus get the raw text
        llama_docs, source_ids = await web_crawler_adapter_instance.crawl_and_get_documents(
            url=content_url,
            method=CrawlMethod.SINGLE_PAGE, # Ensure it's treated as a single item
            ingest_url_content_as_is=True,
            force_refresh=force_refresh,
            max_pages=1
        )
        result_message = await _add_llama_documents_to_index(llama_docs, source_ids)
        return json.dumps({"status": "success", "message": result_message, "documents_added": len(llama_docs)})
    except Exception as e:
        print(f"Error in ingest_raw_text_content_from_single_url: {e}")
        return json.dumps({"status": "error", "message": str(e)})

@mcp_app.tool()
async def query_knowledge_base(query_text: str, top_k: int = 5) -> str:
    """Queries the RAG knowledge base.
    Args:
        query_text: The question.
        top_k: Number of relevant chunks to retrieve for context.
    Returns:
        JSON string containing the answer and source information.
    """
    try:
        if not index_manager.INDEX_INSTANCE:
            await index_manager.initialize_index()
        
        # query_engine.py would contain:
        # query_engine_instance = index_manager.INDEX_INSTANCE.as_query_engine(similarity_top_k=top_k)
        # response = query_engine_instance.query(query_text)
        # result = {"answer": str(response), "sources": [{"text": node.get_content()[:200], "score": node.score, "metadata": node.metadata} for node in response.source_nodes]}
        # return json.dumps(result)
        
        # Placeholder for query_engine.py implementation
        response_placeholder = await query_engine.execute_query(query_text, top_k) # This function needs to be created
        return json.dumps(response_placeholder)

    except Exception as e:
        print(f"Error in query_knowledge_base: {e}")
        return json.dumps({"status": "error", "message": str(e)})

# --- Main Execution & Setup ---
async def main_async_setup():
    """Performs asynchronous setup."""
    print("Performing async setup...")
    initialize_llama_index_settings()
    create_metadata_db_tables() # Create SQLite table for source metadata if not exists
    await index_manager.initialize_index() # Connect to PGVector and prepare LlamaIndex
    global web_crawler_adapter_instance
    web_crawler_adapter_instance = AdaptedWebCrawler() # Pre-initialize
    print("Async setup complete. MCP server starting...")

if __name__ == "__main__":
    # Run async setup tasks
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main_async_setup())
    except Exception as e:
        print(f"Error during async setup: {e}. Server might not function correctly.")
        # Decide if to exit or continue with potentially partial setup
    
    print("Starting MCP server via stdio transport...")
    # This will block and run the MCP server, listening on stdio
    mcp_app.run(transport='stdio')