from typing import List, Optional, Tuple
from .RAGasaurus.web_crawler import WebCrawler as RagasaurusCrawler, CrawlConfig, CrawlMethod, Document as RagasaurusDocument
from llama_index.core.schema import Document as LlamaDocument
from persistence.source_metadata_db import get_source_metadata, update_source_metadata, should_refresh_source
import httpx # For parsing cache headers and making conditional requests
import hashlib
import time
import json
from email.utils import parsedate_to_datetime, formatdate

class AdaptedWebCrawler:
    def __init__(self):
        self.crawler = RagasaurusCrawler() # Initialize the RAGasaurus crawler
        # Ensure RAGasaurus's DATA_DIR is configurable or handled
        os.makedirs(config.CRAWL4AI_OUTPUT_DIR, exist_ok=True) # If Ragasaurus uses it

    def _to_llama_document(self, rag_doc: RagasaurusDocument) -> LlamaDocument:
        # Ragasaurus metadata is already a dict.
        # Ensure 'source_url' or 'source_file' is prominent for LlamaIndex.
        llama_meta = rag_doc.metadata.copy()
        if rag_doc.source_url:
            llama_meta["source_url"] = rag_doc.source_url
        if rag_doc.source_file:
            llama_meta["source_file"] = rag_doc.source_file

        # Create a stable document ID for LlamaIndex based on the source
        # This helps in updating/deleting later.
        doc_id_seed = rag_doc.source_url or rag_doc.source_file or rag_doc.content[:100]
        stable_doc_id = hashlib.md5(doc_id_seed.encode()).hexdigest()
        
        return LlamaDocument(
            text=rag_doc.content,
            metadata=llama_meta,
            doc_id=stable_doc_id, # LlamaIndex uses id_ internally, but doc_id can be used for ref_doc_id
            # excluded_llm_metadata_keys=['source_url'], # Example if you want to control what LLM sees
            # excluded_embed_metadata_keys=[]
        )

    async def crawl_and_get_documents(
        self,
        url: str,
        method: CrawlMethod = CrawlMethod.RECURSIVE,
        max_depth: int = 2,
        max_pages: int = 10,
        output_name: Optional[str] = None,
        force_refresh: bool = False,
        treat_as_raw_text_list: bool = False, # For llms.txt containing list of URLs
        ingest_url_content_as_is: bool = False # For llms.txt itself as a document
    ) -> Tuple[List[LlamaDocument], List[str]]: # Returns LlamaDocuments and list of source_identifiers processed
        
        processed_identifiers = []
        all_llama_documents = []

        if treat_as_raw_text_list:
            # Fetch the content of 'url', parse it as a list of new URLs, then crawl each.
            print(f"Fetching URL list from: {url}")
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url)
                    response.raise_for_status()
                content_urls = [u.strip() for u in response.text.splitlines() if u.strip().startswith('http')]
                print(f"Found {len(content_urls)} URLs in list. Crawling each as single page...")
                for i, content_url in enumerate(content_urls):
                    if max_pages > 0 and i >= max_pages:
                        print(f"Reached max_pages limit ({max_pages}) for URL list processing.")
                        break
                    # Crawl each URL from the list as a single page
                    # Recursively call this function, but for a single page, not as a list
                    docs, ids = await self.crawl_and_get_documents(
                        url=content_url, method=CrawlMethod.SINGLE_PAGE, max_pages=1, 
                        output_name=output_name, force_refresh=force_refresh,
                        treat_as_raw_text_list=False, ingest_url_content_as_is=True # Ingest content of each listed URL
                    )
                    all_llama_documents.extend(docs)
                    processed_identifiers.extend(ids)
                return all_llama_documents, processed_identifiers
            except Exception as e:
                print(f"Error processing URL list {url}: {e}")
                return [], []

        # --- Standard single URL processing (web page, sitemap, or raw text file) ---
        source_identifier = url
        if not force_refresh and not should_refresh_source(source_identifier):
            print(f"Skipping {source_identifier}, content is considered fresh.")
            return [], [source_identifier] # Return empty list if no refresh needed

        # Prepare for potential conditional GET
        headers = {}
        # source_meta = get_source_metadata(source_identifier) # Not used here, Ragasaurus handles its own fetch
        # if source_meta and source_meta.get('etag'):
        #     headers['If-None-Match'] = source_meta['etag']
        # if source_meta and source_meta.get('last_modified_at_source'):
        #     headers['If-Modified-Since'] = formatdate(source_meta['last_modified_at_source'], usegmt=True)
        # The Ragasaurus crawler needs to be adapted to use these headers.
        # For now, Crawl4AI's CacheMode.ENABLED will handle some of this.

        crawl_config_dict = {
            "url": url, "method": method, "max_depth": max_depth, "max_pages": max_pages,
            "output_name": output_name or hashlib.md5(url.encode()).hexdigest(),
            # user_agent, timeout etc. will use Ragasaurus defaults or can be passed
        }
        # Ragasaurus's ingest directly uses kwargs.
        if ingest_url_content_as_is and (url.endswith((".txt", ".md")) or "text/plain" in (await httpx.head(url)).headers.get("content-type","")):
             # Make Ragasaurus treat this URL specifically as a raw text file.
             # Ragasaurus already has some logic for this based on extension/content-type.
             # This flag makes it explicit if needed beyond its auto-detection.
             print(f"Instructing crawler to treat {url} as raw text content.")
             crawl_config_dict["method"] = CrawlMethod.SINGLE_PAGE # Ensure single page for raw content

        ingestion_result = await self.crawler.ingest(**crawl_config_dict)
        
        ragasaurus_docs: List[RagasaurusDocument] = []
        if ingestion_result.status == ingestion_result.status.SUCCESS:
            # The current Ragasaurus ingest doesn't directly return documents in IngestionResult.
            # It saves to a file (details["raw_output_path"]). We need to load that or modify Ragasaurus.
            # For now, let's assume Ragasaurus's _crawl_xxx methods are modified to return docs.
            # Or, if crawl4ai was used, result.markdown is the content.
            # This part needs Ragasaurus's internal structure to be clear on how to get the actual Document objects.
            
            # TEMPORARY: Simulating Ragasaurus returning documents
            # This needs actual integration with how Ragasaurus provides the crawled docs.
            # Let's assume the `_process_crawl4ai_result` or similar in Ragasaurus makes RagasaurusDocument objects.
            # And these are somehow accessible post-ingest.
            # If RAGASAURUS_OUTPUT_DIR/output_name_raw.json contains the data:
            if "raw_output_path" in ingestion_result.details and os.path.exists(ingestion_result.details["raw_output_path"]):
                try:
                    with open(ingestion_result.details["raw_output_path"], 'r') as f:
                        raw_docs_data = json.load(f)
                    for doc_data in raw_docs_data:
                        # Assuming doc_data matches RagasaurusDocument structure for this simulation
                        ragasaurus_docs.append(RagasaurusDocument(**doc_data))
                except Exception as e:
                    print(f"Error reading Ragasaurus output file: {e}")

            print(f"Crawled {len(ragasaurus_docs)} items from {source_identifier}")
        else:
            print(f"Crawling failed for {source_identifier}: {ingestion_result.message}")
            return [], []

        new_llama_documents = []
        for rag_doc in ragasaurus_docs:
            llama_doc = self._to_llama_document(rag_doc)
            new_llama_documents.append(llama_doc)

            # Update source metadata for this specific document (URL)
            # (Ragasaurus `Document` already has `url` in its metadata)
            doc_url = rag_doc.metadata.get("url", source_identifier) # Fallback to main url
            content_hash = hashlib.md5(rag_doc.content.encode()).hexdigest()
            
            # Extract cache info from Ragasaurus metadata if it stored response headers
            # http_response_headers = rag_doc.metadata.get("http_response_headers", {}) # Hypothetical
            # etag = http_response_headers.get("ETag")
            # last_mod_str = http_response_headers.get("Last-Modified")
            # last_mod_ts = parsedate_to_datetime(last_mod_str).timestamp() if last_mod_str else None
            # ... parse cache_control, expires ...
            
            source_meta_entry = {
                "source_type": "web", "last_ingested": time.time(),
                "content_hash": content_hash,
                "llama_index_doc_ids": json.dumps([llama_doc.doc_id]) # Or node_id
                # "etag": etag, "last_modified_at_source": last_mod_ts, ...
            }
            update_source_metadata(doc_url, source_meta_entry)
            if doc_url not in processed_identifiers:
                processed_identifiers.append(doc_url)
        
        return new_llama_documents, processed_identifiers