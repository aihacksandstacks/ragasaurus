"""
Web crawler for RAGasaurus.
Provides functionality for crawling websites and extracting content.
"""

import os
import asyncio
import re
import time
import urllib.parse
from typing import List, Dict, Any, Optional, Set, Tuple, Union
import json
from pathlib import Path
from dataclasses import dataclass, field # Added field for default_factory
from datetime import datetime
import traceback
from enum import Enum

import httpx
# import urllib.parse # Already imported above
from bs4 import BeautifulSoup
from xml.etree import ElementTree # For sitemap parsing

# Assuming these are defined elsewhere in your RAGasaurus project structure
# For self-contained execution, minimal versions are provided below.

# --- Minimal Reproductions of Assumed External Dependencies ---
class IngestionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    SKIPPED = "skipped"

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_url: Optional[str] = None
    source_file: Optional[str] = None
    source_repo: Optional[str] = None # For code repositories
    id: Optional[str] = None # Unique ID for the document

@dataclass
class IngestionResult:
    status: IngestionStatus
    message: str
    source_identifier: Optional[str] = None
    source_type: Optional[str] = None # e.g., "web", "file", "github"
    items_processed: int = 0
    items_succeeded: int = 0
    items_failed: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def complete(self):
        self.end_time = time.time()

    def add_succeeded_item(self):
        self.items_processed += 1
        self.items_succeeded += 1

    def add_failed_item(self):
        self.items_processed += 1
        self.items_failed += 1

# --- Mock Logger and Config (replace with your actual project's setup) ---
class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

logger = MockLogger()

class MockCrawlerConfig:
    def __init__(self):
        self.max_depth = 2
        self.max_pages = 50
        self.timeout = 30
        self.user_agent = "RAGasaurus/1.0.0 (Mock)"
        self.respect_robots_txt = True # This is a placeholder; actual implementation is complex
        self.max_concurrent = 10

class MockConfig:
    def __init__(self):
        self.crawler = MockCrawlerConfig()

_mock_config_instance = MockConfig()

def get_config():
    return _mock_config_instance

DATA_DIR = Path("./ragasaurus_data")
DATA_DIR.mkdir(exist_ok=True)

def add_source(source_type: str, identifier: str, metadata: Dict[str, Any]) -> str:
    # Mock implementation
    logger.info(f"Mock DB: Adding source {source_type} - {identifier} with metadata {metadata}")
    source_id = f"{source_type}_{identifier.replace('/', '_').replace(':', '_')}"
    # In a real scenario, this would interact with a database.
    return source_id
# --- End of Minimal Reproductions ---


class CrawlMethod(str, Enum):
    """Methods for crawling a website."""
    SINGLE_PAGE = "single_page"
    RECURSIVE = "recursive"
    SITEMAP = "sitemap"


@dataclass
class CrawlConfig:
    """Configuration for a web crawl."""
    url: str
    max_depth: int = 2
    max_pages: int = 50
    method: CrawlMethod = CrawlMethod.RECURSIVE
    timeout: int = 30
    user_agent: str = "RAGasaurus/1.0.0"
    respect_robots_txt: bool = True # Note: Full robots.txt parsing is not implemented here
    follow_links: bool = True
    max_concurrent: int = 10
    output_name: Optional[str] = None
    include_patterns: Optional[List[str]] = None # Changed to Optional
    exclude_patterns: Optional[List[str]] = None # Changed to Optional


class WebCrawler: # Removed (Ingester) as Ingester base class not provided
    """Crawler for extracting content from websites."""

    def __init__(self):
        """Initialize the web crawler."""
        # Get configuration
        config = get_config()
        self.default_max_depth = config.crawler.max_depth
        self.default_max_pages = config.crawler.max_pages
        self.default_timeout = config.crawler.timeout
        self.default_user_agent = config.crawler.user_agent
        self.default_respect_robots_txt = config.crawler.respect_robots_txt
        self.default_max_concurrent = config.crawler.max_concurrent

        # Create data directory for crawled content
        self.crawled_data_dir = Path(DATA_DIR) / "crawled"
        self.crawled_data_dir.mkdir(parents=True, exist_ok=True)

        # Check if crawl4ai is installed
        try:
            import crawl4ai
            self.crawl4ai_available = True
            logger.info("Crawl4AI is available for advanced crawling")
        except ImportError:
            self.crawl4ai_available = False
            logger.info("Crawl4AI not available, using built-in HTTP crawler")

    async def ingest(self, **kwargs) -> IngestionResult:
        """
        Crawl a website and extract content.

        Args:
            url: URL to crawl
            max_depth: Maximum depth to crawl
            max_pages: Maximum pages to crawl
            method: Crawl method (single_page, recursive, sitemap)
            timeout: Request timeout in seconds
            user_agent: User agent string
            respect_robots_txt: Whether to respect robots.txt
            follow_links: Whether to follow links in pages
            max_concurrent: Maximum concurrent requests
            output_name: Name for the output files
            include_patterns: Patterns for URLs to include
            exclude_patterns: Patterns for URLs to exclude

        Returns:
            IngestionResult with details about the crawl
        """
        # Create config from parameters
        crawl_cfg = CrawlConfig( # Renamed to avoid conflict with module-level 'config'
            url=kwargs.get("url"),
            max_depth=kwargs.get("max_depth", self.default_max_depth),
            max_pages=kwargs.get("max_pages", self.default_max_pages),
            method=kwargs.get("method", CrawlMethod.RECURSIVE),
            timeout=kwargs.get("timeout", self.default_timeout),
            user_agent=kwargs.get("user_agent", self.default_user_agent),
            respect_robots_txt=kwargs.get("respect_robots_txt", self.default_respect_robots_txt),
            follow_links=kwargs.get("follow_links", True),
            max_concurrent=kwargs.get("max_concurrent", self.default_max_concurrent),
            output_name=kwargs.get("output_name"),
            include_patterns=kwargs.get("include_patterns", []),
            exclude_patterns=kwargs.get("exclude_patterns", [])
        )

        # Validate URL
        if not crawl_cfg.url:
            raise ValueError("URL is required")

        # Create output name if not provided
        if not crawl_cfg.output_name:
            domain = urllib.parse.urlparse(crawl_cfg.url).netloc
            crawl_cfg.output_name = domain.replace(".", "_") if domain else "crawled_site"

        # Create result object
        result = IngestionResult(
            status=IngestionStatus.IN_PROGRESS, # Start as in_progress
            message=f"Starting crawl of {crawl_cfg.url}",
            source_identifier=crawl_cfg.url,
            source_type="web",
            details={"config": crawl_cfg.__dict__}
        )

        crawled_pages_docs: List[Document] = [] # To store Document objects

        try:
            logger.info(f"Starting crawl of {crawl_cfg.url} with method {crawl_cfg.method}")

            if self.crawl4ai_available:
                logger.info("Using Crawl4AI for crawling")
                crawled_pages_docs = await self._crawl_with_crawl4ai(crawl_cfg)
            else:
                if crawl_cfg.method == CrawlMethod.SINGLE_PAGE:
                    crawled_pages_docs = await self._crawl_single_page(crawl_cfg)
                elif crawl_cfg.method == CrawlMethod.SITEMAP:
                    crawled_pages_docs = await self._crawl_sitemap(crawl_cfg)
                else: # Default to RECURSIVE
                    crawled_pages_docs = await self._crawl_recursive(crawl_cfg)

            for _ in crawled_pages_docs: # Iterate over actual documents
                result.add_succeeded_item()

            result.details["pages_crawled"] = len(crawled_pages_docs)

            source_metadata = {
                "domain": urllib.parse.urlparse(crawl_cfg.url).netloc,
                "crawl_time": datetime.now().isoformat(),
                "crawl_method": crawl_cfg.method.value, # Use enum value
                "pages_crawled": len(crawled_pages_docs),
                "max_depth": crawl_cfg.max_depth,
                "max_pages": crawl_cfg.max_pages
            }

            source_id = add_source("web", crawl_cfg.url, source_metadata)
            result.details["source_id"] = source_id

            logger.info(f"Crawl completed with {len(crawled_pages_docs)} pages")

            output_path = self.crawled_data_dir / f"{crawl_cfg.output_name}_raw.json"
            serializable_pages = [self._make_serializable(page_doc) for page_doc in crawled_pages_docs]
            with open(output_path, "w", encoding="utf-8") as f: # Added encoding
                json.dump(serializable_pages, f, indent=2)

            result.details["raw_output_path"] = str(output_path)
            result.message = f"Successfully crawled {len(crawled_pages_docs)} pages from {crawl_cfg.url}"
            result.status = IngestionStatus.SUCCESS
            result.complete()

            # The ingest method should return the IngestionResult, which contains the list of Document objects
            # via the raw_output_path or by adding them to result.details if preferred.
            # For direct use, one might modify this to return crawled_pages_docs directly or as part of result.
            return result

        except Exception as e:
            logger.error(f"Error during crawl: {e}")
            logger.debug(traceback.format_exc())

            result.status = IngestionStatus.ERROR
            result.message = f"Error during crawl: {str(e)}"
            result.error = e
            result.complete()

            return result

    async def _crawl_with_crawl4ai(self, config: CrawlConfig) -> List[Document]:
        try:
            if config.url.endswith((".txt", ".md", ".text")):
                logger.info(f"Detected direct text file URL: {config.url}")
                return await self._crawl_single_page(config)

            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
            from crawl4ai.content_filter_strategy import PruningContentFilter
            from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

            browser_config = BrowserConfig(headless=True, user_agent=config.user_agent, verbose=False)
            pruning_filter = PruningContentFilter(threshold=0.5, threshold_type="fixed", min_word_threshold=10)
            md_generator = DefaultMarkdownGenerator(content_filter=pruning_filter)
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.ENABLED, markdown_generator=md_generator,
                word_count_threshold=10, exclude_external_links=True,
                remove_overlay_elements=True, process_iframes=True,
                wait_until="networkidle", delay_before_return_html=0.5
            )

            crawled_docs: List[Document] = []
            async with AsyncWebCrawler(config=browser_config) as crawler:
                if config.method == CrawlMethod.SINGLE_PAGE:
                    crawl_result = await crawler.arun(url=config.url, config=run_config)
                    if crawl_result.success and crawl_result.markdown:
                        crawled_docs.append(self._process_crawl4ai_result(crawl_result))
                elif config.method == CrawlMethod.SITEMAP:
                    sitemap_urls = await self._parse_sitemap_async(config.url) # Made async
                    if not sitemap_urls:
                        logger.warning(f"No URLs found in sitemap {config.url}")
                    else:
                        if config.max_pages > 0 and len(sitemap_urls) > config.max_pages:
                            sitemap_urls = sitemap_urls[:config.max_pages]
                        
                        from crawl4ai.dispatcher import MemoryAdaptiveDispatcher # Local import
                        dispatcher = MemoryAdaptiveDispatcher(
                            memory_threshold_percent=70.0, check_interval=1.0,
                            max_session_permit=config.max_concurrent
                        )
                        results = await crawler.arun_many(urls=sitemap_urls, config=run_config, dispatcher=dispatcher)
                        for r in results:
                            if r.success and r.markdown:
                                crawled_docs.append(self._process_crawl4ai_result(r))
                else: # RECURSIVE
                    visited = set()
                    to_visit = [(config.url, 1)]
                    base_domain = urllib.parse.urlparse(config.url).netloc

                    while to_visit and len(crawled_docs) < config.max_pages:
                        current_batch = []
                        processed_in_batch = 0
                        while to_visit and len(current_batch) < config.max_concurrent:
                            if len(crawled_docs) + processed_in_batch >= config.max_pages:
                                break
                            url, depth = to_visit.pop(0)
                            if url in visited: continue
                            visited.add(url)
                            current_batch.append((url, depth))
                            processed_in_batch +=1

                        if not current_batch: break
                        batch_urls = [item[0] for item in current_batch]
                        
                        from crawl4ai.dispatcher import MemoryAdaptiveDispatcher # Local import
                        dispatcher = MemoryAdaptiveDispatcher(
                            memory_threshold_percent=70.0, check_interval=1.0,
                            max_session_permit=len(batch_urls) # Can be config.max_concurrent
                        )
                        results = await crawler.arun_many(urls=batch_urls, config=run_config, dispatcher=dispatcher)

                        for (url, depth), r_item in zip(current_batch, results):
                            if r_item.success and r_item.markdown:
                                crawled_docs.append(self._process_crawl4ai_result(r_item))
                                if depth < config.max_depth and config.follow_links:
                                    links_to_add = []
                                    page_links = getattr(r_item, "links", [])
                                    if isinstance(page_links, dict): # Handle dict format from crawl4ai
                                        temp_links = []
                                        for link_list_val in page_links.values():
                                            if isinstance(link_list_val, list):
                                                temp_links.extend(link_list_val)
                                            elif isinstance(link_list_val, str):
                                                temp_links.append(link_list_val)
                                        page_links = temp_links


                                    for link_obj in page_links:
                                        link_url = None
                                        if isinstance(link_obj, str): link_url = link_obj
                                        elif isinstance(link_obj, dict) and 'href' in link_obj: link_url = link_obj['href']
                                        elif hasattr(link_obj, "url"): link_url = link_obj.url
                                        elif hasattr(link_obj, "href"): link_url = link_obj.href
                                        if not link_url: continue
                                        
                                        abs_link_url = urllib.parse.urljoin(url, link_url.strip())
                                        link_domain = urllib.parse.urlparse(abs_link_url).netloc
                                        if link_domain == base_domain and abs_link_url not in visited and (abs_link_url, depth +1) not in to_visit:
                                            links_to_add.append((abs_link_url, depth + 1))
                                    to_visit.extend(links_to_add) # Add to end for BFS-like behavior
            return crawled_docs
        except Exception as e:
            logger.error(f"Error using Crawl4AI: {e}\n{traceback.format_exc()}")
            logger.info("Falling back to built-in crawler")
            if config.method == CrawlMethod.SINGLE_PAGE: return await self._crawl_single_page(config)
            elif config.method == CrawlMethod.SITEMAP: return await self._crawl_sitemap(config)
            else: return await self._crawl_recursive(config)

    async def _crawl_single_page(self, config: CrawlConfig) -> List[Document]:
        try:
            logger.info(f"Crawling single page: {config.url}")
            async with httpx.AsyncClient(follow_redirects=True, timeout=config.timeout) as client:
                headers = {"User-Agent": config.user_agent}
                response = await client.get(config.url, headers=headers)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()

                if "text/plain" in content_type or \
                   any(config.url.endswith(ext) for ext in (".txt", ".md", ".text")):
                    logger.info(f"Detected text file: {config.url}")
                    text_content = response.text
                    title = Path(urllib.parse.urlparse(config.url).path).name or config.url
                    doc_metadata = {
                        "title": title, "url": config.url, "crawl_timestamp": time.time(),
                        "source": "web", "source_type": "text",
                        "domain": urllib.parse.urlparse(config.url).netloc,
                        "status_code": response.status_code, "content_type": content_type
                    }
                    return [Document(content=text_content, metadata=doc_metadata, source_url=config.url)]

                html_content = response.text
                soup = BeautifulSoup(html_content, "html.parser")
                title_tag = soup.find("title")
                title = title_tag.text.strip() if title_tag else (Path(urllib.parse.urlparse(config.url).path).name or config.url)
                
                for script_style in soup(["script", "style", "svg", "noscript", "iframe", "header", "footer", "nav", "aside"]):
                    script_style.extract()
                text_content = soup.get_text(separator=" ", strip=True)
                text_content = re.sub(r'\s\s+', ' ', text_content) # Normalize whitespace

                doc_metadata = {
                    "title": title, "url": config.url, "crawl_timestamp": time.time(),
                    "source": "web", "domain": urllib.parse.urlparse(config.url).netloc,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type", "")
                }
                return [Document(content=text_content, metadata=doc_metadata, source_url=config.url)]
        except Exception as e:
            logger.error(f"Error crawling single page {config.url}: {e}\n{traceback.format_exc()}")
            return []

    async def _parse_sitemap_async(self, url: str) -> List[str]:
        """Asynchronously parse a sitemap and extract URLs."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url)
                response.raise_for_status()
            content = response.content
            # The rest of the parsing logic from _parse_sitemap can be used here,
            # but if a sitemap index points to other sitemaps, those fetches should also be async.
            # For simplicity in this example, we'll call the synchronous _parse_sitemap_content here.
            # A more robust solution would make all HTTP requests within parsing async.
            return self._parse_sitemap_content(content, base_url=url) # Pass base_url for relative sitemap locs
        except Exception as e:
            logger.error(f"Error fetching sitemap {url}: {e}\n{traceback.format_exc()}")
            return []
            
    def _parse_sitemap_content(self, xml_content: bytes, base_url: str) -> List[str]:
        """Helper to parse XML content of a sitemap."""
        urls: List[str] = []
        try:
            root = ElementTree.fromstring(xml_content)
            namespace = '{http://www.sitemaps.org/schemas/sitemap/0.9}'
            
            # Check for sitemap index
            if root.tag == f'{namespace}sitemapindex':
                for sitemap_element in root.findall(f'{namespace}sitemap'):
                    loc_element = sitemap_element.find(f'{namespace}loc')
                    if loc_element is not None and loc_element.text:
                        # This is a URL to another sitemap. Recursively parse it.
                        # This part should ideally be async if called from an async context.
                        # For this structure, we'd collect these and process them outside or make _parse_sitemap_async truly recursive with async calls.
                        # Here, we're just illustrating parsing. A full async solution is more involved.
                        logger.warning(f"Sitemap index found pointing to: {loc_element.text}. Recursive async parsing of sitemap indexes not fully implemented in this synchronous helper.")
                        # To make this work, fetch loc_element.text and parse its content.
                        # For now, we'll just add it as if it were a page, which is incorrect for sitemap indexes.
                        # Correct approach: collect these URLs, then fetch and parse them asynchronously.
                        # urls.append(loc_element.text) # Incorrect for sitemapindex locs
            elif root.tag == f'{namespace}urlset':
                for url_element in root.findall(f'{namespace}url'):
                    loc_element = url_element.find(f'{namespace}loc')
                    if loc_element is not None and loc_element.text:
                        urls.append(urllib.parse.urljoin(base_url, loc_element.text.strip()))
            else:
                logger.warning(f"Unknown root tag in sitemap content from {base_url}: {root.tag}")

        except ElementTree.ParseError as e:
            logger.error(f"XML ParseError for sitemap content from {base_url}: {e}")
        except Exception as e:
            logger.error(f"Error parsing sitemap content from {base_url}: {e}\n{traceback.format_exc()}")
        return urls


    async def _crawl_sitemap(self, config: CrawlConfig) -> List[Document]:
        try:
            logger.info(f"Crawling sitemap: {config.url}")
            sitemap_page_urls = await self._parse_sitemap_async(config.url)
            if not sitemap_page_urls:
                logger.warning(f"No URLs found in sitemap {config.url}")
                return []

            if config.max_pages > 0 and len(sitemap_page_urls) > config.max_pages:
                sitemap_page_urls = sitemap_page_urls[:config.max_pages]
            logger.info(f"Found {len(sitemap_page_urls)} URLs in sitemap to process.")

            all_docs: List[Document] = []
            tasks = []
            semaphore = asyncio.Semaphore(config.max_concurrent)

            async def crawl_task(s_url):
                async with semaphore:
                    single_page_config = dataclasses.replace(config, url=s_url, method=CrawlMethod.SINGLE_PAGE)
                    return await self._crawl_single_page(single_page_config)

            for page_url in sitemap_page_urls:
                tasks.append(crawl_task(page_url))
            
            results_list_of_lists = await asyncio.gather(*tasks, return_exceptions=True)
            
            for item_result in results_list_of_lists:
                if isinstance(item_result, list):
                    all_docs.extend(item_result)
                elif isinstance(item_result, Exception):
                    logger.error(f"Error in sitemap page crawl task: {item_result}")
            return all_docs
        except Exception as e:
            logger.error(f"Error crawling sitemap {config.url}: {e}\n{traceback.format_exc()}")
            return []

    async def _crawl_recursive(self, config: CrawlConfig) -> List[Document]:
        try:
            logger.info(f"Crawling recursively: {config.url} (depth: {config.max_depth}, pages: {config.max_pages})")
            base_domain = urllib.parse.urlparse(config.url).netloc
            visited: Set[str] = set()
            # Use a list of tuples for BFS-like behavior: (url, current_depth)
            to_visit: List[Tuple[str, int]] = [(config.url, 1)]
            all_docs: List[Document] = []
            
            # Using a semaphore for concurrency within the recursive crawl.
            # This means multiple pages at the current level of `to_visit` can be processed concurrently.
            semaphore = asyncio.Semaphore(config.max_concurrent)

            async def process_url(url_to_process: str, current_depth: int):
                if url_to_process in visited or len(all_docs) >= config.max_pages:
                    return None, [] # Return None for doc, empty list for new links

                async with semaphore: # Limit concurrent processing of individual pages
                    if url_to_process in visited or len(all_docs) >= config.max_pages: # Re-check after acquiring semaphore
                        return None, []
                    
                    visited.add(url_to_process)
                    logger.debug(f"Recursively processing {url_to_process} at depth {current_depth}")
                    
                    single_page_config = dataclasses.replace(config, url=url_to_process, method=CrawlMethod.SINGLE_PAGE)
                    # _crawl_single_page returns a list of Documents (usually one)
                    page_docs = await self._crawl_single_page(single_page_config)
                    
                    new_links_to_explore: List[Tuple[str,int]] = []
                    if page_docs: # If page was successfully crawled
                        doc = page_docs[0] # Assuming one doc per single page crawl
                        
                        if current_depth < config.max_depth and config.follow_links:
                            # Extract links from this page's content (if it was HTML)
                            # Need to parse HTML from doc.content or have _crawl_single_page return soup
                            try:
                                soup = BeautifulSoup(doc.content, "html.parser") # Re-parse if content is HTML text
                                for link_tag in soup.find_all("a", href=True):
                                    href = link_tag.get("href", "").strip()
                                    if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
                                        continue
                                    
                                    abs_link = urllib.parse.urljoin(url_to_process, href)
                                    # Basic fragment and query param removal for visited check simplicity
                                    abs_link_normalized = urllib.parse.urlunparse(urllib.parse.urlparse(abs_link)._replace(fragment="", query=""))

                                    link_domain = urllib.parse.urlparse(abs_link).netloc
                                    if link_domain == base_domain and abs_link_normalized not in visited:
                                        new_links_to_explore.append((abs_link_normalized, current_depth + 1))
                            except Exception as e:
                                logger.warning(f"Could not parse links from {url_to_process}: {e}")
                        return doc, new_links_to_explore
                    return None, [] # No doc if crawl failed

            while to_visit and len(all_docs) < config.max_pages:
                # Take a batch of URLs to process concurrently from the current to_visit queue
                # This isn't strictly level by level for BFS due to async nature, but approximates it.
                batch_to_process = []
                while to_visit and len(batch_to_process) < config.max_concurrent : # Form a batch
                    if len(all_docs) + len(batch_to_process) >= config.max_pages: break
                    batch_to_process.append(to_visit.pop(0))

                if not batch_to_process: break

                tasks = [process_url(u, d) for u, d in batch_to_process]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for item_result in results:
                    if isinstance(item_result, Exception):
                        logger.error(f"Error in recursive processing task: {item_result}")
                        continue
                    if item_result is None: # Should not happen if process_url always returns tuple
                        continue

                    doc_processed, links_found = item_result
                    if doc_processed and len(all_docs) < config.max_pages:
                        all_docs.append(doc_processed)
                    
                    for link_info in links_found:
                        # Add to to_visit if not already scheduled and not visited
                        # Check visited again because multiple tasks might identify the same link
                        if link_info[0] not in visited and link_info not in to_visit:
                             # A simple check to avoid adding duplicates to to_visit if already present
                            is_scheduled = any(scheduled_url == link_info[0] for scheduled_url, _ in to_visit)
                            if not is_scheduled:
                                to_visit.append(link_info)
            return all_docs
        except Exception as e:
            logger.error(f"Error in recursive crawl of {config.url}: {e}\n{traceback.format_exc()}")
            return []

    def _process_crawl4ai_result(self, crawl_result: Any) -> Document: # crawl_result is crawl4ai.models.CrawlResult
        url = getattr(crawl_result, "url", "unknown_url")
        content = ""
        if hasattr(crawl_result, "markdown") and crawl_result.markdown:
            if hasattr(crawl_result.markdown, "fit_markdown") and crawl_result.markdown.fit_markdown:
                content = crawl_result.markdown.fit_markdown
            elif hasattr(crawl_result.markdown, "raw_markdown") and crawl_result.markdown.raw_markdown:
                content = crawl_result.markdown.raw_markdown
            elif isinstance(crawl_result.markdown, str): # Fallback if markdown is just a string
                content = crawl_result.markdown
        elif hasattr(crawl_result, "html") and crawl_result.html: # Fallback to HTML if no markdown
            try:
                soup = BeautifulSoup(crawl_result.html, "html.parser")
                for script_style in soup(["script", "style", "svg", "noscript", "iframe", "header", "footer", "nav", "aside"]):
                    script_style.extract()
                content = soup.get_text(separator=" ", strip=True)
                content = re.sub(r'\s\s+', ' ', content)
            except Exception as e:
                logger.warning(f"Error parsing HTML from crawl4ai result for {url}: {e}")
                content = crawl_result.html # Use raw HTML as last resort
        elif hasattr(crawl_result, "text") and crawl_result.text: # Further fallback to plain text
             content = crawl_result.text


        title = getattr(crawl_result.metadata, "title", "") if hasattr(crawl_result, "metadata") and crawl_result.metadata else \
                getattr(crawl_result, "title", Path(urllib.parse.urlparse(url).path).name or url)

        # Basic header extraction from markdown
        headers_extracted = "; ".join([f"{h[0]} {h[1]}" for h in re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)])

        doc_metadata = {
            "title": title, "url": url, "crawl_timestamp": time.time(),
            "source": "web", "domain": urllib.parse.urlparse(url).netloc,
            "content_type": "markdown" if (hasattr(crawl_result, "markdown") and crawl_result.markdown) else "html",
            "headers": headers_extracted,
            "char_count": len(content), "word_count": len(content.split())
        }
        return Document(content=content, metadata=doc_metadata, source_url=url)

    def _make_serializable(self, obj: Any) -> Dict[str, Any]:
        if isinstance(obj, Document):
            return {
                "content": obj.content,
                "metadata": obj.metadata, # Assumes metadata is already serializable
                "source_url": obj.source_url,
                "source_file": obj.source_file,
                "source_repo": obj.source_repo,
                "id": obj.id
            }
        elif dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        elif hasattr(obj, "__dict__"): # For general objects
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_") and self._is_json_serializable(v)}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        else:
            return str(obj) # Fallback

    def _is_json_serializable(self, value: Any) -> bool:
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False

# Example usage (optional, for testing the crawler directly)
async def main_test():
    crawler = WebCrawler()
    
    # Test single page (text file)
    # result_text_file = await crawler.ingest(url="https://raw.githubusercontent.com/ai-pydantic/pydantic-ai/main/README.md", method=CrawlMethod.SINGLE_PAGE)
    # print(f"Text File Crawl Result: Status: {result_text_file.status}, Message: {result_text_file.message}, Pages: {result_text_file.details.get('pages_crawled')}")
    # if result_text_file.status == IngestionStatus.SUCCESS and result_text_file.details.get('raw_output_path'):
    #     with open(result_text_file.details['raw_output_path'], 'r') as f:
    #         print("First doc content sample (text file):", json.load(f)[0]['content'][:200])


    # Test single page (HTML)
    # result_single = await crawler.ingest(url="https://example.com", method=CrawlMethod.SINGLE_PAGE)
    # print(f"Single Page Crawl Result: Status: {result_single.status}, Message: {result_single.message}, Pages: {result_single.details.get('pages_crawled')}")
    # if result_single.status == IngestionStatus.SUCCESS and result_single.details.get('raw_output_path'):
    #     with open(result_single.details['raw_output_path'], 'r') as f:
    #          print("First doc content sample (html):", json.load(f)[0]['content'][:200])

    # Test sitemap (using a known public sitemap, ensure it's allowed by robots.txt for testing)
    # result_sitemap = await crawler.ingest(url="https://www.sitemaps.org/sitemap.xml", method=CrawlMethod.SITEMAP, max_pages=5) # Fictional sitemap for example
    # print(f"Sitemap Crawl Result: Status: {result_sitemap.status}, Message: {result_sitemap.message}, Pages: {result_sitemap.details.get('pages_crawled')}")

    # Test recursive (use with caution on external sites, be respectful)
    # result_recursive = await crawler.ingest(url="YOUR_TEST_SITE_HERE", method=CrawlMethod.RECURSIVE, max_depth=1, max_pages=3)
    # print(f"Recursive Crawl Result: Status: {result_recursive.status}, Message: {result_recursive.message}, Pages: {result_recursive.details.get('pages_crawled')}")

if __name__ == "__main__":
    # Ensure an event loop is running if testing directly
    # Python 3.7+
    # asyncio.run(main_test())
    # For older versions or environments like Jupyter:
    loop = asyncio.get_event_loop()
    if not loop.is_running():
         # loop.run_until_complete(main_test()) # Comment out if not testing directly
         pass
    else:
        print("Asyncio loop already running. Cannot run main_test() directly in this context (e.g. Jupyter).")