from src.service.domo import DomoClient

from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from supabase import Client
from crawl4ai import AsyncWebCrawler, BrowserConfig

from typing import Optional, Dict, Any, List, Annotated

import datetime as dt

from pydantic.functional_serializers import PlainSerializer

def format_datetime(dt: dt.datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

CustomDateTime = Annotated[
    dt.datetime, PlainSerializer(format_datetime, return_type=str)
]

class MCP_Response(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tool_name: Optional[str] = None # To identify which tool generated the response

    def to_json_str(self) -> str:
        # Ensure indent is used and None values are excluded for cleaner output
        return self.model_dump_json(indent=2, exclude_none=True)

class CrawlerMetadata(BaseModel):
    """Standardized metadata for documents stored in Supabase."""
    # Core metadata
    chunk_index: Optional[int] = None  # Made optional
    url: str  # Original URL of the content unit (page, file, etc.)
    source_domain: str  # e.g., "example.com", "github.com"
    crawled_at: CustomDateTime # Timestamp of when crawling/chunking occurred
    crawler_tool: str # Name of the specific mcp tool (e.g., "crawl_single_page")

    # Content-derived metadata (from extract_section_info or file properties)
    document_title: Optional[str] = None # Title of the page or document
    document_description: Optional[str] = None
    document_keywords: Optional[List[str]] = None
    content_headings: Optional[List[str]] = None # Headings within the chunk or document

    # Source-specific metadata
    file_name: Optional[str] = None # For file-based crawls (GitHub, .txt)
    file_path_in_source: Optional[str] = None # e.g., path in GitHub repo, path in website
    source_repo_url: Optional[str] = None # For GitHub crawls, the main repository URL
    content_type: Optional[str] = None # MIME type, if applicable

    # To allow for any other specific metadata not covered
    additional_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation suitable for Supabase, excluding None values."""
        # Pydantic's model_dump handles datetime serialization to ISO format by default if the model field is datetime
        # and the target (like JSON) requires string. Supabase client might handle datetime objects directly.
        return self.model_dump(exclude_none=True)

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client
    domo_client: DomoClient
    
@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP, supabase_client, domo_client) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    
    try:

        print(supabase_client.supabase_url, supabase_client.supabase_key)
        print(domo_client.domo_base_url, domo_client.index_id)
       
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client,
            domo_client=domo_client
        )
    finally:
        # Clean up the crawler
        await crawler.__aexit__(None, None, None)

# Initialize FastMCP server
def get_mcp_server(host, port, crawl4ai_lifespan) -> FastMCP:
    """
    Initializes the FastMCP server with the Crawl4AI context.
    
    Returns:
        FastMCP: The initialized FastMCP server
    """
    
    # Create the MCP server with the lifespan context manager
    return FastMCP(
        "mcp-crawl4ai-rag",
        description="MCP server for RAG and web crawling with Crawl4AI",
        lifespan=crawl4ai_lifespan,
        host=host,
        port=port)
