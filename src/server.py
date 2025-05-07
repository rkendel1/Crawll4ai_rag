from src.service.supabase import get_supabase_client

from mcp.server.fastmcp import FastMCP
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from supabase import Client
from crawl4ai import AsyncWebCrawler, BrowserConfig



# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client
    
@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP, supabase_client) -> AsyncIterator[Crawl4AIContext]:
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
       
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client
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
    