"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""

# Keep necessary imports for server setup and context
import os

import asyncio
from functools import partial
from dotenv import load_dotenv

from src.service.supabase import get_supabase_client
from src.server import get_mcp_server, crawl4ai_lifespan


dotenv_path = '.env' 
load_dotenv(dotenv_path, override=True)

print( os.environ['SUPABASE_URL'], os.environ['SUPABASE_SERVICE_KEY'])

# Create a partial function for the lifespan manager, injecting the supabase client
partial_crawl4ai_lifespan = partial(crawl4ai_lifespan, 
                                    supabase_client=get_supabase_client(
                                        url=os.environ['SUPABASE_URL'],
                                        key=os.environ['SUPABASE_SERVICE_KEY']
))

# Get the MCP server instance
mcp = get_mcp_server(
    host=os.environ['MCP_HOST'], # Use .get for defaults
    port=int(os.environ['MCP_PORT']), # Ensure port is int
    crawl4ai_lifespan=partial_crawl4ai_lifespan
)

import src.tools  as tools
from src.tools import (smart_crawl, 
                             crawl_single_page, 
                             crawl_github_repo,
                             crawl_text_file_tool, 
                             crawl_sitemap_tool,
                             crawl_recursive_webpages_tool, 
                             perform_rag_query,
                             get_available_sources)
tools.mcp = mcp 

mcp.add_tool(tools.smart_crawl)
mcp.add_tool(tools.crawl_single_page)
mcp.add_tool(tools.crawl_github_repo)
mcp.add_tool(tools.crawl_text_file_tool)
mcp.add_tool(tools.crawl_recursive_webpages_tool)
mcp.add_tool(tools.perform_rag_query)
mcp.add_tool(tools.get_available_sources)
mcp.add_tool(tools.crawl_sitemap_tool)

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())