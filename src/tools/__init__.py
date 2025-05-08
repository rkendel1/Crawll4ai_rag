
# Note: The 'mcp' instance will be defined in mcp_server.py.
# The @mcp.tool() decorator will work when these functions are imported 
# into mcp_server.py where 'mcp' is in scope.

# Placeholder for the mcp instance to allow @mcp.tool() decorator.
# This will be replaced by the actual mcp instance when imported.
class PlaceholderMCP:
    def tool(self):
        def decorator(func):
            # Store the original function, it will be registered later
            # when tools.py is imported into mcp_server.py
            func._is_mcp_tool = True 
            return func
        return decorator

mcp = PlaceholderMCP() 

from .crawl import smart_crawl, crawl_single_page, crawl_github_repo, crawl_text_file_tool, crawl_sitemap_tool,crawl_recursive_webpages_tool
from .rag import perform_rag_query, get_available_sources