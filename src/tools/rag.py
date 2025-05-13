from . import mcp

from mcp.server.fastmcp import Context

from src.server import MCP_Response 

from src.service.supabase import search_documents


@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.
    
    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.

    Use the tool to get source domains if the user is asking to use a specific tool or framework.
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string representation of an MCP_Response object.
    """
    tool_name = "perform_rag_query"
    try:
        supabase_client_from_ctx = ctx.request_context.lifespan_context.supabase_client
        filter_metadata = {"source": source} if source and source.strip() else None

        print(filter_metadata)
        
        results = search_documents(
            client=supabase_client_from_ctx,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
        
        formatted_results = [
            {
                "url": r.get("url"),
                "content": r.get("content"),
                "metadata": r.get("metadata"),
                "similarity": r.get("similarity")
            } for r in results
        ]
        response_data = {
            "query": query,
            "source_filter": source,
            "results": formatted_results,
            "count": len(formatted_results)
        }
        return MCP_Response(success=True, message="RAG query performed successfully.", data=response_data, tool_name=tool_name).to_json_str()
    except Exception as e:
        import traceback
        print(f"Error in tool '{tool_name}': {e}\n{traceback.format_exc()}")
        return MCP_Response(success=False, message=f"An error occurred in {tool_name}", error=str(e), data={"query": query, "source_filter": source}, tool_name=tool_name).to_json_str()


@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources based on unique source metadata values.
    
    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database. This is useful for discovering what content is available for querying.
    
    Args:
        ctx: The MCP server provided context
    
    Returns:
        JSON string representation of an MCP_Response object with the list of available sources.
    """
    tool_name = "get_available_sources"
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        result = supabase_client.from_('crawled_pages')\
            .select('metadata')\
            .not_.is_('metadata->>source', 'null')\
            .execute()
            
        unique_sources = set()
        
        if result.data:
            for item in result.data:
                source = item.get('metadata', {}).get('source')
                if source:
                    unique_sources.add(source)
        
        sources_list = sorted(list(unique_sources))
        
        response_data = {
            "sources": sources_list,
            "count": len(sources_list)
        }
        return MCP_Response(success=True, message="Successfully retrieved available sources.", data=response_data, tool_name=tool_name).to_json_str()
    except Exception as e:
        import traceback
        print(f"Error in tool '{tool_name}': {e}\n{traceback.format_exc()}")
        return MCP_Response(success=False, message=f"An error occurred in {tool_name}", error=str(e), tool_name=tool_name).to_json_str()


@mcp.tool()
async def perform_domo_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Perform a query using the Domo search function on the stored content.
    
    This tool searches the Domo database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string representation of an MCP_Response object.
    """
    tool_name = "perform_domo_query"
    try:
        domo_client_from_ctx = ctx.request_context.lifespan_context.domo_client
        filter_metadata = {"source": source} if source and source.strip() else None

        print(filter_metadata)
        
        results = domo_client_from_ctx.search_documents(
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
        formatted_results = [
            {
                "url": r.get("url"),
                "content": r.get("content"),
                "metadata": r.get("metadata"),
                "similarity": r.get("similarity")
            } for r in results
        ]
        response_data = {
            "query": query,
            "source_filter": source,
            "results": formatted_results,
            "count": len(formatted_results)
        }
        return MCP_Response(success=True, message="Domo query performed successfully.", data=response_data, tool_name=tool_name).to_json_str()
    except Exception as e:
        import traceback
        print(f"Error in tool '{tool_name}': {e}\n{traceback.format_exc()}")
        return MCP_Response(success=False, message=f"An error occurred in {tool_name}", error=str(e), data={"query": query, "source_filter": source}, tool_name=tool_name).to_json_str()
