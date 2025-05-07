"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""

from src.crawler_github import crawl_github_repository_async, is_github_repository, extract_repo_info
from src.crawler_crawl4ai import crawl_markdown_file, crawl_batch, crawl_recursive_internal_links
from src.service.supabase import get_supabase_client, add_documents_to_supabase, search_documents
from src.server import get_mcp_server, crawl4ai_lifespan

from src.utils import (
    extract_section_info, 
    smart_chunk_markdown, 
    is_txt, 
    is_sitemap,
    parse_sitemap
)

from crawl4ai import CrawlerRunConfig, CacheMode
from mcp.server.fastmcp import Context

import os
from typing import Optional
from pathlib import Path

import json
from dotenv import load_dotenv
from urllib.parse import urlparse
import asyncio
from functools import partial


# Load environment variables from the project root .env file
# project_root = Path(__file__).resolve().parent.parent
dotenv_path = '.env'
assert load_dotenv(dotenv_path, override=True)



 # Initialize Supabase client
supabase_client = get_supabase_client(url=os.environ['SUPABASE_URL'],
                                      key= os.environ['SUPABASE_SERVICE_KEY'])

partial_crawl4ai_lifespan = partial(crawl4ai_lifespan, supabase_client= supabase_client)


mcp = get_mcp_server(
    host = os.environ['MCP_HOST'],
    port = os.environ['MCP_PORT'],
    crawl4ai_lifespan=partial_crawl4ai_lifespan
)

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase.
    
    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
    
    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)
            
            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = urlparse(url).netloc
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
            
            # Add to Supabase
            add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas)
            
            return json.dumps({
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "content_length": len(result.markdown),
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": result.error_message
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.
    
    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth
    
    All crawled content is chunked and stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 1000)
    
    Returns:
        JSON string with crawl summary and storage information
    """
    try:
        # Get the crawler and Supabase client from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        crawl_results = []
        crawl_type = "webpage"
        
        # Detect URL type and use appropriate crawl method
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No URLs found in sitemap"
                }, indent=2)
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            crawl_type = "webpage"
        
        if not crawl_results:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)
        
        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0
        
        for doc in crawl_results:
            source_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            
            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = urlparse(source_url).netloc
                meta["crawl_type"] = crawl_type
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
                
                chunk_count += 1
        
        # Add to Supabase
        batch_size = 20
        add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, batch_size=batch_size)
        
        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "chunks_stored": chunk_count,
            "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else [])
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources based on unique source metadata values.
    
    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database. This is useful for discovering what content is available for querying.
    
    Args:
        ctx: The MCP server provided context
    
    Returns:
        JSON string with the list of available sources
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Use a direct query with the Supabase client
        # This could be more efficient with a direct Postgres query but
        # I don't want to require users to set a DB_URL environment variable as well
        result = supabase_client.from_('crawled_pages')\
            .select('metadata')\
            .not_.is_('metadata->>source', 'null')\
            .execute()
            
        # Use a set to efficiently track unique sources
        
        unique_sources = set()
        
        # Extract the source values from the result using a set for uniqueness
        if result.data:
            for item in result.data:
                source = item.get('metadata', {}).get('source')
                if source:
                    unique_sources.add(source)
        
        # Convert set to sorted list for consistent output
        sources = sorted(list(unique_sources))
        
        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

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
        JSON string with the search results
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}
        
        # Perform the search
        results = search_documents(
            client=supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            })
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def crawl_github_repo(
    ctx: Context, 
    repo_url: str, 
    max_depth: int = 5, # Note: max_depth is less relevant for ZIP download method
    branch_name: Optional[str] = None, 
    chunk_size: int = 5000,
    save_files_locally: bool = True,
    local_save_dir: str = "EXPORT_GITHUB"
) -> str:
    """
    Crawls a GitHub repository by downloading it as a ZIP, extracts file contents, 
    stores them in Supabase, and saves raw files locally.
    // ... (args description remains the same)
    """
    try:
        if not is_github_repository(repo_url):
            return json.dumps({
                "success": False,
                "repo_url": repo_url,
                "error": "Invalid GitHub repository URL provided."
            }, indent=2)

        # crawler = ctx.request_context.lifespan_context.crawler # No longer needed to pass to async func
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        github_auth_token = os.getenv("GITHUB_TOKEN")
        
        # Extract owner and repo name for metadata and local save path
        try:
            owner, repo, _, _ = extract_repo_info(repo_url)
            # Construct a more specific save directory for this repo
            repo_specific_save_dir = Path(local_save_dir) / owner / repo
        except ValueError:
            # Fallback if extract_repo_info fails
            owner, repo = "unknown_owner", "unknown_repo"
            repo_specific_save_dir = Path(local_save_dir) / "unknown_repo"


        print(f"Starting GitHub repository crawl for: {repo_url} using ZIP download method.")
        
        crawled_files_data = await crawl_github_repository_async(
            # crawler=crawler, # REMOVE THIS LINE - crawler is no longer an argument
            repo_url=repo_url,
            max_depth=max_depth, # Kept for consistency, though less used by ZIP method
            branch_name=branch_name,
            # max_concurrent is not used by the ZIP download method's async function
            broadcast_progress=None, 
            save_raw_content=save_files_locally,
            output_dir_for_raw=str(repo_specific_save_dir),
            github_token=github_auth_token # Pass the token
        )
        
        if not crawled_files_data:
            return json.dumps({
                "success": True, # Crawl might be successful but find no processable files
                "repo_url": repo_url,
                "message": "No files found or processed in the repository.",
                "files_processed": 0,
                "chunks_stored": 0,
                "files_saved_locally": 0
            }, indent=2)
            
        urls_db = []
        chunk_numbers = []
        contents_db = []
        metadatas_db = []
        
        processed_files_count = 0
        total_chunks_stored = 0
        locally_saved_files_count = 0

        # Define text-based content types to process for Supabase
        # This list can be expanded.
        text_content_types_prefixes = [
            'text/', 'application/json', 'application/xml', 'application/javascript', 
            'application/x-python', # Common for .py if not text/x-python
        ]

        for file_data in crawled_files_data:
            if file_data.get('success') and file_data.get('content'):
                processed_files_count += 1
                
                # Check if file was saved by crawl_github_repository_async's internal logic
                # This is an approximation; the function logs saving but doesn't return paths per file.
                # We assume if save_files_locally was true and the file was fetched, it was attempted to be saved.
                if save_files_locally:
                     locally_saved_files_count +=1


                content_type = file_data.get('content_type', 'application/octet-stream').lower()
                
                # Determine if content is text-based and suitable for chunking/embedding
                is_text_content = any(content_type.startswith(prefix) for prefix in text_content_types_prefixes)
                
                if is_text_content:
                    file_content_text = file_data['content']
                    if isinstance(file_content_text, bytes):
                        try:
                            file_content_text = file_content_text.decode('utf-8')
                        except UnicodeDecodeError:
                            print(f"Could not decode file {file_data['title']} as UTF-8, skipping for Supabase.")
                            continue # Skip this file for Supabase if not decodable text

                    chunks = smart_chunk_markdown(file_content_text, chunk_size=chunk_size) # Using existing chunker
                    
                    for i, chunk in enumerate(chunks):
                        urls_db.append(file_data['url']) # HTML URL of the file on GitHub
                        chunk_numbers.append(i)
                        contents_db.append(chunk)
                        
                        meta = extract_section_info(chunk) # Basic metadata from chunk
                        meta["chunk_index"] = i
                        meta["url"] = file_data['url']
                        meta["source"] = urlparse(repo_url).netloc # e.g., github.com
                        meta["repo_url"] = repo_url
                        meta["file_path_in_repo"] = file_data.get('api_path', file_data['title'])
                        meta["file_name"] = file_data['title']
                        meta["content_type"] = content_type
                        meta["crawl_tool"] = "crawl_github_repo"
                        meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__) # Or use datetime
                        metadatas_db.append(meta)
                        total_chunks_stored += 1
                else:
                    print(f"Skipping Supabase storage for non-text content: {file_data['title']} (type: {content_type})")

        if contents_db:
            add_documents_to_supabase(supabase_client, urls_db, chunk_numbers, contents_db, metadatas_db)
            
        return json.dumps({
            "success": True,
            "repo_url": repo_url,
            "files_discovered": len(crawled_files_data),
            "files_processed_for_supabase": processed_files_count, # Files that had content and were successful
            "chunks_stored_in_supabase": total_chunks_stored,
            "files_saved_locally": locally_saved_files_count if save_files_locally else 0,
            "local_save_directory_base": str(repo_specific_save_dir) if save_files_locally else None
        }, indent=2)
        
    except Exception as e:
        print(f"Error in crawl_github_repo for {repo_url}: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({
            "success": False,
            "repo_url": repo_url,
            "error": str(e)
        }, indent=2)

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