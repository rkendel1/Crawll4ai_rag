from . import mcp
from src.server import MCP_Response, CrawlerMetadata

import os
import datetime
import asyncio
import traceback 
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import Context
from crawl4ai import CrawlerRunConfig, CacheMode


# Import functions from your other source files

from src.service.github import crawl_github_repository_async, is_github_repository, extract_repo_info

from src.service.crawl4ai import crawl_markdown_file, crawl_batch, crawl_recursive_internal_links
from src.service.supabase import add_documents_to_supabase
from src.utils import (
    extract_section_info, 
    smart_chunk_markdown, 
    is_txt, 
    is_sitemap,
    parse_sitemap
)

# --- Individual Crawling Tools ---

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase.
    """
    tool_name = "crawl_single_page"
    try:
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            chunks = smart_chunk_markdown(result.markdown)
            urls_db, chunk_numbers, contents, metadatas = [], [], [], []
            
            for i, chunk in enumerate(chunks):
                urls_db.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                section_info = extract_section_info(chunk)
                meta = CrawlerMetadata(
                    chunk_index=i,
                    url=url,
                    source_domain=urlparse(url).netloc,
                    crawled_at=datetime.datetime.now(datetime.timezone.utc),
                    crawler_tool=tool_name,
                    document_title=section_info.get("title"),
                    document_description=section_info.get("description"),
                    document_keywords=section_info.get("keywords"),
                    content_headings=section_info.get("headings")
                )
                metadatas.append(meta.to_supabase_dict())
            
            if contents:
                add_documents_to_supabase(supabase_client, urls_db, chunk_numbers, contents, metadatas)
            
            response_data = {
                "url": url, "chunks_stored": len(chunks), "content_length": len(result.markdown),
                "links_count": {"internal": len(result.links.get("internal", [])), "external": len(result.links.get("external", []))}
            }
            return MCP_Response(success=True, message="Page crawled and content stored.", data=response_data, tool_name=tool_name).to_json_str()
        else:
            return MCP_Response(success=False, message=f"Failed to crawl page: {url}", error=result.error_message, data={"url": url}, tool_name=tool_name).to_json_str()
    except Exception as e:
        print(f"Error in tool '{tool_name}': {e}\n{traceback.format_exc()}")
        return MCP_Response(success=False, message=f"An error occurred in {tool_name} for {url}", error=str(e), data={"url": url}, tool_name=tool_name).to_json_str()

@mcp.tool()
async def crawl_github_repo(
    ctx: Context, 
    repo_url: str, 
    branch_name: Optional[str] = None, 
    chunk_size: int = 5000, 
    save_files_locally: bool = True, 
    local_save_dir: str = "EXPORT_GITHUB"
) -> str:
    """
    Crawls a GitHub repository by downloading it as a ZIP, extracts file contents, 
    stores them in Supabase, and saves raw files locally.
    """
    tool_name = "crawl_github_repo"
    try:
        if not is_github_repository(repo_url):
            return MCP_Response(success=False, message="Invalid GitHub repository URL provided.", data={"repo_url": repo_url}, tool_name=tool_name).to_json_str()

        supabase_client = ctx.request_context.lifespan_context.supabase_client
        github_auth_token = os.getenv("GITHUB_TOKEN")
        
        owner, repo = "unknown_owner", "unknown_repo"
        try:
            owner, repo, _, _ = extract_repo_info(repo_url)
        except ValueError:
            path_segments = urlparse(repo_url).path.strip('/').split('/')
            if len(path_segments) >= 2:
                owner, repo = path_segments[0], path_segments[1]
        
        repo_specific_save_dir = Path(local_save_dir) / owner / repo

        print(f"Starting GitHub repository crawl for: {repo_url} using ZIP download method.")
        
        crawled_files_data = await crawl_github_repository_async(
            repo_url=repo_url,
            branch_name=branch_name,
            broadcast_progress=None, 
            save_raw_content=save_files_locally,
            output_dir_for_raw=str(repo_specific_save_dir),
            github_token=github_auth_token
        )
        
        if not crawled_files_data:
            response_data = {"repo_url": repo_url, "files_processed": 0, "chunks_stored": 0, "files_saved_locally": 0}
            return MCP_Response(success=True, message="No files found or processed in the repository.", data=response_data, tool_name=tool_name).to_json_str()
            
        urls_db, chunk_numbers, contents_db, metadatas_db = [], [], [], []
        processed_files_count, total_chunks_stored, locally_saved_files_count = 0, 0, 0
        
        text_content_types_prefixes = [
            'text/', 'application/json', 'application/xml', 'application/javascript', 
            'application/x-python', 'application/x-sh', 'application/x-csh', 
            'application/rtf', 'application/csv'
        ]
        text_file_extensions = [
            '.md', '.txt', '.py', '.js', '.json', '.yaml', '.yml', '.toml', '.xml', 
            '.css', '.html', '.htm', '.rst', '.sh', '.ps1', '.bat', '.rb', '.java',
            '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.php', '.pl', '.swift', '.kt',
            '.kts', '.scala', '.sql', '.r', '.tex', '.log', '.ini', '.cfg', '.conf',
            '.properties', '.env', '.dockerfile', 'dockerfile', '.gitignore', '.gitattributes',
            '.csv', '.tsv'
        ]

        for file_data in crawled_files_data:
            if file_data.get('success') and file_data.get('content'):
                processed_files_count += 1
                if save_files_locally and file_data.get('local_path'):
                     locally_saved_files_count +=1

                content_type = file_data.get('content_type', 'application/octet-stream').lower()
                file_name_lower = file_data.get('title', '').lower()
                
                is_text_content = any(content_type.startswith(prefix) for prefix in text_content_types_prefixes) or \
                                  any(file_name_lower.endswith(ext) for ext in text_file_extensions)

                if is_text_content:
                    file_content_text = file_data['content']
                    if isinstance(file_content_text, bytes):
                        try:
                            file_content_text = file_content_text.decode('utf-8', errors='replace')
                        except UnicodeDecodeError:
                            print(f"Could not decode file {file_data['title']} as UTF-8, skipping for Supabase.")
                            continue

                    chunks = smart_chunk_markdown(file_content_text, chunk_size=chunk_size)
                    
                    for i, chunk in enumerate(chunks):
                        urls_db.append(file_data['url']) # This is the file's URL from GitHub API
                        chunk_numbers.append(i)
                        contents_db.append(chunk)
                        
                        section_info = extract_section_info(chunk)
                        meta = CrawlerMetadata(
                            chunk_index=i,
                            url=file_data.get('html_url', file_data['url']), # Prefer html_url if available
                            source_domain=urlparse(repo_url).netloc,
                            crawled_at=datetime.datetime.now(datetime.timezone.utc),
                            crawler_tool=tool_name,
                            document_title=section_info.get("title") or file_data['title'],
                            document_description=section_info.get("description"),
                            document_keywords=section_info.get("keywords"),
                            content_headings=section_info.get("headings"),
                            file_name=file_data['title'],
                            # Assuming 'api_path' is the relative path in repo, or use 'path' if available from file_data
                            file_path_in_source=file_data.get('path', file_data.get('api_path', file_data['title'])), 
                            source_repo_url=repo_url,
                            content_type=content_type
                        )
                        metadatas_db.append(meta.to_supabase_dict())
                        total_chunks_stored += 1
                else:
                    print(f"Skipping Supabase storage for non-text/binary content: {file_data['title']} (type: {content_type})")

        if contents_db:
            add_documents_to_supabase(supabase_client, urls_db, chunk_numbers, contents_db, metadatas_db)
            
        response_data = {
            "repo_url": repo_url, "files_discovered": len(crawled_files_data), 
            "files_processed_for_supabase": processed_files_count,
            "chunks_stored_in_supabase": total_chunks_stored, 
            "files_saved_locally": locally_saved_files_count if save_files_locally else 0,
            "local_save_directory_base": str(repo_specific_save_dir) if save_files_locally else None
        }
        return MCP_Response(success=True, message="GitHub repository processed.", data=response_data, tool_name=tool_name).to_json_str()
        
    except Exception as e:
        print(f"Error in tool '{tool_name}' for {repo_url}: {e}\n{traceback.format_exc()}")
        return MCP_Response(success=False, message=f"An error occurred in {tool_name} for {repo_url}", error=str(e), data={"repo_url": repo_url}, tool_name=tool_name).to_json_str()


@mcp.tool()
async def crawl_text_file_tool(ctx: Context, url: str, chunk_size: int = 5000) -> str:
    """
    Crawls a single text file (e.g., llms.txt) and stores its content in Supabase.
    """
    tool_name = "crawl_text_file_tool"
    try:
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        crawl_results = await crawl_markdown_file(crawler, url) 

        if not crawl_results or not crawl_results[0].get('markdown'):
            return MCP_Response(success=False, message="No content found in text file.", data={"url": url}, tool_name=tool_name).to_json_str()

        doc = crawl_results[0]
        source_url = doc['url']
        md = doc['markdown']
        file_name_from_url = Path(urlparse(source_url).path).name
        chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
        
        urls_db, chunk_numbers, contents, metadatas = [], [], [], []
        for i, chunk in enumerate(chunks):
            urls_db.append(source_url)
            chunk_numbers.append(i)
            contents.append(chunk)
            section_info = extract_section_info(chunk) # Text files might not have rich section info
            meta = CrawlerMetadata(
                chunk_index=i,
                url=source_url,
                source_domain=urlparse(source_url).netloc,
                crawled_at=datetime.datetime.now(datetime.timezone.utc),
                crawler_tool=tool_name,
                document_title=section_info.get("title") or file_name_from_url,
                document_description=section_info.get("description"),
                document_keywords=section_info.get("keywords"),
                content_headings=section_info.get("headings"),
                file_name=file_name_from_url,
                file_path_in_source=urlparse(source_url).path
            )
            metadatas.append(meta.to_supabase_dict())
        
        if contents:
            add_documents_to_supabase(supabase_client, urls_db, chunk_numbers, contents, metadatas)

        response_data = {"url": url, "crawl_type": "text_file", "pages_crawled": 1, "chunks_stored": len(chunks), "urls_crawled": [source_url]}
        return MCP_Response(success=True, message="Text file crawled and content stored.", data=response_data, tool_name=tool_name).to_json_str()
    except Exception as e:
        print(f"Error in tool '{tool_name}': {e}\n{traceback.format_exc()}")
        return MCP_Response(success=False, message=f"An error occurred in {tool_name} for {url}", error=str(e), data={"url": url}, tool_name=tool_name).to_json_str()

@mcp.tool()
async def crawl_sitemap_tool(ctx: Context, url: str, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Extracts URLs from a sitemap and crawls them in parallel, storing content in Supabase.
    """
    tool_name = "crawl_sitemap_tool"
    try:
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        sitemap_urls = parse_sitemap(url)
        if not sitemap_urls:
            return MCP_Response(success=False, message="No URLs found in sitemap.", data={"url": url}, tool_name=tool_name).to_json_str()

        crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)

        if not crawl_results:
            return MCP_Response(success=False, message="No content found from sitemap URLs.", data={"url": url, "sitemap_urls_count": len(sitemap_urls)}, tool_name=tool_name).to_json_str()

        urls_db, chunk_numbers, contents, metadatas = [], [], [], []
        chunk_count = 0
        for doc in crawl_results:
            if not doc.get('markdown'): continue 
            source_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            for i, chunk in enumerate(chunks):
                urls_db.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                section_info = extract_section_info(chunk)
                meta = CrawlerMetadata(
                    chunk_index=i,
                    url=source_url,
                    source_domain=urlparse(source_url).netloc,
                    crawled_at=datetime.datetime.now(datetime.timezone.utc),
                    crawler_tool=tool_name,
                    document_title=section_info.get("title"),
                    document_description=section_info.get("description"),
                    document_keywords=section_info.get("keywords"),
                    content_headings=section_info.get("headings"),
                    additional_info={"sitemap_source_url": url}
                )
                metadatas.append(meta.to_supabase_dict())
                chunk_count += 1
        
        if contents:
            add_documents_to_supabase(supabase_client, urls_db, chunk_numbers, contents, metadatas, batch_size=20)
        
        response_data = {"url": url, "crawl_type": "sitemap", "pages_crawled": len(crawl_results), "chunks_stored": chunk_count,
                         "urls_in_sitemap": len(sitemap_urls), "urls_crawled_sample": [doc['url'] for doc in crawl_results if doc.get('url')][:10] + (["..."] if len(crawl_results) > 10 else [])}
        return MCP_Response(success=True, message="Sitemap crawled and content stored.", data=response_data, tool_name=tool_name).to_json_str()
    except Exception as e:
        print(f"Error in tool '{tool_name}': {e}\n{traceback.format_exc()}")
        return MCP_Response(success=False, message=f"An error occurred in {tool_name} for {url}", error=str(e), data={"url": url}, tool_name=tool_name).to_json_str()

@mcp.tool()
async def crawl_recursive_webpages_tool(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Recursively crawls internal links from a starting webpage up to a specified depth.
    """
    tool_name = "crawl_recursive_webpages_tool"
    try:
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)

        if not crawl_results:
            return MCP_Response(success=False, message="No content found during recursive crawl.", data={"url": url, "max_depth": max_depth}, tool_name=tool_name).to_json_str()

        urls_db, chunk_numbers, contents, metadatas = [], [], [], []
        chunk_count = 0
        for doc in crawl_results:
            if not doc.get('markdown'): continue
            source_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            for i, chunk in enumerate(chunks):
                urls_db.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                section_info = extract_section_info(chunk)
                meta = CrawlerMetadata(
                    chunk_index=i,
                    url=source_url,
                    source_domain=urlparse(source_url).netloc,
                    crawled_at=datetime.datetime.now(datetime.timezone.utc),
                    crawler_tool=tool_name,
                    document_title=section_info.get("title"),
                    document_description=section_info.get("description"),
                    document_keywords=section_info.get("keywords"),
                    content_headings=section_info.get("headings"),
                    additional_info={"recursive_crawl_start_url": url, "max_depth_setting": max_depth}
                )
                metadatas.append(meta.to_supabase_dict())
                chunk_count += 1
        
        if contents:
            add_documents_to_supabase(supabase_client, urls_db, chunk_numbers, contents, metadatas, batch_size=20)

        response_data = {"url": url, "crawl_type": "webpage_recursive", "pages_crawled": len(crawl_results), "chunks_stored": chunk_count,
                         "urls_crawled_sample": [doc['url'] for doc in crawl_results if doc.get('url')][:10] + (["..."] if len(crawl_results) > 10 else [])}
        return MCP_Response(success=True, message="Recursive web crawl completed and content stored.", data=response_data, tool_name=tool_name).to_json_str()
    except Exception as e:
        print(f"Error in tool '{tool_name}': {e}\n{traceback.format_exc()}")
        return MCP_Response(success=False, message=f"An error occurred in {tool_name} for {url}", error= str(e), data={"url": url}, tool_name=tool_name).to_json_str()


# --- Router Tool ---
@mcp.tool()
async def smart_crawl(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000, branch_name: Optional[str] = None, save_files_locally: bool = True, local_save_dir: str = "EXPORT_GITHUB") -> str:
    """
    Intelligently routes a URL to the appropriate crawling tool based on its type.
    
    Detects GitHub URLs, sitemaps, text files, or regular webpages and delegates.
    
    Args:
        ctx: The MCP server provided context.
        url: URL to crawl.
        max_depth: Max recursion depth (for webpages, not directly used by GitHub ZIP).
        max_concurrent: Max concurrent sessions (for sitemaps/webpages).
        chunk_size: Max size of content chunks.
        branch_name: Specific branch for GitHub repos.
        save_files_locally: For GitHub repos, whether to save files.
        local_save_dir: For GitHub repos, directory to save files.
    
    Returns:
        JSON string with crawl summary from the delegated tool.
    """
    tool_name = "smart_crawl (router)"
    try:
        if is_github_repository(url):
            print(f"Routing to crawl_github_repo for URL: {url}")
            # Pass relevant GitHub specific parameters
            return await crawl_github_repo(
                ctx, 
                repo_url=url, 
                branch_name=branch_name, 
                chunk_size=chunk_size, # GitHub tool also uses chunk_size for Supabase
                save_files_locally=save_files_locally, 
                local_save_dir=local_save_dir
            )
        elif is_txt(url):
            print(f"Routing to crawl_text_file_tool for URL: {url}")
            return await crawl_text_file_tool(ctx, url, chunk_size=chunk_size)
        elif is_sitemap(url):
            print(f"Routing to crawl_sitemap_tool for URL: {url}")
            return await crawl_sitemap_tool(ctx, url, max_concurrent=max_concurrent, chunk_size=chunk_size)
        else:
            print(f"Routing to crawl_recursive_webpages_tool for URL: {url}")
            return await crawl_recursive_webpages_tool(ctx, url, max_depth=max_depth, max_concurrent=max_concurrent, chunk_size=chunk_size)
    except Exception as e:
        print(f"Error in tool '{tool_name}' for {url}: {e}\n{traceback.format_exc()}")
        return MCP_Response(success=False, message=f"Routing error in {tool_name} for {url}", error=str(e), data={"url": url}, tool_name=tool_name).to_json_str()
