import datetime
import traceback
import os
import logging

from . import mcp
from src.server import MCP_Response, CrawlerMetadata
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import Context
from crawl4ai import CrawlerRunConfig, CacheMode

from src.service.github import (
    extract_repo_info,
    download_github_repo,
    CrawlException,
    validate_github_url,
)
from src.service.crawl4ai import (
    crawl_markdown_file,
    crawl_batch,
    crawl_recursive_internal_links,
)
from src.service.supabase import add_documents_to_supabase

from src.utils.crawler import is_sitemap, is_txt, is_github_repository, parse_sitemap
from src.utils.chunking import smart_chunk_markdown, extract_section_info, enrich_chunks_with_metadata
from src.utils.files import save_raw_content_to_export

DEFAULT_LOCAL_SAVE_DIR = "EXPORT"

default_logger = logging.getLogger("DomoClientLogger")
default_logger.setLevel(logging.INFO)

# --- Individual Crawling Tools ---


@mcp.tool()
async def crawl_single_page(
    ctx: Context,
    url: str,
    save_files_locally: bool = True,
    local_save_dir: str = DEFAULT_LOCAL_SAVE_DIR,
) -> str:
    """
    Crawl a single web page, store its content in Supabase, and save locally using utility.
    """
    tool_name = "crawl_single_page"
    local_file_path = None
    try:
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        domo_client = ctx.request_context.lifespan_context.domo_client
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=run_config)

        if result.success and result.markdown:
            if save_files_locally:
                local_file_path = await save_raw_content_to_export(
                    source_url_for_metadata=url,
                    content=result.markdown,
                    output_folder=local_save_dir,
                    target_relative_path_hint=(
                        f"{Path(urlparse(url).path).name or urlparse(url).netloc}.md"
                        if Path(urlparse(url).path).name or urlparse(url).netloc
                        else "crawled_page.md"
                    ),
                )

            chunks = smart_chunk_markdown(result.markdown)
            urls_db, chunk_numbers, contents, metadatas = enrich_chunks_with_metadata(
                chunks=chunks,
                source_url=url,
                tool_name=tool_name,
                local_file_path=local_file_path,
                file_name_from_url=Path(urlparse(url).path).name or urlparse(url).netloc,
            )

            if contents:
                url_to_full_document = {url: result.markdown}
                add_documents_to_supabase(
                    client=supabase_client,
                    urls=urls_db,
                    chunk_numbers=chunk_numbers,
                    contents=contents,
                    metadatas=metadatas,
                    url_to_full_document=url_to_full_document,
                )
                domo_client.upsert_text_embedding(
                    markdown=result.markdown,
                    url=url,
                    tool_name=tool_name,
                    local_file_path=url_to_full_document,
                    file_name_from_url=Path(urlparse(url).path).name or urlparse(url).netloc
                )

            response_data = {
                "url": url,
                "chunks_stored": len(chunks),
                "content_length": len(result.markdown),
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", [])),
                },
                "local_file_path": local_file_path,
            }
            return MCP_Response(
                success=True,
                message="Page crawled, content stored, and saved locally.",
                data=response_data,
                tool_name=tool_name,
            ).to_json_str()
        else:
            return MCP_Response(
                success=False,
                message=f"Failed to crawl page: {url}",
                error=result.error_message,
                data={"url": url},
                tool_name=tool_name,
            ).to_json_str()
    except Exception as e:
        print(f"Error in tool '{tool_name}': {e}\n{traceback.format_exc()}")
        return MCP_Response(
            success=False,
            message=f"An error occurred in {tool_name} for {url}",
            error=str(e),
            data={"url": url},
            tool_name=tool_name,
        ).to_json_str()


@mcp.tool()
async def crawl_github_repo(
    ctx: Context,
    repo_url: str,
    branch_name: Optional[str] = None,
    chunk_size: int = 5000,
    save_files_locally: bool = True,
    local_save_dir: str = DEFAULT_LOCAL_SAVE_DIR,
) -> str:
    """
    Downloads a GitHub repository ZIP (private or public), processes its content,
    stores it in Supabase, and optionally saves it locally.
    """
    tool_name = "crawl_github_repo"

    if not validate_github_url(repo_url):
        return MCP_Response(
            success=False,
            message="Invalid GitHub repository URL provided.",
            data={"repo_url": repo_url},
            tool_name=tool_name,
        ).to_json_str()

    github_auth_token = os.getenv("GITHUB_TOKEN")

    try:
        # Extract repository info
        owner, repo_name, branch_from_url, _ = extract_repo_info(repo_url)
        branch_name = branch_name or branch_from_url or "main"

        # Download and extract the repository
        extracted_path = download_github_repo(
            repo_url=repo_url,
            extract_to=local_save_dir,
            github_token=github_auth_token,
            branch_name=branch_name,
        )

        # Process repository files
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        domo_client = ctx.request_context.lifespan_context.domo_client
        urls_db, chunk_numbers, contents, metadatas = [], [], [], []
        url_to_full_document = {}

        for file_path in Path(extracted_path).rglob("*"):
            if (
                file_path.is_file()
                # and file_path.suffix in [
                #     ".md",
                #     ".txt",
                #     ".py",
                # ]
            ):  # Process specific file types
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Chunk the content
                chunks = smart_chunk_markdown(content, chunk_size=chunk_size)

                for i, chunk in enumerate(chunks):
                    urls_db.append(str(file_path))
                    chunk_numbers.append(i)
                    contents.append(chunk)
                    section_info = extract_section_info(chunk)
                    meta = CrawlerMetadata(
                        chunk_index=i,
                        url=str(file_path),
                        source_domain="github.com",
                        crawled_at=datetime.datetime.now(datetime.timezone.utc),
                        crawler_tool=tool_name,
                        document_title=section_info.get("title") or file_path.name,
                        document_description=section_info.get("description"),
                        document_keywords=section_info.get("keywords"),
                        content_headings=section_info.get("headings"),
                        additional_info={
                            "repository_url": repo_url,
                            "branch": branch_name,
                            "local_file_path": str(file_path),
                        },
                    )
                    metadatas.append(meta.to_dict())

                # Store the full document content
                url_to_full_document[str(file_path)] = content

        # Add documents to Supabase
        if contents:
            add_documents_to_supabase(
                client=supabase_client,
                urls=urls_db,
                chunk_numbers=chunk_numbers,
                contents=contents,
                metadatas=metadatas,
                url_to_full_document=url_to_full_document,
            )
            domo_client.upsert_text_embedding(
                markdown=content,
                url=str(file_path),
                tool_name=tool_name,
                local_file_path=url_to_full_document,
                file_name_from_url=Path(urlparse(url).path).name or urlparse(url).netloc
            )


        response_data = {
            "repo_url": repo_url,
            "branch": branch_name,
            "extracted_path": extracted_path,
            "auth_used": bool(github_auth_token),
            "files_processed": len(urls_db),
        }
        return MCP_Response(
            success=True,
            message="Repository downloaded, processed, and stored successfully.",
            data=response_data,
            tool_name=tool_name,
        ).to_json_str()

    except CrawlException as e:
        return MCP_Response(
            success=False,
            message=str(e),
            error=str(e),
            data={
                "repo_url": repo_url,
                "auth_used": bool(github_auth_token),
            },
            tool_name=tool_name,
        ).to_json_str()

    except Exception as e:
        print(f"Error in tool '{tool_name}': {e}\n{traceback.format_exc()}")
        return MCP_Response(
            success=False,
            message="Failed to process repository.",
            error=str(e),
            data={"repo_url": repo_url},
            tool_name=tool_name,
        ).to_json_str()


@mcp.tool()
async def crawl_text_file_tool(
    ctx: Context,
    url: str,
    chunk_size: int = 5000,
    save_files_locally: bool = True,
    local_save_dir: str = DEFAULT_LOCAL_SAVE_DIR,
) -> str:
    """
    Crawls a single text file, stores its content in Supabase, and saves locally using utility.
    """
    tool_name = "crawl_text_file_tool"
    local_file_path = None
    try:
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        domo_client = ctx.request_context.lifespan_context.domo_client


        crawl_results = await crawl_markdown_file(crawler, url)

        if not crawl_results or not crawl_results[0].get("markdown"):
            return MCP_Response(
                success=False,
                message="No content found in text file.",
                data={"url": url},
                tool_name=tool_name,
            ).to_json_str()

        doc = crawl_results[0]
        source_url = doc["url"]
        md = doc["markdown"]

        if save_files_locally and md:
            original_filename_hint = (
                Path(urlparse(source_url).path).name or "crawled_text_file.txt"
            )
            local_file_path = await save_raw_content_to_export(
                source_url_for_metadata=source_url,
                content=md,
                output_folder=local_save_dir,
                target_relative_path_hint=original_filename_hint,
            )

        file_name_from_url = Path(urlparse(source_url).path).name
        chunks = smart_chunk_markdown(md, chunk_size=chunk_size)

        urls_db, chunk_numbers, contents, metadatas = enrich_chunks_with_metadata(
            chunks=chunks,
            source_url=source_url,
            tool_name=tool_name,
            local_file_path=local_file_path,
            file_name_from_url=file_name_from_url
        )

        if contents:
            url_to_full_document = {source_url: md}
            add_documents_to_supabase(
                client=supabase_client,
                urls=urls_db,
                chunk_numbers=chunk_numbers,
                contents=contents,
                metadatas=metadatas,
                url_to_full_document=url_to_full_document,
            )
            domo_client.upsert_text_embedding(
                markdown=md,
                url=url,
                tool_name=tool_name,
                local_file_path=url_to_full_document,
                file_name_from_url=Path(urlparse(url).path).name or urlparse(url).netloc
            )
        response_data = {
            "url": url,
            "crawl_type": "text_file",
            "pages_crawled": 1,
            "chunks_stored": len(chunks),
            "urls_crawled": [source_url],
            "local_file_path": local_file_path,
        }
        return MCP_Response(
            success=True,
            message="Text file crawled, content stored, and saved locally.",
            data=response_data,
            tool_name=tool_name,
        ).to_json_str()
    except Exception as e:
        print(f"Error in tool '{tool_name}': {e}\n{traceback.format_exc()}")
        return MCP_Response(
            success=False,
            message=f"An error occurred in {tool_name} for {url}",
            error=str(e),
            data={"url": url},
            tool_name=tool_name,
        ).to_json_str()


@mcp.tool()
async def crawl_sitemap_tool(
    ctx: Context,
    url: str,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
    save_files_locally: bool = True,
    local_save_dir: str = DEFAULT_LOCAL_SAVE_DIR,
) -> str:
    """
    Extracts URLs from a sitemap, crawls them, stores content in Supabase, and saves locally using utility.
    """
    tool_name = "crawl_sitemap_tool"
    saved_file_paths_map: Dict[str, Optional[str]] = {}  # Store URL -> local_path
    try:
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        domo_client = ctx.request_context.lifespan_context.domo_client

        sitemap_urls = parse_sitemap(url)
        if not sitemap_urls:
            return MCP_Response(
                success=False,
                message="No URLs found in sitemap.",
                data={"url": url},
                tool_name=tool_name,
            ).to_json_str()

        crawl_results = await crawl_batch(
            crawler, sitemap_urls, max_concurrent=max_concurrent
        )

        if not crawl_results:
            return MCP_Response(
                success=False,
                message="No content found from sitemap URLs.",
                data={"url": url, "sitemap_urls_count": len(sitemap_urls)},
                tool_name=tool_name,
            ).to_json_str()

        urls_db, chunk_numbers, contents, metadatas = [], [], [], []
        chunk_count = 0
        url_to_full_document = {}

        for doc in crawl_results:
            if not doc.get("markdown"):
                continue
            source_url = doc["url"]
            md = doc["markdown"]
            url_to_full_document[source_url] = md

            local_file_path_for_doc = None
            if save_files_locally and md:
                path_name = (
                    Path(urlparse(source_url).path).name or urlparse(source_url).netloc
                )
                file_hint = (
                    f"{path_name}.md"
                    if path_name
                    else f"{urlparse(source_url).netloc}.md"
                )
                local_file_path_for_doc = await save_raw_content_to_export(
                    source_url_for_metadata=source_url,
                    content=md,
                    output_folder=local_save_dir,
                    target_relative_path_hint=file_hint,
                )
                saved_file_paths_map[source_url] = local_file_path_for_doc

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
                    additional_info={
                        "sitemap_source_url": url,
                        "local_file_path": saved_file_paths_map.get(source_url),
                    },
                )
                metadatas.append(meta.to_dict())
                chunk_count += 1

        if contents:
            add_documents_to_supabase(
                client=supabase_client,
                urls=urls_db,
                chunk_numbers=chunk_numbers,
                contents=contents,
                metadatas=metadatas,
                url_to_full_document=url_to_full_document,
                batch_size=20,
            )
            domo_client.upsert_text_embedding(
                markdown=md,
                url=url,
                tool_name=tool_name,
                local_file_path=url_to_full_document,
                file_name_from_url=Path(urlparse(url).path).name or urlparse(url).netloc
            )


        actual_saved_paths = [p for p in saved_file_paths_map.values() if p]
        response_data = {
            "url": url,
            "crawl_type": "sitemap",
            "pages_crawled": len(crawl_results),
            "chunks_stored": chunk_count,
            "urls_in_sitemap": len(sitemap_urls),
            "locally_saved_files_count": len(actual_saved_paths),
            "locally_saved_files_sample": actual_saved_paths[:5]
            + (["..."] if len(actual_saved_paths) > 5 else []),
            "urls_crawled_sample": [
                doc["url"] for doc in crawl_results if doc.get("url")
            ][:10]
            + (["..."] if len(crawl_results) > 10 else []),
        }
        return MCP_Response(
            success=True,
            message="Sitemap crawled, content stored, and saved locally.",
            data=response_data,
            tool_name=tool_name,
        ).to_json_str()
    except Exception as e:
        print(f"Error in tool '{tool_name}': {e}\n{traceback.format_exc()}")
        return MCP_Response(
            success=False,
            message=f"An error occurred in {tool_name} for {url}",
            error=str(e),
            data={"url": url},
            tool_name=tool_name,
        ).to_json_str()


@mcp.tool()
async def crawl_recursive_webpages_tool(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
    save_files_locally: bool = True,
    local_save_dir: str = DEFAULT_LOCAL_SAVE_DIR,
) -> str:
    """
    Recursively crawls internal links, stores content in Supabase, and saves locally using utility.
    """
    tool_name = "crawl_recursive_webpages_tool"
    saved_file_paths_map: Dict[str, Optional[str]] = {}  # Store URL -> local_path
    try:
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        domo_client = ctx.request_context.lifespan_context.domo_client

        crawl_results = await crawl_recursive_internal_links(
            crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent
        )

        if not crawl_results:
            return MCP_Response(
                success=False,
                message="No content found during recursive crawl.",
                data={"url": url, "max_depth": max_depth},
                tool_name=tool_name,
            ).to_json_str()

        urls_db, chunk_numbers, contents, metadatas = [], [], [], []
        chunk_count = 0
        url_to_full_document = {}

        for doc in crawl_results:
            if not doc.get("markdown"):
                continue
            source_url = doc["url"]
            md = doc["markdown"]
            url_to_full_document[source_url] = md

            local_file_path_for_doc = None
            if save_files_locally and md:
                path_name = (
                    Path(urlparse(source_url).path).name or urlparse(source_url).netloc
                )
                file_hint = (
                    f"{path_name}.md"
                    if path_name
                    else f"{urlparse(source_url).netloc}.md"
                )
                local_file_path_for_doc = await save_raw_content_to_export(
                    source_url_for_metadata=source_url,
                    content=md,
                    output_folder=local_save_dir,
                    target_relative_path_hint=file_hint,
                )
                saved_file_paths_map[source_url] = local_file_path_for_doc

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
                    additional_info={
                        "recursive_crawl_start_url": url,
                        "max_depth_setting": max_depth,
                        "local_file_path": saved_file_paths_map.get(source_url),
                    },
                )
                metadatas.append(meta.to_dict())
                chunk_count += 1

        if contents:
            add_documents_to_supabase(
                client=supabase_client,
                urls=urls_db,
                chunk_numbers=chunk_numbers,
                contents=contents,
                metadatas=metadatas,
                url_to_full_document=url_to_full_document,
                batch_size=20,
            )
            domo_client.upsert_text_embedding(
                markdown=md,
                url=url,
                tool_name=tool_name,
                local_file_path=url_to_full_document,
                file_name_from_url=Path(urlparse(url).path).name or urlparse(url).netloc
            )

        actual_saved_paths = [p for p in saved_file_paths_map.values() if p]
        response_data = {
            "url": url,
            "crawl_type": "webpage_recursive",
            "pages_crawled": len(crawl_results),
            "chunks_stored": chunk_count,
            "locally_saved_files_count": len(actual_saved_paths),
            "locally_saved_files_sample": actual_saved_paths[:5]
            + (["..."] if len(actual_saved_paths) > 5 else []),
            "urls_crawled_sample": [
                doc["url"] for doc in crawl_results if doc.get("url")
            ][:10]
            + (["..."] if len(crawl_results) > 10 else []),
        }
        return MCP_Response(
            success=True,
            message="Recursive web crawl completed, content stored, and saved locally.",
            data=response_data,
            tool_name=tool_name,
        ).to_json_str()
    except Exception as e:
        print(f"Error in tool '{tool_name}': {e}\n{traceback.format_exc()}")
        return MCP_Response(
            success=False,
            message=f"An error occurred in {tool_name} for {url}",
            error=str(e),
            data={"url": url},
            tool_name=tool_name,
        ).to_json_str()


# --- Router Tool ---
@mcp.tool()
async def smart_crawl(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
    branch_name: Optional[str] = None,
    save_files_locally: bool = True,
    local_save_dir: str = DEFAULT_LOCAL_SAVE_DIR,
) -> str:
    """
    Smartly determines the type of URL and routes to the appropriate crawler.
    All crawlers will save files locally by default using the utility function.
    """
    tool_name = "smart_crawl"
    print(
        f"Smart crawl initiated for URL: {url}. Local save: {save_files_locally}, Dir: {local_save_dir}"
    )

    try:
        if is_github_repository(url):
            print(
                f"Smart crawl identified GitHub URL: {url}. Routing to crawl_github_repo."
            )
            return await crawl_github_repo(
                ctx,
                url,
                branch_name=branch_name,
                chunk_size=chunk_size,
                save_files_locally=save_files_locally,
                local_save_dir=local_save_dir,
            )
        elif is_sitemap(url):
            print(
                f"Smart crawl identified Sitemap URL: {url}. Routing to crawl_sitemap_tool."
            )
            return await crawl_sitemap_tool(
                ctx,
                url,
                max_concurrent=max_concurrent,
                chunk_size=chunk_size,
                save_files_locally=save_files_locally,
                local_save_dir=local_save_dir,
            )
        elif is_txt(url):
            print(
                f"Smart crawl identified Text File URL: {url}. Routing to crawl_text_file_tool."
            )
            return await crawl_text_file_tool(
                ctx,
                url,
                chunk_size=chunk_size,
                save_files_locally=save_files_locally,
                local_save_dir=local_save_dir,
            )
        else:
            print(
                f"Smart crawl identified Webpage URL: {url}. Routing to crawl_recursive_webpages_tool."
            )
            return await crawl_recursive_webpages_tool(
                ctx,
                url,
                max_depth=max_depth,
                max_concurrent=max_concurrent,
                chunk_size=chunk_size,
                save_files_locally=save_files_locally,
                local_save_dir=local_save_dir,
            )

    except Exception as e:
        error_message = f"An error occurred in smart_crawl for URL '{url}': {e}"
        print(f"{error_message}\n{traceback.format_exc()}")
        return MCP_Response(
            success=False,
            message=error_message,
            error=str(e),
            data={"url": url},
            tool_name=tool_name,
        ).to_json_str()
