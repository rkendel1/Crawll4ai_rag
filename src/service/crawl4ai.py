"""
Crawler functions that directly interface with Crawl4AI.
"""

from typing import List, Dict, Any
from urllib.parse import urldefrag
from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,
)

from crawl4ai.deep_crawling.filters import URLPatternFilter, FilterChain
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy


def generate_crawl_strategy(urls):

    patterns = [f"{url}*" for url in urls]

    return BestFirstCrawlingStrategy(
        max_depth=10, filter_chain=FilterChain([URLPatternFilter(patterns=patterns)])
    )


async def crawl_markdown_file(
    crawler: AsyncWebCrawler, url: str
) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.

    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(
        deep_crawl_strategy=generate_crawl_strategy(url), magic=True
    )

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{"url": url, "markdown": result.markdown}]
    else:
        print(f"Failed to crawl {url}: {result.error_message}")
        return []


async def crawl_batch(
    crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.

    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        magic=True,
        stream=False,
        deep_crawl_strategy=generate_crawl_strategy(urls),
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    results = await crawler.arun_many(
        urls=urls, config=crawl_config, dispatcher=dispatcher
    )
    return [
        {"url": r.url, "markdown": r.markdown}
        for r in results
        if r.success and r.markdown
    ]


async def crawl_recursive_internal_links(
    crawler: AsyncWebCrawler,
    start_urls: List[str],
    max_depth: int = 3,
    max_concurrent: int = 10,
    results_all: List[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.

    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        magic=True,
        stream=False,
        deep_crawl_strategy=generate_crawl_strategy(start_urls),
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])

    if not results_all:
        results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [
            normalize_url(url)
            for url in current_urls
            if normalize_url(url) not in visited
        ]

        if not urls_to_crawl:
            break

        results_all.append(
            (
                await crawler.arun_many(
                    urls=urls_to_crawl, config=crawl_config, dispatcher=dispatcher
                )
            )
        )

    return [
        {"url": r.url, "markdown": r.markdown}
        for r in results_all
        if r.success and r.markdown
    ]
