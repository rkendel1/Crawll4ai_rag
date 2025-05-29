"""
Integration tests for MCP tools.
"""
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

from crawl4ai_mcp import (
    crawl_single_page,
    smart_crawl_url,
    get_available_sources,
    perform_rag_query,
    is_sitemap,
    is_txt,
    parse_sitemap
)


class TestMCPTools:
    """Test MCP tool functions."""
    
    @pytest.mark.asyncio
    async def test_crawl_single_page_success(self, mock_context):
        """Test successful single page crawling."""
        result = await crawl_single_page(mock_context, "https://example.com")
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["url"] == "https://example.com"
        assert "chunks_stored" in result_data
        assert "content_length" in result_data
    
    @pytest.mark.asyncio
    async def test_crawl_single_page_failure(self, mock_context):
        """Test single page crawling failure."""
        # Mock crawler to return failure
        mock_context.session.lifespan_context.crawler.arun.return_value.success = False
        mock_context.session.lifespan_context.crawler.arun.return_value.error_message = "Failed to crawl"
        
        result = await crawl_single_page(mock_context, "https://example.com")
        
        result_data = json.loads(result)
        assert result_data["success"] is False
        assert result_data["error"] == "Failed to crawl"
    
    @pytest.mark.asyncio
    async def test_crawl_single_page_exception(self, mock_context):
        """Test single page crawling with exception."""
        # Mock crawler to raise exception
        mock_context.session.lifespan_context.crawler.arun.side_effect = Exception("Network error")
        
        result = await crawl_single_page(mock_context, "https://example.com")
        
        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "Network error" in result_data["error"]
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.crawl_recursive_internal_links')
    async def test_smart_crawl_url_webpage(self, mock_crawl_recursive, mock_context):
        """Test smart crawling of regular webpage."""
        mock_crawl_recursive.return_value = [
            {"url": "https://example.com", "markdown": "# Test\nContent"}
        ]
        
        result = await smart_crawl_url(mock_context, "https://example.com")
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["crawl_type"] == "webpage"
        assert result_data["pages_crawled"] == 1
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.parse_sitemap')
    @patch('crawl4ai_mcp.crawl_batch')
    async def test_smart_crawl_url_sitemap(self, mock_crawl_batch, mock_parse_sitemap, mock_context):
        """Test smart crawling of sitemap."""
        mock_parse_sitemap.return_value = ["https://example.com/page1", "https://example.com/page2"]
        mock_crawl_batch.return_value = [
            {"url": "https://example.com/page1", "markdown": "# Page 1"},
            {"url": "https://example.com/page2", "markdown": "# Page 2"}
        ]
        
        result = await smart_crawl_url(mock_context, "https://example.com/sitemap.xml")
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["crawl_type"] == "sitemap"
        assert result_data["pages_crawled"] == 2
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.crawl_markdown_file')
    async def test_smart_crawl_url_txt_file(self, mock_crawl_file, mock_context):
        """Test smart crawling of text file."""
        mock_crawl_file.return_value = [
            {"url": "https://example.com/llms.txt", "markdown": "# LLMs\nContent about LLMs"}
        ]
        
        result = await smart_crawl_url(mock_context, "https://example.com/llms.txt")
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["crawl_type"] == "text_file"
        assert result_data["pages_crawled"] == 1
    
    @pytest.mark.asyncio
    async def test_get_available_sources_success(self, mock_context):
        """Test getting available sources successfully."""
        # Mock Supabase response
        mock_context.session.lifespan_context.supabase_client.from_.return_value.select.return_value.not_.return_value.is_.return_value.execute.return_value.data = [
            {"metadata": {"source": "example.com"}},
            {"metadata": {"source": "test.com"}},
            {"metadata": {"source": "example.com"}}  # Duplicate
        ]
        
        result = await get_available_sources(mock_context)
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["sources"]) == 2  # Duplicates removed
        assert "example.com" in result_data["sources"]
        assert "test.com" in result_data["sources"]
    
    @pytest.mark.asyncio
    async def test_get_available_sources_exception(self, mock_context):
        """Test getting available sources with exception."""
        mock_context.session.lifespan_context.supabase_client.from_.side_effect = Exception("DB error")
        
        result = await get_available_sources(mock_context)
        
        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "DB error" in result_data["error"]
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.search_documents')
    async def test_perform_rag_query_success(self, mock_search, mock_context):
        """Test RAG query successfully."""
        mock_search.return_value = [
            {
                "url": "https://example.com",
                "content": "Test content",
                "metadata": {"source": "example.com"},
                "similarity": 0.9
            }
        ]
        
        result = await perform_rag_query(mock_context, "test query")
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["query"] == "test query"
        assert len(result_data["results"]) == 1
        assert result_data["results"][0]["url"] == "https://example.com"
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.search_documents')
    async def test_perform_rag_query_with_source_filter(self, mock_search, mock_context):
        """Test RAG query with source filter."""
        mock_search.return_value = []
        
        result = await perform_rag_query(mock_context, "test query", source="example.com")
        
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["source_filter"] == "example.com"
        
        # Verify search was called with filter
        mock_search.assert_called_once()
        call_args = mock_search.call_args[1]
        assert call_args["filter_metadata"] == {"source": "example.com"}


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_is_sitemap(self):
        """Test sitemap detection."""
        assert is_sitemap("https://example.com/sitemap.xml") is True
        assert is_sitemap("https://example.com/sitemaps/main.xml") is True
        assert is_sitemap("https://example.com/page.html") is False
        assert is_sitemap("https://example.com/sitemap") is True  # Contains 'sitemap'
    
    def test_is_txt(self):
        """Test text file detection."""
        assert is_txt("https://example.com/llms.txt") is True
        assert is_txt("https://example.com/robots.txt") is True
        assert is_txt("https://example.com/page.html") is False
        assert is_txt("https://example.com/file.json") is False
    
    @patch('crawl4ai_mcp.requests.get')
    def test_parse_sitemap_success(self, mock_get):
        """Test successful sitemap parsing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'''<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/page1</loc></url>
            <url><loc>https://example.com/page2</loc></url>
        </urlset>'''
        mock_get.return_value = mock_response
        
        urls = parse_sitemap("https://example.com/sitemap.xml")
        
        assert len(urls) == 2
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls
    
    @patch('crawl4ai_mcp.requests.get')
    def test_parse_sitemap_failure(self, mock_get):
        """Test sitemap parsing failure."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        urls = parse_sitemap("https://example.com/sitemap.xml")
        
        assert urls == []
    
    @patch('crawl4ai_mcp.requests.get')
    def test_parse_sitemap_invalid_xml(self, mock_get):
        """Test sitemap parsing with invalid XML."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'Invalid XML content'
        mock_get.return_value = mock_response
        
        urls = parse_sitemap("https://example.com/sitemap.xml")
        
        assert urls == [] 