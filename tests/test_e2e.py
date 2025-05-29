"""
End-to-end tests for the MCP server.
"""
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client


class TestMCPServerE2E:
    """End-to-end tests for the MCP server."""
    
    @pytest.mark.asyncio
    @patch('src.crawl4ai_mcp.get_supabase_client')
    @patch('crawl4ai.AsyncWebCrawler')
    async def test_server_startup_stdio(self, mock_crawler_class, mock_supabase):
        """Test server starts up correctly in stdio mode."""
        # Mock the crawler and supabase
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler
        mock_supabase.return_value = MagicMock()
        
        # Import after mocking
        from crawl4ai_mcp import mcp
        
        # Test that server has the expected tools
        tools = await mcp.list_tools()
        tool_names = [tool.name for tool in tools]
        
        expected_tools = [
            "crawl_single_page",
            "smart_crawl_url", 
            "get_available_sources",
            "perform_rag_query"
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tool_names
    
    @pytest.mark.asyncio
    @patch('src.crawl4ai_mcp.get_supabase_client')
    @patch('crawl4ai.AsyncWebCrawler')
    async def test_server_startup_sse(self, mock_crawler_class, mock_supabase):
        """Test server starts up correctly in SSE mode."""
        # Mock the crawler and supabase
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler
        mock_supabase.return_value = MagicMock()
        
        # Import after mocking
        from crawl4ai_mcp import mcp
        
        # Test SSE app creation
        sse_app = mcp.sse_app()
        assert sse_app is not None
        
        # Test that routes are configured
        routes = sse_app.routes
        assert len(routes) > 0
    
    @pytest.mark.asyncio
    @patch('src.utils.create_embeddings_batch')
    @patch('src.utils.add_documents_to_supabase')
    async def test_full_crawl_workflow(self, mock_add_docs, mock_embeddings, mock_context):
        """Test complete crawling workflow."""
        # Setup mocks
        mock_embeddings.return_value = [[0.1] * 1536]
        mock_add_docs.return_value = None
        
        # Mock successful crawl result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "# Test Page\n\nThis is test content."
        mock_result.links = {"internal": [], "external": []}
        mock_context.session.lifespan_context.crawler.arun.return_value = mock_result
        
        # Import and test
        from crawl4ai_mcp import crawl_single_page
        
        result = await crawl_single_page(mock_context, "https://example.com")
        result_data = json.loads(result)
        
        # Verify workflow completed
        assert result_data["success"] is True
        assert result_data["chunks_stored"] > 0
        mock_add_docs.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rag_search_workflow(self, mock_context):
        """Test complete RAG search workflow."""
        # Mock search results
        mock_context.session.lifespan_context.supabase_client.rpc.return_value.execute.return_value.data = [
            {
                "url": "https://example.com",
                "content": "Test content about Python programming",
                "metadata": {"source": "example.com", "chunk_index": 0},
                "similarity": 0.85
            }
        ]
        
        # Import and test
        from crawl4ai_mcp import perform_rag_query
        
        result = await perform_rag_query(mock_context, "Python programming")
        result_data = json.loads(result)
        
        # Verify search workflow
        assert result_data["success"] is True
        assert len(result_data["results"]) == 1
        assert result_data["results"][0]["similarity"] == 0.85


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, mock_context):
        """Test handling of network errors during crawling."""
        # Mock network error
        mock_context.session.lifespan_context.crawler.arun.side_effect = Exception("Connection timeout")
        
        from crawl4ai_mcp import crawl_single_page
        
        result = await crawl_single_page(mock_context, "https://unreachable.com")
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "Connection timeout" in result_data["error"]
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, mock_context):
        """Test handling of database errors."""
        # Mock database error
        mock_context.session.lifespan_context.supabase_client.from_.side_effect = Exception("Database connection failed")
        
        from crawl4ai_mcp import get_available_sources
        
        result = await get_available_sources(mock_context)
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "Database connection failed" in result_data["error"]
    
    @pytest.mark.asyncio
    async def test_invalid_url_handling(self, mock_context):
        """Test handling of invalid URLs."""
        from crawl4ai_mcp import crawl_single_page
        
        result = await crawl_single_page(mock_context, "not-a-valid-url")
        result_data = json.loads(result)
        
        # Should handle gracefully (exact behavior depends on implementation)
        assert "success" in result_data


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_large_document_chunking(self, mock_context):
        """Test chunking of large documents."""
        # Create a large markdown document
        large_content = "# Large Document\n\n" + "This is a paragraph. " * 1000
        
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = large_content
        mock_result.links = {"internal": [], "external": []}
        mock_context.session.lifespan_context.crawler.arun.return_value = mock_result
        
        from crawl4ai_mcp import crawl_single_page
        
        result = await crawl_single_page(mock_context, "https://example.com/large")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["chunks_stored"] > 1  # Should be chunked
        assert result_data["content_length"] == len(large_content)
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.crawl_batch')
    async def test_concurrent_crawling(self, mock_crawl_batch, mock_context):
        """Test concurrent crawling of multiple URLs."""
        # Mock batch crawling results
        mock_crawl_batch.return_value = [
            {"url": f"https://example.com/page{i}", "markdown": f"# Page {i}\nContent {i}"}
            for i in range(10)
        ]
        
        from crawl4ai_mcp import smart_crawl_url
        
        # Test with sitemap that has many URLs
        with patch('crawl4ai_mcp.parse_sitemap') as mock_parse:
            mock_parse.return_value = [f"https://example.com/page{i}" for i in range(10)]
            
            result = await smart_crawl_url(mock_context, "https://example.com/sitemap.xml")
            result_data = json.loads(result)
            
            assert result_data["success"] is True
            assert result_data["pages_crawled"] == 10 