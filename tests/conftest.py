"""
Pytest configuration and shared fixtures for MCP server tests.
"""
import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawl4ai_mcp import FastMCP, Context, Crawl4AIContext
from utils import get_supabase_client


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'SUPABASE_URL': 'https://test.supabase.co',
        'SUPABASE_SERVICE_KEY': 'test-key',
        'OPENAI_API_KEY': 'test-openai-key',
        'HOST': '127.0.0.1',
        'PORT': '8052',
        'TRANSPORT': 'stdio'
    }):
        yield


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing."""
    mock_client = MagicMock()
    mock_client.from_.return_value.select.return_value.not_.return_value.is_.return_value.execute.return_value.data = []
    mock_client.rpc.return_value.execute.return_value.data = []
    return mock_client


@pytest.fixture
def mock_crawler():
    """Mock AsyncWebCrawler for testing."""
    mock_crawler = AsyncMock()
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.markdown = "# Test Content\nThis is test markdown content."
    mock_result.links = {"internal": [], "external": []}
    mock_result.error_message = None
    mock_crawler.arun.return_value = mock_result
    mock_crawler.arun_many.return_value = [mock_result]
    return mock_crawler


@pytest.fixture
def mock_context(mock_crawler, mock_supabase_client):
    """Mock MCP Context for testing."""
    mock_ctx = MagicMock(spec=Context)
    mock_session = MagicMock()
    mock_lifespan_context = MagicMock(spec=Crawl4AIContext)
    mock_lifespan_context.crawler = mock_crawler
    mock_lifespan_context.supabase_client = mock_supabase_client
    mock_session.lifespan_context = mock_lifespan_context
    mock_ctx.session = mock_session
    return mock_ctx


@pytest.fixture
async def mcp_server(mock_env_vars):
    """Create a test MCP server instance."""
    with patch('src.crawl4ai_mcp.get_supabase_client'), \
         patch('crawl4ai.AsyncWebCrawler'):
        # Import here to ensure patches are applied
        from crawl4ai_mcp import mcp
        yield mcp


@pytest.fixture
def sample_urls():
    """Sample URLs for testing."""
    return {
        'regular': 'https://example.com/page',
        'sitemap': 'https://example.com/sitemap.xml',
        'txt_file': 'https://example.com/llms.txt',
        'invalid': 'not-a-url'
    }


@pytest.fixture
def sample_markdown():
    """Sample markdown content for testing."""
    return """# Test Document

This is a test document with multiple sections.

## Section 1

Some content here with **bold** and *italic* text.

```python
def hello_world():
    print("Hello, World!")
```

## Section 2

More content here.

### Subsection

Even more content.
"""


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1] * 1536)
    ]
    return mock_response 