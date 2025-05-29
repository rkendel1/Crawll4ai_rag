# Testing Setup for MCP Crawl4AI RAG

This document provides a quick guide to set up and run tests for the MCP Crawl4AI RAG server.

## Quick Start

1. **Install test dependencies:**
   ```bash
   uv sync --extra test --extra dev
   ```

2. **Run all tests:**
   ```bash
   uv run pytest
   ```

3. **Run tests with coverage:**
   ```bash
   uv run pytest --cov=src --cov-report=html
   ```

4. **Run the comprehensive test script:**
   ```bash
   ./scripts/test.sh
   ```

## Test Categories

- **Unit Tests**: `uv run pytest tests/test_utils.py`
- **Integration Tests**: `uv run pytest tests/test_mcp_tools.py`
- **End-to-End Tests**: `uv run pytest tests/test_e2e.py`

## Key Features

✅ **Comprehensive Coverage**: Unit, integration, and e2e tests  
✅ **Async Support**: Full async/await testing with pytest-asyncio  
✅ **Mocking Strategy**: External APIs and dependencies properly mocked  
✅ **CI/CD Ready**: GitHub Actions workflow included  
✅ **Performance Testing**: Large document and concurrent crawling tests  
✅ **Error Handling**: Network, database, and validation error scenarios  

## Test Structure

```
tests/
├── conftest.py          # Shared fixtures and configuration
├── test_utils.py        # Unit tests for utility functions
├── test_mcp_tools.py    # Integration tests for MCP tools
└── test_e2e.py          # End-to-end workflow tests
```

## Environment Variables

For testing with real services (optional):
```bash
export SUPABASE_URL="your-supabase-url"
export SUPABASE_SERVICE_KEY="your-service-key"
export OPENAI_API_KEY="your-openai-key"
```

Tests will use mock values if these aren't set.

## Coverage Goals

- Overall: >90%
- Critical paths: 100%
- Error handling: 100%

For detailed testing documentation, see [docs/testing.md](docs/testing.md). 