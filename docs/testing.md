# Testing Guide

This document outlines the testing strategy and best practices for the MCP Crawl4AI RAG server.

## Test Structure

The test suite is organized into three main categories:

### 1. Unit Tests (`tests/test_utils.py`)
- Test individual utility functions in isolation
- Mock external dependencies (OpenAI API, Supabase)
- Fast execution, no network calls
- Focus on edge cases and error handling

### 2. Integration Tests (`tests/test_mcp_tools.py`)
- Test MCP tool functions with mocked dependencies
- Verify tool input/output contracts
- Test error handling and edge cases
- Ensure proper JSON response formatting

### 3. End-to-End Tests (`tests/test_e2e.py`)
- Test complete workflows from start to finish
- Verify server startup and configuration
- Test performance characteristics
- Include both success and failure scenarios

## Running Tests

### Prerequisites

1. Install test dependencies:
```bash
uv sync --extra test --extra dev
```

2. Set environment variables (optional):
```bash
export SUPABASE_URL="your-supabase-url"
export SUPABASE_SERVICE_KEY="your-service-key"
export OPENAI_API_KEY="your-openai-key"
```

### Quick Test Run

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test categories
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m e2e
```

### Using the Test Script

For a comprehensive test run with linting and server startup tests:

```bash
./scripts/test.sh
```

This script will:
- Install dependencies
- Run linting checks (flake8, black, isort, mypy)
- Execute all test categories
- Generate coverage reports
- Test server startup in both stdio and SSE modes

## Test Configuration

### Pytest Configuration (`pytest.ini`)

Key settings:
- `asyncio_mode = auto`: Automatically handle async tests
- Test discovery patterns for files, classes, and functions
- Custom markers for test categorization
- Warning filters to reduce noise

### Fixtures (`tests/conftest.py`)

Shared fixtures provide:
- Mock MCP context with crawler and Supabase client
- Sample data for testing (URLs, markdown content)
- Environment variable mocking
- Async event loop configuration

## Writing Tests

### Best Practices

1. **Use descriptive test names** that explain what is being tested:
```python
def test_smart_chunk_markdown_respects_code_block_boundaries(self):
    """Test that chunking doesn't break inside code blocks."""
```

2. **Follow the AAA pattern** (Arrange, Act, Assert):
```python
def test_create_embedding_success(self, mock_openai, mock_openai_response):
    # Arrange
    mock_openai.return_value = mock_openai_response
    
    # Act
    result = create_embedding("test text")
    
    # Assert
    assert len(result) == 1536
    assert all(isinstance(x, float) for x in result)
```

3. **Test both success and failure cases**:
```python
def test_crawl_single_page_success(self, mock_context):
    # Test successful crawling
    
def test_crawl_single_page_failure(self, mock_context):
    # Test crawling failure
    
def test_crawl_single_page_exception(self, mock_context):
    # Test exception handling
```

4. **Use appropriate mocking**:
```python
@patch('utils.openai.embeddings.create')
def test_create_embedding(self, mock_openai):
    # Mock external API calls
```

### Test Categories

Mark tests with appropriate categories:

```python
@pytest.mark.unit
def test_utility_function():
    """Unit test for utility function."""

@pytest.mark.integration  
def test_mcp_tool():
    """Integration test for MCP tool."""

@pytest.mark.e2e
def test_full_workflow():
    """End-to-end test for complete workflow."""

@pytest.mark.slow
def test_performance():
    """Performance test that takes longer to run."""

@pytest.mark.network
def test_with_real_api():
    """Test that requires network access."""
```

## Continuous Integration

### GitHub Actions

The CI pipeline (`.github/workflows/test.yml`) runs:

1. **Linting and formatting checks**:
   - flake8 for code quality
   - black for code formatting
   - isort for import sorting
   - mypy for type checking

2. **Test execution**:
   - Unit and integration tests with coverage
   - End-to-end tests (when secrets are available)
   - Server startup tests

3. **Coverage reporting**:
   - Uploads coverage to Codecov
   - Generates HTML coverage reports

### Test Matrix

Currently tests against:
- Python 3.12 on Ubuntu Latest
- Can be extended to include multiple Python versions and OS

## Mocking Strategy

### External Dependencies

1. **OpenAI API**: Mocked to return predictable embeddings
2. **Supabase**: Mocked to simulate database operations
3. **Crawl4AI**: Mocked to return controlled crawling results
4. **Network requests**: Mocked to avoid external dependencies

### Context Mocking

The `mock_context` fixture provides a complete MCP context with:
- Mocked crawler with configurable responses
- Mocked Supabase client with chainable methods
- Proper session and lifespan context structure

## Coverage Goals

Target coverage metrics:
- **Overall**: >90%
- **Critical paths**: 100% (crawling, embedding, storage)
- **Error handling**: 100%
- **Utility functions**: >95%

## Performance Testing

Performance tests verify:
- Large document chunking efficiency
- Concurrent crawling capabilities
- Memory usage patterns
- Response time characteristics

## Debugging Tests

### Running Individual Tests

```bash
# Run a specific test
uv run pytest tests/test_utils.py::TestMarkdownProcessing::test_smart_chunk_markdown_basic -v

# Run with debugging output
uv run pytest tests/test_utils.py -v -s --tb=long

# Run with pdb on failure
uv run pytest tests/test_utils.py --pdb
```

### Common Issues

1. **Import errors**: Ensure `src` is in Python path (handled by conftest.py)
2. **Async test failures**: Use `@pytest.mark.asyncio` decorator
3. **Mock not working**: Check patch target paths
4. **Environment variables**: Use `mock_env_vars` fixture

## Adding New Tests

When adding new functionality:

1. **Write tests first** (TDD approach)
2. **Add unit tests** for new utility functions
3. **Add integration tests** for new MCP tools
4. **Update fixtures** if new mock objects are needed
5. **Update CI** if new dependencies are required

## Test Data Management

Test data is managed through:
- Fixtures for reusable test data
- Factory functions for generating test objects
- Mock responses that mirror real API responses
- Sample files for testing file operations

This comprehensive testing strategy ensures the MCP server is reliable, maintainable, and performs well under various conditions. 