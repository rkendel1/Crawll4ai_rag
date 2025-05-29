#!/bin/bash

# Test script for local development
set -e

echo "🧪 Running MCP Crawl4AI RAG Tests"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install it first: https://docs.astral.sh/uv/"
    exit 1
fi

print_status "Installing dependencies..."
uv sync --extra test --extra dev

# Set test environment variables
export SUPABASE_URL=${SUPABASE_URL:-"https://test.supabase.co"}
export SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_KEY:-"test-key"}
export OPENAI_API_KEY=${OPENAI_API_KEY:-"test-openai-key"}

# Run linting
echo ""
echo "🔍 Running linting checks..."
print_status "Running flake8..."
uv run flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics

print_status "Running black format check..."
uv run black --check src tests

print_status "Running isort import check..."
uv run isort --check-only src tests

print_status "Running mypy type check..."
uv run mypy src --ignore-missing-imports

# Run tests
echo ""
echo "🧪 Running tests..."

# Unit tests
print_status "Running unit tests..."
uv run pytest tests/test_utils.py -v --tb=short

# Integration tests
print_status "Running integration tests..."
uv run pytest tests/test_mcp_tools.py -v --tb=short

# End-to-end tests (without network)
print_status "Running e2e tests (offline)..."
uv run pytest tests/test_e2e.py -v --tb=short -m "not network"

# Full test suite with coverage
echo ""
print_status "Running full test suite with coverage..."
uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Test server startup
echo ""
echo "🚀 Testing server startup..."

print_status "Testing stdio mode startup..."
timeout 5s uv run python src/crawl4ai_mcp.py &
STDIO_PID=$!
sleep 2
if kill -0 $STDIO_PID 2>/dev/null; then
    print_status "Server started successfully in stdio mode"
    kill $STDIO_PID
else
    print_error "Server failed to start in stdio mode"
fi

print_status "Testing SSE mode startup..."
TRANSPORT=sse HOST=127.0.0.1 PORT=8053 timeout 5s uv run python src/crawl4ai_mcp.py &
SSE_PID=$!
sleep 2
if kill -0 $SSE_PID 2>/dev/null; then
    print_status "Server started successfully in SSE mode"
    kill $SSE_PID
else
    print_error "Server failed to start in SSE mode"
fi

echo ""
print_status "All tests completed successfully! 🎉"
print_warning "Coverage report generated in htmlcov/index.html" 