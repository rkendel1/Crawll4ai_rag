# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Server
```bash
# Using Python directly
uv run src/crawl4ai_mcp.py

# Using Docker
docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Installation & Setup
```bash
# Install dependencies with uv
uv pip install -e .
crawl4ai-setup  # Required after installation

# Database setup
# Run the SQL commands in crawled_pages.sql in your Supabase dashboard
```

### Development Environment
```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install in development mode
uv pip install -e .
```

## Architecture Overview

### Core Components

**MCP Server (`src/crawl4ai_mcp.py`)**
- FastMCP server with async lifespan management
- 9 main tools: crawl_single_page, smart_crawl_url, get_available_sources, perform_rag_query, search_code_examples, ingest_pdf, ingest_openapi, search_api_endpoints, list_ingested_apis
- Configurable RAG strategies via environment variables
- Context management with Crawl4AI, Supabase, and optional reranking models
- PDF and OpenAPI processor instances with configurable parameters

**Utilities (`src/utils.py`)**
- Supabase client management and database operations
- OpenAI embeddings with batch processing and retry logic
- Code block extraction with contextual information
- Hybrid search combining vector and keyword search
- Parallel processing for code example summarization

**Document Processors**
- `PDFProcessor` (`src/pdf_processor.py`): PDF text extraction with pdfplumber and PyPDF2 fallback, page-aware chunking, metadata tracking
- `OpenAPIProcessor` (`src/openapi_processor.py`): OpenAPI/Swagger spec parsing with $ref resolution, 4 chunking strategies (endpoint, schema, combined, operation), comprehensive documentation generation

**Database Schema (`crawled_pages.sql`)**
- Three main tables: `sources`, `crawled_pages`, `code_examples`
- PostgreSQL with pgvector extension for vector similarity search
- Custom stored procedures: `match_crawled_pages`, `match_code_examples`

### RAG Strategy System

The server supports four configurable RAG strategies:

1. **USE_CONTEXTUAL_EMBEDDINGS**: Enhances chunks with LLM-generated context from full documents
2. **USE_HYBRID_SEARCH**: Combines vector similarity with keyword search using PostgreSQL ILIKE
3. **USE_AGENTIC_RAG**: Extracts and indexes code examples separately with AI-generated summaries
4. **USE_RERANKING**: Applies cross-encoder models to reorder search results by relevance

### Data Flow

1. **Content Ingestion**: 
   - Web crawling: URLs processed through smart detection (sitemap, text file, or webpage)
   - PDF processing: Text extraction with page awareness and fallback methods
   - OpenAPI processing: Spec parsing with $ref resolution and strategic chunking
2. **Chunking**: Content split using appropriate strategies:
   - Web content: `smart_chunk_markdown` respecting code blocks and paragraphs  
   - PDF content: Word-based chunking with configurable overlap and page tracking
   - OpenAPI content: Strategy-based chunking (endpoint, schema, combined, operation)
3. **Processing**: Chunks optionally enhanced with contextual embeddings
4. **Storage**: Documents and metadata stored in Supabase with vector embeddings
5. **Search**: RAG queries use vector similarity, optional hybrid search, and reranking

### Key Design Patterns

**Batch Processing**: All operations (embeddings, database inserts, code processing) use batching with configurable sizes and retry logic.

**Parallel Execution**: Heavy operations like contextual embedding generation and code summarization use ThreadPoolExecutor for performance.

**Error Resilience**: Comprehensive retry logic with exponential backoff for API calls and database operations.

**Modular Configuration**: Environment variables control all RAG strategies and can be enabled independently.

## Configuration

### Required Environment Variables
```bash
# Core MCP Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse  # or stdio

# API Keys
OPENAI_API_KEY=your_key_here
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key

# LLM Model for contextual embeddings and summaries
MODEL_CHOICE=gpt-4o-mini  # or gpt-3.5-turbo, etc.

# RAG Strategy Toggles (default: false)
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false  
USE_AGENTIC_RAG=false
USE_RERANKING=false

# Document Processing Configuration (optional)
PDF_CHUNK_SIZE=1000
PDF_CHUNK_OVERLAP=200
OPENAPI_DEFAULT_STRATEGY=combined
```

### Database Requirements
- PostgreSQL with pgvector extension
- Run `crawled_pages.sql` to create required tables and functions
- Supabase service key must have read/write permissions

## Key Implementation Details

### Core Processing

**Embedding Model**: Hardcoded to `text-embedding-3-small` (1536 dimensions) - change in `utils.py:51` and database schema if needed.

**Code Extraction**: Minimum 1000 characters for code blocks, configurable in `utils.py:358`.

**Web Content Chunking**: Default 5000 characters with smart breaking at code blocks, paragraphs, and sentences.

**Concurrent Limits**: Default 10 concurrent browser sessions for crawling, 10 workers for code processing.

**Transport Modes**: Supports both SSE (Server-Sent Events) and stdio transports for MCP communication.

### Document Processing

**PDF Processing**:
- Primary: pdfplumber for better text extraction and layout preservation
- Fallback: PyPDF2 for maximum compatibility
- Page-aware chunking: Tracks which pages each chunk spans
- Configurable word-based chunking with overlap (default: 1000 words, 200 overlap)
- Metadata includes page count, file size, extraction timestamps

**OpenAPI Processing**:
- Full $ref resolution using Prance library
- Support for OpenAPI 3.x and Swagger 2.x formats
- Four chunking strategies:
  - `endpoint`: Individual chunks per API operation
  - `schema`: Individual chunks per data model  
  - `combined`: Endpoints + schemas + overview + security (default)
  - `operation`: Grouped by operationId
- Comprehensive documentation generation with parameters, request/response bodies, security schemes
- Schema relationship tracking and usage examples

### Tool Capabilities

**Document Ingestion**:
- `ingest_pdf`: PDF processing with fallback extraction, page tracking, configurable chunking
- `ingest_openapi`: OpenAPI spec processing with strategy selection, $ref resolution, format validation

**Specialized Search**:
- `search_api_endpoints`: Semantic search for API operations with filtering by API name
- `list_ingested_apis`: Discovery tool for available API documentation and metadata

**Testing and Validation**:
- Comprehensive test suite in `tests/test_ingestion.py` with colored output
- Sample files in `test_data/` for PDF, JSON, and YAML testing
- Error handling validation for missing files, invalid formats, malformed specifications