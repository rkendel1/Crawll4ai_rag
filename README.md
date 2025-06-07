<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling, PDF Processing, and RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [Supabase](https://supabase.com/) for providing AI agents and AI coding assistants with advanced web crawling, document processing, and RAG capabilities.

With this MCP server, you can <b>scrape anything</b>, <b>ingest PDF documents</b>, <b>process OpenAPI specifications</b>, and then <b>use that knowledge anywhere</b> for RAG.

The primary goal is to bring this MCP server into [Archon](https://github.com/coleam00/Archon) as I evolve it to be more of a knowledge engine for AI coding assistants to build AI agents. This first version of the Crawl4AI/RAG MCP server will be improved upon greatly soon, especially making it more configurable so you can use different embedding models and run everything locally with Ollama.

## Overview

This MCP server provides tools that enable AI agents to crawl websites, ingest PDF documents, process OpenAPI specifications, store content in a vector database (Supabase), and perform RAG over all types of ingested content. It follows the best practices for building MCP servers based on the [Mem0 MCP server template](https://github.com/coleam00/mcp-mem0/) I provided on my channel previously.

The server includes several advanced RAG strategies that can be enabled to enhance retrieval quality:
- **Contextual Embeddings** for enriched semantic understanding
- **Hybrid Search** combining vector and keyword search
- **Agentic RAG** for specialized code example extraction
- **Reranking** for improved result relevance using cross-encoder models

See the [Configuration section](#configuration) below for details on how to enable and configure these strategies.

## Vision

The Crawl4AI RAG MCP server is just the beginning. Here's where we're headed:

1. **Integration with Archon**: Building this system directly into [Archon](https://github.com/coleam00/Archon) to create a comprehensive knowledge engine for AI coding assistants to build better AI agents.

2. **Multiple Embedding Models**: Expanding beyond OpenAI to support a variety of embedding models, including the ability to run everything locally with Ollama for complete control and privacy.

3. **Advanced RAG Strategies**: Implementing sophisticated retrieval techniques like contextual retrieval, late chunking, and others to move beyond basic "naive lookups" and significantly enhance the power and precision of the RAG system, especially as it integrates with Archon.

4. **Enhanced Chunking Strategy**: Implementing a Context 7-inspired chunking approach that focuses on examples and creates distinct, semantically meaningful sections for each chunk, improving retrieval precision.

5. **Performance Optimization**: Increasing crawling and indexing speed to make it more realistic to "quickly" index new documentation to then leverage it within the same prompt in an AI coding assistant.

## Features

### Web Crawling
- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
- **Recursive Crawling**: Follows internal links to discover content
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously

### Document Processing
- **PDF Ingestion**: Extract text from PDF documents with intelligent chunking and page awareness
- **OpenAPI Processing**: Parse and index OpenAPI/Swagger specifications with multiple chunking strategies
- **Multiple Formats**: Support for JSON and YAML OpenAPI specifications

### Advanced RAG
- **Content Chunking**: Intelligently splits content by headers and size for better processing
- **Vector Search**: Performs RAG over all content types, optionally filtering by data source for precision
- **Specialized Search**: Dedicated API endpoint search for technical documentation
- **Source Management**: Retrieve and manage available sources for targeted queries

## Tools

The server provides a comprehensive set of tools for web crawling, document ingestion, and knowledge retrieval:

### Web Crawling Tools

1. **`crawl_single_page`**: Quickly crawl a single web page and store its content in the vector database
2. **`smart_crawl_url`**: Intelligently crawl a full website based on the type of URL provided (sitemap, llms-full.txt, or a regular webpage that needs to be crawled recursively)

### Document Ingestion Tools

3. **`ingest_pdf`**: Extract and index content from PDF documents with intelligent chunking and page tracking
4. **`ingest_openapi`**: Parse and index OpenAPI specifications with configurable chunking strategies

### Knowledge Retrieval Tools

5. **`get_available_sources`**: Get a list of all available sources (domains) in the database
6. **`perform_rag_query`**: Search for relevant content using semantic search with optional source filtering
7. **`search_api_endpoints`**: Search specifically for API endpoints from ingested OpenAPI specifications
8. **`list_ingested_apis`**: List all OpenAPI specifications that have been ingested into the system

### Conditional Tools

9. **`search_code_examples`** (requires `USE_AGENTIC_RAG=true`): Search specifically for code examples and their summaries from crawled documentation. This tool provides targeted code snippet retrieval for AI coding assistants.

---

### Detailed Tool Documentation

#### Document Ingestion

##### `ingest_pdf`
Ingest PDF documents into the vector database for RAG queries.

**Parameters:**
- `file_path` (string, required): Path to the PDF file
- Automatically detects absolute vs relative paths
- Supports fallback extraction methods for maximum compatibility

**Features:**
- Page-aware chunking with configurable word count and overlap
- Metadata extraction including page numbers, file size, and extraction timestamps
- Fallback to PyPDF2 if primary extraction method fails
- Integration with all existing RAG strategies (contextual embeddings, hybrid search, etc.)

**Example:**
```python
result = await mcp.ingest_pdf("/docs/api-documentation.pdf")
# Returns: JSON with success status, chunk count, page count, and processing details
```

##### `ingest_openapi`
Ingest OpenAPI specifications with intelligent chunking strategies.

**Parameters:**
- `file_path` (string, required): Path to the OpenAPI spec file (JSON or YAML)
- `strategy` (string, optional): Chunking strategy (default: "combined")
  - `"endpoint"`: Create chunks for each API endpoint with full documentation
  - `"schema"`: Create chunks for each data model/schema definition
  - `"combined"`: Create chunks for both endpoints and schemas plus overview and security (recommended)
  - `"operation"`: Group endpoints by operationId for related operations

**Features:**
- Full $ref resolution for complex specifications
- Support for both OpenAPI 3.x and Swagger 2.x formats
- Comprehensive endpoint documentation with parameters, request/response bodies, and security
- Schema relationship tracking and usage examples
- Security scheme documentation extraction

**Examples:**
```python
# Basic ingestion with default strategy
result = await mcp.ingest_openapi("/specs/payment-api.yaml")

# Endpoint-focused ingestion for API reference
result = await mcp.ingest_openapi("/specs/user-api.json", strategy="endpoint")

# Schema-focused ingestion for data model documentation
result = await mcp.ingest_openapi("/specs/models.yaml", strategy="schema")
```

#### Specialized Search

##### `search_api_endpoints`
Search for specific API endpoints using semantic search.

**Parameters:**
- `query` (string, required): Search query describing the functionality
- `api_name` (string, optional): Filter results by specific API name
- `match_count` (integer, optional): Maximum results to return (default: 10)

**Features:**
- Semantic search across endpoint descriptions, summaries, and documentation
- Filtering by API name for targeted searches
- Returns comprehensive endpoint details including HTTP methods, parameters, and responses
- Integration with hybrid search and reranking when enabled

**Examples:**
```python
# Find payment-related endpoints
endpoints = await mcp.search_api_endpoints("payment processing")

# Search within specific API
endpoints = await mcp.search_api_endpoints("user authentication", api_name="User Service API")

# Find all POST endpoints
endpoints = await mcp.search_api_endpoints("POST create", match_count=20)
```

##### `list_ingested_apis`
Discover all APIs available in the system.

**Features:**
- Lists all ingested OpenAPI specifications with metadata
- Provides API versions, descriptions, and ingestion timestamps
- Helps users understand what documentation is available for queries
- Shows chunk counts and processing statistics

**Example:**
```python
apis = await mcp.list_ingested_apis()
# Returns: List of APIs with titles, versions, descriptions, and ingestion details
```

## Prerequisites

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/) if running the MCP server as a container (recommended)
- [Python 3.12+](https://www.python.org/downloads/) if running the MCP server directly through uv
- [Supabase](https://supabase.com/) (database for RAG)
- [OpenAI API key](https://platform.openai.com/api-keys) (for generating embeddings)

## Installation

### Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Build the Docker image:
   ```bash
   docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
   ```

3. Create a `.env` file based on the configuration section below

### Using uv directly (no Docker)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   .venv\Scripts\activate
   # on Mac/Linux: source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -e .
   crawl4ai-setup
   ```

5. Create a `.env` file based on the configuration section below

## Database Setup

Before running the server, you need to set up the database with the pgvector extension:

1. Go to the SQL Editor in your Supabase dashboard (create a new project first if necessary)

2. Create a new query and paste the contents of `crawled_pages.sql`

3. Run the query to create the necessary tables and functions

## Configuration

Create a `.env` file in the project root with the following variables:

```
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key

# LLM for summaries and contextual embeddings
MODEL_CHOICE=gpt-4.1-nano

# RAG Strategies (set to "true" or "false", default to "false")
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key

# Document Processing Configuration (optional)
PDF_CHUNK_SIZE=1000
PDF_CHUNK_OVERLAP=200
OPENAPI_DEFAULT_STRATEGY=combined
```

### RAG Strategy Options

The Crawl4AI RAG MCP server supports four powerful RAG strategies that can be enabled independently:

#### 1. **USE_CONTEXTUAL_EMBEDDINGS**
When enabled, this strategy enhances each chunk's embedding with additional context from the entire document. The system passes both the full document and the specific chunk to an LLM (configured via `MODEL_CHOICE`) to generate enriched context that gets embedded alongside the chunk content.

- **When to use**: Enable this when you need high-precision retrieval where context matters, such as technical documentation where terms might have different meanings in different sections.
- **Trade-offs**: Slower indexing due to LLM calls for each chunk, but significantly better retrieval accuracy.
- **Cost**: Additional LLM API calls during indexing.

#### 2. **USE_HYBRID_SEARCH**
Combines traditional keyword search with semantic vector search to provide more comprehensive results. The system performs both searches in parallel and intelligently merges results, prioritizing documents that appear in both result sets.

- **When to use**: Enable this when users might search using specific technical terms, function names, or when exact keyword matches are important alongside semantic understanding.
- **Trade-offs**: Slightly slower search queries but more robust results, especially for technical content.
- **Cost**: No additional API costs, just computational overhead.

#### 3. **USE_AGENTIC_RAG**
Enables specialized code example extraction and storage. When crawling documentation, the system identifies code blocks (≥300 characters), extracts them with surrounding context, generates summaries, and stores them in a separate vector database table specifically designed for code search.

- **When to use**: Essential for AI coding assistants that need to find specific code examples, implementation patterns, or usage examples from documentation.
- **Trade-offs**: Significantly slower crawling due to code extraction and summarization, requires more storage space.
- **Cost**: Additional LLM API calls for summarizing each code example.
- **Benefits**: Provides a dedicated `search_code_examples` tool that AI agents can use to find specific code implementations.

#### 4. **USE_RERANKING**
Applies cross-encoder reranking to search results after initial retrieval. Uses a lightweight cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to score each result against the original query, then reorders results by relevance.

- **When to use**: Enable this when search precision is critical and you need the most relevant results at the top. Particularly useful for complex queries where semantic similarity alone might not capture query intent.
- **Trade-offs**: Adds ~100-200ms to search queries depending on result count, but significantly improves result ordering.
- **Cost**: No additional API costs - uses a local model that runs on CPU.
- **Benefits**: Better result relevance, especially for complex queries. Works with both regular RAG search and code example search.

### Recommended Configurations

**For general documentation RAG:**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=true
```

**For AI coding assistant with code examples:**
```
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
```

**For fast, basic RAG:**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=false
```

### Document Processing Configuration

The server supports additional configuration for PDF and OpenAPI processing:

#### PDF Processing Options

- **`PDF_CHUNK_SIZE`** (default: 1000): Number of words per PDF chunk
- **`PDF_CHUNK_OVERLAP`** (default: 200): Number of overlapping words between chunks

**When to adjust:**
- Increase chunk size for large documents with continuous content
- Increase overlap for better context preservation across chunks
- Decrease chunk size for more granular search results

#### OpenAPI Processing Options

- **`OPENAPI_DEFAULT_STRATEGY`** (default: "combined"): Default chunking strategy for OpenAPI ingestion

**Available strategies:**
- `"endpoint"`: Optimize for API endpoint documentation
- `"schema"`: Optimize for data model documentation  
- `"combined"`: Balanced approach with all content types (recommended)
- `"operation"`: Group by operation ID for related endpoints

### Recommended Configurations by Use Case

**For API documentation with mixed content:**
```
PDF_CHUNK_SIZE=800
PDF_CHUNK_OVERLAP=150
OPENAPI_DEFAULT_STRATEGY=combined
USE_HYBRID_SEARCH=true
USE_RERANKING=true
```

**For large technical manuals:**
```
PDF_CHUNK_SIZE=1500
PDF_CHUNK_OVERLAP=300
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
```

**For API reference documentation:**
```
OPENAPI_DEFAULT_STRATEGY=endpoint
USE_AGENTIC_RAG=true
USE_HYBRID_SEARCH=true
USE_RERANKING=true
```

## Running the Server

### Using Docker

```bash
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Using Python

```bash
uv run src/crawl4ai_mcp.py
```

The server will start and listen on the configured host and port.

## Usage Examples

This section provides comprehensive examples for using the new PDF and OpenAPI ingestion capabilities.

### PDF Document Ingestion

#### Basic PDF Ingestion
```bash
# In Claude, Cursor, or any MCP client:
"Please ingest the API documentation at /docs/payment-api.pdf"
```

**Expected Response:**
```json
{
  "success": true,
  "file_path": "/docs/payment-api.pdf",
  "filename": "payment-api.pdf",
  "chunks_stored": 45,
  "total_pages": 12,
  "total_words": 8750,
  "source_id": "payment-api.pdf"
}
```

#### Batch PDF Processing
```bash
# Process multiple PDFs in a directory
"Ingest all PDF files from the /documentation/apis/ folder"

# The assistant would then process each PDF individually:
# - payment-gateway.pdf
# - user-management.pdf 
# - webhook-integration.pdf
```

#### PDF with Custom Configuration
```bash
# For large technical manuals with dense content
"Ingest the technical manual at /manuals/system-architecture.pdf. 
Use larger chunk sizes since this is a comprehensive document."
```

### OpenAPI Specification Ingestion

#### Basic OpenAPI Ingestion
```bash
# JSON format
"Please ingest the OpenAPI specification at /specs/user-service.json"

# YAML format  
"Ingest the payment API spec from /specs/payment-api.yaml"
```

**Expected Response:**
```json
{
  "success": true,
  "file_path": "/specs/payment-api.yaml",
  "filename": "payment-api.yaml",
  "api_title": "Payment Processing API",
  "api_version": "2.1.0",
  "strategy": "combined",
  "chunks_stored": 23,
  "chunk_breakdown": {
    "overview": 1,
    "endpoint": 8,
    "schema": 12,
    "security": 2
  },
  "total_words": 5420,
  "source_id": "Payment Processing API"
}
```

#### Strategy-Specific Ingestion

**Endpoint-Focused Strategy:**
```bash
"Ingest the API spec at /specs/rest-api.yaml using the endpoint strategy 
for better API reference documentation"
```

**Schema-Focused Strategy:**
```bash
"Process the data models specification at /specs/models.json using 
the schema strategy to focus on data structures"
```

**Operation-Grouped Strategy:**
```bash
"Ingest /specs/complex-api.yaml using the operation strategy to group 
related endpoints by their operation IDs"
```

### Querying Ingested Content

#### General RAG Queries
```bash
# Search across all ingested content
"How do I authenticate with the payment API?"

# Search with source filtering
"Find information about user roles in the User Management API documentation"

# Complex queries spanning multiple documents
"What are the security requirements for payment processing across all our APIs?"
```

#### API-Specific Searches
```bash
# Find specific endpoints
"Show me all endpoints related to payment processing"

# Search for specific HTTP methods
"Find all POST endpoints that create new resources"

# Look for authentication endpoints
"What endpoints are available for user authentication and authorization?"
```

#### Advanced Query Examples

**Discovering Available APIs:**
```bash
"What APIs have been ingested into the system?"
```

**Finding Implementation Examples:**
```bash
"Show me code examples for integrating with the payment gateway"
# (when USE_AGENTIC_RAG=true)
```

**Schema and Data Model Queries:**
```bash
"What fields are required for creating a new user?"

"Show me the complete data model for payment transactions"

"What are the validation rules for customer information?"
```

### Workflow Examples

#### Complete API Documentation Workflow
```bash
1. "First, list what APIs are currently available in the system"

2. "Ingest the new Payment API v3 specification from /specs/payment-v3.yaml"

3. "What are the main differences between the payment endpoints in v2 and v3?"

4. "Show me all webhook-related endpoints in the Payment API"

5. "Generate integration examples for processing credit card payments"
```

#### Technical Documentation Workflow
```bash
1. "Ingest the system architecture guide from /docs/architecture.pdf"

2. "Process the API specifications from /specs/ using the combined strategy"

3. "How does the authentication system work across all services?"

4. "What are the recommended practices for error handling based on the documentation?"

5. "Create a summary of all security considerations mentioned in the docs"
```

#### Development Support Workflow
```bash
1. "List all available APIs to see what documentation we have"

2. "Find all endpoints related to user management"

3. "What are the required parameters for creating a new user account?"

4. "Show me example request/response payloads for user creation"

5. "What error codes should I handle when implementing user registration?"
```

### Integration Patterns

#### With AI Coding Assistants
```bash
# Claude Code / Cursor workflow
1. "Ingest our internal API docs from /company/api-docs.pdf"
2. "Based on the API documentation, help me implement user authentication"
3. "Generate TypeScript interfaces for all the user-related endpoints"
4. "Create error handling for all the possible API error responses"
```

#### With Documentation Systems
```bash
# Automated documentation updates
1. "Ingest the latest API specifications from /deployment/specs/"
2. "Compare the new API structure with what's currently documented"
3. "Identify any breaking changes or new endpoints"
4. "Generate migration guides for developers"
```

#### With Testing Frameworks
```bash
# Test generation from API specs
1. "Process the API specification to understand all endpoints"
2. "What test cases should I write for the payment processing endpoints?"
3. "Generate sample test data based on the API schemas"
4. "What edge cases should I test based on the API documentation?"
```

### Troubleshooting Common Scenarios

#### PDF Processing Issues
```bash
# If PDF extraction fails
"The PDF ingestion failed. Try processing /docs/manual.pdf with fallback extraction methods"

# For scanned PDFs or complex layouts
"This PDF seems to have complex formatting. Can you extract what text is available?"
```

#### OpenAPI Processing Issues
```bash
# For invalid specifications
"The OpenAPI spec validation failed. What are the specific issues with /specs/api.yaml?"

# For large specifications
"This OpenAPI spec is very large. Process it using the endpoint strategy to focus on API reference"
```

#### Search and Retrieval Issues
```bash
# When searches return no results
"List all available sources to see what documentation is indexed"

# For improving search results
"Use hybrid search to find information about authentication tokens"
```

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "crawl4ai-rag": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8051/sse"
>     }
>   }
> }
> ```
>
> **Note for Docker users**: Use `host.docker.internal` instead of `localhost` if your client is running in a different container. This will apply if you are using this MCP server within n8n!

### Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "python",
      "args": ["path/to/crawl4ai-mcp/src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "-e", "OPENAI_API_KEY", 
               "-e", "SUPABASE_URL", 
               "-e", "SUPABASE_SERVICE_KEY", 
               "mcp/crawl4ai"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

## Building Your Own Server

This implementation provides a foundation for building more complex MCP servers with web crawling capabilities. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies
3. Modify the `utils.py` file for any helper functions you need
4. Extend the crawling capabilities by adding more specialized crawlers