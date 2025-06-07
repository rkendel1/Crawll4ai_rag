# PRD - PDF and OpenAPI Support for Crawl4AI RAG MCP Servermd

### Summary
This PRD outlines the addition of PDF and OpenAPI specification ingestion capabilities to the existing Crawl4AI RAG MCP Server. These enhancements will enable the server to process and index private API documentation stored in PDF format and OpenAPI specifications, making them searchable through the existing RAG infrastructure.

### Background
The Crawl4AI RAG MCP Server currently supports web crawling and content indexing. Many organizations maintain their API documentation in PDF format or as OpenAPI specifications. Adding support for these formats will significantly expand the server's utility for developers working with private API documentation.

### Goals and Objectives
1. **Primary Goals**
   - Enable ingestion of PDF documents into the vector database
   - Support parsing and indexing of OpenAPI specifications (v3.x)
   - Maintain compatibility with existing RAG query functionality
   - Provide specialized search capabilities for API endpoints and schemas

2. **Success Metrics**
   - Successfully ingest PDFs up to 1000 pages
   - Parse OpenAPI specs with full $ref resolution
   - Maintain sub-2 second query latency
   - 95%+ accuracy in endpoint discovery queries

### User Stories
1. **As a developer**, I want to upload PDF API documentation so that I can search through it using natural language queries
2. **As a support engineer**, I want to ingest OpenAPI specs so that I can quickly find endpoint details and schemas
3. **As an AI coding assistant user**, I want to query API documentation to generate accurate code examples

### Functional Requirements

#### Phase 1: PDF Support
**FR-PDF-1**: PDF Ingestion
- System shall accept PDF files via file path
- System shall extract text content from all pages
- System shall preserve page structure and numbering
- System shall handle PDFs up to 1000 pages
- System shall introduce a new MCP tool: `ingest_pdf`

**FR-PDF-2**: PDF Chunking
- System shall split PDF content into chunks of ~1000 words
- System shall support the various RAG chunking strategies: Contextual Embeddings, Hybrid Search, Agentic RAG and Reranking
- System shall maintain 200-word overlap between chunks
- System shall preserve metadata (source, page numbers, extraction date)

**FR-PDF-3**: PDF Storage
- System shall generate embeddings for each chunk
- System shall store chunks in existing Supabase vector database
- System shall maintain source attribution for each chunk

#### Phase 2: OpenAPI Support
**FR-OAS-1**: OpenAPI Parsing
- System shall support OpenAPI 3.0.x specifications
- System shall handle both JSON and YAML formats
- System shall resolve all $ref references
- System shall validate spec structure
- System shall introduce a new MCP tool: `ingest_openapi`

**FR-OAS-2**: OpenAPI Chunking Strategies
- System shall support multiple chunking strategies:
  - By endpoint (each operation as a chunk)
  - By schema (each model definition as a chunk)
  - Combined (both endpoints and schemas)
- System shall generate comprehensive documentation for each chunk

**FR-OAS-3**: OpenAPI Metadata
- System shall extract and store:
  - API title and version
  - Endpoint paths and methods
  - Schema names and relationships
  - Operation IDs

**FR-OAS-4**: Specialized Search
- System shall provide endpoint-specific search functionality
- System shall enable filtering by API name
- System shall support schema relationship queries

### Non-Functional Requirements

**NFR-1**: Performance
- PDF processing: <30 seconds for 100-page document
- OpenAPI processing: <10 seconds for typical spec
- Query latency: <2 seconds for specialized searches

**NFR-2**: Scalability
- Support concurrent processing of multiple documents
- Handle OpenAPI specs up to 50MB in size

**NFR-3**: Reliability
- Graceful error handling for malformed documents
- Detailed error messages for troubleshooting
- Rollback capability for failed ingestions

**NFR-4**: Security
- No storage of sensitive data from examples
- Respect existing authentication mechanisms
- Audit logging for all ingestion operations

### Technical Requirements

**Dependencies**:
- PDF Processing: `pdfplumber>=0.10.0`, `pypdf2>=3.0.0`, `unstructured[pdf]>=0.10.0`
- OpenAPI Processing: `openapi-spec-validator>=0.6.0`, `prance>=23.0.0`, `jsonschema>=4.0.0`

**Integration Points**:
- Existing Supabase vector database
- Existing OpenAI embedding generation
- Existing MCP tool interface

### API Design

#### New MCP Tools

1. **ingest_pdf**
   ```python
   async def ingest_pdf(file_path: str) -> str
   ```
   - Input: Path to PDF file
   - Output: Success message with chunk count
   - Errors: File not found, parsing errors, database errors

2. **ingest_openapi**
   ```python
   async def ingest_openapi(file_path: str, strategy: str = "combined") -> str
   ```
   - Input: Path to OpenAPI spec, chunking strategy
   - Output: Success message with chunk count
   - Errors: Invalid spec, parsing errors, database errors

3. **search_api_endpoints**
   ```python
   async def search_api_endpoints(query: str, api_name: str = None) -> List[Dict]
   ```
   - Input: Search query, optional API filter
   - Output: List of matching endpoints with details
   - Errors: Database errors

### Data Model Extensions

**Metadata Schema for PDFs**:
```json
{
  "source": "string (file path)",
  "type": "pdf",
  "pages": "integer",
  "extracted_at": "timestamp",
  "chunk_index": "integer"
}
```

**Metadata Schema for OpenAPI**:
```json
{
  "source": "string (file path)",
  "type": "openapi",
  "version": "string (OpenAPI version)",
  "title": "string (API title)",
  "api_version": "string (API version)",
  "endpoint": "string (optional)",
  "operation_id": "string (optional)",
  "schema_name": "string (optional)"
}
```

### Implementation Phases

**Phase 1**: PDF Support
- Implement PDF processor module
- Add ingest_pdf MCP tool
- Test with various PDF formats
- Update documentation

**Phase 2**: OpenAPI Support
- Implement OpenAPI processor module
- Add ingest_openapi and search_api_endpoints tools
- Test with complex OpenAPI specs
- Update documentation

### Testing Requirements

**Unit Tests**:
- PDF text extraction accuracy
- OpenAPI $ref resolution
- Chunking strategies
- Error handling

**Integration Tests**:
- End-to-end ingestion workflows
- RAG query accuracy
- Performance benchmarks

**User Acceptance Tests**:
- Real API documentation ingestion
- Query result relevance
- Claude Code and Cursor MCP integration

### Documentation Requirements
- Update README with new tool usage
- Add examples for each ingestion type
- Document chunking strategies
- Provide troubleshooting guide