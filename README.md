<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [Supabase](https://supabase.com/) for providing AI agents and AI coding assistants with advanced web crawling and RAG capabilities.

With this MCP server, you can <b>scrape anything</b> and then <b>use that knowledge anywhere</b> for RAG.

## Features

- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
- **Recursive Crawling**: Follows internal links to discover content
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously
- **Content Chunking**: Intelligently splits content by headers and size for better processing
- **Vector Search**: Performs RAG over crawled content, optionally filtering by data source for precision
- **Source Retrieval**: Retrieve sources available for filtering to guide the RAG process

## Tools

The server provides four essential web crawling and search tools:

1. **`crawl_single_page`**: Quickly crawl a single web page and store its content in the vector database
2. **`smart_crawl_url`**: Intelligently crawl a full website based on the type of URL provided (sitemap, llms-full.txt, or a regular webpage that needs to be crawled recursively)
3. **`get_available_sources`**: Get a list of all available sources (domains) in the database
4. **`perform_rag_query`**: Search for relevant content using semantic search with optional source filtering

## Prerequisites

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/) if running the MCP server as a container (recommended)
- [Python 3.12+](https://www.python.org/downloads/) if running the MCP server directly through uv
- [Supabase](https://supabase.com/) (database for RAG)
- API keys for your chosen providers:
  - [OpenAI API key](https://platform.openai.com/api-keys) (if using OpenAI for embeddings or context generation)
  - [AWS credentials](https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html) with access to Bedrock models (if using Bedrock for embeddings or context generation).
    - For embeddings, the server uses `amazon.titan-embed-text-v1`.
    - For context generation, the server is configured to use Claude 3.5 Sonnet via the `BEDROCK_CONTEXT_MODEL_ID` (e.g., `anthropic.claude-3-5-sonnet-20240620-v1:0`).

## Installation

### Using Docker (Recommended)

#### Pull from Docker Hub

The pre-built image is published to Docker Hub under the `ignaciocardenas/mcp-crawl4ai-rag-softworks` repository. To download the latest version:

```bash
docker pull ignaciocardenas/mcp-crawl4ai-rag-softworks:latest
```

#### Configure and Run with Custom Credentials

You can pass your credentials directly as environment variables when running the container:

```bash
# Run in detached mode, exposing port 8051 and passing creds
docker run -d \
  --name crawl4ai-rag \
  -p 8051:8051 \
  -e OPENAI_API_KEY=your_openai_api_key \
  -e SUPABASE_URL=your_supabase_url \
  -e SUPABASE_SERVICE_KEY=your_supabase_service_key \
  -e AWS_ACCESS_KEY_ID=your_aws_access_key_id \
  -e AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key \
  -e AWS_REGION=your_aws_region \
  -e EMBEDDINGS_PROVIDER="openai" \
  -e CONTEXT_PROVIDER="openai" \
  -e BEDROCK_CONTEXT_MODEL_ID="anthropic.claude-3-5-sonnet-20240620-v1:0" \
  ignaciocardenas/mcp-crawl4ai-rag-softworks:latest
```

Or load all variables from a local `.env` file (recommended for security):

```bash
# Create a .env file containing all required keys:
# # Server Transport Configuration:
# # TRANSPORT: "sse" or "stdio". 
# #   - "sse": Server runs as an HTTP server with Server-Sent Events. HOST and PORT are required.
# #   - "stdio": Server communicates over standard input/output. HOST and PORT are ignored.
# HOST=0.0.0.0 # Required if TRANSPORT is "sse". The host address for the SSE server.
# PORT=8051    # Required if TRANSPORT is "sse". The port for the SSE server.
# TRANSPORT=sse
# 
# # API and Service Credentials:
# OPENAI_API_KEY=...
# SUPABASE_URL=...
# SUPABASE_SERVICE_KEY=...
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# AWS_REGION=...
# 
# # Provider Configuration:
# # EMBEDDINGS_PROVIDER: "openai" or "bedrock". (Optional, default: "openai")
# #   - If "bedrock", uses 'amazon.titan-embed-text-v1'.
# EMBEDDINGS_PROVIDER="openai"
# # CONTEXT_PROVIDER: "openai" or "bedrock". (Optional, default: "openai")
# #   - If "bedrock", uses the model specified in BEDROCK_CONTEXT_MODEL_ID.
# CONTEXT_PROVIDER="openai"
# # BEDROCK_CONTEXT_MODEL_ID: (Required if CONTEXT_PROVIDER is "bedrock")
# #   Specifies the Claude 3.5 Sonnet model ID for context generation (e.g., "anthropic.claude-3-5-sonnet-20240620-v1:0").
# BEDROCK_CONTEXT_MODEL_ID="anthropic.claude-3-5-sonnet-20240620-v1:0"

docker run -d \
  --name crawl4ai-rag \
  --env-file ./path/to/.env \
  -p 8051:8051 \
  ignaciocardenas/mcp-crawl4ai-rag-softworks:latest
```

## Database Setup

Before running the server, you need to set up the database with the pgvector extension:

1. Go to the SQL Editor in your Supabase dashboard (create a new project first if necessary)

2. Create a new query and paste the following SQL code:

```sql
-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table crawled_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,  -- Added content column
    metadata jsonb not null default '{}'::jsonb,  -- Added metadata column
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
create index on crawled_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_crawled_pages_metadata on crawled_pages using gin (metadata);

CREATE INDEX idx_crawled_pages_source ON crawled_pages ((metadata->>'source'));

-- Create a function to search for documentation chunks
create or replace function match_crawled_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    content,
    metadata,
    1 - (crawled_pages.embedding <=> query_embedding) as similarity
  from crawled_pages
  where metadata @> filter
  order by crawled_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Enable RLS on the table
alter table crawled_pages enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on crawled_pages
  for select
  to public
  using (true);

```

3. Run the query to create the necessary tables and functions

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# MCP Server Transport Configuration
# TRANSPORT: "sse" or "stdio". 
#   - "sse": Server runs as an HTTP server with Server-Sent Events. HOST and PORT are required.
#   - "stdio": Server communicates over standard input/output. HOST and PORT are ignored.
TRANSPORT=sse

# HOST: Required if TRANSPORT is "sse". The host address for the SSE server.
HOST=0.0.0.0

# PORT: Required if TRANSPORT is "sse". The port for the SSE server.
PORT=8051

# OpenAI API Configuration (Required if using OpenAI models)
OPENAI_API_KEY=your_openai_api_key

# Supabase Configuration (Required for RAG)
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key

# AWS Configuration (Required if using Bedrock models)
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=your_aws_region

# Embeddings and Context Providers
# EMBEDDINGS_PROVIDER: "openai" or "bedrock". (Optional, default: "openai")
#   - If "bedrock", the server uses 'amazon.titan-embed-text-v1' for embeddings.
EMBEDDINGS_PROVIDER="openai"

# CONTEXT_PROVIDER: "openai" or "bedrock". (Optional, default: "openai")
#   - If "bedrock", the server uses the model specified in BEDROCK_CONTEXT_MODEL_ID for context generation.
CONTEXT_PROVIDER="openai"

# BEDROCK_CONTEXT_MODEL_ID: (Required if CONTEXT_PROVIDER is "bedrock")
#   Specifies the Claude 3.5 Sonnet model ID for context generation (e.g., "anthropic.claude-3-5-sonnet-20240620-v1:0").
BEDROCK_CONTEXT_MODEL_ID="anthropic.claude-3-5-sonnet-20240620-v1:0"
```

## Running the Server

### Using Docker

```bash
docker run -d --env-file .env -p 8051:8051 ignaciocardenas/mcp-crawl4ai-rag-softworks:latest
```

This command:

- Runs the container in detached mode (`-d`)
- Loads environment variables from the `.env` file
- Maps port 8051 from the container to port 8051 on the host
- Uses the latest version of the image

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
>
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
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key",
        "AWS_ACCESS_KEY_ID": "your_aws_access_key_id",
        "AWS_SECRET_ACCESS_KEY": "your_aws_secret_access_key",
        "AWS_REGION": "your_aws_region",
        "EMBEDDINGS_PROVIDER": "openai", 
        "CONTEXT_PROVIDER": "openai",
        "BEDROCK_CONTEXT_MODEL_ID": "anthropic.claude-3-5-sonnet-20240620-v1:0"
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
               "-e", "AWS_ACCESS_KEY_ID",
               "-e", "AWS_SECRET_ACCESS_KEY",
               "-e", "AWS_REGION",
               "-e", "EMBEDDINGS_PROVIDER",
               "-e", "CONTEXT_PROVIDER",
               "-e", "BEDROCK_CONTEXT_MODEL_ID",
               "ignaciocardenas/mcp-crawl4ai-rag-softworks:latest"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key",
        "AWS_ACCESS_KEY_ID": "your_aws_access_key_id",
        "AWS_SECRET_ACCESS_KEY": "your_aws_secret_access_key",
        "AWS_REGION": "your_aws_region",
        "EMBEDDINGS_PROVIDER": "openai",
        "CONTEXT_PROVIDER": "openai",
        "BEDROCK_CONTEXT_MODEL_ID": "anthropic.claude-3-5-sonnet-20240620-v1:0"
      }
    }
  }
}
```