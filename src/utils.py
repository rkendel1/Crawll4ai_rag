"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
from supabase import create_client, Client
from urllib.parse import urlparse
import openai
from providers.aws_bedrock import create_titan_embeddings_batch, create_titan_embedding, invoke_bedrock_model
import logging

# Load OpenAI API key for embeddings
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_ACCESS_SECRET = os.getenv("AWS_ACCESS_SECRET")

CONTEXT_PROVIDER = os.getenv("CONTEXT_PROVIDER", "openai").lower()
BEDROCK_CONTEXT_MODEL_ID = os.getenv("BEDROCK_CONTEXT_MODEL_ID")

if EMBEDDINGS_PROVIDER == "bedrock" or CONTEXT_PROVIDER == "bedrock":
    # Set AWS credentials for boto3 if using Bedrock for embeddings or context
    if AWS_ACCESS_KEY and AWS_ACCESS_SECRET:
        os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY
        os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_ACCESS_SECRET
    if AWS_REGION:
        os.environ["AWS_REGION"] = AWS_REGION

# Set default to Claude if CONTEXT_PROVIDER is bedrock and BEDROCK_CONTEXT_MODEL_ID is not set or is deepseek
if CONTEXT_PROVIDER == "bedrock":
    if not BEDROCK_CONTEXT_MODEL_ID or BEDROCK_CONTEXT_MODEL_ID.startswith("deepseek"):
        BEDROCK_CONTEXT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"


def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.

    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")

    return create_client(url, key)


def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.
    Args:
        texts: List of texts to create embeddings for
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    if EMBEDDINGS_PROVIDER == "bedrock":
        region = AWS_REGION or "us-east-1"
        return create_titan_embeddings_batch(texts, region=region)
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error creating batch embeddings: {e}")
        return [[0.0] * 1536 for _ in range(len(texts))]


def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using the selected provider.
    Args:
        text: Text to create an embedding for
    Returns:
        List of floats representing the embedding
    """
    if EMBEDDINGS_PROVIDER == "bedrock":
        region = AWS_REGION or "us-east-1"
        return create_titan_embedding(text, region=region)
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return [0.0] * 1536


def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.

    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    logger = logging.getLogger("crawl4ai.context")
    model_choice = os.getenv("MODEL_CHOICE")  # This is for OpenAI model

    if not model_choice and CONTEXT_PROVIDER == "openai":
        logger.info("[CONTEXT] MODEL_CHOICE (for OpenAI context) is not set. Skipping contextual embedding.")
        return chunk, False

    if CONTEXT_PROVIDER == "bedrock" and not BEDROCK_CONTEXT_MODEL_ID:
        logger.info("[CONTEXT] CONTEXT_PROVIDER is 'bedrock' but BEDROCK_CONTEXT_MODEL_ID is not set. Skipping contextual embedding.")
        return chunk, False

    try:
        # Create the prompt for generating contextual information
        # Reduced full_document length to avoid overly long prompts for context generation
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        context = ""
        if CONTEXT_PROVIDER == "openai":
            if not model_choice:
                logger.info("[CONTEXT] OpenAI model_choice not set for contextual embedding. Skipping.")
                return chunk, False
            logger.info("[CONTEXT] Calling OpenAI model '%s' for context generation.", model_choice)
            response = openai.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200  # Max tokens for the generated context
            )
            context = response.choices[0].message.content.strip()

        elif CONTEXT_PROVIDER == "bedrock":
            if not BEDROCK_CONTEXT_MODEL_ID:
                logger.info("[CONTEXT] Bedrock context model ID not set. Skipping contextual embedding.")
                return chunk, False

            region = AWS_REGION or "us-east-1"  # Default to us-east-1 if not set
            logger.info("[CONTEXT] Calling Bedrock model '%s' for context generation in region '%s'.", BEDROCK_CONTEXT_MODEL_ID, region)
            # Call the centralized invoke_bedrock_model function
            # Max tokens here refers to the expected output length for the context.
            context = invoke_bedrock_model(
                model_id=BEDROCK_CONTEXT_MODEL_ID,
                prompt=prompt,
                max_tokens=200,  # Max tokens for the generated context
                temperature=0.3,
                # top_p is managed by invoke_bedrock_model default or specific model logic
                region=region
            ).strip()

        else:
            logger.info("[CONTEXT] Unknown CONTEXT_PROVIDER: %s. Skipping contextual embedding.", CONTEXT_PROVIDER)
            return chunk, False

        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        logger.info("[CONTEXT] Context generated successfully using provider '%s'.", CONTEXT_PROVIDER)

        return contextual_text, True

    except Exception as e:
        logger.error("[CONTEXT] Error generating contextual embedding: %s. Using original chunk instead.", e)
        return chunk, False


def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.

    Args:
        args: Tuple containing (url, content, full_document)

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)


def add_documents_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20
) -> None:
    """
    Add documents to the Supabase crawled_pages table in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.

    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))

    # Delete existing records for these URLs in a single operation
    try:
        if unique_urls:
            # Use the .in_() filter to delete all records with matching URLs
            client.table("crawled_pages").delete().in_(
                "url", unique_urls).execute()
    except Exception as e:
        print(
            f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
        # Fallback: delete records one by one
        for url in unique_urls:
            try:
                client.table("crawled_pages").delete().eq("url", url).execute()
            except Exception as inner_e:
                print(f"Error deleting record for URL {url}: {inner_e}")
                # Continue with the next URL even if one fails

    # Check if contextual embeddings should be used
    context_provider = os.getenv("CONTEXT_PROVIDER", "openai").lower()
    bedrock_context_model_id = os.getenv("BEDROCK_CONTEXT_MODEL_ID")
    model_choice = os.getenv("MODEL_CHOICE")
    use_contextual_embeddings = bool(model_choice) or (context_provider == "bedrock" and bedrock_context_model_id)

    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))

        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]

        # Apply contextual embedding to each chunk if MODEL_CHOICE is set
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for j, content in enumerate(batch_contents):
                url = batch_urls[j]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))

            # Process in parallel using ThreadPoolExecutor
            contextual_contents = []
            max_workers = os.getenv("MAX_WORKERS_FOR_SUPABASE", 1)  # Set max workers for threading
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(max_workers)) as executor:  # Reduced concurrency to avoid 429 errors
                # Submit all tasks and collect results
                future_to_idx = {executor.submit(process_chunk_with_context, arg): idx
                                 for idx, arg in enumerate(process_args)}

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        contextual_contents.append(result)
                        if success:
                            batch_metadatas[idx]["contextual_embedding"] = True
                    except Exception as e:
                        print(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        contextual_contents.append(batch_contents[idx])

            # Sort results back into original order if needed
            if len(contextual_contents) != len(batch_contents):
                print(
                    f"Warning: Expected {len(batch_contents)} results but got {len(contextual_contents)}")
                # Use original contents as fallback
                contextual_contents = batch_contents
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents

        # Create embeddings for the entire batch at once
        batch_embeddings = create_embeddings_batch(contextual_contents)

        batch_data = []
        for j in range(len(contextual_contents)):
            # Extract metadata fields
            chunk_size = len(contextual_contents[j])

            # Prepare data for insertion
            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": contextual_contents[j],  # Store original content
                "metadata": {
                    "chunk_size": chunk_size,
                    **batch_metadatas[j]
                },
                # Use embedding from contextual content
                "embedding": batch_embeddings[j]
            }

            batch_data.append(data)

        # Insert batch into Supabase
        try:
            client.table("crawled_pages").insert(batch_data).execute()
        except Exception as e:
            print(f"Error inserting batch into Supabase: {e}")


def search_documents(
    client: Client,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in Supabase using vector similarity.

    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter

    Returns:
        List of matching documents
    """
    # Create embedding for the query
    query_embedding = create_embedding(query)

    # Execute the search using the match_crawled_pages function
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count
        }

        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            # Pass the dictionary directly, not JSON-encoded
            params['filter'] = filter_metadata

        result = client.rpc('match_crawled_pages', params).execute()

        return result.data
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []
