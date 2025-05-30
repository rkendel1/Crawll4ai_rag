"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
import asyncio # Added for asyncio.to_thread if needed, and for async functions
from supabase import create_client, Client
from urllib.parse import urlparse
import openai
from openai import AsyncOpenAI # For async LLM calls

# Initialize OpenAI clients
# Synchronous client for existing functions (embeddings, contextual_embedding)
openai.api_key = os.getenv("OPENAI_API_KEY") 
# Asynchronous client for new query expansion function
# Ensure OPENAI_API_KEY is loaded before this
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.
    
    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")
    
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
        
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small", # Hardcoding embedding model for now, will change this later to be more dynamic
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error creating batch embeddings: {e}")
        # Return empty embeddings if there's an error
        return [[0.0] * 1536 for _ in range(len(texts))]

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
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
    model_choice = os.getenv("MODEL_CHOICE")
    
    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        # Extract the generated context
        context = response.choices[0].message.content.strip()
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

async def expand_query_with_llm(original_query: str, model_choice: str) -> List[str]:
    """
    Expand the original query with LLM-generated alternative formulations.
    Returns a list of expanded queries, or an empty list if failed.
    """
    if not model_choice or not original_query:
        return []

    prompt = f"""Given the following user query for information retrieval:
"{original_query}"

Please generate 2-3 alternative formulations or related questions that would be helpful for retrieving a broader set of relevant documents.
Return your answer as a JSON list of strings. For example:
["alternative query 1", "related question 2"]
Return only the JSON list and nothing else.
"""
    try:
        response = await aclient.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that expands user queries for information retrieval. You must respond with only a valid JSON list of strings."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
            response_format={"type": "json_object"} # Ensure JSON output
        )
        content = response.choices[0].message.content
        if content:
            # The response_format="json_object" should ensure it's a JSON object,
            # but the prompt asks for a list. Let's assume the LLM might wrap it in a key.
            # Or it might return a JSON string that is a list.
            try:
                # Attempt to parse the entire content as a JSON list
                expanded_queries = json.loads(content)
                if isinstance(expanded_queries, list) and all(isinstance(q, str) for q in expanded_queries):
                    return expanded_queries
                # If it's a dict, look for a key that contains a list of strings
                elif isinstance(expanded_queries, dict):
                    for key, value in expanded_queries.items():
                        if isinstance(value, list) and all(isinstance(q, str) for q in value):
                            return value
                print(f"LLM returned JSON, but not in the expected list format: {content}")
                return []
            except json.JSONDecodeError:
                # Fallback for cases where LLM doesn't strictly adhere to JSON list,
                # e.g. if it returns newline-separated strings despite the prompt.
                # This is less ideal if JSON was expected.
                # Given response_format="json_object", this fallback might be less necessary.
                cleaned_content = content.strip().replace("```json", "").replace("```", "").strip()
                if cleaned_content.startswith('[') and cleaned_content.endswith(']'):
                    try:
                        expanded_queries = json.loads(cleaned_content)
                        if isinstance(expanded_queries, list) and all(isinstance(q, str) for q in expanded_queries):
                            return expanded_queries
                    except json.JSONDecodeError:
                        print(f"Failed to parse LLM response as JSON list after cleaning: {cleaned_content}")
                        pass # Fall through to trying newline separation if robust parsing fails

                # As a last resort, try splitting by newlines if it's not a JSON list
                # This is a weaker fallback.
                if not (cleaned_content.startswith('[') and cleaned_content.endswith(']')):
                    queries = [q.strip() for q in cleaned_content.split('\n') if q.strip()]
                    # Filter out any potential non-query lines if the LLM added extra text
                    # (though it was instructed not to)
                    # A simple heuristic: queries are usually not extremely short or long.
                    queries = [q for q in queries if 2 < len(q.split()) < 20 and 'query' not in q.lower() and 'question' not in q.lower()]
                    if queries:
                        return queries
                
                print(f"Could not parse LLM response for query expansion: {content}")
                return []
        return []
    except Exception as e:
        print(f"Error calling LLM for query expansion: {e}")
        return []

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
            client.table("crawled_pages").delete().in_("url", unique_urls).execute()
    except Exception as e:
        print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
        # Fallback: delete records one by one
        for url in unique_urls:
            try:
                client.table("crawled_pages").delete().eq("url", url).execute()
            except Exception as inner_e:
                print(f"Error deleting record for URL {url}: {inner_e}")
                # Continue with the next URL even if one fails
    
    # Check if MODEL_CHOICE is set for contextual embeddings
    model_choice = os.getenv("MODEL_CHOICE")
    use_contextual_embeddings = bool(model_choice)
    
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
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
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
                print(f"Warning: Expected {len(batch_contents)} results but got {len(contextual_contents)}")
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
                "embedding": batch_embeddings[j]  # Use embedding from contextual content
            }
            
            batch_data.append(data)
        
        # Insert batch into Supabase
        try:
            client.table("crawled_pages").insert(batch_data).execute()
        except Exception as e:
            print(f"Error inserting batch into Supabase: {e}")

def search_documents(
    client: Client,
    queries: List[str], # Changed from query: str
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in Supabase using vector similarity for multiple queries.
    Consolidates and de-duplicates results.
    
    Args:
        client: Supabase client
        queries: List of query texts (original + expanded)
        match_count: Maximum number of results to return in the final list
        filter_metadata: Optional metadata filter
        
    Returns:
        Consolidated list of matching documents
    """
    if not queries:
        return []

    all_embeddings = create_embeddings_batch(queries)
    if not all_embeddings or len(all_embeddings) != len(queries):
        # Fallback to original query if batch embedding fails for some reason
        if queries:
             single_embedding = create_embedding(queries[0])
             if single_embedding and any(v != 0.0 for v in single_embedding): # Check if not empty embedding
                 all_embeddings = [single_embedding]
             else:
                 print("Failed to create embeddings for any query.")
                 return []
        else: # Should not happen if initial check `if not queries:` passes
            return []
            
    all_results = []
    # Determine how many results to fetch per query.
    # A simple approach: fetch match_count for the primary query,
    # and fewer for expanded queries to control total results before deduplication.
    # Or, fetch match_count for all and rely on deduplication and final trim.
    # For now, let's fetch match_count for each to maximize chances of finding relevant docs.
    
    for i, query_embedding in enumerate(all_embeddings):
        # Skip if embedding is effectively empty (all zeros)
        if not any(v != 0.0 for v in query_embedding):
            print(f"Skipping query '{queries[i]}' due to empty embedding.")
            continue

        try:
            params = {
                'query_embedding': query_embedding,
                'match_count': match_count # Fetch match_count for each query variant
            }
            if filter_metadata:
                params['filter'] = filter_metadata
            
            result = client.rpc('match_crawled_pages', params).execute()
            if result.data:
                all_results.extend(result.data)
        except Exception as e:
            print(f"Error searching documents for query '{queries[i]}': {e}")
            continue # Continue with other queries

    # Consolidate and de-duplicate results
    unique_documents = {}
    for doc in all_results:
        # Create a unique ID for each document chunk.
        # Assuming 'id' is the primary key from 'crawled_pages' table returned by RPC.
        # If 'id' is not available or not unique per chunk, use url + chunk_number.
        doc_id_val = doc.get('id') # Supabase typically returns an 'id' for each row.
        if doc_id_val is None: # Fallback if 'id' is not in the result
            doc_id_val = f"{doc.get('url')}_{doc.get('chunk_number', doc.get('metadata', {}).get('chunk_index'))}"


        if doc_id_val not in unique_documents:
            unique_documents[doc_id_val] = doc
        else:
            # Optional: If a document is found via multiple queries,
            # one might implement a re-scoring logic here.
            # For now, first-come, first-served.
            pass 
            
    consolidated_results = list(unique_documents.values())
    
    # Sort by similarity if available and desired.
    # The current Supabase RPC `match_crawled_pages` should return them sorted by similarity for each call.
    # After merging, this order is partially lost. A simple re-sort if similarity is consistent:
    # consolidated_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    # However, 'similarity' scores from different query embeddings might not be directly comparable.
    # For now, we rely on the initial sort and deduplication.

    # Trim to final match_count
    return consolidated_results[:match_count]

async def _get_relevance_score_for_document(
    doc_content: str, original_query: str, reranker_model_choice: str
) -> Optional[float]:
    """
    Helper async function to get relevance score for a single document.
    Returns a float score or None if scoring fails.
    """
    prompt = f"""Given the User Query and the Document content below:

User Query:
{original_query}

Document:
{doc_content[:4000]} # Truncate doc_content to avoid excessive token usage

Based on the Document's content, how relevant is it to the User Query?
Please provide a relevance score as a single floating-point number between 0.0 (not relevant) and 1.0 (highly relevant).
Output only the numerical score and nothing else. For example: 0.75
"""
    try:
        response = await aclient.chat.completions.create(
            model=reranker_model_choice,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that evaluates document relevance to a query and returns a numerical score.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=10, # Expecting just a number like "0.75"
        )
        score_text = response.choices[0].message.content.strip()
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score)) # Clamp score between 0.0 and 1.0
        except ValueError:
            print(f"Could not parse relevance score from LLM response: '{score_text}' for query '{original_query}'")
            return None
    except Exception as e:
        print(f"Error calling LLM for relevance scoring (query: '{original_query}'): {e}")
        return None

async def rerank_retrieved_documents(
    original_query: str,
    documents: List[Dict[str, Any]],
    reranker_model_choice: str,
    # api_key is implicitly used by the aclient, so not needed as a direct param if aclient is global
) -> List[Dict[str, Any]]:
    """
    Re-ranks retrieved documents based on LLM-evaluated relevance to the original query.
    """
    if not reranker_model_choice or not documents:
        print("Skipping reranking: No reranker model specified or no documents to rank.")
        return documents

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping reranking: OPENAI_API_KEY not found.")
        return documents

    print(f"Reranking {len(documents)} documents using model: {reranker_model_choice} for query: '{original_query}'")

    # Create a list of tasks for asyncio.gather
    tasks = [
        _get_relevance_score_for_document(
            doc.get("content", ""), original_query, reranker_model_choice
        )
        for doc in documents
    ]
    
    # Run all scoring tasks concurrently
    scores = await asyncio.gather(*tasks, return_exceptions=True) # return_exceptions to handle individual failures

    # Add scores to documents and handle potential errors from gather
    for i, doc in enumerate(documents):
        score_result = scores[i]
        if isinstance(score_result, Exception):
            print(f"Exception during scoring document {i}: {score_result}")
            doc["relevance_score"] = 0.0  # Assign default low score on error
        elif score_result is None:
            doc["relevance_score"] = 0.0 # Assign default low score if parsing failed or LLM error
        else:
            doc["relevance_score"] = score_result
            
    # Sort documents by relevance_score in descending order
    # If scores are equal, maintain original relative order (Python's sort is stable)
    documents.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    
    print(f"Reranking completed. Top scores: {[doc.get('relevance_score') for doc in documents[:5]]}")
    return documents