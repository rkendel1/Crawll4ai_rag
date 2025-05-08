from src.utils.embeddings import create_embeddings_batch, create_embedding, process_chunk_with_context

from supabase import create_client, Client as SupabaseClient
from typing import List, Dict, Any, Optional
import os
import concurrent.futures


def get_supabase_client(url, key) -> SupabaseClient:
    """
    Get a Supabase client with the URL and key from environment variables.
    
    Returns:
        Supabase client instance
    """
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")
    
    return create_client(supabase_url=url, supabase_key=key)

def _delete_existing_documents(client: SupabaseClient, unique_urls: List[str]) -> None:
    """Helper function to delete existing documents for given URLs."""
    if not unique_urls:
        return
    try:
        client.table("crawled_pages").delete().in_("url", unique_urls).execute()
        print(f"Successfully deleted existing records for {len(unique_urls)} URLs.")
    except Exception as e:
        print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
        for url in unique_urls:
            try:
                client.table("crawled_pages").delete().eq("url", url).execute()
            except Exception as inner_e:
                print(f"Error deleting record for URL {url}: {inner_e}")

def _generate_contextual_texts_for_batch(
    batch_original_contents: List[str],
    batch_urls: List[str],
    batch_metadatas: List[Dict[str, Any]], # This list will be modified in-place
    url_to_full_document: Dict[str, str]
) -> List[str]:
    """
    Generates contextual texts for a batch of contents.
    Modifies batch_metadatas in-place to add 'contextual_embedding_applied' flag.
    Returns a list of texts to be used for embedding.
    """
    print(f"Generating contextual embeddings for a batch of {len(batch_original_contents)} items...")
    tasks_args = []
    for k, original_chunk_content in enumerate(batch_original_contents):
        current_url = batch_urls[k]
        full_document_text = url_to_full_document.get(current_url, "")
        if not full_document_text:
            print(f"Warning: No full document found for URL {current_url}. Using chunk as full document for context.")
            full_document_text = original_chunk_content
        tasks_args.append((full_document_text, original_chunk_content))

    temp_contextual_contents = [None] * len(batch_original_contents)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {
            executor.submit(process_chunk_with_context, arg_pair[0], arg_pair[1]): idx 
            for idx, arg_pair in enumerate(tasks_args)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            original_index = future_to_index[future]
            try:
                contextual_text, success = future.result()
                temp_contextual_contents[original_index] = contextual_text
                batch_metadatas[original_index]["contextual_embedding_applied"] = success
            except Exception as exc:
                print(f"Chunk at original index {original_index} generated an exception: {exc}")
                temp_contextual_contents[original_index] = batch_original_contents[original_index]
                batch_metadatas[original_index]["contextual_embedding_applied"] = False
    
    # Ensure all placeholders were filled
    for idx, content in enumerate(temp_contextual_contents):
        if content is None: # Should not happen if logic is correct, but as a safeguard
            print(f"Warning: Contextual content at index {idx} was None. Falling back to original.")
            temp_contextual_contents[idx] = batch_original_contents[idx]
            batch_metadatas[idx]["contextual_embedding_applied"] = False
            
    return temp_contextual_contents

def _prepare_supabase_batch_data(
    batch_urls: List[str],
    batch_chunk_numbers: List[int],
    batch_original_contents: List[str],
    batch_embeddings: List[List[float]],
    batch_metadatas: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Prepares the list of dictionaries for Supabase insertion for a single batch."""
    batch_data_to_insert = []
    for j in range(len(batch_original_contents)):
        original_content_chunk = batch_original_contents[j]
        current_metadata = batch_metadatas[j]
        data = {
            "url": batch_urls[j],
            "chunk_number": batch_chunk_numbers[j],
            "content": original_content_chunk,
            "metadata": {
                "chunk_size": len(original_content_chunk),
                **current_metadata
            },
            "embedding": batch_embeddings[j]
        }
        batch_data_to_insert.append(data)
    return batch_data_to_insert

def add_documents_to_supabase(
    client: SupabaseClient, 
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20,
) -> None:
    """
    Add documents to the Supabase crawled_pages table in batches.
    Deletes existing records, generates contextual embeddings if configured,
    and inserts new data.
    """
    unique_urls = list(set(urls))
    _delete_existing_documents(client, unique_urls)
    
    model_choice = os.getenv("MODEL_CHOICE")
    use_contextual_embeddings = bool(model_choice)
    
    for i in range(0, len(contents), batch_size):
        batch_start_index = i
        batch_end_index = min(i + batch_size, len(contents))
        
        current_batch_urls = urls[batch_start_index:batch_end_index]
        current_batch_chunk_numbers = chunk_numbers[batch_start_index:batch_end_index]
        current_batch_original_contents = contents[batch_start_index:batch_end_index]
        current_batch_metadatas = metadatas[batch_start_index:batch_end_index] # Slicing creates a copy
        
        contents_for_embedding: List[str]
        if use_contextual_embeddings:
            contents_for_embedding = _generate_contextual_texts_for_batch(
                current_batch_original_contents,
                current_batch_urls,
                current_batch_metadatas, # Pass the sliced copy to be modified
                url_to_full_document
            )
        else:
            contents_for_embedding = list(current_batch_original_contents) # Use a copy
            for meta_item in current_batch_metadatas:
                 meta_item["contextual_embedding_applied"] = False
        
        batch_embeddings = create_embeddings_batch(contents_for_embedding)
        
        batch_data_to_insert = _prepare_supabase_batch_data(
            current_batch_urls,
            current_batch_chunk_numbers,
            current_batch_original_contents,
            batch_embeddings,
            current_batch_metadatas
        )
        
        if batch_data_to_insert:
            try:
                client.table("crawled_pages").insert(batch_data_to_insert).execute()
                print(f"Successfully inserted batch {i//batch_size + 1} ({len(batch_data_to_insert)} items) into Supabase.")
            except Exception as e:
                print(f"Error inserting batch {i//batch_size + 1} into Supabase: {e}")
        else:
            print(f"Batch {i//batch_size + 1} had no data to insert.")

def search_documents(
    client: SupabaseClient, 
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
            params['filter'] = filter_metadata  # Pass the dictionary directly, not JSON-encoded
        
        result = client.rpc('match_crawled_pages', params).execute()
        
        return result.data
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []