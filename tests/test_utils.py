"""
Unit tests for utility functions.
"""
import pytest
from unittest.mock import patch, MagicMock
import json

from utils import (
    smart_chunk_markdown, 
    extract_section_info,
    create_embedding,
    create_embeddings_batch,
    add_documents_to_supabase,
    search_documents
)


class TestMarkdownProcessing:
    """Test markdown processing functions."""
    
    def test_smart_chunk_markdown_basic(self, sample_markdown):
        """Test basic markdown chunking."""
        chunks = smart_chunk_markdown(sample_markdown, chunk_size=100)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 150 for chunk in chunks)  # Allow some overflow
        assert all(chunk.strip() for chunk in chunks)  # No empty chunks
    
    def test_smart_chunk_markdown_code_blocks(self):
        """Test chunking respects code block boundaries."""
        text = "Some text\n\n```python\ndef function():\n    pass\n```\n\nMore text"
        chunks = smart_chunk_markdown(text, chunk_size=30)
        
        # Should not break inside code blocks
        code_chunk = next((chunk for chunk in chunks if '```' in chunk), None)
        assert code_chunk is not None
        assert code_chunk.count('```') % 2 == 0  # Even number of backticks
    
    def test_smart_chunk_markdown_paragraphs(self):
        """Test chunking respects paragraph boundaries."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = smart_chunk_markdown(text, chunk_size=15)
        
        # Should break at paragraph boundaries when possible
        assert len(chunks) >= 2
    
    def test_extract_section_info(self):
        """Test section info extraction."""
        chunk = "# Main Header\n\n## Sub Header\n\nSome content here."
        info = extract_section_info(chunk)
        
        assert "headers" in info
        assert "char_count" in info
        assert "word_count" in info
        assert "# Main Header" in info["headers"]
        assert "## Sub Header" in info["headers"]
        assert info["char_count"] == len(chunk)
        assert info["word_count"] == len(chunk.split())


class TestEmbeddings:
    """Test embedding functions."""
    
    @patch('utils.openai.embeddings.create')
    def test_create_embedding(self, mock_openai, mock_openai_response):
        """Test single embedding creation."""
        mock_openai.return_value = mock_openai_response
        
        result = create_embedding("test text")
        
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)
        mock_openai.assert_called_once()
    
    @patch('utils.openai.embeddings.create')
    def test_create_embeddings_batch(self, mock_openai, mock_openai_response):
        """Test batch embedding creation."""
        mock_openai.return_value = mock_openai_response
        mock_openai_response.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(3)]
        
        texts = ["text 1", "text 2", "text 3"]
        results = create_embeddings_batch(texts)
        
        assert len(results) == 3
        assert all(len(emb) == 1536 for emb in results)
        mock_openai.assert_called_once_with(
            model="text-embedding-3-small",
            input=texts
        )
    
    @patch('utils.openai.embeddings.create')
    def test_create_embedding_error_handling(self, mock_openai):
        """Test embedding creation error handling."""
        mock_openai.side_effect = Exception("API Error")
        
        result = create_embedding("test text")
        
        # Should return zero embedding on error
        assert len(result) == 1536
        assert all(x == 0.0 for x in result)


class TestSupabaseOperations:
    """Test Supabase operations."""
    
    @patch('utils.create_embeddings_batch')
    def test_add_documents_to_supabase(self, mock_embeddings, mock_supabase_client):
        """Test adding documents to Supabase."""
        mock_embeddings.return_value = [[0.1] * 1536, [0.2] * 1536]
        
        urls = ["http://example.com/1", "http://example.com/2"]
        chunk_numbers = [0, 1]
        contents = ["Content 1", "Content 2"]
        metadatas = [{"key": "value1"}, {"key": "value2"}]
        url_to_full_document = {"http://example.com/1": "Full doc 1", "http://example.com/2": "Full doc 2"}
        
        add_documents_to_supabase(
            mock_supabase_client, 
            urls, 
            chunk_numbers, 
            contents, 
            metadatas, 
            url_to_full_document
        )
        
        # Verify delete was called
        mock_supabase_client.table.return_value.delete.return_value.in_.assert_called()
        
        # Verify insert was called
        mock_supabase_client.table.return_value.insert.assert_called()
    
    @patch('utils.create_embedding')
    def test_search_documents(self, mock_embedding, mock_supabase_client):
        """Test document search."""
        mock_embedding.return_value = [0.1] * 1536
        mock_supabase_client.rpc.return_value.execute.return_value.data = [
            {"url": "http://example.com", "content": "test", "similarity": 0.9}
        ]
        
        results = search_documents(
            mock_supabase_client,
            "test query",
            match_count=5
        )
        
        assert len(results) == 1
        assert results[0]["url"] == "http://example.com"
        mock_supabase_client.rpc.assert_called_once()
    
    @patch('utils.create_embedding')
    def test_search_documents_with_filter(self, mock_embedding, mock_supabase_client):
        """Test document search with metadata filter."""
        mock_embedding.return_value = [0.1] * 1536
        mock_supabase_client.rpc.return_value.execute.return_value.data = []
        
        search_documents(
            mock_supabase_client,
            "test query",
            match_count=5,
            filter_metadata={"source": "example.com"}
        )
        
        # Verify filter was passed to RPC call
        call_args = mock_supabase_client.rpc.call_args[1]
        assert "filter" in call_args
        assert call_args["filter"] == {"source": "example.com"} 