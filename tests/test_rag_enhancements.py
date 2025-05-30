import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import json
import os

# Assuming PYTHONPATH is set to include the project root
from src.crawl4ai_mcp import extract_section_info
from src.utils import expand_query_with_llm, rerank_retrieved_documents

class TestExtractSectionInfo(unittest.TestCase):
    def test_hierarchical_headers(self):
        chunk = """
# Section 1
Some text
## Subsection 1.1
More text
### Subsubsection 1.1.1
Even more text
# Section 2
## Subsection 2.1
Final text
"""
        expected_header_path = [
            "Section 1",
            "Section 1 > Subsection 1.1",
            "Section 1 > Subsection 1.1 > Subsubsection 1.1.1",
            "Section 2",
            "Section 2 > Subsection 2.1",
        ]
        info = extract_section_info(chunk)
        self.assertEqual(info["header_path"], expected_header_path)
        self.assertIn("# Section 1; ## Subsection 1.1; ### Subsubsection 1.1.1; # Section 2; ## Subsection 2.1", info["headers"])

    def test_keyword_extraction(self):
        chunk = "This is a test chunk with some important test keywords. Keywords are good for search."
        # "important", "keywords", "search", "test", "chunk" (or "good") could be top 5 depending on exact TF logic for ties
        # STOP_WORDS includes "this", "is", "a", "with", "some", "are", "for"
        info = extract_section_info(chunk)
        self.assertIsNotNone(info["keywords"])
        self.assertEqual(len(info["keywords"]), 5) # TOP_N_KEYWORDS = 5
        self.assertIn("test", info["keywords"])
        self.assertIn("keywords", info["keywords"])
        self.assertNotIn("is", info["keywords"]) # stop word

    def test_no_headers(self):
        chunk = "This is a simple text without any headers."
        info = extract_section_info(chunk)
        self.assertIsNone(info["header_path"])
        self.assertEqual(info["headers"], "")

    def test_no_keywords(self):
        chunk = "a the is of and to" # All stop words
        info = extract_section_info(chunk)
        self.assertIsNone(info["keywords"])

    def test_empty_chunk(self):
        chunk = ""
        info = extract_section_info(chunk)
        self.assertIsNone(info["header_path"])
        self.assertIsNone(info["keywords"])
        self.assertEqual(info["char_count"], 0)
        self.assertEqual(info["word_count"], 0)

    def test_mixed_content_headers_and_keywords(self):
        chunk = """
# Title
This document is about apples and oranges.
Apples are red, oranges are orange.
## Fruit Types
Discussing apples.
### Apple Details
Green apples.
        """
        info = extract_section_info(chunk)
        self.assertEqual(info["header_path"], ["Title", "Title > Fruit Types", "Title > Fruit Types > Apple Details"])
        self.assertIn("apples", info["keywords"])
        self.assertIn("oranges", info["keywords"])


class TestExpandQueryWithLLM(unittest.IsolatedAsyncioTestCase):

    @patch('src.utils.aclient.chat.completions.create', new_callable=AsyncMock)
    async def test_expand_query_successful_json_list(self, mock_openai_call):
        mock_openai_call.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["expanded query 1", "expanded query 2"]'))]
        )
        result = await expand_query_with_llm("original query", "gpt-test")
        self.assertEqual(result, ["expanded query 1", "expanded query 2"])
        mock_openai_call.assert_called_once()

    @patch('src.utils.aclient.chat.completions.create', new_callable=AsyncMock)
    async def test_expand_query_successful_json_object(self, mock_openai_call):
        mock_openai_call.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"queries": ["expanded query 1", "query 2"]}'))]
        )
        result = await expand_query_with_llm("original query", "gpt-test")
        self.assertEqual(result, ["expanded query 1", "query 2"])

    async def test_expand_query_no_model_choice(self):
        result = await expand_query_with_llm("original query", "")
        self.assertEqual(result, [])

    @patch('src.utils.aclient.chat.completions.create', new_callable=AsyncMock)
    async def test_expand_query_api_failure(self, mock_openai_call):
        mock_openai_call.side_effect = Exception("API Error")
        result = await expand_query_with_llm("original query", "gpt-test")
        self.assertEqual(result, [])

    @patch('src.utils.aclient.chat.completions.create', new_callable=AsyncMock)
    async def test_expand_query_bad_json_response(self, mock_openai_call):
        mock_openai_call.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='not a json list'))]
        )
        # This might print an error but should return empty list
        result = await expand_query_with_llm("original query", "gpt-test")
        self.assertEqual(result, [])
    
    @patch('src.utils.aclient.chat.completions.create', new_callable=AsyncMock)
    async def test_expand_query_newline_fallback(self, mock_openai_call):
        # Test the weaker fallback if JSON parsing fails and it's not a JSON list
        # This depends on the specific fallback logic in expand_query_with_llm
        # For this test, assume it might try to split by newline if not JSON
        mock_openai_call.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='expanded query one\nanother query two'))]
        )
        # The current implementation heavily relies on `response_format={"type": "json_object"}`
        # so this fallback might be harder to trigger or might result in an empty list if strict JSON is expected.
        # If JSON parsing fails completely, it will print error and return [].
        # The fallback for newline was more of a safeguard.
        # Let's assume the strict JSON parsing leads to [] here as per current code.
        result = await expand_query_with_llm("original query", "gpt-test")
        # Depending on how robust the non-JSON parsing is, this might be [] or the parsed list.
        # Given `response_format={"type": "json_object"}`, if the response is not JSON, it's an error.
        # The fallback to newline split is weak and might be removed if `json_object` is reliable.
        # For now, expecting [] if it's not valid JSON.
        self.assertEqual(result, [])


class TestRerankRetrievedDocuments(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.documents = [
            {"id": "doc1", "content": "alpha content", "similarity": 0.8},
            {"id": "doc2", "content": "bravo content", "similarity": 0.9},
            {"id": "doc3", "content": "charlie content", "similarity": 0.7},
        ]

    @patch('src.utils.aclient.chat.completions.create', new_callable=AsyncMock)
    async def test_rerank_successful(self, mock_openai_call):
        # Mock responses for each document
        mock_openai_call.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="0.7"))]), # doc1
            MagicMock(choices=[MagicMock(message=MagicMock(content="0.95"))]),# doc2
            MagicMock(choices=[MagicMock(message=MagicMock(content="0.85"))]),# doc3
        ]
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "testkey"}):
            reranked_docs = await rerank_retrieved_documents(
                "query", self.documents, "gpt-reranker"
            )

        self.assertEqual(len(reranked_docs), 3)
        self.assertEqual(reranked_docs[0]["id"], "doc2") # score 0.95
        self.assertEqual(reranked_docs[1]["id"], "doc3") # score 0.85
        self.assertEqual(reranked_docs[2]["id"], "doc1") # score 0.7
        self.assertAlmostEqual(reranked_docs[0]["relevance_score"], 0.95)
        self.assertAlmostEqual(reranked_docs[1]["relevance_score"], 0.85)
        self.assertAlmostEqual(reranked_docs[2]["relevance_score"], 0.7)

    async def test_rerank_no_model_choice(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "testkey"}):
            reranked_docs = await rerank_retrieved_documents(
                "query", self.documents, ""
            )
        self.assertEqual(reranked_docs, self.documents) # Should return original

    async def test_rerank_no_api_key(self):
        # Temporarily remove OPENAI_API_KEY from environ for this test
        original_api_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            reranked_docs = await rerank_retrieved_documents(
                "query", self.documents, "gpt-reranker"
            )
            self.assertEqual(reranked_docs, self.documents) # Should return original
        finally:
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key


    @patch('src.utils.aclient.chat.completions.create', new_callable=AsyncMock)
    async def test_rerank_api_call_fails_for_some(self, mock_openai_call):
        mock_openai_call.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="0.9"))]), # doc1
            Exception("API Error for doc2"),                                  # doc2 fails
            MagicMock(choices=[MagicMock(message=MagicMock(content="0.8"))]), # doc3
        ]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "testkey"}):
            reranked_docs = await rerank_retrieved_documents(
                "query", self.documents, "gpt-reranker"
            )

        self.assertEqual(len(reranked_docs), 3)
        # Expected order: doc1 (0.9), doc3 (0.8), doc2 (0.0 default on error)
        self.assertEqual(reranked_docs[0]["id"], "doc1")
        self.assertEqual(reranked_docs[1]["id"], "doc3")
        self.assertEqual(reranked_docs[2]["id"], "doc2")
        self.assertAlmostEqual(reranked_docs[0]["relevance_score"], 0.9)
        self.assertAlmostEqual(reranked_docs[1]["relevance_score"], 0.8)
        self.assertAlmostEqual(reranked_docs[2]["relevance_score"], 0.0) # Default score

    @patch('src.utils.aclient.chat.completions.create', new_callable=AsyncMock)
    async def test_rerank_unparseable_score(self, mock_openai_call):
        mock_openai_call.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="high"))]), # Unparseable
            MagicMock(choices=[MagicMock(message=MagicMock(content="0.8"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="0.7"))]),
        ]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "testkey"}):
            reranked_docs = await rerank_retrieved_documents(
                "query", self.documents, "gpt-reranker"
            )
        # Expected order: doc2 (0.8), doc3 (0.7), doc1 (0.0 default)
        self.assertEqual(reranked_docs[0]["id"], "doc2")
        self.assertEqual(reranked_docs[1]["id"], "doc3")
        self.assertEqual(reranked_docs[2]["id"], "doc1")
        self.assertAlmostEqual(reranked_docs[0]["relevance_score"], 0.8)
        self.assertAlmostEqual(reranked_docs[1]["relevance_score"], 0.7)
        self.assertAlmostEqual(reranked_docs[2]["relevance_score"], 0.0) # Default score

    async def test_rerank_empty_documents_list(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "testkey"}):
            reranked_docs = await rerank_retrieved_documents(
                "query", [], "gpt-reranker"
            )
        self.assertEqual(reranked_docs, [])


if __name__ == '__main__':
    unittest.main()
