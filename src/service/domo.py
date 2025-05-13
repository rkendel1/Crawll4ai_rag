# here we need to generate a connection to the Domo API
# and create a class that will handle the connection
# We will also need methods for creating data in domo

import json
import logging
from pathlib import Path
from typing import Any, List, Dict
import base64
from urllib.parse import urlparse
import requests
import re

# Set up logging
logging.basicConfig(level=logging.INFO)

class DomoClient:
    def __init__(self, index_id: str, host: str, developer_token: str):
        """Initialize the DomoClient with environment variables and constants."""
        self.domo_host = host
        self.domo_developer_token = developer_token
        self.domo_base_url = f"https://{self.domo_host}/api"
        self.logger = logging.getLogger(__name__)
        self.index_id = index_id
        self.common_headers = {
            "X-DOMO-Developer-Token": self.domo_developer_token,
            "Accept": "application/json"
        }

    def create_vector_index(self, index_id: str) -> Dict[str, Any]:
        """
        Create a new vectorDB index.

        :param index_id: The name of the vector index.
        :return: The response from the API.
        """
        url = f"{self.domo_base_url}/api/recall/v1/indexes"
        body = {
            "indexId": index_id,
            "embeddingModel": "domo.domo_ai.domo-embed-text-multilingual-v1:cohere",
        }
        response = requests.post(url, headers=self.common_headers, json=body)
        response.raise_for_status()
        return response.json()

    def embed_image(self, file_id: str, embed_type: str = "image/png") -> List[float]:
        """
        Create an embedding for an uploaded image using Domo's vectorDB endpoints.

        :param file_id: The ID of the file to embed.
        :param domo_access_token: The Domo access token.
        :param media_type: The media type of the image (default is "image/png").
        :return: The vector embedding for the image.
        """
        headers = {"X-Domo-Developer-Token": self.domo_developer_token}
        result = requests.get(f"{self.domo_base_url}/data/v1/data-files/{file_id}", headers=headers)
        result.raise_for_status()

        blob = result.content
        base64_image = base64.b64encode(blob).decode('utf-8')

        # Validate the base64 string
        validation_result = self.is_valid_base64_image(base64_image)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid base64 image: {validation_result['error']}")

        image_embedding_url = f"{self.domo_base_url}/ai/v1/embedding/image"
        body = {
            "input": [
                {
                    "type": "base64",
                    "mediaType": embed_type,
                    "data": base64_image,
                }
            ],
            "model": "domo.domo_ai",
        }
        response = requests.post(image_embedding_url, headers=self.common_headers, json=body)
        response.raise_for_status()
        return response.json()["embeddings"][0]

    def upsert_text_embedding(self, markdown: List[str], url: str, tool_name: str, local_file_path: str, file_name_from_url: str) -> Dict[str, Any]:
        """
        Upsert a text embedding into the vectorDB.

        :param index_id: The ID of the vector index.
        :param text: The text to embed.
        :meta: The metadata to associate with the text.
        :return: The response from the API.
        """
        from src.utils.chunking import smart_chunk_markdown
        from src.utils.chunking import enrich_chunks_with_metadata

        chunks = smart_chunk_markdown(markdown, 1200)
        _, __, contents, metadatas = enrich_chunks_with_metadata(
            chunks=chunks,
            source_url=url,
            tool_name=tool_name,
            local_file_path=local_file_path,
            file_name_from_url=file_name_from_url,
        )
        
        nodes = []
        self.logger.info(f"Preparing {len(contents)} text embeddings to index {self.index_id}")
        for i in range(len(contents)):
            content = contents[i]
            meta = metadatas[i]

            # Create a document with content and metadata
            document = {
                "content": content,
                "metadata": meta
            }
            enriched_content = json.dumps(document)
            nodes.append({
                "content": enriched_content,
                "type": "TEXT",
            })

        url = f"{self.domo_base_url}/recall/v1/indexes/{self.index_id}/upsert"
        body = {
            "nodes": nodes
        }
        response = requests.post(url, headers=self.common_headers, json=body)
        response.raise_for_status()
        self.logger.info(f"Upserted {len(nodes)} text embeddings to index {self.index_id}")
        return response.json()
    def upsert_image_embedding(self, index_id: str, file_id: str, file_name: str) -> Dict[str, Any]:
        """
        Upsert an image embedding into the vectorDB.

        :param index_id: The ID of the vector index.
        :param file_id: The ID of the file to embed.
        :param file_name: The name of the file.
        :return: The response from the API.
        """
        embedding = self.embed_image(file_id, "IMAGE")

        url = f"{self.domo_base_url}/recall/v1/indexes/{index_id}/upsert"
        body = {
            "nodes": [
                {
                    "content": file_name,
                    "type": "IMAGE",
                    "embedding": embedding,
                    "properties": {
                        "file_id": file_id,
                    },
                }
            ]
        }
        response = requests.post(url, headers=self.common_headers, json=body)
        response.raise_for_status()
        return response.json()

    def upload_file(self, file, file_name: str, description: str, public: bool = True) -> Dict[str, Any]:
        """
        Upload a file to the Domo API.

        :param file: The file object to upload.
        :param file_name: The name of the file to be uploaded.
        :param description: A description for the file.
        :param public: Whether the file should be public (default is True).
        :return: The response from the API.
        """
        url = f"{self.domo_base_url}/data/v1/data-files?name={file_name}&public={str(public).lower()}&description={description}"
        files = {'file': (file_name, file)}
        response = requests.post(url, headers=self.common_headers, files=files)
        response.raise_for_status()
        return response.json()

    def is_valid_base64_image(self, base64_string: str) -> Dict[str, Any]:
        """
        Validate if a string is a valid base64 encoded image.

        :param base64_string: The base64 string to validate.
        :return: A dictionary indicating validity and format.
        """
        base64_regex = r'^[A-Za-z0-9+/]+={0,2}$'

        if len(base64_string) % 4 != 0:
            return {"valid": False, "error": "Length not a multiple of 4"}

        if not re.match(base64_regex, base64_string):
            return {"valid": False, "error": "Invalid base64 characters"}

        try:
            decoded = base64.b64decode(base64_string[:100], validate=True)
            signatures = {
                "jpeg": b"\xff\xd8\xff",
                "png": b"\x89PNG",
                "gif": b"GIF8",
                "bmp": b"BM",
                "webp": b"RIFF",
                "tiff": b"II*\x00",
            }

            for format, signature in signatures.items():
                if decoded.startswith(signature):
                    return {"valid": True, "format": format}

            return {"valid": True, "format": "unknown"}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def base64_to_file(self, base64_string: str) -> Any:
        """
        Convert a base64 string to a file object.

        :param base64_string: The base64 string to convert.
        :return: A file object.
        """
        mime_match = re.match(r'^data:(.+);base64,', base64_string)
        if not mime_match:
            raise ValueError("Invalid base64 string")

        mime_type = mime_match.group(1)
        extension = mime_type.split('/')[1]

        byte_string = base64.b64decode(base64_string.split(',')[1])
        file_name = f"file.{extension}"

        with open(file_name, "wb") as file:
            file.write(byte_string)

        return file_name

    def get_query_text_results(self, input_text: str, top_k: int) -> Dict[str, Any]:
        """
        Query the vectorDB index for text results.

        :param input_text: The input text to query.
        :param top_k: The number of top results to retrieve.
        :return: The response from the API.
        """
        try:
            url = f"{self.domo_base_url}/recall/v1/indexes/{self.index_id}/query"
            body = {
                "input": input_text,
                "topK": top_k
            }
            response = requests.post(url, headers=self.common_headers, json=body)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            self.logger.error(f"Error querying text results: {error}")
            raise

