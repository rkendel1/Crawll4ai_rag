import os
import json
from json import JSONDecodeError  # Added JSONDecodeError
from typing import List, Optional, Any, Dict
import logging
import boto3
from botocore.exceptions import ClientError
_bedrock_client = None
logger = logging.getLogger(__name__)


def get_bedrock_client(region: Optional[str] = None):
    global _bedrock_client
    if _bedrock_client is not None:
        return _bedrock_client
    effective_region = region or os.getenv("AWS_REGION")
    _bedrock_client = boto3.client(
        "bedrock-runtime", region_name=effective_region)
    return _bedrock_client


def create_titan_embeddings_batch(texts: List[str], region: Optional[str] = None) -> List[List[float]]:
    """
    Crea embeddings usando AWS Bedrock Titan para una lista de textos.
    Args:
        texts: Lista de textos a embeddar
        region: Región AWS (opcional)
    Returns:
        Lista de embeddings (cada embedding es una lista de floats)
    """
    if not texts:
        return []
    client = get_bedrock_client(region)
    model_id = "amazon.titan-embed-text-v1"
    embeddings = []
    for text_item in texts:
        body = {"inputText": text_item}
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json"
            )
            response_body_bytes = response["body"].read()
            result = json.loads(response_body_bytes)
            embeddings.append(result["embedding"])
        except ClientError as ce:
            logger.error(
                "AWS ClientError creando embedding con Bedrock para '%s...': %s", text_item[:50], ce)
            embeddings.append([0.0] * 1536)
        except JSONDecodeError as jde:
            logger.error("JSONDecodeError creando embedding con Bedrock para '%s...': %s. Respuesta: %s", text_item[:50], jde, response.get(
                "body").read().decode('utf-8', errors='ignore') if response and hasattr(response.get("body"), "read") else "No response body")
            embeddings.append([0.0] * 1536)
        except Exception as e:  # General fallback
            logger.error("Error general creando embedding con Bedrock para '%s...': %s",
                         text_item[:50], e, exc_info=True)
            embeddings.append([0.0] * 1536)
    return embeddings


def create_titan_embedding(text: str, region: Optional[str] = None) -> List[float]:
    """
    Crea un embedding para un solo texto usando AWS Bedrock Titan.
    Args:
        text: Texto a embeddar
        region: Región AWS (opcional)
    Returns:
        Embedding como lista de floats
    """
    batch = create_titan_embeddings_batch([text], region=region)
    return batch[0] if batch else [0.0] * 1536


def invoke_bedrock_model(
    model_id: str,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    region: Optional[str] = None
) -> str:
    """
    Invokes the AWS Bedrock model (Claude or DeepSeek) for text generation.
    Args:
        model_id: The ID of the Bedrock model to use.
        prompt: The input prompt for the model.
        max_tokens: The maximum number of tokens to generate.
        temperature: Controls randomness in generation.
        top_p: Controls nucleus sampling.
        region: AWS region (optional).
    Returns:
        The generated text from the model.
    Raises:
        ValueError: If the model_id is not a supported model or an API error occurs.
    """
    client = get_bedrock_client(region)

    # Use Claude as default for Bedrock
    if model_id.startswith("anthropic.claude"):
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        content_type = "application/json"
        accept = "application/json"
        def extract_claude_response(r: Dict[str, Any]) -> Optional[str]:
            if r.get("content"):
                return r["content"]
            if r.get("completion"):
                return r["completion"]
            return None
        response_body_raw_bytes = None
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                accept=accept,
                contentType=content_type
            )
            response_body_raw_bytes = response.get("body").read()
            if not response_body_raw_bytes:
                raise ValueError(f"Received empty response body from model {model_id}.")
            response_body = json.loads(response_body_raw_bytes.decode('utf-8'))
            generated_text = extract_claude_response(response_body)
            if generated_text is None:
                raise ValueError(f"Failed to parse or extract generated text from model {model_id}. Response body: {response_body}")
            return generated_text
        except Exception as e:
            logger.error("Error invoking Bedrock Claude model %s: %s", model_id, e, exc_info=True)
            if response_body_raw_bytes is not None:
                logger.error("Raw response body from %s: %s", model_id, response_body_raw_bytes.decode('utf-8', errors='ignore'))
            raise ValueError(f"Error invoking Bedrock Claude model {model_id}: {e}") from e
    # Fallback: DeepSeek or other models (legacy)
    elif model_id.startswith("deepseek"):
        client = get_bedrock_client(region)

        # Ensure only deepseek models are processed
        if not model_id.startswith("deepseek"):  # Check for "deepseek" prefix.
            error_msg = f"Unsupported model_id: {model_id}. This function is exclusively for DeepSeek models (e.g., deepseek.r1-v1:0)."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Payload for deepseek models
        request_body: Dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        # Response parsing for deepseek models
        def parsed_response_extractor(r: Dict[str, Any]) -> Optional[str]:
            if r.get("choices") and isinstance(r.get("choices"), list) and len(r["choices"]) > 0:
                message = r["choices"][0].get("message")
                if message and message.get("role") == "assistant":
                    return message.get("content")
            logger.error(
                "Could not parse or find content in response from DeepSeek model '%s': %s", model_id, r)
            return None

        response_body_raw_bytes: Optional[bytes] = None
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                accept="application/json",
                contentType="application/json"
            )
            response_body_raw_bytes = response.get("body").read()
            if not response_body_raw_bytes:
                err_msg = f"Received empty response body from model {model_id}."
                logger.error(err_msg)
                raise ValueError(err_msg)

            response_body = json.loads(response_body_raw_bytes.decode(
                'utf-8'))  # Decode bytes to string before parsing
            generated_text = parsed_response_extractor(response_body)

            if generated_text is None:
                err_msg = f"Failed to parse or extract generated text from model {model_id}. Response body: {response_body}"
                logger.error(err_msg)
                raise ValueError(err_msg)
            return generated_text

        except ClientError as ce:
            logger.error("AWS ClientError invoking Bedrock model %s: %s",
                         model_id, ce, exc_info=True)
            logger.error("Request body sent to %s: %s",
                         model_id, json.dumps(request_body))
            if response_body_raw_bytes is not None:
                logger.error("Raw response body from %s: %s", model_id,
                             response_body_raw_bytes.decode('utf-8', errors='ignore'))
            raise ValueError(
                f"AWS ClientError invoking Bedrock model {model_id}: {ce}") from ce
        except JSONDecodeError as jde:
            logger.error("JSONDecodeError invoking Bedrock model %s: %s",
                         model_id, jde, exc_info=True)
            logger.error("Request body sent to %s: %s",
                         model_id, json.dumps(request_body))
            if response_body_raw_bytes is not None:
                logger.error("Raw response body from %s: %s", model_id,
                             response_body_raw_bytes.decode('utf-8', errors='ignore'))
            raise ValueError(
                f"JSONDecodeError invoking Bedrock model {model_id}: {jde}") from jde
        except Exception as e:
            logger.error("General error invoking Bedrock model %s: %s",
                         model_id, e, exc_info=True)
            logger.error("Request body sent to %s: %s",
                         model_id, json.dumps(request_body))
            if response_body_raw_bytes is not None:
                logger.error("Raw response body from %s: %s", model_id,
                             response_body_raw_bytes.decode('utf-8', errors='ignore'))
            raise ValueError(
                f"Error invoking Bedrock model {model_id}: {e}") from e
    else:
        raise ValueError(f"Unsupported model_id: {model_id}. Only Claude and DeepSeek are supported.")
