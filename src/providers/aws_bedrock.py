import os
import json
from json import JSONDecodeError  # Added JSONDecodeError
from typing import List, Optional, Any, Dict
import logging
import boto3
import time  # Added for sleep
import random  # Added for jitter
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError, BotoCoreError

logger = logging.getLogger(__name__)  # Standard logger

_bedrock_client = None


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
) -> Optional[str]:
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
        The generated text from the model or None if an error occurs.
    """
    client = get_bedrock_client(region)

    max_retries = 5
    base_delay = 1  # seconds
    max_delay = 60  # seconds

    for attempt in range(max_retries):
        response_obj = None  # Initialize response_obj at the start of each retry attempt
        try:
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

                response_obj = client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body),
                    accept=accept,
                    contentType=content_type
                )
                response_body_raw_bytes = response_obj.get("body").read()
                if not response_body_raw_bytes:
                    raise ValueError(
                        f"Received empty response body from model {model_id}.")
                response_body = json.loads(
                    response_body_raw_bytes.decode('utf-8'))
                generated_text = response_body.get("content") or response_body.get("completion")
                if generated_text is None:
                    raise ValueError(
                        f"Failed to parse or extract generated text from model {model_id}. Response body: {response_body}")
                return generated_text

            elif model_id.startswith("deepseek"):
                request_body = {
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

                response_obj = client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body),
                    accept="application/json",
                    contentType="application/json"
                )
                response_body_raw_bytes = response_obj.get("body").read()
                if not response_body_raw_bytes:
                    raise ValueError(
                        f"Received empty response body from model {model_id}.")
                response_body = json.loads(
                    response_body_raw_bytes.decode('utf-8'))
                choices = response_body.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message = choices[0].get("message")
                    if message and message.get("role") == "assistant":
                        return message.get("content")
                raise ValueError(
                    f"Failed to parse or extract generated text from model {model_id}. Response body: {response_body}")

            else:
                raise ValueError(f"Unsupported model_id: {model_id}. Only Claude and DeepSeek are supported.")

        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                if attempt < max_retries - 1:
                    delay = min(max_delay, base_delay * (2 ** attempt) + random.uniform(0, 1))
                    logger.warning(
                        f"ThrottlingException encountered for model {model_id}. Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Max retries reached for model {model_id} due to ThrottlingException. Last error: {e}")
                    return None
            elif e.response['Error']['Code'] == 'AccessDeniedException':
                logger.error(
                    f"AccessDeniedException when invoking Bedrock model {model_id}. Check IAM permissions. Error: {e}")
                return None
            else:
                logger.error(
                    f"ClientError when invoking Bedrock model {model_id}: {e}")
                if response_obj and hasattr(response_obj.get("body"), "read"):
                    error_response_body = response_obj.get("body").read().decode('utf-8', errors='ignore')
                    logger.error(f"Error response body from Bedrock: {error_response_body}")
                return None
        except BotoCoreError as e:
            logger.error(
                f"BotoCoreError when invoking Bedrock model {model_id}: {e}")
            if attempt < max_retries - 1:
                delay = min(max_delay, base_delay * (2 ** attempt) + random.uniform(0, 1))
                logger.warning(
                    f"BotoCoreError encountered. Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                logger.error(
                    f"Max retries reached for model {model_id} due to BotoCoreError. Last error: {e}")
                return None
        except NoCredentialsError:
            logger.error(f"No AWS credentials found. Please configure them.")
            return None
        except PartialCredentialsError:
            logger.error(f"Incomplete AWS credentials. Please check your configuration.")
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error invoking Bedrock model {model_id}: {e}")
            if response_obj and hasattr(response_obj.get("body"), "read"):
                error_response_body = response_obj.get("body").read().decode('utf-8', errors='ignore')
                logger.error(f"Error response body (if any) during unexpected error: {error_response_body}")
            return None

    return None
