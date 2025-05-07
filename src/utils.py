"""
Utility functions for the Crawl4AI MCP server.
"""
import os
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse
import openai
from pathlib import Path
import re
import logging
from datetime import datetime
import requests
from xml.etree import ElementTree
# Load OpenAI API key for embeddings

openai.api_key = os.environ["OPENAI_API_KEY"]


logger = logging.getLogger(__name__)


def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.
    
    Args:
        sitemap_url: URL of the sitemap
        
    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.
    
    Args:
        chunk: Markdown chunk
        
    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }


def _generate_filename_stem_from_url(url: str, max_len: int = 200) -> str:
    """
    Converts a URL to a safe filename stem (without extension).
    The sanitization aims to be simple and create a readable string.
    """
    # Start with the part after "://" if present
    if "://" in url:
        stem = url.split("://", 1)[1]
    else:
        stem = url

    # Replace common URL characters that are problematic in filenames
    stem = stem.replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_").replace("%", "_")
    
    # Replace any remaining non-alphanumeric characters (excluding dot, hyphen, underscore) with an underscore
    stem = re.sub(r'[^\w.\-_]', '_', stem)
    
    # Consolidate multiple underscores and remove leading/trailing underscores
    stem = re.sub(r'_+', '_', stem).strip('_')

    # If the stem is empty after sanitization, use a default
    if not stem:
        stem = "untitled_page"
        logger.warning(f"URL '{url}' resulted in an empty filename stem, using '{stem}'.")

    # Truncate if the stem is too long
    if len(stem) > max_len:
        stem = stem[:max_len]
        logger.debug(f"Filename stem for URL '{url}' was truncated to '{stem}'.")
        # Ensure it doesn't end with an underscore after truncation
        stem = stem.strip('_')
        if not stem: # Recapture if stripping made it empty
             stem = "untitled_page_truncated"


    return stem


async def save_html_to_export(url: str, html_content: str, output_folder: str = "EXPORT") -> str:
    """
    Save HTML content to the specified output_folder with a filename generated from the URL.
    Prepends an HTML comment with the original URL and export date.
    Handles filename collisions by appending a counter (e.g., filename_1.html, filename_2.html).
    """
    export_dir = Path(output_folder)
    export_dir.mkdir(parents=True, exist_ok=True)

    base_stem = _generate_filename_stem_from_url(url)
    extension = ".html"
    
    # Initial filename
    current_filename = f"{base_stem}{extension}"
    file_path = export_dir / current_filename
    
    # Handle potential filename collisions
    counter = 1
    while file_path.exists():
        current_filename = f"{base_stem}_{counter}{extension}"
        file_path = export_dir / current_filename
        counter += 1
        if counter > 1000:  # Safety break to prevent infinite loops or too many files
            error_msg = (
                f"Could not find a unique filename for URL '{url}' in '{export_dir}' "
                f"after 1000 attempts. Last tried: '{current_filename}'"
            )
            logger.error(error_msg)
            raise OSError(error_msg)

    # Prepare the frontmatter-like comment
    export_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    frontmatter_comment = f"<!--\nOriginal URL: {url}\nExport Date: {export_timestamp}\n-->\n"
    
    content_to_write = frontmatter_comment + html_content

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content_to_write)
        logger.debug(f"Saved HTML content from URL '{url}' to '{file_path}' with metadata comment.")
    except IOError as e:
        logger.error(f"Failed to write HTML content from URL '{url}' to '{file_path}': {e}")
        raise  # Re-raise the exception after logging

    return str(file_path)



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



def _sanitize_filename_component(name: str) -> str:
    """
    Sanitizes a single path component (directory name or filename part).
    Removes characters that are generally problematic in file systems and ensures
    it's not an empty or problematic component like "." or "..".
    """
    if not name:
        return "_empty_component_"
    
    # Replace problematic characters with underscore
    # Characters to avoid: < > : " / \ | ? * and control characters (0-31)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', name)
    
    # Remove leading/trailing spaces and dots for Windows compatibility
    name = name.strip('. ')
    
    # Ensure component is not empty or just dots after sanitization
    if not name or name == "." or name == "..":
        return "_sanitized_component_"
        
    return name


def _generate_filename_stem_from_url(url: str, max_len: int = 200) -> str:
    """
    Converts a URL to a safe filename stem (without extension).
    The sanitization aims to be simple and create a readable string.
    """
    # Start with the part after "://" if present
    if "://" in url:
        stem = url.split("://", 1)[1]
    else:
        stem = url

    # Replace common URL characters that are problematic in filenames
    stem = stem.replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_").replace("%", "_")
    
    # Replace any remaining non-alphanumeric characters (excluding dot, hyphen, underscore) with an underscore
    stem = re.sub(r'[^\w.\-_]', '_', stem)
    
    # Consolidate multiple underscores and remove leading/trailing underscores
    stem = re.sub(r'_+', '_', stem).strip('_')

    # If the stem is empty after sanitization, use a default
    if not stem:
        stem = "untitled_page"
        logger.warning(f"URL '{url}' resulted in an empty filename stem, using '{stem}'.")

    # Truncate if the stem is too long
    if len(stem) > max_len:
        stem = stem[:max_len]
        logger.debug(f"Filename stem for URL '{url}' was truncated to '{stem}'.")
        # Ensure it doesn't end with an underscore after truncation
        stem = stem.strip('_')
        if not stem: # Recapture if stripping made it empty
             stem = "untitled_page_truncated"


    return stem


async def save_raw_content_to_export(
    source_url_for_metadata: str,
    content: Union[str, bytes],
    output_folder: str = "EXPORT",
    target_relative_path_hint: Optional[str] = None
) -> str:
    """
    Save raw content (string or bytes) to the specified output_folder.
    A filename can be suggested via target_relative_path_hint (can include subdirs).
    If no hint, filename is generated from the source_url_for_metadata.
    Prepends an HTML comment with metadata if content is HTML.
    Handles filename collisions by appending a counter.
    """
    base_export_dir = Path(output_folder)
    # Parent directory of the actual file will be created just before writing

    initial_relative_path: Path
    effective_extension: str

    if target_relative_path_hint:
        # Sanitize each component of the provided hint
        # Path("foo/bar.txt").parts -> ("foo", "bar.txt")
        # Path("baz.md").parts -> ("baz.md",)
        original_parts = Path(target_relative_path_hint).parts
        sanitized_components = [_sanitize_filename_component(part) for part in original_parts]
        
        # Filter out any components that became empty or problematic after sanitization
        # (though _sanitize_filename_component tries to prevent this)
        valid_sanitized_components = [comp for comp in sanitized_components if comp and comp not in ["_empty_component_", "_sanitized_component_"]]

        if not valid_sanitized_components:
            logger.warning(
                f"Target relative path hint '{target_relative_path_hint}' sanitized to empty or invalid components. "
                f"Falling back to URL-based naming for {source_url_for_metadata}."
            )
            target_relative_path_hint = None # Force fallback to URL-based naming

    # This 'if' condition is re-evaluated after the sanitization check above
    if target_relative_path_hint:
        # Re-sanitize and construct path (might be redundant if sanitization above is perfect, but safe)
        sanitized_components = [_sanitize_filename_component(part) for part in Path(target_relative_path_hint).parts]
        initial_relative_path = Path(*[comp for comp in sanitized_components if comp]) # Ensure no empty parts join

        _ext_from_target = initial_relative_path.suffix.lower()
        # Check if suffix is a "valid" extension (e.g. not just "." from "filename.")
        if _ext_from_target and len(_ext_from_target) > 1:
            effective_extension = _ext_from_target
        elif isinstance(content, str):
            effective_extension = ".txt"
        else:
            effective_extension = ".bin"
        
        # If initial_relative_path had no suffix, or an invalid one, append the determined one.
        # Ensure we don't add extension to a directory-like path hint (e.g. "folder/")
        if (not initial_relative_path.suffix or len(initial_relative_path.suffix) <=1 ) and initial_relative_path.name:
             initial_relative_path = initial_relative_path.with_suffix(effective_extension)
        elif not initial_relative_path.name: # Hint was likely a directory path like "folder/"
            # In this case, we need a filename. Fallback to URL-based name inside this dir.
            generated_stem_for_dir = _generate_filename_stem_from_url(source_url_for_metadata)
            if isinstance(content, str): # Determine extension again for this generated name
                stripped_content = content.strip().lower()
                effective_extension = ".html" if stripped_content.startswith("<!doctype html") or stripped_content.startswith("<html") else ".txt"
            else:
                effective_extension = ".bin"
            initial_relative_path = initial_relative_path / f"{generated_stem_for_dir}{effective_extension}"

    else: # No target_relative_path_hint or it was invalidated
        generated_stem = _generate_filename_stem_from_url(source_url_for_metadata)
        if isinstance(content, str):
            stripped_content = content.strip().lower()
            if stripped_content.startswith("<!doctype html") or stripped_content.startswith("<html"):
                effective_extension = ".html"
            else:
                effective_extension = ".txt"
        else:
            effective_extension = ".bin"
        initial_relative_path = Path(f"{generated_stem}{effective_extension}")

    # Collision handling
    current_path_candidate = base_export_dir / initial_relative_path
    
    # For collision logic, separate the stem and the true parent directory of the intended file
    original_stem = current_path_candidate.stem
    # Parent dir relative to base_export_dir. If current_path_candidate is at base_export_dir root, this is Path('.')
    path_parent_dir = current_path_candidate.parent 
    
    counter = 1
    final_path_to_save = current_path_candidate
    
    while final_path_to_save.exists():
        new_filename_for_collision = f"{original_stem}_{counter}{effective_extension}"
        final_path_to_save = path_parent_dir / new_filename_for_collision
        counter += 1
        if counter > 1000:
            error_msg = (
                f"Could not find a unique filename for source '{source_url_for_metadata}' "
                f"(target hint: {target_relative_path_hint or 'from_url'}) in '{path_parent_dir}' "
                f"after 1000 attempts. Last tried: '{final_path_to_save.name}'"
            )
            logger.error(error_msg)
            raise OSError(error_msg)

    # Ensure parent directory for the final file path exists
    final_path_to_save.parent.mkdir(parents=True, exist_ok=True)

    content_to_write = content
    if isinstance(content, str) and effective_extension.lower() in ['.html', '.htm']:
        export_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        frontmatter_comment = f"<!--\nOriginal URL: {source_url_for_metadata}\nExport Date: {export_timestamp}\n-->\n"
        content_to_write = frontmatter_comment + content

    try:
        if isinstance(content_to_write, str):
            with open(final_path_to_save, "w", encoding="utf-8") as f:
                f.write(content_to_write)
        else: # content is bytes
            with open(final_path_to_save, "wb") as f:
                f.write(content_to_write) # type: ignore [arg-type] # content_to_write is bytes here
        logger.debug(f"Saved content from source '{source_url_for_metadata}' to '{final_path_to_save}'.")
    except IOError as e:
        logger.error(f"Failed to write content from source '{source_url_for_metadata}' to '{final_path_to_save}': {e}")
        raise

    return str(final_path_to_save)
