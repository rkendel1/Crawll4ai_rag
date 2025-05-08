
import re
import logging

logger = logging.getLogger(__name__)

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