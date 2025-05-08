from pathlib import Path
from .convert import _generate_filename_stem_from_url, _sanitize_filename_component
from typing import Union, Optional
import logging
import datetime as dt

logger = logging.getLogger(__name__)


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
    export_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
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
        export_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
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
