"""
GitHub repository crawler.

This module provides functions for crawling GitHub repositories, primarily by
downloading the repository as a ZIP archive and processing its contents.
"""
import os
import re
import time
import requests
import logging
import asyncio
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple, Optional,Callable, Awaitable
import zipfile
import io
import shutil 

# Set up logging
logging.basicConfig(level=logging.INFO)

# GitHub API Constants
GITHUB_API_BASE = "https://api.github.com"
DEFAULT_API_HEADERS = {
    'User-Agent': 'GitHubCrawler/1.0', 
    'Accept': 'application/vnd.github.v3+json'
}


def download_github_repo(
    repo_url: str, 
    extract_to: str = ".", 
    github_token: Optional[str] = None,
    branch_name: Optional[str] = None
) -> str:
    """
    Downloads and extracts a specific or default branch of a GitHub repository.
    """
    if not repo_url.startswith("https://github.com/"):
        raise ValueError("URL must be a GitHub repository URL.")

    parsed_url = urlparse(repo_url)
    path_segments = parsed_url.path.strip('/').split('/')
    
    if len(path_segments) < 2:
        raise ValueError("Invalid GitHub URL format. Expected owner/repo.")
    owner, repo = path_segments[0], path_segments[1]

    headers = DEFAULT_API_HEADERS.copy()
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    branch_to_download = branch_name
    if not branch_to_download:
        api_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
        logging.info(f"Fetching repository metadata from {api_url} to find default branch...")
        try:
            response = requests.get(api_url, headers=headers, timeout=15)
            response.raise_for_status()
            repo_info = response.json()
            branch_to_download = repo_info.get("default_branch")
            if not branch_to_download:
                logging.error(f"Could not determine default branch for {owner}/{repo}. API response: {repo_info}")
                raise ValueError(f"Default branch not found for {owner}/{repo}.")
            logging.info(f"Default branch for {owner}/{repo} is: {branch_to_download}")
        except requests.RequestException as e:
            logging.error(f"Error fetching repo metadata for {owner}/{repo}: {e}")
            raise
    else:
        logging.info(f"Using specified branch for {owner}/{repo}: {branch_to_download}")


    zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch_to_download}.zip"
    logging.info(f"Downloading repository ZIP from {zip_url} ...")
    
    try:
        zip_response = requests.get(zip_url, headers=headers, timeout=600) 
        zip_response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Error downloading ZIP for {owner}/{repo}@{branch_to_download}: {e}")
        raise
    
    os.makedirs(extract_to, exist_ok=True)
    
    extracted_folder_name = f"{repo}-{branch_to_download}"
    final_extracted_path = os.path.join(extract_to, extracted_folder_name)

    if os.path.exists(final_extracted_path):
        logging.info(f"Removing existing directory: {final_extracted_path}")
        shutil.rmtree(final_extracted_path)
    
    logging.info(f"Extracting ZIP content to a temporary location...")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zf:
            zip_root_dir_name = ""
            for member in zf.namelist():
                if '/' in member:
                    zip_root_dir_name = member.split('/')[0]
                    break
            if not zip_root_dir_name:
                raise zipfile.BadZipFile("ZIP file does not seem to have a root directory.")

            temp_extract_parent = os.path.join(extract_to, "_temp_extract")
            if os.path.exists(temp_extract_parent): 
                shutil.rmtree(temp_extract_parent)
            os.makedirs(temp_extract_parent, exist_ok=True)
            
            zf.extractall(temp_extract_parent)
            
            source_extracted_content_path = os.path.join(temp_extract_parent, zip_root_dir_name)
            if os.path.exists(source_extracted_content_path):
                shutil.move(source_extracted_content_path, final_extracted_path)
                logging.info(f"Repository content extracted to: {final_extracted_path}")
            else:
                raise FileNotFoundError(f"Expected extracted content not found at {source_extracted_content_path}")

            shutil.rmtree(temp_extract_parent)

    except zipfile.BadZipFile as e_zip:
        logging.error(f"Error extracting ZIP for {owner}/{repo}@{branch_to_download}: {e_zip}")
        raise
    except Exception as e_extract: 
        logging.error(f"An error occurred during extraction process: {e_extract}")
        if os.path.exists(final_extracted_path): 
            shutil.rmtree(final_extracted_path)
        raise
        
    return final_extracted_path

def _get_content_type_from_filename(filename: str) -> str:
    """Determines a best-guess content type from a filename."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.md', '.markdown'):
        return 'text/markdown'
    elif ext == '.py':
        return 'text/x-python'
    elif ext == '.js':
        return 'application/javascript'
    elif ext in ('.html', '.htm'):
        return 'text/html'
    elif ext == '.css':
        return 'text/css'
    elif ext == '.json':
        return 'application/json'
    elif ext == '.xml':
        return 'application/xml'
    elif ext == '.txt':
        return 'text/plain'
    # Add more mappings as needed
    return 'application/octet-stream' # Default for unknown binary or text


def is_github_repository(url: str) -> bool:
    """
    Check if a URL is a GitHub repository.
    """
    return bool(re.match(r'https?://github\.com/[^/]+/[^/]+(/.*)?$', url))


def extract_repo_info(url: str) -> Tuple[str, str, str, str]:
    """
    Extract repository information (username, repo_name, branch, path) from a GitHub URL.
    """
    logging.debug(f"Extracting repo info from URL: {url}")
    parsed_url = urlparse(url)
    if parsed_url.hostname != 'github.com':
        raise ValueError(f"Not a github.com URL: {url}")

    path_parts = parsed_url.path.strip('/').split('/')
    
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub repository URL (requires user/repo): {url}")
    
    username, repo_name = path_parts[0], path_parts[1]
    branch = '' 
    api_path = ''

    if len(path_parts) > 2:
        if path_parts[2] == 'tree' or path_parts[2] == 'blob':
            if len(path_parts) > 3:
                branch = path_parts[3]
                if len(path_parts) > 4:
                    api_path = '/'.join(path_parts[4:])
            else: 
                logging.warning(f"URL {url} has 'tree' or 'blob' but no branch specified, will use default.")
        else: 
            api_path = '/'.join(path_parts[2:])

    if not branch: 
        try:
            # This API call is only to determine the default branch if not specified in URL
            # and if download_github_repo isn't called with an explicit branch_name.
            # download_github_repo has its own logic for this.
            # This part of extract_repo_info might be redundant if download_github_repo always handles it.
            # For now, keeping it for cases where extract_repo_info might be called independently.
            repo_api_url = f"{GITHUB_API_BASE}/repos/{username}/{repo_name}"
            response = requests.get(repo_api_url, headers=DEFAULT_API_HEADERS, timeout=10)
            response.raise_for_status()
            repo_data = response.json()
            branch = repo_data.get('default_branch', 'main')
            logging.info(f"Determined default branch for {username}/{repo_name}: {branch}")
        except Exception as e:
            logging.warning(f"Could not determine default branch for {username}/{repo_name} via API in extract_repo_info: {e}. Defaulting to 'main'.")
            branch = 'main' # Fallback if API call fails
            
    logging.info(f"Extracted info: user='{username}', repo='{repo_name}', branch='{branch}', api_path='{api_path}'")
    return username, repo_name, branch, api_path


async def crawl_github_repository_async(
    repo_url: str,
    max_depth: int = 10, # Note: max_depth is not directly used by ZIP download method
    branch_name: Optional[str] = None,
    broadcast_progress: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    save_raw_content: bool = True, 
    output_dir_for_raw: str = "EXPORT_GITHUB",
    github_token: Optional[str] = None 
) -> List[Dict[str, Any]]:
    """
    Asynchronously "crawls" a GitHub repository by downloading it as a ZIP,
    extracting, and processing its files.
    """
    logging.info(f"Starting GitHub repository processing (via ZIP download) for URL: {repo_url}")
    all_fetched_results: List[Dict[str, Any]] = []
    
    try:
        # Use extract_repo_info to get owner and repo for path construction
        # It also helps in determining the branch if not explicitly provided,
        # though download_github_repo will also do this.
        owner_name, repo_name_from_url, actual_branch_hint, _ = extract_repo_info(repo_url)
        # If branch_name is provided to this function, it takes precedence.
        # Otherwise, use the hint from extract_repo_info (which might be default).
        effective_branch_name = branch_name if branch_name else actual_branch_hint

    except ValueError as e: # Handle cases where extract_repo_info fails
        logging.error(f"Could not parse repo_url '{repo_url}' with extract_repo_info: {e}. Using fallback naming.")
        parsed_url_for_names = urlparse(repo_url)
        path_segments_for_names = parsed_url_for_names.path.strip('/').split('/')
        owner_name = path_segments_for_names[0] if len(path_segments_for_names) > 0 else "unknown_owner"
        repo_name_from_url = path_segments_for_names[1] if len(path_segments_for_names) > 1 else "unknown_repo"
        effective_branch_name = branch_name # Use provided branch_name or None

    extraction_base_dir = os.path.join(output_dir_for_raw, owner_name, repo_name_from_url)
    os.makedirs(extraction_base_dir, exist_ok=True)

    extracted_repo_path: Optional[str] = None
    actual_branch_downloaded = ""

    try:
        if broadcast_progress:
            await broadcast_progress({"type": "crawl_status", "status": "downloading_repo", "url": repo_url, "timestamp": time.time()})

        extracted_repo_path = await asyncio.to_thread(
            download_github_repo,
            repo_url=repo_url,
            extract_to=extraction_base_dir, 
            github_token=github_token,
            branch_name=effective_branch_name # Pass the determined branch
        )
        if extracted_repo_path:
            # Infer actual branch downloaded from the path name (e.g. .../repo_name-main)
            actual_branch_downloaded = os.path.basename(extracted_repo_path).split(f"{repo_name_from_url}-", 1)[-1]


        if not extracted_repo_path or not os.path.isdir(extracted_repo_path):
            logging.error(f"Failed to download or extract repository: {repo_url}")
            if broadcast_progress:
                await broadcast_progress({"type": "crawl_error", "url": repo_url, "message": "Failed to download or extract repository.", "timestamp": time.time()})
            return []

        logging.info(f"Repository content extracted to: {extracted_repo_path}")
        if broadcast_progress:
            await broadcast_progress({"type": "crawl_status", "status": "processing_files", "url": repo_url, "path": extracted_repo_path, "timestamp": time.time()})

        file_count = 0
        for root, _, files in os.walk(extracted_repo_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                # Ensure relative_path uses forward slashes for URL construction
                relative_path = os.path.relpath(file_path, extracted_repo_path).replace(os.sep, '/')
                
                # Construct HTML URL using the branch name confirmed from the downloaded folder
                html_url = f"https://github.com/{owner_name}/{repo_name_from_url}/blob/{actual_branch_downloaded}/{relative_path}"
                download_url_approx = f"https://raw.githubusercontent.com/{owner_name}/{repo_name_from_url}/{actual_branch_downloaded}/{relative_path}"

                content: Any = None
                content_type = _get_content_type_from_filename(filename)
                file_success = False
                error_message = None

                try:
                    if content_type.startswith('text/') or \
                       any(filename.endswith(ext) for ext in ['.md', '.txt', '.py', '.js', '.json', '.yaml', '.toml', '.xml', '.css', '.html', '.htm', '.rst', '.sh', '.ps1', '.bat']):
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                    else: 
                        with open(file_path, 'rb') as f:
                            content = f.read()
                    file_success = True
                    file_count += 1
                except Exception as e_read:
                    logging.error(f"Error reading file {file_path}: {e_read}")
                    content = f"Error reading file: {e_read}"
                    error_message = str(e_read)
                
                file_data = {
                    'url': html_url,
                    'download_url': download_url_approx, 
                    'title': filename,
                    'api_path': relative_path, 
                    'content': content,
                    'content_type': content_type,
                    'is_file': True,
                    'success': file_success,
                    'error_message': error_message,
                    'local_path': file_path 
                }
                all_fetched_results.append(file_data)

                if broadcast_progress:
                    await broadcast_progress({
                        "type": "file_progress", "url": html_url,
                        "status": "success" if file_success else "error",
                        "size": len(content) if content and isinstance(content, (str, bytes)) else 0,
                        "message": f"Processed: {relative_path}", "timestamp": time.time()
                    })
        
        logging.info(f"Processed {file_count} files from the downloaded repository.")

    except Exception as e:
        logging.error(f"Error during GitHub repository processing for {repo_url}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        if broadcast_progress:
            await broadcast_progress({"type": "crawl_error", "url": repo_url, "message": str(e), "timestamp": time.time()})
        return [] 
    finally:
        if not save_raw_content and extracted_repo_path and os.path.isdir(extracted_repo_path):
            try:
                logging.info(f"Cleaning up temporary extracted directory: {extracted_repo_path}")
                shutil.rmtree(extracted_repo_path)
                # Clean up parent directories if they become empty
                current_dir_to_check = os.path.dirname(extracted_repo_path)
                while current_dir_to_check != output_dir_for_raw and \
                      os.path.exists(current_dir_to_check) and \
                      not os.listdir(current_dir_to_check):
                    os.rmdir(current_dir_to_check)
                    current_dir_to_check = os.path.dirname(current_dir_to_check)

            except Exception as e_cleanup:
                logging.error(f"Error cleaning up extracted directory {extracted_repo_path}: {e_cleanup}")

    successful_fetches = sum(1 for r in all_fetched_results if r['success'])
    logging.info(f"GitHub repository processing finished. Successfully processed {successful_fetches} of {len(all_fetched_results)} files found in ZIP.")
    if broadcast_progress:
        await broadcast_progress({
            "type": "crawl_status", "status": "completed", "url": repo_url,
            "total_files_discovered_in_zip": len(all_fetched_results),
            "successful_files_processed": successful_fetches,
            "timestamp": time.time()
        })
            
    return all_fetched_results