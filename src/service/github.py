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
from typing import List, Dict, Any, Tuple, Optional, Callable, Awaitable
import zipfile
import io
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)

# GitHub API Constants
GITHUB_API_BASE = "https://api.github.com"
DEFAULT_API_HEADERS = {
    "User-Agent": "GitHubCrawler/1.0",
    "Accept": "application/vnd.github.v3+json",
}


class CrawlException(Exception):
    """Custom exception for crawl-related errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"CrawlException: {self.message}"


def validate_github_url(repo_url: str) -> bool:
    """
    Validates that a URL has the correct GitHub repository format.

    Args:
        repo_url: The GitHub repository URL to validate

    Returns:
        bool: True if the URL is a valid GitHub repository URL, False otherwise
    """
    if not repo_url.startswith("https://github.com/"):
        return False

    parsed_url = urlparse(repo_url)
    path_segments = parsed_url.path.strip("/").split("/")

    if len(path_segments) < 2:
        return False

    return True


def extract_repo_info(url: str) -> Tuple[str, str, Optional[str], str]:
    """
    Extract repository information (username, repo_name, branch, path) from a GitHub URL.
    """
    parsed_url = urlparse(url)
    if parsed_url.hostname != "github.com":
        raise ValueError(f"Not a github.com URL: {url}")

    path_parts = parsed_url.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub repository URL (requires user/repo): {url}")

    username, repo_name = path_parts[0], path_parts[1]
    branch = None
    api_path = ""

    if len(path_parts) > 2:
        if path_parts[2] in ("tree", "blob") and len(path_parts) > 3:
            branch = path_parts[3]
            if len(path_parts) > 4:
                api_path = "/".join(path_parts[4:])
        else:
            api_path = "/".join(path_parts[2:])

    return username, repo_name, branch, api_path


def download_github_repo_with_token(
    owner: str, repo: str, branch: str, github_token: str
) -> bytes:
    """
    Download a GitHub repo ZIP archive using the API endpoint with authentication.
    """
    api_zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{branch}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GitHubCrawler/1.0",
    }
    response = requests.get(api_zip_url, headers=headers, timeout=600)
    response.raise_for_status()
    return response.content


def download_github_repo_public(owner: str, repo: str, branch: str) -> bytes:
    """
    Download a public GitHub repo ZIP archive using the public URL (no authentication).
    """
    public_zip_url = (
        f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
    )
    headers = {"User-Agent": "GitHubCrawler/1.0"}
    response = requests.get(public_zip_url, headers=headers, timeout=600)
    response.raise_for_status()
    return response.content


def get_branch_to_download(
    owner: str,
    repo: str,
    branch_name: Optional[str] = None,
    github_token: Optional[str] = None,
) -> str:
    """
    Determines the branch to download from a GitHub repository.

    Args:
        owner: Repository owner/username
        repo: Repository name
        branch_name: Optional explicitly specified branch name
        headers: Optional HTTP headers for API requests (for authentication)

    Returns:
        The branch name to use for download
    """

    headers = DEFAULT_API_HEADERS.copy() or {}

    if github_token:
        headers["Authorization"] = f"token {github_token}"

    if branch_name:
        logging.info(f"Using specified branch for {owner}/{repo}: {branch_name}")
        return branch_name

    # Need to query the API for the default branch
    api_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
    logging.info(
        f"Fetching repository metadata from {api_url} to find default branch..."
    )
    try:
        response = requests.get(api_url, headers=headers, timeout=15)
        response.raise_for_status()
        repo_info = response.json()
        default_branch = repo_info.get("default_branch")
        if not default_branch:
            logging.error(
                f"Could not determine default branch for {owner}/{repo}. API response: {repo_info}"
            )
            raise ValueError(f"Default branch not found for {owner}/{repo}.")
        logging.info(f"Default branch for {owner}/{repo} is: {default_branch}")
        return default_branch
    except requests.RequestException as e:
        logging.error(f"Error fetching repo metadata for {owner}/{repo}: {e}")
        raise e from e


def download_github_repo(
    repo_url: str,
    extract_to: str = ".",
    github_token: Optional[str] = None,
    branch_name: Optional[str] = None,
) -> str:
    """
    Downloads and extracts a GitHub repository.

    Args:
        repo_url: GitHub repository URL
        extract_to: Directory to extract the repository to
        github_token: Optional GitHub authentication token (for private repos)
        branch_name: Branch to download (defaults to repository's default branch)

    Returns:
        Path to the extracted repository
    """
    from src.utils.files import download_and_extract_zip

    # Validate URL
    if not validate_github_url(repo_url):
        raise ValueError("Invalid GitHub repository URL.")

    # Extract repository info
    owner, repo, _, _ = extract_repo_info(repo_url)

    # Determine branch to download
    branch_to_download = get_branch_to_download(owner, repo, branch_name, github_token)

    # Get ZIP content based on authentication
    try:
        if github_token:
            logging.info(
                f"Using GitHub API with token to download {owner}/{repo}@{branch_to_download}"
            )
            zip_content = download_github_repo_with_token(
                owner, repo, branch_to_download, github_token
            )
        else:
            logging.info(
                f"Using public GitHub URL to download {owner}/{repo}@{branch_to_download}"
            )
            zip_content = download_github_repo_public(owner, repo, branch_to_download)

        # Extract repository
        extracted_path = download_and_extract_zip(
            zip_content=zip_content, extract_to=extract_to, cleanup_temp=True
        )

        return extracted_path

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            auth_message = "" if github_token else " (no authentication token provided)"
            raise CrawlException(
                f"Repository not found or is private{auth_message}. Set GITHUB_TOKEN environment variable if this is a private repository."
            ) from e

        elif e.response.status_code == 401 or e.response.status_code == 403:
            raise CrawlException(
                "Authentication failed or insufficient permissions. Check your GITHUB_TOKEN."
            ) from e
        else:
            raise CrawlException(
                f"HTTP error when accessing GitHub repository: {str(e)}"
            ) from e

    except Exception as e:
        logging.error(
            f"Error downloading or extracting repository {owner}/{repo}@{branch_to_download}: {e}"
        )
        raise CrawlException(
            f"Error downloading or extracting repository {owner}/{repo}@{branch_to_download}: {e}"
        ) from e
