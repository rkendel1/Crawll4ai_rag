"""
PDF Processing Module for Crawl4AI RAG MCP Server
Handles extraction and chunking of PDF documents
"""

import pdfplumber
from typing import List, Dict, Optional
import hashlib
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Processes PDF files for ingestion into vector database"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Number of words per chunk
            overlap: Number of overlapping words between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    async def process_pdf(self, file_path: str) -> List[Dict]:
        """
        Extract and chunk PDF content
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of chunks with content and metadata
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: For PDF parsing errors
        """
        chunks = []
        
        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # Extract metadata
                metadata = {
                    "source": str(path.absolute()),
                    "filename": path.name,
                    "type": "pdf",
                    "pages": len(pdf.pages),
                    "extracted_at": datetime.now().isoformat(),
                    "file_size": path.stat().st_size
                }
                
                # Log extraction start
                logger.info(f"Extracting PDF: {path.name} ({len(pdf.pages)} pages)")
                
                # Extract text from all pages
                full_text = ""
                page_boundaries = []  # Track page boundaries for metadata
                
                for i, page in enumerate(pdf.pages):
                    page_start = len(full_text)
                    page_text = page.extract_text()
                    
                    if page_text:
                        # Add page separator
                        if full_text:
                            full_text += "\n\n"
                        full_text += f"--- Page {i+1} ---\n\n{page_text}"
                        page_boundaries.append((page_start, len(full_text), i+1))
                    else:
                        logger.warning(f"No text extracted from page {i+1}")
                
                # Chunk the content
                chunks = self._chunk_text(full_text, metadata, page_boundaries)
                
                logger.info(f"Created {len(chunks)} chunks from PDF")
                
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise Exception(f"Failed to process PDF: {str(e)}")
        
        return chunks
    
    def _chunk_text(self, text: str, metadata: Dict, 
                    page_boundaries: List[tuple]) -> List[Dict]:
        """
        Split text into overlapping chunks with page awareness
        
        Args:
            text: Full text content
            metadata: Document metadata
            page_boundaries: List of (start, end, page_num) tuples
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        words = text.split()
        
        if not words:
            return chunks
        
        # Create chunks with overlap
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Determine which pages this chunk spans
            chunk_start_char = len(" ".join(words[:i]))
            chunk_end_char = chunk_start_char + len(chunk_text)
            
            pages_in_chunk = set()
            for start, end, page_num in page_boundaries:
                if (chunk_start_char <= end and chunk_end_char >= start):
                    pages_in_chunk.add(page_num)
            
            # Generate unique chunk ID
            chunk_id = hashlib.md5(
                f"{metadata['source']}_{i}_{chunk_text[:50]}".encode()
            ).hexdigest()
            
            chunk_metadata = {
                **metadata,
                "chunk_index": len(chunks),
                "chunk_id": chunk_id,
                "pages_in_chunk": sorted(list(pages_in_chunk)),
                "word_count": len(chunk_words)
            }
            
            chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks
    
    async def extract_with_fallback(self, file_path: str) -> List[Dict]:
        """
        Try multiple PDF extraction methods as fallback
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of chunks
        """
        # First try pdfplumber
        try:
            return await self.process_pdf(file_path)
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying pypdf2")
        
        # Fallback to pypdf2
        try:
            from PyPDF2 import PdfReader
            
            chunks = []
            reader = PdfReader(file_path)
            full_text = ""
            
            metadata = {
                "source": file_path,
                "type": "pdf",
                "pages": len(reader.pages),
                "extracted_at": datetime.now().isoformat(),
                "extraction_method": "pypdf2"
            }
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text += f"\n\n--- Page {i+1} ---\n\n{text}"
            
            chunks = self._chunk_text(full_text, metadata, [])
            return chunks
            
        except Exception as e:
            logger.error(f"All PDF extraction methods failed: {e}")
            raise Exception(f"Unable to extract PDF content: {str(e)}")