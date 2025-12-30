"""
Document processing utilities for loading and chunking documents.
"""

import os
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles loading and chunking of documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_documents(self, documents_dir: str = "documents") -> List[Document]:
        """
        Load all documents from the specified directory.
        
        Args:
            documents_dir: Path to the documents directory
            
        Returns:
            List of Document objects
        """
        documents_path = Path(documents_dir)
        if not documents_path.exists():
            return []
        
        all_documents = []
        
        # Supported file extensions
        supported_extensions = {".txt", ".md", ".pdf"}
        
        # Load all supported files
        for file_path in documents_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    if file_path.suffix.lower() == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                    else:
                        loader = TextLoader(str(file_path), encoding="utf-8")
                    
                    docs = loader.load()
                    all_documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        return all_documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def process_documents(self, documents_dir: str = "documents") -> List[Document]:
        """
        Load and chunk documents from the specified directory.
        
        Args:
            documents_dir: Path to the documents directory
            
        Returns:
            List of chunked Document objects
        """
        documents = self.load_documents(documents_dir)
        if not documents:
            return []
        
        chunks = self.chunk_documents(documents)
        return chunks

