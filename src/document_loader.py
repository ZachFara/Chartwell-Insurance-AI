"""Document loading utilities for the RAG Agent."""

from pathlib import Path
from typing import List
import pandas as pd
from llama_index.readers.file import PDFReader
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser


class DocumentLoader:
    """Handles loading documents from various file types."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.pdf_reader = PDFReader()
        self.supported_extensions = {'.pdf', '.csv'}
        
        # Store chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize node parser with custom settings
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n"  # Split on paragraphs for insurance docs
        )
    
    def load_from_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of Document objects
        """
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        # Process all files in directory
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    file_docs = self._load_single_file(file_path)
                    # Apply chunking to the loaded documents
                    chunked_docs = self._apply_chunking(file_docs)
                    documents.extend(chunked_docs)
                    print(f"Loaded and chunked {len(file_docs)} documents from {file_path.name} into {len(chunked_docs)} chunks")
                except Exception as e:
                    print(f"Warning: Failed to load {file_path.name}: {e}")
        
        return documents
    
    def _apply_chunking(self, documents: List[Document]) -> List[Document]:
        """Apply chunking to documents using the configured node parser."""
        chunked_docs = []
        for doc in documents:
            # Convert document to nodes and back to documents
            nodes = self.node_parser.get_nodes_from_documents([doc])
            for node in nodes:
                # Create new document from node
                chunked_doc = Document(
                    text=node.text,
                    metadata={**doc.metadata, 'chunk_size': self.chunk_size, 'chunk_overlap': self.chunk_overlap}
                )
                chunked_docs.append(chunked_doc)
        return chunked_docs
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load documents from a single file."""
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._load_pdf(file_path)
        elif extension == '.csv':
            return self._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """Load documents from a PDF file."""
        return self.pdf_reader.load_data(file_path)
    
    def _load_csv(self, file_path: Path) -> List[Document]:
        """
        Load documents from a 2-column CSV file.
        First column is treated as key, second as value.
        """
        documents = []
        
        try:
            df = pd.read_csv(file_path)
            
            if df.shape[1] < 2:
                print(f"Warning: CSV {file_path.name} has less than 2 columns, skipping")
                return documents
            
            # Process each row as key-value pair
            for idx, row in df.iterrows():
                key = str(row.iloc[0]) if pd.notna(row.iloc[0]) else f"row_{idx}_key"
                value = str(row.iloc[1]) if pd.notna(row.iloc[1]) else f"row_{idx}_value"
                
                doc = Document(
                    text=f"{key}: {value}",
                    metadata={
                        "key": key,
                        "value": value,
                        "source": str(file_path),
                        "row_index": idx
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            raise ValueError(f"Error processing CSV file {file_path}: {e}")
        
        return documents
    
    def get_supported_extensions(self) -> set:
        """Get set of supported file extensions."""
        return self.supported_extensions.copy()
    
    def __repr__(self):
        return f"DocumentLoader(supported_extensions={self.supported_extensions})"
