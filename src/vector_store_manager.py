"""Vector store management for the RAG Agent."""

from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Document
from pinecone import Pinecone, ServerlessSpec


class VectorStoreManager:
    """Manages vector store operations for both local and Pinecone storage."""
    
    def __init__(
        self, 
        use_pinecone: bool = False, 
        pinecone_config: Optional[dict] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.use_pinecone = use_pinecone
        self.pinecone_config = pinecone_config or {}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        self._pinecone_client = None
    
    def create_index(self, documents: List[Document]) -> VectorStoreIndex:
        """
        Create a vector store index from documents.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            VectorStoreIndex instance
        """
        if not documents:
            raise ValueError("No documents provided to create index")
        
        print(f"Creating {'Pinecone' if self.use_pinecone else 'local'} vector index...")
        
        if self.use_pinecone:
            # Create Pinecone-backed index
            vector_store = self._get_pinecone_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context
            )
            print(f"Created Pinecone index with {len(documents)} documents")
        else:
            # Create local vector index
            self.index = VectorStoreIndex.from_documents(documents)
            print(f"Created local vector index with {len(documents)} documents")
        
        return self.index
    
    def connect_to_existing_index(self) -> VectorStoreIndex:
        """
        Connect to an existing Pinecone index without uploading new documents.
        Only works with Pinecone storage.
        
        Returns:
            VectorStoreIndex instance connected to existing index
        """
        if not self.use_pinecone:
            raise ValueError("connect_to_existing_index only works with Pinecone storage. Set use_pinecone=True")
        
        print("Connecting to existing Pinecone index...")
        
        # Create Pinecone-backed index from existing data
        vector_store = self._get_pinecone_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index from existing vector store (no documents needed)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        
        print("Successfully connected to existing Pinecone index")
        return self.index
    
    def _get_pinecone_vector_store(self) -> PineconeVectorStore:
        """Create or get existing Pinecone vector store."""
        # Validate configuration
        api_key = self.pinecone_config.get('api_key')
        if not api_key:
            raise ValueError("Pinecone API key is required for Pinecone mode")
        
        index_name = self.pinecone_config.get('index_name', 'chartwell-insurance')
        namespace = self.pinecone_config.get('namespace', 'llama-namespace')
        cloud = self.pinecone_config.get('cloud', 'aws')
        region = self.pinecone_config.get('region', 'us-east-1')
        
        # Initialize Pinecone client if not already done
        if self._pinecone_client is None:
            self._pinecone_client = Pinecone(api_key=api_key)
        
        # Create index if it doesn't exist
        self._ensure_index_exists(index_name, cloud, region)
        
        # Get the index and create vector store
        pinecone_index = self._pinecone_client.Index(index_name)
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            namespace=namespace
        )
        
        print(f"Using Pinecone index '{index_name}' with namespace '{namespace}'")
        return vector_store
    
    def _ensure_index_exists(self, index_name: str, cloud: str, region: str):
        """Ensure the Pinecone index exists, create if it doesn't."""
        existing_indexes = [index.name for index in self._pinecone_client.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {index_name}")
            self._pinecone_client.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            print(f"Successfully created Pinecone index: {index_name}")
        else:
            print(f"Using existing Pinecone index: {index_name}")
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """Get the current vector index."""
        return self.index
    
    def reset(self):
        """Reset the vector store manager."""
        self.index = None
        self._pinecone_client = None
        print("Vector store manager reset")
    
    def get_stats(self) -> dict:
        """Get statistics about the current vector store."""
        return {
            'has_index': self.index is not None,
            'storage_type': 'pinecone' if self.use_pinecone else 'local',
            'pinecone_configured': bool(self.pinecone_config.get('api_key'))
        }
    
    def __repr__(self):
        storage_type = 'Pinecone' if self.use_pinecone else 'Local'
        return f"VectorStoreManager(storage_type={storage_type}, has_index={self.index is not None})"
