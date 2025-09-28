# Import a bunch of llama-index stuff
from llama_index.core import Settings
from llama_index.core.tools import RetrieverTool
from llama_index.agent.openai import OpenAIAgent

# Import our clean components
from .configuration import Configuration
from .document_loader import DocumentLoader
from .vector_store_manager import VectorStoreManager


class Agent:
    def __init__(
        self, 
        name: str, 
        use_pinecone: bool = False,
        # Tunable parameters
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_top_k: int = 5,
        system_prompt_override: str = None
    ):
        self.name = name
        
        # Store tuning parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.system_prompt_override = system_prompt_override
        
        # Initialize components with parameters
        self.config = Configuration()
        self.document_loader = DocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store_manager = VectorStoreManager(
            use_pinecone=use_pinecone,
            pinecone_config=self.config.get_pinecone_config() if use_pinecone else None,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.agent = None
        
        print(f"Initialized {self.name} with {self.config}")
        print(f"Tuning parameters: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, top_k={similarity_top_k}")
        if use_pinecone and not self.config.has_pinecone_config():
            print("Warning: Pinecone mode requested but no Pinecone configuration found in .env")

    def ingest_directory(self, directory_path: str):
        """
        Ingest all files in a directory and create a vector store index.
        Uses the DocumentLoader to load documents and VectorStoreManager to create index.
        """
        # Load documents using DocumentLoader
        documents = self.document_loader.load_from_directory(directory_path)
        
        if documents:
            # Create index using VectorStoreManager
            index = self.vector_store_manager.create_index(documents)
            
            # Create query tool
            query_tool = self._create_query_tool(index)
            
            # Create agent
            self._create_agent([query_tool])
            
            print(f"Successfully created agent with {len(documents)} documents")
        else:
            print("No documents found to index")

    def connect_to_existing_index(self):
        """
        Connect to an existing Pinecone index without uploading new documents.
        This allows you to use a pre-populated index without re-ingesting documents.
        Only works when use_pinecone=True.
        """
        if not self.vector_store_manager.use_pinecone:
            return "Error: connect_to_existing_index only works with Pinecone storage. Initialize agent with use_pinecone=True"
        
        try:
            # Connect to existing index using VectorStoreManager
            index = self.vector_store_manager.connect_to_existing_index()
            
            # Create query tool
            query_tool = self._create_query_tool(index)
            
            # Create agent
            self._create_agent([query_tool])
            
            print("Successfully connected to existing Pinecone index")
            return "Agent connected to existing index and ready to chat!"
            
        except Exception as e:
            error_msg = f"Failed to connect to existing index: {str(e)}"
            print(error_msg)
            return error_msg

    def chat(self, message: str, get_response = False) -> str:
        """Chat with the agent. The agent will decide whether to use the document search tool."""
        if self.agent is None:
            return "Please ingest documents first using ingest_directory()"
        
        if get_response:
            response = self.agent.chat(message)
            return response
        else:
            response = self.agent.chat(message)
            return str(response)

    def reset(self):
        """Reset the agent and clear all indexed documents."""
        self.vector_store_manager.reset()
        self.agent = None
        print("Agent reset. Call ingest_directory() to load new documents.")

    def get_index_stats(self) -> dict:
        """Get information about the current setup."""
        stats = self.vector_store_manager.get_stats()
        stats.update({
            'agent_name': self.name,
            'has_agent': self.agent is not None,
            'config_status': str(self.config)
        })
        return stats

    def _create_query_tool(self, index) -> RetrieverTool:
        """Create a retriever tool from the index that returns raw document chunks."""
        # Create retriever with configurable parameters
        retriever = index.as_retriever(
            similarity_top_k=self.similarity_top_k,  # Use instance parameter
            retriever_mode="default"
        )
        
        # Use RetrieverTool instead of QueryEngineTool to get raw chunks
        return RetrieverTool.from_defaults(
            retriever=retriever,
            name="insurance_documents",
            description=(
                f"Search through insurance documents and contracts to find relevant information. "
                f"This tool retrieves the top {self.similarity_top_k} most relevant document chunks "
                f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap}) about insurance policies, "
                f"coverage details, terms, conditions, and contract information. "
                f"Returns unfiltered document content without summarization."
            )
        )


    def _create_agent(self, tools):
        """Create an OpenAI agent with the provided tools."""
        # Use custom system prompt if provided, otherwise use default
        if self.system_prompt_override:
            system_prompt = self.system_prompt_override
        else:
            system_prompt = (
                f"You are {self.name}, an AI assistant for Chartwell Insurance designed to help our customer service team "
                "provide accurate and professional responses to customer queries and emails. "
                f"\n\nYou have access to a document search tool that returns the top {self.similarity_top_k} most relevant "
                f"document chunks (size: {self.chunk_size} chars, overlap: {self.chunk_overlap}) from our insurance files. "
                "When you receive search results, carefully analyze ALL the retrieved content to provide comprehensive answers. "
                "\n\nKey guidelines:"
                "\n- Always search for relevant information before answering insurance-related questions"
                "\n- Use all retrieved document chunks to form complete, accurate responses"
                "\n- Preserve specific details, numbers, dates, and exact terms from the documents"
                "\n- If multiple chunks contain related information, synthesize them together"
                "\n- Reference specific documents or sections when relevant"
                "\n- Maintain a professional tone representing Chartwell Insurance"
                "\n- Be thorough and don't lose important details from the source material"
                "\n\nDo not mention the search process explicitly - present information as your knowledge base. "
                "Focus on being helpful, accurate, and professional in all interactions."
            )
        
        self.agent = OpenAIAgent.from_tools(
            tools,
            llm=Settings.llm,
            verbose=True,
            system_prompt=system_prompt
        )

    def __repr__(self):
        return f"Agent(name='{self.name}', storage='{self.vector_store_manager}', has_agent={self.agent is not None})"

if __name__ == "__main__":
    # Example usage with clean component architecture
    
    print("=== Testing Local Vector Storage ===")
    agent_local = Agent("Insurance Assistant", use_pinecone=False)
    print(f"Local agent: {agent_local}")
    
    print("=== Testing Pinecone Vector Storage ===")
    agent_pinecone = Agent("Insurance Assistant", use_pinecone=True)
    print(f"Pinecone agent: {agent_pinecone}")
    
    # Example of ingesting documents (uncomment to test)
    # agent_local.ingest_directory("/Users/zfara/Repositories/Chartwell/Chartwell-Insurance-AI/data/raw/")
    # response = agent_local.chat("What documents do you have access to?")
    # print(f"Response: {response}")
    
    print(f"Local agent stats: {agent_local.get_index_stats()}")
    print(f"Pinecone agent stats: {agent_pinecone.get_index_stats()}")
    
    print("Clean component architecture initialized successfully!")

    # Example 1: Ingest new documents to Pinecone (uncomment to test)
    # agent_pinecone.ingest_directory("/Users/zfara/Repositories/Chartwell/Chartwell-Insurance-AI/data/raw/")
    # response = agent_pinecone.chat("Can you summarize the Chubb contract?")
    # print(f"Pinecone agent response: {response}")
    
    # Example 2: Connect to existing Pinecone index (uncomment to test)
    # result = agent_pinecone.connect_to_existing_index()
    # print(f"Connection result: {result}")
    # if "ready to chat" in result:
    #     response = agent_pinecone.chat("What insurance documents are available?")
    #     print(f"Response from existing index: {response}")
    
    pass