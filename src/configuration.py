"""Configuration management for the RAG Agent."""

import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from typing import Dict, Any, Optional

class Configuration:
    """Handles all configuration loading from environment variables or Streamlit secrets."""
    
    def __init__(self, load_method: str = "auto"):
        """
        Initialize configuration.
        
        Args:
            load_method: "auto", "env", or "streamlit"
        """
        self.config = {}
        self.load_method = load_method
        self._load_configuration()
        self._configure_llama_index()
    
    def _load_configuration(self):
        """Load configuration based on the specified method."""
        if self.load_method == "auto":
            self._auto_detect_and_load()
        elif self.load_method == "env":
            self._load_from_env()
        elif self.load_method == "streamlit":
            self._load_from_streamlit()
        else:
            raise ValueError("load_method must be 'auto', 'env', or 'streamlit'")
    
    def _auto_detect_and_load(self):
        """Auto-detect the environment and load configuration accordingly."""
        try:
            # Try to import streamlit and check if we're in a Streamlit environment
            import streamlit as st
            
            # Check if we're actually running in Streamlit context
            # This is safer than checking st.secrets directly
            try:
                # Try to access secrets without causing an error
                _ = dict(st.secrets)
                print("🎯 Auto-detected Streamlit environment - loading from secrets")
                self._load_from_streamlit()
                return
            except Exception:
                # Streamlit is installed but not in Streamlit context or no secrets
                pass
                
        except ImportError:
            # Streamlit not installed
            pass
        
        # Fall back to .env file
        print("🎯 Auto-detected local environment - loading from .env file")
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from .env file."""
        load_dotenv()
        
        # Core API keys
        self.config.update({
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'llama_cloud_api_key': os.getenv('LLAMA_CLOUD_API_KEY'),
            
            # Pinecone configuration
            'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
            'pinecone_index_name': os.getenv('PINECONE_INDEX_NAME', 'chartwell-insurance'),
            'pinecone_namespace': os.getenv('PINECONE_NAMESPACE', 'llama-namespace'),
            'pinecone_cloud': os.getenv('PINECONE_CLOUD', 'aws'),
            'pinecone_region': os.getenv('PINECONE_REGION', 'us-east-1'),
            
            # LLM configuration
            'llm_model': os.getenv('LLM_MODEL', 'gpt-4o'),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002'),
            
            # Agent configuration
            'agent_verbose': os.getenv('AGENT_VERBOSE', 'true').lower() == 'true',
        })
        
        print("✅ Configuration loaded from .env file")
    
    def _load_from_streamlit(self):
        """Load configuration from Streamlit secrets."""
        try:
            import streamlit as st
            
            # Core API keys
            self.config.update({
                'openai_api_key': st.secrets.get('OPENAI_API_KEY'),
                'llama_cloud_api_key': st.secrets.get('LLAMA_CLOUD_API_KEY'),
                
                # Pinecone configuration
                'pinecone_api_key': st.secrets.get('PINECONE_API_KEY'),
                'pinecone_index_name': st.secrets.get('PINECONE_INDEX_NAME', 'chartwell-insurance'),
                'pinecone_namespace': st.secrets.get('PINECONE_NAMESPACE', 'llama-namespace'),
                'pinecone_cloud': st.secrets.get('PINECONE_CLOUD', 'aws'),
                'pinecone_region': st.secrets.get('PINECONE_REGION', 'us-east-1'),
                
                # LLM configuration
                'llm_model': st.secrets.get('LLM_MODEL', 'gpt-4o'),
                'embedding_model': st.secrets.get('EMBEDDING_MODEL', 'text-embedding-ada-002'),
                
                # Agent configuration
                'agent_verbose': st.secrets.get('AGENT_VERBOSE', 'true').lower() == 'true',
            })
            
            print("✅ Configuration loaded from Streamlit secrets")
            
        except ImportError:
            raise ImportError("Streamlit is not installed. Install with: pip install streamlit")
    
    def _configure_llama_index(self):
        """Configure global LlamaIndex settings."""
        if not self.config.get('openai_api_key'):
            print("⚠️  Warning: OpenAI API key not found. Some features may not work.")
            return
        
        Settings.llm = OpenAI(
            model=self.config['llm_model'],
            api_key=self.config['openai_api_key']
        )
        Settings.embed_model = OpenAIEmbedding(
            model=self.config['embedding_model'],
            api_key=self.config['openai_api_key']
        )
        
        print(f"✅ LlamaIndex configured with {self.config['llm_model']}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def get_openai_config(self) -> Dict[str, str]:
        """Get OpenAI configuration."""
        return {
            'api_key': self.config.get('openai_api_key'),
            'llm_model': self.config.get('llm_model'),
            'embedding_model': self.config.get('embedding_model')
        }
    
    def get_pinecone_config(self) -> Dict[str, str]:
        """Get Pinecone configuration."""
        return {
            'api_key': self.config.get('pinecone_api_key'),
            'index_name': self.config.get('pinecone_index_name'),
            'namespace': self.config.get('pinecone_namespace'),
            'cloud': self.config.get('pinecone_cloud'),
            'region': self.config.get('pinecone_region')
        }
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            'verbose': self.config.get('agent_verbose'),
            'llm_model': self.config.get('llm_model')
        }
    
    def has_pinecone_config(self) -> bool:
        """Check if Pinecone configuration is available."""
        return self.config.get('pinecone_api_key') is not None
    
    def validate_required_keys(self, required_keys: list) -> bool:
        """Validate that required configuration keys are present."""
        missing_keys = []
        for key in required_keys:
            if not self.config.get(key):
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ Missing required configuration keys: {', '.join(missing_keys)}")
            return False
        
        print("✅ All required configuration keys are present")
        return True
    
    def print_config_summary(self, hide_sensitive: bool = True):
        """Print a summary of the current configuration."""
        print("\n📋 Configuration Summary:")
        print("=" * 40)
        
        for key, value in self.config.items():
            if hide_sensitive and 'api_key' in key.lower():
                # Show only first 8 and last 4 characters of API keys
                if value and len(value) > 12:
                    masked_value = f"{value[:8]}...{value[-4:]}"
                else:
                    masked_value = "***masked***" if value else "Not set"
                print(f"  {key}: {masked_value}")
            else:
                print(f"  {key}: {value}")
        
        print("=" * 40)
    
    def update_config(self, key: str, value: Any):
        """Update a configuration value."""
        self.config[key] = value
        print(f"✅ Updated {key} in configuration")
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get a copy of the entire configuration dictionary."""
        return self.config.copy()
