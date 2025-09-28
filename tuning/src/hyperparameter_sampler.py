"""
Hyperparameter sampling and management for tuning runs.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional


class HyperparameterSampler:
    """
    Handles hyperparameter space definition and sampling.
    """
    
    def __init__(self, system_prompts_file: str = 'tuning/data/system_prompts/system_prompts.json'):
        """Initialize hyperparameter space."""
        # Load system prompts from file
        with open(system_prompts_file, 'r') as f:
            self.system_prompts_dict = json.load(f)
        self.system_prompts = [v for k, v in self.system_prompts_dict.items()]
        
        # Create mapping from prompt text to prompt name for identification
        self.prompt_text_to_name = {v: k for k, v in self.system_prompts_dict.items()}
        
        # Define hyperparameter space
        self.hyperparameters = { 
            'chunk_size': np.arange(256, 1025, 128).tolist(),
            'chunk_overlap': np.arange(0, 257, 32).tolist(),
            'similarity_top_k': np.arange(1, 11).tolist(),
            'system_prompt_override': [None] + self.system_prompts 
        }
    
    def sample_hyperparameters(self) -> Dict[str, Any]:
        """Sample a random set of hyperparameters."""
        return { 
            'chunk_size': int(np.random.choice(self.hyperparameters['chunk_size'])),
            'chunk_overlap': int(np.random.choice(self.hyperparameters['chunk_overlap'])),
            'similarity_top_k': int(np.random.choice(self.hyperparameters['similarity_top_k'])),
            'system_prompt_override': np.random.choice(self.hyperparameters['system_prompt_override'])
        }
    
    def get_hyperparameter_space(self) -> Dict[str, List]:
        """Get the full hyperparameter space."""
        return self.hyperparameters.copy()
    
    def get_system_prompts(self) -> List[str]:
        """Get list of available system prompts."""
        return self.system_prompts.copy()
    
    def get_prompt_name(self, prompt_text: str) -> str:
        """Get the name/identifier for a system prompt."""
        if prompt_text is None:
            return "default"
        
        # Look up the name for this prompt text
        return self.prompt_text_to_name.get(prompt_text, "unknown_custom")
    
    def get_prompt_names(self) -> List[str]:
        """Get list of available system prompt names."""
        return ["default"] + list(self.system_prompts_dict.keys())
