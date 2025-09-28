"""
Main hyperparameter tuning orchestrator.
Coordinates the entire tuning process.
"""

import sys
import os
import pandas as pd

# Add repository root to path to import from src
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

from src.configuration import Configuration
from src.agent import Agent
from tuning.src.evaluator import TuningEvaluator
from tuning.src.hyperparameter_sampler import HyperparameterSampler
from tuning.src.results_manager import ResultsManager


class HyperparameterTuner:
    """
    Main class for orchestrating hyperparameter tuning runs.
    """
    
    def __init__(self, questions_file: str = 'tuning/data/eval/sample_questions.csv'):
        """Initialize the tuner."""
        # Load configuration
        self.config = Configuration()
        openai_api_key = self.config.get('openai_api_key')
        
        # Initialize components
        self.evaluator = TuningEvaluator(openai_api_key)
        self.sampler = HyperparameterSampler()
        self.results_manager = ResultsManager()
        
        # Pass sampler to results manager for prompt identification
        self.results_manager.set_prompt_sampler(self.sampler)
        
        # Load questions dataset
        self.questions_df = pd.read_csv(questions_file)
        print(f"Loaded {len(self.questions_df)} evaluation questions")
    
    def run_tuning(self, iterations: int = 10) -> None:
        """
        Run hyperparameter tuning for specified number of iterations.
        
        Args:
            iterations: Number of tuning iterations to run
        """
        print(f"Starting hyperparameter tuning with {iterations} iterations...")
        print("=" * 60)
        
        for i in range(iterations):
            try:
                print(f"Starting tuning iteration {i+1}/{iterations}")
                
                # Sample hyperparameters
                sampled_params = self.sampler.sample_hyperparameters()
                print(f"Sampled hyperparameters: {sampled_params}")
                
                # Initialize agent with sampled hyperparameters
                agent = Agent(
                    name="TuningAgent",
                    use_pinecone=False,  # Use local indexing for tuning, not Pinecone
                    chunk_size=sampled_params['chunk_size'],
                    chunk_overlap=sampled_params['chunk_overlap'],
                    similarity_top_k=sampled_params['similarity_top_k'],
                    system_prompt_override=sampled_params['system_prompt_override']
                )
                
                # Ingest documents (assumes documents are in 'data/documents/')
                # For local indexing, we need to ingest documents each time
                agent.ingest_directory('data/raw/')  # Use the same directory as your original
                
                # Evaluate agent
                evaluation_results, relevancy_scores, faithfulness_scores, response_times = \
                    self.evaluator.evaluate_dataset(agent, self.questions_df)
                
                # Save and print results
                self.results_manager.save_iteration_results(
                    i+1, sampled_params, evaluation_results,
                    relevancy_scores, faithfulness_scores, response_times,
                    self.questions_df
                )
                
                self.results_manager.print_iteration_results(
                    i+1, iterations, sampled_params,
                    relevancy_scores, faithfulness_scores, response_times,
                    self.questions_df
                )
                
            except Exception as e:
                print(f"Error in iteration {i+1}: {str(e)}")
                continue
        
        # Print final summary
        self._print_final_summary()
    
    def _print_final_summary(self) -> None:
        """Print final summary of tuning results."""
        print("Hyperparameter tuning complete! Results saved in tuning/results/")
        
        # Get best configuration
        best_config = self.results_manager.get_best_configuration()
        
        if best_config:
            print("\nBest performing configuration:")
            for key, value in best_config.items():
                if key != 'iteration':
                    print(f"  {key}: {value}")
        else:
            print("No valid configurations found in results.")
