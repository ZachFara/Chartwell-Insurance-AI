"""
Results management for hyperparameter tuning runs.
Handles saving detailed results and iteration summaries.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any


class ResultsManager:
    """
    Manages saving and organizing tuning results.
    """
    
    def __init__(self, results_dir: str = 'tuning/results'):
        """Initialize results manager."""
        self.results_dir = results_dir
        self.prompt_sampler = None
        os.makedirs(self.results_dir, exist_ok=True)
    
    def set_prompt_sampler(self, sampler):
        """Set the hyperparameter sampler for prompt identification."""
        self.prompt_sampler = sampler
    
    def save_iteration_results(
        self,
        iteration: int,
        sampled_params: Dict[str, Any],
        evaluation_results: List[Dict],
        relevancy_scores: List[float],
        faithfulness_scores: List[float],
        response_times: List[float],
        questions_df: pd.DataFrame
    ) -> None:
        """Save results for a single iteration."""
        timestamp = datetime.now().isoformat()
        
        # Get the specific system prompt name
        if self.prompt_sampler:
            system_prompt_name = self.prompt_sampler.get_prompt_name(sampled_params['system_prompt_override'])
        else:
            system_prompt_name = 'custom' if sampled_params['system_prompt_override'] else 'default'
        
        # Add iteration info to each evaluation result
        for result in evaluation_results:
            result.update({
                'iteration': iteration,
                'timestamp': timestamp,
                'chunk_size': sampled_params['chunk_size'],
                'chunk_overlap': sampled_params['chunk_overlap'],
                'similarity_top_k': sampled_params['similarity_top_k'],
                'system_prompt_name': system_prompt_name,
                'system_prompt': sampled_params['system_prompt_override'][:50] + "..." if sampled_params['system_prompt_override'] else None
            })
        
        # Save detailed results
        self._save_detailed_results(evaluation_results)
        
        # Save iteration summary
        self._save_iteration_summary(
            iteration, timestamp, sampled_params, 
            relevancy_scores, faithfulness_scores, response_times, questions_df
        )
    
    def _save_detailed_results(self, evaluation_results: List[Dict]) -> None:
        """Save detailed results to CSV."""
        results_file = os.path.join(self.results_dir, 'detailed_results.csv')
        results_df = pd.DataFrame(evaluation_results)
        
        if os.path.exists(results_file):
            # Append to existing file
            results_df.to_csv(results_file, mode='a', header=False, index=False)
        else:
            # Create new file with headers
            results_df.to_csv(results_file, index=False)
    
    def _save_iteration_summary(
        self,
        iteration: int,
        timestamp: str,
        sampled_params: Dict[str, Any],
        relevancy_scores: List[float],
        faithfulness_scores: List[float],
        response_times: List[float],
        questions_df: pd.DataFrame
    ) -> None:
        """Save iteration summary to CSV."""
        # Compute statistics
        valid_relevancy_scores = [s for s in relevancy_scores if s is not None]
        valid_faithfulness_scores = [s for s in faithfulness_scores if s is not None]
        valid_response_times = [t for t in response_times if t is not None]
        
        avg_relevancy = np.mean(valid_relevancy_scores) if valid_relevancy_scores else 0.0
        avg_faithfulness = np.mean(valid_faithfulness_scores) if valid_faithfulness_scores else 0.0
        avg_response_time = np.mean(valid_response_times) if valid_response_times else 0.0
        
        # Get the specific system prompt name
        if self.prompt_sampler:
            system_prompt_name = self.prompt_sampler.get_prompt_name(sampled_params['system_prompt_override'])
        else:
            system_prompt_name = 'custom' if sampled_params['system_prompt_override'] else 'default'
        
        # Create summary data
        summary_data = {
            'iteration': iteration,
            'timestamp': timestamp,
            'chunk_size': sampled_params['chunk_size'],
            'chunk_overlap': sampled_params['chunk_overlap'],
            'similarity_top_k': sampled_params['similarity_top_k'],
            'system_prompt_name': system_prompt_name,
            'system_prompt_type': 'custom' if sampled_params['system_prompt_override'] else 'default',
            'avg_relevancy': avg_relevancy,
            'avg_faithfulness': avg_faithfulness,
            'avg_response_time': avg_response_time,
            'valid_responses': len(valid_relevancy_scores),
            'total_questions': len(questions_df),
            'individual_relevancy_scores': str(valid_relevancy_scores),  # Convert to string for CSV
            'individual_faithfulness_scores': str(valid_faithfulness_scores),
            'individual_response_times': str(valid_response_times)
        }
        
        # Save to CSV
        summary_file = os.path.join(self.results_dir, 'iteration_summary.csv')
        summary_df = pd.DataFrame([summary_data])
        
        if os.path.exists(summary_file):
            summary_df.to_csv(summary_file, mode='a', header=False, index=False)
        else:
            summary_df.to_csv(summary_file, index=False)
    
    def print_iteration_results(
        self,
        iteration: int,
        total_iterations: int,
        sampled_params: Dict[str, Any],
        relevancy_scores: List[float],
        faithfulness_scores: List[float],
        response_times: List[float],
        questions_df: pd.DataFrame
    ) -> None:
        """Print iteration results to console."""
        # Compute statistics
        valid_relevancy_scores = [s for s in relevancy_scores if s is not None]
        valid_faithfulness_scores = [s for s in faithfulness_scores if s is not None]
        valid_response_times = [t for t in response_times if t is not None]
        
        # Log individual scores for debugging
        print(f"  Individual relevancy scores: {valid_relevancy_scores}")
        print(f"  Individual faithfulness scores: {valid_faithfulness_scores}")
        
        avg_relevancy = np.mean(valid_relevancy_scores) if valid_relevancy_scores else 0.0
        avg_faithfulness = np.mean(valid_faithfulness_scores) if valid_faithfulness_scores else 0.0
        avg_response_time = np.mean(valid_response_times) if valid_response_times else 0.0
        
        print(f"Iteration {iteration} results:")
        print(f"  Average Relevancy: {avg_relevancy:.3f} (range: {min(valid_relevancy_scores) if valid_relevancy_scores else 'N/A'}-{max(valid_relevancy_scores) if valid_relevancy_scores else 'N/A'})")
        print(f"  Average Faithfulness: {avg_faithfulness:.3f}")
        print(f"  Average Response Time: {avg_response_time:.2f}s")
        print(f"  Valid Responses: {len(valid_relevancy_scores)}/{len(questions_df)}")
        print(f"Completed iteration {iteration}/{total_iterations}")
        print("-" * 50)
    
    def get_best_configuration(self) -> Dict[str, Any]:
        """Get the best performing configuration from results."""
        summary_file = os.path.join(self.results_dir, 'iteration_summary.csv')
        
        if not os.path.exists(summary_file):
            return {}
        
        try:
            df = pd.read_csv(summary_file)
            if len(df) == 0:
                return {}
            
            # Find best configuration based on combined relevancy and faithfulness
            df['combined_score'] = (df['avg_relevancy'] + df['avg_faithfulness']) / 2
            best_idx = df['combined_score'].idxmax()
            best_config = df.iloc[best_idx]
            
            return {
                'iteration': best_config['iteration'],
                'chunk_size': best_config['chunk_size'],
                'chunk_overlap': best_config['chunk_overlap'],
                'similarity_top_k': best_config['similarity_top_k'],
                'system_prompt_type': best_config['system_prompt_type'],
                'avg_relevancy': best_config['avg_relevancy'],
                'avg_faithfulness': best_config['avg_faithfulness'],
                'combined_score': best_config['combined_score']
            }
        except Exception as e:
            print(f"Error finding best configuration: {e}")
            return {}
