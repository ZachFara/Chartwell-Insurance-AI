"""
Clean, refactored hyperparameter tuning script.
Maintains exact same functionality as the original main.py.
"""

import argparse
import sys
import os

# Add the repository root to Python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from tuning.src.tuning_orchestrator import HyperparameterTuner


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for RAG agent.")
    parser.add_argument('--iterations', type=int, default=10, 
                       help='Number of tuning iterations to run')
    parser.add_argument('--questions-file', type=str, 
                       default='tuning/data/eval/sample_questions.csv',
                       help='Path to evaluation questions CSV file')
    
    args = parser.parse_args()
    
    # Initialize and run tuner
    tuner = HyperparameterTuner(questions_file=args.questions_file)
    tuner.run_tuning(iterations=args.iterations)
