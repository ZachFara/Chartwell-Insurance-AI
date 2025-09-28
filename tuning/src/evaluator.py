"""
Evaluation components for hyperparameter tuning.
Handles relevancy and faithfulness evaluation of agent responses.
"""

import time
import difflib
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from llama_index.core.evaluation import FaithfulnessEvaluator, AnswerRelevancyEvaluator
from llama_index.llms.openai import OpenAI


class TuningEvaluator:
    """
    Handles evaluation of agent responses during hyperparameter tuning.
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        """Initialize evaluators."""
        self.openai = OpenAI(model=model, api_key=openai_api_key)
        self.faithfulness_evaluator = FaithfulnessEvaluator(llm=self.openai)
        self.relevancy_evaluator = AnswerRelevancyEvaluator(llm=self.openai)
    
    def evaluate_single_response(
        self, 
        agent, 
        question: str, 
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single agent response.
        
        Args:
            agent: The agent to evaluate
            question: The question to ask
            expected_answer: Expected answer for faithfulness evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        
        try:
            # Get agent's answer
            answer = agent.chat(question, get_response=True)
            response_time = time.time() - start_time
            
            # Extract the actual response text for storage
            response_text = answer.response if hasattr(answer, 'response') else str(answer)
            
            # Debug: Check what type we have
            print(f"    Answer type: {type(answer)}, has .response: {hasattr(answer, 'response')}")
            
            # Verify we have a proper response object
            if not hasattr(answer, 'response'):
                print(f"    ERROR: Expected AgentChatResponse but got {type(answer)}")
                raise ValueError(f"Agent returned {type(answer)} instead of AgentChatResponse")
            
            # Evaluate relevancy
            relevancy_score = self._evaluate_relevancy(question, answer)
            
            # Evaluate faithfulness if expected answer provided
            faithfulness_score = None
            if expected_answer:
                faithfulness_score = self._evaluate_faithfulness(
                    question, answer, response_text, expected_answer
                )
            
            return {
                'success': True,
                'answer': answer,
                'response_text': response_text,
                'response_time': response_time,
                'relevancy_score': relevancy_score,
                'faithfulness_score': faithfulness_score,
                'error': None
            }
            
        except Exception as e:
            print(f"    Error evaluating question: {str(e)}")
            return {
                'success': False,
                'answer': None,
                'response_text': f"ERROR: {str(e)}",
                'response_time': None,
                'relevancy_score': 0.0,
                'faithfulness_score': 0.0,
                'error': str(e)
            }
    
    def _evaluate_relevancy(self, question: str, answer) -> float:
        """Evaluate relevancy of answer to question."""
        try:
            print(f"    Evaluating relevancy with answer type: {type(answer)}")
            relevancy_result = self.relevancy_evaluator.evaluate_response(
                query=question,
                response=answer  # Pass the full AgentChatResponse object
            )
            # Handle different return types from the evaluator
            if hasattr(relevancy_result, 'score'):
                relevancy_score = relevancy_result.score
            elif hasattr(relevancy_result, 'passing'):
                relevancy_score = 1.0 if relevancy_result.passing else 0.0
            elif isinstance(relevancy_result, (int, float)):
                relevancy_score = float(relevancy_result)
            else:
                relevancy_score = 0.5  # Default score if we can't determine
        except Exception as eval_error:
            print(f"    Evaluation error: {eval_error}")
            relevancy_score = 0.0
        
        # Debug: Print the actual score we got
        print(f"    Relevancy score: {relevancy_score}")
        return relevancy_score
    
    def _evaluate_faithfulness(
        self, 
        question: str, 
        answer, 
        response_text: str, 
        expected_answer: str
    ) -> float:
        """Evaluate faithfulness of answer against expected answer."""
        try:
            # For faithfulness evaluation, we need to use the evaluate method instead of evaluate_response
            # and pass the expected answer differently
            faithfulness_result = self.faithfulness_evaluator.evaluate(
                query=question,
                response=answer.response,  # Use the text response
                contexts=[expected_answer]  # Expected answer as context
            )
            # Handle different return types from the faithfulness evaluator
            if hasattr(faithfulness_result, 'score'):
                faithfulness_score = faithfulness_result.score
            elif hasattr(faithfulness_result, 'passing'):
                faithfulness_score = 1.0 if faithfulness_result.passing else 0.0
            elif isinstance(faithfulness_result, (int, float)):
                faithfulness_score = float(faithfulness_result)
            else:
                faithfulness_score = 0.5
        except Exception as faith_error:
            print(f"    Faithfulness evaluation error: {faith_error}")
            # Try alternative approach - use the expected answer as ground truth in a different way
            try:
                # Simple semantic similarity as fallback
                similarity = difflib.SequenceMatcher(None, response_text.lower(), expected_answer.lower()).ratio()
                faithfulness_score = similarity
                print(f"    Using fallback similarity score: {faithfulness_score:.3f}")
            except Exception as fallback_error:
                print(f"    Fallback similarity calculation failed: {fallback_error}")
                faithfulness_score = 0.0
        
        return faithfulness_score
    
    def evaluate_dataset(
        self, 
        agent, 
        questions_df: pd.DataFrame
    ) -> Tuple[List[Dict], List[float], List[float], List[float]]:
        """
        Evaluate agent on a dataset of questions.
        
        Returns:
            Tuple of (evaluation_results, relevancy_scores, faithfulness_scores, response_times)
        """
        evaluation_results = []
        relevancy_scores = []
        faithfulness_scores = []
        response_times = []
        
        print(f"Evaluating on {len(questions_df)} questions...")
        
        for idx, row in questions_df.iterrows():
            question = row['question']
            expected_category = row['expected_category']
            keywords = row['keywords']
            evaluation_notes = row['evaluation_notes']
            expected_answer = row.get('expected_answer', None)  # Get expected answer if available
            
            print(f"  Question {idx+1}/{len(questions_df)}: {question[:50]}...")
            
            # Evaluate single response
            result = self.evaluate_single_response(agent, question, expected_answer)
            
            # Extract results
            if result['success']:
                response_times.append(result['response_time'])
                relevancy_scores.append(result['relevancy_score'])
                faithfulness_scores.append(result['faithfulness_score'] if result['faithfulness_score'] is not None else 0.0)
                
                # Store detailed results
                evaluation_results.append({
                    'question_idx': idx,
                    'question': question,
                    'answer': result['response_text'],
                    'expected_answer': expected_answer,
                    'expected_category': expected_category,
                    'keywords': keywords,
                    'evaluation_notes': evaluation_notes,
                    'relevancy_score': result['relevancy_score'],
                    'faithfulness_score': result['faithfulness_score'] if result['faithfulness_score'] is not None else 0.0,
                    'response_time': result['response_time'],
                })
            else:
                # Handle error case
                response_times.append(None)
                relevancy_scores.append(0.0)
                faithfulness_scores.append(0.0)
                evaluation_results.append({
                    'question_idx': idx,
                    'question': question,
                    'answer': result['response_text'],
                    'expected_answer': expected_answer,
                    'expected_category': expected_category,
                    'keywords': keywords,
                    'evaluation_notes': evaluation_notes,
                    'relevancy_score': 0.0,
                    'faithfulness_score': 0.0,
                    'response_time': None,
                })
        
        return evaluation_results, relevancy_scores, faithfulness_scores, response_times
