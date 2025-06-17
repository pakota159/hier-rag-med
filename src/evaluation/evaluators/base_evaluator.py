"""
Base evaluator class for medical RAG systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
from loguru import logger


class BaseEvaluator(ABC):
    """Abstract base class for model evaluators."""
    
    def __init__(self, config: Dict):
        """Initialize evaluator with model configuration."""
        self.config = config
        self.model_name = config.get("name", "unknown_model")
        self.system_path = config.get("system_path", "")
        self.enabled = config.get("enabled", True)
        
        logger.info(f"Initialized {self.model_name} evaluator")
    
    @abstractmethod
    def setup_model(self) -> None:
        """Initialize and setup the model for evaluation."""
        pass
    
    @abstractmethod
    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """Generate response for a given question."""
        pass
    
    @abstractmethod
    def retrieve_documents(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        pass
    
    def evaluate_benchmark(self, benchmark) -> Dict:
        """Evaluate model on a specific benchmark."""
        if not self.enabled:
            return {"error": "Evaluator disabled", "status": "skipped"}
        
        logger.info(f"Starting {benchmark.name} evaluation for {self.model_name}")
        start_time = time.time()
        
        try:
            # Setup model
            self.setup_model()
            
            # Get benchmark questions
            questions = benchmark.get_questions()
            results = []
            
            # Process each question
            for i, question in enumerate(questions):
                try:
                    result = self._evaluate_single_question(question, benchmark)
                    results.append(result)
                    
                    if (i + 1) % 50 == 0:
                        logger.info(f"Processed {i + 1}/{len(questions)} questions")
                        
                except Exception as e:
                    logger.error(f"Error processing question {i}: {e}")
                    results.append({
                        "question_id": question.get("id", f"q_{i}"),
                        "error": str(e),
                        "status": "failed"
                    })
            
            # Calculate final metrics
            metrics = benchmark.calculate_metrics(results)
            evaluation_time = time.time() - start_time
            
            final_results = {
                "model_name": self.model_name,
                "benchmark_name": benchmark.name,
                "evaluation_time": evaluation_time,
                "total_questions": len(questions),
                "successful_evaluations": len([r for r in results if "error" not in r]),
                "metrics": metrics,
                "individual_results": results,
                "status": "completed"
            }
            
            logger.info(f"Completed {benchmark.name} evaluation: {metrics.get('accuracy', 0):.2f}% accuracy")
            return final_results
            
        except Exception as e:
            logger.error(f"Evaluation failed for {self.model_name} on {benchmark.name}: {e}")
            return {
                "model_name": self.model_name,
                "benchmark_name": benchmark.name,
                "error": str(e),
                "status": "failed"
            }
    
    def _evaluate_single_question(self, question: Dict, benchmark) -> Dict:
        """Evaluate model on a single question."""
        question_text = question.get("question", "")
        question_id = question.get("id", "unknown")
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(question_text)
        
        # Generate response
        response = self.generate_response(question_text)
        
        # Evaluate using benchmark-specific criteria
        evaluation_result = benchmark.evaluate_response(question, response, retrieved_docs)
        
        # Add timing and metadata
        evaluation_result.update({
            "model_name": self.model_name,
            "question_id": question_id,
            "retrieved_docs_count": len(retrieved_docs)
        })
        
        return evaluation_result
    
    def get_model_info(self) -> Dict:
        """Get information about the model."""
        return {
            "name": self.model_name,
            "system_path": self.system_path,
            "enabled": self.enabled,
            "config": self.config
        }
    
    def validate_setup(self) -> bool:
        """Validate that the model is properly setup."""
        try:
            # Test basic functionality
            test_query = "What is diabetes?"
            docs = self.retrieve_documents(test_query, top_k=1)
            response = self.generate_response(test_query)
            
            return len(docs) > 0 and len(response) > 0
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False