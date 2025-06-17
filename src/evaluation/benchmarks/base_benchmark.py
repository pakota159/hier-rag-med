"""
Base benchmark class for medical RAG evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import random
from loguru import logger


class BaseBenchmark(ABC):
    """Abstract base class for medical benchmarks."""
    
    def __init__(self, config: Dict):
        """Initialize benchmark with configuration."""
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.sample_size = config.get("sample_size", 1000)
        self.random_seed = config.get("random_seed", 42)
        self.data = None
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        logger.info(f"Initialized {self.name} benchmark (sample_size: {self.sample_size})")
    
    @abstractmethod
    def load_dataset(self) -> List[Dict]:
        """Load and return the benchmark dataset."""
        pass
    
    @abstractmethod
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate a model response against ground truth."""
        pass
    
    def get_questions(self) -> List[Dict]:
        """Get benchmark questions for evaluation."""
        if self.data is None:
            self.data = self.load_dataset()
        
        # Sample questions if needed
        if len(self.data) > self.sample_size:
            sampled_data = random.sample(self.data, self.sample_size)
            logger.info(f"Sampled {self.sample_size} questions from {len(self.data)} total")
            return sampled_data
        
        return self.data
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics from individual results."""
        if not results:
            return {"error": "No results to calculate metrics"}
        
        # Calculate basic metrics
        total_score = sum(r.get("score", 0) for r in results)
        avg_score = total_score / len(results)
        
        # Count correct answers
        correct_count = sum(1 for r in results if r.get("correct", False))
        accuracy = correct_count / len(results)
        
        return {
            "total_questions": len(results),
            "average_score": avg_score,
            "accuracy": accuracy * 100,
            "correct_answers": correct_count,
            "benchmark_name": self.name
        }
    
    def save_results(self, results: List[Dict], output_path: Path) -> None:
        """Save evaluation results to file."""
        output_data = {
            "benchmark": self.name,
            "config": self.config,
            "results": results,
            "metrics": self.calculate_metrics(results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved {len(results)} results to {output_path}")
    
    def validate_question(self, question: Dict) -> bool:
        """Validate that a question has required fields."""
        required_fields = ["question", "answer"]
        return all(field in question for field in required_fields)