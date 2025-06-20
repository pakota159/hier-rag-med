"""
FIXED: Base benchmark class for medical RAG evaluation.
Corrects sample size handling to properly support unlimited datasets.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import random
import math
from loguru import logger


class BaseBenchmark(ABC):
    """Abstract base class for medical benchmarks with FIXED sample handling."""
    
    def __init__(self, config: Dict):
        """Initialize benchmark with configuration."""
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        
        # FIXED: Proper sample size handling
        configured_sample_size = config.get("sample_size", config.get("max_samples", 1000))
        
        # Handle unlimited datasets (full mode)
        if configured_sample_size is None or configured_sample_size == float('inf'):
            self.sample_size = None  # Unlimited
            self.is_unlimited = True
            logger.info(f"   📊 {self.name}: UNLIMITED samples (full dataset)")
        else:
            self.sample_size = int(configured_sample_size)
            self.is_unlimited = False
            logger.info(f"   📊 {self.name}: LIMITED to {self.sample_size} samples")
        
        self.random_seed = config.get("random_seed", 42)
        self.data = None
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        logger.info(f"Initialized {self.name} benchmark")
    
    @abstractmethod
    def load_dataset(self) -> List[Dict]:
        """Load and return the benchmark dataset."""
        pass
    
    @abstractmethod
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate a model response against ground truth."""
        pass
    
    def get_questions(self) -> List[Dict]:
        """Get benchmark questions for evaluation with FIXED sampling logic."""
        if self.data is None:
            self.data = self.load_dataset()
        
        total_available = len(self.data)
        logger.info(f"   📋 {self.name}: {total_available} questions available")
        
        # FIXED: Handle unlimited vs limited sampling
        if self.is_unlimited or self.sample_size is None:
            logger.info(f"   ✅ {self.name}: Using ALL {total_available} questions (full dataset)")
            return self.data
        elif total_available <= self.sample_size:
            logger.info(f"   ✅ {self.name}: Using ALL {total_available} questions (less than limit)")
            return self.data
        else:
            # Sample questions if dataset is larger than requested sample size
            sampled_data = random.sample(self.data, self.sample_size)
            logger.info(f"   ✂️ {self.name}: Sampled {self.sample_size} from {total_available} questions")
            return sampled_data
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get information about the current sampling configuration."""
        if self.data is None:
            self.data = self.load_dataset()
        
        total_available = len(self.data)
        
        if self.is_unlimited:
            effective_sample_size = total_available
            sampling_mode = "unlimited"
        elif self.sample_size is None:
            effective_sample_size = total_available
            sampling_mode = "unlimited"
        else:
            effective_sample_size = min(total_available, self.sample_size)
            sampling_mode = "limited"
        
        return {
            "benchmark_name": self.name,
            "total_available": total_available,
            "configured_limit": self.sample_size,
            "effective_sample_size": effective_sample_size,
            "sampling_mode": sampling_mode,
            "is_unlimited": self.is_unlimited,
            "utilization_percentage": (effective_sample_size / total_available) * 100 if total_available > 0 else 0
        }
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics from individual results."""
        if not results:
            return {"error": "No results to calculate metrics"}
        
        # Add sample information to metrics
        sample_info = self.get_sample_info()
        
        # Calculate basic metrics
        total_score = sum(r.get("score", 0) for r in results)
        avg_score = total_score / len(results)
        
        # Count correct answers
        correct_count = sum(1 for r in results if r.get("correct", False))
        accuracy = correct_count / len(results)
        
        metrics = {
            "total_questions": len(results),
            "average_score": avg_score,
            "accuracy": accuracy * 100,
            "correct_answers": correct_count,
            "benchmark_name": self.name,
            "sample_info": sample_info
        }
        
        # Log sample utilization
        utilization = sample_info["utilization_percentage"]
        if utilization < 100:
            logger.info(f"   📊 {self.name}: Used {sample_info['effective_sample_size']} of {sample_info['total_available']} questions ({utilization:.1f}%)")
        else:
            logger.info(f"   📊 {self.name}: Used FULL dataset ({sample_info['total_available']} questions)")
        
        return metrics
    
    def save_results(self, results: List[Dict], output_path: Path) -> None:
        """Save evaluation results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Include sample information in saved results
        save_data = {
            "benchmark_name": self.name,
            "sample_info": self.get_sample_info(),
            "results": results,
            "metrics": self.calculate_metrics(results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"💾 Saved {len(results)} results to {output_path}")
    
    def validate_config(self) -> bool:
        """Validate benchmark configuration."""
        required_fields = ["name"]
        for field in required_fields:
            if field not in self.config:
                logger.error(f"❌ Missing required config field: {field}")
                return False
        
        return True