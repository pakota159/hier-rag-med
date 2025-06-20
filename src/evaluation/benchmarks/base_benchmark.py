"""
FIXED: Base benchmark class for medical RAG evaluation.
Corrects sample size handling and accuracy calculation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import random
import math
import numpy as np
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
        """
        Calculate aggregate metrics from individual results - FIXED VERSION.
        
        This version properly handles overall_score as the primary accuracy metric
        and converts it to percentage for display.
        """
        if not results:
            return {"error": "No results to calculate metrics"}
        
        # Add sample information to metrics
        sample_info = self.get_sample_info()
        
        # FIXED: Properly handle overall_score as the main metric
        valid_scores = []
        failed_count = 0
        
        for r in results:
            # Skip failed results
            if "error" in r or r.get("status") == "failed":
                failed_count += 1
                continue
                
            # Get the overall score from the result
            score = r.get("overall_score", r.get("score", 0.0))
            if score is None:
                score = 0.0
            
            # Handle numpy types
            if hasattr(score, 'item'):
                score = float(score.item())
            else:
                score = float(score)
                
            valid_scores.append(score)
        
        # Calculate metrics
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            min_score = min(valid_scores)
            max_score = max(valid_scores)
            
            # FIXED: Accuracy is the average overall_score converted to percentage
            accuracy_percentage = avg_score * 100
            
            # Binary accuracy: count scores above threshold as "correct"
            threshold = 0.3  # 30% threshold for medical evaluation
            correct_count = sum(1 for score in valid_scores if score > threshold)
            binary_accuracy = (correct_count / len(valid_scores) * 100) if valid_scores else 0.0
            
        else:
            avg_score = 0.0
            min_score = 0.0
            max_score = 0.0
            accuracy_percentage = 0.0
            binary_accuracy = 0.0
            correct_count = 0
        
        metrics = {
            "total_questions": len(results),
            "valid_results": len(valid_scores),
            "failed_results": failed_count,
            "average_score": avg_score,
            "accuracy": accuracy_percentage,  # FIXED: Main accuracy metric
            "binary_accuracy": binary_accuracy,  # Alternative binary accuracy
            "correct_answers": correct_count,
            "min_score": min_score,
            "max_score": max_score,
            "benchmark_name": self.name,
            "sample_info": sample_info,
            "score_distribution": {
                "scores": valid_scores,
                "mean": avg_score,
                "std": float(np.std(valid_scores)) if valid_scores else 0.0,
                "median": float(np.median(valid_scores)) if valid_scores else 0.0
            }
        }
        
        # Log sample utilization with CORRECTED accuracy
        utilization = sample_info["utilization_percentage"]
        if utilization < 100:
            logger.info(f"   📊 {self.name}: Used {sample_info['effective_sample_size']} of {sample_info['total_available']} questions ({utilization:.1f}%)")
        else:
            logger.info(f"   📊 {self.name}: Used FULL dataset ({sample_info['total_available']} questions)")
        
        # FIXED: Log the corrected accuracy
        if len(valid_scores) > 0:
            logger.info(f"   🎯 {self.name}: Accuracy: {accuracy_percentage:.1f}% (range: {min_score*100:.1f}%-{max_score*100:.1f}%)")
        else:
            logger.warning(f"   ⚠️ {self.name}: No valid results to calculate accuracy")
        
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
            json.dump(save_data, f, indent=2, default=str)  # Added default=str for numpy types
        
        logger.info(f"💾 Saved {len(results)} results to {output_path}")
    
    def validate_config(self) -> bool:
        """Validate benchmark configuration."""
        required_fields = ["name"]
        for field in required_fields:
            if field not in self.config:
                logger.error(f"❌ Missing required config field: {field}")
                return False
        
        return True
    
    def get_evaluation_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate evaluation summary for this benchmark."""
        metrics = self.calculate_metrics(results)
        
        summary = {
            "benchmark_name": self.name,
            "total_questions": metrics.get("total_questions", 0),
            "accuracy": metrics.get("accuracy", 0.0),
            "average_score": metrics.get("average_score", 0.0),
            "score_range": {
                "min": metrics.get("min_score", 0.0) * 100,
                "max": metrics.get("max_score", 0.0) * 100
            },
            "performance_category": self._categorize_performance(metrics.get("accuracy", 0.0)),
            "sample_info": metrics.get("sample_info", {}),
            "recommendations": self._generate_recommendations(metrics)
        }
        
        return summary
    
    def _categorize_performance(self, accuracy: float) -> str:
        """Categorize performance based on accuracy."""
        if accuracy >= 70:
            return "excellent"
        elif accuracy >= 60:
            return "good"
        elif accuracy >= 50:
            return "fair"
        elif accuracy >= 30:
            return "poor"
        else:
            return "very_poor"
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate recommendations based on performance."""
        recommendations = []
        accuracy = metrics.get("accuracy", 0.0)
        
        if accuracy < 30:
            recommendations.append("Consider reviewing fundamental medical knowledge base")
            recommendations.append("Check if retrieval system is finding relevant documents")
        elif accuracy < 50:
            recommendations.append("Focus on improving reasoning chain quality")
            recommendations.append("Enhance context integration mechanisms")
        elif accuracy < 70:
            recommendations.append("Fine-tune response generation for better precision")
            recommendations.append("Optimize retrieval ranking algorithms")
        else:
            recommendations.append("Performance is excellent - consider testing on harder datasets")
            recommendations.append("Focus on edge case handling and robustness")
        
        # Add sample-specific recommendations
        valid_results = metrics.get("valid_results", 0)
        total_questions = metrics.get("total_questions", 0)
        
        if valid_results < total_questions:
            recommendations.append("Investigate and fix causes of failed evaluations")
        
        return recommendations