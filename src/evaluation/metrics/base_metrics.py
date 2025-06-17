"""
Base metrics class for medical RAG evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union
import numpy as np
from loguru import logger


class BaseMetrics(ABC):
    """Abstract base class for evaluation metrics."""
    
    def __init__(self, config: Dict = None):
        """Initialize base metrics."""
        self.config = config or {}
        self.name = self.__class__.__name__
        
    @abstractmethod
    def calculate(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """Calculate metrics given predictions and references."""
        pass
    
    def validate_inputs(self, predictions: List[str], references: List[str]) -> bool:
        """Validate input data."""
        if len(predictions) != len(references):
            logger.error(f"Predictions ({len(predictions)}) and references ({len(references)}) length mismatch")
            return False
        
        if not predictions or not references:
            logger.error("Empty predictions or references")
            return False
        
        return True
    
    def normalize_scores(self, scores: Dict[str, float], scale: str = "0-100") -> Dict[str, float]:
        """Normalize scores to specified scale."""
        if scale == "0-100":
            return {k: v * 100 if 0 <= v <= 1 else v for k, v in scores.items()}
        elif scale == "0-1":
            return {k: v / 100 if v > 1 else v for k, v in scores.items()}
        return scores
    
    def aggregate_scores(self, score_list: List[float], method: str = "mean") -> float:
        """Aggregate list of scores using specified method."""
        if not score_list:
            return 0.0
        
        if method == "mean":
            return np.mean(score_list)
        elif method == "median":
            return np.median(score_list)
        elif method == "max":
            return np.max(score_list)
        elif method == "min":
            return np.min(score_list)
        else:
            return np.mean(score_list)
    
    def calculate_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval for scores."""
        if len(scores) < 2:
            return {"mean": np.mean(scores) if scores else 0, "ci_lower": 0, "ci_upper": 0}
        
        mean_score = np.mean(scores)
        std_error = np.std(scores, ddof=1) / np.sqrt(len(scores))
        
        # Use t-distribution for small samples
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
        margin_error = t_value * std_error
        
        return {
            "mean": mean_score,
            "ci_lower": mean_score - margin_error,
            "ci_upper": mean_score + margin_error,
            "std_error": std_error
        }
    
    def get_metric_info(self) -> Dict[str, Any]:
        """Get information about this metric."""
        return {
            "name": self.name,
            "type": "base_metric",
            "config": self.config,
            "description": self.__doc__ or "Base metric class"
        }