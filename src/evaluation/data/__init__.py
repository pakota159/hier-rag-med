"""
Evaluation Data Module for HierRAGMed
Handles loading, processing, and validation of benchmark datasets for comprehensive evaluation.
"""

from .data_loader import BenchmarkDataLoader
from .data_processor import DataProcessor  
from .data_validator import DataValidator

__version__ = "0.1.0"
__all__ = [
    "BenchmarkDataLoader",
    "DataProcessor", 
    "DataValidator"
]