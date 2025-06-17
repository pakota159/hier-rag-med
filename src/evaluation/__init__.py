"""
Evaluation Module for HierRAGMed
Comprehensive evaluation framework for medical RAG systems with MIRAGE, MedReason, PubMedQA, and MS MARCO benchmarks.
"""

# Core Evaluation Classes
from .evaluators.base_evaluator import BaseEvaluator
from .evaluators.kg_evaluator import KGEvaluator
from .evaluators.hierarchical_evaluator import HierarchicalEvaluator
from .evaluators.comparative_evaluator import ComparativeEvaluator

# Benchmark Implementations
from .benchmarks.base_benchmark import BaseBenchmark
from .benchmarks.mirage_benchmark import MIRAGEBenchmark
from .benchmarks.medreason_benchmark import MedReasonBenchmark
from .benchmarks.pubmedqa_benchmark import PubMedQABenchmark
from .benchmarks.msmarco_benchmark import MSMARCOBenchmark

# Metrics
from .metrics.base_metrics import BaseMetrics
from .metrics.qa_metrics import QAMetrics
from .metrics.retrieval_metrics import RetrievalMetrics
from .metrics.clinical_metrics import ClinicalMetrics
from .metrics.combined_metrics import CombinedMetrics

# Data Management
from .data.data_loader import BenchmarkDataLoader
from .data.data_processor import DataProcessor
from .data.data_validator import DataValidator

# Utilities
from .utils.result_processor import ResultProcessor
from .utils.visualization import EvaluationVisualizer
from .utils.report_generator import ReportGenerator
from .utils.statistical_analysis import StatisticalAnalysis

# Main Evaluation Functions
from .run_evaluation import run_evaluation, run_single_benchmark
from .compare_models import compare_models, generate_comparison_report

__version__ = "0.1.0"
__all__ = [
    # Core Evaluators
    "BaseEvaluator",
    "KGEvaluator", 
    "HierarchicalEvaluator",
    "ComparativeEvaluator",
    
    # Benchmarks
    "BaseBenchmark",
    "MIRAGEBenchmark",
    "MedReasonBenchmark",
    "PubMedQABenchmark",
    "MSMARCOBenchmark",
    
    # Metrics
    "BaseMetrics",
    "QAMetrics",
    "RetrievalMetrics", 
    "ClinicalMetrics",
    "CombinedMetrics",
    
    # Data Management
    "BenchmarkDataLoader",
    "DataProcessor",
    "DataValidator",
    
    # Utilities
    "ResultProcessor",
    "EvaluationVisualizer",
    "ReportGenerator",
    "StatisticalAnalysis",
    
    # Main Functions
    "run_evaluation",
    "run_single_benchmark",
    "compare_models",
    "generate_comparison_report"
]