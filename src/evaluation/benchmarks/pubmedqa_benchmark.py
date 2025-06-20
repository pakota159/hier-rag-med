# src/evaluation/benchmarks/pubmedqa_benchmark.py
"""
PubMedQA Benchmark - DISABLED
This benchmark has been disabled because PubMedQA questions are already included in the MIRAGE benchmark.
"""

from .base_benchmark import BaseBenchmark
from loguru import logger
from typing import Dict, List, Any


class PubMedQABenchmark(BaseBenchmark):
    """
    PubMedQA benchmark - DISABLED.
    
    This benchmark is disabled because:
    1. PubMedQA questions are already included in the MIRAGE benchmark
    2. MIRAGE provides a more comprehensive evaluation covering both clinical reasoning and research questions
    3. Avoiding duplicate evaluation of the same question types
    
    Use MIRAGE benchmark instead for research literature QA evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "PubMedQA (DISABLED)"
        self.disabled = True
        self.disabled_reason = "PubMedQA questions are already included in the MIRAGE benchmark"
        
        logger.warning("⚠️ PubMedQA benchmark is DISABLED")
        logger.info("   Reason: Questions already covered by MIRAGE benchmark")
        logger.info("   Alternative: Use MIRAGE benchmark for research literature evaluation")
    
    def load_dataset(self) -> List[Dict]:
        """Return empty dataset as this benchmark is disabled."""
        logger.error("❌ PubMedQA benchmark is disabled - use MIRAGE instead")
        return []
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[str]) -> Dict[str, Any]:
        """Return empty evaluation as this benchmark is disabled."""
        logger.error("❌ PubMedQA evaluation disabled - use MIRAGE instead")
        return {
            "error": "PubMedQA benchmark is disabled",
            "reason": self.disabled_reason,
            "alternative": "Use MIRAGE benchmark for research literature evaluation",
            "overall_score": 0.0
        }
    
    def get_evaluation_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Return disabled status summary."""
        return {
            "benchmark": "PubMedQA",
            "status": "DISABLED",
            "reason": self.disabled_reason,
            "alternative": "MIRAGE benchmark includes PubMedQA-style research questions",
            "total_questions": 0,
            "message": "Use MIRAGE benchmark for comprehensive medical QA evaluation including research literature"
        }
    
    @staticmethod
    def is_enabled() -> bool:
        """Return False as this benchmark is disabled."""
        return False
    
    @staticmethod
    def get_alternative() -> str:
        """Return alternative benchmark recommendation."""
        return "MIRAGE"
    
    @staticmethod
    def get_disable_reason() -> str:
        """Return reason for disabling."""
        return "PubMedQA questions are already included in the MIRAGE benchmark"