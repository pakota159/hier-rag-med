"""
Fully Updated MIRAGE Benchmark with Real Dataset Loading
Uses the official MIRAGE benchmark from Teddy-XiongGZ/MIRAGE repository
"""

import re
import json
import sys
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Any, Set

from .base_benchmark import BaseBenchmark
from loguru import logger

# Add project root to path for data loader import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.data.data_loader import BenchmarkDataLoader

class MIRAGEBenchmark(BaseBenchmark):
    """MIRAGE benchmark with official dataset loading."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MIRAGE"
        self.data_loader = BenchmarkDataLoader(config)
        
        # Initialize evaluation models
        self.similarity_model = None
        self._init_evaluation_models()
        
        # Medical patterns for enhanced evaluation
        self.medical_patterns = {
            'symptoms': ['pain', 'fever', 'nausea', 'vomiting', 'diarrhea', 'fatigue', 'headache'],
            'conditions': ['diabetes', 'hypertension', 'heart failure', 'stroke', 'asthma', 'cancer'],
            'treatments': ['medication', 'therapy', 'treatment', 'management', 'intervention', 'surgery'],
            'investigations': ['blood test', 'x-ray', 'ct scan', 'mri', 'ecg', 'examination', 'biopsy']
        }
        
        self.reasoning_indicators = {
            'systematic': ['systematic', 'step-by-step', 'approach', 'method', 'process'],
            'differential': ['differential', 'consider', 'rule out', 'exclude', 'possible'],
            'evidence': ['evidence', 'studies', 'research', 'guidelines', 'literature'],
            'causality': ['because', 'due to', 'caused by', 'leads to', 'results in']
        }
    
    def load_dataset(self) -> List[Dict]:
        """Load official MIRAGE dataset with caching."""
        logger.info(f"🔄 Loading official MIRAGE benchmark...")
        
        try:
            # Use the centralized data loader
            max_samples = self.sample_size if not self.is_unlimited else None
            data = self.data_loader.load_benchmark_data("mirage", max_samples=max_samples)
            
            if data and len(data) > 0:
                logger.info(f"✅ Loaded {len(data)} MIRAGE questions from data loader")
                return data
            else:
                raise ConnectionError("Failed to load MIRAGE benchmark data. Please check your internet connection and the data source.")
            
        except Exception as e:
            logger.error(f"❌ Failed to load official MIRAGE data: {e}")
            raise e
    
    def _init_evaluation_models(self):
        """Initialize models for evaluation metrics."""
        try:
            # Try to load sentence transformers for semantic similarity
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Loaded sentence transformer for evaluation")
        except ImportError:
            logger.warning("⚠️ Sentence transformers not available, using basic evaluation")
            self.similarity_model = None
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate a model response against MIRAGE question."""
        correct_answer = question.get("answer", "")
        options = question.get("options", [])
        
        # Exact match evaluation
        exact_match = self._check_exact_match(response, correct_answer, options)
        
        # Semantic similarity (if available)
        semantic_score = self._calculate_semantic_similarity(response, correct_answer)
        
        # Medical terminology usage
        medical_score = self._evaluate_medical_terminology(response, question)
        
        # Reasoning quality
        reasoning_score = self._evaluate_reasoning_quality(response, question)
        
        return {
            "exact_match": exact_match,
            "semantic_similarity": semantic_score,
            "medical_terminology": medical_score,
            "reasoning_quality": reasoning_score,
            "overall_score": (exact_match * 0.4 + semantic_score * 0.3 + 
                            medical_score * 0.15 + reasoning_score * 0.15),
            "question_difficulty": question.get("difficulty", "medium"),
            "medical_specialty": question.get("medical_specialty", "general"),
            "reasoning_type": question.get("reasoning_type", "factual_recall")
        }
    
    def _check_exact_match(self, response: str, correct_answer: str, options: List[str]) -> bool:
        """Check for exact match with answer."""
        response_clean = response.strip().lower()
        correct_clean = correct_answer.strip().lower()
        
        # Direct match
        if correct_clean in response_clean:
            return True
        
        # Check if response contains any of the options that match correct answer
        for option in options:
            option_clean = option.strip().lower()
            if option_clean == correct_clean and option_clean in response_clean:
                return True
        
        return False
    
    def _calculate_semantic_similarity(self, response: str, correct_answer: str) -> float:
        """Calculate semantic similarity between response and correct answer."""
        if not self.similarity_model:
            return 0.0
        
        try:
            embeddings = self.similarity_model.encode([response, correct_answer])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except:
            return 0.0
    
    def _evaluate_medical_terminology(self, response: str, question: Dict) -> float:
        """Evaluate proper use of medical terminology."""
        response_lower = response.lower()
        
        # Count medical terms used appropriately
        medical_term_count = 0
        total_medical_terms = 0
        
        for category, terms in self.medical_patterns.items():
            for term in terms:
                total_medical_terms += 1
                if term in response_lower:
                    medical_term_count += 1
        
        if total_medical_terms == 0:
            return 0.5  # Neutral score if no medical terms expected
        
        return medical_term_count / total_medical_terms
    
    def _evaluate_reasoning_quality(self, response: str, question: Dict) -> float:
        """Evaluate quality of medical reasoning in response."""
        response_lower = response.lower()
        reasoning_type = question.get("reasoning_type", "factual_recall")
        
        # Look for reasoning indicators based on question type
        if reasoning_type in self.reasoning_indicators:
            indicators = self.reasoning_indicators[reasoning_type]
            found_indicators = sum(1 for indicator in indicators if indicator in response_lower)
            return min(found_indicators / len(indicators), 1.0)
        
        # General reasoning quality indicators
        reasoning_words = ["because", "therefore", "due to", "caused by", "leads to", "results in"]
        found_reasoning = sum(1 for word in reasoning_words if word in response_lower)
        
        return min(found_reasoning / 3, 1.0)  # Normalize to max 1.0