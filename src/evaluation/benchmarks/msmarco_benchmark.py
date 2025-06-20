"""
Enhanced MS MARCO Benchmark with Real Dataset Loading - FIXED VERSION
"""

import re
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from .base_benchmark import BaseBenchmark
from loguru import logger

from src.evaluation.data.data_loader import BenchmarkDataLoader

# Add project root to path for data loader import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

class MSMARCOBenchmark(BaseBenchmark):
    """MS MARCO benchmark with real dataset loading."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MS MARCO"
        self.data_loader = BenchmarkDataLoader(config)
        
        # Initialize evaluation models
        self.similarity_model = None
        self._init_evaluation_models()
    
    def load_dataset(self) -> List[Dict]:
        """Load MS MARCO dataset using centralized data loader."""
        logger.info(f"🔄 Loading MS MARCO dataset using data loader...")
        
        try:
            # FIXED: Proper unlimited sample handling
            if self.is_unlimited or self.sample_size is None:
                max_samples_param = None  # Unlimited
                logger.info(f"   📊 MS MARCO: Loading FULL dataset (unlimited)")
            else:
                max_samples_param = self.sample_size
                logger.info(f"   📊 MS MARCO: Loading LIMITED dataset ({self.sample_size} samples)")
            
            # Use the centralized data loader
            data = self.data_loader.load_benchmark_data(
                benchmark_name="msmarco",
                split="test",
                max_samples=max_samples_param  # FIXED: Use proper parameter
            )
            
            if data and len(data) > 0:
                # Convert to MS MARCO format if needed
                formatted_data = []
                for item in data:
                    formatted_item = {
                        "id": item.get("question_id", item.get("id", f"msmarco_{len(formatted_data)}")),
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "context": item.get("context", ""),
                        "reasoning_type": item.get("reasoning_type", "information_retrieval"),
                        "medical_specialty": item.get("medical_specialty", "general")
                    }
                    formatted_data.append(formatted_item)
                
                logger.info(f"✅ Loaded {len(formatted_data)} MS MARCO questions via data loader")
                return formatted_data
            
        except Exception as e:
            logger.error(f"❌ Data loader failed for MS MARCO: {e}")
        
        # Fallback to minimal synthetic data
        logger.warning("⚠️ Using minimal fallback dataset for MS MARCO")
        return self._generate_minimal_fallback()
    
    def _generate_minimal_fallback(self) -> List[Dict]:
        """Generate minimal fallback dataset if all else fails."""
        return [
            {
                "id": "msmarco_fallback_001",
                "question": "What are the symptoms of diabetes?",
                "answer": "Increased thirst, frequent urination, fatigue, blurred vision",
                "context": "Diabetes mellitus symptoms include polydipsia, polyuria, and fatigue.",
                "reasoning_type": "information_retrieval"
            },
            {
                "id": "msmarco_fallback_002",
                "question": "How is hypertension treated?",
                "answer": "Lifestyle changes, ACE inhibitors, diuretics, calcium channel blockers",
                "context": "Hypertension treatment includes non-pharmacological and pharmacological approaches.",
                "reasoning_type": "information_retrieval"
            }
        ]
    
    def _init_evaluation_models(self):
        """Initialize models for evaluation."""
        try:
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.warning("sentence-transformers not available for similarity scoring")
            self.similarity_model = None
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate a model response against ground truth."""
        expected_answer = question.get("answer", "")
        
        # Basic exact match
        exact_match = response.lower().strip() == expected_answer.lower().strip()
        
        # Semantic similarity if model available
        semantic_score = 0.0
        if self.similarity_model and expected_answer:
            try:
                embeddings = self.similarity_model.encode([response, expected_answer])
                semantic_score = float(np.dot(embeddings[0], embeddings[1]) / 
                                     (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
            except:
                semantic_score = 0.0
        
        # Retrieval relevance scoring
        retrieval_score = self._score_retrieval_relevance(response, retrieved_docs)
        
        # Overall score
        score = (exact_match * 0.3 + semantic_score * 0.4 + retrieval_score * 0.3)
        
        return {
            "score": score,
            "correct": exact_match or semantic_score > 0.8,
            "exact_match": exact_match,
            "semantic_similarity": semantic_score,
            "retrieval_relevance": retrieval_score,
            "reasoning_type": question.get("reasoning_type", "information_retrieval")
        }
    
    def _score_retrieval_relevance(self, response: str, retrieved_docs: List[Dict]) -> float:
        """Score retrieval relevance."""
        if not retrieved_docs:
            return 0.0
        
        response_lower = response.lower()
        relevance_scores = []
        
        for doc in retrieved_docs[:5]:  # Top 5 documents
            doc_text = doc.get("text", "").lower()
            
            # Simple overlap scoring
            response_words = set(response_lower.split())
            doc_words = set(doc_text.split())
            
            if len(response_words) > 0:
                overlap = len(response_words.intersection(doc_words)) / len(response_words)
                relevance_scores.append(overlap)
        
        return np.mean(relevance_scores) if relevance_scores else 0.0