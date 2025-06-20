"""
Enhanced MIRAGE Benchmark with Real Dataset Loading - FIXED VERSION
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

class MIRAGEBenchmark(BaseBenchmark):
    """MIRAGE benchmark with real dataset loading."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MIRAGE"
        self.data_loader = BenchmarkDataLoader(config)
        
        # Initialize semantic similarity model
        self.similarity_model = None
        self._init_evaluation_models()
    
    def load_dataset(self) -> List[Dict]:
        """Load MIRAGE dataset using centralized data loader."""
        logger.info(f"🔄 Loading MIRAGE dataset using data loader...")
        
        try:
            # FIXED: Proper unlimited sample handling
            if self.is_unlimited or self.sample_size is None:
                max_samples_param = None  # Unlimited
                logger.info(f"   📊 MIRAGE: Loading FULL dataset (unlimited)")
            else:
                max_samples_param = self.sample_size
                logger.info(f"   📊 MIRAGE: Loading LIMITED dataset ({self.sample_size} samples)")
            
            # Use the centralized data loader
            data = self.data_loader.load_benchmark_data(
                benchmark_name="mirage",
                split="test",
                max_samples=max_samples_param  # FIXED: Use proper parameter
            )
            
            if data and len(data) > 0:
                # Convert to MIRAGE format if needed
                formatted_data = []
                for item in data:
                    formatted_item = {
                        "id": item.get("question_id", item.get("id", f"mirage_{len(formatted_data)}")),
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "type": item.get("question_type", item.get("type", "clinical")),
                        "category": item.get("medical_specialty", item.get("category", "general")),
                        "context": item.get("context", ""),
                        "options": item.get("options", [])
                    }
                    formatted_data.append(formatted_item)
                
                logger.info(f"✅ Loaded {len(formatted_data)} MIRAGE questions via data loader")
                return formatted_data
            
        except Exception as e:
            logger.error(f"❌ Data loader failed for MIRAGE: {e}")
        
        # Fallback to minimal synthetic data if data loader fails completely
        logger.warning("⚠️ Using minimal fallback dataset for MIRAGE")
        return self._generate_minimal_fallback()
    
    def _generate_minimal_fallback(self) -> List[Dict]:
        """Generate minimal fallback dataset if all else fails."""
        return [
            {
                "id": "mirage_fallback_001",
                "question": "What is the first-line treatment for type 2 diabetes?",
                "answer": "Metformin",
                "type": "clinical",
                "category": "endocrinology"
            },
            {
                "id": "mirage_fallback_002", 
                "question": "What are the signs of myocardial infarction?",
                "answer": "Chest pain, shortness of breath, nausea",
                "type": "clinical",
                "category": "cardiology"
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
        
        # Clinical relevance scoring
        clinical_score = self._score_clinical_relevance(response, question)
        
        # Overall score
        score = (exact_match * 0.4 + semantic_score * 0.4 + clinical_score * 0.2)
        
        return {
            "score": score,
            "correct": exact_match or semantic_score > 0.8,
            "exact_match": exact_match,
            "semantic_similarity": semantic_score,
            "clinical_relevance": clinical_score,
            "question_type": question.get("type", "clinical")
        }
    
    def _score_clinical_relevance(self, response: str, question: Dict) -> float:
        """Score clinical relevance of response."""
        # Simple keyword-based scoring
        medical_terms = ["treatment", "diagnosis", "symptom", "medication", "therapy", 
                        "clinical", "patient", "medical", "condition", "disease"]
        
        response_lower = response.lower()
        term_count = sum(1 for term in medical_terms if term in response_lower)
        
        return min(term_count / 5.0, 1.0)  # Normalize to 0-1