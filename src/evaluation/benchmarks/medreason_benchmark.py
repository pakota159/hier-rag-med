"""
Enhanced MedReason Benchmark with Real Dataset Loading - FIXED VERSION
"""

import re
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Set

from .base_benchmark import BaseBenchmark
from loguru import logger

from src.evaluation.data.data_loader import BenchmarkDataLoader

# Add project root to path for data loader import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

class MedReasonBenchmark(BaseBenchmark):
    """MedReason benchmark with real dataset loading."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MedReason"
        self.data_loader = BenchmarkDataLoader(config)
        
        # Keep the medical patterns for evaluation
        self.medical_patterns = {
            'symptoms': ['pain', 'fever', 'nausea', 'vomiting', 'diarrhea', 'fatigue'],
            'conditions': ['diabetes', 'hypertension', 'heart failure', 'stroke', 'asthma'],
            'treatments': ['medication', 'therapy', 'treatment', 'management', 'intervention'],
            'investigations': ['blood test', 'x-ray', 'ct scan', 'mri', 'ecg', 'examination']
        }
        
        self.reasoning_indicators = {
            'systematic': ['systematic', 'step-by-step', 'approach', 'method', 'process'],
            'differential': ['differential', 'consider', 'rule out', 'exclude', 'possible'],
            'evidence': ['evidence', 'studies', 'research', 'guidelines', 'literature'],
            'causality': ['because', 'due to', 'caused by', 'leads to', 'results in']
        }
    
    def load_dataset(self) -> List[Dict]:
        """Load MedReason dataset using centralized data loader."""
        logger.info(f"🔄 Loading MedReason dataset using data loader...")
        
        try:
            # FIXED: Proper unlimited sample handling
            if self.is_unlimited or self.sample_size is None:
                max_samples_param = None  # Unlimited
                logger.info(f"   📊 MedReason: Loading FULL dataset (unlimited)")
            else:
                max_samples_param = self.sample_size
                logger.info(f"   📊 MedReason: Loading LIMITED dataset ({self.sample_size} samples)")
            
            # Use the centralized data loader
            data = self.data_loader.load_benchmark_data(
                benchmark_name="medreason",
                split="test",
                max_samples=max_samples_param  # FIXED: Use proper parameter
            )
            
            if data and len(data) > 0:
                # Convert to MedReason format if needed
                formatted_data = []
                for item in data:
                    formatted_item = {
                        "id": item.get("question_id", item.get("id", f"medreason_{len(formatted_data)}")),
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "reasoning_type": item.get("reasoning_type", "clinical_reasoning"),
                        "context": item.get("context", ""),
                        "reasoning_chain": item.get("reasoning_chain", [])
                    }
                    formatted_data.append(formatted_item)
                
                logger.info(f"✅ Loaded {len(formatted_data)} MedReason questions via data loader")
                return formatted_data
            
        except Exception as e:
            logger.error(f"❌ Data loader failed for MedReason: {e}")
        
        # Fallback to minimal synthetic data
        logger.warning("⚠️ Using minimal fallback dataset for MedReason")
        return self._generate_minimal_fallback()
    
    def _generate_minimal_fallback(self) -> List[Dict]:
        """Generate minimal fallback dataset if all else fails."""
        return [
            {
                "id": "medreason_fallback_001",
                "question": "A patient presents with chest pain. What is your diagnostic approach?",
                "answer": "systematic evaluation including history, ECG, troponins",
                "reasoning_type": "diagnostic_approach"
            },
            {
                "id": "medreason_fallback_002",
                "question": "How would you manage acute myocardial infarction?",
                "answer": "immediate reperfusion therapy, antiplatelet agents, beta-blockers",
                "reasoning_type": "treatment_planning"
            }
        ]
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate a model response against ground truth."""
        expected_answer = question.get("answer", "")
        reasoning_type = question.get("reasoning_type", "clinical_reasoning")
        
        # Basic exact match
        exact_match = response.lower().strip() == expected_answer.lower().strip()
        
        # Reasoning quality scoring
        reasoning_score = self._score_reasoning_quality(response, reasoning_type)
        
        # Medical pattern matching
        pattern_score = self._score_medical_patterns(response)
        
        # Overall score
        score = (exact_match * 0.3 + reasoning_score * 0.4 + pattern_score * 0.3)
        
        return {
            "score": score,
            "correct": exact_match or reasoning_score > 0.7,
            "exact_match": exact_match,
            "reasoning_quality": reasoning_score,
            "medical_pattern_score": pattern_score,
            "reasoning_type": reasoning_type
        }
    
    def _score_reasoning_quality(self, response: str, reasoning_type: str) -> float:
        """Score the quality of medical reasoning in response."""
        response_lower = response.lower()
        
        # Get relevant indicators for this reasoning type
        indicators = self.reasoning_indicators.get(reasoning_type.split('_')[0], [])
        if not indicators:
            indicators = self.reasoning_indicators['systematic']  # Default
        
        # Count reasoning indicators
        indicator_count = sum(1 for indicator in indicators if indicator in response_lower)
        
        # Score based on presence of reasoning indicators
        reasoning_score = min(indicator_count / len(indicators), 1.0)
        
        return reasoning_score
    
    def _score_medical_patterns(self, response: str) -> float:
        """Score based on medical pattern matching."""
        response_lower = response.lower()
        total_patterns = 0
        found_patterns = 0
        
        for category, patterns in self.medical_patterns.items():
            total_patterns += len(patterns)
            found_patterns += sum(1 for pattern in patterns if pattern in response_lower)
        
        return found_patterns / total_patterns if total_patterns > 0 else 0.0