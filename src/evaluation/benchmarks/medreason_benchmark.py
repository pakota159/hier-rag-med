"""
Complete MedReason Benchmark implementation with all required methods
src/evaluation/benchmarks/medreason_benchmark.py
"""

import re
import json
from typing import Dict, List, Any
from pathlib import Path

from .base_benchmark import BaseBenchmark
from loguru import logger

class MedReasonBenchmark(BaseBenchmark):
    """MedReason benchmark using MedGraphRAG evaluation methods."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MedReason"
    
    def load_dataset(self) -> List[Dict]:
        """Load MedReason dataset."""
        # Sample MedReason-style questions
        sample_data = [
            {
                "id": "medreason_001",
                "question": "A patient presents with chest pain. What is your diagnostic approach?",
                "answer": "systematic evaluation including history, ECG, troponins",
                "reasoning_type": "diagnostic_approach"
            },
            {
                "id": "medreason_002",
                "question": "How would you manage a patient with acute myocardial infarction?",
                "answer": "immediate reperfusion therapy, antiplatelet agents, beta-blockers, ACE inhibitors",
                "reasoning_type": "treatment_planning"
            },
            {
                "id": "medreason_003", 
                "question": "What are the differential diagnoses for shortness of breath?",
                "answer": "heart failure, asthma, COPD, pneumonia, pulmonary embolism",
                "reasoning_type": "differential_diagnosis"
            },
            {
                "id": "medreason_004",
                "question": "Explain the pathophysiology of type 2 diabetes mellitus.",
                "answer": "insulin resistance leading to beta-cell dysfunction and hyperglycemia",
                "reasoning_type": "pathophysiology"
            },
            {
                "id": "medreason_005",
                "question": "What investigations would you order for suspected stroke?",
                "answer": "CT brain, blood glucose, ECG, full blood count, coagulation studies",
                "reasoning_type": "investigation_planning"
            }
        ]
        
        logger.info(f"✅ Loaded {len(sample_data)} MedReason questions")
        return sample_data
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate single MedReason response."""
        
        expected = question.get("answer", "")
        reasoning_type = question.get("reasoning_type", "general")
        
        # Calculate reasoning components
        reasoning_quality = self._assess_reasoning_quality(response, reasoning_type)
        clinical_accuracy = self._assess_clinical_accuracy(response, expected)
        diagnostic_process = self._assess_diagnostic_process(response)
        knowledge_integration = self._assess_knowledge_integration(response, retrieved_docs)
        
        # Overall score (weighted combination)
        overall_score = (
            reasoning_quality * 0.4 +
            clinical_accuracy * 0.3 +
            diagnostic_process * 0.2 +
            knowledge_integration * 0.1
        )
        
        return {
            "question_id": question.get("id"),
            "score": overall_score * 100,
            "correct": overall_score > 0.6,
            "metrics": {
                "reasoning_quality": reasoning_quality,
                "clinical_accuracy": clinical_accuracy, 
                "diagnostic_process": diagnostic_process,
                "knowledge_integration": knowledge_integration,
                "overall_score": overall_score
            },
            "response": response,
            "expected": expected,
            "reasoning_type": reasoning_type
        }
    
    def _assess_reasoning_quality(self, response: str, reasoning_type: str) -> float:
        """Assess medical reasoning quality based on type."""
        response_lower = response.lower()
        
        # Base reasoning indicators
        base_indicators = {
            'systematic_approach': ['systematic', 'approach', 'step', 'first', 'next', 'then'],
            'clinical_thinking': ['consider', 'rule out', 'differential', 'likely', 'unlikely'],
            'evidence_based': ['evidence', 'studies', 'guidelines', 'literature', 'research'],
            'medical_knowledge': ['pathophysiology', 'mechanism', 'etiology', 'epidemiology']
        }
        
        # Type-specific indicators
        type_indicators = {
            'diagnostic_approach': ['history', 'examination', 'investigation', 'diagnosis'],
            'treatment_planning': ['treatment', 'management', 'therapy', 'medication'],
            'differential_diagnosis': ['differential', 'consider', 'rule out', 'possibilities'],
            'pathophysiology': ['mechanism', 'pathway', 'process', 'leads to'],
            'investigation_planning': ['test', 'investigation', 'imaging', 'laboratory']
        }
        
        total_score = 0.0
        
        # Score base reasoning
        for category, indicators in base_indicators.items():
            indicator_count = sum(1 for indicator in indicators if indicator in response_lower)
            category_score = min(indicator_count / 2, 1.0)
            total_score += category_score * 0.6  # 60% weight to base reasoning
        
        # Score type-specific reasoning
        if reasoning_type in type_indicators:
            indicators = type_indicators[reasoning_type]
            indicator_count = sum(1 for indicator in indicators if indicator in response_lower)
            type_score = min(indicator_count / 2, 1.0)
            total_score += type_score * 0.4  # 40% weight to type-specific
        
        return min(total_score / len(base_indicators), 1.0)
    
    def _assess_clinical_accuracy(self, response: str, expected: str) -> float:
        """Assess clinical accuracy using medical concept matching."""
        if not expected:
            return 0.7  # Default score
        
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        # Extract medical concepts
        response_concepts = self._extract_medical_concepts(response_lower)
        expected_concepts = self._extract_medical_concepts(expected_lower)
        
        if not expected_concepts:
            # Fallback to word overlap
            return self._calculate_word_overlap(response_lower, expected_lower)
        
        # Calculate concept overlap
        overlap = len(response_concepts.intersection(expected_concepts))
        accuracy = overlap / len(expected_concepts) if expected_concepts else 0.7
        
        return min(accuracy, 1.0)
    
    def _extract_medical_concepts(self, text: str) -> set:
        """Extract medical concepts from text."""
        # Medical patterns
        medical_patterns = [
            r'\b\w*cardio\w*\b', r'\b\w*vascular\w*\b', r'\b\w*diabetes\w*\b',
            r'\b\w*hypertension\w*\b', r'\b\w*syndrome\w*\b', r'\b\w*disease\w*\b',
            r'\b\w*therapy\w*\b', r'\b\w*treatment\w*\b', r'\b\w*medication\w*\b'
        ]
        
        concepts = set()
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.update(match.lower() for match in matches)
        
        # Specific medical terms
        medical_terms = [
            'ecg', 'troponin', 'chest pain', 'myocardial infarction', 'heart failure',
            'insulin', 'glucose', 'metformin', 'beta-blocker', 'ace inhibitor',
            'pneumonia', 'asthma', 'copd', 'stroke', 'ct scan'
        ]
        
        for term in medical_terms:
            if term in text:
                concepts.add(term)
        
        return concepts
    
    def _calculate_word_overlap(self, response: str, expected: str) -> float:
        """Calculate word overlap as fallback."""
        response_words = set(re.findall(r'\b\w+\b', response))
        expected_words = set(re.findall(r'\b\w+\b', expected))
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        response_words -= stop_words
        expected_words -= stop_words
        
        if not expected_words:
            return 0.7
        
        overlap = len(response_words.intersection(expected_words))
        return overlap / len(expected_words)
    
    def _assess_diagnostic_process(self, response: str) -> float:
        """Assess diagnostic process structure."""
        response_lower = response.lower()
        
        # Diagnostic process components
        process_steps = {
            'history_taking': ['history', 'patient reports', 'presenting', 'onset', 'duration'],
            'physical_examination': ['examination', 'physical', 'vital signs', 'inspection'],
            'investigation': ['test', 'laboratory', 'imaging', 'ecg', 'blood', 'x-ray'],
            'interpretation': ['results', 'findings', 'shows', 'reveals', 'indicates'],
            'management': ['treatment', 'management', 'therapy', 'medication', 'plan']
        }
        
        step_scores = []
        for step, keywords in process_steps.items():
            step_score = sum(1 for keyword in keywords if keyword in response_lower)
            normalized_score = min(step_score / 2, 1.0)
            step_scores.append(normalized_score)
        
        return sum(step_scores) / len(step_scores)
    
    def _assess_knowledge_integration(self, response: str, retrieved_docs: List[Dict]) -> float:
        """Assess integration of retrieved knowledge."""
        if not retrieved_docs:
            return 0.6  # Default if no docs
        
        response_lower = response.lower()
        
        # Knowledge integration indicators
        integration_indicators = [
            'based on', 'according to', 'evidence suggests', 'studies show',
            'research indicates', 'literature supports', 'guidelines recommend'
        ]
        
        integration_count = sum(1 for indicator in integration_indicators if indicator in response_lower)
        
        # Citation patterns
        citation_patterns = ['study', 'trial', 'research', 'analysis', 'findings']
        citation_count = sum(1 for pattern in citation_patterns if pattern in response_lower)
        
        # Combine scores
        integration_score = min(integration_count / 2, 1.0)
        citation_score = min(citation_count / 3, 1.0)
        
        return (integration_score * 0.6 + citation_score * 0.4)