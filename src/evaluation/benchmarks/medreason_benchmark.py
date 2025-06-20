"""
Updated MedReason Benchmark using methods from MedGraphRAG
Place in: src/evaluation/benchmarks/medreason_benchmark.py
"""

import re
from typing import Dict, List, Any
from .base_benchmark import BaseBenchmark

class MedReasonBenchmark(BaseBenchmark):
    """Updated MedReason benchmark using MedGraphRAG evaluation methods."""
    
    def __init__(self, data_path: str, config: Dict[str, Any]):
        super().__init__(data_path, config)
        self.name = "MedReason"
    
    def evaluate_single(self, question: Dict, response: str, retrieved_docs: List[Dict] = None) -> Dict[str, Any]:
        """Evaluate single MedReason response."""
        
        expected = question.get("answer", "")
        
        # Calculate reasoning components
        reasoning_quality = self._assess_reasoning_quality(response)
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
            "expected": expected
        }
    
    def _assess_reasoning_quality(self, response: str) -> float:
        """Assess medical reasoning quality."""
        response_lower = response.lower()
        
        # Reasoning indicators with weights
        reasoning_components = {
            'differential_diagnosis': {
                'patterns': ['differential', 'consider', 'rule out', 'ddx', 'possibilities include'],
                'weight': 0.25
            },
            'clinical_evidence': {
                'patterns': ['evidence', 'supports', 'indicates', 'suggests', 'pattern'],
                'weight': 0.2
            },
            'systematic_approach': {
                'patterns': ['first', 'next', 'then', 'systematic', 'approach', 'step'],
                'weight': 0.2
            },
            'medical_knowledge': {
                'patterns': ['pathophysiology', 'mechanism', 'etiology', 'epidemiology'],
                'weight': 0.15
            },
            'critical_thinking': {
                'patterns': ['however', 'although', 'contrast', 'compare', 'analysis'],
                'weight': 0.2
            }
        }
        
        total_score = 0.0
        for component, info in reasoning_components.items():
            patterns = info['patterns']
            weight = info['weight']
            
            pattern_count = sum(1 for pattern in patterns if pattern in response_lower)
            component_score = min(pattern_count / 2, 1.0)  # Normalize
            total_score += component_score * weight
        
        return min(total_score, 1.0)
    
    def _assess_clinical_accuracy(self, response: str, expected: str) -> float:
        """Assess clinical accuracy using medical entity matching."""
        if not expected:
            return 0.7  # Default score
        
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        # Extract key medical concepts
        response_concepts = self._extract_medical_concepts(response_lower)
        expected_concepts = self._extract_medical_concepts(expected_lower)
        
        if not expected_concepts:
            # Fallback to keyword matching
            return self._calculate_keyword_overlap(response_lower, expected_lower)
        
        # Calculate concept overlap
        overlap = len(response_concepts.intersection(expected_concepts))
        accuracy = overlap / len(expected_concepts) if expected_concepts else 0.7
        
        return min(accuracy, 1.0)
    
    def _extract_medical_concepts(self, text: str) -> set:
        """Extract medical concepts from text."""
        # Common medical terms and concepts
        medical_patterns = [
            r'\b\w*cardio\w*\b', r'\b\w*vascular\w*\b', r'\b\w*diabetes\w*\b',
            r'\b\w*hypertension\w*\b', r'\b\w*syndrome\w*\b', r'\b\w*disease\w*\b',
            r'\b\w*diagnosis\w*\b', r'\b\w*treatment\w*\b', r'\b\w*therapy\w*\b',
            r'\b\w*medication\w*\b', r'\b\w*symptom\w*\b', r'\b\w*condition\w*\b'
        ]
        
        concepts = set()
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.update(match.lower() for match in matches)
        
        # Add specific medical terms
        medical_terms = [
            'metformin', 'insulin', 'glucose', 'chest pain', 'ecg', 'troponin',
            'myocardial infarction', 'acute coronary syndrome', 'angina',
            'heart failure', 'arrhythmia', 'ischemia'
        ]
        
        for term in medical_terms:
            if term in text:
                concepts.add(term)
        
        return concepts
    
    def _calculate_keyword_overlap(self, response: str, expected: str) -> float:
        """Calculate keyword overlap as fallback."""
        response_words = set(re.findall(r'\b\w+\b', response))
        expected_words = set(re.findall(r'\b\w+\b', expected))
        
        # Remove common stop words
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
            'history_taking': ['history', 'patient reports', 'presenting', 'chief complaint'],
            'physical_examination': ['examination', 'physical', 'inspect', 'palpate', 'auscult'],
            'investigation': ['test', 'laboratory', 'imaging', 'ecg', 'blood', 'x-ray'],
            'interpretation': ['results', 'findings', 'shows', 'reveals', 'indicates'],
            'management': ['treatment', 'management', 'therapy', 'medication', 'plan']
        }
        
        step_scores = []
        for step, keywords in process_steps.items():
            step_score = sum(1 for keyword in keywords if keyword in response_lower)
            normalized_score = min(step_score / 2, 1.0)  # Normalize to 0-1
            step_scores.append(normalized_score)
        
        # Average across all steps
        return sum(step_scores) / len(step_scores)
    
    def _assess_knowledge_integration(self, response: str, retrieved_docs: List[Dict]) -> float:
        """Assess integration of retrieved knowledge."""
        if not retrieved_docs:
            return 0.6  # Default if no docs
        
        response_lower = response.lower()
        
        # Check for evidence of knowledge integration
        integration_indicators = [
            'based on', 'according to', 'evidence suggests', 'studies show',
            'research indicates', 'literature supports', 'clinical trials',
            'meta-analysis', 'systematic review'
        ]
        
        integration_count = sum(1 for indicator in integration_indicators if indicator in response_lower)
        
        # Check for specific citations or references
        citation_patterns = [
            r'study', r'trial', r'research', r'analysis', r'findings'
        ]
        
        citation_count = sum(1 for pattern in citation_patterns 
                           if re.search(pattern, response_lower))
        
        # Combine scores
        integration_score = min(integration_count / 2, 1.0)
        citation_score = min(citation_count / 3, 1.0)
        
        return (integration_score * 0.6 + citation_score * 0.4)