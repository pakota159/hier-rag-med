"""
Complete MedReason Benchmark implementation with debugging and proper metrics calculation
src/evaluation/benchmarks/medreason_benchmark.py
"""

import re
import json
import numpy as np
from typing import Dict, List, Any, Set
from pathlib import Path

from .base_benchmark import BaseBenchmark
from loguru import logger

class MedReasonBenchmark(BaseBenchmark):
    """MedReason benchmark using MedGraphRAG evaluation methods."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MedReason"
        
        # Medical concept patterns for assessment
        self.medical_patterns = {
            'symptoms': [
                'pain', 'fever', 'nausea', 'vomiting', 'diarrhea', 'fatigue', 'weakness',
                'shortness', 'breath', 'chest pain', 'headache', 'dizziness', 'sweating'
            ],
            'conditions': [
                'diabetes', 'hypertension', 'heart failure', 'myocardial', 'stroke', 'asthma',
                'copd', 'pneumonia', 'infection', 'cancer', 'arthritis', 'depression'
            ],
            'treatments': [
                'medication', 'therapy', 'surgery', 'treatment', 'management', 'intervention',
                'antihypertensive', 'antibiotic', 'analgesic', 'insulin', 'metformin'
            ],
            'investigations': [
                'blood test', 'x-ray', 'ct scan', 'mri', 'ecg', 'echo', 'biopsy',
                'laboratory', 'imaging', 'examination', 'assessment'
            ]
        }
        
        # Clinical reasoning indicators
        self.reasoning_indicators = {
            'systematic': ['systematic', 'step-by-step', 'approach', 'method', 'process'],
            'differential': ['differential', 'consider', 'rule out', 'exclude', 'possible'],
            'evidence': ['evidence', 'studies', 'research', 'guidelines', 'literature'],
            'causality': ['because', 'due to', 'caused by', 'leads to', 'results in']
        }
        
    def load_dataset(self) -> List[Dict]:
        """Load MedReason dataset."""
        # Enhanced MedReason-style questions with better coverage
        sample_data = [
            {
                "id": "medreason_001",
                "question": "A 55-year-old patient presents with chest pain. What is your systematic diagnostic approach?",
                "answer": "systematic evaluation including detailed history, physical examination, ECG, cardiac enzymes, chest X-ray",
                "reasoning_type": "diagnostic_approach"
            },
            {
                "id": "medreason_002",
                "question": "How would you manage a patient with acute ST-elevation myocardial infarction?",
                "answer": "immediate reperfusion therapy, dual antiplatelet therapy, beta-blockers, ACE inhibitors, statin therapy",
                "reasoning_type": "treatment_planning"
            },
            {
                "id": "medreason_003", 
                "question": "What are the differential diagnoses for acute shortness of breath in an elderly patient?",
                "answer": "acute heart failure, pneumonia, COPD exacerbation, pulmonary embolism, pneumothorax",
                "reasoning_type": "differential_diagnosis"
            },
            {
                "id": "medreason_004",
                "question": "Explain the pathophysiology of type 2 diabetes mellitus.",
                "answer": "insulin resistance in peripheral tissues leading to compensatory hyperinsulinemia, eventual beta-cell dysfunction and relative insulin deficiency",
                "reasoning_type": "pathophysiology"
            },
            {
                "id": "medreason_005",
                "question": "What investigations would you order for a patient with suspected acute stroke?",
                "answer": "immediate CT brain, blood glucose, ECG, full blood count, coagulation studies, renal function",
                "reasoning_type": "investigation_planning"
            }
        ]
        
        logger.info(f"✅ Loaded {len(sample_data)} MedReason questions")
        return sample_data
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate single MedReason response."""
        
        expected = question.get("answer", "")
        reasoning_type = question.get("reasoning_type", "general")
        question_id = question.get("id", "unknown")
        
        # Debug logging
        logger.debug(f"Evaluating MedReason question {question_id}")
        logger.debug(f"Response length: {len(response) if response else 0}")
        
        # Handle empty or None responses
        if not response or not response.strip():
            logger.warning(f"Empty response for question {question_id}")
            return {
                "question_id": question_id,
                "score": 0.0,
                "correct": False,
                "metrics": {
                    "reasoning_quality": 0.0,
                    "clinical_accuracy": 0.0,
                    "diagnostic_process": 0.0,
                    "knowledge_integration": 0.0,
                    "overall_score": 0.0
                },
                "response": response,
                "expected": expected,
                "reasoning_type": reasoning_type,
                "error": "Empty response"
            }
        
        try:
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
            
            # Ensure score is between 0 and 1
            overall_score = max(0.0, min(1.0, overall_score))
            
            # Debug logging
            logger.debug(f"Question {question_id} scores:")
            logger.debug(f"  Reasoning: {reasoning_quality:.3f}")
            logger.debug(f"  Clinical: {clinical_accuracy:.3f}")
            logger.debug(f"  Diagnostic: {diagnostic_process:.3f}")
            logger.debug(f"  Knowledge: {knowledge_integration:.3f}")
            logger.debug(f"  Overall: {overall_score:.3f}")
            
            result = {
                "question_id": question_id,
                "score": overall_score * 100,  # Convert to percentage
                "correct": overall_score > 0.6,  # Threshold for "correct"
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
            
            logger.debug(f"Final result for {question_id}: score={result['score']:.2f}, correct={result['correct']}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating MedReason response for {question_id}: {e}")
            return {
                "question_id": question_id,
                "score": 0.0,
                "correct": False,
                "metrics": {
                    "reasoning_quality": 0.0,
                    "clinical_accuracy": 0.0,
                    "diagnostic_process": 0.0,
                    "knowledge_integration": 0.0,
                    "overall_score": 0.0
                },
                "response": response,
                "expected": expected,
                "reasoning_type": reasoning_type,
                "error": str(e)
            }
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate MedReason-specific metrics from evaluation results."""
        logger.info(f"Calculating metrics for {len(results)} MedReason results")
        
        if not results:
            logger.warning("No results to calculate metrics for MedReason")
            return {
                "accuracy": 0.0,
                "total_questions": 0,
                "correct_answers": 0,
                "average_score": 0.0,
                "reasoning_quality": 0.0,
                "clinical_accuracy": 0.0,
                "diagnostic_process": 0.0,
                "knowledge_integration": 0.0,
                "benchmark_name": self.name
            }
        
        # Filter out error results
        valid_results = [r for r in results if "error" not in r and r.get("score") is not None]
        
        logger.info(f"Valid results: {len(valid_results)}/{len(results)}")
        
        if not valid_results:
            logger.warning("No valid results for MedReason")
            return {
                "accuracy": 0.0,
                "total_questions": len(results),
                "correct_answers": 0,
                "average_score": 0.0,
                "error_rate": 100.0,
                "benchmark_name": self.name
            }
        
        # Calculate metrics
        total_questions = len(valid_results)
        correct_answers = sum(1 for r in valid_results if r.get("correct", False))
        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0.0
        
        # Calculate component averages
        avg_score = np.mean([r.get("score", 0) for r in valid_results])
        avg_reasoning = np.mean([r.get("metrics", {}).get("reasoning_quality", 0) for r in valid_results])
        avg_clinical = np.mean([r.get("metrics", {}).get("clinical_accuracy", 0) for r in valid_results])
        avg_diagnostic = np.mean([r.get("metrics", {}).get("diagnostic_process", 0) for r in valid_results])
        avg_knowledge = np.mean([r.get("metrics", {}).get("knowledge_integration", 0) for r in valid_results])
        
        # Debug logging
        logger.info(f"MedReason Final Metrics:")
        logger.info(f"  Accuracy: {accuracy:.2f}%")
        logger.info(f"  Correct: {correct_answers}/{total_questions}")
        logger.info(f"  Avg Score: {avg_score:.2f}")
        logger.info(f"  Reasoning: {avg_reasoning:.3f}")
        logger.info(f"  Clinical: {avg_clinical:.3f}")
        
        metrics = {
            "accuracy": accuracy,
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "average_score": avg_score,
            "reasoning_quality": avg_reasoning,
            "clinical_accuracy": avg_clinical,
            "diagnostic_process": avg_diagnostic,
            "knowledge_integration": avg_knowledge,
            "failed_questions": len(results) - len(valid_results),
            "error_rate": ((len(results) - len(valid_results)) / len(results)) * 100 if results else 0.0,
            "benchmark_name": self.name
        }
        
        return metrics
    
    def _assess_reasoning_quality(self, response: str, reasoning_type: str) -> float:
        """Assess medical reasoning quality based on type."""
        if not response:
            return 0.0
            
        response_lower = response.lower()
        total_score = 0.0
        
        # Base reasoning indicators
        base_score = 0.0
        for category, indicators in self.reasoning_indicators.items():
            indicator_count = sum(1 for indicator in indicators if indicator in response_lower)
            category_score = min(indicator_count / 2.0, 1.0)
            base_score += category_score
        
        base_score = base_score / len(self.reasoning_indicators)
        total_score += base_score * 0.6
        
        # Type-specific indicators
        type_indicators = {
            'diagnostic_approach': ['history', 'examination', 'investigation', 'diagnosis', 'assessment'],
            'treatment_planning': ['treatment', 'management', 'therapy', 'medication', 'intervention'],
            'differential_diagnosis': ['differential', 'consider', 'rule out', 'possibilities', 'exclude'],
            'pathophysiology': ['mechanism', 'pathway', 'process', 'leads to', 'causes'],
            'investigation_planning': ['test', 'investigation', 'imaging', 'laboratory', 'blood']
        }
        
        if reasoning_type in type_indicators:
            indicators = type_indicators[reasoning_type]
            indicator_count = sum(1 for indicator in indicators if indicator in response_lower)
            type_score = min(indicator_count / 3.0, 1.0)
            total_score += type_score * 0.4
        else:
            total_score += 0.3  # Default bonus for unknown types
        
        return min(total_score, 1.0)
    
    def _assess_clinical_accuracy(self, response: str, expected: str) -> float:
        """Assess clinical accuracy using medical concept matching."""
        if not response:
            return 0.0
            
        if not expected:
            return 0.5  # Default score when no reference available
        
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        # Extract medical concepts
        response_concepts = self._extract_medical_concepts(response_lower)
        expected_concepts = self._extract_medical_concepts(expected_lower)
        
        if not expected_concepts:
            # Fallback to word overlap
            return self._calculate_word_overlap(response_lower, expected_lower)
        
        # Calculate concept overlap using Jaccard similarity
        if not response_concepts:
            return 0.1  # Some credit for effort
            
        intersection = len(response_concepts.intersection(expected_concepts))
        union = len(response_concepts.union(expected_concepts))
        
        if union == 0:
            return 0.5
            
        jaccard_similarity = intersection / union
        
        # Also consider word-level overlap as fallback
        word_overlap = self._calculate_word_overlap(response_lower, expected_lower)
        
        # Combine both scores
        combined_score = (jaccard_similarity * 0.7) + (word_overlap * 0.3)
        
        return min(combined_score, 1.0)
    
    def _extract_medical_concepts(self, text: str) -> Set[str]:
        """Extract medical concepts from text."""
        concepts = set()
        
        # Extract concepts from predefined patterns
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    concepts.add(pattern)
        
        # Extract potential medical terms (words ending in common medical suffixes)
        medical_suffixes = ['osis', 'itis', 'emia', 'uria', 'pathy', 'ology', 'gram', 'scopy']
        words = text.split()
        
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word)
            if len(word_clean) > 4:
                if any(word_clean.endswith(suffix) for suffix in medical_suffixes):
                    concepts.add(word_clean)
        
        return concepts
    
    def _calculate_word_overlap(self, response: str, expected: str) -> float:
        """Calculate simple word overlap as fallback measure."""
        if not response or not expected:
            return 0.0
            
        response_words = set(response.split())
        expected_words = set(expected.split())
        
        if not expected_words:
            return 0.5
            
        if not response_words:
            return 0.0
            
        intersection = len(response_words.intersection(expected_words))
        return intersection / len(expected_words)
    
    def _assess_diagnostic_process(self, response: str) -> float:
        """Assess diagnostic reasoning process quality."""
        if not response:
            return 0.0
            
        response_lower = response.lower()
        
        # Check for systematic diagnostic approach
        diagnostic_steps = [
            'history', 'examination', 'investigation', 'differential', 'diagnosis'
        ]
        
        step_score = sum(1 for step in diagnostic_steps if step in response_lower)
        step_score = min(step_score / len(diagnostic_steps), 1.0)
        
        # Check for clinical reasoning structure
        structure_indicators = [
            'first', 'initially', 'then', 'next', 'finally', 'because', 'therefore'
        ]
        
        structure_score = sum(1 for indicator in structure_indicators if indicator in response_lower)
        structure_score = min(structure_score / 3.0, 1.0)
        
        # Combine scores
        total_score = (step_score * 0.6) + (structure_score * 0.4)
        
        return min(total_score, 1.0)
    
    def _assess_knowledge_integration(self, response: str, retrieved_docs: List[Dict]) -> float:
        """Assess how well the response integrates retrieved knowledge."""
        if not response:
            return 0.0
            
        # Base score for having a response
        base_score = 0.3
        
        # If no retrieved docs, return base score
        if not retrieved_docs:
            return base_score
        
        response_lower = response.lower()
        
        # Check if response mentions concepts from retrieved documents
        integration_score = 0.0
        total_docs = len(retrieved_docs)
        
        for doc in retrieved_docs[:5]:  # Check top 5 documents
            doc_content = doc.get('content', '') or doc.get('text', '')
            if doc_content:
                doc_lower = doc_content.lower()
                
                # Find common words (excluding stop words)
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
                
                response_words = set(response_lower.split()) - stop_words
                doc_words = set(doc_lower.split()) - stop_words
                
                if response_words and doc_words:
                    overlap = len(response_words.intersection(doc_words))
                    doc_integration = overlap / len(response_words)
                    integration_score += doc_integration
        
        if total_docs > 0:
            integration_score = integration_score / total_docs
        
        # Combine base score with integration score
        final_score = base_score + (integration_score * 0.7)
        
        return min(final_score, 1.0)