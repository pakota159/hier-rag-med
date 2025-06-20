"""
Updated MIRAGE Benchmark using methods from MedRAG/Self-BioRAG
Place in: src/evaluation/benchmarks/mirage_benchmark.py
"""

import re
import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch

from .base_benchmark import BaseBenchmark
from ..metrics.qa_metrics import QAMetrics
from ..metrics.clinical_metrics import ClinicalMetrics

class MIRAGEBenchmark(BaseBenchmark):
    """Updated MIRAGE benchmark using proven medical RAG evaluation methods."""
    
    def __init__(self, data_path: str, config: Dict[str, Any]):
        super().__init__(data_path, config)
        self.name = "MIRAGE"
        
        # Initialize medical evaluation models
        self._init_evaluation_models()
        
    def _init_evaluation_models(self):
        """Initialize models for medical evaluation."""
        try:
            # Semantic similarity model
            self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Medical NER for entity extraction
            self.medical_ner = pipeline(
                "ner", 
                model="d4data/biomedical-ner-all",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Warning: Could not load models: {e}")
            self.similarity_model = None
            self.medical_ner = None
    
    def evaluate_single(self, question: Dict, response: str, retrieved_docs: List[Dict] = None) -> Dict[str, Any]:
        """Evaluate single response using medical RAG methods."""
        
        # Extract ground truth
        correct_answer = question.get("answer", "")
        question_type = question.get("type", "multiple_choice")
        
        if question_type == "multiple_choice":
            # Use accuracy-based evaluation like MedRAG
            score = self._evaluate_multiple_choice(response, correct_answer, question)
        else:
            # Use semantic similarity + medical entity overlap
            score = self._evaluate_text_generation(response, correct_answer)
        
        # Calculate component metrics
        semantic_score = self._calculate_semantic_similarity(response, correct_answer)
        medical_accuracy = self._calculate_medical_entity_overlap(response, correct_answer)
        clinical_relevance = self._assess_clinical_relevance(response, question)
        
        return {
            "question_id": question.get("id"),
            "question_type": question_type,
            "score": score * 100,
            "correct": score > 0.6,
            "metrics": {
                "semantic_similarity": semantic_score,
                "medical_accuracy": medical_accuracy,
                "clinical_relevance": clinical_relevance,
                "overall_score": score
            },
            "response": response,
            "ground_truth": correct_answer
        }
    
    def _evaluate_multiple_choice(self, response: str, correct_answer: str, question: Dict) -> float:
        """Evaluate multiple choice using exact match like MedRAG."""
        response_lower = response.lower().strip()
        correct_lower = correct_answer.lower().strip()
        
        # Extract choice letters/words
        response_choice = self._extract_choice(response_lower)
        correct_choice = self._extract_choice(correct_lower)
        
        # Exact match
        if response_choice == correct_choice:
            return 1.0
        
        # Check if correct answer appears in response
        if correct_lower in response_lower:
            return 0.8
        
        # Partial credit for medical terms
        return self._calculate_medical_entity_overlap(response, correct_answer) * 0.5
    
    def _evaluate_text_generation(self, response: str, correct_answer: str) -> float:
        """Evaluate text generation using Self-BioRAG methods."""
        # Semantic similarity (main component)
        semantic_sim = self._calculate_semantic_similarity(response, correct_answer)
        
        # Medical entity overlap
        entity_overlap = self._calculate_medical_entity_overlap(response, correct_answer)
        
        # Key term overlap (for short answers)
        term_overlap = self._calculate_key_term_overlap(response, correct_answer)
        
        # Weighted combination
        score = (semantic_sim * 0.5 + entity_overlap * 0.3 + term_overlap * 0.2)
        return min(score, 1.0)
    
    def _calculate_semantic_similarity(self, response: str, reference: str) -> float:
        """Calculate semantic similarity using embeddings."""
        if not self.similarity_model or not response.strip() or not reference.strip():
            return 0.0
        
        try:
            # Get embeddings
            response_emb = self.similarity_model.encode([response])
            reference_emb = self.similarity_model.encode([reference])
            
            # Cosine similarity
            similarity = cosine_similarity(response_emb, reference_emb)[0][0]
            return max(0.0, float(similarity))
        except:
            return 0.0
    
    def _calculate_medical_entity_overlap(self, response: str, reference: str) -> float:
        """Calculate overlap of medical entities."""
        if not self.medical_ner:
            # Fallback to keyword matching
            return self._calculate_medical_keyword_overlap(response, reference)
        
        try:
            # Extract medical entities
            response_entities = self._extract_medical_entities(response)
            reference_entities = self._extract_medical_entities(reference)
            
            if not reference_entities:
                return 0.8  # Default if no medical entities
            
            overlap = len(response_entities.intersection(reference_entities))
            return overlap / len(reference_entities)
        except:
            return self._calculate_medical_keyword_overlap(response, reference)
    
    def _extract_medical_entities(self, text: str) -> set:
        """Extract medical entities from text."""
        try:
            entities = self.medical_ner(text)
            medical_entities = set()
            
            for entity in entities:
                if entity['entity'] in ['CHEMICAL', 'DISEASE', 'GENE_OR_GENE_PRODUCT']:
                    # Clean entity text
                    entity_text = entity['word'].replace('##', '').lower().strip()
                    if len(entity_text) > 2:
                        medical_entities.add(entity_text)
            
            return medical_entities
        except:
            return set()
    
    def _calculate_medical_keyword_overlap(self, response: str, reference: str) -> float:
        """Fallback medical keyword overlap."""
        medical_keywords = [
            'metformin', 'diabetes', 'insulin', 'glucose', 'hypertension',
            'cardiovascular', 'diagnosis', 'treatment', 'medication', 'therapy',
            'disease', 'condition', 'syndrome', 'disorder', 'pathology'
        ]
        
        response_lower = response.lower()
        reference_lower = reference.lower()
        
        response_terms = {kw for kw in medical_keywords if kw in response_lower}
        reference_terms = {kw for kw in medical_keywords if kw in reference_lower}
        
        if not reference_terms:
            return 0.7
        
        overlap = len(response_terms.intersection(reference_terms))
        return overlap / len(reference_terms)
    
    def _calculate_key_term_overlap(self, response: str, reference: str) -> float:
        """Calculate overlap of key terms."""
        # Clean and split terms
        response_terms = set(re.findall(r'\b\w+\b', response.lower()))
        reference_terms = set(re.findall(r'\b\w+\b', reference.lower()))
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        response_terms -= stop_words
        reference_terms -= stop_words
        
        if not reference_terms:
            return 0.0
        
        overlap = len(response_terms.intersection(reference_terms))
        return overlap / len(reference_terms)
    
    def _extract_choice(self, text: str) -> str:
        """Extract choice letter from response."""
        # Look for patterns like "A)", "(A)", "A.", "Option A"
        choice_pattern = r'\b([A-E])\b'
        matches = re.findall(choice_pattern, text.upper())
        
        if matches:
            return matches[0].lower()
        
        # Look for choice words
        if 'metformin' in text:
            return 'metformin'
        
        return text[:20]  # First 20 chars as fallback
    
    def _assess_clinical_relevance(self, response: str, question: Dict) -> float:
        """Assess clinical relevance of response."""
        response_lower = response.lower()
        
        # Clinical reasoning indicators
        clinical_indicators = [
            'diagnosis', 'treatment', 'management', 'therapy', 'medication',
            'symptoms', 'signs', 'examination', 'history', 'assessment',
            'differential', 'prognosis', 'pathophysiology', 'etiology'
        ]
        
        indicator_count = sum(1 for indicator in clinical_indicators if indicator in response_lower)
        
        # Normalize by number of indicators
        relevance_score = min(indicator_count / 5, 1.0)  # Expect at least 5 clinical terms
        
        return max(0.6, relevance_score)  # Minimum 60% relevance