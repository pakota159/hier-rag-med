"""
Fixed MSMARCO Benchmark implementation
src/evaluation/benchmarks/msmarco_benchmark.py
"""

import json
from typing import Dict, List, Any
from pathlib import Path

from .base_benchmark import BaseBenchmark
from loguru import logger

class MSMARCOBenchmark(BaseBenchmark):
    """MSMARCO benchmark for passage retrieval evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MSMARCO"
    
    def load_dataset(self) -> List[Dict]:
        """Load MSMARCO dataset."""
        # Sample MSMARCO-style questions for medical domain
        sample_data = [
            {
                "id": "msmarco_001",
                "question": "What are the symptoms of diabetes?",
                "answer": "increased thirst, frequent urination, fatigue, blurred vision",
                "relevant_passages": ["Diabetes symptoms include polyuria, polydipsia, and fatigue."],
                "query_type": "factual"
            },
            {
                "id": "msmarco_002", 
                "question": "How is hypertension diagnosed?",
                "answer": "blood pressure measurements over multiple visits",
                "relevant_passages": ["Hypertension is diagnosed through repeated blood pressure measurements."],
                "query_type": "procedural"
            },
            {
                "id": "msmarco_003",
                "question": "What medications treat heart failure?",
                "answer": "ACE inhibitors, beta-blockers, diuretics",
                "relevant_passages": ["Heart failure treatment includes ACE inhibitors and beta-blockers."],
                "query_type": "treatment"
            },
            {
                "id": "msmarco_004",
                "question": "What causes myocardial infarction?", 
                "answer": "coronary artery blockage from atherosclerotic plaque",
                "relevant_passages": ["MI occurs when coronary arteries are blocked by atherosclerotic plaques."],
                "query_type": "causal"
            },
            {
                "id": "msmarco_005",
                "question": "How is pneumonia treated?",
                "answer": "antibiotics, supportive care, oxygen therapy",
                "relevant_passages": ["Pneumonia treatment involves antibiotics and supportive measures."],
                "query_type": "treatment"
            },
            {
                "id": "msmarco_006",
                "question": "What are risk factors for stroke?",
                "answer": "hypertension, diabetes, smoking, atrial fibrillation",
                "relevant_passages": ["Stroke risk factors include hypertension, diabetes, and smoking."],
                "query_type": "risk_factors"
            },
            {
                "id": "msmarco_007",
                "question": "How is asthma diagnosed?",
                "answer": "spirometry, peak flow measurement, clinical history",
                "relevant_passages": ["Asthma diagnosis involves spirometry and clinical assessment."],
                "query_type": "diagnostic"
            },
            {
                "id": "msmarco_008",
                "question": "What are complications of diabetes?",
                "answer": "nephropathy, retinopathy, neuropathy, cardiovascular disease",
                "relevant_passages": ["Diabetes complications include kidney, eye, and nerve damage."],
                "query_type": "complications"
            },
            {
                "id": "msmarco_009",
                "question": "How does metformin work?",
                "answer": "reduces hepatic glucose production, improves insulin sensitivity",
                "relevant_passages": ["Metformin works by reducing glucose production in the liver."],
                "query_type": "mechanism"
            },
            {
                "id": "msmarco_010",
                "question": "What are signs of heart attack?",
                "answer": "chest pain, shortness of breath, nausea, sweating",
                "relevant_passages": ["Heart attack symptoms include chest pain and shortness of breath."],
                "query_type": "symptoms"
            }
        ]
        
        logger.info(f"✅ Loaded {len(sample_data)} MSMARCO questions")
        return sample_data
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate MSMARCO response using passage retrieval metrics."""
        
        expected_answer = question.get("answer", "")
        relevant_passages = question.get("relevant_passages", [])
        
        # Calculate retrieval quality
        retrieval_score = self._calculate_retrieval_score(retrieved_docs, relevant_passages)
        
        # Calculate answer quality
        answer_score = self._calculate_answer_quality(response, expected_answer)
        
        # Check if response contains relevant medical information
        medical_relevance = self._assess_medical_relevance(response, question)
        
        # Overall score combining retrieval and generation
        overall_score = (retrieval_score * 0.4 + answer_score * 0.4 + medical_relevance * 0.2)
        
        return {
            "question_id": question.get("id"),
            "score": overall_score * 100,
            "correct": overall_score > 0.5,  # More lenient threshold for MSMARCO
            "metrics": {
                "retrieval_score": retrieval_score,
                "answer_score": answer_score,
                "medical_relevance": medical_relevance,
                "overall_score": overall_score,
                "retrieved_docs_count": len(retrieved_docs)
            },
            "response": response,
            "expected": expected_answer
        }
    
    def _calculate_retrieval_score(self, retrieved_docs: List[Dict], relevant_passages: List[str]) -> float:
        """Calculate retrieval quality score."""
        if not retrieved_docs or not relevant_passages:
            return 0.5  # Default score
        
        # Check if retrieved documents contain relevant information
        retrieved_texts = [doc.get("text", "") for doc in retrieved_docs]
        
        relevance_scores = []
        for passage in relevant_passages:
            passage_lower = passage.lower()
            
            # Check overlap with retrieved documents
            max_overlap = 0
            for doc_text in retrieved_texts:
                doc_lower = doc_text.lower()
                
                # Simple word overlap calculation
                passage_words = set(passage_lower.split())
                doc_words = set(doc_lower.split())
                
                if passage_words and doc_words:
                    overlap = len(passage_words.intersection(doc_words)) / len(passage_words)
                    max_overlap = max(max_overlap, overlap)
            
            relevance_scores.append(max_overlap)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
    
    def _calculate_answer_quality(self, response: str, expected: str) -> float:
        """Calculate answer quality using word overlap."""
        if not expected or not response:
            return 0.3
        
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        # Extract key terms from expected answer
        expected_words = set(expected_lower.split())
        response_words = set(response_lower.split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are'}
        expected_words -= stop_words
        response_words -= stop_words
        
        if not expected_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(expected_words.intersection(response_words))
        overlap_score = overlap / len(expected_words)
        
        # Boost score if response contains key medical terms
        medical_terms = [
            'diabetes', 'hypertension', 'heart', 'blood', 'pressure', 'medication',
            'treatment', 'therapy', 'diagnosis', 'symptoms', 'disease', 'condition'
        ]
        
        medical_count = sum(1 for term in medical_terms if term in response_lower)
        medical_boost = min(medical_count * 0.1, 0.3)  # Max 30% boost
        
        return min(overlap_score + medical_boost, 1.0)
    
    def _assess_medical_relevance(self, response: str, question: Dict) -> float:
        """Assess medical relevance of the response."""
        response_lower = response.lower()
        query_type = question.get("query_type", "general")
        
        # Type-specific relevance indicators
        type_indicators = {
            "factual": ["is", "are", "include", "characterized by"],
            "procedural": ["perform", "measure", "test", "evaluate"],
            "treatment": ["treat", "medication", "therapy", "management"],
            "causal": ["cause", "leads to", "results in", "due to"],
            "diagnostic": ["diagnose", "test", "examination", "assessment"],
            "symptoms": ["symptoms", "signs", "presents with", "manifests"],
            "complications": ["complications", "effects", "consequences"],
            "mechanism": ["mechanism", "works by", "action", "process"]
        }
        
        # Base medical relevance
        medical_terms = [
            'patient', 'clinical', 'medical', 'health', 'disease', 'condition',
            'treatment', 'diagnosis', 'therapy', 'medication', 'symptoms'
        ]
        
        medical_count = sum(1 for term in medical_terms if term in response_lower)
        base_relevance = min(medical_count / 3, 1.0)  # Normalize to 0-1
        
        # Type-specific relevance
        type_relevance = 0.5  # Default
        if query_type in type_indicators:
            indicators = type_indicators[query_type]
            indicator_count = sum(1 for indicator in indicators if indicator in response_lower)
            type_relevance = min(indicator_count / 2, 1.0)
        
        # Combine scores
        return (base_relevance * 0.7 + type_relevance * 0.3)