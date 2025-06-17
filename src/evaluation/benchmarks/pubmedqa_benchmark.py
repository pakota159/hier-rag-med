"""
PubMedQA benchmark implementation for research literature evaluation.
"""

import json
from typing import Dict, List
from pathlib import Path
from loguru import logger

from .base_benchmark import BaseBenchmark


class PubMedQABenchmark(BaseBenchmark):
    """PubMedQA benchmark for research literature QA evaluation."""
    
    def __init__(self, config: Dict):
        """Initialize PubMedQA benchmark."""
        super().__init__(config)
        self.dataset_url = config.get("dataset_url", "")
        self.split = config.get("split", "test")
    
    def load_dataset(self) -> List[Dict]:
        """Load PubMedQA dataset."""
        try:
            # Try to load from cache
            cache_path = Path("data/evaluation/cache/pubmedqa_dataset.json")
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded PubMedQA dataset from cache: {len(data)} questions")
                return data
            
            # Try to load from Hugging Face datasets
            try:
                from datasets import load_dataset
                dataset = load_dataset("pubmed_qa", "pqa_labeled", split=self.split)
                data = [item for item in dataset]
                logger.info(f"Loaded PubMedQA from Hugging Face: {len(data)} items")
                return data
            except Exception:
                pass
            
            # Generate sample data
            logger.warning("Using sample PubMedQA data - replace with actual dataset")
            return self._generate_sample_data()
            
        except Exception as e:
            logger.error(f"Failed to load PubMedQA dataset: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> List[Dict]:
        """Generate sample PubMedQA-style questions."""
        research_questions = [
            {
                "question": "Do statins reduce cardiovascular mortality in primary prevention?",
                "context": "Multiple randomized controlled trials have evaluated statin therapy for primary prevention of cardiovascular disease. Large meta-analyses suggest significant reduction in cardiovascular events.",
                "answer": "yes",
                "long_answer": "Yes, statins significantly reduce cardiovascular mortality in primary prevention. Meta-analyses show 25-30% reduction in major cardiovascular events.",
                "pmid": "12345678",
                "study_type": "meta_analysis"
            },
            {
                "question": "Is aspirin effective for stroke prevention in atrial fibrillation?",
                "context": "Aspirin has been compared to oral anticoagulants for stroke prevention in atrial fibrillation patients. Studies show variable efficacy compared to warfarin and newer anticoagulants.",
                "answer": "maybe",
                "long_answer": "Aspirin provides some stroke prevention in atrial fibrillation but is less effective than oral anticoagulants.",
                "pmid": "23456789",
                "study_type": "clinical_trial"
            },
            {
                "question": "Does vitamin D supplementation prevent fractures in elderly?",
                "context": "Multiple studies have investigated vitamin D supplementation for fracture prevention. Results vary based on dosing, compliance, and baseline vitamin D levels.",
                "answer": "maybe",
                "long_answer": "Vitamin D supplementation may prevent fractures in elderly, particularly when combined with calcium and in vitamin D deficient populations.",
                "pmid": "34567890",
                "study_type": "systematic_review"
            },
            {
                "question": "Are ACE inhibitors superior to ARBs for heart failure?",
                "context": "Both ACE inhibitors and ARBs are recommended for heart failure treatment. Head-to-head comparisons show similar efficacy with different side effect profiles.",
                "answer": "no",
                "long_answer": "ACE inhibitors are not superior to ARBs for heart failure. Both show similar mortality benefits with comparable efficacy.",
                "pmid": "45678901",
                "study_type": "comparative_study"
            }
        ]
        
        # Replicate for sample size
        questions = research_questions * (self.sample_size // len(research_questions) + 1)
        
        # Add unique IDs
        for i, q in enumerate(questions[:self.sample_size]):
            q["id"] = f"pubmedqa_{i:04d}"
        
        return questions[:self.sample_size]
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate model response for research literature QA."""
        
        # Extract components
        correct_answer = question.get("answer", "").lower()
        long_answer = question.get("long_answer", "")
        study_type = question.get("study_type", "unknown")
        
        # Evaluate answer accuracy (yes/no/maybe)
        answer_accuracy = self._evaluate_answer_accuracy(response, correct_answer)
        
        # Evaluate evidence synthesis
        evidence_score = self._evaluate_evidence_synthesis(response, long_answer)
        
        # Evaluate research quality assessment
        research_quality = self._evaluate_research_quality(response, study_type)
        
        # Evaluate citation and source usage
        citation_score = self._evaluate_citation_usage(response, retrieved_docs)
        
        # Calculate overall score
        overall_score = (answer_accuracy * 0.4 + 
                        evidence_score * 0.3 + 
                        research_quality * 0.2 + 
                        citation_score * 0.1)
        
        return {
            "question_id": question.get("id"),
            "study_type": study_type,
            "score": overall_score * 100,
            "correct": overall_score > 0.6,
            "metrics": {
                "answer_accuracy": answer_accuracy,
                "evidence_synthesis": evidence_score,
                "research_quality": research_quality,
                "citation_usage": citation_score
            },
            "response": response,
            "expected_answer": correct_answer,
            "expected_explanation": long_answer
        }
    
    def _evaluate_answer_accuracy(self, response: str, correct_answer: str) -> float:
        """Evaluate accuracy of yes/no/maybe answer."""
        response_lower = response.lower()
        
        # Extract likely answer from response
        if "yes" in response_lower and "no" not in response_lower:
            predicted = "yes"
        elif "no" in response_lower and "yes" not in response_lower:
            predicted = "no"
        elif "maybe" in response_lower or "possibly" in response_lower or "uncertain" in response_lower:
            predicted = "maybe"
        else:
            # Default based on confidence indicators
            if "definitely" in response_lower or "clearly" in response_lower:
                predicted = "yes"
            elif "not" in response_lower or "unlikely" in response_lower:
                predicted = "no"
            else:
                predicted = "maybe"
        
        return 1.0 if predicted == correct_answer else 0.0
    
    def _evaluate_evidence_synthesis(self, response: str, expected_explanation: str) -> float:
        """Evaluate quality of evidence synthesis."""
        response_lower = response.lower()
        expected_lower = expected_explanation.lower()
        
        # Check for evidence terms
        evidence_terms = ["study", "trial", "research", "evidence", "data", "analysis", "meta-analysis"]
        evidence_count = sum(1 for term in evidence_terms if term in response_lower)
        evidence_score = min(evidence_count / len(evidence_terms), 1.0)
        
        # Check semantic overlap with expected explanation
        response_words = set(response_lower.split())
        expected_words = set(expected_lower.split())
        
        if expected_words:
            semantic_overlap = len(response_words.intersection(expected_words)) / len(expected_words)
        else:
            semantic_overlap = 0.5
        
        return (evidence_score * 0.6 + semantic_overlap * 0.4)
    
    def _evaluate_research_quality(self, response: str, study_type: str) -> float:
        """Evaluate research quality assessment."""
        response_lower = response.lower()
        
        # Check for research quality indicators
        quality_terms = ["randomized", "controlled", "meta-analysis", "systematic", "peer-reviewed", "placebo"]
        quality_count = sum(1 for term in quality_terms if term in response_lower)
        quality_score = min(quality_count / len(quality_terms), 1.0)
        
        # Study type specific terms
        study_specific_terms = {
            "meta_analysis": ["meta-analysis", "pooled", "combined", "systematic"],
            "clinical_trial": ["randomized", "controlled", "trial", "intervention"],
            "systematic_review": ["systematic", "review", "literature", "comprehensive"],
            "comparative_study": ["comparison", "versus", "compared", "control"]
        }
        
        if study_type in study_specific_terms:
            specific_terms = study_specific_terms[study_type]
            specific_count = sum(1 for term in specific_terms if term in response_lower)
            specific_score = min(specific_count / len(specific_terms), 1.0)
        else:
            specific_score = 0.5
        
        return (quality_score * 0.6 + specific_score * 0.4)
    
    def _evaluate_citation_usage(self, response: str, retrieved_docs: List[Dict]) -> float:
        """Evaluate proper citation and source usage."""
        if not retrieved_docs:
            return 0.5
        
        response_lower = response.lower()
        
        # Check for citation indicators
        citation_terms = ["according to", "study shows", "research indicates", "evidence suggests", "based on"]
        citation_count = sum(1 for term in citation_terms if term in response_lower)
        citation_score = min(citation_count / len(citation_terms), 1.0)
        
        # Check if response incorporates retrieved document content
        doc_overlap = 0
        for doc in retrieved_docs[:3]:  # Check top 3 documents
            doc_text = doc.get("text", "").lower()
            doc_words = set(doc_text.split())
            response_words = set(response_lower.split())
            
            if doc_words:
                overlap = len(response_words.intersection(doc_words)) / len(doc_words)
                doc_overlap = max(doc_overlap, overlap)
        
        return (citation_score * 0.4 + doc_overlap * 0.6)