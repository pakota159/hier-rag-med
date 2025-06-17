"""
MIRAGE benchmark implementation for clinical and research QA evaluation.
"""

import json
import requests
from typing import Dict, List
from pathlib import Path
from loguru import logger

from .base_benchmark import BaseBenchmark


class MIRAGEBenchmark(BaseBenchmark):
    """MIRAGE Medical QA benchmark for clinical and research tasks."""
    
    def __init__(self, config: Dict):
        """Initialize MIRAGE benchmark."""
        super().__init__(config)
        self.dataset_url = config.get("dataset_url", "")
        self.clinical_enabled = config.get("tasks", {}).get("clinical", True)
        self.research_enabled = config.get("tasks", {}).get("research", True)
    
    def load_dataset(self) -> List[Dict]:
        """Load MIRAGE dataset from source."""
        try:
            # Try to load from local cache first
            cache_path = Path("data/evaluation/cache/mirage_dataset.json")
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded MIRAGE dataset from cache: {len(data)} questions")
                return data
            
            # Generate sample MIRAGE-style questions for now
            logger.warning("Using sample MIRAGE data - replace with actual dataset")
            return self._generate_sample_data()
            
        except Exception as e:
            logger.error(f"Failed to load MIRAGE dataset: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> List[Dict]:
        """Generate sample MIRAGE-style questions."""
        clinical_questions = [
            {
                "question": "A 45-year-old patient presents with chest pain and shortness of breath. What is the most likely diagnosis?",
                "answer": "Acute coronary syndrome should be considered given the presentation of chest pain and dyspnea.",
                "type": "clinical",
                "category": "diagnosis"
            },
            {
                "question": "What is the first-line treatment for type 2 diabetes mellitus?",
                "answer": "Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus.",
                "type": "clinical", 
                "category": "treatment"
            },
            {
                "question": "What are the risk factors for developing hypertension?",
                "answer": "Risk factors include age, family history, obesity, high sodium intake, physical inactivity, and excessive alcohol consumption.",
                "type": "clinical",
                "category": "risk_factors"
            }
        ]
        
        research_questions = [
            {
                "question": "What does recent evidence suggest about the efficacy of statins in primary prevention of cardiovascular disease?",
                "answer": "Recent meta-analyses show statins reduce cardiovascular events by 25-30% in primary prevention with acceptable safety profiles.",
                "type": "research",
                "category": "evidence_synthesis"
            },
            {
                "question": "According to current literature, what is the role of aspirin in stroke prevention?",
                "answer": "Low-dose aspirin reduces ischemic stroke risk but increases hemorrhagic stroke risk, requiring careful risk-benefit assessment.",
                "type": "research",
                "category": "literature_review"
            }
        ]
        
        questions = []
        if self.clinical_enabled:
            questions.extend(clinical_questions * 100)  # Replicate for sample size
        if self.research_enabled:
            questions.extend(research_questions * 100)
        
        # Add unique IDs
        for i, q in enumerate(questions):
            q["id"] = f"mirage_{i:04d}"
        
        return questions[:self.sample_size]
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate model response for MIRAGE benchmark."""
        
        # Extract ground truth
        correct_answer = question.get("answer", "")
        question_type = question.get("type", "unknown")
        
        # Calculate semantic similarity (simplified)
        similarity_score = self._calculate_similarity(response, correct_answer)
        
        # Medical accuracy assessment (simplified)
        medical_accuracy = self._assess_medical_accuracy(response, correct_answer)
        
        # Clinical relevance for clinical questions
        clinical_relevance = 1.0
        if question_type == "clinical":
            clinical_relevance = self._assess_clinical_relevance(response, question)
        
        # Research quality for research questions  
        research_quality = 1.0
        if question_type == "research":
            research_quality = self._assess_research_quality(response, retrieved_docs)
        
        # Calculate overall score
        overall_score = (similarity_score * 0.4 + 
                        medical_accuracy * 0.3 + 
                        clinical_relevance * 0.2 + 
                        research_quality * 0.1)
        
        return {
            "question_id": question.get("id"),
            "question_type": question_type,
            "score": overall_score * 100,
            "correct": overall_score > 0.6,
            "metrics": {
                "similarity_score": similarity_score,
                "medical_accuracy": medical_accuracy,
                "clinical_relevance": clinical_relevance,
                "research_quality": research_quality
            },
            "response": response,
            "ground_truth": correct_answer
        }
    
    def _calculate_similarity(self, response: str, reference: str) -> float:
        """Calculate semantic similarity between response and reference."""
        # Simplified similarity calculation
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())
        
        if not reference_words:
            return 0.0
        
        intersection = len(response_words.intersection(reference_words))
        union = len(response_words.union(reference_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _assess_medical_accuracy(self, response: str, reference: str) -> float:
        """Assess medical accuracy of response."""
        # Check for key medical terms
        medical_terms = ["diagnosis", "treatment", "medication", "symptoms", "disease", "condition"]
        
        response_lower = response.lower()
        reference_lower = reference.lower()
        
        # Simple keyword-based assessment
        response_terms = sum(1 for term in medical_terms if term in response_lower)
        reference_terms = sum(1 for term in medical_terms if term in reference_lower)
        
        if reference_terms == 0:
            return 0.8  # Default score if no medical terms in reference
        
        return min(response_terms / reference_terms, 1.0)
    
    def _assess_clinical_relevance(self, response: str, question: Dict) -> float:
        """Assess clinical relevance of response."""
        category = question.get("category", "")
        
        # Category-specific keywords
        category_keywords = {
            "diagnosis": ["diagnosis", "condition", "disease", "syndrome"],
            "treatment": ["treatment", "therapy", "medication", "management"],
            "risk_factors": ["risk", "factors", "causes", "predisposing"]
        }
        
        if category in category_keywords:
            keywords = category_keywords[category]
            response_lower = response.lower()
            keyword_count = sum(1 for keyword in keywords if keyword in response_lower)
            return min(keyword_count / len(keywords), 1.0)
        
        return 0.8  # Default relevance score
    
    def _assess_research_quality(self, response: str, retrieved_docs: List[Dict]) -> float:
        """Assess research quality based on retrieved documents."""
        if not retrieved_docs:
            return 0.5
        
        # Check if response incorporates evidence from retrieved docs
        response_lower = response.lower()
        evidence_indicators = ["study", "research", "evidence", "meta-analysis", "trial"]
        
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in response_lower)
        return min(evidence_count / len(evidence_indicators), 1.0)