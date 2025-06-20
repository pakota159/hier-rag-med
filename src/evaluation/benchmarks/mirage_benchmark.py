# src/evaluation/benchmarks/mirage_benchmark.py
"""
Updated MIRAGE Benchmark with Official QADataset Integration
Uses src.utils.QADataset from the official MIRAGE repository
"""

import re
import json
import sys
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Any, Set, Optional

from .base_benchmark import BaseBenchmark
from loguru import logger

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import official MIRAGE QADataset utility
try:
    from mirage.src.utils import QADataset
    MIRAGE_UTILS_AVAILABLE = True
    logger.info("✅ Successfully imported QADataset from src.utils")
except ImportError:
    MIRAGE_UTILS_AVAILABLE = False
    logger.warning("⚠️ QADataset from src.utils not available - using fallback data loader")

from src.evaluation.data.data_loader import BenchmarkDataLoader


class MIRAGEBenchmark(BaseBenchmark):
    """
    MIRAGE benchmark with official QADataset integration.
    Now uses the official src.utils.QADataset from MIRAGE repository.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MIRAGE"
        self.data_loader = BenchmarkDataLoader(config)
        
        # Initialize evaluation models
        self.similarity_model = None
        self._init_evaluation_models()
        
        # MIRAGE-specific configuration
        self.use_official_dataset = config.get("use_official_dataset", True)
        self.dataset_source = config.get("dataset_source", "https://github.com/Teddy-XiongGZ/MIRAGE")
        
        # Medical patterns for enhanced evaluation
        self.medical_patterns = {
            'symptoms': ['pain', 'fever', 'nausea', 'vomiting', 'diarrhea', 'fatigue', 'headache', 'shortness of breath'],
            'conditions': ['diabetes', 'hypertension', 'heart failure', 'stroke', 'asthma', 'cancer', 'pneumonia'],
            'treatments': ['medication', 'therapy', 'treatment', 'management', 'intervention', 'surgery', 'procedure'],
            'investigations': ['blood test', 'x-ray', 'ct scan', 'mri', 'ecg', 'examination', 'biopsy', 'ultrasound'],
            'anatomy': ['heart', 'lung', 'liver', 'kidney', 'brain', 'stomach', 'intestine', 'blood vessel']
        }
        
        self.reasoning_indicators = {
            'systematic': ['systematic', 'step-by-step', 'approach', 'method', 'process', 'algorithm'],
            'differential': ['differential', 'consider', 'rule out', 'exclude', 'possible', 'likely', 'unlikely'],
            'evidence': ['evidence', 'studies', 'research', 'guidelines', 'literature', 'recommendation'],
            'causality': ['because', 'due to', 'caused by', 'leads to', 'results in', 'associated with']
        }
    
    def load_dataset(self) -> List[Dict]:
        """Load official MIRAGE dataset using QADataset utility."""
        logger.info(f"🔄 Loading official MIRAGE benchmark...")
        
        try:
            # Try to use official QADataset first
            if MIRAGE_UTILS_AVAILABLE and self.use_official_dataset:
                return self._load_with_official_qadataset()
            else:
                # Fallback to centralized data loader
                return self._load_with_fallback_loader()
                
        except Exception as e:
            logger.error(f"❌ Failed to load MIRAGE data: {e}")
            raise e
    
    def _load_with_official_qadataset(self) -> List[Dict]:
        """Load data using the official QADataset from src.utils."""
        logger.info("📚 Using official MIRAGE QADataset...")
        
        try:
            # Initialize QADataset with MIRAGE configuration
            qa_dataset = QADataset(
                dataset_name="mirage",
                split="test" if not self.sample_size else "train",
                cache_dir=self.data_loader.cache_dir
            )
            
            # Load the dataset
            raw_data = qa_dataset.load_data()
            
            # Process data into our format
            processed_data = []
            for idx, item in enumerate(raw_data):
                processed_item = self._process_mirage_item(item, idx)
                processed_data.append(processed_item)
            
            # Apply sample limit if specified
            if self.sample_size and not self.is_unlimited:
                processed_data = processed_data[:self.sample_size]
                logger.info(f"   ✂️ Limited to {self.sample_size} samples")
            
            logger.info(f"✅ Loaded {len(processed_data)} MIRAGE questions using official QADataset")
            return processed_data
            
        except Exception as e:
            logger.warning(f"⚠️ Official QADataset failed: {e}. Falling back to manual loader.")
            return self._load_with_fallback_loader()
    
    def _load_with_fallback_loader(self) -> List[Dict]:
        """Fallback to centralized data loader."""
        logger.info("📚 Using fallback MIRAGE data loader...")
        
        max_samples = self.sample_size if not self.is_unlimited else None
        data = self.data_loader.load_benchmark_data("mirage", max_samples=max_samples)
        
        if data and len(data) > 0:
            logger.info(f"✅ Loaded {len(data)} MIRAGE questions from fallback loader")
            return data
        else:
            raise ConnectionError("Failed to load MIRAGE benchmark data from both official and fallback sources.")
    
    def _process_mirage_item(self, item: Dict, idx: int) -> Dict:
        """Process a single MIRAGE item into standardized format."""
        return {
            "question_id": item.get("id", f"mirage_{idx}"),
            "question": item.get("question", ""),
            "context": item.get("context", ""),
            "answer": item.get("answer", ""),
            "explanation": item.get("explanation", ""),
            "options": item.get("options", []),
            "question_type": item.get("question_type", self._classify_question_type(item.get("question", ""))),
            "medical_specialty": item.get("medical_specialty", self._classify_medical_specialty(item.get("question", ""))),
            "reasoning_type": item.get("reasoning_type", self._classify_reasoning_type(item.get("question", ""))),
            "difficulty": item.get("difficulty", "medium"),
            "benchmark": "mirage",
            
            # MIRAGE-specific fields
            "clinical_scenario": item.get("clinical_scenario", ""),
            "patient_info": item.get("patient_info", {}),
            "diagnostic_category": item.get("diagnostic_category", ""),
            "evidence_level": item.get("evidence_level", ""),
        }
    
    def _init_evaluation_models(self):
        """Initialize models for evaluation metrics."""
        try:
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Initialized similarity model for evaluation")
        except ImportError:
            logger.warning("⚠️ SentenceTransformers not available - some metrics will be limited")
            self.similarity_model = None
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of medical question."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['which', 'what is the most', 'best', 'choose']):
            return "multiple_choice"
        elif any(word in question_lower for word in ['diagnose', 'diagnosis', 'likely cause']):
            return "diagnostic"
        elif any(word in question_lower for word in ['treatment', 'therapy', 'management']):
            return "therapeutic"
        elif any(word in question_lower for word in ['prognosis', 'outcome', 'survival']):
            return "prognostic"
        elif any(word in question_lower for word in ['test', 'investigation', 'workup']):
            return "diagnostic_workup"
        else:
            return "general_medical"
    
    def _classify_medical_specialty(self, question: str) -> str:
        """Classify the medical specialty of the question."""
        question_lower = question.lower()
        
        specialty_keywords = {
            'cardiology': ['heart', 'cardiac', 'ecg', 'myocardial', 'arrhythmia'],
            'pulmonology': ['lung', 'respiratory', 'breathing', 'asthma', 'copd'],
            'gastroenterology': ['stomach', 'intestine', 'digestive', 'liver', 'gallbladder'],
            'neurology': ['brain', 'neurologic', 'seizure', 'stroke', 'headache'],
            'endocrinology': ['diabetes', 'thyroid', 'hormone', 'insulin', 'glucose'],
            'infectious_disease': ['infection', 'antibiotic', 'fever', 'sepsis', 'pneumonia'],
            'oncology': ['cancer', 'tumor', 'malignant', 'chemotherapy', 'metastasis'],
            'emergency_medicine': ['emergency', 'trauma', 'acute', 'urgent', 'critical']
        }
        
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return specialty
        
        return "general_medicine"
    
    def _classify_reasoning_type(self, question: str) -> str:
        """Classify the type of clinical reasoning required."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['step', 'approach', 'systematic']):
            return "systematic_reasoning"
        elif any(word in question_lower for word in ['differential', 'consider', 'rule out']):
            return "differential_diagnosis"
        elif any(word in question_lower for word in ['evidence', 'study', 'research']):
            return "evidence_based"
        elif any(word in question_lower for word in ['cause', 'mechanism', 'pathophysiology']):
            return "causal_reasoning"
        else:
            return "clinical_judgment"
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[str]) -> Dict[str, Any]:
        """
        Evaluate response using MIRAGE-specific metrics.
        
        Args:
            question: Question data including answer and context
            response: Generated response to evaluate
            retrieved_docs: List of retrieved document snippets
            
        Returns:
            Dictionary containing evaluation scores and metrics
        """
        logger.debug(f"🔍 Evaluating MIRAGE response for question {question.get('question_id', 'unknown')}")
        
        # Initialize metrics
        metrics = {
            "exact_match": 0.0,
            "semantic_similarity": 0.0,
            "medical_accuracy": 0.0,
            "clinical_relevance": 0.0,
            "reasoning_quality": 0.0,
            "evidence_support": 0.0,
            "overall_score": 0.0
        }
        
        try:
            correct_answer = question.get("answer", "").strip()
            if not correct_answer:
                logger.warning("No correct answer provided for evaluation")
                return metrics
            
            # 1. Exact match evaluation
            metrics["exact_match"] = self._calculate_exact_match(response, correct_answer)
            
            # 2. Semantic similarity (if model available)
            if self.similarity_model:
                metrics["semantic_similarity"] = self._calculate_semantic_similarity(response, correct_answer)
            
            # 3. Medical accuracy assessment
            metrics["medical_accuracy"] = self._assess_medical_accuracy(response, question)
            
            # 4. Clinical relevance evaluation
            metrics["clinical_relevance"] = self._assess_clinical_relevance(response, question)
            
            # 5. Reasoning quality assessment
            metrics["reasoning_quality"] = self._assess_reasoning_quality(response, question)
            
            # 6. Evidence support evaluation
            metrics["evidence_support"] = self._assess_evidence_support(response, retrieved_docs, question)
            
            # 7. Calculate overall score
            metrics["overall_score"] = self._calculate_overall_score(metrics)
            
            logger.debug(f"   📊 Evaluation complete. Overall score: {metrics['overall_score']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            
        return metrics
    
    def _calculate_exact_match(self, response: str, correct_answer: str) -> float:
        """Calculate exact match score."""
        response_clean = self._normalize_answer(response)
        correct_clean = self._normalize_answer(correct_answer)
        return 1.0 if response_clean == correct_clean else 0.0
    
    def _calculate_semantic_similarity(self, response: str, correct_answer: str) -> float:
        """Calculate semantic similarity using sentence transformer."""
        try:
            embeddings = self.similarity_model.encode([response, correct_answer])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            logger.debug(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def _assess_medical_accuracy(self, response: str, question: Dict) -> float:
        """Assess medical accuracy of the response."""
        score = 0.0
        response_lower = response.lower()
        
        # Check for medical terminology alignment
        question_text = question.get("question", "").lower()
        context = question.get("context", "").lower()
        
        # Extract medical terms from question and context
        medical_terms = set()
        for category, terms in self.medical_patterns.items():
            for term in terms:
                if term in question_text or term in context:
                    medical_terms.add(term)
        
        # Score based on medical term usage
        if medical_terms:
            matching_terms = sum(1 for term in medical_terms if term in response_lower)
            score += (matching_terms / len(medical_terms)) * 0.5
        
        # Check for contraindications or harmful advice
        harmful_patterns = ['not recommended', 'contraindicated', 'dangerous', 'harmful']
        if any(pattern in response_lower for pattern in harmful_patterns):
            score += 0.3  # Bonus for safety awareness
        
        # Penalty for overly generic responses
        generic_patterns = ['depends', 'varies', 'consult doctor', 'see physician']
        generic_count = sum(1 for pattern in generic_patterns if pattern in response_lower)
        if generic_count > 1:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _assess_clinical_relevance(self, response: str, question: Dict) -> float:
        """Assess clinical relevance of the response."""
        score = 0.5  # Base score
        response_lower = response.lower()
        
        question_type = question.get("question_type", "")
        
        # Type-specific relevance checks
        if question_type == "diagnostic":
            diagnostic_terms = ['diagnosis', 'symptoms', 'signs', 'findings', 'presentation']
            score += sum(0.1 for term in diagnostic_terms if term in response_lower)
        
        elif question_type == "therapeutic":
            treatment_terms = ['treatment', 'therapy', 'medication', 'management', 'intervention']
            score += sum(0.1 for term in treatment_terms if term in response_lower)
        
        elif question_type == "prognostic":
            prognosis_terms = ['prognosis', 'outcome', 'survival', 'recovery', 'course']
            score += sum(0.1 for term in prognosis_terms if term in response_lower)
        
        return max(0.0, min(1.0, score))
    
    def _assess_reasoning_quality(self, response: str, question: Dict) -> float:
        """Assess quality of clinical reasoning in response."""
        score = 0.0
        response_lower = response.lower()
        
        # Check for reasoning indicators
        reasoning_score = 0.0
        for category, indicators in self.reasoning_indicators.items():
            category_matches = sum(1 for indicator in indicators if indicator in response_lower)
            if category_matches > 0:
                reasoning_score += 0.25
        
        score += min(1.0, reasoning_score)
        
        # Check for structured reasoning
        structure_indicators = ['first', 'second', 'then', 'next', 'finally', 'therefore']
        structure_score = sum(0.1 for indicator in structure_indicators if indicator in response_lower)
        score += min(0.3, structure_score)
        
        # Check response length (neither too short nor too verbose)
        word_count = len(response.split())
        if 20 <= word_count <= 200:
            score += 0.2
        elif word_count < 10:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _assess_evidence_support(self, response: str, retrieved_docs: List[str], question: Dict) -> float:
        """Assess how well the response is supported by retrieved evidence."""
        if not retrieved_docs:
            return 0.0
        
        score = 0.0
        response_lower = response.lower()
        
        # Check overlap with retrieved documents
        doc_text = " ".join(retrieved_docs).lower()
        
        # Calculate term overlap
        response_terms = set(response_lower.split())
        doc_terms = set(doc_text.split())
        
        if doc_terms:
            overlap = len(response_terms.intersection(doc_terms)) / len(response_terms)
            score += overlap * 0.6
        
        # Check for specific medical evidence
        evidence_terms = ['study', 'research', 'guidelines', 'evidence', 'trial', 'recommendation']
        evidence_score = sum(0.1 for term in evidence_terms if term in doc_text)
        score += min(0.4, evidence_score)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        weights = {
            "exact_match": 0.3,
            "semantic_similarity": 0.2,
            "medical_accuracy": 0.25,
            "clinical_relevance": 0.1,
            "reasoning_quality": 0.1,
            "evidence_support": 0.05
        }
        
        overall = sum(metrics.get(metric, 0.0) * weight for metric, weight in weights.items())
        return max(0.0, min(1.0, overall))
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        return re.sub(r'[^\w\s]', '', answer.lower().strip())
    
    def get_evaluation_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for MIRAGE evaluation."""
        if not results:
            return {"error": "No results to summarize"}
        
        # Calculate aggregate metrics
        metrics = ["exact_match", "semantic_similarity", "medical_accuracy", 
                  "clinical_relevance", "reasoning_quality", "evidence_support", "overall_score"]
        
        summary = {
            "total_questions": len(results),
            "average_scores": {},
            "score_distribution": {},
            "performance_by_type": {},
            "benchmark": "MIRAGE"
        }
        
        # Calculate averages
        for metric in metrics:
            scores = [r.get(metric, 0.0) for r in results]
            summary["average_scores"][metric] = np.mean(scores) if scores else 0.0
            
            # Score distribution
            summary["score_distribution"][metric] = {
                "min": np.min(scores) if scores else 0.0,
                "max": np.max(scores) if scores else 0.0,
                "std": np.std(scores) if scores else 0.0,
                "median": np.median(scores) if scores else 0.0
            }
        
        # Performance by question type
        question_types = set(r.get("question_type", "unknown") for r in results)
        for qtype in question_types:
            type_results = [r for r in results if r.get("question_type") == qtype]
            if type_results:
                type_scores = [r.get("overall_score", 0.0) for r in type_results]
                summary["performance_by_type"][qtype] = {
                    "count": len(type_results),
                    "average_score": np.mean(type_scores),
                    "accuracy": np.mean([r.get("exact_match", 0.0) for r in type_results])
                }
        
        return summary