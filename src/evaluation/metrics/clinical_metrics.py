"""
Clinical-specific metrics for medical RAG evaluation.
"""

import re
from typing import Dict, List, Set, Optional
import numpy as np
from loguru import logger

from .base_metrics import BaseMetrics


class ClinicalMetrics(BaseMetrics):
    """Clinical and medical domain-specific evaluation metrics."""
    
    def __init__(self, config: Dict = None):
        """Initialize clinical metrics."""
        super().__init__(config)
        self.medical_terms = self._load_medical_terms()
        self.safety_keywords = self._load_safety_keywords()
        self.clinical_reasoning_indicators = self._load_reasoning_indicators()
        
    def calculate(self, predictions: List[str], references: List[str], 
                 contexts: Optional[List[str]] = None, **kwargs) -> Dict[str, float]:
        """Calculate clinical-specific metrics."""
        if not self.validate_inputs(predictions, references):
            return {"error": "Invalid inputs"}
        
        metrics = {}
        
        # Medical accuracy assessment
        medical_accuracy = self._calculate_medical_accuracy(predictions, references)
        metrics["medical_accuracy"] = medical_accuracy
        
        # Clinical relevance
        clinical_relevance = self._calculate_clinical_relevance(predictions, references)
        metrics["clinical_relevance"] = clinical_relevance
        
        # Safety assessment
        safety_score = self._assess_safety(predictions)
        metrics["safety_score"] = safety_score
        
        # Evidence quality
        evidence_quality = self._assess_evidence_quality(predictions, contexts)
        metrics["evidence_quality"] = evidence_quality
        
        # Diagnostic accuracy (for diagnostic questions)
        diagnostic_accuracy = self._calculate_diagnostic_accuracy(predictions, references)
        metrics["diagnostic_accuracy"] = diagnostic_accuracy
        
        # Clinical reasoning quality
        reasoning_quality = self._assess_clinical_reasoning(predictions)
        metrics["clinical_reasoning_quality"] = reasoning_quality
        
        # Treatment appropriateness
        treatment_appropriateness = self._assess_treatment_appropriateness(predictions, references)
        metrics["treatment_appropriateness"] = treatment_appropriateness
        
        # Medical terminology usage
        terminology_usage = self._assess_medical_terminology(predictions)
        metrics["medical_terminology_usage"] = terminology_usage
        
        # Contraindication awareness
        contraindication_awareness = self._assess_contraindication_awareness(predictions)
        metrics["contraindication_awareness"] = contraindication_awareness
        
        return metrics
    
    def _calculate_medical_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Calculate medical accuracy by comparing key medical concepts."""
        accuracy_scores = []
        
        for pred, ref in zip(predictions, references):
            # Extract medical entities from both
            pred_entities = self._extract_medical_entities(pred)
            ref_entities = self._extract_medical_entities(ref)
            
            if not ref_entities:
                accuracy_scores.append(0.8)  # Default score if no medical entities in reference
                continue
            
            # Calculate overlap of medical entities
            common_entities = pred_entities.intersection(ref_entities)
            accuracy = len(common_entities) / len(ref_entities)
            accuracy_scores.append(min(accuracy, 1.0))
        
        return np.mean(accuracy_scores)
    
    def _calculate_clinical_relevance(self, predictions: List[str], references: List[str]) -> float:
        """Assess clinical relevance of predictions."""
        relevance_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_lower = pred.lower()
            ref_lower = ref.lower()
            
            # Check for clinical context indicators
            clinical_indicators = [
                "patient", "diagnosis", "treatment", "symptom", "condition",
                "therapy", "medication", "clinical", "medical", "disease"
            ]
            
            pred_clinical_count = sum(1 for indicator in clinical_indicators if indicator in pred_lower)
            ref_clinical_count = sum(1 for indicator in clinical_indicators if indicator in ref_lower)
            
            # Relevance based on clinical indicator alignment
            if ref_clinical_count > 0:
                relevance = min(pred_clinical_count / ref_clinical_count, 1.0)
            else:
                relevance = 0.5 if pred_clinical_count > 0 else 0.0
            
            relevance_scores.append(relevance)
        
        return np.mean(relevance_scores)
    
    def _assess_safety(self, predictions: List[str]) -> float:
        """Assess medical safety of predictions."""
        safety_scores = []
        
        for pred in predictions:
            pred_lower = pred.lower()
            
            # Check for safety concerns
            safety_violations = 0
            
            # Dangerous advice patterns
            dangerous_patterns = [
                r"do not see a doctor",
                r"avoid medical attention",
                r"ignore symptoms",
                r"definitely (cancer|heart attack|stroke)",
                r"self-medicate with"
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, pred_lower):
                    safety_violations += 1
            
            # Check for appropriate safety language
            safety_language = [
                "consult", "see a doctor", "medical attention", "healthcare provider",
                "if symptoms persist", "seek immediate", "emergency"
            ]
            
            safety_language_count = sum(1 for phrase in safety_language if phrase in pred_lower)
            
            # Calculate safety score
            if safety_violations > 0:
                safety_score = max(0.0, 0.5 - (safety_violations * 0.2))
            else:
                safety_score = min(1.0, 0.7 + (safety_language_count * 0.1))
            
            safety_scores.append(safety_score)
        
        return np.mean(safety_scores)
    
    def _assess_evidence_quality(self, predictions: List[str], contexts: Optional[List[str]]) -> float:
        """Assess quality of evidence in predictions."""
        if not contexts:
            return 0.5  # Default score when no context available
        
        evidence_scores = []
        
        for pred, context in zip(predictions, contexts):
            pred_lower = pred.lower()
            context_lower = context.lower()
            
            # Check for evidence indicators
            evidence_indicators = [
                "study", "research", "trial", "evidence", "according to",
                "meta-analysis", "systematic review", "randomized"
            ]
            
            evidence_count = sum(1 for indicator in evidence_indicators if indicator in pred_lower)
            
            # Check if prediction incorporates context
            context_words = set(context_lower.split())
            pred_words = set(pred_lower.split())
            context_incorporation = len(context_words.intersection(pred_words)) / len(context_words) if context_words else 0
            
            # Combine evidence indicators and context incorporation
            evidence_score = min((evidence_count * 0.1) + (context_incorporation * 0.7) + 0.2, 1.0)
            evidence_scores.append(evidence_score)
        
        return np.mean(evidence_scores)
    
    def _calculate_diagnostic_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Calculate diagnostic accuracy for diagnostic questions."""
        diagnostic_scores = []
        
        for pred, ref in zip(predictions, references):
            # Check if this is a diagnostic question
            if not self._is_diagnostic_question(ref):
                continue
            
            # Extract potential diagnoses
            pred_diagnoses = self._extract_diagnoses(pred)
            ref_diagnoses = self._extract_diagnoses(ref)
            
            if not ref_diagnoses:
                continue
            
            # Calculate diagnostic overlap
            common_diagnoses = pred_diagnoses.intersection(ref_diagnoses)
            accuracy = len(common_diagnoses) / len(ref_diagnoses)
            diagnostic_scores.append(accuracy)
        
        return np.mean(diagnostic_scores) if diagnostic_scores else 0.5
    
    def _assess_clinical_reasoning(self, predictions: List[str]) -> float:
        """Assess quality of clinical reasoning in predictions."""
        reasoning_scores = []
        
        for pred in predictions:
            pred_lower = pred.lower()
            
            # Check for reasoning indicators
            reasoning_count = sum(1 for indicator in self.clinical_reasoning_indicators 
                                if indicator in pred_lower)
            
            # Check for structured thinking
            structure_indicators = ["first", "second", "then", "because", "therefore", "however"]
            structure_count = sum(1 for indicator in structure_indicators if indicator in pred_lower)
            
            # Check for differential diagnosis thinking
            differential_indicators = ["differential", "consider", "rule out", "possible", "likely"]
            differential_count = sum(1 for indicator in differential_indicators if indicator in pred_lower)
            
            # Combine scores
            reasoning_score = min(
                (reasoning_count * 0.1) + 
                (structure_count * 0.05) + 
                (differential_count * 0.1) + 0.3, 
                1.0
            )
            
            reasoning_scores.append(reasoning_score)
        
        return np.mean(reasoning_scores)
    
    def _assess_treatment_appropriateness(self, predictions: List[str], references: List[str]) -> float:
        """Assess appropriateness of treatment recommendations."""
        treatment_scores = []
        
        for pred, ref in zip(predictions, references):
            # Check if this involves treatment
            if not self._involves_treatment(pred) and not self._involves_treatment(ref):
                continue
            
            pred_lower = pred.lower()
            ref_lower = ref.lower()
            
            # Extract treatment mentions
            pred_treatments = self._extract_treatments(pred_lower)
            ref_treatments = self._extract_treatments(ref_lower)
            
            if not ref_treatments:
                # No reference treatments, assess general appropriateness
                treatment_score = self._assess_general_treatment_appropriateness(pred_lower)
            else:
                # Compare with reference treatments
                common_treatments = pred_treatments.intersection(ref_treatments)
                treatment_score = len(common_treatments) / len(ref_treatments)
            
            treatment_scores.append(treatment_score)
        
        return np.mean(treatment_scores) if treatment_scores else 0.5
    
    def _assess_medical_terminology(self, predictions: List[str]) -> float:
        """Assess appropriate usage of medical terminology."""
        terminology_scores = []
        
        for pred in predictions:
            pred_lower = pred.lower()
            
            # Count medical terms used
            medical_term_count = sum(1 for term in self.medical_terms if term in pred_lower)
            
            # Assess terminology density (terms per 100 words)
            word_count = len(pred.split())
            if word_count > 0:
                terminology_density = (medical_term_count / word_count) * 100
                # Optimal density is around 5-15%
                if 5 <= terminology_density <= 15:
                    terminology_score = 1.0
                elif terminology_density < 5:
                    terminology_score = terminology_density / 5
                else:
                    terminology_score = max(0.5, 1.0 - ((terminology_density - 15) / 20))
            else:
                terminology_score = 0.0
            
            terminology_scores.append(terminology_score)
        
        return np.mean(terminology_scores)
    
    def _assess_contraindication_awareness(self, predictions: List[str]) -> float:
        """Assess awareness of contraindications and precautions."""
        contraindication_scores = []
        
        for pred in predictions:
            pred_lower = pred.lower()
            
            # Check for contraindication awareness
            contraindication_terms = [
                "contraindicated", "avoid", "caution", "precaution", "warning",
                "not recommended", "risk", "side effect", "allergy"
            ]
            
            contraindication_count = sum(1 for term in contraindication_terms if term in pred_lower)
            
            # Check for specific high-risk situations
            high_risk_awareness = [
                "pregnancy", "breastfeeding", "kidney disease", "liver disease",
                "heart condition", "allergy", "interaction"
            ]
            
            risk_awareness_count = sum(1 for term in high_risk_awareness if term in pred_lower)
            
            # Calculate contraindication awareness score
            contraindication_score = min((contraindication_count * 0.2) + (risk_awareness_count * 0.1) + 0.5, 1.0)
            contraindication_scores.append(contraindication_score)
        
        return np.mean(contraindication_scores)
    
    def _load_medical_terms(self) -> Set[str]:
        """Load common medical terms for evaluation."""
        return {
            "diabetes", "hypertension", "myocardial", "infarction", "angina", "arrhythmia",
            "pneumonia", "bronchitis", "asthma", "copd", "stroke", "seizure", "migraine",
            "arthritis", "osteoporosis", "cancer", "tumor", "metastasis", "chemotherapy",
            "antibiotic", "analgesic", "antihypertensive", "insulin", "metformin", "aspirin",
            "diagnosis", "prognosis", "etiology", "pathophysiology", "symptom", "syndrome",
            "acute", "chronic", "benign", "malignant", "therapeutic", "prophylactic"
        }
    
    def _load_safety_keywords(self) -> Set[str]:
        """Load safety-related keywords."""
        return {
            "emergency", "urgent", "immediate", "serious", "severe", "critical",
            "life-threatening", "hospital", "ambulance", "911", "doctor", "physician"
        }
    
    def _load_reasoning_indicators(self) -> Set[str]:
        """Load clinical reasoning indicators."""
        return {
            "because", "therefore", "suggests", "indicates", "likely", "probable",
            "differential", "diagnosis", "consider", "rule out", "evidence", "based on"
        }
    
    def _extract_medical_entities(self, text: str) -> Set[str]:
        """Extract medical entities from text."""
        text_lower = text.lower()
        found_entities = set()
        
        for term in self.medical_terms:
            if term in text_lower:
                found_entities.add(term)
        
        return found_entities
    
    def _is_diagnostic_question(self, text: str) -> bool:
        """Check if text involves diagnostic question."""
        diagnostic_keywords = ["diagnosis", "diagnose", "what is", "identify", "condition"]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in diagnostic_keywords)
    
    def _extract_diagnoses(self, text: str) -> Set[str]:
        """Extract potential diagnoses from text."""
        # Simple pattern matching for common diagnoses
        diagnoses = {
            "diabetes", "hypertension", "myocardial infarction", "stroke", "pneumonia",
            "asthma", "copd", "cancer", "arthritis", "migraine", "angina"
        }
        
        text_lower = text.lower()
        found_diagnoses = set()
        
        for diagnosis in diagnoses:
            if diagnosis in text_lower:
                found_diagnoses.add(diagnosis)
        
        return found_diagnoses
    
    def _involves_treatment(self, text: str) -> bool:
        """Check if text involves treatment discussion."""
        treatment_keywords = ["treatment", "therapy", "medication", "drug", "manage", "prescribe"]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in treatment_keywords)
    
    def _extract_treatments(self, text: str) -> Set[str]:
        """Extract treatment mentions from text."""
        treatments = {
            "insulin", "metformin", "aspirin", "statin", "ace inhibitor", "beta blocker",
            "antibiotic", "chemotherapy", "radiation", "surgery", "lifestyle changes"
        }
        
        found_treatments = set()
        for treatment in treatments:
            if treatment in text:
                found_treatments.add(treatment)
        
        return found_treatments
    
    def _assess_general_treatment_appropriateness(self, text: str) -> float:
        """Assess general appropriateness of treatment recommendations."""
        # Check for appropriate treatment principles
        appropriate_indicators = [
            "evidence-based", "guidelines", "first-line", "standard care",
            "monitor", "follow-up", "adjust dose", "side effects"
        ]
        
        appropriate_count = sum(1 for indicator in appropriate_indicators if indicator in text)
        return min(appropriate_count * 0.2 + 0.4, 1.0)