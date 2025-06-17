"""
MedReason benchmark implementation for clinical reasoning evaluation.
"""

import json
from typing import Dict, List
from pathlib import Path
from loguru import logger

from .base_benchmark import BaseBenchmark


class MedReasonBenchmark(BaseBenchmark):
    """MedReason benchmark for clinical reasoning chain evaluation."""
    
    def __init__(self, config: Dict):
        """Initialize MedReason benchmark."""
        super().__init__(config)
        self.dataset_url = config.get("dataset_url", "")
    
    def load_dataset(self) -> List[Dict]:
        """Load MedReason dataset."""
        try:
            # Try to load from cache
            cache_path = Path("data/evaluation/cache/medreason_dataset.json")
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded MedReason dataset from cache: {len(data)} questions")
                return data
            
            # Try to load from Hugging Face datasets
            try:
                from datasets import load_dataset
                dataset = load_dataset("UCSC-VLAA/MedReason", split="train")
                data = [item for item in dataset]
                logger.info(f"Loaded MedReason from Hugging Face: {len(data)} items")
                return data
            except Exception:
                pass
            
            # Generate sample data
            logger.warning("Using sample MedReason data - replace with actual dataset")
            return self._generate_sample_data()
            
        except Exception as e:
            logger.error(f"Failed to load MedReason dataset: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> List[Dict]:
        """Generate sample MedReason-style questions."""
        reasoning_questions = [
            {
                "question": "A 65-year-old male presents with crushing chest pain radiating to the left arm. What is your reasoning process?",
                "reasoning_chain": [
                    "Pattern recognition: Classic presentation of acute coronary syndrome",
                    "Hypothesis formation: Most likely STEMI vs NSTEMI vs unstable angina",
                    "Evidence gathering: Need EKG, troponins, vital signs",
                    "Differential diagnosis: Also consider aortic dissection, PE, GERD",
                    "Final assessment: High suspicion for acute MI, proceed with ACS protocol"
                ],
                "answer": "This presentation is highly suspicious for acute myocardial infarction requiring immediate evaluation and treatment.",
                "medical_specialty": "cardiology"
            },
            {
                "question": "A 28-year-old female presents with polyuria, polydipsia, and weight loss. Walk through your diagnostic reasoning.",
                "reasoning_chain": [
                    "Pattern recognition: Classic triad suggests diabetes mellitus",
                    "Age consideration: Young age raises suspicion for Type 1 DM",
                    "Hypothesis testing: Check glucose, HbA1c, ketones, autoantibodies",
                    "Differential diagnosis: Type 1 vs Type 2 vs MODY vs secondary diabetes",
                    "Risk assessment: Evaluate for diabetic ketoacidosis"
                ],
                "answer": "The presentation strongly suggests diabetes mellitus, likely Type 1 given the age and acute presentation.",
                "medical_specialty": "endocrinology"
            },
            {
                "question": "A 45-year-old presents with sudden severe headache. Describe your reasoning approach.",
                "reasoning_chain": [
                    "Red flag recognition: Sudden severe headache is concerning",
                    "Critical diagnosis consideration: Subarachnoid hemorrhage, meningitis",
                    "History gathering: Previous headaches, trauma, fever, neurologic symptoms",
                    "Physical examination: Neurologic exam, meningeal signs, fundoscopy",
                    "Immediate workup: CT head, consider lumbar puncture if CT negative"
                ],
                "answer": "Sudden severe headache requires urgent evaluation to rule out subarachnoid hemorrhage and other dangerous causes.",
                "medical_specialty": "emergency_medicine"
            }
        ]
        
        # Replicate for sample size
        questions = reasoning_questions * (self.sample_size // len(reasoning_questions) + 1)
        
        # Add unique IDs
        for i, q in enumerate(questions[:self.sample_size]):
            q["id"] = f"medreason_{i:04d}"
        
        return questions[:self.sample_size]
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate model response for clinical reasoning."""
        
        # Extract components
        correct_reasoning = question.get("reasoning_chain", [])
        correct_answer = question.get("answer", "")
        specialty = question.get("medical_specialty", "general")
        
        # Evaluate reasoning depth
        reasoning_score = self._evaluate_reasoning_depth(response, correct_reasoning)
        
        # Evaluate medical accuracy
        accuracy_score = self._evaluate_medical_accuracy(response, correct_answer)
        
        # Evaluate clinical workflow alignment
        workflow_score = self._evaluate_clinical_workflow(response)
        
        # Evaluate specialty knowledge
        specialty_score = self._evaluate_specialty_knowledge(response, specialty)
        
        # Calculate overall score
        overall_score = (reasoning_score * 0.4 + 
                        accuracy_score * 0.3 + 
                        workflow_score * 0.2 + 
                        specialty_score * 0.1)
        
        return {
            "question_id": question.get("id"),
            "specialty": specialty,
            "score": overall_score * 100,
            "correct": overall_score > 0.6,
            "metrics": {
                "reasoning_depth": reasoning_score,
                "medical_accuracy": accuracy_score,
                "clinical_workflow": workflow_score,
                "specialty_knowledge": specialty_score
            },
            "response": response,
            "expected_reasoning": correct_reasoning
        }
    
    def _evaluate_reasoning_depth(self, response: str, expected_reasoning: List[str]) -> float:
        """Evaluate depth and quality of clinical reasoning."""
        response_lower = response.lower()
        
        # Check for reasoning indicators
        reasoning_indicators = [
            "pattern", "hypothesis", "differential", "evidence", "assessment",
            "because", "therefore", "suggests", "indicates", "likely"
        ]
        
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
        max_reasoning = len(reasoning_indicators)
        
        # Check for step-by-step reasoning
        step_indicators = ["first", "second", "next", "then", "finally", "step"]
        step_count = sum(1 for indicator in step_indicators if indicator in response_lower)
        
        # Combine scores
        reasoning_depth = min(reasoning_count / max_reasoning, 1.0)
        step_structure = min(step_count / 3, 1.0)  # Expect at least 3 steps
        
        return (reasoning_depth * 0.7 + step_structure * 0.3)
    
    def _evaluate_medical_accuracy(self, response: str, expected_answer: str) -> float:
        """Evaluate medical accuracy of the reasoning."""
        # Simple keyword overlap for now
        response_words = set(response.lower().split())
        expected_words = set(expected_answer.lower().split())
        
        if not expected_words:
            return 0.8
        
        overlap = len(response_words.intersection(expected_words))
        return min(overlap / len(expected_words), 1.0)
    
    def _evaluate_clinical_workflow(self, response: str) -> float:
        """Evaluate adherence to clinical workflow patterns."""
        response_lower = response.lower()
        
        # Check for clinical workflow components
        workflow_components = [
            "history", "examination", "assessment", "plan", "diagnosis",
            "investigation", "treatment", "follow-up", "monitor"
        ]
        
        component_count = sum(1 for comp in workflow_components if comp in response_lower)
        return min(component_count / len(workflow_components), 1.0)
    
    def _evaluate_specialty_knowledge(self, response: str, specialty: str) -> float:
        """Evaluate specialty-specific knowledge demonstration."""
        response_lower = response.lower()
        
        # Specialty-specific terms
        specialty_terms = {
            "cardiology": ["ecg", "troponin", "catheterization", "stent", "arrhythmia"],
            "endocrinology": ["hormone", "glucose", "insulin", "thyroid", "diabetes"],
            "emergency_medicine": ["triage", "acute", "urgent", "immediate", "protocol"],
            "general": ["symptom", "diagnosis", "treatment", "patient", "medical"]
        }
        
        terms = specialty_terms.get(specialty, specialty_terms["general"])
        term_count = sum(1 for term in terms if term in response_lower)
        
        return min(term_count / len(terms), 1.0)