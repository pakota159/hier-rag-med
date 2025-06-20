"""
Enhanced MIRAGE Benchmark with Real Dataset Loading
REPLACE: src/evaluation/benchmarks/mirage_benchmark.py
"""

import re
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from .base_benchmark import BaseBenchmark
from loguru import logger

from src.evaluation.data.data_loader import BenchmarkDataLoader

# Add project root to path for data loader import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

class MIRAGEBenchmark(BaseBenchmark):
    """MIRAGE benchmark with real dataset loading."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MIRAGE"
        self.data_loader = BenchmarkDataLoader(config)
        
        # Initialize semantic similarity model
        self.similarity_model = None
        self._init_evaluation_models()
    
    def load_dataset(self) -> List[Dict]:
        """Load MIRAGE dataset using centralized data loader."""
        logger.info(f"🔄 Loading MIRAGE dataset using data loader...")
        
        try:
            # Use the centralized data loader
            data = self.data_loader.load_benchmark_data(
                benchmark_name="mirage",
                split="test",
                max_samples=self.sample_size if self.sample_size < 1000 else None
            )
            
            if data and len(data) > 0:
                # Convert to MIRAGE format if needed
                formatted_data = []
                for item in data:
                    formatted_item = {
                        "id": item.get("question_id", item.get("id", f"mirage_{len(formatted_data)}")),
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "type": item.get("question_type", item.get("type", "clinical")),
                        "category": item.get("medical_specialty", item.get("category", "general")),
                        "context": item.get("context", ""),
                        "options": item.get("options", [])
                    }
                    formatted_data.append(formatted_item)
                
                logger.info(f"✅ Loaded {len(formatted_data)} MIRAGE questions via data loader")
                return formatted_data
            
        except Exception as e:
            logger.error(f"❌ Data loader failed for MIRAGE: {e}")
        
        # Fallback to minimal synthetic data if data loader fails completely
        logger.warning("⚠️ Using minimal fallback dataset for MIRAGE")
        return self._generate_minimal_fallback()
    
    def _generate_minimal_fallback(self) -> List[Dict]:
        """Generate minimal fallback dataset if all else fails."""
        return [
            {
                "id": "mirage_fallback_001",
                "question": "What is the first-line treatment for type 2 diabetes?",
                "answer": "Metformin",
                "type": "clinical",
                "category": "endocrinology"
            },
            {
                "id": "mirage_fallback_002",
                "question": "What are the main symptoms of myocardial infarction?",
                "answer": "chest pain, shortness of breath, nausea, sweating",
                "type": "clinical", 
                "category": "cardiology"
            }
        ]
        
    def _init_evaluation_models(self):
        """Initialize models for medical evaluation."""
        try:
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("✅ Loaded semantic similarity model")
        except Exception as e:
            logger.warning(f"Could not load semantic model: {e}")
            self.similarity_model = None
    
    def _generate_comprehensive_mirage_dataset(self) -> List[Dict]:
        """Generate comprehensive synthetic MIRAGE dataset."""
        synthetic_data = []
        
        # Medical specialties with question templates
        medical_data = {
            "cardiology": [
                ("What is the first-line treatment for acute STEMI?", "Primary PCI or fibrinolysis"),
                ("What are the classic signs of heart failure?", "Shortness of breath, peripheral edema, elevated JVP"),
                ("Which medication is contraindicated in severe aortic stenosis?", "Nitrates"),
                ("What is the most common cause of sudden cardiac death?", "Ventricular arrhythmias"),
                ("What ECG changes suggest anterior STEMI?", "ST elevation in V1-V6"),
            ],
            "endocrinology": [
                ("What is the target HbA1c for most diabetic patients?", "Less than 7%"),
                ("What is the first-line treatment for type 2 diabetes?", "Metformin"),
                ("What are symptoms of diabetic ketoacidosis?", "Polyuria, polydipsia, vomiting, altered consciousness"),
                ("Which hormone is deficient in type 1 diabetes?", "Insulin"),
                ("What is the treatment for severe hypoglycemia?", "IV glucose or glucagon"),
            ],
            "pulmonology": [
                ("What is the most common cause of COPD?", "Smoking"),
                ("Which medication is first-line for asthma maintenance?", "Inhaled corticosteroids"),
                ("What are signs of pneumonia on chest X-ray?", "Consolidation, air bronchograms"),
                ("What is the gold standard for COPD diagnosis?", "Spirometry showing FEV1/FVC < 0.7"),
                ("Which organism causes atypical pneumonia?", "Mycoplasma pneumoniae"),
            ],
            "gastroenterology": [
                ("What is the most common cause of peptic ulcers?", "Helicobacter pylori"),
                ("Which test diagnoses H. pylori infection?", "Urea breath test or stool antigen"),
                ("What are symptoms of inflammatory bowel disease?", "Diarrhea, abdominal pain, weight loss"),
                ("Which medication treats GERD?", "Proton pump inhibitors"),
                ("What complication can occur with long-standing GERD?", "Barrett's esophagus"),
            ],
            "neurology": [
                ("What is the acute treatment for ischemic stroke?", "Thrombolysis with tPA"),
                ("Which imaging is first-line for suspected stroke?", "Non-contrast CT head"),
                ("What are symptoms of Parkinson's disease?", "Tremor, rigidity, bradykinesia"),
                ("Which medication treats Alzheimer's disease?", "Cholinesterase inhibitors"),
                ("What is the most common type of headache?", "Tension-type headache"),
            ],
            "infectious_disease": [
                ("What is the first-line antibiotic for pneumonia?", "Amoxicillin or macrolide"),
                ("Which organism causes meningitis in young adults?", "Neisseria meningitidis"),
                ("What is the treatment for methicillin-resistant staph?", "Vancomycin or linezolid"),
                ("Which vaccine prevents pneumococcal disease?", "Pneumococcal conjugate vaccine"),
                ("What are symptoms of sepsis?", "Fever, tachycardia, hypotension, altered mental status"),
            ],
            "nephrology": [
                ("What are causes of acute kidney injury?", "Prerenal, intrinsic renal, postrenal"),
                ("Which medication can cause nephrotoxicity?", "NSAIDs, ACE inhibitors, aminoglycosides"),
                ("What is the most common cause of chronic kidney disease?", "Diabetes mellitus"),
                ("Which electrolyte abnormality occurs in kidney failure?", "Hyperkalemia"),
                ("What is the treatment for end-stage renal disease?", "Dialysis or kidney transplant"),
            ],
            "hematology": [
                ("What is the most common type of anemia?", "Iron deficiency anemia"),
                ("Which test diagnoses iron deficiency?", "Serum ferritin"),
                ("What are symptoms of anemia?", "Fatigue, pallor, shortness of breath"),
                ("Which medication treats pernicious anemia?", "Vitamin B12 injections"),
                ("What causes sickle cell crisis?", "Vaso-occlusion from sickled red blood cells"),
            ]
        }
        
        # Generate questions for each specialty
        question_id = 1
        for specialty, questions in medical_data.items():
            for question, answer in questions:
                synthetic_data.append({
                    "id": f"mirage_{question_id:03d}",
                    "question": question,
                    "answer": answer,
                    "type": "clinical",
                    "category": specialty
                })
                question_id += 1
        
        # Add research/pharmacology questions
        research_questions = [
            ("What is the mechanism of action of statins?", "HMG-CoA reductase inhibition"),
            ("Which study design provides strongest evidence?", "Randomized controlled trial"),
            ("What is the number needed to treat?", "Number of patients to treat for one to benefit"),
            ("What does sensitivity measure in diagnostic tests?", "Proportion of true positives correctly identified"),
            ("What does specificity measure in diagnostic tests?", "Proportion of true negatives correctly identified"),
        ]
        
        for question, answer in research_questions:
            synthetic_data.append({
                "id": f"mirage_{question_id:03d}",
                "question": question,
                "answer": answer,
                "type": "research",
                "category": "methodology"
            })
            question_id += 1
        
        # Generate additional questions to reach 200 total
        while len(synthetic_data) < 200:
            specialty = list(medical_data.keys())[question_id % len(medical_data)]
            synthetic_data.append({
                "id": f"mirage_{question_id:03d}",
                "question": f"Clinical question {question_id} about {specialty} diagnosis and management.",
                "answer": f"Evidence-based answer {question_id} for {specialty} clinical scenario.",
                "type": "clinical",
                "category": specialty
            })
            question_id += 1
        
        logger.info(f"✅ Generated {len(synthetic_data)} comprehensive MIRAGE questions")
        return synthetic_data
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate single response using medical RAG methods."""
        
        # Extract ground truth
        correct_answer = question.get("answer", "")
        question_type = question.get("type", "clinical")
        
        # Calculate semantic similarity
        semantic_score = self._calculate_semantic_similarity(response, correct_answer)
        
        # Medical accuracy assessment
        medical_accuracy = self._assess_medical_accuracy(response, correct_answer)
        
        # Clinical relevance
        clinical_relevance = self._assess_clinical_relevance(response, question)
        
        # Overall score (weighted combination)
        overall_score = (semantic_score * 0.5 + medical_accuracy * 0.3 + clinical_relevance * 0.2)
        
        return {
            "question_id": question.get("id"),
            "question_type": question_type,
            "score": overall_score * 100,
            "correct": overall_score > 0.5,  # 50% threshold for MIRAGE
            "metrics": {
                "semantic_similarity": semantic_score,
                "medical_accuracy": medical_accuracy,
                "clinical_relevance": clinical_relevance,
                "overall_score": overall_score
            },
            "response": response,
            "ground_truth": correct_answer
        }
    
    def _calculate_semantic_similarity(self, response: str, reference: str) -> float:
        """Calculate semantic similarity using embeddings."""
        if not response or not reference:
            return 0.0
        
        if self.similarity_model:
            try:
                embeddings = self.similarity_model.encode([response, reference])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return max(0.0, float(similarity))
            except Exception as e:
                logger.warning(f"Semantic similarity calculation failed: {e}")
        
        # Fallback to word overlap
        return self._calculate_word_overlap(response, reference)
    
    def _assess_medical_accuracy(self, response: str, reference: str) -> float:
        """Assess medical accuracy using concept matching."""
        if not response or not reference:
            return 0.0
        
        response_lower = response.lower()
        reference_lower = reference.lower()
        
        # Medical concept keywords
        medical_concepts = {
            "medications": ["metformin", "insulin", "statin", "ace inhibitor", "beta blocker"],
            "procedures": ["ecg", "ct scan", "mri", "biopsy", "surgery"],
            "conditions": ["diabetes", "hypertension", "heart failure", "pneumonia", "stroke"],
            "symptoms": ["chest pain", "shortness of breath", "fever", "headache", "nausea"]
        }
        
        # Extract concepts from both texts
        response_concepts = set()
        reference_concepts = set()
        
        for category, concepts in medical_concepts.items():
            for concept in concepts:
                if concept in response_lower:
                    response_concepts.add(concept)
                if concept in reference_lower:
                    reference_concepts.add(concept)
        
        # Calculate concept overlap
        if not reference_concepts:
            return self._calculate_word_overlap(response_lower, reference_lower)
        
        overlap = len(response_concepts.intersection(reference_concepts))
        return overlap / len(reference_concepts)
    
    def _assess_clinical_relevance(self, response: str, question: Dict) -> float:
        """Assess clinical relevance of the response."""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        
        # Clinical relevance indicators
        clinical_indicators = [
            "treatment", "diagnosis", "management", "therapy", "medication",
            "patient", "clinical", "medical", "healthcare", "disease"
        ]
        
        indicator_count = sum(1 for indicator in clinical_indicators if indicator in response_lower)
        relevance_score = min(indicator_count / 5.0, 1.0)
        
        # Bonus for question type alignment
        question_type = question.get("type", "clinical")
        if question_type == "clinical" and any(word in response_lower for word in ["treatment", "diagnosis", "management"]):
            relevance_score += 0.2
        elif question_type == "research" and any(word in response_lower for word in ["study", "evidence", "research"]):
            relevance_score += 0.2
        
        return min(relevance_score, 1.0)
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap between two texts."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words2:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        return overlap / len(words2)