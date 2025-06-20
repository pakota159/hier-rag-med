"""
Enhanced MSMARCO Benchmark with Real Dataset Loading
REPLACE: src/evaluation/benchmarks/msmarco_benchmark.py
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

class MSMARCOBenchmark(BaseBenchmark):
    """MSMARCO benchmark with real dataset loading, filtered for medical content."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MSMARCO"
        self.data_loader = BenchmarkDataLoader(config)
    
    def load_dataset(self) -> List[Dict]:
        """Load MSMARCO dataset using centralized data loader."""
        logger.info(f"🔄 Loading MSMARCO dataset using data loader...")
        
        try:
            # Use the centralized data loader
            data = self.data_loader.load_benchmark_data(
                benchmark_name="msmarco",
                split="validation",  # MSMARCO typically uses validation split
                max_samples=self.sample_size if self.sample_size < 1000 else None
            )
            
            if data and len(data) > 0:
                # Convert to MSMARCO format if needed
                formatted_data = []
                for item in data:
                    formatted_item = {
                        "id": item.get("question_id", item.get("id", f"msmarco_{len(formatted_data)}")),
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "context": item.get("context", ""),
                        "query_type": item.get("query_type", "factoid"),
                        "reasoning_type": "information_retrieval"
                    }
                    formatted_data.append(formatted_item)
                
                logger.info(f"✅ Loaded {len(formatted_data)} MSMARCO questions via data loader")
                return formatted_data
            
        except Exception as e:
            logger.error(f"❌ Data loader failed for MSMARCO: {e}")
        
        # Fallback to minimal synthetic data
        logger.warning("⚠️ Using minimal fallback dataset for MSMARCO")
        return self._generate_minimal_fallback()
    
    def _generate_minimal_fallback(self) -> List[Dict]:
        """Generate minimal fallback dataset if all else fails."""
        return [
            {
                "id": "msmarco_fallback_001",
                "question": "What are the symptoms of diabetes?",
                "answer": "increased thirst, frequent urination, fatigue, blurred vision",
                "context": "Diabetes symptoms include polyuria, polydipsia, and fatigue.",
                "query_type": "factoid"
            },
            {
                "id": "msmarco_fallback_002", 
                "question": "How is hypertension treated?",
                "answer": "lifestyle changes, medication, monitoring",
                "context": "Hypertension treatment includes diet, exercise, and antihypertensive drugs.",
                "query_type": "factoid"
            }
        ]
    
    def _is_medical_query(self, query: str) -> bool:
        """Check if a query is medical-related."""
        if not query:
            return False
            
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.medical_keywords)
    
    def _generate_comprehensive_msmarco_dataset(self) -> List[Dict]:
        """Generate comprehensive synthetic MSMARCO dataset for medical domain."""
        synthetic_data = []
        
        # Medical query templates by category
        medical_queries = {
            "symptoms": [
                ("What are the symptoms of diabetes?", "increased thirst, frequent urination, fatigue, blurred vision"),
                ("What causes chest pain?", "heart problems, lung issues, muscle strain, anxiety"),
                ("What are signs of heart attack?", "chest pain, shortness of breath, nausea, sweating"),
                ("What are symptoms of pneumonia?", "cough, fever, difficulty breathing, chest pain"),
                ("What causes headaches?", "tension, migraines, dehydration, eye strain"),
            ],
            "treatments": [
                ("How is diabetes treated?", "medication, diet changes, exercise, insulin"),
                ("What medications treat high blood pressure?", "ACE inhibitors, diuretics, beta blockers, calcium channel blockers"),
                ("How do you treat pneumonia?", "antibiotics, rest, fluids, oxygen therapy"),
                ("What is the treatment for depression?", "therapy, medication, lifestyle changes, support groups"),
                ("How is cancer treated?", "surgery, chemotherapy, radiation therapy, targeted therapy"),
            ],
            "causes": [
                ("What causes diabetes?", "genetics, obesity, lack of exercise, poor diet"),
                ("What causes high blood pressure?", "genetics, diet, stress, obesity, lack of exercise"),
                ("What causes heart disease?", "high cholesterol, smoking, diabetes, high blood pressure"),
                ("What causes stroke?", "blood clots, bleeding in brain, blocked arteries"),
                ("What causes kidney disease?", "diabetes, high blood pressure, genetic disorders"),
            ],
            "diagnosis": [
                ("How is diabetes diagnosed?", "blood tests, glucose tolerance test, HbA1c test"),
                ("How do doctors diagnose heart disease?", "ECG, stress test, cardiac catheterization, echocardiogram"),
                ("How is cancer diagnosed?", "biopsy, imaging tests, blood tests, physical examination"),
                ("How do you diagnose pneumonia?", "chest X-ray, blood tests, sputum culture"),
                ("How is depression diagnosed?", "clinical interview, questionnaires, physical exam"),
            ],
            "prevention": [
                ("How can you prevent diabetes?", "healthy diet, regular exercise, maintain healthy weight"),
                ("How do you prevent heart disease?", "exercise, healthy diet, don't smoke, manage stress"),
                ("How can cancer be prevented?", "healthy lifestyle, avoid tobacco, limit alcohol, regular screening"),
                ("How do you prevent stroke?", "control blood pressure, exercise, healthy diet, don't smoke"),
                ("How can you prevent kidney disease?", "control diabetes and blood pressure, healthy diet"),
            ],
            "medications": [
                ("What is metformin used for?", "type 2 diabetes treatment, lowers blood sugar"),
                ("What does aspirin do?", "reduces pain, inflammation, fever, prevents blood clots"),
                ("What is insulin?", "hormone that regulates blood sugar, used to treat diabetes"),
                ("What are antibiotics?", "medications that fight bacterial infections"),
                ("What do statins do?", "lower cholesterol, reduce heart disease risk"),
            ]
        }
        
        # Generate questions for each category
        question_id = 1
        for category, queries in medical_queries.items():
            for question, answer in queries:
                synthetic_data.append({
                    "id": f"msmarco_{question_id:04d}",
                    "question": question,
                    "answer": answer,
                    "context": f"Medical information about {category}: {answer}",
                    "query_type": "factoid",
                    "category": category,
                    "reasoning_type": "information_retrieval"
                })
                question_id += 1
        
        # Add more complex medical queries
        complex_queries = [
            ("What is the difference between type 1 and type 2 diabetes?", "Type 1 is autoimmune, Type 2 is insulin resistance"),
            ("When should you see a doctor for chest pain?", "Severe pain, shortness of breath, nausea, sweating"),
            ("What are the side effects of chemotherapy?", "Nausea, fatigue, hair loss, increased infection risk"),
            ("How long does it take to recover from surgery?", "Depends on type of surgery, typically weeks to months"),
            ("What lifestyle changes help with high blood pressure?", "Exercise, diet, weight loss, stress management"),
        ]
        
        for question, answer in complex_queries:
            synthetic_data.append({
                "id": f"msmarco_{question_id:04d}",
                "question": question,
                "answer": answer,
                "context": f"Medical guidance: {answer}",
                "query_type": "description",
                "reasoning_type": "information_retrieval"
            })
            question_id += 1
        
        # Generate additional questions to reach 1000 total
        categories = list(medical_queries.keys())
        while len(synthetic_data) < 1000:
            category = categories[question_id % len(categories)]
            
            synthetic_data.append({
                "id": f"msmarco_{question_id:04d}",
                "question": f"Medical query {question_id} about {category} information?",
                "answer": f"Medical answer {question_id} providing {category} information.",
                "context": f"Medical passage {question_id} containing relevant {category} information.",
                "query_type": "factoid",
                "category": category,
                "reasoning_type": "information_retrieval"
            })
            question_id += 1
        
        logger.info(f"✅ Generated {len(synthetic_data)} comprehensive MSMARCO questions")
        return synthetic_data
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate single MSMARCO response focusing on information retrieval quality."""
        
        expected_answer = question.get("answer", "")
        question_id = question.get("id", "unknown")
        
        # Handle empty responses
        if not response or not response.strip():
            return {
                "question_id": question_id,
                "score": 0.0,
                "correct": False,
                "metrics": {
                    "answer_relevance": 0.0,
                    "information_completeness": 0.0,
                    "retrieval_quality": 0.0,
                    "overall_score": 0.0
                },
                "response": response or "",
                "expected": expected_answer,
                "error": "Empty response"
            }
        
        try:
            # Calculate retrieval-focused metrics
            answer_relevance = self._assess_answer_relevance(response, expected_answer)
            information_completeness = self._assess_information_completeness(response, question)
            retrieval_quality = self._assess_retrieval_quality(response, retrieved_docs)
            
            # Overall score weighted for information retrieval
            overall_score = (
                answer_relevance * 0.5 +
                information_completeness * 0.3 +
                retrieval_quality * 0.2
            )
            
            return {
                "question_id": question_id,
                "score": overall_score * 100,
                "correct": overall_score > 0.6,  # 60% threshold for MSMARCO
                "metrics": {
                    "answer_relevance": answer_relevance,
                    "information_completeness": information_completeness,
                    "retrieval_quality": retrieval_quality,
                    "overall_score": overall_score
                },
                "response": response,
                "expected": expected_answer
            }
            
        except Exception as e:
            logger.error(f"Error evaluating MSMARCO response for {question_id}: {e}")
            return {
                "question_id": question_id,
                "score": 0.0,
                "correct": False,
                "metrics": {
                    "answer_relevance": 0.0,
                    "information_completeness": 0.0,
                    "retrieval_quality": 0.0,
                    "overall_score": 0.0
                },
                "response": response,
                "expected": expected_answer,
                "error": str(e)
            }
    
    def _assess_answer_relevance(self, response: str, expected: str) -> float:
        """Assess how relevant the response is to the expected answer."""
        if not response:
            return 0.0
        if not expected:
            return 0.5  # Give some credit if no expected answer
        
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        # Word overlap scoring
        response_words = set(response_lower.split())
        expected_words = set(expected_lower.split())
        
        if not expected_words:
            return 0.5
        if not response_words:
            return 0.0
        
        # Calculate multiple similarity metrics
        intersection = len(response_words.intersection(expected_words))
        union = len(response_words.union(expected_words))
        
        # Jaccard similarity
        jaccard = intersection / union if union > 0 else 0.0
        
        # Overlap ratio
        overlap_ratio = intersection / len(expected_words)
        
        # Take the better score
        return max(jaccard, overlap_ratio * 0.7)
    
    def _assess_information_completeness(self, response: str, question: Dict) -> float:
        """Assess how complete the information is in the response."""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        
        # Information completeness indicators
        completeness_indicators = [
            "treatment", "cause", "symptom", "diagnosis", "prevention",
            "medication", "therapy", "management", "risk", "effect"
        ]
        
        # Check for medical information richness
        medical_terms = [
            "medical", "clinical", "health", "disease", "condition",
            "patient", "doctor", "hospital", "drug", "medicine"
        ]
        
        # Score based on information content
        indicator_count = sum(1 for indicator in completeness_indicators if indicator in response_lower)
        medical_count = sum(1 for term in medical_terms if term in response_lower)
        
        # Length bonus for comprehensive answers
        length_score = min(len(response.split()) / 20.0, 0.3)  # Up to 30% for length
        
        completeness_score = (
            min(indicator_count / 3.0, 0.4) +  # Up to 40% for content indicators
            min(medical_count / 3.0, 0.3) +    # Up to 30% for medical terms  
            length_score                        # Up to 30% for length
        )
        
        return min(completeness_score, 1.0)
    
    def _assess_retrieval_quality(self, response: str, retrieved_docs: List[Dict]) -> float:
        """Assess how well the response uses retrieved information."""
        if not response:
            return 0.0
        
        base_score = 0.3  # Base score for having a response
        
        if not retrieved_docs:
            return base_score
        
        response_lower = response.lower()
        retrieval_score = 0.0
        
        # Check integration with retrieved documents
        for doc in retrieved_docs[:5]:  # Check top 5 documents
            doc_content = doc.get('content', '') or doc.get('text', '')
            if doc_content:
                doc_lower = doc_content.lower()
                
                # Calculate word overlap
                doc_words = set(word for word in doc_lower.split() if len(word) > 3)
                response_words = set(word for word in response_lower.split() if len(word) > 3)
                
                if doc_words and response_words:
                    overlap = len(response_words.intersection(doc_words))
                    if overlap > 0:
                        doc_score = min(overlap / 10.0, 0.1)  # Up to 10% per doc
                        retrieval_score += doc_score
        
        return min(base_score + retrieval_score, 1.0)