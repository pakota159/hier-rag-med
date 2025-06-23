#!/usr/bin/env python3
"""
CORRECTED MIRAGE Benchmark with Official QADataset Integration
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
    logger.info("✅ Successfully imported QADataset from MIRAGE")
except ImportError as e:
    MIRAGE_UTILS_AVAILABLE = False
    logger.warning(f"⚠️ QADataset from MIRAGE not available: {e} - using fallback data loader")

from src.evaluation.data.data_loader import BenchmarkDataLoader


class MIRAGEBenchmark(BaseBenchmark):
    """
    MIRAGE benchmark with official QADataset integration.
    Uses the official QADataset from MIRAGE repository when available.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MIRAGE"
        self.data_loader = BenchmarkDataLoader(config)
        
        # Initialize evaluation models
        self._init_evaluation_models()
        
        # Medical specialties mapping for analysis
        self.medical_specialties = {
            'cardiology': ['heart', 'cardiac', 'cardiovascular', 'myocardial'],
            'neurology': ['brain', 'neural', 'neurological', 'cognitive'],
            'oncology': ['cancer', 'tumor', 'malignant', 'chemotherapy'],
            'infectious_disease': ['infection', 'bacterial', 'viral', 'antibiotic'],
            'endocrinology': ['diabetes', 'thyroid', 'hormone', 'endocrine'],
            'gastroenterology': ['digestive', 'stomach', 'intestinal', 'liver'],
            'pulmonology': ['lung', 'respiratory', 'breathing', 'pulmonary'],
            'psychiatry': ['mental', 'psychological', 'psychiatric', 'depression'],
            'dermatology': ['skin', 'dermal', 'dermatological', 'rash'],
            'orthopedics': ['bone', 'joint', 'orthopedic', 'fracture']
        }
        
        # Question type patterns
        self.question_patterns = {
            'multiple_choice': ['A)', 'B)', 'C)', 'D)', 'E)'],
            'diagnostic': ['diagnosis', 'most likely', 'what is the'],
            'treatment': ['treatment', 'therapy', 'management'],
            'pathophysiology': ['mechanism', 'pathophysiology', 'how does'],
            'pharmacology': ['drug', 'medication', 'pharmacology']
        }

    def load_dataset(self) -> List[Dict]:
        """Load MIRAGE dataset using official QADataset when available."""
        logger.info("📊 Loading MIRAGE dataset...")
        
        # Try official QADataset first
        if MIRAGE_UTILS_AVAILABLE:
            try:
                return self._load_with_official_qadataset()
            except Exception as e:
                logger.warning(f"⚠️ Official QADataset failed: {e}. Using fallback method.")
        
        # Fallback to data loader
        return self._load_with_data_loader()

    def load_test_data(self) -> List[Dict]:
        """Load test data for evaluation (compatibility method)."""
        return self.get_questions()

    def _load_with_official_qadataset(self) -> List[Dict]:
        """Load MIRAGE data using official QADataset."""
        logger.info("📊 Using official MIRAGE QADataset...")
        
        # Detect MIRAGE directory
        mirage_paths = [
            Path("mirage"),
            Path("../mirage"),
            Path("../../mirage"),
            project_root / "mirage"
        ]
        
        mirage_dir = None
        for path in mirage_paths:
            if path.exists() and (path / "src" / "utils.py").exists():
                mirage_dir = path
                break
        
        if not mirage_dir:
            raise FileNotFoundError("MIRAGE directory not found")
        
        # Initialize QADataset
        qa_dataset = QADataset()
        
        # Load all MIRAGE datasets
        all_questions = []
        datasets = ['mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq']
        
        for dataset_name in datasets:
            try:
                data = qa_dataset.load_data(dataset_name)
                formatted_data = self._format_qadataset_questions(data, dataset_name)
                all_questions.extend(formatted_data)
                logger.info(f"   ✅ Loaded {len(formatted_data)} questions from {dataset_name}")
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to load {dataset_name}: {e}")
        
        logger.info(f"📊 Total MIRAGE questions loaded: {len(all_questions)}")
        return all_questions

    def _load_with_data_loader(self) -> List[Dict]:
        """Load MIRAGE data using fallback data loader."""
        logger.info("📊 Using fallback MIRAGE data loader...")
        return self.data_loader.load_benchmark_data("mirage", max_samples=self.sample_size)

    def _format_qadataset_questions(self, data: List[Dict], source_dataset: str) -> List[Dict]:
        """Format questions from QADataset to standard format."""
        formatted_questions = []
        
        for i, item in enumerate(data):
            # Extract question components
            question_text = item.get('question', item.get('query', ''))
            options = item.get('options', item.get('choices', []))
            correct_answer = item.get('answer', item.get('correct_answer', ''))
            explanation = item.get('explanation', item.get('rationale', ''))
            
            # Determine medical specialty
            medical_specialty = self._classify_medical_specialty(question_text)
            
            # Determine question type
            question_type = self._classify_question_type(question_text, options)
            
            formatted_question = {
                'id': f"{source_dataset}_{i}",
                'question': question_text,
                'options': options,
                'correct_answer': correct_answer,
                'explanation': explanation,
                'medical_specialty': medical_specialty,
                'question_type': question_type,
                'source_dataset': source_dataset,
                'reasoning_type': 'medical_reasoning'
            }
            
            formatted_questions.append(formatted_question)
        
        return formatted_questions

    def _classify_medical_specialty(self, question_text: str) -> str:
        """Classify question into medical specialty based on content."""
        question_lower = question_text.lower()
        
        for specialty, keywords in self.medical_specialties.items():
            if any(keyword in question_lower for keyword in keywords):
                return specialty
        
        return 'general_medicine'

    def _classify_question_type(self, question_text: str, options: List[str]) -> str:
        """Classify question type based on structure and content."""
        question_lower = question_text.lower()
        
        # Check for multiple choice
        if options and len(options) > 1:
            return 'multiple_choice'
        
        # Check for specific question patterns
        for q_type, patterns in self.question_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                return q_type
        
        return 'knowledge_retrieval'

    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """Evaluate model response against MIRAGE ground truth."""
        try:
            correct_answer = question.get('correct_answer', '')
            question_type = question.get('question_type', 'multiple_choice')
            
            # Extract answer from response
            predicted_answer = self._extract_answer(response, question.get('options', []))
            
            # Calculate basic accuracy
            is_correct = self._compare_answers(predicted_answer, correct_answer, question_type)
            
            # Additional metrics
            confidence_score = self._calculate_confidence(response)
            reasoning_quality = self._assess_reasoning_quality(response, question.get('explanation', ''))
            
            result = {
                'question_id': question.get('id', ''),
                'predicted_answer': predicted_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'accuracy': 1.0 if is_correct else 0.0,
                'confidence_score': confidence_score,
                'reasoning_quality': reasoning_quality,
                'question_type': question_type,
                'medical_specialty': question.get('medical_specialty', 'general'),
                'overall_score': 1.0 if is_correct else 0.0,
                'retrieval_docs_count': len(retrieved_docs)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating MIRAGE response: {e}")
            return {
                'question_id': question.get('id', ''),
                'error': str(e),
                'accuracy': 0.0,
                'overall_score': 0.0,
                'is_correct': False
            }

    def _extract_answer(self, response: str, options: List[str]) -> str:
        """Extract answer from model response."""
        response_clean = response.strip().upper()
        
        # Look for explicit answer patterns
        patterns = [
            r'(?:ANSWER|ANS)[:\s]*([A-E])',
            r'(?:THE ANSWER IS|ANSWER IS)[:\s]*([A-E])',
            r'\b([A-E])\)',
            r'^([A-E])[.:\s]',
            r'OPTION[:\s]*([A-E])'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_clean)
            if match:
                return match.group(1)
        
        # If no clear pattern, look for option letters at start of lines
        lines = response.split('\n')
        for line in lines:
            line_clean = line.strip().upper()
            if len(line_clean) >= 1 and line_clean[0] in 'ABCDE':
                return line_clean[0]
        
        # Last resort: return first found letter
        for char in response_clean:
            if char in 'ABCDE':
                return char
        
        return ''

    def _compare_answers(self, predicted: str, correct: str, question_type: str) -> bool:
        """Compare predicted and correct answers."""
        if not predicted or not correct:
            return False
        
        predicted_clean = predicted.strip().upper()
        correct_clean = correct.strip().upper()
        
        if question_type == 'multiple_choice':
            return predicted_clean == correct_clean
        else:
            # For non-MCQ, use semantic similarity if available
            if self.similarity_model:
                try:
                    pred_embedding = self.similarity_model.encode([predicted_clean])
                    correct_embedding = self.similarity_model.encode([correct_clean])
                    similarity = np.dot(pred_embedding[0], correct_embedding[0])
                    return similarity > 0.7
                except:
                    pass
            
            # Fallback to string matching
            return predicted_clean in correct_clean or correct_clean in predicted_clean

    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score based on response characteristics."""
        confidence_indicators = {
            'high': ['definitely', 'certainly', 'clearly', 'obviously', 'without doubt'],
            'medium': ['likely', 'probably', 'most likely', 'appears to be'],
            'low': ['might', 'could be', 'possibly', 'unsure', 'not certain']
        }
        
        response_lower = response.lower()
        
        high_count = sum(1 for indicator in confidence_indicators['high'] if indicator in response_lower)
        medium_count = sum(1 for indicator in confidence_indicators['medium'] if indicator in response_lower)
        low_count = sum(1 for indicator in confidence_indicators['low'] if indicator in response_lower)
        
        if high_count > 0:
            return 0.9
        elif medium_count > 0:
            return 0.7
        elif low_count > 0:
            return 0.3
        else:
            return 0.5  # Default medium confidence

    def _assess_reasoning_quality(self, response: str, ground_truth_explanation: str) -> float:
        """Assess quality of reasoning in response."""
        if not ground_truth_explanation:
            return 0.5
        
        if self.similarity_model:
            try:
                response_embedding = self.similarity_model.encode([response])
                explanation_embedding = self.similarity_model.encode([ground_truth_explanation])
                similarity = np.dot(response_embedding[0], explanation_embedding[0])
                return float(similarity)
            except:
                pass
        
        # Fallback: keyword overlap
        response_words = set(response.lower().split())
        explanation_words = set(ground_truth_explanation.lower().split())
        
        if len(explanation_words) == 0:
            return 0.5
        
        overlap = len(response_words.intersection(explanation_words))
        return min(1.0, overlap / len(explanation_words))

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        
        import re
        # Remove extra whitespace and punctuation, convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _init_evaluation_models(self) -> None:
        """Initialize models needed for evaluation."""
        try:
            # Try to initialize sentence transformer for semantic similarity
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Initialized similarity model for MIRAGE evaluation")
        except ImportError:
            logger.warning("⚠️ sentence-transformers not available, using basic evaluation")
            self.similarity_model = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize similarity model: {e}")
            self.similarity_model = None