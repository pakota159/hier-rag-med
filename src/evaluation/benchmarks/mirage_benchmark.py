#!/usr/bin/env python3
"""
MIRAGE Benchmark with Local Submodule Support
Uses local MIRAGE submodule instead of downloading online
"""

import re
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Set, Optional

from .base_benchmark import BaseBenchmark
from loguru import logger

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class MIRAGEBenchmark(BaseBenchmark):
    """
    MIRAGE benchmark using local submodule data.
    Loads from local mirage/benchmark.json instead of downloading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MIRAGE"
        
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

    def load_dataset(self) -> List[Dict]:
        """Load MIRAGE dataset from local submodule."""
        logger.info("📊 Loading MIRAGE dataset from local submodule...")
        
        # Try to find local MIRAGE submodule
        mirage_paths = [
            project_root / "mirage",  # Standard submodule location
            Path("mirage"),           # Current directory
            Path("../mirage"),        # Parent directory
        ]
        
        mirage_dir = None
        benchmark_file = None
        
        for path in mirage_paths:
            potential_file = path / "benchmark.json"
            if potential_file.exists():
                mirage_dir = path
                benchmark_file = potential_file
                break
        
        if not benchmark_file:
            logger.error("❌ MIRAGE benchmark.json not found in local submodule")
            logger.error("💡 Expected locations:")
            for path in mirage_paths:
                logger.error(f"   - {path / 'benchmark.json'}")
            logger.error("🔧 To fix:")
            logger.error("   cd mirage && wget https://github.com/Teddy-XiongGZ/MIRAGE/raw/main/benchmark.json")
            return []
        
        try:
            logger.info(f"📁 Loading from: {benchmark_file}")
            
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                # Handle different JSON structures
                questions = []
                if "questions" in data:
                    questions = data["questions"]
                elif "data" in data:
                    questions = data["data"]
                else:
                    # Assume it's a flat structure of questions
                    questions = [data]
            elif isinstance(data, list):
                questions = data
            else:
                logger.error(f"❌ Unexpected data format in {benchmark_file}")
                return []
            
            # Process and format questions
            formatted_questions = []
            for i, item in enumerate(questions):
                formatted_item = self._format_mirage_question(item, i)
                if formatted_item:
                    formatted_questions.append(formatted_item)
            
            logger.info(f"✅ Loaded {len(formatted_questions)} MIRAGE questions from local file")
            return formatted_questions
            
        except Exception as e:
            logger.error(f"❌ Failed to load MIRAGE data: {e}")
            return []

    def load_test_data(self) -> List[Dict]:
        """Load test data for evaluation (compatibility method)."""
        return self.get_questions()

    def _format_mirage_question(self, item: Dict, index: int) -> Optional[Dict]:
        """Format a single MIRAGE question to standard format."""
        try:
            # Extract question components with flexible field names
            question_text = (item.get('question') or 
                           item.get('query') or 
                           item.get('text') or '')
            
            options = (item.get('options') or 
                      item.get('choices') or 
                      item.get('answers') or [])
            
            correct_answer = (item.get('answer') or 
                            item.get('correct_answer') or 
                            item.get('label') or '')
            
            explanation = (item.get('explanation') or 
                         item.get('rationale') or 
                         item.get('reasoning') or '')
            
            # Skip items without essential fields
            if not question_text:
                return None
            
            # Determine medical specialty
            medical_specialty = self._classify_medical_specialty(question_text)
            
            # Determine question type
            question_type = self._classify_question_type(question_text, options)
            
            formatted_question = {
                'id': item.get('id', f"mirage_{index}"),
                'question': question_text,
                'options': options,
                'correct_answer': correct_answer,
                'explanation': explanation,
                'medical_specialty': medical_specialty,
                'question_type': question_type,
                'source_dataset': item.get('source', 'mirage'),
                'reasoning_type': 'medical_reasoning'
            }
            
            return formatted_question
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to format question {index}: {e}")
            return None

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
        if any(pattern in question_lower for pattern in ['diagnosis', 'most likely', 'what is the']):
            return 'diagnostic'
        elif any(pattern in question_lower for pattern in ['treatment', 'therapy', 'management']):
            return 'treatment'
        elif any(pattern in question_lower for pattern in ['mechanism', 'pathophysiology', 'how does']):
            return 'pathophysiology'
        elif any(pattern in question_lower for pattern in ['drug', 'medication', 'pharmacology']):
            return 'pharmacology'
        
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