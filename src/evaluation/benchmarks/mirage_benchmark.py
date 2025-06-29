#!/usr/bin/env python3
"""
MIRAGE Benchmark - Simple Local File Only Version
Completely bypasses QADataset and only uses local benchmark.json
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

# DON'T import QADataset at all - only use local files
logger.info("📁 MIRAGE: Using local file parsing only (no QADataset)")


class MIRAGEBenchmark(BaseBenchmark):
    """
    MIRAGE benchmark using ONLY local submodule data.
    No QADataset - just loads from local mirage/benchmark.json.
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
        """Load MIRAGE dataset from local submodule ONLY."""
        logger.info("📊 Loading MIRAGE dataset from local submodule...")
        
        # Temporarily enable debug logging
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        
        # DIAGNOSTIC: Check what files actually exist
        self._diagnostic_check()
        
        # ONLY try local file parsing
        logger.info("🔍 Loading from local benchmark.json file...")
        result = self._load_from_local_file()
        
        # Reset logging level
        logging.getLogger().setLevel(logging.INFO)
        
        return result

    def _diagnostic_check(self):
        """Diagnostic check to see what files exist."""
        logger.info("🔍 DIAGNOSTIC: Checking MIRAGE submodule structure...")
        
        # Check current working directory
        import os
        logger.info(f"   Current working directory: {os.getcwd()}")
        
        # Check for mirage directory and files
        mirage_paths = [
            project_root / "mirage",
            Path("mirage"),
            Path("../mirage"),
        ]
        
        for path in mirage_paths:
            logger.info(f"   Checking path: {path}")
            if path.exists():
                logger.info(f"   ✅ Directory exists: {path}")
                # List files in the directory
                try:
                    files = list(path.iterdir())
                    logger.info(f"   📁 Files in {path}:")
                    for file in files[:10]:  # Show first 10 files
                        logger.info(f"      - {file.name}")
                    if len(files) > 10:
                        logger.info(f"      ... and {len(files) - 10} more files")
                    
                    # Specifically check for benchmark.json
                    benchmark_file = path / "benchmark.json"
                    if benchmark_file.exists():
                        logger.info(f"   ✅ benchmark.json found: {benchmark_file}")
                        logger.info(f"   📏 File size: {benchmark_file.stat().st_size} bytes")
                        
                        # Try to peek at the file structure
                        try:
                            with open(benchmark_file, 'r', encoding='utf-8') as f:
                                # Read first 1000 characters to see structure
                                preview = f.read(1000)
                                logger.info(f"   📖 File preview (first 1000 chars): {preview[:500]}...")
                        except Exception as e:
                            logger.info(f"   ⚠️ Could not read file: {e}")
                    else:
                        logger.info(f"   ❌ benchmark.json NOT found at: {benchmark_file}")
                        
                except Exception as e:
                    logger.info(f"   ⚠️ Error listing directory: {e}")
            else:
                logger.info(f"   ❌ Directory does not exist: {path}")
        
        logger.info("🔍 DIAGNOSTIC: End of structure check")

    def _load_from_local_file(self) -> List[Dict]:
        """Load MIRAGE data from local benchmark.json file."""
        logger.info("📁 Loading from local benchmark.json file...")
        
        # Try to find local MIRAGE submodule
        mirage_paths = [
            project_root / "mirage",  # Standard submodule location
            Path("mirage"),           # Current directory
            Path("../mirage"),        # Parent directory
        ]
        
        benchmark_file = None
        for path in mirage_paths:
            potential_file = path / "benchmark.json"
            if potential_file.exists():
                benchmark_file = potential_file
                break
        
        if not benchmark_file:
            logger.error("❌ MIRAGE benchmark.json not found in local submodule")
            logger.error("💡 Expected locations:")
            for path in mirage_paths:
                logger.error(f"   - {path / 'benchmark.json'}")
            return []
        
        try:
            logger.info(f"📁 Loading from: {benchmark_file}")
            
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"📊 Raw data type: {type(data)}")
            if isinstance(data, dict):
                logger.info(f"📊 Dict keys: {list(data.keys())}")
            elif isinstance(data, list):
                logger.info(f"📊 List length: {len(data)}")
            
            # Handle MIRAGE benchmark.json structure
            all_questions = []
            
            # MIRAGE benchmark.json typically has this structure:
            # {"dataset_name": [questions], "dataset_name2": [questions], ...}
            if isinstance(data, dict):
                for dataset_name, questions in data.items():
                    logger.info(f"   📖 Processing key '{dataset_name}': type={type(questions)}")
                    if isinstance(questions, list):
                        logger.info(f"   📖 Processing {dataset_name}: {len(questions)} questions")
                        for i, item in enumerate(questions):
                            formatted_item = self._format_mirage_question(item, f"{dataset_name}_{i}", dataset_name)
                            if formatted_item:
                                all_questions.append(formatted_item)
                    elif isinstance(questions, dict):
                        # Sometimes questions are nested deeper
                        logger.info(f"   📖 {dataset_name} is a dict with keys: {list(questions.keys())[:10]}...")
                        
                        # Process each individual question
                        for sub_key, question_data in questions.items():
                            # Each sub_key is like 'anatomy-000', 'anatomy-001', etc.
                            # and question_data is the actual question object
                            logger.debug(f"   🔍 Processing {dataset_name}.{sub_key}: type={type(question_data)}")
                            
                            if isinstance(question_data, dict):
                                # This is a single question object
                                formatted_item = self._format_mirage_question(question_data, f"{dataset_name}_{sub_key}", dataset_name)
                                if formatted_item:
                                    all_questions.append(formatted_item)
                                else:
                                    logger.debug(f"   ⚠️ Failed to format {dataset_name}.{sub_key}")
                            elif isinstance(question_data, list):
                                # Sometimes it might be a list of questions
                                logger.info(f"   📖 Processing {dataset_name}.{sub_key}: {len(question_data)} questions")
                                for i, item in enumerate(question_data):
                                    formatted_item = self._format_mirage_question(item, f"{dataset_name}_{sub_key}_{i}", f"{dataset_name}_{sub_key}")
                                    if formatted_item:
                                        all_questions.append(formatted_item)
                            else:
                                logger.debug(f"   ⚠️ Skipping {dataset_name}.{sub_key}: unexpected type {type(question_data)}")
                    else:
                        logger.debug(f"   ⚠️ Skipping {dataset_name}: not a list or dict (type: {type(questions)})")
                        
            elif isinstance(data, list):
                # If it's just a list of questions
                logger.info(f"   📖 Processing direct list: {len(data)} questions")
                for i, item in enumerate(data):
                    formatted_item = self._format_mirage_question(item, f"mirage_{i}", "mirage")
                    if formatted_item:
                        all_questions.append(formatted_item)
            else:
                logger.error(f"❌ Unexpected data format in {benchmark_file}: {type(data)}")
                return []
            
            logger.info(f"✅ Loaded {len(all_questions)} MIRAGE questions from local file")
            return all_questions
            
        except Exception as e:
            logger.error(f"❌ Failed to load MIRAGE data: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return []

    def load_test_data(self) -> List[Dict]:
        """Load test data for evaluation (compatibility method)."""
        return self.get_questions()

    def _format_mirage_question(self, item: Dict, question_id: str, dataset_name: str) -> Optional[Dict]:
        """Format a single MIRAGE question to standard format."""
        try:
            logger.debug(f"🔍 Formatting question {question_id}: {type(item)}")
            
            # Handle various MIRAGE question formats
            if isinstance(item, str):
                # Sometimes questions are just strings
                return {
                    'id': question_id,
                    'question': item,
                    'options': [],
                    'correct_answer': '',
                    'explanation': '',
                    'medical_specialty': self._classify_medical_specialty(item),
                    'question_type': 'knowledge_retrieval',
                    'source_dataset': dataset_name,
                    'reasoning_type': 'medical_reasoning'
                }
            
            if not isinstance(item, dict):
                logger.debug(f"⚠️ Skipping non-dict item: {type(item)}")
                return None
            
            logger.debug(f"🔍 Question {question_id} keys: {list(item.keys())}")
            
            # Extract question components with flexible field names
            question_text = (item.get('question') or 
                           item.get('query') or 
                           item.get('text') or 
                           item.get('prompt') or 
                           item.get('input') or '')
            
            # Handle options in various formats
            options = []
            if 'options' in item:
                options = item['options']
            elif 'choices' in item:
                options = item['choices']
            elif 'answers' in item:
                options = item['answers']
            else:
                # Look for A, B, C, D options
                for key in ['A', 'B', 'C', 'D', 'E']:
                    if key in item:
                        options.append(item[key])
            
            # Convert options to list if it's a dict
            if isinstance(options, dict):
                options = list(options.values())
            
            correct_answer = (item.get('answer') or 
                            item.get('correct_answer') or 
                            item.get('label') or 
                            item.get('target') or 
                            item.get('output') or '')
            
            explanation = (item.get('explanation') or 
                         item.get('rationale') or 
                         item.get('reasoning') or 
                         item.get('solution') or '')
            
            logger.debug(f"🔍 Question {question_id}: question_text='{question_text[:50]}...', options={len(options)}, answer='{correct_answer}'")
            
            # Skip items without essential fields
            if not question_text:
                logger.debug(f"⚠️ Skipping item without question text: {list(item.keys())}")
                return None
            
            # Determine medical specialty and question type
            medical_specialty = self._classify_medical_specialty(question_text)
            question_type = self._classify_question_type(question_text, options)
            
            formatted_question = {
                'id': question_id,
                'question': question_text,
                'options': options,
                'correct_answer': correct_answer,
                'explanation': explanation,
                'medical_specialty': medical_specialty,
                'question_type': question_type,
                'source_dataset': dataset_name,
                'reasoning_type': 'medical_reasoning'
            }
            
            logger.debug(f"✅ Successfully formatted question {question_id}")
            return formatted_question
            
        except Exception as e:
            logger.warning(f"❌ Failed to format MIRAGE question {question_id}: {e}")
            logger.debug(f"   Item: {item}")
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