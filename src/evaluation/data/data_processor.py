"""
Data processor for HierRAGMed evaluation system.
Processes and normalizes benchmark data for consistent evaluation.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from loguru import logger
import pandas as pd
import numpy as np
from collections import defaultdict


class DataProcessor:
    """Process and normalize benchmark data for evaluation."""
    
    def __init__(self, config: Dict):
        """Initialize data processor."""
        self.config = config
        self.processing_config = config.get("data_processing", {})
        
        # Text processing settings
        self.max_context_length = self.processing_config.get("max_context_length", 2048)
        self.max_question_length = self.processing_config.get("max_question_length", 512)
        self.normalize_text = self.processing_config.get("normalize_text", True)
        self.remove_duplicates = self.processing_config.get("remove_duplicates", True)
        
        # Medical text patterns
        self.medical_abbreviations = {
            "MI": "myocardial infarction",
            "HTN": "hypertension", 
            "DM": "diabetes mellitus",
            "COPD": "chronic obstructive pulmonary disease",
            "CHF": "congestive heart failure",
            "CAD": "coronary artery disease",
            "CVA": "cerebrovascular accident",
            "PE": "pulmonary embolism",
            "DVT": "deep vein thrombosis",
            "UTI": "urinary tract infection"
        }
        
        # Question type patterns
        self.question_type_patterns = {
            "diagnosis": ["diagnos", "condition", "disease", "disorder"],
            "treatment": ["treat", "therap", "medication", "drug", "management"],
            "prognosis": ["prognos", "outcome", "survival", "mortality"],
            "etiology": ["cause", "etiology", "risk factor", "pathogenesis"],
            "symptoms": ["symptom", "sign", "presentation", "manifest"],
            "pharmacology": ["drug", "medication", "pharmacol", "dose", "side effect"]
        }
    
    def process_benchmark_data(self, benchmark_name: str, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw benchmark data into standardized format."""
        logger.info(f"🔄 Processing {benchmark_name} data...")
        logger.info(f"   Raw samples: {len(raw_data)}")
        
        # Apply benchmark-specific processing
        if benchmark_name == "mirage":
            processed_data = self._process_mirage_data(raw_data)
        elif benchmark_name == "medreason":
            processed_data = self._process_medreason_data(raw_data)
        elif benchmark_name == "pubmedqa":
            processed_data = self._process_pubmedqa_data(raw_data)
        elif benchmark_name == "msmarco":
            processed_data = self._process_msmarco_data(raw_data)
        else:
            # Generic processing
            processed_data = self._process_generic_data(raw_data)
        
        # Apply common processing steps
        processed_data = self._apply_common_processing(processed_data)
        
        # Remove duplicates if enabled
        if self.remove_duplicates:
            processed_data = self._remove_duplicates(processed_data)
        
        # Add metadata
        processed_data = self._add_processing_metadata(processed_data, benchmark_name)
        
        logger.info(f"   ✅ Processed samples: {len(processed_data)}")
        return processed_data
    
    def _process_mirage_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process MIRAGE benchmark data."""
        processed_data = []
        
        for item in raw_data:
            processed_item = {
                "question_id": item.get("question_id", ""),
                "question": self._clean_text(item.get("question", "")),
                "context": self._clean_text(item.get("context", "")),
                "answer": self._clean_text(item.get("answer", "")),
                "options": [self._clean_text(opt) for opt in item.get("options", [])],
                "question_type": self._classify_question_type(item.get("question", "")),
                "medical_specialty": item.get("medical_specialty", "general"),
                "reasoning_type": item.get("reasoning_type", "clinical"),
                "difficulty": self._assess_difficulty(item),
                "benchmark": "mirage"
            }
            processed_data.append(processed_item)
        
        return processed_data
    
    def _process_medreason_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process MedReason benchmark data."""
        processed_data = []
        
        for item in raw_data:
            # Extract reasoning chain if available
            reasoning_chain = item.get("reasoning_chain", [])
            if isinstance(reasoning_chain, str):
                reasoning_chain = self._parse_reasoning_chain(reasoning_chain)
            
            processed_item = {
                "question_id": item.get("question_id", ""),
                "question": self._clean_text(item.get("question", "")),
                "context": self._clean_text(item.get("context", "")),
                "answer": self._clean_text(item.get("answer", "")),
                "reasoning_chain": reasoning_chain,
                "question_type": self._classify_question_type(item.get("question", "")),
                "medical_specialty": item.get("medical_specialty", "general"),
                "reasoning_type": "knowledge_graph_guided",
                "complexity": len(reasoning_chain) if reasoning_chain else 1,
                "benchmark": "medreason"
            }
            processed_data.append(processed_item)
        
        return processed_data
    
    def _process_pubmedqa_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process PubMedQA benchmark data."""
        processed_data = []
        
        for item in raw_data:
            processed_item = {
                "question_id": item.get("question_id", ""),
                "question": self._clean_text(item.get("question", "")),
                "context": self._clean_text(item.get("context", "")),
                "answer": self._normalize_yes_no_answer(item.get("answer", "")),
                "long_answer": self._clean_text(item.get("long_answer", "")),
                "question_type": "yes_no_question",
                "medical_specialty": item.get("medical_specialty", "biomedical_research"),
                "reasoning_type": "evidence_based",
                "evidence_quality": self._assess_evidence_quality(item.get("context", "")),
                "benchmark": "pubmedqa"
            }
            processed_data.append(processed_item)
        
        return processed_data
    
    def _process_msmarco_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process MS MARCO benchmark data."""
        processed_data = []
        
        for item in raw_data:
            processed_item = {
                "question_id": item.get("question_id", ""),
                "question": self._clean_text(item.get("question", "")),
                "context": self._clean_text(item.get("context", "")),
                "answer": self._clean_text(item.get("answer", "")),
                "question_type": self._classify_question_type(item.get("question", "")),
                "medical_specialty": item.get("medical_specialty", "general"),
                "reasoning_type": "information_retrieval",
                "context_relevance": self._assess_context_relevance(
                    item.get("question", ""), item.get("context", "")
                ),
                "benchmark": "msmarco"
            }
            processed_data.append(processed_item)
        
        return processed_data
    
    def _process_generic_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generic processing for unknown benchmark formats."""
        processed_data = []
        
        for item in raw_data:
            processed_item = {
                "question_id": item.get("question_id", item.get("id", f"generic_{len(processed_data)}")),
                "question": self._clean_text(item.get("question", item.get("query", ""))),
                "context": self._clean_text(item.get("context", item.get("passage", ""))),
                "answer": self._clean_text(item.get("answer", item.get("response", ""))),
                "question_type": self._classify_question_type(item.get("question", "")),
                "medical_specialty": "general",
                "reasoning_type": "generic",
                "benchmark": "custom"
            }
            processed_data.append(processed_item)
        
        return processed_data
    
    def _apply_common_processing(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply common processing steps to all data."""
        processed_data = []
        
        for item in data:
            # Expand medical abbreviations
            if self.normalize_text:
                item["question"] = self._expand_medical_abbreviations(item["question"])
                item["context"] = self._expand_medical_abbreviations(item["context"])
                item["answer"] = self._expand_medical_abbreviations(item["answer"])
            
            # Truncate long texts
            item["question"] = self._truncate_text(item["question"], self.max_question_length)
            item["context"] = self._truncate_text(item["context"], self.max_context_length)
            
            # Skip empty questions
            if not item["question"].strip():
                continue
            
            processed_data.append(item)
        
        return processed_data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def _expand_medical_abbreviations(self, text: str) -> str:
        """Expand common medical abbreviations."""
        for abbrev, full_form in self.medical_abbreviations.items():
            # Match abbreviation as whole word
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
        
        return text
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text
        
        # Truncate at word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # Only truncate at word boundary if close
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of medical question."""
        question_lower = question.lower()
        
        for question_type, patterns in self.question_type_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                return question_type
        
        return "general"
    
    def _assess_difficulty(self, item: Dict[str, Any]) -> str:
        """Assess the difficulty level of a question."""
        question = item.get("question", "")
        context = item.get("context", "")
        
        # Simple heuristics for difficulty assessment
        complexity_indicators = [
            "differential diagnosis", "multiple", "complex", "rare", "syndrome",
            "pathophysiology", "mechanism", "multifactorial"
        ]
        
        combined_text = (question + " " + context).lower()
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in combined_text)
        
        if complexity_score >= 3:
            return "hard"
        elif complexity_score >= 1:
            return "medium"
        else:
            return "easy"
    
    def _assess_evidence_quality(self, context: str) -> str:
        """Assess the quality of evidence in context."""
        context_lower = context.lower()
        
        high_quality_indicators = [
            "randomized controlled trial", "meta-analysis", "systematic review",
            "peer-reviewed", "published", "journal"
        ]
        
        medium_quality_indicators = [
            "study", "research", "clinical trial", "evidence", "data"
        ]
        
        high_score = sum(1 for indicator in high_quality_indicators if indicator in context_lower)
        medium_score = sum(1 for indicator in medium_quality_indicators if indicator in context_lower)
        
        if high_score >= 1:
            return "high"
        elif medium_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _assess_context_relevance(self, question: str, context: str) -> float:
        """Assess how relevant the context is to the question."""
        if not question or not context:
            return 0.0
        
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        question_words -= stop_words
        context_words -= stop_words
        
        if not question_words:
            return 0.0
        
        overlap = len(question_words.intersection(context_words))
        relevance = overlap / len(question_words)
        
        return min(relevance, 1.0)
    
    def _normalize_yes_no_answer(self, answer: str) -> str:
        """Normalize yes/no answers."""
        answer_lower = answer.lower().strip()
        
        if answer_lower in ["yes", "y", "true", "1", "positive"]:
            return "yes"
        elif answer_lower in ["no", "n", "false", "0", "negative"]:
            return "no"
        else:
            return answer_lower
    
    def _parse_reasoning_chain(self, reasoning_text: str) -> List[Dict[str, Any]]:
        """Parse reasoning chain from text format."""
        if not reasoning_text:
            return []
        
        # Simple parsing - split by numbers or bullet points
        steps = re.split(r'\d+\.|•|-', reasoning_text)
        
        reasoning_chain = []
        for i, step in enumerate(steps):
            step = step.strip()
            if step:
                reasoning_chain.append({
                    "step": i + 1,
                    "description": step,
                    "type": "reasoning_step"
                })
        
        return reasoning_chain
    
    def _remove_duplicates(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate questions based on question text."""
        seen_questions = set()
        unique_data = []
        
        for item in data:
            question_key = item["question"].lower().strip()
            if question_key not in seen_questions:
                seen_questions.add(question_key)
                unique_data.append(item)
        
        duplicates_removed = len(data) - len(unique_data)
        if duplicates_removed > 0:
            logger.info(f"   🧹 Removed {duplicates_removed} duplicate questions")
        
        return unique_data
    
    def _add_processing_metadata(self, data: List[Dict[str, Any]], benchmark_name: str) -> List[Dict[str, Any]]:
        """Add processing metadata to each item."""
        for item in data:
            item["processing_metadata"] = {
                "processed_at": pd.Timestamp.now().isoformat(),
                "processor_version": "1.0.0",
                "benchmark_name": benchmark_name,
                "text_normalized": self.normalize_text,
                "max_context_length": self.max_context_length,
                "max_question_length": self.max_question_length
            }
        
        return data
    
    def get_processing_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate processing statistics for the dataset."""
        if not data:
            return {}
        
        stats = {
            "total_samples": len(data),
            "question_types": defaultdict(int),
            "medical_specialties": defaultdict(int),
            "reasoning_types": defaultdict(int),
            "difficulty_levels": defaultdict(int),
            "text_statistics": {
                "avg_question_length": 0,
                "avg_context_length": 0,
                "avg_answer_length": 0
            },
            "quality_metrics": {
                "complete_samples": 0,
                "samples_with_context": 0,
                "samples_with_reasoning": 0
            }
        }
        
        question_lengths = []
        context_lengths = []
        answer_lengths = []
        
        for item in data:
            # Count categories
            stats["question_types"][item.get("question_type", "unknown")] += 1
            stats["medical_specialties"][item.get("medical_specialty", "unknown")] += 1
            stats["reasoning_types"][item.get("reasoning_type", "unknown")] += 1
            stats["difficulty_levels"][item.get("difficulty", "unknown")] += 1
            
            # Collect text lengths
            question_lengths.append(len(item.get("question", "")))
            context_lengths.append(len(item.get("context", "")))
            answer_lengths.append(len(item.get("answer", "")))
            
            # Quality metrics
            if all([item.get("question"), item.get("answer")]):
                stats["quality_metrics"]["complete_samples"] += 1
            
            if item.get("context"):
                stats["quality_metrics"]["samples_with_context"] += 1
            
            if item.get("reasoning_chain") or item.get("reasoning_type") != "generic":
                stats["quality_metrics"]["samples_with_reasoning"] += 1
        
        # Calculate averages
        if question_lengths:
            stats["text_statistics"]["avg_question_length"] = np.mean(question_lengths)
            stats["text_statistics"]["avg_context_length"] = np.mean(context_lengths)
            stats["text_statistics"]["avg_answer_length"] = np.mean(answer_lengths)
        
        # Convert defaultdicts to regular dicts
        stats["question_types"] = dict(stats["question_types"])
        stats["medical_specialties"] = dict(stats["medical_specialties"])
        stats["reasoning_types"] = dict(stats["reasoning_types"])
        stats["difficulty_levels"] = dict(stats["difficulty_levels"])
        
        return stats
    
    def save_processed_data(self, data: List[Dict[str, Any]], output_path: Path) -> None:
        """Save processed data to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"   💾 Saved processed data to {output_path}")
    
    def load_processed_data(self, input_path: Path) -> List[Dict[str, Any]]:
        """Load processed data from file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"   📂 Loaded processed data from {input_path}")
        return data
    
    def create_data_splits(self, data: List[Dict[str, Any]], 
                          train_ratio: float = 0.7, 
                          val_ratio: float = 0.15, 
                          test_ratio: float = 0.15,
                          stratify_by: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Create train/validation/test splits from data."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        if stratify_by and stratify_by in data[0]:
            # Stratified split
            splits = self._create_stratified_splits(data, train_ratio, val_ratio, test_ratio, stratify_by)
        else:
            # Random split
            splits = self._create_random_splits(data, train_ratio, val_ratio, test_ratio)
        
        logger.info(f"   ✂️ Created data splits: train={len(splits['train'])}, "
                   f"val={len(splits['validation'])}, test={len(splits['test'])}")
        
        return splits
    
    def _create_random_splits(self, data: List[Dict[str, Any]], 
                            train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, List[Dict[str, Any]]]:
        """Create random train/validation/test splits."""
        np.random.seed(42)  # For reproducibility
        shuffled_data = data.copy()
        np.random.shuffle(shuffled_data)
        
        n_total = len(shuffled_data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            "train": shuffled_data[:n_train],
            "validation": shuffled_data[n_train:n_train + n_val],
            "test": shuffled_data[n_train + n_val:]
        }
        
        return splits
    
    def _create_stratified_splits(self, data: List[Dict[str, Any]], 
                                train_ratio: float, val_ratio: float, test_ratio: float,
                                stratify_by: str) -> Dict[str, List[Dict[str, Any]]]:
        """Create stratified splits based on a specific field."""
        # Group data by stratification field
        groups = defaultdict(list)
        for item in data:
            key = item.get(stratify_by, "unknown")
            groups[key].append(item)
        
        splits = {"train": [], "validation": [], "test": []}
        
        # Split each group proportionally
        for group_items in groups.values():
            np.random.seed(42)
            np.random.shuffle(group_items)
            
            n_total = len(group_items)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            splits["train"].extend(group_items[:n_train])
            splits["validation"].extend(group_items[n_train:n_train + n_val])
            splits["test"].extend(group_items[n_train + n_val:])
        
        # Shuffle the final splits
        for split in splits.values():
            np.random.shuffle(split)
        
        return splits