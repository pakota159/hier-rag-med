"""
Benchmark data loader for HierRAGMed evaluation system.
Loads and preprocesses benchmark datasets for evaluation.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from loguru import logger
import requests
import zipfile
import tempfile
from datasets import load_dataset
import yaml


class BenchmarkDataLoader:
    """Load benchmark datasets for medical RAG evaluation."""
    
    def __init__(self, config: Dict):
        """Initialize benchmark data loader."""
        self.config = config
        self.data_dir = Path(config.get("data_dir", "src/evaluation/data"))
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = {
            "mirage": {
                "source": "huggingface",
                "dataset_name": "mirage_benchmark",
                "url": "https://github.com/mirage-project/MIRAGE",
                "file_format": "json",
                "cache_file": "mirage_data.json"
            },
            "medreason": {
                "source": "huggingface", 
                "dataset_name": "UCSC-VLAA/MedReason",
                "url": "https://huggingface.co/datasets/UCSC-VLAA/MedReason",
                "file_format": "json",
                "cache_file": "medreason_data.json"
            },
            "pubmedqa": {
                "source": "huggingface",
                "dataset_name": "pubmed_qa",
                "url": "https://huggingface.co/datasets/pubmed_qa",
                "file_format": "json", 
                "cache_file": "pubmedqa_data.json"
            },
            "msmarco": {
                "source": "microsoft",
                "dataset_name": "ms_marco",
                "url": "https://microsoft.github.io/msmarco/",
                "file_format": "tsv",
                "cache_file": "msmarco_data.json"
            }
        }
    
    def load_benchmark_data(self, benchmark_name: str, split: str = "test", 
                          max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data for a specific benchmark."""
        logger.info(f"📚 Loading {benchmark_name} benchmark data...")
        
        if benchmark_name not in self.dataset_configs:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        config = self.dataset_configs[benchmark_name]
        cache_file = self.cache_dir / config["cache_file"]
        
        # Try to load from cache first
        if cache_file.exists():
            logger.info(f"   📂 Loading from cache: {cache_file}")
            return self._load_from_cache(cache_file, max_samples)
        
        # Load from source
        if benchmark_name == "mirage":
            data = self._load_mirage_data()
        elif benchmark_name == "medreason":
            data = self._load_medreason_data(split)
        elif benchmark_name == "pubmedqa":
            data = self._load_pubmedqa_data(split)
        elif benchmark_name == "msmarco":
            data = self._load_msmarco_data(split)
        else:
            raise ValueError(f"No loader implemented for {benchmark_name}")
        
        # Cache the loaded data
        self._save_to_cache(data, cache_file)
        
        # Apply sample limit
        if max_samples and len(data) > max_samples:
            data = data[:max_samples]
            logger.info(f"   ✂️ Limited to {max_samples} samples")
        
        logger.info(f"   ✅ Loaded {len(data)} samples for {benchmark_name}")
        return data
    
    def _load_mirage_data(self) -> List[Dict[str, Any]]:
        """Load MIRAGE benchmark data."""
        try:
            # MIRAGE dataset format
            data = []
            
            # Sample MIRAGE-style questions for demonstration
            # In real implementation, this would load from the actual MIRAGE dataset
            sample_data = [
                {
                    "question_id": "mirage_001",
                    "question": "What is the most likely diagnosis for a 45-year-old male with chest pain, sweating, and shortness of breath?",
                    "context": "Patient presents with acute onset chest pain radiating to left arm, diaphoresis, and dyspnea. ECG shows ST elevation in leads II, III, aVF.",
                    "answer": "Inferior myocardial infarction",
                    "options": ["Angina pectoris", "Inferior myocardial infarction", "Pulmonary embolism", "Aortic dissection"],
                    "reasoning_type": "clinical_diagnosis",
                    "medical_specialty": "cardiology"
                },
                {
                    "question_id": "mirage_002", 
                    "question": "Which medication is contraindicated in a patient with severe asthma?",
                    "context": "Patient has severe persistent asthma with frequent exacerbations requiring oral corticosteroids.",
                    "answer": "Beta-blockers",
                    "options": ["Beta-blockers", "ACE inhibitors", "Calcium channel blockers", "Diuretics"],
                    "reasoning_type": "pharmacology",
                    "medical_specialty": "pulmonology"
                }
            ]
            
            data.extend(sample_data)
            logger.info(f"   📋 Loaded {len(data)} MIRAGE samples")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load MIRAGE data: {e}")
            return []
    
    def _load_medreason_data(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load MedReason benchmark data."""
        try:
            # Load MedReason dataset from HuggingFace
            dataset = load_dataset("UCSC-VLAA/MedReason", split=split)
            
            data = []
            for item in dataset:
                formatted_item = {
                    "question_id": item.get("id", f"medreason_{len(data)}"),
                    "question": item.get("question", ""),
                    "context": item.get("context", ""),
                    "answer": item.get("answer", ""),
                    "reasoning_chain": item.get("reasoning_chain", []),
                    "reasoning_type": "knowledge_graph_guided",
                    "medical_specialty": item.get("specialty", "general")
                }
                data.append(formatted_item)
            
            logger.info(f"   📋 Loaded {len(data)} MedReason samples")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load MedReason from HuggingFace: {e}")
            # Fallback to sample data
            return self._get_sample_medreason_data()
    
    def _load_pubmedqa_data(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load PubMedQA benchmark data."""
        try:
            # Load PubMedQA dataset from HuggingFace
            dataset = load_dataset("pubmed_qa", "pqa_labeled", split=split)
            
            data = []
            for item in dataset:
                formatted_item = {
                    "question_id": item.get("pubid", f"pubmedqa_{len(data)}"),
                    "question": item.get("question", ""),
                    "context": " ".join(item.get("context", {}).get("contexts", [])),
                    "answer": item.get("final_decision", ""),
                    "long_answer": item.get("long_answer", ""),
                    "reasoning_type": "evidence_based",
                    "medical_specialty": "biomedical_research"
                }
                data.append(formatted_item)
            
            logger.info(f"   📋 Loaded {len(data)} PubMedQA samples")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load PubMedQA from HuggingFace: {e}")
            return self._get_sample_pubmedqa_data()
    
    def _load_msmarco_data(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load MS MARCO benchmark data."""
        try:
            # Load MS MARCO dataset
            dataset = load_dataset("ms_marco", "v1.1", split=split)
            
            data = []
            for item in dataset:
                if self._is_medical_query(item.get("query", "")):
                    formatted_item = {
                        "question_id": item.get("query_id", f"msmarco_{len(data)}"),
                        "question": item.get("query", ""),
                        "context": " ".join(item.get("passages", {}).get("passage_text", [])),
                        "answer": item.get("wellFormedAnswers", [""])[0] if item.get("wellFormedAnswers") else "",
                        "reasoning_type": "information_retrieval",
                        "medical_specialty": "general"
                    }
                    data.append(formatted_item)
            
            logger.info(f"   📋 Loaded {len(data)} MS MARCO medical samples")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load MS MARCO from HuggingFace: {e}")
            return self._get_sample_msmarco_data()
    
    def _is_medical_query(self, query: str) -> bool:
        """Check if a query is medical-related."""
        medical_keywords = [
            "disease", "symptom", "treatment", "diagnosis", "medicine", "drug",
            "patient", "health", "medical", "clinical", "therapy", "cancer",
            "diabetes", "heart", "blood", "doctor", "hospital", "pain"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in medical_keywords)
    
    def _get_sample_medreason_data(self) -> List[Dict[str, Any]]:
        """Get sample MedReason data for testing."""
        return [
            {
                "question_id": "medreason_sample_001",
                "question": "What is the reasoning chain for diagnosing Type 2 diabetes?",
                "context": "Patient presents with polyuria, polydipsia, and unexplained weight loss.",
                "answer": "Screen with fasting glucose → Confirm with HbA1c → Assess complications",
                "reasoning_chain": [
                    {"step": 1, "action": "Screen with fasting glucose", "rationale": "Initial screening test"},
                    {"step": 2, "action": "Confirm with HbA1c", "rationale": "Gold standard confirmation"},
                    {"step": 3, "action": "Assess complications", "rationale": "Comprehensive care"}
                ],
                "reasoning_type": "knowledge_graph_guided",
                "medical_specialty": "endocrinology"
            }
        ]
    
    def _get_sample_pubmedqa_data(self) -> List[Dict[str, Any]]:
        """Get sample PubMedQA data for testing."""
        return [
            {
                "question_id": "pubmedqa_sample_001",
                "question": "Does aspirin prevent cardiovascular events?",
                "context": "Multiple randomized controlled trials have shown aspirin reduces cardiovascular events in high-risk patients.",
                "answer": "yes",
                "long_answer": "Aspirin has been shown to reduce cardiovascular events in patients with high cardiovascular risk.",
                "reasoning_type": "evidence_based",
                "medical_specialty": "cardiology"
            }
        ]
    
    def _get_sample_msmarco_data(self) -> List[Dict[str, Any]]:
        """Get sample MS MARCO medical data for testing."""
        return [
            {
                "question_id": "msmarco_sample_001", 
                "question": "What are the symptoms of pneumonia?",
                "context": "Pneumonia symptoms include cough, fever, chills, shortness of breath, and chest pain.",
                "answer": "Pneumonia symptoms include cough, fever, chills, shortness of breath, and chest pain.",
                "reasoning_type": "information_retrieval",
                "medical_specialty": "pulmonology"
            }
        ]
    
    def _load_from_cache(self, cache_file: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data from cache file."""
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        if max_samples and len(data) > max_samples:
            data = data[:max_samples]
        
        return data
    
    def _save_to_cache(self, data: List[Dict[str, Any]], cache_file: Path) -> None:
        """Save data to cache file."""
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"   💾 Cached data to {cache_file}")
    
    def get_available_benchmarks(self) -> List[str]:
        """Get list of available benchmark datasets."""
        return list(self.dataset_configs.keys())
    
    def get_benchmark_info(self, benchmark_name: str) -> Dict[str, Any]:
        """Get information about a specific benchmark."""
        if benchmark_name not in self.dataset_configs:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        config = self.dataset_configs[benchmark_name].copy()
        cache_file = self.cache_dir / config["cache_file"]
        config["cached"] = cache_file.exists()
        
        if config["cached"]:
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                config["cached_samples"] = len(data)
            except:
                config["cached_samples"] = 0
        
        return config