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
    
    def _load_mirage_data(self) -> List[Dict[str, Any]]:
        """Load MIRAGE benchmark data with multiple fallback options."""
        try:
            # Try multiple potential MIRAGE sources
            sources = [
                ("mirage-project/MIRAGE", "test"),
                ("clinical-reasoning/mirage", "test"), 
                ("medical-rag/mirage", "test"),
                ("cmedcorp/MIRAGE", "test")
            ]
            
            for source, split in sources:
                try:
                    logger.info(f"   Trying MIRAGE source: {source}")
                    dataset = load_dataset(source, split=split)
                    
                    data = []
                    for item in dataset:
                        formatted_item = {
                            "question_id": item.get("id", f"mirage_{len(data)}"),
                            "question": item.get("question", ""),
                            "context": item.get("context", ""),
                            "answer": item.get("answer", ""),
                            "options": item.get("options", []),
                            "reasoning_type": item.get("reasoning_type", "clinical"),
                            "medical_specialty": item.get("medical_specialty", "general")
                        }
                        data.append(formatted_item)
                    
                    if len(data) > 0:
                        logger.info(f"   📋 Loaded {len(data)} MIRAGE samples from {source}")
                        return data
                        
                except Exception as e:
                    logger.debug(f"   Could not load MIRAGE from {source}: {e}")
                    continue
            
            # If all sources fail, return empty list to trigger fallback
            logger.warning("Could not load MIRAGE from any HuggingFace source")
            return []
            
        except Exception as e:
            logger.error(f"MIRAGE loading completely failed: {e}")
            return []

    def _load_medreason_data(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load MedReason benchmark data with improved error handling."""
        try:
            # Try the main MedReason dataset first
            try:
                logger.info("   Trying UCSC-VLAA/MedReason...")
                dataset = load_dataset("UCSC-VLAA/MedReason", split=split)
                
                data = []
                for item in dataset:
                    formatted_item = {
                        "question_id": item.get("id", f"medreason_{len(data)}"),
                        "question": item.get("question", ""),
                        "context": item.get("context", ""),
                        "answer": item.get("answer", ""),
                        "reasoning_chain": item.get("reasoning_chain", []),
                        "reasoning_type": item.get("reasoning_type", "knowledge_graph_guided"),
                        "medical_specialty": item.get("specialty", "general")
                    }
                    data.append(formatted_item)
                
                if len(data) > 0:
                    logger.info(f"   📋 Loaded {len(data)} MedReason samples")
                    return data
                    
            except Exception as e:
                logger.debug(f"   Could not load from UCSC-VLAA/MedReason: {e}")
            
            # Try alternative sources
            alternative_sources = [
                ("medical-reasoning/medreason", split),
                ("clinical-kg/med-reason", split),
                ("biomedical/med-reasoning", split)
            ]
            
            for source, split_name in alternative_sources:
                try:
                    logger.info(f"   Trying alternative MedReason source: {source}")
                    dataset = load_dataset(source, split=split_name)
                    
                    data = []
                    for item in dataset:
                        formatted_item = {
                            "question_id": item.get("id", f"medreason_alt_{len(data)}"),
                            "question": item.get("question", ""),
                            "context": item.get("context", ""),
                            "answer": item.get("answer", ""),
                            "reasoning_chain": item.get("reasoning_chain", []),
                            "reasoning_type": "knowledge_graph_guided",
                            "medical_specialty": item.get("specialty", "general")
                        }
                        data.append(formatted_item)
                    
                    if len(data) > 0:
                        logger.info(f"   📋 Loaded {len(data)} MedReason samples from {source}")
                        return data
                        
                except Exception as e:
                    logger.debug(f"   Could not load from {source}: {e}")
                    continue
            
            # Fallback to sample data
            logger.warning("Could not load MedReason from HuggingFace, using fallback")
            return self._get_sample_medreason_data()
            
        except Exception as e:
            logger.error(f"MedReason loading failed: {e}")
            return self._get_sample_medreason_data()

    def _load_pubmedqa_data(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load PubMedQA benchmark data with multiple configuration support."""
        try:
            # Try different PubMedQA configurations
            configurations = [
                ("pubmed_qa", "pqa_labeled"),
                ("pubmed_qa", "pqa_artificial"),
                ("pubmed_qa", "pqa_unlabeled"),
                ("biomedical-qa/pubmedqa", None)
            ]
            
            for dataset_name, config_name in configurations:
                try:
                    logger.info(f"   Trying PubMedQA: {dataset_name} {config_name}")
                    
                    if config_name:
                        dataset = load_dataset(dataset_name, config_name, split=split)
                    else:
                        dataset = load_dataset(dataset_name, split=split)
                    
                    data = []
                    for item in dataset:
                        # Handle context properly - it can be in different formats
                        context_data = item.get("context", {})
                        if isinstance(context_data, dict):
                            context = " ".join(context_data.get("contexts", []))
                        elif isinstance(context_data, list):
                            context = " ".join(context_data)
                        else:
                            context = str(context_data)
                        
                        formatted_item = {
                            "question_id": item.get("pubid", f"pubmedqa_{len(data)}"),
                            "question": item.get("question", ""),
                            "context": context,
                            "answer": item.get("final_decision", item.get("answer", "")),
                            "long_answer": item.get("long_answer", ""),
                            "reasoning_type": "evidence_based",
                            "medical_specialty": "biomedical_research"
                        }
                        data.append(formatted_item)
                    
                    if len(data) > 0:
                        logger.info(f"   📋 Loaded {len(data)} PubMedQA samples from {dataset_name}")
                        return data
                        
                except Exception as e:
                    logger.debug(f"   Could not load from {dataset_name} {config_name}: {e}")
                    continue
            
            # Fallback to sample data
            logger.warning("Could not load PubMedQA from HuggingFace, using fallback")
            return self._get_sample_pubmedqa_data()
            
        except Exception as e:
            logger.error(f"PubMedQA loading failed: {e}")
            return self._get_sample_pubmedqa_data()

    def _load_msmarco_data(self, split: str = "validation") -> List[Dict[str, Any]]:
        """Load MS MARCO benchmark data filtered for medical content."""
        try:
            # Try different MS MARCO configurations
            configurations = [
                ("ms_marco", "v1.1"),
                ("ms_marco", "v2.1"),
                ("microsoft/ms_marco", "v1.1")
            ]
            
            for dataset_name, config_name in configurations:
                try:
                    logger.info(f"   Trying MSMARCO: {dataset_name} {config_name}")
                    
                    # Load a subset for processing (MSMARCO is very large)
                    dataset = load_dataset(dataset_name, config_name, split=f"{split}[:10000]")
                    
                    data = []
                    medical_count = 0
                    
                    for item in dataset:
                        query = item.get("query", "")
                        
                        # Filter for medical content
                        if self._is_medical_query(query):
                            # Handle different MS MARCO formats
                            passages = item.get("passages", {})
                            if isinstance(passages, dict):
                                passage_texts = passages.get("passage_text", [])
                                is_selected = passages.get("is_selected", [])
                                # Get selected passages or first few
                                if is_selected:
                                    context_parts = [text for text, selected in zip(passage_texts, is_selected) if selected]
                                else:
                                    context_parts = passage_texts[:3]
                                context = " ".join(context_parts)
                            else:
                                context = str(passages)
                            
                            # Get answers
                            answers = item.get("answers", [])
                            if isinstance(answers, list) and answers:
                                answer = answers[0]
                            else:
                                answer = item.get("wellFormedAnswers", [""])[0] if item.get("wellFormedAnswers") else ""
                            
                            formatted_item = {
                                "question_id": item.get("query_id", f"msmarco_{medical_count}"),
                                "question": query,
                                "context": context,
                                "answer": answer,
                                "reasoning_type": "information_retrieval",
                                "medical_specialty": "general"
                            }
                            data.append(formatted_item)
                            medical_count += 1
                            
                            # Stop at reasonable number for evaluation
                            if medical_count >= 1000:
                                break
                    
                    if len(data) > 0:
                        logger.info(f"   📋 Loaded {len(data)} MS MARCO medical samples from {dataset_name}")
                        return data
                        
                except Exception as e:
                    logger.debug(f"   Could not load from {dataset_name} {config_name}: {e}")
                    continue
            
            # Fallback to sample data
            logger.warning("Could not load MSMARCO from HuggingFace, using fallback")
            return self._get_sample_msmarco_data()
            
        except Exception as e:
            logger.error(f"MSMARCO loading failed: {e}")
            return self._get_sample_msmarco_data()

    def download_and_cache_datasets(self, force_refresh: bool = False) -> Dict[str, bool]:
        """Download and cache all benchmark datasets."""
        logger.info("🔄 Downloading and caching benchmark datasets...")
        
        results = {}
        
        for benchmark_name in self.dataset_configs.keys():
            try:
                cache_file = self.cache_dir / self.dataset_configs[benchmark_name]["cache_file"]
                
                # Skip if cached and not forcing refresh
                if cache_file.exists() and not force_refresh:
                    logger.info(f"   ✅ {benchmark_name} already cached")
                    results[benchmark_name] = True
                    continue
                
                logger.info(f"   🔄 Downloading {benchmark_name}...")
                
                # Load dataset
                data = self.load_benchmark_data(benchmark_name, max_samples=None)
                
                if data and len(data) > 0:
                    # Cache the data
                    self._save_to_cache(data, cache_file)
                    logger.info(f"   ✅ Cached {len(data)} {benchmark_name} samples")
                    results[benchmark_name] = True
                else:
                    logger.warning(f"   ⚠️ No data loaded for {benchmark_name}")
                    results[benchmark_name] = False
                    
            except Exception as e:
                logger.error(f"   ❌ Failed to download {benchmark_name}: {e}")
                results[benchmark_name] = False
        
        return results

    def get_dataset_statistics(self) -> Dict[str, Dict]:
        """Get statistics about available datasets."""
        stats = {}
        
        for benchmark_name in self.dataset_configs.keys():
            try:
                cache_file = self.cache_dir / self.dataset_configs[benchmark_name]["cache_file"]
                
                if cache_file.exists():
                    data = self._load_from_cache(cache_file)
                    
                    # Calculate statistics
                    stats[benchmark_name] = {
                        "total_samples": len(data),
                        "cached": True,
                        "cache_file_size_mb": cache_file.stat().st_size / (1024 * 1024),
                        "sample_fields": list(data[0].keys()) if data else [],
                        "reasoning_types": list(set(item.get("reasoning_type", "unknown") for item in data)),
                        "medical_specialties": list(set(item.get("medical_specialty", "general") for item in data))
                    }
                else:
                    stats[benchmark_name] = {
                        "total_samples": 0,
                        "cached": False,
                        "status": "not_downloaded"
                    }
                    
            except Exception as e:
                stats[benchmark_name] = {
                    "error": str(e),
                    "cached": False
                }
        
        return stats
    
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