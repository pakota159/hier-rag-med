# src/evaluation/data/data_loader.py
"""
Updated BenchmarkDataLoader with Official MIRAGE QADataset Support
Integrates with src.utils.QADataset and disables PubMedQA (already in MIRAGE)
"""

import json
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from loguru import logger
import tempfile
from datasets import load_dataset
import yaml

# Import official MIRAGE QADataset utility
try:
    from mirage.src.utils import QADataset
    MIRAGE_UTILS_AVAILABLE = True
    logger.info("✅ Successfully imported QADataset from src.utils")
except ImportError:
    MIRAGE_UTILS_AVAILABLE = False
    logger.warning("⚠️ QADataset from src.utils not available - using fallback implementation")


class BenchmarkDataLoader:
    """Load benchmark datasets for medical RAG evaluation - Updated with MIRAGE QADataset integration."""
    
    def __init__(self, config: Dict):
        """Initialize benchmark data loader."""
        self.config = config
        self.data_dir = Path(config.get("data_dir", "src/evaluation/data"))
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Updated dataset configurations with MIRAGE QADataset integration
        self.dataset_configs = {
            "mirage": {
                "source": "official_qadataset",  # Use official QADataset first
                "fallback_source": "github",
                "dataset_name": "MIRAGE",
                "url": "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json",
                "backup_urls": [
                    "https://github.com/Teddy-XiongGZ/MIRAGE/raw/main/benchmark.json"
                ],
                "file_format": "json",
                "cache_file": "mirage_official_data.json",
                "use_qadataset": True
            },
            "medreason": {
                "source": "huggingface", 
                "dataset_name": "UCSC-VLAA/MedReason",
                "backup_sources": [
                    "bigbio/med_qa",
                    "augtoma/usmle_qa"
                ],
                "file_format": "json",
                "cache_file": "medreason_data.json"
            },
            # PubMedQA DISABLED - Already included in MIRAGE
            "pubmedqa": {
                "enabled": False,
                "disabled_reason": "PubMedQA questions are already included in MIRAGE benchmark",
                "alternative": "Use MIRAGE benchmark for research literature QA",
                "cache_file": "pubmedqa_disabled.json"
            },
            "msmarco": {
                "source": "huggingface",
                "dataset_name": "microsoft/ms_marco",
                "config": "v1.1",
                "backup_sources": [
                    "ms_marco"
                ],
                "file_format": "json",
                "cache_file": "msmarco_data.json"
            }
        }
    
    def load_benchmark_data(self, benchmark_name: str, split: str = "train", 
                          max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data for a specific benchmark - Updated with MIRAGE QADataset support."""
        logger.info(f"📚 Loading {benchmark_name} benchmark data...")
        
        if benchmark_name not in self.dataset_configs:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        config = self.dataset_configs[benchmark_name]
        
        # Check if benchmark is disabled (like PubMedQA)
        if not config.get("enabled", True):
            logger.warning(f"⚠️ {benchmark_name} benchmark is DISABLED")
            logger.info(f"   Reason: {config.get('disabled_reason', 'Not specified')}")
            logger.info(f"   Alternative: {config.get('alternative', 'Use another benchmark')}")
            return []
        
        cache_file = self.cache_dir / config["cache_file"]
        
        # Try to load from cache first
        if cache_file.exists():
            logger.info(f"   📂 Loading from cache: {cache_file}")
            return self._load_from_cache(cache_file, max_samples)
        
        # Load from source based on benchmark type
        if benchmark_name == "mirage":
            data = self._load_mirage_with_qadataset(split)
        elif benchmark_name == "medreason":
            data = self._load_medreason_data_fixed(split)
        elif benchmark_name == "pubmedqa":
            # PubMedQA is disabled
            logger.error("❌ PubMedQA is disabled - use MIRAGE instead")
            return []
        elif benchmark_name == "msmarco":
            data = self._load_msmarco_data_fixed(split)
        else:
            raise ValueError(f"No loader implemented for {benchmark_name}")
        
        # Cache the loaded data
        if data:
            self._save_to_cache(data, cache_file)
        
        # Apply sample limit
        if max_samples and len(data) > max_samples:
            data = data[:max_samples]
            logger.info(f"   ✂️ Limited to {max_samples} samples")
        
        logger.info(f"   ✅ Loaded {len(data)} samples for {benchmark_name}")
        return data

    def _load_mirage_with_qadataset(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load MIRAGE data using official QADataset utility."""
        logger.info("📚 Loading MIRAGE using official QADataset...")
        
        try:
            # Try official QADataset first
            if MIRAGE_UTILS_AVAILABLE:
                logger.info("   🎯 Using official QADataset from src.utils")
                qa_dataset = QADataset(
                    dataset_name="mirage",
                    split=split,
                    cache_dir=str(self.cache_dir)
                )
                
                raw_data = qa_dataset.load_data()
                
                if raw_data:
                    # Process into standardized format
                    processed_data = []
                    for idx, item in enumerate(raw_data):
                        processed_item = {
                            "question_id": item.get("id", f"mirage_{idx}"),
                            "question": item.get("question", ""),
                            "context": item.get("context", ""),
                            "answer": item.get("answer", ""),
                            "explanation": item.get("explanation", ""),
                            "options": item.get("options", []),
                            "question_type": item.get("question_type", "multiple_choice"),
                            "medical_specialty": item.get("medical_specialty", "general_medicine"),
                            "reasoning_type": item.get("reasoning_type", "clinical_reasoning"),
                            "difficulty": item.get("difficulty", "medium"),
                            "benchmark": "mirage",
                            "source": "official_qadataset"
                        }
                        processed_data.append(processed_item)
                    
                    logger.info(f"   ✅ Loaded {len(processed_data)} MIRAGE samples using QADataset")
                    return processed_data
                else:
                    logger.warning("   ⚠️ QADataset returned empty data, falling back to manual loading")
            
            # Fallback to manual loading
            return self._load_official_mirage_data_fallback()
            
        except Exception as e:
            logger.warning(f"   ⚠️ QADataset loading failed: {e}. Using fallback method.")
            return self._load_official_mirage_data_fallback()
    
    def _load_official_mirage_data_fallback(self) -> List[Dict[str, Any]]:
        """Fallback method to load MIRAGE data from GitHub repository."""
        logger.info("   📚 Using fallback MIRAGE data loading from GitHub...")
        
        config = self.dataset_configs["mirage"]
        urls_to_try = [config["url"]] + config.get("backup_urls", [])
        
        for url in urls_to_try:
            try:
                logger.info(f"   🔗 Trying URL: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if isinstance(data, list):
                    raw_questions = data
                elif isinstance(data, dict):
                    # Handle different JSON structures
                    raw_questions = data.get("questions", data.get("data", []))
                else:
                    logger.warning(f"   ⚠️ Unexpected data format from {url}")
                    continue
                
                if raw_questions:
                    # Process into standardized format
                    processed_data = []
                    for idx, item in enumerate(raw_questions):
                        processed_item = {
                            "question_id": item.get("id", f"mirage_fallback_{idx}"),
                            "question": item.get("question", item.get("query", "")),
                            "context": item.get("context", item.get("background", "")),
                            "answer": item.get("answer", item.get("correct_answer", "")),
                            "explanation": item.get("explanation", item.get("rationale", "")),
                            "options": item.get("options", item.get("choices", [])),
                            "question_type": "multiple_choice",
                            "medical_specialty": "general_medicine",
                            "reasoning_type": "clinical_reasoning",
                            "difficulty": "medium",
                            "benchmark": "mirage",
                            "source": "github_fallback"
                        }
                        processed_data.append(processed_item)
                    
                    logger.info(f"   ✅ Loaded {len(processed_data)} MIRAGE samples from fallback")
                    return processed_data
                
            except Exception as e:
                logger.debug(f"   Failed to load from {url}: {e}")
                continue
        
        logger.error("   ❌ All MIRAGE URLs failed")
        return []

    def _load_medreason_data_fixed(self, split: str = "train") -> List[Dict[str, Any]]:
        """Load MedReason benchmark data with fallbacks."""
        try:
            sources = [
                ("UCSC-VLAA/MedReason", split),
                ("bigbio/med_qa", "med_qa_en_bigbio_qa", split),
                ("augtoma/usmle_qa", split)
            ]
            
            for source_info in sources:
                try:
                    if len(source_info) == 3:
                        source, config, split_name = source_info
                        logger.info(f"   Trying MedReason source: {source} with config: {config}")
                        dataset = load_dataset(source, config, split=split_name, trust_remote_code=True)
                    else:
                        source, split_name = source_info
                        logger.info(f"   Trying MedReason source: {source}")
                        dataset = load_dataset(source, split=split_name, trust_remote_code=True)
                    
                    data = []
                    for item in dataset:
                        formatted_item = {
                            "question_id": item.get("id", f"medreason_{len(data)}"),
                            "question": item.get("question", item.get("input", "")),
                            "context": item.get("context", item.get("reasoning_chain", "")),
                            "answer": item.get("answer", item.get("target", "")),
                            "reasoning_chain": item.get("reasoning_chain", []),
                            "reasoning_type": "diagnostic_reasoning",
                            "medical_specialty": "general_medicine",
                            "explanation": item.get("explanation", ""),
                            "benchmark": "medreason"
                        }
                        data.append(formatted_item)
                    
                    if len(data) > 0:
                        logger.info(f"   📋 Loaded {len(data)} MedReason samples from {source}")
                        return data
                        
                except Exception as e:
                    logger.debug(f"   Could not load MedReason from {source}: {e}")
                    continue
            
            logger.warning("Could not load MedReason from any HuggingFace source")
            return []
            
        except Exception as e:
            logger.error(f"MedReason loading failed: {e}")
            return []

    def _load_msmarco_data_fixed(self, split: str = "dev") -> List[Dict[str, Any]]:
        """Load MS MARCO benchmark data with medical filtering."""
        try:
            sources = [
                ("microsoft/ms_marco", "v1.1", split),
                ("ms_marco", split)
            ]
            
            for source_info in sources:
                try:
                    if len(source_info) == 3:
                        source, config, split_name = source_info
                        logger.info(f"   Trying MS MARCO source: {source} with config: {config}")
                        dataset = load_dataset(source, config, split=split_name, trust_remote_code=True)
                    else:
                        source, split_name = source_info
                        logger.info(f"   Trying MS MARCO source: {source}")
                        dataset = load_dataset(source, split=split_name, trust_remote_code=True)
                    
                    data = []
                    medical_keywords = [
                        'health', 'medical', 'disease', 'treatment', 'medicine', 
                        'doctor', 'hospital', 'patient', 'symptom', 'therapy'
                    ]
                    
                    for item in dataset:
                        query = item.get("query", item.get("question", ""))
                        
                        # Filter for medical relevance
                        if any(keyword in query.lower() for keyword in medical_keywords):
                            formatted_item = {
                                "question_id": item.get("query_id", f"msmarco_{len(data)}"),
                                "query": query,
                                "passages": item.get("passages", []),
                                "relevant_passages": item.get("relevant_passages", []),
                                "query_type": "medical_retrieval",
                                "benchmark": "msmarco"
                            }
                            data.append(formatted_item)
                            
                            # Limit to reasonable size for medical subset
                            if len(data) >= 5000:
                                break
                    
                    if len(data) > 0:
                        logger.info(f"   📋 Loaded {len(data)} medical MS MARCO samples from {source}")
                        return data
                        
                except Exception as e:
                    logger.debug(f"   Could not load MS MARCO from {source}: {e}")
                    continue
            
            logger.warning("Could not load MS MARCO from any HuggingFace source")
            return []
            
        except Exception as e:
            logger.error(f"MS MARCO loading failed: {e}")
            return []
    
    def _load_from_cache(self, cache_file: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data from cache file."""
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if max_samples and len(data) > max_samples:
                data = data[:max_samples]
            
            return data
        except Exception as e:
            logger.error(f"Failed to load from cache {cache_file}: {e}")
            return []
    
    def _save_to_cache(self, data: List[Dict[str, Any]], cache_file: Path):
        """Save data to cache file."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"   💾 Cached data to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    def get_available_benchmarks(self) -> List[str]:
        """Get list of available benchmarks."""
        return [name for name, config in self.dataset_configs.items() 
                if config.get("enabled", True)]
    
    def get_disabled_benchmarks(self) -> Dict[str, str]:
        """Get list of disabled benchmarks with reasons."""
        return {
            name: config.get("disabled_reason", "Unknown reason")
            for name, config in self.dataset_configs.items()
            if not config.get("enabled", True)
        }
    
    def is_benchmark_enabled(self, benchmark_name: str) -> bool:
        """Check if a benchmark is enabled."""
        config = self.dataset_configs.get(benchmark_name, {})
        return config.get("enabled", True)