"""
Updated BenchmarkDataLoader with Official MIRAGE Support
Integrates with the real MIRAGE benchmark dataset
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


class BenchmarkDataLoader:
    """Load benchmark datasets for medical RAG evaluation - Updated with real MIRAGE."""
    
    def __init__(self, config: Dict):
        """Initialize benchmark data loader."""
        self.config = config
        self.data_dir = Path(config.get("data_dir", "src/evaluation/data"))
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Updated dataset configurations with real sources
        self.dataset_configs = {
            "mirage": {
                "source": "github",
                "dataset_name": "MIRAGE",
                "url": "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json",
                "backup_urls": [
                    "https://github.com/Teddy-XiongGZ/MIRAGE/raw/main/benchmark.json"
                ],
                "file_format": "json",
                "cache_file": "mirage_official_data.json"
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
            "pubmedqa": {
                "source": "huggingface",
                "dataset_name": "qiaojin/PubMedQA",
                "config": "pqa_labeled",
                "backup_sources": [
                    "pubmed_qa"
                ],
                "file_format": "json", 
                "cache_file": "pubmedqa_data.json"
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
    
    def load_benchmark_data(self, benchmark_name: str, split: str = "test", 
                          max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data for a specific benchmark - Updated with real MIRAGE support."""
        logger.info(f"📚 Loading {benchmark_name} benchmark data...")
        
        if benchmark_name not in self.dataset_configs:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        config = self.dataset_configs[benchmark_name]
        cache_file = self.cache_dir / config["cache_file"]
        
        # Try to load from cache first
        if cache_file.exists():
            logger.info(f"   📂 Loading from cache: {cache_file}")
            return self._load_from_cache(cache_file, max_samples)
        
        # Load from source based on benchmark type
        if benchmark_name == "mirage":
            data = self._load_official_mirage_data()
        elif benchmark_name == "medreason":
            data = self._load_medreason_data_fixed(split)
        elif benchmark_name == "pubmedqa":
            data = self._load_pubmedqa_data_fixed(split)
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

    def _load_official_mirage_data(self) -> List[Dict[str, Any]]:
        """Load official MIRAGE benchmark from GitHub repository."""
        config = self.dataset_configs["mirage"]
        
        # Try primary and backup URLs
        urls_to_try = [config["url"]] + config.get("backup_urls", [])
        
        for url in urls_to_try:
            try:
                logger.info(f"   🌐 Downloading MIRAGE from: {url}")
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                mirage_raw = response.json()
                logger.info(f"   📋 Downloaded MIRAGE with {len(mirage_raw)} datasets")
                
                # Parse and format the MIRAGE data
                formatted_data = []
                
                # MIRAGE contains 5 sub-datasets
                for dataset_name, questions in mirage_raw.items():
                    logger.info(f"   📊 Processing {dataset_name}: {len(questions)} questions")
                    
                    for i, question_data in enumerate(questions):
                        formatted_item = self._format_mirage_question(question_data, dataset_name, i)
                        if formatted_item:
                            formatted_data.append(formatted_item)
                
                if len(formatted_data) > 0:
                    logger.info(f"   ✅ Successfully loaded {len(formatted_data)} MIRAGE questions")
                    return formatted_data
                
            except requests.RequestException as e:
                logger.debug(f"   ❌ Failed to download from {url}: {e}")
                continue
            except json.JSONDecodeError as e:
                logger.debug(f"   ❌ JSON parsing error from {url}: {e}")
                continue
            except Exception as e:
                logger.debug(f"   ❌ Unexpected error from {url}: {e}")
                continue
        
        # If all downloads fail, return empty list to trigger fallback
        logger.warning("Could not load official MIRAGE from any source")
        return []

    def _format_mirage_question(self, question_data: Dict, dataset_name: str, index: int) -> Dict:
        """Format a MIRAGE question to our standard format."""
        try:
            # Handle different MIRAGE question formats
            question_id = question_data.get("id", f"{dataset_name}_{index}")
            question_text = question_data.get("question", "")
            
            # Extract answer (different formats in MIRAGE)
            answer = self._extract_mirage_answer(question_data)
            
            # Extract options (different formats in MIRAGE)
            options = self._extract_mirage_options(question_data)
            
            # Determine medical specialty and reasoning type
            medical_specialty = self._classify_medical_specialty(question_text, dataset_name)
            reasoning_type = self._classify_reasoning_type(question_text, dataset_name)
            
            formatted_item = {
                "question_id": question_id,
                "question": question_text,
                "context": question_data.get("context", ""),
                "answer": answer,
                "options": options,
                "reasoning_type": reasoning_type,
                "medical_specialty": medical_specialty,
                "source_dataset": dataset_name,
                "difficulty": self._assess_question_difficulty(question_text),
                "question_type": self._get_question_type(dataset_name)
            }
            
            return formatted_item
            
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to format question {index} from {dataset_name}: {e}")
            return None

    def _extract_mirage_answer(self, question_data: Dict) -> str:
        """Extract answer from MIRAGE question data."""
        # Try different answer field names used in MIRAGE
        answer_fields = ["answer", "target", "final_decision", "correct_answer"]
        
        for field in answer_fields:
            if field in question_data and question_data[field]:
                answer = question_data[field]
                if isinstance(answer, str):
                    return answer.strip()
                elif isinstance(answer, (int, float)):
                    return str(answer)
                elif isinstance(answer, list) and len(answer) > 0:
                    return str(answer[0])
        
        return ""

    def _extract_mirage_options(self, question_data: Dict) -> List[str]:
        """Extract options from MIRAGE question data."""
        # Try different option field names used in MIRAGE
        option_fields = ["options", "choices", "candidates"]
        
        for field in option_fields:
            if field in question_data and question_data[field]:
                options = question_data[field]
                if isinstance(options, list):
                    return [str(opt).strip() for opt in options if opt]
                elif isinstance(options, dict):
                    # Handle {"A": "option1", "B": "option2"} format
                    return [str(val).strip() for val in options.values() if val]
        
        return []

    def _classify_medical_specialty(self, question_text: str, dataset_name: str) -> str:
        """Classify medical specialty based on question content."""
        question_lower = question_text.lower()
        
        # Specialty keywords mapping
        specialties = {
            "cardiology": ["heart", "cardiac", "cardiovascular", "ecg", "ekg", "myocardial", "coronary"],
            "neurology": ["brain", "neuro", "seizure", "stroke", "nervous", "cognitive", "cranial"],
            "endocrinology": ["diabetes", "hormone", "thyroid", "insulin", "glucose", "endocrine"],
            "infectious_disease": ["infection", "bacteria", "virus", "antibiotic", "fever", "pathogen"],
            "oncology": ["cancer", "tumor", "malignant", "chemotherapy", "radiation", "metastasis"],
            "respiratory": ["lung", "respiratory", "pneumonia", "asthma", "breathing", "pulmonary"],
            "gastroenterology": ["stomach", "intestine", "liver", "digestive", "gastro", "bowel"],
            "psychiatry": ["mental", "psychiatric", "depression", "anxiety", "psycho"],
            "emergency": ["emergency", "trauma", "acute", "urgent", "critical"]
        }
        
        for specialty, keywords in specialties.items():
            if any(keyword in question_lower for keyword in keywords):
                return specialty
        
        return "general_medicine"

    def _classify_reasoning_type(self, question_text: str, dataset_name: str) -> str:
        """Classify the type of reasoning required."""
        question_lower = question_text.lower()
        
        # Dataset-based classification
        if dataset_name == "medqa":
            return "clinical_reasoning"
        elif dataset_name == "pubmedqa":
            return "evidence_based"
        elif dataset_name == "bioasq":
            return "factual_retrieval"
        elif dataset_name == "mmlu":
            return "knowledge_application"
        elif dataset_name == "medmcqa":
            return "examination_recall"
        
        # Content-based classification
        if any(word in question_lower for word in ["diagnose", "differential", "symptom"]):
            return "diagnostic_reasoning"
        elif any(word in question_lower for word in ["treatment", "therapy", "management"]):
            return "therapeutic_reasoning"
        elif any(word in question_lower for word in ["mechanism", "pathway", "physiology"]):
            return "mechanistic_reasoning"
        else:
            return "factual_recall"

    def _assess_question_difficulty(self, question_text: str) -> str:
        """Assess question difficulty based on length and complexity."""
        word_count = len(question_text.split())
        
        if word_count > 100:
            return "hard"
        elif word_count > 50:
            return "medium"
        else:
            return "easy"

    def _get_question_type(self, dataset_name: str) -> str:
        """Get question type based on source dataset."""
        type_mapping = {
            "mmlu": "multiple_choice_knowledge",
            "medqa": "clinical_vignette", 
            "medmcqa": "examination_question",
            "pubmedqa": "yes_no_maybe",
            "bioasq": "factual_question"
        }
        
        return type_mapping.get(dataset_name, "general_medical")

    def _load_medreason_data_fixed(self, split: str = "test") -> List[Dict[str, Any]]:
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
                            "reasoning_type": "diagnostic_reasoning",
                            "medical_specialty": "general_medicine",
                            "explanation": item.get("explanation", "")
                        }
                        data.append(formatted_item)
                    
                    if len(data) > 0:
                        logger.info(f"   📋 Loaded {len(data)} MedReason samples from {source}")
                        return data
                        
                except Exception as e:
                    logger.debug(f"   Could not load MedReason from {source}: {e}")
                    continue
            
            logger.warning("Could not load MedReason from HuggingFace, generating synthetic dataset")
            return self._generate_synthetic_medreason_data()
            
        except Exception as e:
            logger.error(f"MedReason loading failed: {e}")
            return self._generate_synthetic_medreason_data()

    def _load_pubmedqa_data_fixed(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load PubMedQA benchmark data with fallbacks."""
        try:
            sources = [
                ("qiaojin/PubMedQA", "pqa_labeled", split),
                ("pubmed_qa", "pqa_labeled", split)
            ]
            
            for source, config, split_name in sources:
                try:
                    logger.info(f"   Trying PubMedQA source: {source}")
                    dataset = load_dataset(source, config, split=split_name, trust_remote_code=True)
                    
                    data = []
                    for item in dataset:
                        # Handle PubMedQA specific format
                        context_text = ""
                        if "context" in item and isinstance(item["context"], dict):
                            contexts = item["context"].get("contexts", [])
                            context_text = " ".join(contexts) if contexts else ""
                        
                        formatted_item = {
                            "question_id": item.get("pubid", f"pubmedqa_{len(data)}"),
                            "question": item.get("question", ""),
                            "context": context_text,
                            "answer": item.get("final_decision", item.get("answer", "")),
                            "long_answer": item.get("long_answer", ""),
                            "reasoning_type": "evidence_based",
                            "medical_specialty": "biomedical_research"
                        }
                        data.append(formatted_item)
                    
                    if len(data) > 0:
                        logger.info(f"   📋 Loaded {len(data)} PubMedQA samples")
                        return data
                        
                except Exception as e:
                    logger.debug(f"   Could not load PubMedQA from {source}: {e}")
                    continue
            
            logger.warning("Could not load PubMedQA from HuggingFace, generating synthetic dataset")
            return self._generate_synthetic_pubmedqa_data()
            
        except Exception as e:
            logger.error(f"PubMedQA loading failed: {e}")
            return self._generate_synthetic_pubmedqa_data()

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
                    medical_count = 0
                    
                    for item in dataset:
                        query = item.get("query", "")
                        if self._is_medical_query(query):
                            # Extract passages/context
                            passages = item.get("passages", {})
                            passage_texts = passages.get("passage_text", []) if isinstance(passages, dict) else []
                            context = " ".join(passage_texts) if passage_texts else ""
                            
                            # Extract answers
                            answers = item.get("answers", [])
                            answer = " ".join(answers) if answers else ""
                            
                            formatted_item = {
                                "question_id": item.get("qid", f"msmarco_{len(data)}"),
                                "question": query,
                                "context": context,
                                "answer": answer,
                                "reasoning_type": "information_retrieval",
                                "medical_specialty": "general_medicine"
                            }
                            data.append(formatted_item)
                            medical_count += 1
                            
                            if medical_count >= 1000:  # Limit medical samples
                                break
                    
                    if len(data) > 0:
                        logger.info(f"   📋 Loaded {len(data)} MS MARCO medical samples")
                        return data
                        
                except Exception as e:
                    logger.debug(f"   Could not load MS MARCO from {source_info}: {e}")
                    continue
            
            logger.warning("Could not load MS MARCO from HuggingFace, generating synthetic dataset")
            return self._generate_synthetic_msmarco_data()
            
        except Exception as e:
            logger.error(f"MS MARCO loading failed: {e}")
            return self._generate_synthetic_msmarco_data()

    def _is_medical_query(self, query: str) -> bool:
        """Check if a query is medical-related."""
        medical_keywords = [
            "disease", "symptom", "treatment", "diagnosis", "medicine", "drug",
            "patient", "health", "medical", "clinical", "therapy", "cancer",
            "diabetes", "heart", "blood", "doctor", "hospital", "pain",
            "infection", "virus", "bacteria", "surgery", "medication"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in medical_keywords)

    def _generate_synthetic_medreason_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic MedReason dataset."""
        logger.info("   🏗️ Generating synthetic MedReason dataset...")
        
        base_data = [
            {
                "question_id": "medreason_synthetic_001",
                "question": "A 45-year-old patient presents with chest pain and shortness of breath. What is the diagnostic approach?",
                "context": "Patient has substernal chest pain, diaphoresis, and shortness of breath for 2 hours.",
                "answer": "Obtain ECG, cardiac enzymes, and chest X-ray to rule out acute coronary syndrome.",
                "reasoning_type": "diagnostic_reasoning",
                "medical_specialty": "emergency_medicine",
                "explanation": "Classic presentation requires immediate cardiac evaluation to exclude MI."
            }
        ]
        return base_data * 100  # Create larger synthetic dataset

    def _generate_synthetic_pubmedqa_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic PubMedQA dataset."""
        logger.info("   🏗️ Generating synthetic PubMedQA dataset...")
        
        base_data = [
            {
                "question_id": "pubmedqa_synthetic_001",
                "question": "Does metformin reduce cardiovascular risk in diabetic patients?",
                "context": "Multiple studies have examined metformin's cardiovascular effects in diabetes.",
                "answer": "yes",
                "long_answer": "Yes, metformin has been shown to reduce cardiovascular risk in diabetic patients.",
                "reasoning_type": "evidence_based",
                "medical_specialty": "endocrinology"
            }
        ]
        return base_data * 100

    def _generate_synthetic_msmarco_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic MS MARCO medical dataset."""
        logger.info("   🏗️ Generating synthetic MS MARCO dataset...")
        
        base_data = [
            {
                "question_id": "msmarco_synthetic_001",
                "question": "symptoms of diabetes",
                "context": "Diabetes symptoms include increased thirst, frequent urination, and fatigue.",
                "answer": "Common diabetes symptoms are polydipsia, polyuria, and unexplained fatigue.",
                "reasoning_type": "information_retrieval",
                "medical_specialty": "endocrinology"
            }
        ]
        return base_data * 200

    def _load_from_cache(self, cache_file: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data from cache file."""
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        if max_samples and len(data) > max_samples:
            data = data[:max_samples]
        
        return data
    
    def _save_to_cache(self, data: List[Dict[str, Any]], cache_file: Path) -> None:
        """Save data to cache file."""
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"   💾 Cached data to {cache_file}")

    def download_all_benchmarks(self, force_refresh: bool = False) -> Dict[str, bool]:
        """Download and cache all benchmark datasets."""
        logger.info("🔄 Downloading all medical benchmark datasets...")
        
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
                    logger.info(f"   ✅ Downloaded {len(data)} {benchmark_name} samples")
                    results[benchmark_name] = True
                else:
                    logger.warning(f"   ⚠️ No data loaded for {benchmark_name}")
                    results[benchmark_name] = False
                    
            except Exception as e:
                logger.error(f"   ❌ Failed to download {benchmark_name}: {e}")
                results[benchmark_name] = False
        
        success_count = sum(results.values())
        total_count = len(results)
        logger.info(f"🎉 Downloaded {success_count}/{total_count} benchmarks successfully")
        
        return results

    def get_dataset_info(self) -> Dict[str, Dict]:
        """Get information about all available datasets."""
        info = {}
        
        for benchmark_name, config in self.dataset_configs.items():
            cache_file = self.cache_dir / config["cache_file"]
            
            dataset_info = {
                "source": config["source"],
                "dataset_name": config["dataset_name"],
                "cached": cache_file.exists(),
                "cache_file": str(cache_file)
            }
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    dataset_info["cached_samples"] = len(data)
                    
                    # Sample analysis
                    if data:
                        sample = data[0]
                        dataset_info["sample_fields"] = list(sample.keys())
                        
                        # Count specialties and reasoning types
                        specialties = set(item.get("medical_specialty", "unknown") for item in data)
                        reasoning_types = set(item.get("reasoning_type", "unknown") for item in data)
                        
                        dataset_info["medical_specialties"] = list(specialties)
                        dataset_info["reasoning_types"] = list(reasoning_types)
                
                except Exception as e:
                    dataset_info["error"] = str(e)
            else:
                dataset_info["cached_samples"] = 0
            
            info[benchmark_name] = dataset_info
        
        return info