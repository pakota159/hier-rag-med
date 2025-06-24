"""
Enhanced Configuration module for Basic Reasoning system.
Updated for medical Q&A optimization and MIRAGE benchmark performance.
"""

import os
import platform
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from loguru import logger


class Config:
    """Enhanced configuration manager for Hierarchical Medical Q&A system."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize enhanced configuration for medical Q&A."""
        self.environment = self._detect_environment()
        
        # Determine config file based on environment
        if config_path:
            self.config_path = config_path
        elif self.environment in ["runpod_gpu", "cuda_gpu"]:
            self.config_path = Path(__file__).parent / "config_gpu.yaml"
        else:
            self.config_path = Path(__file__).parent / "config.yaml"
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"🔧 Initialized Enhanced Config for {self.environment}")

    def _detect_environment(self) -> str:
        """Enhanced environment detection."""
        # Check for RunPod environment
        if os.getenv("RUNPOD_POD_ID"):
            return "runpod_gpu"
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            return "cuda_gpu"
        
        # Check for Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps_local"
        
        # Default to CPU
        return "cpu_local"

    def _load_config(self) -> Dict:
        """Load configuration with enhanced medical Q&A defaults."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Loaded config from {self.config_path}")
            return config
        else:
            logger.warning(f"⚠️ Config file not found: {self.config_path}")
            logger.info("🔧 Using enhanced default configuration for medical Q&A")
            return self._get_enhanced_default_config()

    def _get_enhanced_default_config(self) -> Dict:
        """Enhanced default configuration optimized for medical Q&A."""
        # Auto-detect optimal settings based on environment
        if self.environment in ["runpod_gpu", "cuda_gpu"]:
            device = "cuda"
            batch_size = 32
            tier1_top_k = 8
            temperature = 0.2  # Lower for more consistent answers
        elif self.environment == "mps_local":
            device = "mps"
            batch_size = 16
            tier1_top_k = 5
            temperature = 0.3
        else:
            device = "cpu"
            batch_size = 8
            tier1_top_k = 3
            temperature = 0.4
        
        return {
            "data_dir": "data/foundation",
            "models": {
                "embedding": {
                    "name": "sentence-transformers/all-MiniLM-L6-v2",  # Fallback
                    "medical_embedding": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "device": device,
                    "batch_size": batch_size,
                    "use_medical_embedding": True
                },
                "llm": {
                    "name": "mistral:7b-instruct",
                    "temperature": temperature,
                    "context_window": 4096,
                    "device": device
                }
            },
            "hierarchical_retrieval": {
                "tier1_top_k": tier1_top_k,
                "tier2_top_k": 4,
                "tier3_top_k": 3,
                "enable_evidence_stratification": True,
                "enable_temporal_weighting": True,
                "medical_specialty_boost": True,
                "balanced_tier_distribution": True
            },
            "processing": {
                "chunk_size": 512,
                "chunk_overlap": 100,
                "min_content_length": 50,
                "enable_medical_entity_recognition": True,
                "target_tier_distribution": {
                    "tier1": 0.30,  # 30% Pattern Recognition
                    "tier2": 0.40,  # 40% Clinical Reasoning
                    "tier3": 0.30   # 30% Evidence Confirmation
                }
            },
            "prompts": {
                "system": """You are a medical knowledge expert answering multiple choice questions.
Analyze each question systematically using hierarchical medical reasoning.

CRITICAL REQUIREMENTS:
- Select ONLY ONE answer: A, B, C, D, or E
- Format your final answer as: "Answer: [LETTER]"
- Base your answer on medical evidence and established knowledge
- Be precise, accurate, and evidence-based

RESPONSE FORMAT:
1. Brief analysis of the question topic
2. Systematic evaluation of key options
3. Selection of the best answer
4. Final answer: "Answer: [LETTER]"

EXAMPLE:
Question: What is the most common cause of community-acquired pneumonia?
Options: A) Mycoplasma B) Streptococcus pneumoniae C) Haemophilus influenzae D) Staphylococcus aureus

Analysis: Community-acquired pneumonia etiology varies by population and setting. S. pneumoniae remains the leading bacterial cause in most populations, particularly in adults without comorbidities.

Answer: B""",
                "tier1_prompt": """TIER 1 - MEDICAL PATTERN RECOGNITION:
Analyze the basic medical concepts, definitions, anatomy, and fundamental knowledge patterns.

Focus areas:
• Medical terminology and definitions
• Basic anatomy and physiology
• Fundamental disease concepts
• Clinical presentations and symptoms
• Basic classifications and categories

Use this foundational knowledge to understand what the question is asking about.""",
                "tier2_prompt": """TIER 2 - CLINICAL REASONING:
Apply clinical knowledge, pathophysiology, and diagnostic reasoning.

Focus areas:
• Disease mechanisms and pathophysiology
• Clinical decision-making processes
• Diagnostic criteria and differential diagnosis
• Treatment principles and management approaches
• Risk factors and prognostic indicators

Use this clinical knowledge to systematically evaluate each answer option.""",
                "tier3_prompt": """TIER 3 - EVIDENCE-BASED CONFIRMATION:
Confirm with established medical facts, guidelines, and research evidence.

Focus areas:
• Clinical practice guidelines (WHO, AHA, ESC, etc.)
• Evidence-based medicine and research findings
• Established medical standards and protocols
• Authoritative medical recommendations
• Definitive diagnostic and treatment criteria

Use this authoritative knowledge to select the most accurate answer."""
            },
            "web": {
                "host": "0.0.0.0",
                "port": 8503,
                "cors_origins": ["*"]
            },
            "evaluation": {
                "enable_answer_extraction": True,
                "answer_validation_strict": True,
                "require_final_answer_format": True,
                "log_reasoning_steps": True
            }
        }

    def _setup_logging(self):
        """Setup enhanced logging with medical Q&A focus."""
        log_dir = self.get_data_dir("logs")
        log_file = log_dir / f"enhanced_medical_qa_{self.environment}.log"
        
        logger.add(
            log_file,
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | Enhanced Medical Q&A | {message}"
        )

    def get_data_dir(self, subdir: str) -> Path:
        """Get data directory path with automatic creation."""
        data_dir = Path(self.config["data_dir"]) / subdir
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def save(self) -> None:
        """Save enhanced configuration to file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"💾 Saved Enhanced Medical Q&A configuration to {self.config_path}")

    def get_device_info(self) -> Dict:
        """Get comprehensive device information for medical Q&A optimization."""
        info = {
            "environment": self.environment,
            "config_file": self.config_path.name,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "optimization_level": "medical_qa_enhanced"
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "recommended_batch_size": 32
            })
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info.update({
                "mps_device": "Apple Silicon GPU",
                "recommended_batch_size": 16
            })
        else:
            info.update({
                "device": "CPU",
                "recommended_batch_size": 8
            })
        
        return info

    def get_medical_qa_settings(self) -> Dict:
        """Get optimized settings for medical Q&A performance."""
        return {
            "temperature": self.config["models"]["llm"]["temperature"],
            "tier_distribution": {
                "tier1_top_k": self.config["hierarchical_retrieval"]["tier1_top_k"],
                "tier2_top_k": self.config["hierarchical_retrieval"]["tier2_top_k"],
                "tier3_top_k": self.config["hierarchical_retrieval"]["tier3_top_k"]
            },
            "answer_extraction": self.config.get("evaluation", {}).get("enable_answer_extraction", True),
            "strict_validation": self.config.get("evaluation", {}).get("answer_validation_strict", True)
        }

    def optimize_for_benchmark(self, benchmark_name: str = "mirage") -> None:
        """Optimize configuration for specific medical benchmarks."""
        if benchmark_name.lower() == "mirage":
            # MIRAGE-specific optimizations
            self.config["models"]["llm"]["temperature"] = 0.2  # More deterministic
            self.config["hierarchical_retrieval"]["tier1_top_k"] += 1  # More pattern recognition
            self.config["hierarchical_retrieval"]["tier3_top_k"] += 1  # More evidence
            
            logger.info("🎯 Optimized configuration for MIRAGE benchmark")
        
        elif benchmark_name.lower() == "medqa":
            # MedQA-specific optimizations
            self.config["models"]["llm"]["temperature"] = 0.3
            self.config["hierarchical_retrieval"]["tier2_top_k"] += 1  # More clinical reasoning
            
            logger.info("🎯 Optimized configuration for MedQA benchmark")

    def __getitem__(self, key: str):
        """Get configuration value."""
        return self.config[key]

    def __setitem__(self, key: str, value):
        """Set configuration value."""
        self.config[key] = value

    def get(self, key: str, default=None):
        """Get configuration value with default."""
        return self.config.get(key, default)