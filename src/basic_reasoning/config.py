"""
Enhanced Configuration module for Basic Reasoning system.
Updated to support Microsoft BiomedNLP-PubMedBERT medical embedding.

File: src/basic_reasoning/config.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
import yaml
from loguru import logger


class Config:
    """Enhanced configuration manager with medical embedding support."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration with medical embedding support."""
        self.config_path = config_path or Path(__file__).parent / "config.yaml"
        
        # Auto-detect environment
        self.environment = self._detect_environment()
        logger.info(f"🔧 Detected environment: {self.environment}")
        
        # Load configuration
        self.config = self._load_config()

        # Create data directories
        self._create_directories()
        
        # Set up logging
        self._setup_logging()

    def _detect_environment(self) -> str:
        """Auto-detect the current environment."""
        if "RUNPOD_POD_ID" in os.environ or "RUNPOD_POD_HOSTNAME" in os.environ:
            return "runpod_gpu"
        elif "CUDA_VISIBLE_DEVICES" in os.environ and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            return "cuda_gpu"
        elif sys.platform == "darwin" and "arm64" in os.uname().machine.lower():
            return "mps_local"
        else:
            return "cpu_local"

    def _load_config(self) -> Dict:
        """Load configuration from file with fallback to enhanced defaults."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Loaded config from {self.config_path}")
            
            # Validate medical embedding configuration
            self._validate_medical_config(config)
            return config
        else:
            logger.warning(f"⚠️ Config file not found: {self.config_path}")
            logger.info("🔧 Using enhanced default configuration for medical Q&A")
            return self._get_enhanced_default_config()

    def _validate_medical_config(self, config: Dict) -> None:
        """Validate medical embedding configuration."""
        embedding_config = config.get("models", {}).get("embedding", {})
        
        # Check if medical embedding is properly configured
        if embedding_config.get("use_medical_embedding", True):
            medical_model = embedding_config.get("name", "")
            if "BiomedNLP-PubMedBERT" not in medical_model:
                logger.warning("⚠️ Medical embedding not configured, using default")
                embedding_config["name"] = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
                embedding_config["use_medical_embedding"] = True

    def _get_enhanced_default_config(self) -> Dict:
        """Enhanced default configuration optimized for medical Q&A."""
        # Auto-detect optimal settings based on environment
        if self.environment in ["runpod_gpu", "cuda_gpu"]:
            device = "cuda"
            batch_size = 32
            tier1_top_k = 8
            temperature = 0.2  # Lower for more consistent answers
            chunk_size = 384
        elif self.environment == "mps_local":
            device = "mps"
            batch_size = 8
            tier1_top_k = 5
            temperature = 0.3
            chunk_size = 384
        else:
            device = "cpu"
            batch_size = 4
            tier1_top_k = 3
            temperature = 0.4
            chunk_size = 256  # Smaller for CPU
        
        return {
            "data_dir": "data/foundation",
            "models": {
                "embedding": {
                    "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "fallback_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "device": device,
                    "batch_size": batch_size,
                    "max_length": 512,
                    "normalize_embeddings": True,
                    "trust_remote_code": False,
                    "use_medical_embedding": True,
                    "model_kwargs": {
                        "torch_dtype": "float16" if device != "cpu" else "float32",
                        "attn_implementation": "eager"
                    },
                    "device_settings": {
                        "cuda": {"batch_size": 32, "mixed_precision": True},
                        "mps": {"batch_size": 8, "mixed_precision": False},
                        "cpu": {"batch_size": 4, "mixed_precision": False}
                    }
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
                "tier2_top_k": max(4, tier1_top_k - 1),
                "tier3_top_k": max(3, tier1_top_k - 2),
                "enable_evidence_stratification": True,
                "enable_temporal_weighting": True,
                "medical_specialty_boost": True,
                "balanced_tier_distribution": True,
                "medical_entity_boost": 1.2,
                "clinical_context_window": 3
            },
            "processing": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_size // 4,  # 25% overlap
                "min_content_length": 50,
                "enable_medical_entity_recognition": True,
                "preserve_medical_terminology": True,
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
4. Final answer: "Answer: [LETTER]" """,
                "tier1_prompt": """TIER 1 - MEDICAL PATTERN RECOGNITION:
Analyze the basic medical concepts, definitions, anatomy, and fundamental knowledge patterns.
Focus on established medical facts, terminology, and basic pathophysiology.""",
                "tier2_prompt": """TIER 2 - CLINICAL REASONING:
Apply clinical reasoning and diagnostic thinking to the medical scenario.
Consider differential diagnosis, clinical presentations, and diagnostic approaches.""",
                "tier3_prompt": """TIER 3 - EVIDENCE CONFIRMATION:
Evaluate evidence-based medicine, treatment guidelines, and research findings.
Consider latest clinical guidelines, treatment protocols, and research evidence."""
            },
            "evaluation": {
                "enable_medical_validation": True,
                "check_medical_terminology": True,
                "validate_clinical_accuracy": True
            }
        }

    def _create_directories(self) -> None:
        """Create necessary data directories."""
        directories = [
            "vector_db", "logs", "processed", "benchmarks", "cache"
        ]
        
        for directory in directories:
            self.get_data_dir(directory)

    def _setup_logging(self) -> None:
        """Setup logging with enhanced medical system identification."""
        log_dir = self.get_data_dir("logs")
        log_file = log_dir / f"basic_reasoning_{self.environment}.log"
        
        logger.add(
            log_file,
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | MedRAG | {message}"
        )

    def get_data_dir(self, subdir: str) -> Path:
        """Get data directory path and create if it doesn't exist."""
        data_dir = Path(self.config["data_dir"]) / subdir
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def get_device_info(self) -> Dict:
        """Get current device and environment information."""
        embedding_config = self.config["models"]["embedding"]
        
        return {
            "environment": self.environment,
            "device": embedding_config["device"],
            "batch_size": embedding_config["batch_size"],
            "embedding_model": embedding_config["name"],
            "use_medical_embedding": embedding_config.get("use_medical_embedding", True),
            "medical_optimized": "BiomedNLP" in embedding_config["name"]
        }

    def get_embedding_config(self) -> Dict:
        """Get complete embedding configuration."""
        return self.config["models"]["embedding"]

    def save(self) -> None:
        """Save configuration to file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        logger.info(f"💾 Saved configuration to {self.config_path}")

    def update_embedding_model(self, model_name: str, **kwargs) -> None:
        """Update embedding model configuration."""
        embedding_config = self.config["models"]["embedding"]
        embedding_config["name"] = model_name
        embedding_config.update(kwargs)
        
        # Set medical embedding flag
        embedding_config["use_medical_embedding"] = "BiomedNLP" in model_name
        
        logger.info(f"🔄 Updated embedding model to: {model_name}")

    def validate_medical_setup(self) -> bool:
        """Validate that medical embedding setup is correct."""
        embedding_config = self.config["models"]["embedding"]
        
        checks = {
            "medical_model": "BiomedNLP" in embedding_config.get("name", ""),
            "medical_flag": embedding_config.get("use_medical_embedding", False),
            "fallback_configured": "fallback_name" in embedding_config,
            "device_configured": "device" in embedding_config
        }
        
        all_passed = all(checks.values())
        
        if all_passed:
            logger.info("✅ Medical embedding configuration validated successfully")
        else:
            failed_checks = [k for k, v in checks.items() if not v]
            logger.warning(f"⚠️ Medical config validation failed: {failed_checks}")
        
        return all_passed

    def __getitem__(self, key: str):
        """Get configuration value."""
        return self.config[key]

    def __setitem__(self, key: str, value):
        """Set configuration value."""
        self.config[key] = value