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
import torch
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
        
        # Apply environment-specific device configuration
        self._apply_environment_config()

        # Create data directories
        self._create_directories()
        
        # Set up logging
        self._setup_logging()

    def _detect_environment(self) -> str:
        """Auto-detect the current environment with correct device priority."""
        # Check for RunPod environment first (highest priority for GPU)
        if (os.path.exists("/workspace") or 
            "RUNPOD_POD_ID" in os.environ or 
            "RUNPOD_POD_HOSTNAME" in os.environ or
            "runpod" in os.environ.get("HOSTNAME", "").lower()):
            return "runpod_gpu"
        
        # Check for general CUDA availability
        elif torch.cuda.is_available():
            return "cuda_gpu"
        
        # Check for MPS (Apple Silicon) - only if not in RunPod and CUDA not available
        elif (sys.platform == "darwin" and 
              hasattr(torch.backends, 'mps') and 
              torch.backends.mps.is_available()):
            return "mps_local"
        
        # Default to CPU
        else:
            return "cpu_local"

    def _apply_environment_config(self) -> None:
        """Apply environment-specific configuration settings."""
        models_config = self.config.get("models", {})
        
        if self.environment in ["runpod_gpu", "cuda_gpu"]:
            # GPU environments - use CUDA
            self._set_device_config(models_config, "cuda", batch_size_multiplier=2.0)
            logger.info("🚀 Applied CUDA GPU optimizations")
            
        elif self.environment == "mps_local":
            # Apple Silicon - use MPS
            self._set_device_config(models_config, "mps", batch_size_multiplier=1.0)
            logger.info("🍎 Applied MPS (Apple Silicon) optimizations")
            
        else:
            # CPU fallback
            self._set_device_config(models_config, "cpu", batch_size_multiplier=0.5)
            logger.info("💻 Applied CPU configuration")

    def _set_device_config(self, models_config: Dict, device: str, batch_size_multiplier: float) -> None:
        """Set device configuration for all model components."""
        # Configure embedding model
        if "embedding" not in models_config:
            models_config["embedding"] = {}
        
        embedding_config = models_config["embedding"]
        embedding_config["device"] = device
        
        # Adjust batch size based on device capability
        base_batch_size = embedding_config.get("batch_size", 16)
        new_batch_size = int(base_batch_size * batch_size_multiplier)
        embedding_config["batch_size"] = max(1, new_batch_size)  # Ensure at least 1
        
        # Configure LLM if present
        if "llm" in models_config:
            models_config["llm"]["device"] = device
            if "batch_size" in models_config["llm"]:
                llm_base_batch = models_config["llm"]["batch_size"]
                models_config["llm"]["batch_size"] = max(1, int(llm_base_batch * batch_size_multiplier))
        
        # Configure system-level models
        for system_name in ["kg_system", "hierarchical_system"]:
            if system_name in models_config:
                models_config[system_name]["device"] = device
                if "embedding" in models_config[system_name]:
                    models_config[system_name]["embedding"]["device"] = device

    def _load_config(self) -> Dict:
        """Load configuration from file with fallback to enhanced defaults."""
        # Try to load GPU config first for RunPod environments
        if self.environment == "runpod_gpu":
            gpu_config_path = Path(__file__).parent.parent / "evaluation" / "configs" / "gpu_runpod_config.yaml"
            if gpu_config_path.exists():
                try:
                    with open(gpu_config_path, "r") as f:
                        config = yaml.safe_load(f)
                    logger.info(f"✅ Loaded GPU config from {gpu_config_path}")
                    self._validate_medical_config(config)
                    return config
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load GPU config: {e}")
        
        # Try main config file
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
        return {
            "data_dir": "data",
            "models": {
                "embedding": {
                    "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "fallback_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "device": "auto",  # Will be set by _apply_environment_config
                    "batch_size": 16,
                    "max_length": 512,
                    "normalize_embeddings": True,
                    "use_medical_embedding": True
                },
                "llm": {
                    "name": "mistral:7b-instruct",
                    "device": "auto",  # Will be set by _apply_environment_config
                    "batch_size": 16,
                    "temperature": 0.7,
                    "context_window": 4096,
                    "max_new_tokens": 512
                }
            },
            "retrieval": {
                "tier1_top_k": 5,
                "tier2_top_k": 4,
                "tier3_top_k": 3,
                "similarity_threshold": 0.7,
                "enable_reranking": True
            },
            "generation": {
                "system_prompt": "You are a medical knowledge assistant specialized in clinical reasoning and evidence-based medicine.",
                "enable_answer_extraction": True,
                "force_answer_format": True
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
            "medical_optimized": "BiomedNLP" in embedding_config["name"],
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
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
            "device_set": embedding_config.get("device") is not None,
            "valid_device": embedding_config.get("device") in ["cuda", "mps", "cpu"]
        }
        
        all_valid = all(checks.values())
        
        if all_valid:
            logger.info("✅ Medical embedding setup validated successfully")
        else:
            failed_checks = [k for k, v in checks.items() if not v]
            logger.warning(f"⚠️ Medical setup validation failed: {failed_checks}")
        
        return all_valid

    def __getitem__(self, key: str):
        """Get configuration value."""
        return self.config[key]

    def __setitem__(self, key: str, value):
        """Set configuration value."""
        self.config[key] = value

    def get(self, key: str, default=None):
        """Get configuration value with default."""
        return self.config.get(key, default)