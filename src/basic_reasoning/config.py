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
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                logger.info(f"✅ Loaded config from {self.config_path}")
            else:
                config = {}
                logger.warning(f"⚠️ Config file not found: {self.config_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            config = {}
        
        # Apply defaults
        default_config = self._get_default_config()
        merged_config = self._deep_merge(default_config, config)
        
        return merged_config

    def _get_default_config(self) -> Dict:
        """Get enhanced default configuration with processing section."""
        return {
            "data_dir": "data",
            "logs_dir": "logs",
            "models": {
                "embedding": {
                    "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "device": "cpu",
                    "batch_size": 16,
                    "use_medical_embedding": True,
                    "max_length": 512,
                    "trust_remote_code": False
                },
                "llm": {
                    "name": "mistral:7b-instruct",
                    "temperature": 0.7,
                    "context_window": 4096,
                    "device": "cpu",
                    "batch_size": 8
                },
                "kg_system": {
                    "enabled": True,
                    "device": "cpu"
                },
                "hierarchical_system": {
                    "enabled": True,
                    "device": "cpu"
                }
            },
            "processing": {
                "chunk_size": 512,
                "chunk_overlap": 100,
                "min_chunk_size": 50,
                "max_chunk_size": 1024,
                "enable_semantic_chunking": True,
                "preserve_sentence_boundaries": True,
                "hierarchy_tiers": {
                    "tier1": {
                        "name": "pattern_recognition",
                        "description": "Fast pattern matching for medical concepts",
                        "chunk_size": 256,
                        "overlap": 50
                    },
                    "tier2": {
                        "name": "hypothesis_testing", 
                        "description": "Systematic evidence collection",
                        "chunk_size": 512,
                        "overlap": 100
                    },
                    "tier3": {
                        "name": "confirmation",
                        "description": "Comprehensive verification",
                        "chunk_size": 1024,
                        "overlap": 150
                    }
                }
            },
            "retrieval": {
                "top_k": 5,
                "hybrid_search": True,
                "alpha": 0.5,
                "similarity_threshold": 0.1,
                "hierarchical_search": {
                    "tier1_k": 10,
                    "tier2_k": 5,
                    "tier3_k": 3,
                    "combine_strategy": "weighted"
                }
            },
            "prompts": {
                "system": """You are a medical knowledge assistant. Answer the multiple choice question based on the provided medical context.

Instructions:
1. Read the question and all answer options carefully
2. Use the provided medical context to determine the correct answer
3. Select ONLY ONE answer: A, B, C, D, or E
4. End your response with exactly: "Answer: [LETTER]"

Format your response as:
[Your reasoning based on the medical context]

Answer: [A/B/C/D/E]""",
                
                "tier1_system": """You are analyzing medical patterns and concepts. Focus on:
- Basic medical terminology and definitions
- Anatomical structures and functions
- Pattern recognition in symptoms
- Basic diagnostic criteria

Answer the multiple choice question with: Answer: [LETTER]""",
                
                "tier2_system": """You are performing clinical reasoning and hypothesis testing. Focus on:
- Pathophysiology and disease mechanisms
- Differential diagnosis considerations
- Clinical reasoning chains
- Evidence-based analysis

Answer the multiple choice question with: Answer: [LETTER]""",
                
                "tier3_system": """You are providing comprehensive medical verification. Focus on:
- Evidence-based medical facts
- Clinical guidelines and protocols
- Confirmatory diagnostic information
- Authoritative medical knowledge

Answer the multiple choice question with: Answer: [LETTER]""",
                
                "question_template": """Question: {question}

Medical Context:
{context}

Options:
{options}

Based on the medical context provided, select the correct answer."""
            },
            "vector_db": {
                "persist_directory": "data/vector_db",
                "collection_names": {
                    "tier1": "tier1_pattern_recognition",
                    "tier2": "tier2_hypothesis_testing", 
                    "tier3": "tier3_confirmation"
                },
                "distance_metric": "cosine",
                "max_batch_size": 1000
            },
            "web": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors_origins": ["*"]
            },
            "logging": {
                "level": "INFO",
                "format": "{time} | {level} | {name}:{function}:{line} - {message}",
                "file_rotation": "10 MB"
            },
            "evaluation": {
                "enable_medical_validation": True,
                "check_medical_terminology": True,
                "benchmarks": {
                    "mirage": {"enabled": True},
                    "medreason": {"enabled": True},
                    "pubmedqa": {"enabled": True},
                    "msmarco": {"enabled": True}
                }
            }
        }

    def _deep_merge(self, default: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.config["data_dir"],
            self.config["logs_dir"],
            f"{self.config['data_dir']}/vector_db",
            f"{self.config['data_dir']}/foundation",
            f"{self.config['data_dir']}/processed",
            f"{self.config['data_dir']}/raw"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"📁 Created directories: {directories}")

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_level = self.config["logging"]["level"]
        log_format = self.config["logging"]["format"]
        
        # Configure loguru
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            level=log_level,
            format=log_format,
            colorize=True
        )
        
        # Add file logging
        log_file = Path(self.config["logs_dir"]) / "hierarchical_system.log"
        logger.add(
            log_file,
            level=log_level,
            format=log_format,
            rotation=self.config["logging"]["file_rotation"],
            retention="7 days"
        )

    def get_data_dir(self, subdir: str = "") -> Path:
        """Get data directory path with optional subdirectory."""
        if subdir:
            data_dir = Path(self.config["data_dir"]) / subdir
        else:
            data_dir = Path(self.config["data_dir"])
        
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
    
    def get_processing_config(self) -> Dict:
        """Get processing configuration."""
        return self.config["processing"]

    def save(self) -> None:
        """Save configuration to file."""
        with open(self.config_path, "w", encoding="utf-8") as f:
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
            "valid_device": embedding_config.get("device") in ["cuda", "mps", "cpu"],
            "processing_config": "processing" in self.config
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