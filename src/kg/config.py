"""
Smart configuration module for KG HierRAGMed.
Auto-detects environment and loads appropriate config.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import yaml
import os
import torch
from loguru import logger


class Config:
    """Smart configuration manager that adapts to environment."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration with auto-detection."""
        
        # Determine config directory
        if config_path is None:
            self.config_dir = Path(__file__).parent
        elif config_path.is_file():
            self.config_dir = config_path.parent
        else:
            self.config_dir = config_path
        
        self.environment = self._detect_environment()
        self.config_path = self._get_config_path()
        self.config = self._load_config()

        # Create directories
        self.get_data_dir("logs")
        
        # Set up logging
        self._setup_logging()

    def _detect_environment(self) -> str:
        """Auto-detect the current environment."""
        
        # Check for RunPod environment
        if os.path.exists("/workspace") and "runpod" in os.environ.get("HOSTNAME", "").lower():
            return "runpod_gpu"
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            return "cuda_gpu"
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps_local"
        
        # Default to CPU
        return "cpu_local"

    def _get_config_path(self) -> Path:
        """Get the appropriate config file path based on environment."""
        
        # Priority order for config files
        config_candidates = []
        
        if self.environment in ["runpod_gpu", "cuda_gpu"]:
            # GPU environments - prefer GPU config
            config_candidates = [
                self.config_dir / "config_gpu.yaml",
                self.config_dir / "config_cuda.yaml",
                self.config_dir / "config.yaml"
            ]
        elif self.environment == "mps_local":
            # Apple Silicon - prefer MPS config
            config_candidates = [
                self.config_dir / "config_mps.yaml",
                self.config_dir / "config_local.yaml",
                self.config_dir / "config.yaml"
            ]
        else:
            # CPU fallback
            config_candidates = [
                self.config_dir / "config_cpu.yaml",
                self.config_dir / "config_local.yaml",
                self.config_dir / "config.yaml"
            ]
        
        # Return first existing config
        for config_path in config_candidates:
            if config_path.exists():
                logger.info(f"🎯 KG Config: {config_path.name} (environment: {self.environment})")
                return config_path
        
        # If no specific config found, use default
        default_path = self.config_dir / "config.yaml"
        logger.warning(f"⚠️ Using default KG config: {default_path}")
        return default_path

    def _load_config(self) -> Dict:
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.warning(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # Apply environment-specific optimizations
        config = self._optimize_for_environment(config)
        
        logger.info(f"✅ Loaded KG config for {self.environment} environment")
        return config

    def _optimize_for_environment(self, config: Dict) -> Dict:
        """Apply environment-specific optimizations to config."""
        
        # Ensure models section exists
        if "models" not in config:
            config["models"] = {}
        
        # Auto-configure device settings based on detected environment
        if self.environment in ["runpod_gpu", "cuda_gpu"]:
            # GPU optimizations
            if "embedding" in config["models"]:
                config["models"]["embedding"]["device"] = "cuda"
                config["models"]["embedding"]["batch_size"] = max(
                    config["models"]["embedding"].get("batch_size", 16), 32
                )
            if "llm" in config["models"]:
                config["models"]["llm"]["device"] = "cuda"
        
        elif self.environment == "mps_local":
            # Apple Silicon optimizations
            if "embedding" in config["models"]:
                config["models"]["embedding"]["device"] = "mps"
            if "llm" in config["models"]:
                config["models"]["llm"]["device"] = "mps"
        
        else:
            # CPU fallback
            if "embedding" in config["models"]:
                config["models"]["embedding"]["device"] = "cpu"
                config["models"]["embedding"]["batch_size"] = min(
                    config["models"]["embedding"].get("batch_size", 16), 8
                )
            if "llm" in config["models"]:
                config["models"]["llm"]["device"] = "cpu"
        
        # Add environment metadata
        config["_environment"] = {
            "detected": self.environment,
            "config_file": self.config_path.name,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
        
        return config

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        
        # Auto-detect device
        if self.environment in ["runpod_gpu", "cuda_gpu"]:
            device = "cuda"
            batch_size = 32
        elif self.environment == "mps_local":
            device = "mps"
            batch_size = 16
        else:
            device = "cpu"
            batch_size = 8
        
        return {
            "data_dir": "data/kg_raw",
            "models": {
                "embedding": {
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "device": device,
                    "batch_size": batch_size
                },
                "llm": {
                    "name": "mistral:7b-instruct",
                    "temperature": 0.7,
                    "context_window": 4096,
                    "device": device
                }
            },
            "retrieval": {
                "top_k": 10,
                "hybrid_search": True,
                "alpha": 0.5
            },
            "processing": {
                "chunk_size": 512,
                "chunk_overlap": 100
            },
            "prompts": {
                "system": """You are an advanced medical assistant with access to comprehensive medical literature, 
clinical documentation, and medical terminology. Use this knowledge to provide accurate, 
evidence-based responses. Always prioritize patient safety and cite relevant medical sources.""",
                "system_with_citations": """You are an advanced medical assistant with access to PubMed research, clinical documentation,
and MeSH medical terminology. Provide evidence-based answers with proper citations using 
document numbers. Distinguish between research evidence [1], clinical documentation [2], 
and medical terminology [3] when citing sources."""
            },
            "web": {
                "host": "0.0.0.0",
                "port": 8001,
                "cors_origins": ["*"]
            }
        }

    def _setup_logging(self):
        """Setup logging with environment-specific settings."""
        log_dir = self.get_data_dir("logs")
        log_file = log_dir / f"kg_system_{self.environment}.log"
        
        logger.add(
            log_file,
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | KG | {message}"
        )

    def get_data_dir(self, subdir: str) -> Path:
        """Get data directory path."""
        data_dir = Path(self.config["data_dir"]) / subdir
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def save(self) -> None:
        """Save configuration to file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Saved KG configuration to {self.config_path}")

    def get_device_info(self) -> Dict:
        """Get current device information."""
        info = {
            "environment": self.environment,
            "config_file": self.config_path.name,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        
        return info

    def __getitem__(self, key: str):
        """Get configuration value."""
        return self.config[key]

    def __setitem__(self, key: str, value):
        """Set configuration value."""
        self.config[key] = value

    def get(self, key: str, default=None):
        """Get configuration value with default."""
        return self.config.get(key, default)