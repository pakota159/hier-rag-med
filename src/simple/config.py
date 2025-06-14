"""
Configuration module for HierRAGMed.
"""

from pathlib import Path
from typing import Dict, Optional

import yaml
from loguru import logger


class Config:
    """Configuration manager for HierRAGMed."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration."""
        self.config_path = config_path or Path("config.yaml")
        self.config = self._load_config()

        # Create directories
        self.get_data_dir("logs")
        
        # Set up logging
        logger.add(
            self.get_data_dir("logs") / "hierragmed.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )

    def _load_config(self) -> Dict:
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.warning(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "data_dir": "data",
            "models": {
                "embedding": {
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "device": "mps",  # M1 Mac
                    "batch_size": 16
                },
                "llm": {
                    "name": "mistral:7b-instruct",
                    "temperature": 0.7,
                    "context_window": 4096
                }
            },
            "retrieval": {
                "top_k": 5,
                "hybrid_search": True,
                "alpha": 0.5
            },
            "processing": {
                "chunk_size": 512,
                "chunk_overlap": 100
            },
            "prompts": {
                "system": "You are a helpful medical assistant. Use the provided context to answer the question accurately and concisely.",
                "system_with_citations": "You are a helpful medical assistant. Use the provided context to answer the question accurately and concisely. Cite your sources using the provided document numbers."
            }
        }

    def get_data_dir(self, subdir: str) -> Path:
        """Get data directory path."""
        data_dir = Path(self.config["data_dir"]) / subdir
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def save(self) -> None:
        """Save configuration to file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Saved configuration to {self.config_path}")

    def __getitem__(self, key: str):
        """Get configuration value."""
        return self.config[key]