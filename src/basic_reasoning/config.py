"""
Configuration for Basic Reasoning - Hierarchical Diagnostic RAG System
Uses data/foundation/ exclusively
"""

from pathlib import Path
from typing import Dict, Optional
import yaml
from loguru import logger


class Config:
    """Configuration manager for Basic Reasoning system."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize basic reasoning configuration."""
        self.config_path = config_path or Path(__file__).parent / "config.yaml"
        self.config = self._load_config()

        # Create directories
        self.get_data_dir("logs")
        
        # Set up logging
        logger.add(
            self.get_data_dir("logs") / "basic_reasoning.log",
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
        """Get default hierarchical configuration."""
        return {
            "data_dir": "data/foundation",  # Foundation datasets only
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
            "hierarchical_retrieval": {
                "tier1_top_k": 5,   # Pattern Recognition
                "tier2_top_k": 3,   # Hypothesis Testing
                "tier3_top_k": 2,   # Confirmation
                "enable_evidence_stratification": True,
                "enable_temporal_weighting": True
            },
            "processing": {
                "chunk_size": 512,
                "chunk_overlap": 100
            },
            "prompts": {
                "system": "You are a hierarchical diagnostic reasoning assistant using three-tier clinical decision patterns.",
                "tier1_prompt": "Identify relevant medical patterns and initial differential diagnoses.",
                "tier2_prompt": "Test hypotheses using evidence-based reasoning chains.", 
                "tier3_prompt": "Confirm diagnosis with comprehensive clinical evidence."
            },
            "web": {
                "host": "0.0.0.0",
                "port": 8503  # Different port from simple (8501) and kg (8502)
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
        logger.info(f"Saved basic reasoning configuration to {self.config_path}")

    def __getitem__(self, key: str):
        """Get configuration value."""
        return self.config[key]