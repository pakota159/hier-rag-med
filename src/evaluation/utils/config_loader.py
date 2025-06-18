#!/usr/bin/env python3
"""
GPU-focused configuration loader for HierRAGMed evaluation on RunPod.
Simplified for GPU-only operation with intelligent config selection.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """GPU-specific configuration container."""
    device: str = "cuda"
    embedding_batch_size: int = 128
    llm_batch_size: int = 32
    mixed_precision: bool = True
    compile_models: bool = True
    memory_fraction: float = 0.85
    pin_memory: bool = True
    num_workers: int = 8


@dataclass
class EvaluationConfig:
    """Complete evaluation configuration."""
    platform: str
    results_dir: str
    data_dir: str
    gpu_config: GPUConfig
    models: Dict[str, Any]
    benchmarks: Dict[str, Any]
    performance: Dict[str, Any]
    logging: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ConfigLoader:
    """
    GPU-focused configuration loader for RunPod evaluation.
    Prioritizes GPU configurations and applies intelligent optimizations.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path(__file__).parent.parent / "configs"
        self.loaded_config = None
        
        # Configuration file priority (highest to lowest)
        self.config_files = [
            "gpu_runpod_config.yaml",      # Primary RunPod config
            "config.yaml",                 # Fallback config
            "evaluation_config.yaml",      # Alternative name
            "default_config.yaml"          # Last resort
        ]
    
    def load_config(self, config_path: Optional[Path] = None) -> EvaluationConfig:
        """
        Load and optimize configuration for GPU evaluation.
        
        Args:
            config_path: Specific config file path (optional)
            
        Returns:
            Optimized EvaluationConfig
        """
        if config_path:
            # Load specific config file
            raw_config = self._load_yaml_config(config_path)
            logger.info(f"📄 Loaded config from: {config_path}")
        else:
            # Try config files in priority order
            raw_config = self._load_priority_config()
        
        # Apply GPU optimizations
        optimized_config = self._optimize_for_gpu(raw_config)
        
        # Validate configuration
        self._validate_config(optimized_config)
        
        # Convert to structured config
        self.loaded_config = self._create_evaluation_config(optimized_config)
        
        logger.info(f"✅ Configuration loaded and optimized for GPU evaluation")
        return self.loaded_config
    
    def _load_priority_config(self) -> Dict[str, Any]:
        """Load configuration using priority order."""
        for config_file in self.config_files:
            config_path = self.config_dir / config_file
            
            if config_path.exists():
                try:
                    config = self._load_yaml_config(config_path)
                    logger.info(f"📄 Loaded config from: {config_path}")
                    return config
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {config_path}: {e}")
                    continue
        
        # If no config file found, use defaults
        logger.warning("⚠️ No config file found, using GPU defaults")
        return self._get_default_gpu_config()
    
    def _load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise ValueError("Configuration must be a dictionary")
            
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    def _optimize_for_gpu(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply GPU-specific optimizations to configuration."""
        # Ensure GPU device is set
        self._ensure_gpu_device(config)
        
        # Apply GPU batch size optimizations
        self._optimize_batch_sizes(config)
        
        # Enable GPU performance features
        self._enable_gpu_features(config)
        
        # Set GPU-specific paths
        self._set_gpu_paths(config)
        
        # Configure GPU monitoring
        self._configure_gpu_monitoring(config)
        
        logger.info("🔧 Applied GPU optimizations to configuration")
        return config
    
    def _ensure_gpu_device(self, config: Dict[str, Any]) -> None:
        """Ensure all models use GPU device."""
        models = config.setdefault("models", {})
        
        for model_name, model_config in models.items():
            if isinstance(model_config, dict):
                model_config["device"] = "cuda"
                
                # Remove non-GPU device settings
                if "device_settings" in model_config:
                    del model_config["device_settings"]
                if "auto_detect_device" in model_config:
                    del model_config["auto_detect_device"]
    
    def _optimize_batch_sizes(self, config: Dict[str, Any]) -> None:
        """Optimize batch sizes for GPU performance."""
        models = config.setdefault("models", {})
        
        # GPU-optimized batch sizes for RTX 4090
        gpu_batch_sizes = {
            "embedding": 128,
            "llm": 32
        }
        
        for model_name, optimal_batch in gpu_batch_sizes.items():
            if model_name in models and isinstance(models[model_name], dict):
                models[model_name]["batch_size"] = optimal_batch
        
        # Optimize benchmark batch sizes
        benchmarks = config.setdefault("benchmarks", {})
        benchmark_batch_sizes = {
            "mirage": 32,
            "medreason": 16,
            "pubmedqa": 64,
            "msmarco": 128
        }
        
        for benchmark_name, optimal_batch in benchmark_batch_sizes.items():
            if benchmark_name in benchmarks and isinstance(benchmarks[benchmark_name], dict):
                benchmarks[benchmark_name]["batch_size"] = optimal_batch
                benchmarks[benchmark_name]["num_workers"] = 8
                benchmarks[benchmark_name]["pin_memory"] = True
    
    def _enable_gpu_features(self, config: Dict[str, Any]) -> None:
        """Enable GPU performance features."""
        # Performance section
        performance = config.setdefault("performance", {})
        performance.update({
            "use_gpu_acceleration": True,
            "mixed_precision": True,
            "compile_models": True,
            "memory_efficient": True,
            "cuda_deterministic": False,
            "cuda_benchmark": True,
            "gpu_memory_fraction": 0.85,
            "parallel_processing": True,
            "max_workers": 8,
            "prefetch_factor": 2
        })
        
        # Model-specific GPU features
        models = config.setdefault("models", {})
        for model_name, model_config in models.items():
            if isinstance(model_config, dict) and model_name in ["kg_system", "hierarchical_system"]:
                model_config.update({
                    "mixed_precision": True,
                    "compile_model": True,
                    "memory_efficient": True,
                    "gradient_checkpointing": False  # Disabled for inference
                })
    
    def _set_gpu_paths(self, config: Dict[str, Any]) -> None:
        """Set appropriate paths for GPU environment."""
        # Check if running in container/RunPod
        if os.path.exists("/workspace"):
            # RunPod/container paths
            config["data_dir"] = "/workspace/data"
            config["results_dir"] = "/workspace/evaluation/results"
            config["cache_dir"] = "/workspace/evaluation/cache"
            config["logs_dir"] = "/workspace/evaluation/logs"
        else:
            # Local development paths
            config.setdefault("data_dir", "data")
            config.setdefault("results_dir", "evaluation/results")
            config.setdefault("cache_dir", "evaluation/cache")
            config.setdefault("logs_dir", "evaluation/logs")
    
    def _configure_gpu_monitoring(self, config: Dict[str, Any]) -> None:
        """Configure GPU monitoring settings."""
        monitoring = config.setdefault("monitoring", {})
        monitoring.update({
            "enable_gpu_monitoring": True,
            "gpu_poll_interval": 10,
            "log_gpu_memory": True,
            "log_gpu_utilization": True,
            "track_inference_time": True,
            "track_memory_usage": True,
            "track_throughput": True
        })
        
        # GPU-specific logging
        logging_config = config.setdefault("logging", {})
        logging_config.update({
            "log_gpu_stats": True,
            "gpu_stats_interval": 30,
            "log_timing": True,
            "log_memory_usage": True,
            "log_batch_sizes": True
        })
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate GPU configuration."""
        required_sections = ["models", "benchmarks", "performance"]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate GPU settings
        models = config.get("models", {})
        for model_name, model_config in models.items():
            if isinstance(model_config, dict):
                device = model_config.get("device", "")
                if device != "cuda":
                    logger.warning(f"⚠️ Model {model_name} not using CUDA device: {device}")
        
        # Validate paths exist or can be created
        for path_key in ["results_dir", "cache_dir", "logs_dir"]:
            if path_key in config:
                path = Path(config[path_key])
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.warning(f"⚠️ Cannot create directory {path}: {e}")
    
    def _create_evaluation_config(self, config: Dict[str, Any]) -> EvaluationConfig:
        """Create structured evaluation configuration."""
        # Extract GPU configuration
        gpu_config = GPUConfig(
            device="cuda",
            embedding_batch_size=config.get("models", {}).get("embedding", {}).get("batch_size", 128),
            llm_batch_size=config.get("models", {}).get("llm", {}).get("batch_size", 32),
            mixed_precision=config.get("performance", {}).get("mixed_precision", True),
            compile_models=config.get("performance", {}).get("compile_models", True),
            memory_fraction=config.get("performance", {}).get("gpu_memory_fraction", 0.85),
            pin_memory=True,
            num_workers=config.get("performance", {}).get("max_workers", 8)
        )
        
        return EvaluationConfig(
            platform=config.get("platform", "RunPod GPU"),
            results_dir=config.get("results_dir", "evaluation/results"),
            data_dir=config.get("data_dir", "data"),
            gpu_config=gpu_config,
            models=config.get("models", {}),
            benchmarks=config.get("benchmarks", {}),
            performance=config.get("performance", {}),
            logging=config.get("logging", {})
        )
    
    def _get_default_gpu_config(self) -> Dict[str, Any]:
        """Get default GPU configuration as fallback."""
        return {
            "platform": "RunPod GPU",
            "environment": "production",
            "gpu_optimized": True,
            
            "results_dir": "evaluation/results",
            "data_dir": "data",
            "cache_dir": "evaluation/cache",
            "logs_dir": "evaluation/logs",
            
            "models": {
                "embedding": {
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "device": "cuda",
                    "batch_size": 128,
                    "mixed_precision": True
                },
                "llm": {
                    "name": "mistral:7b-instruct",
                    "device": "cuda",
                    "batch_size": 32,
                    "mixed_precision": True
                },
                "kg_system": {
                    "enabled": True,
                    "device": "cuda",
                    "mixed_precision": True,
                    "compile_model": True
                },
                "hierarchical_system": {
                    "enabled": True,
                    "device": "cuda",
                    "mixed_precision": True,
                    "compile_model": True
                }
            },
            
            "benchmarks": {
                "mirage": {
                    "enabled": True,
                    "batch_size": 32,
                    "num_workers": 8,
                    "pin_memory": True
                },
                "medreason": {
                    "enabled": True,
                    "batch_size": 16,
                    "num_workers": 8,
                    "pin_memory": True
                },
                "pubmedqa": {
                    "enabled": True,
                    "batch_size": 64,
                    "num_workers": 8,
                    "pin_memory": True
                },
                "msmarco": {
                    "enabled": True,
                    "batch_size": 128,
                    "num_workers": 8,
                    "pin_memory": True
                }
            },
            
            "performance": {
                "use_gpu_acceleration": True,
                "mixed_precision": True,
                "compile_models": True,
                "memory_efficient": True,
                "gpu_memory_fraction": 0.85,
                "parallel_processing": True,
                "max_workers": 8
            },
            
            "logging": {
                "level": "INFO",
                "log_gpu_stats": True,
                "gpu_stats_interval": 30
            },
            
            "monitoring": {
                "enable_gpu_monitoring": True,
                "gpu_poll_interval": 10,
                "log_gpu_memory": True,
                "log_gpu_utilization": True
            }
        }
    
    def get_gpu_optimized_config(self, gpu_name: str = "RTX 4090") -> Dict[str, Any]:
        """
        Get GPU-specific optimized configuration.
        
        Args:
            gpu_name: Name of the GPU for optimization
            
        Returns:
            GPU-optimized configuration
        """
        base_config = self.loaded_config.to_dict() if self.loaded_config else self._get_default_gpu_config()
        
        # GPU-specific optimizations
        if "RTX 4090" in gpu_name or "4090" in gpu_name:
            # RTX 4090 optimizations (24GB VRAM)
            optimizations = {
                "embedding_batch_size": 128,
                "llm_batch_size": 32,
                "memory_fraction": 0.85,
                "benchmark_batches": {"mirage": 32, "medreason": 16, "pubmedqa": 64, "msmarco": 128},
                "max_workers": 8
            }
        elif "RTX 3080" in gpu_name or "3080" in gpu_name:
            # RTX 3080 optimizations (10GB VRAM)
            optimizations = {
                "embedding_batch_size": 64,
                "llm_batch_size": 16,
                "memory_fraction": 0.8,
                "benchmark_batches": {"mirage": 16, "medreason": 8, "pubmedqa": 32, "msmarco": 64},
                "max_workers": 6
            }
        elif "RTX 3090" in gpu_name or "3090" in gpu_name:
            # RTX 3090 optimizations (24GB VRAM)
            optimizations = {
                "embedding_batch_size": 96,
                "llm_batch_size": 24,
                "memory_fraction": 0.85,
                "benchmark_batches": {"mirage": 24, "medreason": 12, "pubmedqa": 48, "msmarco": 96},
                "max_workers": 8
            }
        elif "A100" in gpu_name:
            # A100 optimizations (40GB/80GB VRAM)
            optimizations = {
                "embedding_batch_size": 256,
                "llm_batch_size": 64,
                "memory_fraction": 0.9,
                "benchmark_batches": {"mirage": 64, "medreason": 32, "pubmedqa": 128, "msmarco": 256},
                "max_workers": 12
            }
        else:
            # Generic GPU optimizations
            optimizations = {
                "embedding_batch_size": 32,
                "llm_batch_size": 8,
                "memory_fraction": 0.7,
                "benchmark_batches": {"mirage": 8, "medreason": 4, "pubmedqa": 16, "msmarco": 32},
                "max_workers": 4
            }
        
        # Apply optimizations
        self._apply_gpu_optimizations(base_config, optimizations)
        
        logger.info(f"🎯 Applied {gpu_name} specific optimizations")
        return base_config
    
    def _apply_gpu_optimizations(self, config: Dict[str, Any], optimizations: Dict[str, Any]) -> None:
        """Apply GPU-specific optimizations to configuration."""
        # Update model batch sizes
        models = config.setdefault("models", {})
        if "embedding" in models:
            models["embedding"]["batch_size"] = optimizations["embedding_batch_size"]
        if "llm" in models:
            models["llm"]["batch_size"] = optimizations["llm_batch_size"]
        
        # Update performance settings
        performance = config.setdefault("performance", {})
        performance["gpu_memory_fraction"] = optimizations["memory_fraction"]
        performance["max_workers"] = optimizations["max_workers"]
        
        # Update benchmark batch sizes
        benchmarks = config.setdefault("benchmarks", {})
        for benchmark_name, batch_size in optimizations["benchmark_batches"].items():
            if benchmark_name in benchmarks:
                benchmarks[benchmark_name]["batch_size"] = batch_size
    
    def save_config(self, config: EvaluationConfig, output_path: Path) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            output_path: Output file path
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"💾 Configuration saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            raise
    
    def update_config_for_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration based on detected environment.
        
        Args:
            config: Base configuration
            
        Returns:
            Environment-specific configuration
        """
        # Detect environment
        environment = self._detect_environment()
        
        if environment == "runpod":
            # RunPod-specific settings
            config.update({
                "platform": "RunPod GPU",
                "data_dir": "/workspace/data",
                "results_dir": "/workspace/evaluation/results",
                "cache_dir": "/workspace/evaluation/cache",
                "logs_dir": "/workspace/evaluation/logs"
            })
            
            # RunPod Streamlit settings
            streamlit_config = config.setdefault("streamlit", {})
            streamlit_config.update({
                "server": {
                    "address": "0.0.0.0",
                    "port": 8501,
                    "enableXsrfProtection": False,
                    "enableCORS": True
                }
            })
            
        elif environment == "docker":
            # Docker container settings
            config.update({
                "platform": "Docker GPU",
                "data_dir": "/app/data",
                "results_dir": "/app/evaluation/results",
                "cache_dir": "/app/evaluation/cache",
                "logs_dir": "/app/evaluation/logs"
            })
            
        elif environment == "local":
            # Local development settings
            config.update({
                "platform": "Local GPU",
                "data_dir": "data",
                "results_dir": "evaluation/results",
                "cache_dir": "evaluation/cache",
                "logs_dir": "evaluation/logs"
            })
        
        logger.info(f"🌍 Updated configuration for {environment} environment")
        return config
    
    def _detect_environment(self) -> str:
        """Detect the current execution environment."""
        if os.path.exists("/workspace") and os.path.exists("/.runpod"):
            return "runpod"
        elif os.path.exists("/.dockerenv"):
            return "docker"
        else:
            return "local"
    
    def get_benchmark_config(self, benchmark_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            
        Returns:
            Benchmark-specific configuration
        """
        if not self.loaded_config:
            self.load_config()
        
        benchmarks = self.loaded_config.benchmarks
        
        if benchmark_name not in benchmarks:
            raise ValueError(f"Benchmark '{benchmark_name}' not found in configuration")
        
        return benchmarks[benchmark_name]
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model-specific configuration
        """
        if not self.loaded_config:
            self.load_config()
        
        models = self.loaded_config.models
        
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        
        return models[model_name]
    
    def validate_gpu_requirements(self) -> bool:
        """
        Validate that GPU requirements are met.
        
        Returns:
            True if GPU requirements are satisfied
        """
        try:
            import torch
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                logger.error("❌ CUDA not available")
                return False
            
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            required_memory = 8.0  # Minimum 8GB for evaluation
            
            if gpu_memory < required_memory:
                logger.error(f"❌ Insufficient GPU memory: {gpu_memory:.1f}GB < {required_memory}GB")
                return False
            
            # Check compute capability
            props = torch.cuda.get_device_properties(0)
            compute_capability = props.major + props.minor * 0.1
            min_compute_capability = 6.0  # Minimum for modern features
            
            if compute_capability < min_compute_capability:
                logger.warning(f"⚠️ Low compute capability: {compute_capability:.1f} < {min_compute_capability}")
            
            logger.info("✅ GPU requirements validated")
            return True
            
        except ImportError:
            logger.error("❌ PyTorch not available")
            return False
        except Exception as e:
            logger.error(f"❌ GPU validation failed: {e}")
            return False
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get Streamlit-specific configuration."""
        if not self.loaded_config:
            self.load_config()
        
        # Default Streamlit config
        streamlit_config = {
            "server": {
                "address": "0.0.0.0",
                "port": 8501,
                "enableXsrfProtection": False,
                "enableCORS": True
            },
            "theme": {
                "base": "dark",
                "primaryColor": "#FF6B6B",
                "backgroundColor": "#0E1117",
                "secondaryBackgroundColor": "#262730",
                "textColor": "#FAFAFA"
            },
            "maxUploadSize": 200,
            "maxMessageSize": 200,
            "show_gpu_metrics": True,
            "refresh_interval": 5,
            "enable_real_time_monitoring": True
        }
        
        # Update with config file settings if available
        config_dict = self.loaded_config.to_dict()
        if "streamlit" in config_dict:
            streamlit_config.update(config_dict["streamlit"])
        
        return streamlit_config


# Singleton instance for global access
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: Optional[Path] = None) -> ConfigLoader:
    """
    Get or create the global configuration loader instance.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    
    return _config_loader


def load_evaluation_config(config_path: Optional[Path] = None) -> EvaluationConfig:
    """
    Load evaluation configuration with GPU optimizations.
    
    Args:
        config_path: Specific config file path (optional)
        
    Returns:
        Loaded and optimized EvaluationConfig
    """
    loader = get_config_loader()
    return loader.load_config(config_path)


def get_gpu_config(gpu_name: str = "RTX 4090") -> Dict[str, Any]:
    """
    Get GPU-optimized configuration.
    
    Args:
        gpu_name: Name of the GPU for optimization
        
    Returns:
        GPU-optimized configuration
    """
    loader = get_config_loader()
    return loader.get_gpu_optimized_config(gpu_name)


def validate_gpu_environment() -> bool:
    """
    Validate GPU environment for evaluation.
    
    Returns:
        True if environment is valid
    """
    loader = get_config_loader()
    return loader.validate_gpu_requirements()


# Convenience functions
def create_streamlit_config_file(output_path: Path = None) -> None:
    """Create Streamlit configuration file for RunPod."""
    if output_path is None:
        output_path = Path(".streamlit/config.toml")
    
    loader = get_config_loader()
    streamlit_config = loader.get_streamlit_config()
    
    # Create .streamlit directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to TOML format
    toml_content = f"""[server]
address = "{streamlit_config['server']['address']}"
port = {streamlit_config['server']['port']}
enableXsrfProtection = {str(streamlit_config['server']['enableXsrfProtection']).lower()}
enableCORS = {str(streamlit_config['server']['enableCORS']).lower()}

[theme]
base = "{streamlit_config['theme']['base']}"
primaryColor = "{streamlit_config['theme']['primaryColor']}"
backgroundColor = "{streamlit_config['theme']['backgroundColor']}"
secondaryBackgroundColor = "{streamlit_config['theme']['secondaryBackgroundColor']}"
textColor = "{streamlit_config['theme']['textColor']}"

[browser]
gatherUsageStats = false

[runner]
magicEnabled = false
installTracer = false

[logger]
level = "info"

[client]
caching = true
displayEnabled = true

[global]
maxUploadSize = {streamlit_config['maxUploadSize']}
maxMessageSize = {streamlit_config['maxMessageSize']}
"""
    
    with open(output_path, 'w') as f:
        f.write(toml_content)
    
    logger.info(f"📄 Streamlit config created: {output_path}")