"""
Configuration loader for HierRAGMed evaluation system.
Handles GPU optimization, environment detection, and path management.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger

try:
    import torch
except ImportError:
    torch = None


@dataclass
class GPUConfig:
    """GPU-specific configuration settings."""
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
    platform: str = "Generic"
    results_dir: str = "evaluation/results"
    data_dir: str = "data"
    gpu_config: GPUConfig = None
    models: Dict = None
    benchmarks: Dict = None
    performance: Dict = None
    logging: Dict = None
    
    def __post_init__(self):
        if self.gpu_config is None:
            self.gpu_config = GPUConfig()
        if self.models is None:
            self.models = {}
        if self.benchmarks is None:
            self.benchmarks = {}
        if self.performance is None:
            self.performance = {}
        if self.logging is None:
            self.logging = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        config_dict = asdict(self)
        # Flatten GPU config
        config_dict["gpu"] = config_dict.pop("gpu_config")
        return config_dict


class ConfigLoader:
    """Load and optimize configuration for different environments."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration loader."""
        self.config_dir = config_dir or Path(__file__).parent.parent / "configs"
        self.loaded_config: Optional[EvaluationConfig] = None
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔧 Config loader initialized for: {self.config_dir}")
    
    def load_config(self, config_path: Optional[Path] = None) -> EvaluationConfig:
        """Load configuration from file or create default."""
        # Load base configuration
        base_config = self._load_base_config(config_path)
        
        # Apply environment optimizations
        optimized_config = self.update_config_for_environment(base_config)
        
        # Convert to EvaluationConfig
        eval_config = self._create_evaluation_config(optimized_config)
        
        self.loaded_config = eval_config
        return eval_config
    
    def _load_base_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load base configuration from file with fallback."""
        config_paths = []
        
        if config_path:
            config_paths.append(config_path)
        
        # Priority order for config files
        config_paths.extend([
            self.config_dir / "gpu_runpod_config.yaml",
            self.config_dir.parent / "config.yaml",
            Path(__file__).parent.parent / "config.yaml"
        ])
        
        for path in config_paths:
            try:
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    logger.info(f"📄 Loaded config from: {path}")
                    return config
            except Exception as e:
                logger.warning(f"⚠️ Failed to load config from {path}: {e}")
                continue
        
        # Fallback to default configuration
        logger.warning("⚠️ Using default configuration")
        return self._get_default_config()
    
    def update_config_for_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration based on detected environment.
        
        Args:
            config: Base configuration
            
        Returns:
            Environment-optimized configuration
        """
        # Detect environment
        is_runpod = os.path.exists("/workspace") or "RUNPOD_POD_ID" in os.environ
        is_docker = os.path.exists("/.dockerenv")
        has_gpu = torch.cuda.is_available() if torch is not None else False
        
        logger.info(f"🔍 Environment Detection:")
        logger.info(f"   RunPod: {is_runpod}")
        logger.info(f"   Docker: {is_docker}")
        logger.info(f"   GPU Available: {has_gpu}")
        
        # Apply environment-specific settings
        if is_runpod:
            logger.info("🚀 Applying RunPod optimizations...")
            # Use RunPod environment settings
            env_config = config.get("environments", {}).get("cloud", {})
            config.update(env_config)
            
            # Set RunPod-specific paths
            self._set_runpod_paths(config)
            
            # Enable GPU optimizations if available
            if has_gpu:
                self._apply_gpu_optimizations(config)
                self._configure_gpu_monitoring(config)
                
        elif is_docker:
            logger.info("🐳 Applying Docker optimizations...")
            # Use Docker environment settings
            env_config = config.get("environments", {}).get("docker", {})
            config.update(env_config)
            
            # Docker paths
            config.update({
                "data_dir": "/app/data",
                "results_dir": "/app/evaluation/results",
                "cache_dir": "/app/evaluation/cache",
                "logs_dir": "/app/evaluation/logs"
            })
            
            if has_gpu:
                self._apply_gpu_optimizations(config)
                
        else:
            logger.info("💻 Applying local development settings...")
            # Use local environment settings  
            env_config = config.get("environments", {}).get("local", {})
            config.update(env_config)
            
            # Local paths (relative)
            self._set_local_paths(config)
            
            # Moderate GPU optimizations for local development
            if has_gpu:
                self._apply_conservative_gpu_optimizations(config)
        
        # Update platform identifier
        config["platform"] = "RunPod GPU" if is_runpod and has_gpu else "Local GPU" if has_gpu else "Local CPU"
        config["environment"] = "production" if is_runpod else "development"
        config["gpu_optimized"] = has_gpu
        
        logger.info(f"✅ Configuration updated for {config['platform']} environment")
        return config
    
    def _set_runpod_paths(self, config: Dict[str, Any]) -> None:
        """Set appropriate paths for RunPod environment."""
        # RunPod/container paths - Fixed to use project directory
        project_base = "/workspace/hierragmed"
        
        # Verify project directory exists
        if os.path.exists(project_base):
            config["data_dir"] = f"{project_base}/data"
            config["results_dir"] = f"{project_base}/evaluation/results"
            config["cache_dir"] = f"{project_base}/evaluation/cache"
            config["logs_dir"] = f"{project_base}/evaluation/logs"
            logger.info(f"📁 Using project directory: {project_base}")
        else:
            # Fallback if project not in expected location
            logger.warning(f"⚠️ Project directory {project_base} not found, using /workspace")
            config["data_dir"] = "/workspace/data"
            config["results_dir"] = "/workspace/evaluation/results"
            config["cache_dir"] = "/workspace/evaluation/cache"
            config["logs_dir"] = "/workspace/evaluation/logs"
        
        # Create directories if they don't exist
        self._ensure_directories_exist(config)
    
    def _set_local_paths(self, config: Dict[str, Any]) -> None:
        """Set paths for local development."""
        config.setdefault("data_dir", "data")
        config.setdefault("results_dir", "evaluation/results")
        config.setdefault("cache_dir", "evaluation/cache")
        config.setdefault("logs_dir", "evaluation/logs")
        
        # Create directories if they don't exist
        self._ensure_directories_exist(config)
    
    def _ensure_directories_exist(self, config: Dict[str, Any]) -> None:
        """Create directories if they don't exist."""
        for dir_key in ["data_dir", "results_dir", "cache_dir", "logs_dir"]:
            if dir_key in config:
                dir_path = Path(config[dir_key])
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"📁 Ensured directory exists: {dir_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not create directory {dir_path}: {e}")
    
    def _apply_gpu_optimizations(self, config: Dict[str, Any]) -> None:
        """Apply GPU optimizations based on detected hardware."""
        # Get GPU memory info
        if torch and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"🎮 GPU Memory: {gpu_memory_gb:.1f} GB")
            
            # Adjust batch sizes based on GPU memory
            if gpu_memory_gb >= 20:  # RTX 4090 or better
                embedding_batch = 128
                llm_batch = 32
            elif gpu_memory_gb >= 12:  # RTX 3080/4070 tier
                embedding_batch = 64
                llm_batch = 16
            else:  # Lower-end GPUs
                embedding_batch = 32
                llm_batch = 8
        else:
            embedding_batch = 16
            llm_batch = 4
        
        # Update model configurations
        if "models" not in config:
            config["models"] = {}
        
        models_config = config["models"]
        
        # Embedding model settings
        if "embedding" not in models_config:
            models_config["embedding"] = {}
        models_config["embedding"].update({
            "device": "cuda",
            "batch_size": embedding_batch,
            "pin_memory": True,
            "use_mixed_precision": True
        })
        
        # LLM settings
        if "llm" not in models_config:
            models_config["llm"] = {}
        models_config["llm"].update({
            "device": "cuda",
            "batch_size": llm_batch,
            "use_flash_attention": True
        })
        
        # System-level optimizations
        config.setdefault("performance", {}).update({
            "num_workers": 8,
            "pin_memory": True,
            "non_blocking": True,
            "gpu_memory_fraction": 0.85
        })
        
        logger.info(f"🚀 Applied GPU optimizations: embedding_batch={embedding_batch}, llm_batch={llm_batch}")
    
    def _apply_conservative_gpu_optimizations(self, config: Dict[str, Any]) -> None:
        """Apply conservative GPU optimizations for local development."""
        if "models" not in config:
            config["models"] = {}
        
        models_config = config["models"]
        
        # Conservative settings for local development
        if "embedding" not in models_config:
            models_config["embedding"] = {}
        models_config["embedding"].update({
            "device": "cuda" if torch and torch.cuda.is_available() else "cpu",
            "batch_size": 32,
            "pin_memory": False
        })
        
        if "llm" not in models_config:
            models_config["llm"] = {}
        models_config["llm"].update({
            "device": "cuda" if torch and torch.cuda.is_available() else "cpu",
            "batch_size": 8
        })
        
        config.setdefault("performance", {}).update({
            "num_workers": 4,
            "pin_memory": False,
            "gpu_memory_fraction": 0.7
        })
        
        logger.info("💻 Applied conservative GPU optimizations for local development")
    
    def _configure_gpu_monitoring(self, config: Dict[str, Any]) -> None:
        """Configure GPU monitoring settings."""
        config.setdefault("monitoring", {}).update({
            "gpu_metrics": True,
            "memory_tracking": True,
            "performance_logging": True,
            "interval_seconds": 10
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data_dir": "data",
            "results_dir": "evaluation/results",
            "cache_dir": "evaluation/cache",
            "logs_dir": "evaluation/logs",
            "models": {
                "embedding": {
                    "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "device": "cpu",
                    "batch_size": 16,
                    "max_length": 512
                },
                "llm": {
                    "name": "mistral:7b-instruct",
                    "device": "cpu",
                    "batch_size": 8,
                    "temperature": 0.7
                },
                "hierarchical_system": {"enabled": True},
                "kg_system": {"enabled": True}
            },
            "processing": {
                "chunk_size": 512,
                "chunk_overlap": 100,
                "hierarchy_tiers": {
                    "tier1": {"name": "pattern_recognition", "chunk_size": 256},
                    "tier2": {"name": "hypothesis_testing", "chunk_size": 512},
                    "tier3": {"name": "confirmation", "chunk_size": 1024}
                }
            },
            "benchmarks": {
                "mirage": {"enabled": True},
                "medreason": {"enabled": True},
                "pubmedqa": {"enabled": True},
                "msmarco": {"enabled": True}
            },
            "logging": {
                "level": "INFO",
                "format": "{time} | {level} | {message}"
            },
            "performance": {
                "num_workers": 4,
                "pin_memory": False,
                "gpu_memory_fraction": 0.8
            },
            "environments": {
                "local": {
                    "max_concurrent_requests": 5,
                    "cache_size": "1GB"
                },
                "cloud": {
                    "max_concurrent_requests": 20,
                    "cache_size": "5GB"
                },
                "docker": {
                    "max_concurrent_requests": 10,
                    "cache_size": "2GB"
                }
            }
        }
    
    def _create_evaluation_config(self, config_dict: Dict[str, Any]) -> EvaluationConfig:
        """Create EvaluationConfig from dictionary."""
        # Extract GPU configuration
        gpu_config = GPUConfig(
            device=config_dict.get("models", {}).get("embedding", {}).get("device", "cpu"),
            embedding_batch_size=config_dict.get("models", {}).get("embedding", {}).get("batch_size", 16),
            llm_batch_size=config_dict.get("models", {}).get("llm", {}).get("batch_size", 8),
            memory_fraction=config_dict.get("performance", {}).get("gpu_memory_fraction", 0.8),
            num_workers=config_dict.get("performance", {}).get("num_workers", 4),
            pin_memory=config_dict.get("performance", {}).get("pin_memory", False)
        )
        
        return EvaluationConfig(
            platform=config_dict.get("platform", "Generic"),
            results_dir=config_dict.get("results_dir", "evaluation/results"),
            data_dir=config_dict.get("data_dir", "data"),
            gpu_config=gpu_config,
            models=config_dict.get("models", {}),
            benchmarks=config_dict.get("benchmarks", {}),
            performance=config_dict.get("performance", {}),
            logging=config_dict.get("logging", {})
        )
    
    def validate_gpu_requirements(self) -> bool:
        """Validate GPU requirements for evaluation."""
        if not torch:
            logger.error("❌ PyTorch not available")
            return False
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available, falling back to CPU")
            return True  # CPU is still valid
        
        try:
            # Test GPU access
            device = torch.device("cuda")
            torch.cuda.empty_cache()
            
            # Check memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024**3)
            
            if memory_gb < 4:
                logger.warning(f"⚠️ Low GPU memory: {memory_gb:.1f} GB")
                return False
            
            logger.info(f"✅ GPU validation passed: {memory_gb:.1f} GB available")
            return True
            
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
        
        return streamlit_config


# Singleton instance for global access
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: Optional[Path] = None) -> ConfigLoader:
    """Get or create the global configuration loader instance."""
    global _config_loader
    
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    
    return _config_loader


def load_evaluation_config(config_path: Optional[Path] = None) -> EvaluationConfig:
    """Load evaluation configuration with GPU optimizations."""
    loader = get_config_loader()
    return loader.load_config(config_path)


def validate_gpu_environment() -> bool:
    """Validate GPU environment for evaluation."""
    loader = get_config_loader()
    return loader.validate_gpu_requirements()


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