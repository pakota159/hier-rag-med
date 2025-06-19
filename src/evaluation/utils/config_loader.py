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
        
        # GPU optimization mappings
        self.gpu_optimizations = {
            "RTX 4090": {
                "embedding_batch_size": 128,
                "llm_batch_size": 32,
                "memory_fraction": 0.85,
                "max_workers": 8,
                "benchmark_batches": {
                    "mirage": 32,
                    "medreason": 16, 
                    "pubmedqa": 64,
                    "msmarco": 128
                }
            },
            "RTX 3090": {
                "embedding_batch_size": 64,
                "llm_batch_size": 16,
                "memory_fraction": 0.80,
                "max_workers": 6,
                "benchmark_batches": {
                    "mirage": 16,
                    "medreason": 8,
                    "pubmedqa": 32,
                    "msmarco": 64
                }
            },
            "A100": {
                "embedding_batch_size": 256,
                "llm_batch_size": 64,
                "memory_fraction": 0.90,
                "max_workers": 12,
                "benchmark_batches": {
                    "mirage": 64,
                    "medreason": 32,
                    "pubmedqa": 128,
                    "msmarco": 256
                }
            }
        }
    
    def load_config(self, config_path: Optional[Path] = None) -> EvaluationConfig:
        """
        Load evaluation configuration with environment optimization.
        
        Args:
            config_path: Specific config file path (optional)
            
        Returns:
            Optimized EvaluationConfig
        """
        # Try to load configuration from multiple sources
        config_dict = self._load_config_dict(config_path)
        
        # Update for current environment
        config_dict = self.update_config_for_environment(config_dict)
        
        # Create structured configuration
        self.loaded_config = self._create_evaluation_config(config_dict)
        
        logger.info(f"✅ Configuration loaded for {self.loaded_config.platform}")
        return self.loaded_config
    
    def _load_config_dict(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load configuration dictionary from file."""
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
        Update configuration based on detected environment - FIXED VERSION.
        
        Args:
            config: Base configuration
            
        Returns:
            Environment-optimized configuration
        """
        # Detect environment
        is_runpod = os.path.exists("/workspace")
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
            
            # Set RunPod-specific paths - FIXED
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
        """Set appropriate paths for RunPod environment - FIXED VERSION."""
        # RunPod/container paths - CORRECTED to use project directory
        project_base = "/workspace/hierragmed"  # Fixed: was missing hierragmed
        
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
        if not torch or not torch.cuda.is_available():
            return
        
        try:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🔥 Optimizing for GPU: {gpu_name}")
            
            # Get GPU-specific optimizations
            optimizations = None
            for gpu_type, opts in self.gpu_optimizations.items():
                if gpu_type in gpu_name:
                    optimizations = opts
                    break
            
            if not optimizations:
                # Generic optimizations for unknown GPU
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                optimizations = self._get_generic_gpu_optimizations(memory_gb)
            
            # Apply optimizations to config
            self._apply_optimizations_to_config(config, optimizations)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply GPU optimizations: {e}")
    
    def _apply_conservative_gpu_optimizations(self, config: Dict[str, Any]) -> None:
        """Apply conservative GPU optimizations for local development."""
        # Smaller batch sizes for local development
        models = config.setdefault("models", {})
        if "embedding" in models:
            models["embedding"]["batch_size"] = min(64, models["embedding"].get("batch_size", 32))
        if "llm" in models:
            models["llm"]["batch_size"] = min(16, models["llm"].get("batch_size", 8))
        
        # Conservative performance settings
        performance = config.setdefault("performance", {})
        performance.update({
            "use_gpu_acceleration": True,
            "mixed_precision": False,  # More conservative
            "compile_models": False,   # More conservative
            "memory_efficient": True,
            "gpu_memory_fraction": 0.7,  # Leave more headroom
            "parallel_processing": True,
            "max_workers": 4,  # Fewer workers
            "prefetch_factor": 1
        })
    
    def _get_generic_gpu_optimizations(self, memory_gb: float) -> Dict[str, Any]:
        """Get generic GPU optimizations based on memory."""
        if memory_gb >= 20:  # High-end GPU
            embedding_batch = 96
            llm_batch = 24
            workers = 8
        elif memory_gb >= 12:  # Mid-range GPU
            embedding_batch = 48
            llm_batch = 12
            workers = 6
        elif memory_gb >= 8:   # Entry-level GPU
            embedding_batch = 24
            llm_batch = 6
            workers = 4
        else:                  # Low-memory GPU
            embedding_batch = 12
            llm_batch = 3
            workers = 2
        
        return {
            "embedding_batch_size": embedding_batch,
            "llm_batch_size": llm_batch,
            "memory_fraction": 0.8,
            "max_workers": workers,
            "benchmark_batches": {
                "mirage": embedding_batch // 4,
                "medreason": embedding_batch // 8,
                "pubmedqa": embedding_batch // 2,
                "msmarco": embedding_batch
            }
        }
    
    def _apply_optimizations_to_config(self, config: Dict[str, Any], optimizations: Dict[str, Any]) -> None:
        """Apply optimizations to configuration."""
        # Update model batch sizes
        models = config.setdefault("models", {})
        if "embedding" in models:
            models["embedding"]["batch_size"] = optimizations["embedding_batch_size"]
        if "llm" in models:
            models["llm"]["batch_size"] = optimizations["llm_batch_size"]
        
        # Update performance settings
        performance = config.setdefault("performance", {})
        performance.update({
            "use_gpu_acceleration": True,
            "mixed_precision": True,
            "compile_models": True,
            "memory_efficient": True,
            "gpu_memory_fraction": optimizations["memory_fraction"],
            "parallel_processing": True,
            "max_workers": optimizations["max_workers"],
            "prefetch_factor": 2
        })
        
        # Update benchmark batch sizes
        benchmarks = config.setdefault("benchmarks", {})
        for benchmark_name, batch_size in optimizations["benchmark_batches"].items():
            if benchmark_name in benchmarks:
                benchmarks[benchmark_name]["batch_size"] = batch_size
    
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
    
    def _create_evaluation_config(self, config: Dict[str, Any]) -> EvaluationConfig:
        """Create structured evaluation configuration."""
        # Extract GPU configuration
        gpu_config = GPUConfig(
            device="cuda" if torch and torch.cuda.is_available() else "cpu",
            embedding_batch_size=config.get("models", {}).get("embedding", {}).get("batch_size", 128),
            llm_batch_size=config.get("models", {}).get("llm", {}).get("batch_size", 32),
            mixed_precision=config.get("performance", {}).get("mixed_precision", True),
            compile_models=config.get("performance", {}).get("compile_models", True),
            memory_fraction=config.get("performance", {}).get("gpu_memory_fraction", 0.85),
            pin_memory=True,
            num_workers=config.get("performance", {}).get("max_workers", 8)
        )
        
        return EvaluationConfig(
            platform=config.get("platform", "Generic"),
            results_dir=config.get("results_dir", "evaluation/results"),
            data_dir=config.get("data_dir", "data"),
            gpu_config=gpu_config,
            models=config.get("models", {}),
            benchmarks=config.get("benchmarks", {}),
            performance=config.get("performance", {}),
            logging=config.get("logging", {})
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration as fallback."""
        return {
            "platform": "Generic",
            "results_dir": "evaluation/results",
            "data_dir": "data",
            "cache_dir": "evaluation/cache",
            "logs_dir": "evaluation/logs",
            "models": {
                "embedding": {"device": "auto", "batch_size": 32},
                "llm": {"device": "auto", "batch_size": 16},
                "kg_system": {"enabled": True},
                "hierarchical_system": {"enabled": True}
            },
            "benchmarks": {
                "mirage": {"enabled": True},
                "medreason": {"enabled": True},
                "pubmedqa": {"enabled": True},
                "msmarco": {"enabled": True}
            },
            "performance": {
                "use_gpu_acceleration": False,
                "mixed_precision": False,
                "compile_models": False,
                "memory_efficient": True,
                "parallel_processing": True,
                "max_workers": 4
            },
            "logging": {"level": "INFO"}
        }
    
    def get_gpu_optimized_config(self, gpu_name: str = "RTX 4090") -> Dict[str, Any]:
        """Get GPU-optimized configuration for specific hardware."""
        config = self._get_default_config()
        
        if gpu_name in self.gpu_optimizations:
            optimizations = self.gpu_optimizations[gpu_name]
            self._apply_optimizations_to_config(config, optimizations)
        
        return config
    
    def validate_gpu_requirements(self) -> bool:
        """Validate that GPU requirements are met."""
        try:
            if not torch:
                logger.error("❌ PyTorch not available")
                return False
            
            if not torch.cuda.is_available():
                logger.error("❌ CUDA not available")
                return False
            
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            required_memory = 8.0  # Minimum 8GB for evaluation
            
            if gpu_memory < required_memory:
                logger.error(f"❌ Insufficient GPU memory: {gpu_memory:.1f}GB < {required_memory}GB")
                return False
            
            logger.info("✅ GPU requirements validated")
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