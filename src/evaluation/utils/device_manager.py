#!/usr/bin/env python3
"""
GPU-only Device Manager for HierRAGMed evaluation on RunPod.
Simplified for CUDA-only operation with RTX 4090 optimization.
"""

import os
import torch
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU information container."""
    name: str
    memory_total: float  # GB
    memory_free: float   # GB
    memory_used: float   # GB
    compute_capability: Tuple[int, int]
    cuda_version: str
    driver_version: str
    device_id: int = 0


class DeviceManager:
    """
    GPU-only device manager for RunPod evaluation.
    Handles CUDA device initialization, memory management, and optimization.
    """
    
    def __init__(self, device_id: int = 0, memory_fraction: float = 0.85):
        """
        Initialize GPU device manager.
        
        Args:
            device_id: CUDA device ID (default: 0)
            memory_fraction: Fraction of GPU memory to use (default: 0.85)
        """
        self.device_id = device_id
        self.memory_fraction = memory_fraction
        self.device = None
        self.gpu_info = None
        
        # Initialize GPU
        self._initialize_gpu()
    
    def _initialize_gpu(self) -> None:
        """Initialize GPU device and verify CUDA availability."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "❌ CUDA not available! This evaluation system requires GPU support. "
                "Please ensure CUDA is properly installed and a compatible GPU is available."
            )
        
        # Check if requested device exists
        if self.device_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"❌ GPU device {self.device_id} not found! "
                f"Available devices: {torch.cuda.device_count()}"
            )
        
        # Set device
        torch.cuda.set_device(self.device_id)
        self.device = torch.device(f"cuda:{self.device_id}")
        
        # Configure GPU optimizations
        self._configure_gpu()
        
        # Get GPU information
        self.gpu_info = self._get_gpu_info()
        
        logger.info(f"🚀 GPU initialized: {self.gpu_info.name}")
        logger.info(f"💾 GPU Memory: {self.gpu_info.memory_total:.1f} GB total")
        logger.info(f"🔥 CUDA Version: {self.gpu_info.cuda_version}")
        logger.info(f"⚡ Compute Capability: {self.gpu_info.compute_capability}")
    
    def _configure_gpu(self) -> None:
        """Configure GPU settings for optimal performance."""
        # Set CUDA environment variables for RTX 4090 optimization
        os.environ.update({
            "CUDA_VISIBLE_DEVICES": str(self.device_id),
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "CUDA_LAUNCH_BLOCKING": "0",  # Async CUDA operations
            "TOKENIZERS_PARALLELISM": "false",  # Avoid tokenizer warnings
        })
        
        # Enable CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul on Ampere
        torch.backends.cudnn.allow_tf32 = True        # Faster convolutions on Ampere
        torch.backends.cudnn.benchmark = True         # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False    # Faster but non-deterministic
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        logger.info("🔧 GPU optimizations configured")
    
    def _get_gpu_info(self) -> GPUInfo:
        """Get comprehensive GPU information."""
        props = torch.cuda.get_device_properties(self.device_id)
        
        # Memory information (in GB)
        memory_total = props.total_memory / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(self.device_id) / (1024**3)
        memory_allocated = torch.cuda.memory_allocated(self.device_id) / (1024**3)
        memory_free = memory_total - memory_reserved
        
        return GPUInfo(
            name=props.name,
            memory_total=memory_total,
            memory_free=memory_free,
            memory_used=memory_allocated,
            compute_capability=(props.major, props.minor),
            cuda_version=torch.version.cuda,
            driver_version=torch.cuda.get_device_properties(self.device_id).name,
            device_id=self.device_id
        )
    
    def get_device(self) -> torch.device:
        """Get the CUDA device."""
        return self.device
    
    def get_gpu_info(self) -> GPUInfo:
        """Get current GPU information."""
        return self._get_gpu_info()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get detailed GPU memory statistics."""
        memory_stats = torch.cuda.memory_stats(self.device_id)
        
        return {
            "allocated_gb": torch.cuda.memory_allocated(self.device_id) / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved(self.device_id) / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated(self.device_id) / (1024**3),
            "max_reserved_gb": torch.cuda.max_memory_reserved(self.device_id) / (1024**3),
            "total_gb": self.gpu_info.memory_total,
            "free_gb": self.gpu_info.memory_total - torch.cuda.memory_reserved(self.device_id) / (1024**3),
            "utilization_percent": (torch.cuda.memory_allocated(self.device_id) / torch.cuda.get_device_properties(self.device_id).total_memory) * 100,
            "num_alloc_retries": memory_stats.get("num_alloc_retries", 0),
            "num_ooms": memory_stats.get("num_ooms", 0)
        }
    
    def optimize_batch_size(self, base_batch_size: int, memory_per_sample: float) -> int:
        """
        Optimize batch size based on available GPU memory.
        
        Args:
            base_batch_size: Base batch size to start with
            memory_per_sample: Estimated memory per sample in GB
            
        Returns:
            Optimized batch size
        """
        available_memory = self.gpu_info.memory_total * self.memory_fraction
        max_batch_size = int(available_memory / max(memory_per_sample, 0.001))  # Avoid division by zero
        
        # Use minimum of base batch size and memory-limited batch size
        optimized_batch_size = min(base_batch_size, max_batch_size)
        
        # Ensure at least batch size of 1
        optimized_batch_size = max(1, optimized_batch_size)
        
        logger.info(f"📊 Batch size optimized: {base_batch_size} → {optimized_batch_size}")
        logger.info(f"💾 Available memory: {available_memory:.1f} GB")
        logger.info(f"📦 Memory per sample: {memory_per_sample:.3f} GB")
        
        return optimized_batch_size
    
    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        torch.cuda.empty_cache()
        logger.debug("🧹 GPU memory cache cleared")
    
    def reset_peak_memory_stats(self) -> None:
        """Reset peak memory statistics."""
        torch.cuda.reset_peak_memory_stats(self.device_id)
        logger.debug("📊 Peak memory statistics reset")
    
    def set_memory_fraction(self, fraction: float) -> None:
        """
        Set GPU memory fraction.
        
        Args:
            fraction: Fraction of GPU memory to use (0.0 to 1.0)
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("Memory fraction must be between 0.0 and 1.0")
        
        self.memory_fraction = fraction
        logger.info(f"💾 GPU memory fraction set to {fraction:.2f}")
    
    def check_memory_usage(self, threshold: float = 0.9) -> bool:
        """
        Check if GPU memory usage exceeds threshold.
        
        Args:
            threshold: Memory usage threshold (0.0 to 1.0)
            
        Returns:
            True if memory usage is below threshold
        """
        stats = self.get_memory_stats()
        usage_ratio = stats["allocated_gb"] / stats["total_gb"]
        
        if usage_ratio > threshold:
            logger.warning(f"⚠️ High GPU memory usage: {usage_ratio:.2f} > {threshold:.2f}")
            return False
        
        return True
    
    def get_optimal_settings(self) -> Dict[str, any]:
        """
        Get optimal settings based on GPU capabilities.
        
        Returns:
            Dictionary of optimal settings
        """
        # RTX 4090 specific optimizations
        if "RTX 4090" in self.gpu_info.name or "4090" in self.gpu_info.name:
            return {
                "embedding_batch_size": 128,
                "llm_batch_size": 32,
                "mixed_precision": True,
                "compile_models": True,
                "memory_efficient": True,
                "gradient_checkpointing": False,
                "max_workers": 8,
                "pin_memory": True
            }
        
        # RTX 3080/3090 optimizations
        elif any(gpu in self.gpu_info.name for gpu in ["RTX 3080", "RTX 3090", "3080", "3090"]):
            return {
                "embedding_batch_size": 64,
                "llm_batch_size": 16,
                "mixed_precision": True,
                "compile_models": True,
                "memory_efficient": True,
                "gradient_checkpointing": False,
                "max_workers": 6,
                "pin_memory": True
            }
        
        # A100/V100 optimizations
        elif any(gpu in self.gpu_info.name for gpu in ["A100", "V100"]):
            return {
                "embedding_batch_size": 256,
                "llm_batch_size": 64,
                "mixed_precision": True,
                "compile_models": True,
                "memory_efficient": True,
                "gradient_checkpointing": False,
                "max_workers": 12,
                "pin_memory": True
            }
        
        # Generic GPU optimizations
        else:
            # Calculate based on memory
            memory_gb = self.gpu_info.memory_total
            
            if memory_gb >= 20:  # High-end GPU
                embedding_batch = 96
                llm_batch = 24
            elif memory_gb >= 12:  # Mid-range GPU
                embedding_batch = 48
                llm_batch = 12
            elif memory_gb >= 8:   # Entry-level GPU
                embedding_batch = 24
                llm_batch = 6
            else:                  # Low-memory GPU
                embedding_batch = 12
                llm_batch = 3
            
            return {
                "embedding_batch_size": embedding_batch,
                "llm_batch_size": llm_batch,
                "mixed_precision": True,
                "compile_models": True,
                "memory_efficient": True,
                "gradient_checkpointing": memory_gb < 12,
                "max_workers": min(8, max(2, int(memory_gb / 3))),
                "pin_memory": True
            }
    
    def monitor_gpu(self) -> Dict[str, any]:
        """
        Monitor GPU status and return comprehensive metrics.
        
        Returns:
            Dictionary with GPU monitoring data
        """
        memory_stats = self.get_memory_stats()
        gpu_info = self.get_gpu_info()
        
        return {
            "device_name": gpu_info.name,
            "device_id": self.device_id,
            "cuda_version": gpu_info.cuda_version,
            "compute_capability": f"{gpu_info.compute_capability[0]}.{gpu_info.compute_capability[1]}",
            "memory_total_gb": memory_stats["total_gb"],
            "memory_allocated_gb": memory_stats["allocated_gb"],
            "memory_reserved_gb": memory_stats["reserved_gb"],
            "memory_free_gb": memory_stats["free_gb"],
            "memory_utilization_percent": memory_stats["utilization_percent"],
            "max_memory_allocated_gb": memory_stats["max_allocated_gb"],
            "max_memory_reserved_gb": memory_stats["max_reserved_gb"],
            "memory_allocation_retries": memory_stats["num_alloc_retries"],
            "out_of_memory_errors": memory_stats["num_ooms"],
            "memory_fraction_setting": self.memory_fraction,
            "device_available": True,
            "status": "healthy" if memory_stats["num_ooms"] == 0 else "warning"
        }
    
    def validate_environment(self) -> bool:
        """
        Validate GPU environment for evaluation.
        
        Returns:
            True if environment is valid for evaluation
        """
        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                logger.error("❌ CUDA not available")
                return False
            
            # Check minimum memory requirement (4GB)
            if self.gpu_info.memory_total < 4.0:
                logger.error(f"❌ Insufficient GPU memory: {self.gpu_info.memory_total:.1f} GB < 4.0 GB")
                return False
            
            # Check compute capability (minimum 6.0 for modern features)
            compute_version = self.gpu_info.compute_capability[0] + self.gpu_info.compute_capability[1] * 0.1
            if compute_version < 6.0:
                logger.warning(f"⚠️ Old compute capability: {compute_version:.1f} < 6.0")
            
            # Test basic GPU operations
            test_tensor = torch.randn(100, 100, device=self.device)
            result = torch.matmul(test_tensor, test_tensor)
            del test_tensor, result
            torch.cuda.empty_cache()
            
            logger.info("✅ GPU environment validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU environment validation failed: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of device manager."""
        if self.gpu_info:
            return (f"DeviceManager(device={self.device}, "
                   f"gpu='{self.gpu_info.name}', "
                   f"memory={self.gpu_info.memory_total:.1f}GB)")
        else:
            return f"DeviceManager(device={self.device})"


# Singleton instance for global access
_device_manager: Optional[DeviceManager] = None


def get_device_manager(device_id: int = 0, memory_fraction: float = 0.85) -> DeviceManager:
    """
    Get or create the global device manager instance.
    
    Args:
        device_id: CUDA device ID
        memory_fraction: Fraction of GPU memory to use
        
    Returns:
        DeviceManager instance
    """
    global _device_manager
    
    if _device_manager is None:
        _device_manager = DeviceManager(device_id, memory_fraction)
    
    return _device_manager


def initialize_gpu(device_id: int = 0, memory_fraction: float = 0.85) -> DeviceManager:
    """
    Initialize GPU for evaluation.
    
    Args:
        device_id: CUDA device ID
        memory_fraction: Fraction of GPU memory to use
        
    Returns:
        Initialized DeviceManager
    """
    device_manager = get_device_manager(device_id, memory_fraction)
    
    if not device_manager.validate_environment():
        raise RuntimeError("GPU environment validation failed")
    
    return device_manager


# Convenience functions
def get_device() -> torch.device:
    """Get the current CUDA device."""
    return get_device_manager().get_device()


def get_gpu_info() -> GPUInfo:
    """Get current GPU information."""
    return get_device_manager().get_gpu_info()


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    get_device_manager().clear_memory()


def monitor_gpu() -> Dict[str, any]:
    """Monitor GPU status."""
    return get_device_manager().monitor_gpu()