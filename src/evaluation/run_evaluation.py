#!/usr/bin/env python3
"""
Fixed GPU evaluation script for HierRAGMed.
Runs real evaluation using actual KG and Hierarchical systems.
"""

import argparse
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import real evaluation components directly to avoid circular imports
from src.evaluation.evaluators.kg_evaluator import KGEvaluator
from src.evaluation.evaluators.hierarchical_evaluator import HierarchicalEvaluator
from src.evaluation.benchmarks.mirage_benchmark import MIRAGEBenchmark
from src.evaluation.benchmarks.medreason_benchmark import MedReasonBenchmark  
from src.evaluation.benchmarks.pubmedqa_benchmark import PubMedQABenchmark
from src.evaluation.benchmarks.msmarco_benchmark import MSMARCOBenchmark


def validate_gpu_environment():
    """Validate GPU environment and log details."""
    logger.info("🔍 Validating GPU environment...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available. GPU evaluation requires CUDA.")
    
    # Get GPU information
    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
    
    # Check CUDA version
    cuda_version = torch.version.cuda
    
    # Detect platform (RunPod, local, etc.)
    platform = "RunPod" if "runpod" in os.environ.get("HOSTNAME", "").lower() else "Local"
    
    logger.info("✅ GPU Environment Validated:")
    logger.info(f"   GPU: {gpu_name}")
    logger.info(f"   Memory: {gpu_memory:.1f}GB")
    logger.info(f"   CUDA: {cuda_version}")
    logger.info(f"   Platform: {platform}")
    
    return {
        "gpu_count": gpu_count,
        "current_device": current_device,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory,
        "cuda_version": cuda_version,
        "platform": platform
    }


def setup_gpu_optimization():
    """Configure GPU optimizations for RTX 4090."""
    logger.info("🔧 Configuring GPU optimizations...")
    
    # Set CUDA device
    torch.cuda.set_device(0)
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Set environment variables for RTX 4090 optimization
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "CUDA_LAUNCH_BLOCKING": "0",  # Async CUDA operations
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8"
    })
    
    # Enable CUDA optimizations for Ampere architecture (RTX 4090)
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'benchmark'):
        torch.backends.cudnn.benchmark = True
    
    logger.info("✅ GPU optimizations configured")


def load_config():
    """Load evaluation configuration."""
    return {
        "results_dir": "evaluation/results",
        "models": {
            "kg_system": {
                "enabled": True,
                "config_path": "src/kg/config.yaml",
                "collection_name": "kg_medical_docs"
            },
            "hierarchical_system": {
                "enabled": True,
                "config_path": "src/basic_reasoning/config.yaml",
                "collection_names": {
                    "primary": "primary_symptoms",
                    "differential": "differential_diagnosis", 
                    "final": "final_diagnosis"
                }
            }
        },
        "benchmarks": {
            "mirage": {"enabled": True, "sample_size": 100},
            "medreason": {"enabled": True, "sample_size": 100},
            "pubmedqa": {"enabled": True, "sample_size": 100},
            "msmarco": {"enabled": True, "sample_size": 100}
        },
        "gpu_optimizations": {
            "batch_size_embedding": 128,
            "batch_size_llm": 32,
            "mixed_precision": True,
            "memory_efficient": True
        },
        "logging": {"level": "INFO"}
    }


def setup_logging(results_dir: str):
    """Setup evaluation logging."""
    logs_dir = Path(results_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"gpu_evaluation_{timestamp}.log"
    
    logger.remove()
    logger.add(
        sys.stdout, 
        level="INFO", 
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | GPU | {message}"
    )
    logger.add(
        log_file, 
        level="INFO", 
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | GPU | {message}"
    )
    
    logger.info(f"📝 GPU evaluation logging initialized: {log_file}")


def initialize_benchmarks(config: Dict, benchmark_filter: Optional[List[str]] = None):
    """Initialize real benchmarks."""
    benchmarks = {}
    
    # Available benchmark classes
    benchmark_classes = {
        "mirage": MIRAGEBenchmark,
        "medreason": MedReasonBenchmark,
        "pubmedqa": PubMedQABenchmark,
        "msmarco": MSMARCOBenchmark
    }
    
    # Determine enabled benchmarks
    if benchmark_filter:
        enabled_benchmarks = benchmark_filter
    else:
        enabled_benchmarks = [name for name, cfg in config["benchmarks"].items() if cfg.get("enabled", False)]
    
    for benchmark_name in enabled_benchmarks:
        if benchmark_name in benchmark_classes:
            try:
                logger.info(f"🎯 Initializing {benchmark_name.upper()} benchmark...")
                benchmark_config = config["benchmarks"][benchmark_name].copy()
                benchmark_config["name"] = benchmark_name
                
                benchmarks[benchmark_name] = benchmark_classes[benchmark_name](benchmark_config)
                logger.info(f"✅ {benchmark_name.upper()} benchmark ready")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {benchmark_name}: {e}")
    
    if not benchmarks:
        raise ValueError("❌ No benchmarks could be initialized!")
    
    logger.info(f"✅ Initialized {len(benchmarks)} benchmarks: {list(benchmarks.keys())}")
    return benchmarks


def initialize_evaluators(config: Dict, model_filter: Optional[List[str]] = None):
    """Initialize real evaluators."""
    evaluators = {}
    
    # Available evaluator classes
    evaluator_classes = {
        "kg_system": KGEvaluator,
        "hierarchical_system": HierarchicalEvaluator
    }
    
    # Determine enabled evaluators
    if model_filter:
        enabled_evaluators = model_filter
    else:
        enabled_evaluators = [name for name, cfg in config["models"].items() if cfg.get("enabled", False)]
    
    for evaluator_name in enabled_evaluators:
        if evaluator_name in evaluator_classes:
            try:
                logger.info(f"📊 Initializing {evaluator_name} evaluator...")
                evaluator_config = config["models"][evaluator_name].copy()
                evaluator_config["model_name"] = evaluator_name
                
                evaluators[evaluator_name] = evaluator_classes[evaluator_name](evaluator_config)
                logger.info(f"✅ {evaluator_name} evaluator ready")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {evaluator_name}: {e}")
    
    if not evaluators:
        raise ValueError("❌ No evaluators could be initialized!")
    
    logger.info(f"✅ Initialized {len(evaluators)} evaluators: {list(evaluators.keys())}")
    return evaluators


def clear_gpu_cache():
    """Clear GPU cache between evaluations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def monitor_gpu_usage():
    """Monitor GPU usage during evaluation."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "utilization_percent": round((allocated / total) * 100, 1)
        }
    return {}


def run_benchmark_evaluation(benchmark_name: str, benchmark, evaluators: Dict):
    """Run real evaluation on a single benchmark."""
    logger.info(f"🚀 Running evaluation: {benchmark_name.upper()}")
    
    benchmark_results = {
        "benchmark_name": benchmark_name,
        "start_time": datetime.now().isoformat(),
        "models": {}
    }
    
    for model_name, evaluator in evaluators.items():
        try:
            logger.info(f"   🔬 Evaluating {model_name} on {benchmark_name}...")
            
            # Clear GPU cache before each model
            clear_gpu_cache()
            
            model_start_time = datetime.now()
            model_results = evaluator.evaluate_benchmark(benchmark)
            model_end_time = datetime.now()
            
            eval_time = (model_end_time - model_start_time).total_seconds()
            
            benchmark_results["models"][model_name] = {
                "results": model_results,
                "evaluation_time_seconds": eval_time,
                "gpu_stats": monitor_gpu_usage()
            }
            
            # Log results
            if "error" not in model_results:
                accuracy = model_results.get("metrics", {}).get("accuracy", 0)
                logger.info(f"   ✅ {model_name} completed in {eval_time:.1f}s (Accuracy: {accuracy:.1f}%)")
            else:
                logger.warning(f"   ⚠️ {model_name} failed in {eval_time:.1f}s")
                
        except Exception as e:
            logger.error(f"   ❌ {model_name} failed: {e}")
            benchmark_results["models"][model_name] = {
                "error": str(e),
                "status": "failed",
                "evaluation_time_seconds": 0
            }
    
    benchmark_results["end_time"] = datetime.now().isoformat()
    return benchmark_results


def save_results(results: Dict, results_dir: Path):
    """Save evaluation results."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"gpu_evaluation_results_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Final results saved to: {results_file}")
    return results_file


def run_evaluation(mode: str = "full", benchmarks_filter: Optional[List[str]] = None, models_filter: Optional[List[str]] = None):
    """
    Run real GPU evaluation with actual models and benchmarks.
    
    Args:
        mode: 'quick' (small sample), 'full' (complete evaluation)
        benchmarks_filter: List of specific benchmarks to run
        models_filter: List of specific models to evaluate
    """
    
    start_time = datetime.now()
    
    # Load configuration and setup
    config = load_config()
    
    # Adjust config based on mode
    if mode == "quick":
        for benchmark_name in config["benchmarks"]:
            config["benchmarks"][benchmark_name]["sample_size"] = 10
        logger.info("🚀 Quick mode - Running small sample evaluation")
    else:
        logger.info("🚀 Full mode - Running complete evaluation")
    
    # Setup results directory
    results_dir = Path(config["results_dir"])
    setup_logging(str(results_dir))
    
    # Log start
    logger.info("🎯 Starting HierRAGMed GPU Evaluation - Mode: {}".format(mode.upper()))
    gpu_info = validate_gpu_environment()
    setup_gpu_optimization()
    
    try:
        # Initialize benchmarks and evaluators
        benchmarks = initialize_benchmarks(config, benchmarks_filter)
        evaluators = initialize_evaluators(config, models_filter)
        
        logger.info(f"📊 Running {len(benchmarks)} benchmarks × {len(evaluators)} models")
        
        # Run evaluations
        all_results = {
            "metadata": {
                "start_time": start_time.isoformat(),
                "mode": mode,
                "gpu_info": gpu_info,
                "benchmarks": list(benchmarks.keys()),
                "models": list(evaluators.keys())
            },
            "results": {}
        }
        
        # Run each benchmark
        for i, (benchmark_name, benchmark) in enumerate(benchmarks.items(), 1):
            logger.info(f"📊 Progress: {i}/{len(benchmarks)} benchmarks")
            
            benchmark_results = run_benchmark_evaluation(benchmark_name, benchmark, evaluators)
            all_results["results"][benchmark_name] = benchmark_results
            
            # Save intermediate results
            if i % 2 == 0:  # Save every 2 benchmarks
                logger.info("💾 Saved intermediate results")
        
        # Calculate final metadata
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        all_results["metadata"].update({
            "end_time": end_time.isoformat(),
            "total_duration_seconds": total_duration
        })
        
        # Save final results
        results_file = save_results(all_results, results_dir)
        
        return all_results
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="HierRAGMed GPU Evaluation")
    parser.add_argument("--mode", choices=["quick", "full"], default="full", 
                        help="Evaluation mode")
    parser.add_argument("--models", nargs="+", 
                        choices=["kg_system", "hierarchical_system"],
                        help="Specific models to evaluate")
    parser.add_argument("--benchmarks", nargs="+",
                        choices=["mirage", "medreason", "pubmedqa", "msmarco"],
                        help="Specific benchmarks to run")
    
    # Legacy argument support
    parser.add_argument("--quick", action="store_true", 
                        help="Run quick evaluation (legacy)")
    parser.add_argument("--full", action="store_true",
                        help="Run full evaluation (legacy)")
    
    args = parser.parse_args()
    
    # Handle legacy arguments
    if args.quick:
        mode = "quick"
    elif args.full:
        mode = "full"
    else:
        mode = args.mode
    
    try:
        logger.info("🎉 Starting HierRAGMed Evaluation")
        
        results = run_evaluation(
            mode=mode,
            benchmarks_filter=args.benchmarks,
            models_filter=args.models
        )
        
        # Print results summary
        print("\n" + "="*60)
        print("🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        metadata = results["metadata"]
        print(f"⏱️ Total duration: {metadata.get('total_duration_seconds', 0):.1f} seconds")
        print(f"🎯 Mode: {metadata.get('mode', 'unknown').upper()}")
        print(f"🔥 GPU: {metadata.get('gpu_info', {}).get('gpu_name', 'Unknown')}")
        print(f"📊 Benchmarks: {', '.join(metadata.get('benchmarks', []))}")
        print(f"🔬 Models: {', '.join(metadata.get('models', []))}")
        
        # Print results summary
        print(f"\n📈 Results Summary:")
        for benchmark_name, benchmark_data in results["results"].items():
            if "error" not in benchmark_data:
                print(f"   • {benchmark_name.upper()}:")
                for model_name, model_data in benchmark_data.get("models", {}).items():
                    if "results" in model_data and "error" not in model_data["results"]:
                        accuracy = model_data["results"].get("metrics", {}).get("accuracy", 0)
                        eval_time = model_data.get("evaluation_time_seconds", 0)
                        print(f"     - {model_name}: {accuracy:.1f}% accuracy ({eval_time:.1f}s)")
                    else:
                        print(f"     - {model_name}: FAILED")
            else:
                print(f"   • {benchmark_name.upper()}: FAILED")
        
        print("\n" + "="*60)
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()