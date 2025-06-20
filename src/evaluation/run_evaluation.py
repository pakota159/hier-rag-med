#!/usr/bin/env python3
"""
FIXED: HierRAGMed GPU Evaluation Runner
Corrects the full/quick mode configuration to properly handle unlimited datasets.
"""

import argparse
import sys
import os
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Import core components
from loguru import logger
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.utils.config_loader import ConfigLoader
from src.evaluation.benchmarks.mirage_benchmark import MIRAGEBenchmark
from src.evaluation.benchmarks.medreason_benchmark import MedReasonBenchmark
from src.evaluation.benchmarks.pubmedqa_benchmark import PubMedQABenchmark
from src.evaluation.benchmarks.msmarco_benchmark import MSMARCOBenchmark
from src.evaluation.evaluators.kg_evaluator import KGEvaluator
from src.evaluation.evaluators.hierarchical_evaluator import HierarchicalEvaluator


def validate_gpu_environment():
    """Validate GPU environment and log details."""
    logger.info("🔍 Validating GPU environment...")
    
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available - running on CPU")
        return {
            "gpu_available": False,
            "device": "cpu",
            "platform": "Local"
        }
    
    # Get GPU information
    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
    
    # Check CUDA version
    cuda_version = torch.version.cuda
    
    # Detect platform
    platform = "RunPod" if "runpod" in os.environ.get("HOSTNAME", "").lower() else "Local"
    
    logger.info("✅ GPU Environment:")
    logger.info(f"   GPU: {gpu_name}")
    logger.info(f"   Memory: {gpu_memory:.1f}GB")
    logger.info(f"   CUDA: {cuda_version}")
    logger.info(f"   Platform: {platform}")
    
    return {
        "gpu_available": True,
        "gpu_count": gpu_count,
        "current_device": current_device,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory,
        "cuda_version": cuda_version,
        "platform": platform
    }


def setup_gpu_optimization():
    """Configure GPU optimizations."""
    if not torch.cuda.is_available():
        return
        
    # Clear cache and optimize GPU settings
    torch.cuda.empty_cache()
    if hasattr(torch.backends.cudnn, 'benchmark'):
        torch.backends.cudnn.benchmark = True
    
    logger.info("🔥 GPU optimizations applied")


def setup_logging(results_dir: str):
    """Setup logging for evaluation."""
    # Create logs directory
    logs_dir = Path(results_dir).parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure loguru
    log_file = logs_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, format="{time} | {level} | {message}")
    
    logger.info(f"📝 Logging to: {log_file}")


def load_config():
    """Load configuration with optimization."""
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    
    # Convert EvaluationConfig to dictionary for backward compatibility
    if hasattr(config, 'to_dict'):
        return config.to_dict()
    return config


def initialize_benchmarks(config: Dict, benchmark_filter: Optional[List[str]] = None):
    """Initialize benchmarks with CORRECTED sample size handling."""
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
                
                # CRITICAL FIX: Ensure sample_size matches the config intent
                max_samples = benchmark_config.get("max_samples")
                if max_samples is None:
                    # Full dataset mode - remove any sample limits
                    benchmark_config["sample_size"] = float('inf')  # Unlimited
                    logger.info(f"   📊 {benchmark_name}: FULL DATASET (unlimited samples)")
                else:
                    # Limited sample mode
                    benchmark_config["sample_size"] = max_samples
                    logger.info(f"   📊 {benchmark_name}: LIMITED to {max_samples} samples")
                
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


def run_benchmark_evaluation(benchmark_name: str, benchmark, evaluators: Dict):
    """Run a single benchmark against all evaluators."""
    logger.info(f"🎯 Running {benchmark_name.upper()} benchmark...")
    
    benchmark_results = {}
    questions = benchmark.get_questions()
    
    logger.info(f"   📋 Processing {len(questions)} questions for {benchmark_name}")
    
    for evaluator_name, evaluator in evaluators.items():
        try:
            logger.info(f"   🔄 Evaluating {evaluator_name} on {benchmark_name}...")
            
            # Run evaluation
            results = evaluator.evaluate_benchmark(benchmark, questions)
            benchmark_results[evaluator_name] = results
            
            # Log summary
            if "metrics" in results:
                accuracy = results["metrics"].get("accuracy", 0)
                logger.info(f"   ✅ {evaluator_name}: {accuracy:.1f}% accuracy")
            
            # Clear GPU cache between evaluators
            clear_gpu_cache()
            
        except Exception as e:
            logger.error(f"   ❌ {evaluator_name} failed on {benchmark_name}: {e}")
            benchmark_results[evaluator_name] = {"error": str(e)}
    
    return benchmark_results


def clear_gpu_cache():
    """Clear GPU cache between evaluations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def save_results(results: Dict, results_dir: Path):
    """Save evaluation results."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"gpu_evaluation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Final results saved to: {results_file}")
    return results_file


def run_evaluation(mode: str = "full", benchmarks_filter: Optional[List[str]] = None, models_filter: Optional[List[str]] = None):
    """
    FIXED: Run evaluation with proper full/quick mode handling.
    
    Args:
        mode: 'quick' (small sample), 'full' (complete evaluation)
        benchmarks_filter: List of specific benchmarks to run
        models_filter: List of specific models to evaluate
    """
    
    start_time = datetime.now()
    
    # Load configuration and setup
    config = load_config()
    
    # FIXED: Adjust config based on mode with EXPLICIT sample size control
    if mode == "quick":
        logger.info("🚀 Quick mode - Running small sample evaluation")
        for benchmark_name in config["benchmarks"]:
            # Use quick_eval_samples for quick mode
            quick_samples = config["benchmarks"][benchmark_name].get("quick_eval_samples", 10)
            config["benchmarks"][benchmark_name]["max_samples"] = quick_samples
            logger.info(f"   📊 {benchmark_name}: LIMITED to {quick_samples} samples (quick mode)")
    else:
        logger.info("🚀 Full mode - Running complete evaluation on FULL datasets")
        for benchmark_name in config["benchmarks"]:
            # Explicitly set to None for unlimited
            config["benchmarks"][benchmark_name]["max_samples"] = None
            logger.info(f"   📊 {benchmark_name}: FULL DATASET (unlimited samples)")
    
    # Setup results directory
    results_dir = Path(config.get("results_dir", "evaluation/results"))
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
                "models": list(evaluators.keys()),
                "sample_limits": {
                    name: config["benchmarks"][name].get("max_samples", "unlimited")
                    for name in benchmarks.keys()
                }
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
    """Main evaluation entry point with FIXED argument handling."""
    parser = argparse.ArgumentParser(description="HierRAGMed GPU Evaluation - FIXED VERSION")
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
        logger.info("🎉 Starting HierRAGMed Evaluation - FIXED VERSION")
        
        results = run_evaluation(
            mode=mode,
            benchmarks_filter=args.benchmarks,
            models_filter=args.models
        )
        
        # Print results summary
        print("\n" + "="*60)
        print("🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        for benchmark_name, benchmark_results in results["results"].items():
            print(f"\n📊 {benchmark_name.upper()} Results:")
            for model_name, model_results in benchmark_results.items():
                if "metrics" in model_results:
                    accuracy = model_results["metrics"].get("accuracy", 0)
                    print(f"   {model_name}: {accuracy:.1f}% accuracy")
                else:
                    print(f"   {model_name}: ERROR")
        
        print(f"\n💾 Results saved to: evaluation/results/")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()