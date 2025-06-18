#!/usr/bin/env python3
"""
GPU-optimized evaluation runner for HierRAGMed on RunPod.
Simplified for CUDA-only deployment with RTX 4090 optimization.
"""

import argparse
import sys
import os
import torch
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import json
from datetime import datetime
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import evaluation components
from src.evaluation.benchmarks.mirage_benchmark import MIRAGEBenchmark
from src.evaluation.benchmarks.medreason_benchmark import MedReasonBenchmark
from src.evaluation.benchmarks.pubmedqa_benchmark import PubMedQABenchmark
from src.evaluation.benchmarks.msmarco_benchmark import MSMARCOBenchmark

from src.evaluation.evaluators.kg_evaluator import KGEvaluator
from src.evaluation.evaluators.hierarchical_evaluator import HierarchicalEvaluator

from src.evaluation.utils.result_processor import ResultProcessor
from src.evaluation.utils.report_generator import ReportGenerator


def setup_gpu_environment() -> None:
    """Setup GPU environment for RunPod RTX 4090."""
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available! This script requires GPU.")
    
    # Set CUDA device
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    
    # GPU memory optimization for RTX 4090
    torch.cuda.empty_cache()
    
    # Set optimal settings for RTX 4090
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async CUDA operations
    
    # Log GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"🚀 GPU: {gpu_name}")
    logger.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    logger.info(f"🔥 CUDA Version: {torch.version.cuda}")


def load_evaluation_config(config_path: Path) -> Dict:
    """Load GPU-optimized evaluation configuration."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply GPU optimizations
        config = apply_gpu_optimizations(config)
        
        logger.info(f"✅ Loaded GPU config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        raise


def apply_gpu_optimizations(config: Dict) -> Dict:
    """Apply RTX 4090 specific optimizations to config."""
    # Force CUDA device
    if "models" in config:
        for model_key in config["models"]:
            if isinstance(config["models"][model_key], dict):
                config["models"][model_key]["device"] = "cuda"
                
                # Optimize batch sizes for RTX 4090
                if "batch_size" in config["models"][model_key]:
                    if model_key == "embedding":
                        config["models"][model_key]["batch_size"] = 128  # Up from 16
                    elif model_key == "llm":
                        config["models"][model_key]["batch_size"] = 32   # Up from 8
    
    # GPU-specific evaluation settings
    if "evaluation" in config:
        config["evaluation"]["parallel_processing"] = True
        config["evaluation"]["gpu_batch_size"] = 32
        config["evaluation"]["memory_optimization"] = True
    
    # Set high-performance defaults
    config["performance"] = {
        "use_gpu_acceleration": True,
        "mixed_precision": True,
        "compile_models": True,
        "memory_efficient": True
    }
    
    logger.info("🔧 Applied RTX 4090 optimizations")
    return config


def setup_logging(config: Dict) -> None:
    """Setup logging for GPU evaluation."""
    log_config = config.get("logging", {})
    
    # Create results directory
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup loguru with GPU-specific formatting
    logger.add(
        results_dir / "gpu_evaluation.log",
        rotation="1 day",
        retention="7 days",
        level=log_config.get("level", "INFO"),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>GPU</cyan> | {message}"
    )


def initialize_benchmarks(config: Dict) -> Dict:
    """Initialize all enabled benchmarks with GPU optimization."""
    benchmarks = {}
    benchmark_config = config["benchmarks"]
    
    if benchmark_config["mirage"]["enabled"]:
        logger.info("🎯 Initializing MIRAGE benchmark (GPU-optimized)...")
        benchmarks["mirage"] = MIRAGEBenchmark(benchmark_config["mirage"])
    
    if benchmark_config["medreason"]["enabled"]:
        logger.info("🧠 Initializing MedReason benchmark (GPU-optimized)...")
        benchmarks["medreason"] = MedReasonBenchmark(benchmark_config["medreason"])
    
    if benchmark_config["pubmedqa"]["enabled"]:
        logger.info("📚 Initializing PubMedQA benchmark (GPU-optimized)...")
        benchmarks["pubmedqa"] = PubMedQABenchmark(benchmark_config["pubmedqa"])
    
    if benchmark_config["msmarco"]["enabled"]:
        logger.info("🔍 Initializing MS MARCO benchmark (GPU-optimized)...")
        benchmarks["msmarco"] = MSMARCOBenchmark(benchmark_config["msmarco"])
    
    logger.info(f"✅ Initialized {len(benchmarks)} GPU-optimized benchmarks")
    return benchmarks


def initialize_evaluators(config: Dict) -> Dict:
    """Initialize all enabled model evaluators with GPU optimization."""
    evaluators = {}
    model_config = config["models"]
    
    if model_config["kg_system"]["enabled"]:
        logger.info("📊 Initializing KG System evaluator (GPU-optimized)...")
        evaluators["kg_system"] = KGEvaluator(model_config["kg_system"])
    
    if model_config["hierarchical_system"]["enabled"]:
        logger.info("🏗️ Initializing Hierarchical System evaluator (GPU-optimized)...")
        evaluators["hierarchical_system"] = HierarchicalEvaluator(model_config["hierarchical_system"])
    
    logger.info(f"✅ Initialized {len(evaluators)} GPU-optimized evaluators")
    return evaluators


def monitor_gpu_usage() -> Dict:
    """Monitor GPU usage during evaluation."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        usage_stats = {
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved, 
            "memory_total_gb": memory_total,
            "memory_utilization_percent": (memory_allocated / memory_total) * 100
        }
        
        logger.info(f"🔥 GPU Memory: {memory_allocated:.1f}GB / {memory_total:.1f}GB ({usage_stats['memory_utilization_percent']:.1f}%)")
        return usage_stats
    
    return {}


def run_single_benchmark(
    benchmark_name: str,
    benchmark,
    evaluators: Dict,
    config: Dict
) -> Dict:
    """Run evaluation on a single benchmark with GPU optimization."""
    logger.info(f"🚀 Running {benchmark_name.upper()} benchmark on GPU...")
    
    benchmark_results = {
        "benchmark_name": benchmark_name,
        "start_time": datetime.now().isoformat(),
        "models": {},
        "gpu_stats": {}
    }
    
    # Monitor GPU before benchmark
    benchmark_results["gpu_stats"]["pre_benchmark"] = monitor_gpu_usage()
    
    for model_name, evaluator in evaluators.items():
        try:
            logger.info(f"   🔬 Evaluating {model_name} on {benchmark_name}...")
            
            # Clear GPU cache before each model
            torch.cuda.empty_cache()
            
            model_start_time = datetime.now()
            model_results = evaluator.evaluate(benchmark)
            model_end_time = datetime.now()
            
            # Calculate evaluation time
            eval_time = (model_end_time - model_start_time).total_seconds()
            
            benchmark_results["models"][model_name] = {
                "results": model_results,
                "evaluation_time_seconds": eval_time,
                "gpu_stats": monitor_gpu_usage()
            }
            
            logger.info(f"   ✅ {model_name} completed in {eval_time:.1f}s")
            
        except Exception as e:
            logger.error(f"   ❌ {model_name} failed on {benchmark_name}: {e}")
            benchmark_results["models"][model_name] = {
                "error": str(e),
                "evaluation_time_seconds": 0,
                "gpu_stats": monitor_gpu_usage()
            }
    
    benchmark_results["end_time"] = datetime.now().isoformat()
    benchmark_results["gpu_stats"]["post_benchmark"] = monitor_gpu_usage()
    
    return benchmark_results


def run_evaluation(
    config_path: Optional[Path] = None,
    benchmarks_filter: Optional[List[str]] = None,
    models_filter: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run comprehensive GPU-optimized evaluation on RunPod.
    
    Args:
        config_path: Path to evaluation config file
        benchmarks_filter: List of benchmark names to run (None = all)
        models_filter: List of model names to evaluate (None = all)
        output_dir: Custom output directory
    
    Returns:
        Dict containing all evaluation results
    """
    # Setup GPU environment first
    setup_gpu_environment()
    
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "gpu_runpod_config.yaml"
    
    config = load_evaluation_config(config_path)
    setup_logging(config)
    
    # Override output directory if specified
    if output_dir:
        config["results_dir"] = str(output_dir)
    
    logger.info("🎯 Starting HierRAGMed GPU Evaluation on RunPod")
    logger.info(f"   Config: {config_path}")
    logger.info(f"   Results: {config['results_dir']}")
    
    # Initialize components
    benchmarks = initialize_benchmarks(config)
    evaluators = initialize_evaluators(config)
    
    # Apply filters
    if benchmarks_filter:
        benchmarks = {k: v for k, v in benchmarks.items() if k in benchmarks_filter}
        logger.info(f"🔍 Filtered benchmarks: {list(benchmarks.keys())}")
    
    if models_filter:
        evaluators = {k: v for k, v in evaluators.items() if k in models_filter}
        logger.info(f"🔍 Filtered models: {list(evaluators.keys())}")
    
    # Run evaluations
    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "config_path": str(config_path),
            "benchmarks": list(benchmarks.keys()),
            "models": list(evaluators.keys()),
            "platform": "RunPod GPU",
            "gpu_info": {
                "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown",
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
                "cuda_version": torch.version.cuda
            }
        },
        "results": {}
    }
    
    total_benchmarks = len(benchmarks)
    for i, (benchmark_name, benchmark) in enumerate(benchmarks.items(), 1):
        try:
            logger.info(f"📊 Progress: {i}/{total_benchmarks} benchmarks")
            benchmark_results = run_single_benchmark(
                benchmark_name, benchmark, evaluators, config
            )
            all_results["results"][benchmark_name] = benchmark_results
            
            # Save intermediate results for long evaluations
            if i % 2 == 0:  # Save every 2 benchmarks
                intermediate_file = Path(config["results_dir"]) / f"intermediate_results_{i}.json"
                with open(intermediate_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                logger.info(f"💾 Saved intermediate results to {intermediate_file}")
            
        except Exception as e:
            logger.error(f"❌ Benchmark {benchmark_name} failed: {e}")
            all_results["results"][benchmark_name] = {"error": str(e)}
    
    # Process and save results
    logger.info("📊 Processing evaluation results...")
    result_processor = ResultProcessor(config)
    processed_results = result_processor.process_results(all_results)
    
    # Generate reports
    logger.info("📝 Generating evaluation reports...")
    report_generator = ReportGenerator(config)
    report_generator.generate_comprehensive_report(processed_results)
    
    # Final GPU stats
    final_gpu_stats = monitor_gpu_usage()
    processed_results["metadata"]["final_gpu_stats"] = final_gpu_stats
    
    logger.info("🎉 GPU evaluation completed successfully!")
    return processed_results


def main():
    """Command line interface for GPU evaluation."""
    parser = argparse.ArgumentParser(description="HierRAGMed GPU Evaluation on RunPod")
    parser.add_argument("--config", type=Path, help="Path to evaluation config file")
    parser.add_argument("--benchmarks", nargs="+", help="Specific benchmarks to run")
    parser.add_argument("--models", nargs="+", help="Specific models to evaluate")
    parser.add_argument("--output-dir", help="Custom output directory")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation mode")
    
    args = parser.parse_args()
    
    try:
        # Run evaluation
        results = run_evaluation(
            config_path=args.config,
            benchmarks_filter=args.benchmarks,
            models_filter=args.models,
            output_dir=args.output_dir
        )
        
        print("\n🎉 Evaluation completed successfully!")
        print(f"📄 Results saved to: {results['metadata'].get('results_dir', 'results/')}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()