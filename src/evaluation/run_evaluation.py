#!/usr/bin/env python3
"""
GPU-optimized evaluation runner for HierRAGMed.
Comprehensive evaluation framework with MIRAGE, MedReason, PubMedQA, and MS MARCO benchmarks.
Optimized for RunPod RTX 4090 deployment with full integration.
"""

import argparse
import sys
import os
import gc
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json
from datetime import datetime, timedelta
import time
from loguru import logger
import numpy as np

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
from src.evaluation.evaluators.comparative_evaluator import ComparativeEvaluator

from src.evaluation.utils.result_processor import ResultProcessor
from src.evaluation.utils.report_generator import ReportGenerator
from src.evaluation.utils.visualization import EvaluationVisualizer
from src.evaluation.utils.statistical_analysis import StatisticalAnalysis
from src.evaluation.utils.config_loader import ConfigLoader

from src.evaluation.data.data_loader import BenchmarkDataLoader
from src.evaluation.data.data_validator import DataValidator

# Import metrics
from src.evaluation.metrics.combined_metrics import CombinedMetrics


def setup_gpu_environment() -> Dict[str, Any]:
    """Setup GPU environment with comprehensive diagnostics."""
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "gpu_name": "N/A",
        "gpu_memory_gb": 0,
        "cuda_version": "N/A",
        "pytorch_version": torch.__version__
    }
    
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available! Running on CPU.")
        return gpu_info
    
    # Set CUDA device
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    
    # GPU memory optimization for RTX 4090
    torch.cuda.empty_cache()
    gc.collect()
    
    # Set optimal settings for RTX 4090
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async CUDA operations
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
    os.environ["OMP_NUM_THREADS"] = "8"
    
    # Update GPU info
    gpu_info.update({
        "device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "cuda_version": torch.version.cuda,
        "current_device": torch.cuda.current_device()
    })
    
    # Log GPU info
    logger.info(f"🚀 GPU: {gpu_info['gpu_name']}")
    logger.info(f"💾 GPU Memory: {gpu_info['gpu_memory_gb']:.1f} GB")
    logger.info(f"🔥 CUDA Version: {gpu_info['cuda_version']}")
    logger.info(f"🐍 PyTorch Version: {gpu_info['pytorch_version']}")
    
    return gpu_info


def load_evaluation_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate evaluation configuration."""
    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        
        # Validate required sections
        required_sections = ["models", "benchmarks", "evaluation", "output"]
        for section in required_sections:
            if section not in config:
                logger.warning(f"⚠️ Missing config section: {section}")
                config[section] = {}
        
        # Set default paths if not specified
        if "results_dir" not in config:
            config["results_dir"] = "evaluation/results"
        if "cache_dir" not in config:
            config["cache_dir"] = "evaluation/cache"
        if "logs_dir" not in config:
            config["logs_dir"] = "evaluation/logs"
        
        # Apply GPU optimizations
        config = apply_gpu_optimizations(config)
        
        logger.info(f"✅ Loaded configuration from {config_path or 'default'}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        raise


def apply_gpu_optimizations(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply RTX 4090 specific optimizations to config."""
    if not torch.cuda.is_available():
        return config
    
    # GPU-specific model optimizations
    if "models" in config:
        # Embedding model optimizations
        if "embedding" in config["models"]:
            config["models"]["embedding"].update({
                "device": "cuda",
                "batch_size": 128,  # RTX 4090 optimization
                "max_length": 512,
                "normalize_embeddings": True
            })
        
        # LLM optimizations
        if "llm" in config["models"]:
            config["models"]["llm"].update({
                "device": "cuda",
                "batch_size": 32,  # RTX 4090 optimization
                "mixed_precision": True,
                "compile_model": True
            })
        
        # System-specific optimizations
        for system in ["kg_system", "hierarchical_system"]:
            if system in config["models"]:
                config["models"][system].update({
                    "device": "cuda",
                    "memory_efficient": True,
                    "mixed_precision": True,
                    "compile_model": True
                })
    
    # Performance optimizations
    config["performance"] = config.get("performance", {})
    config["performance"].update({
        "use_gpu_acceleration": True,
        "mixed_precision": True,
        "compile_models": True,
        "memory_efficient": True,
        "gpu_memory_fraction": 0.85,
        "parallel_processing": True,
        "max_workers": 4,
        "prefetch_factor": 2
    })
    
    logger.info("🔧 Applied GPU optimizations to configuration")
    return config


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup comprehensive logging with GPU monitoring."""
    logs_dir = Path(config.get("logs_dir", "evaluation/logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"gpu_evaluation_{timestamp}.log"
    
    # Configure logger
    logger.remove()  # Remove default handler
    
    # Console handler
    logger.add(
        sys.stdout,
        level=config.get("logging", {}).get("console_level", "INFO"),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | GPU | <level>{message}</level>",
        colorize=True
    )
    
    # File handler
    logger.add(
        log_file,
        level=config.get("logging", {}).get("level", "INFO"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | GPU | {message}",
        rotation="100 MB",
        retention="7 days"
    )
    
    logger.info(f"📝 Logging initialized: {log_file}")


def initialize_benchmarks(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize all enabled benchmarks with GPU optimization."""
    benchmarks = {}
    benchmark_config = config.get("benchmarks", {})
    
    # Initialize data loader and validator
    data_loader = BenchmarkDataLoader(config)
    data_validator = DataValidator(config)
    
    # MIRAGE Benchmark
    if benchmark_config.get("mirage", {}).get("enabled", False):
        logger.info("🎯 Initializing MIRAGE benchmark...")
        try:
            mirage_config = benchmark_config["mirage"]
            mirage_config["data_loader"] = data_loader
            mirage_config["data_validator"] = data_validator
            benchmarks["mirage"] = MIRAGEBenchmark(mirage_config)
            logger.info("✅ MIRAGE benchmark initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MIRAGE: {e}")
    
    # MedReason Benchmark
    if benchmark_config.get("medreason", {}).get("enabled", False):
        logger.info("🧠 Initializing MedReason benchmark...")
        try:
            medreason_config = benchmark_config["medreason"]
            medreason_config["data_loader"] = data_loader
            medreason_config["data_validator"] = data_validator
            benchmarks["medreason"] = MedReasonBenchmark(medreason_config)
            logger.info("✅ MedReason benchmark initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MedReason: {e}")
    
    # PubMedQA Benchmark
    if benchmark_config.get("pubmedqa", {}).get("enabled", False):
        logger.info("📚 Initializing PubMedQA benchmark...")
        try:
            pubmedqa_config = benchmark_config["pubmedqa"]
            pubmedqa_config["data_loader"] = data_loader
            pubmedqa_config["data_validator"] = data_validator
            benchmarks["pubmedqa"] = PubMedQABenchmark(pubmedqa_config)
            logger.info("✅ PubMedQA benchmark initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize PubMedQA: {e}")
    
    # MS MARCO Benchmark
    if benchmark_config.get("msmarco", {}).get("enabled", False):
        logger.info("🔍 Initializing MS MARCO benchmark...")
        try:
            msmarco_config = benchmark_config["msmarco"]
            msmarco_config["data_loader"] = data_loader
            msmarco_config["data_validator"] = data_validator
            benchmarks["msmarco"] = MSMARCOBenchmark(msmarco_config)
            logger.info("✅ MS MARCO benchmark initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MS MARCO: {e}")
    
    if not benchmarks:
        raise ValueError("❌ No benchmarks enabled or successfully initialized!")
    
    logger.info(f"✅ Initialized {len(benchmarks)} benchmarks: {list(benchmarks.keys())}")
    return benchmarks


def initialize_evaluators(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize all enabled model evaluators with GPU optimization."""
    evaluators = {}
    model_config = config.get("models", {})
    
    # Initialize combined metrics
    combined_metrics = CombinedMetrics(config)
    
    # KG System Evaluator
    if model_config.get("kg_system", {}).get("enabled", False):
        logger.info("📊 Initializing KG System evaluator...")
        try:
            kg_config = model_config["kg_system"]
            kg_config["metrics"] = combined_metrics
            evaluators["kg_system"] = KGEvaluator(kg_config)
            logger.info("✅ KG System evaluator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize KG evaluator: {e}")
    
    # Hierarchical System Evaluator
    if model_config.get("hierarchical_system", {}).get("enabled", False):
        logger.info("🏗️ Initializing Hierarchical System evaluator...")
        try:
            hierarchical_config = model_config["hierarchical_system"]
            hierarchical_config["metrics"] = combined_metrics
            evaluators["hierarchical_system"] = HierarchicalEvaluator(hierarchical_config)
            logger.info("✅ Hierarchical System evaluator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hierarchical evaluator: {e}")
    
    if not evaluators:
        raise ValueError("❌ No evaluators enabled or successfully initialized!")
    
    logger.info(f"✅ Initialized {len(evaluators)} evaluators: {list(evaluators.keys())}")
    return evaluators


def monitor_gpu_usage() -> Dict[str, Any]:
    """Monitor comprehensive GPU usage during evaluation."""
    if not torch.cuda.is_available():
        return {"error": "No GPU available"}
    
    try:
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        usage_stats = {
            "timestamp": datetime.now().isoformat(),
            "memory_allocated_gb": round(memory_allocated, 2),
            "memory_reserved_gb": round(memory_reserved, 2),
            "memory_total_gb": round(memory_total, 2),
            "memory_utilization_percent": round((memory_allocated / memory_total) * 100, 1),
            "memory_reserved_percent": round((memory_reserved / memory_total) * 100, 1),
            "free_memory_gb": round(memory_total - memory_allocated, 2)
        }
        
        logger.debug(f"🔥 GPU Memory: {memory_allocated:.1f}GB / {memory_total:.1f}GB ({usage_stats['memory_utilization_percent']:.1f}%)")
        return usage_stats
    except Exception as e:
        logger.error(f"Failed to monitor GPU usage: {e}")
        return {"error": str(e)}


def clear_gpu_cache() -> None:
    """Clear GPU cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def run_single_benchmark(
    benchmark_name: str,
    benchmark: Any,
    evaluators: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run evaluation on a single benchmark with comprehensive GPU optimization."""
    logger.info(f"🚀 Running {benchmark_name.upper()} benchmark...")
    
    benchmark_results = {
        "benchmark_name": benchmark_name,
        "start_time": datetime.now().isoformat(),
        "models": {},
        "gpu_stats": {},
        "benchmark_info": {
            "name": getattr(benchmark, 'name', benchmark_name),
            "description": getattr(benchmark, 'description', ''),
            "version": getattr(benchmark, 'version', '1.0')
        }
    }
    
    # Monitor GPU before benchmark
    benchmark_results["gpu_stats"]["pre_benchmark"] = monitor_gpu_usage()
    
    total_models = len(evaluators)
    for i, (model_name, evaluator) in enumerate(evaluators.items(), 1):
        try:
            logger.info(f"   🔬 Evaluating {model_name} on {benchmark_name} ({i}/{total_models})...")
            
            # Clear GPU cache before each model
            clear_gpu_cache()
            
            model_start_time = datetime.now()
            
            # Run evaluation with error handling
            try:
                model_results = evaluator.evaluate_benchmark(benchmark)
            except Exception as eval_error:
                logger.error(f"   ❌ Evaluation failed for {model_name}: {eval_error}")
                model_results = {
                    "error": str(eval_error),
                    "status": "failed",
                    "model_name": model_name,
                    "benchmark_name": benchmark_name
                }
            
            model_end_time = datetime.now()
            eval_time = (model_end_time - model_start_time).total_seconds()
            
            # Store model results
            benchmark_results["models"][model_name] = {
                "results": model_results,
                "evaluation_time_seconds": eval_time,
                "start_time": model_start_time.isoformat(),
                "end_time": model_end_time.isoformat(),
                "gpu_stats": monitor_gpu_usage()
            }
            
            # Log completion
            if "error" not in model_results:
                accuracy = model_results.get("metrics", {}).get("accuracy", 0)
                logger.info(f"   ✅ {model_name} completed in {eval_time:.1f}s (Accuracy: {accuracy:.1f}%)")
            else:
                logger.warning(f"   ⚠️ {model_name} completed with errors in {eval_time:.1f}s")
                
        except Exception as e:
            logger.error(f"   ❌ {model_name} failed on {benchmark_name}: {e}")
            benchmark_results["models"][model_name] = {
                "error": str(e),
                "status": "failed",
                "evaluation_time_seconds": 0,
                "gpu_stats": monitor_gpu_usage()
            }
    
    benchmark_results["end_time"] = datetime.now().isoformat()
    benchmark_results["gpu_stats"]["post_benchmark"] = monitor_gpu_usage()
    
    # Calculate benchmark summary
    successful_models = [m for m in benchmark_results["models"].values() if "error" not in m.get("results", {})]
    benchmark_results["summary"] = {
        "total_models": total_models,
        "successful_models": len(successful_models),
        "failed_models": total_models - len(successful_models),
        "total_time_seconds": (datetime.fromisoformat(benchmark_results["end_time"]) - 
                             datetime.fromisoformat(benchmark_results["start_time"])).total_seconds()
    }
    
    logger.info(f"✅ {benchmark_name} completed: {len(successful_models)}/{total_models} models successful")
    return benchmark_results


def run_evaluation(
    config_path: Optional[Path] = None,
    benchmarks_filter: Optional[List[str]] = None,
    models_filter: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    quick_mode: bool = False
) -> Dict[str, Any]:
    """
    Run comprehensive GPU-optimized evaluation.
    
    Args:
        config_path: Path to evaluation config file
        benchmarks_filter: List of benchmark names to run (None = all)
        models_filter: List of model names to evaluate (None = all)
        output_dir: Custom output directory
        quick_mode: Enable quick evaluation mode
    
    Returns:
        Dict containing all evaluation results
    """
    start_time = datetime.now()
    
    # Setup GPU environment
    gpu_info = setup_gpu_environment()
    
    # Load configuration
    config = load_evaluation_config(config_path)
    setup_logging(config)
    
    # Override output directory if specified
    if output_dir:
        config["results_dir"] = str(output_dir)
    
    # Apply quick mode settings
    if quick_mode:
        logger.info("⚡ Quick mode enabled - applying optimizations...")
        config = apply_quick_mode_optimizations(config)
    
    logger.info("🎯 Starting HierRAGMed GPU Evaluation")
    logger.info(f"   Config: {config_path or 'default'}")
    logger.info(f"   Results: {config['results_dir']}")
    logger.info(f"   Platform: {config.get('platform', 'Unknown')}")
    
    # Create output directories
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    try:
        benchmarks = initialize_benchmarks(config)
        evaluators = initialize_evaluators(config)
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        raise
    
    # Apply filters
    if benchmarks_filter:
        benchmarks = {k: v for k, v in benchmarks.items() if k in benchmarks_filter}
        logger.info(f"🔍 Filtered benchmarks: {list(benchmarks.keys())}")
    
    if models_filter:
        evaluators = {k: v for k, v in evaluators.items() if k in models_filter}
        logger.info(f"🔍 Filtered models: {list(evaluators.keys())}")
    
    # Initialize result tracking
    all_results = {
        "metadata": {
            "timestamp": start_time.isoformat(),
            "config_path": str(config_path) if config_path else "default",
            "benchmarks": list(benchmarks.keys()),
            "models": list(evaluators.keys()),
            "platform": config.get("platform", "Unknown"),
            "gpu_info": gpu_info,
            "quick_mode": quick_mode,
            "version": "0.1.0"
        },
        "results": {}
    }
    
    # Run evaluations
    total_benchmarks = len(benchmarks)
    logger.info(f"📊 Starting evaluation: {total_benchmarks} benchmarks × {len(evaluators)} models")
    
    for i, (benchmark_name, benchmark) in enumerate(benchmarks.items(), 1):
        try:
            logger.info(f"📊 Progress: {i}/{total_benchmarks} benchmarks")
            
            benchmark_results = run_single_benchmark(
                benchmark_name, benchmark, evaluators, config
            )
            all_results["results"][benchmark_name] = benchmark_results
            
            # Save intermediate results for long evaluations
            if i % 2 == 0 or i == total_benchmarks:
                intermediate_file = results_dir / f"intermediate_results_{i:02d}.json"
                with open(intermediate_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                logger.info(f"💾 Saved intermediate results to {intermediate_file}")
            
            # Clear cache between benchmarks
            clear_gpu_cache()
            
        except Exception as e:
            logger.error(f"❌ Benchmark {benchmark_name} failed: {e}")
            all_results["results"][benchmark_name] = {
                "error": str(e),
                "status": "failed",
                "benchmark_name": benchmark_name
            }
    
    # Calculate final metadata
    end_time = datetime.now()
    all_results["metadata"].update({
        "end_time": end_time.isoformat(),
        "total_duration_seconds": (end_time - start_time).total_seconds(),
        "final_gpu_stats": monitor_gpu_usage()
    })
    
    # Process and save results
    logger.info("📊 Processing evaluation results...")
    try:
        result_processor = ResultProcessor(config)
        processed_results = result_processor.process_results(all_results)
        
        # Generate comprehensive reports
        logger.info("📝 Generating evaluation reports...")
        report_generator = ReportGenerator(config)
        report_generator.generate_comprehensive_report(processed_results)
        
        # Generate visualizations
        logger.info("📈 Generating visualizations...")
        visualizer = EvaluationVisualizer(config)
        visualizer.generate_evaluation_plots(processed_results)
        
        # Perform statistical analysis
        logger.info("📊 Performing statistical analysis...")
        statistical_analyzer = StatisticalAnalysis(config)
        statistical_results = statistical_analyzer.analyze_results(processed_results)
        
        # Save final results
        final_results_file = results_dir / f"evaluation_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(final_results_file, 'w') as f:
            json.dump(processed_results, f, indent=2)
        
        # Save statistical analysis
        stats_file = results_dir / f"statistical_analysis_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(statistical_results, f, indent=2)
        
        logger.info(f"💾 Final results saved to: {final_results_file}")
        logger.info(f"📊 Statistical analysis saved to: {stats_file}")
        
        return processed_results
        
    except Exception as e:
        logger.error(f"❌ Post-processing failed: {e}")
        # Still save raw results
        raw_results_file = results_dir / f"raw_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(raw_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"💾 Raw results saved to: {raw_results_file}")
        
        return all_results


def apply_quick_mode_optimizations(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply optimizations for quick evaluation mode."""
    # Reduce batch sizes for faster processing
    if "models" in config:
        for model_type in ["embedding", "llm"]:
            if model_type in config["models"]:
                config["models"][model_type]["batch_size"] = max(1, config["models"][model_type].get("batch_size", 16) // 2)
    
    # Reduce dataset sizes for benchmarks
    if "benchmarks" in config:
        for benchmark_name in config["benchmarks"]:
            if isinstance(config["benchmarks"][benchmark_name], dict):
                config["benchmarks"][benchmark_name]["max_samples"] = min(100, config["benchmarks"][benchmark_name].get("max_samples", 1000))
    
    logger.info("⚡ Applied quick mode optimizations")
    return config


def run_comparative_analysis(
    results: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run comparative analysis between different model systems."""
    logger.info("🔍 Running comparative analysis...")
    
    try:
        comparative_evaluator = ComparativeEvaluator(config)
        comparison_results = comparative_evaluator.compare_systems(results)
        
        logger.info("✅ Comparative analysis completed")
        return comparison_results
    except Exception as e:
        logger.error(f"❌ Comparative analysis failed: {e}")
        return {"error": str(e)}


def main():
    """Command line interface for evaluation."""
    parser = argparse.ArgumentParser(description="HierRAGMed GPU Evaluation System")
    parser.add_argument("--config", type=Path, help="Path to evaluation config file")
    parser.add_argument("--benchmarks", nargs="+", help="Specific benchmarks to run", 
                       choices=["mirage", "medreason", "pubmedqa", "msmarco"])
    parser.add_argument("--models", nargs="+", help="Specific models to evaluate",
                       choices=["kg_system", "hierarchical_system"])
    parser.add_argument("--output-dir", help="Custom output directory")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation mode")
    parser.add_argument("--compare", action="store_true", help="Run comparative analysis")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    try:
        # Set logging level
        if args.verbose:
            logger.remove()
            logger.add(sys.stdout, level="DEBUG")
        
        # Run evaluation
        results = run_evaluation(
            config_path=args.config,
            benchmarks_filter=args.benchmarks,
            models_filter=args.models,
            output_dir=args.output_dir,
            quick_mode=args.quick
        )
        
        # Run comparative analysis if requested
        if args.compare and len(results.get("results", {})) > 1:
            comparison_results = run_comparative_analysis(results, results.get("metadata", {}))
            results["comparative_analysis"] = comparison_results
        
        # Print summary
        print("\n" + "="*60)
        print("🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        metadata = results.get("metadata", {})
        print(f"📄 Results saved to: {metadata.get('final_gpu_stats', {}).get('results_dir', 'results/')}")
        print(f"⏱️ Total duration: {metadata.get('total_duration_seconds', 0):.1f} seconds")
        print(f"🔥 GPU: {metadata.get('gpu_info', {}).get('gpu_name', 'Unknown')}")
        
        # Print benchmark results summary
        if "results" in results:
            print(f"\n📊 Benchmark Results:")
            for benchmark_name, benchmark_data in results["results"].items():
                if "error" not in benchmark_data:
                    successful = benchmark_data.get("summary", {}).get("successful_models", 0)
                    total = benchmark_data.get("summary", {}).get("total_models", 0)
                    print(f"   • {benchmark_name.upper()}: {successful}/{total} models successful")
                    
                    # Print model accuracies
                    for model_name, model_data in benchmark_data.get("models", {}).items():
                        if "results" in model_data and "error" not in model_data["results"]:
                            accuracy = model_data["results"].get("metrics", {}).get("accuracy", 0)
                            print(f"     - {model_name}: {accuracy:.1f}% accuracy")
                else:
                    print(f"   • {benchmark_name.upper()}: FAILED - {benchmark_data.get('error', 'Unknown error')}")
        
        print("\n" + "="*60)
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


def validate_environment():
    """Validate the environment before running evaluation."""
    logger.info("🔍 Validating environment...")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8+ required")
    
    # Check required packages
    required_packages = [
        "torch", "transformers", "sentence_transformers",
        "chromadb", "langchain", "yaml", "numpy", "pandas"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise RuntimeError(f"Missing required packages: {missing_packages}")
    
    # Check GPU availability if configured
    if torch.cuda.is_available():
        logger.info(f"✅ GPU validation passed: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ No GPU detected - running on CPU")
    
    logger.info("✅ Environment validation completed")


def cleanup_evaluation():
    """Cleanup resources after evaluation."""
    logger.info("🧹 Cleaning up evaluation resources...")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Run garbage collection
    gc.collect()
    
    logger.info("✅ Cleanup completed")


def check_disk_space(min_gb: float = 5.0) -> bool:
    """Check if sufficient disk space is available."""
    import shutil
    
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        if free_gb < min_gb:
            logger.warning(f"⚠️ Low disk space: {free_gb:.1f}GB free (minimum {min_gb}GB recommended)")
            return False
        
        logger.info(f"💾 Disk space: {free_gb:.1f}GB free")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not check disk space: {e}")
        return True


def save_evaluation_metadata(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Save evaluation metadata for reproducibility."""
    metadata_file = Path(config["results_dir"]) / "evaluation_metadata.json"
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_info": get_git_info(),
        "system_info": get_system_info(),
        "config_hash": hash(str(config)),
        "python_version": sys.version,
        "package_versions": get_package_versions()
    }
    
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Saved evaluation metadata to {metadata_file}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save metadata: {e}")


def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    try:
        import subprocess
        
        # Get git commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Get git branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Check if working directory is clean
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        return {
            "commit_hash": commit_hash,
            "branch": branch,
            "clean_working_directory": len(status) == 0,
            "status": status if status else "clean"
        }
    except Exception:
        return {"error": "Git information not available"}


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "disk_free_gb": psutil.disk_usage('.').free / (1024**3),
        "hostname": platform.node()
    }


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    packages = {
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    
    # Try to get versions of other packages
    optional_packages = [
        "transformers", "sentence_transformers", "chromadb", 
        "langchain", "pandas", "scikit-learn"
    ]
    
    for package_name in optional_packages:
        try:
            package = __import__(package_name)
            packages[package_name] = getattr(package, "__version__", "unknown")
        except ImportError:
            packages[package_name] = "not installed"
    
    return packages


# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor performance metrics during evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = []
        self.start_time = None
        
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.metrics = []
        
    def record_metric(self, name: str, value: Any, timestamp: Optional[float] = None):
        """Record a performance metric."""
        if timestamp is None:
            timestamp = time.time()
            
        self.metrics.append({
            "name": name,
            "value": value,
            "timestamp": timestamp,
            "elapsed_seconds": timestamp - self.start_time if self.start_time else 0
        })
        
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all recorded metrics."""
        return self.metrics.copy()
        
    def save_metrics(self, output_path: Path):
        """Save metrics to file."""
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)


# Utility functions for evaluation orchestration
def estimate_evaluation_time(benchmarks: Dict, evaluators: Dict, config: Dict) -> timedelta:
    """Estimate total evaluation time based on configuration."""
    # Base time estimates (in minutes)
    benchmark_times = {
        "mirage": 15,
        "medreason": 20,
        "pubmedqa": 10,
        "msmarco": 25
    }
    
    evaluator_multipliers = {
        "kg_system": 1.0,
        "hierarchical_system": 1.5
    }
    
    total_minutes = 0
    for benchmark_name in benchmarks:
        base_time = benchmark_times.get(benchmark_name, 15)
        for evaluator_name in evaluators:
            multiplier = evaluator_multipliers.get(evaluator_name, 1.0)
            total_minutes += base_time * multiplier
    
    # Add overhead (20%)
    total_minutes *= 1.2
    
    return timedelta(minutes=total_minutes)


def create_evaluation_summary(results: Dict[str, Any]) -> str:
    """Create a text summary of evaluation results."""
    summary_lines = []
    summary_lines.append("HIERRAGMED EVALUATION SUMMARY")
    summary_lines.append("=" * 50)
    
    metadata = results.get("metadata", {})
    summary_lines.append(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
    summary_lines.append(f"Platform: {metadata.get('platform', 'Unknown')}")
    summary_lines.append(f"Duration: {metadata.get('total_duration_seconds', 0):.1f} seconds")
    
    if "gpu_info" in metadata:
        gpu_info = metadata["gpu_info"]
        summary_lines.append(f"GPU: {gpu_info.get('gpu_name', 'Unknown')}")
        summary_lines.append(f"GPU Memory: {gpu_info.get('gpu_memory_gb', 0):.1f} GB")
    
    summary_lines.append("")
    summary_lines.append("BENCHMARK RESULTS:")
    summary_lines.append("-" * 30)
    
    for benchmark_name, benchmark_data in results.get("results", {}).items():
        if "error" in benchmark_data:
            summary_lines.append(f"{benchmark_name.upper()}: FAILED")
            continue
            
        summary_lines.append(f"{benchmark_name.upper()}:")
        for model_name, model_data in benchmark_data.get("models", {}).items():
            if "results" in model_data and "error" not in model_data["results"]:
                metrics = model_data["results"].get("metrics", {})
                accuracy = metrics.get("accuracy", 0)
                summary_lines.append(f"  {model_name}: {accuracy:.1f}% accuracy")
    
    return "\n".join(summary_lines)


if __name__ == "__main__":
    # Validate environment before starting
    try:
        validate_environment()
        check_disk_space(min_gb=5.0)
    except Exception as e:
        logger.error(f"❌ Environment validation failed: {e}")
        sys.exit(1)
    
    # Run main evaluation
    try:
        main()
    finally:
        cleanup_evaluation()