#!/usr/bin/env python3
"""
HierRAGMed Evaluation Runner
Updated with simplified --quick and --full modes and --benchmark filtering
"""

import sys
import time
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import evaluation components
try:
    from src.evaluation.benchmarks.mirage_benchmark import MIRAGEBenchmark
    from src.evaluation.benchmarks.medreason_benchmark import MedReasonBenchmark
    from src.evaluation.benchmarks.msmarco_benchmark import MSMARCOBenchmark
    from src.evaluation.benchmarks.pubmedqa_benchmark import PubMedQABenchmark
    
    from src.evaluation.evaluators.kg_evaluator import KGEvaluator
    from src.evaluation.evaluators.hierarchical_evaluator import HierarchicalEvaluator
    
    # Import utility functions directly from the files that exist
    from src.evaluation.utils.config_loader import ConfigLoader
    
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)


def setup_logging(log_dir: str) -> None:
    """Setup logging configuration."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_path / "evaluation.log",
        rotation="100 MB",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )


def validate_gpu_environment() -> Dict:
    """Validate GPU environment and return system info."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            logger.info(f"🔥 GPU: {gpu_name}")
            logger.info(f"   Memory: {gpu_memory:.1f}GB total")
            logger.info(f"   CUDA: {torch.version.cuda}")
            
            return {
                "gpu_available": True, 
                "gpu_count": gpu_count,
                "gpu_name": gpu_name,
                "gpu_memory_gb": gpu_memory
            }
        else:
            logger.warning("⚠️ CUDA not available")
            return {"gpu_available": False}
            
    except ImportError:
        logger.warning("⚠️ PyTorch not available")
        return {"gpu_available": False}
    except Exception as e:
        logger.warning(f"⚠️ GPU validation failed: {e}")
        return {"gpu_available": False, "error": str(e)}


def setup_gpu_optimization() -> None:
    """Setup GPU optimizations."""
    try:
        import torch
        if torch.cuda.is_available():
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.85)
            # Enable cudNN benchmark
            torch.backends.cudnn.benchmark = True
            logger.info("✅ GPU optimizations configured")
        else:
            logger.info("ℹ️ No GPU available, skipping optimizations")
    except ImportError:
        logger.warning("⚠️ PyTorch not available for GPU optimization")
    except Exception as e:
        logger.warning(f"⚠️ GPU optimization setup failed: {e}")


def load_evaluation_config(config_path: Optional[Path] = None) -> Dict:
    """Load evaluation configuration."""
    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        return config.to_dict()
    except Exception as e:
        logger.warning(f"⚠️ Failed to load config with ConfigLoader: {e}")
        logger.info("📄 Using fallback configuration")
        
        # Fallback configuration
        return {
            "benchmarks": {
                "mirage": {"enabled": True, "sample_size": 1000},
                "medreason": {"enabled": True, "sample_size": 500},
                "msmarco": {"enabled": True, "sample_size": 1000},
                "pubmedqa": {"enabled": True, "sample_size": 500}
            },
            "systems": {
                "kg_system": {"enabled": True},
                "hierarchical_system": {"enabled": True}
            },
            "performance": {
                "max_workers": 4,
                "batch_size": 32
            }
        }


def initialize_benchmarks(config: Dict, benchmark_filter: Optional[str] = None) -> Dict[str, object]:
    """Initialize benchmark objects with optional filtering."""
    benchmarks = {}
    benchmark_config = config.get("benchmarks", {})
    
    # Define available benchmarks
    available_benchmarks = {
        "mirage": (MIRAGEBenchmark, "🧠 Initializing MIRAGE benchmark..."),
        "medreason": (MedReasonBenchmark, "🔗 Initializing MedReason benchmark..."),
        "msmarco": (MSMARCOBenchmark, "🔍 Initializing MS MARCO benchmark..."),
        "pubmedqa": (PubMedQABenchmark, "📚 Initializing PubMedQA benchmark...")
    }
    
    # Filter benchmarks if specified
    if benchmark_filter:
        if benchmark_filter not in available_benchmarks:
            logger.error(f"❌ Unknown benchmark: {benchmark_filter}")
            return {}
        available_benchmarks = {benchmark_filter: available_benchmarks[benchmark_filter]}
        logger.info(f"🎯 Running single benchmark: {benchmark_filter}")
    
    # Initialize selected benchmarks
    for benchmark_name, (benchmark_class, init_msg) in available_benchmarks.items():
        benchmark_cfg = benchmark_config.get(benchmark_name, {})
        
        # Skip if explicitly disabled
        if not benchmark_cfg.get("enabled", True):
            logger.info(f"🚫 {benchmark_name.upper()} benchmark DISABLED")
            continue
        
        try:
            logger.info(init_msg)
            benchmarks[benchmark_name] = benchmark_class(benchmark_cfg)
            logger.info(f"   ✅ {benchmark_name.upper()} ready")
        except Exception as e:
            logger.error(f"❌ Failed to initialize {benchmark_name}: {e}")
            continue
    
    logger.info(f"🎯 Initialized {len(benchmarks)} active benchmarks")
    return benchmarks


def initialize_evaluators(config: Dict, models_filter: Optional[Set[str]] = None) -> Dict[str, object]:
    """Initialize evaluator objects with optional model filtering."""
    evaluators = {}
    system_config = config.get("systems", {})
    
    # Define available evaluators
    available_evaluators = {
        "kg_system": (KGEvaluator, "🔗 Initializing KG System Evaluator..."),
        "hierarchical_system": (HierarchicalEvaluator, "🏗️ Initializing Hierarchical System Evaluator...")
    }
    
    # Filter evaluators if specified
    if models_filter:
        # Only initialize requested models
        filtered_evaluators = {}
        for model_name in models_filter:
            if model_name in available_evaluators:
                filtered_evaluators[model_name] = available_evaluators[model_name]
            else:
                logger.warning(f"⚠️ Unknown model: {model_name}")
        available_evaluators = filtered_evaluators
        logger.info(f"🤖 Running specific models: {', '.join(models_filter)}")
    
    # Initialize selected evaluators
    for evaluator_name, (evaluator_class, init_msg) in available_evaluators.items():
        evaluator_cfg = system_config.get(evaluator_name, {})
        
        # Skip if explicitly disabled (unless specifically requested)
        if not models_filter and not evaluator_cfg.get("enabled", True):
            logger.info(f"🚫 {evaluator_name} evaluator DISABLED")
            continue
        
        try:
            logger.info(init_msg)
            evaluators[evaluator_name] = evaluator_class(evaluator_cfg)
            logger.info(f"   ✅ {evaluator_name} Evaluator ready")
        except Exception as e:
            logger.error(f"❌ Failed to initialize {evaluator_name}: {e}")
            continue
    
    logger.info(f"🎯 Initialized {len(evaluators)} system evaluators")
    return evaluators


def run_single_benchmark(benchmark_name: str, benchmark: object, evaluators: Dict, config: Dict) -> Dict:
    """Run evaluation on a single benchmark."""
    logger.info(f"🚀 Starting {benchmark_name.upper()} evaluation")
    
    benchmark_results = {
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "results": {},
        "timing": {}
    }
    
    benchmark_start = time.time()
    
    try:
        # Load benchmark data
        test_data = benchmark.load_test_data()
        logger.info(f"📊 Loaded {len(test_data)} test cases for {benchmark_name}")
        
        # Run each evaluator
        for evaluator_name, evaluator in evaluators.items():
            logger.info(f"🔄 Running {evaluator_name} on {benchmark_name}")
            
            evaluator_start = time.time()
            try:
                # Run evaluation
                evaluation_results = evaluator.evaluate(test_data, benchmark)
                
                evaluator_time = time.time() - evaluator_start
                
                benchmark_results["results"][evaluator_name] = {
                    "status": "completed",
                    "metrics": evaluation_results["metrics"],
                    "processing_time": evaluator_time,
                    "questions_processed": len(test_data),
                    "details": evaluation_results.get("details", {})
                }
                
                logger.info(f"   ✅ {evaluator_name}: {evaluation_results['metrics'].get('accuracy', 0):.1f}% accuracy in {evaluator_time:.1f}s")
                
            except Exception as e:
                evaluator_time = time.time() - evaluator_start
                logger.error(f"   ❌ {evaluator_name} failed: {e}")
                
                benchmark_results["results"][evaluator_name] = {
                    "status": "failed",
                    "error": str(e),
                    "processing_time": evaluator_time
                }
        
        benchmark_results["status"] = "completed"
        benchmark_results["timing"]["total"] = time.time() - benchmark_start
        
    except Exception as e:
        logger.error(f"❌ Benchmark {benchmark_name} failed: {e}")
        benchmark_results["status"] = "failed"
        benchmark_results["error"] = str(e)
        benchmark_results["timing"]["total"] = time.time() - benchmark_start
    
    benchmark_results["end_time"] = datetime.now().isoformat()
    return benchmark_results


def save_evaluation_results(results: Dict, results_dir: Path, models_filter: Optional[Set[str]] = None) -> Path:
    """Save evaluation results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename based on models and timestamp
    if models_filter:
        models_str = "_".join(sorted(models_filter))
        filename = f"evaluation_{models_str}_{timestamp}.json"
    else:
        filename = f"evaluation_all_models_{timestamp}.json"
    
    results_file = results_dir / filename
    
    try:
        # Convert any non-serializable objects to strings
        serializable_results = json.loads(json.dumps(results, default=str))
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {results_file}")
        return results_file
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        raise


def run_evaluation(
    config_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    quick_mode: bool = False,
    full_mode: bool = False,
    benchmark_filter: Optional[str] = None,
    models_filter: Optional[Set[str]] = None
) -> Dict:
    """
    Run HierRAGMed evaluation with multiple systems and benchmarks.
    
    Args:
        config_path: Path to evaluation configuration file
        results_dir: Directory to save results
        quick_mode: Whether to run quick evaluation with 200 samples
        full_mode: Whether to run on full dataset
        benchmark_filter: Run only specific benchmark
        models_filter: Run only specific models
        
    Returns:
        Dictionary containing all evaluation results
    """
    # Setup
    if results_dir is None:
        results_dir = Path("evaluation/results")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(results_dir))
    
    logger.info("🚀 Starting HierRAGMed Evaluation")
    logger.info("=" * 60)
    
    # Log mode information
    if quick_mode:
        logger.info("⚡ Quick mode enabled - limiting to 200 samples per benchmark")
    elif full_mode:
        logger.info("🔥 Full mode enabled - using complete datasets")
    else:
        logger.info("📊 Standard mode - using default sample sizes from config")
    
    # Log filtering information
    if benchmark_filter:
        logger.info(f"🎯 Running single benchmark: {benchmark_filter}")
    if models_filter:
        logger.info(f"🤖 Running specific models: {', '.join(models_filter)}")
    
    # Validate environment
    env_info = validate_gpu_environment()
    setup_gpu_optimization()
    
    # Load configuration
    config = load_evaluation_config(config_path)
    
    # Apply mode-specific settings
    if quick_mode:
        logger.info("⚡ Applying quick mode settings...")
        for benchmark_name in config.get("benchmarks", {}):
            if config["benchmarks"][benchmark_name].get("enabled", True):
                config["benchmarks"][benchmark_name]["sample_size"] = 200
                logger.info(f"   {benchmark_name}: limited to 200 samples")
    elif full_mode:
        logger.info("🔥 Applying full mode settings...")
        for benchmark_name in config.get("benchmarks", {}):
            if config["benchmarks"][benchmark_name].get("enabled", True):
                # Remove sample size limit or set to very high number
                config["benchmarks"][benchmark_name]["sample_size"] = None
                logger.info(f"   {benchmark_name}: using full dataset")
    
    # Initialize components
    benchmarks = initialize_benchmarks(config, benchmark_filter)
    evaluators = initialize_evaluators(config, models_filter)
    
    if not benchmarks:
        logger.error("❌ No benchmarks initialized")
        return {"error": "No benchmarks available"}
    
    if not evaluators:
        logger.error("❌ No evaluators initialized")
        return {"error": "No evaluators available"}
    
    # Run evaluations
    all_results = {
        "evaluation_id": f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "start_time": datetime.now().isoformat(),
        "config": {
            "quick_mode": quick_mode,
            "full_mode": full_mode,
            "benchmark_filter": benchmark_filter,
            "models_filter": list(models_filter) if models_filter else None
        },
        "environment": env_info,
        "benchmarks": {},
        "status": "running"
    }
    
    evaluation_start = time.time()
    
    try:
        # Run each benchmark
        for benchmark_name, benchmark in benchmarks.items():
            logger.info(f"\n📊 Starting {benchmark_name.upper()} benchmark")
            benchmark_results = run_single_benchmark(
                benchmark_name, benchmark, evaluators, config
            )
            all_results["benchmarks"][benchmark_name] = benchmark_results
        
        all_results["status"] = "completed"
        total_time = time.time() - evaluation_start
        all_results["total_time"] = total_time
        
        # Save results with proper serialization
        results_file = save_evaluation_results(all_results, results_dir, models_filter)
        
        logger.info(f"🎉 Evaluation completed in {total_time:.2f}s")
        
        # Print summary
        print_evaluation_summary(all_results)
        
        return all_results
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        all_results["status"] = "failed"
        all_results["error"] = str(e)
        
        # Try to save partial results
        try:
            results_file = save_evaluation_results(all_results, results_dir, models_filter)
        except:
            logger.error("❌ Failed to save partial results")
        
        return all_results


def print_evaluation_summary(results: Dict) -> None:
    """Print evaluation summary."""
    logger.info("\n" + "=" * 60)
    logger.info("📊 EVALUATION SUMMARY")
    logger.info("=" * 60)
    
    total_questions = 0
    total_time = 0
    
    for benchmark_name, benchmark_data in results.get("benchmarks", {}).items():
        logger.info(f"\n🎯 {benchmark_name.upper()}")
        logger.info("-" * 30)
        
        if benchmark_data.get("status") == "completed":
            benchmark_time = benchmark_data.get("timing", {}).get("total", 0)
            total_time += benchmark_time
            
            for system_name, system_data in benchmark_data.get("results", {}).items():
                if "metrics" in system_data:
                    accuracy = system_data["metrics"].get("accuracy", 0)
                    questions_processed = system_data.get("questions_processed", 0)
                    processing_time = system_data.get("processing_time", 0)
                    
                    total_questions += questions_processed
                    
                    logger.info(f"   {system_name}: {accuracy:.1f}% ({questions_processed} questions, {processing_time:.1f}s)")
                else:
                    logger.error(f"   {system_name}: {system_data.get('status', 'failed')}")
        else:
            logger.error(f"   ❌ {benchmark_data.get('status', 'failed')}")
    
    # Overall summary
    logger.info(f"\n📈 OVERALL RESULTS:")
    logger.info(f"   Total Questions: {total_questions}")
    logger.info(f"   Total Time: {total_time:.1f}s")
    if total_questions > 0:
        logger.info(f"   Average Time per Question: {total_time/total_questions:.2f}s")


def main():
    """Command line interface for evaluation runner."""
    parser = argparse.ArgumentParser(description="HierRAGMed Evaluation Runner")
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument("--results-dir", type=Path, help="Directory to save results")
    
    # Mode arguments - mutually exclusive
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true", 
                           help="Run quick evaluation with 200 samples per benchmark")
    mode_group.add_argument("--full", action="store_true", 
                           help="Run evaluation on full datasets (no sample limit)")
    
    # Filtering arguments
    parser.add_argument("--benchmark", type=str, 
                       choices=["mirage", "medreason", "msmarco", "pubmedqa"], 
                       help="Run specific benchmark only")
    parser.add_argument("--models", type=str, 
                       help="Comma-separated list of models to evaluate (e.g., 'hierarchical_system' or 'kg_system,hierarchical_system')")
    
    args = parser.parse_args()
    
    try:
        # Parse models filter
        models_filter = None
        if args.models:
            models_filter = set(model.strip() for model in args.models.split(","))
            available_models = {"kg_system", "hierarchical_system"}
            invalid_models = models_filter - available_models
            if invalid_models:
                logger.error(f"❌ Invalid models: {invalid_models}")
                logger.info(f"Available models: {available_models}")
                sys.exit(1)
        
        # Run evaluation
        results = run_evaluation(
            config_path=args.config,
            results_dir=args.results_dir,
            quick_mode=args.quick,
            full_mode=args.full,
            benchmark_filter=args.benchmark,
            models_filter=models_filter
        )
        
        if results.get("status") == "failed":
            sys.exit(1)
        
        logger.info("🎉 Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()