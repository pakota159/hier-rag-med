#!/usr/bin/env python3
"""
Updated HierRAGMed Evaluation Runner with --models parameter support
Includes fixes for JSON serialization and comprehensive error handling
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import json
from datetime import datetime

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import evaluation components
from src.evaluation.benchmarks.mirage_benchmark import MIRAGEBenchmark
from src.evaluation.benchmarks.medreason_benchmark import MedReasonBenchmark
from src.evaluation.benchmarks.msmarco_benchmark import MSMARCOBenchmark
from src.evaluation.benchmarks.pubmedqa_benchmark import PubMedQABenchmark

from src.evaluation.evaluators.kg_evaluator import KGEvaluator
from src.evaluation.evaluators.hierarchical_evaluator import HierarchicalEvaluator

from src.evaluation.utils.config_loader import ConfigLoader
from src.evaluation.utils.result_processor import ResultProcessor
from src.evaluation.utils.device_manager import DeviceManager


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
        device_manager = DeviceManager()
        gpu_info = device_manager.get_gpu_info()
        
        logger.info(f"🔥 GPU: {gpu_info.name}")
        logger.info(f"   Memory: {gpu_info.memory_total:.1f}GB total")
        logger.info(f"   CUDA: {gpu_info.cuda_version}")
        
        return {"gpu_available": True, "gpu_info": gpu_info}
        
    except Exception as e:
        logger.warning(f"⚠️ GPU validation failed: {e}")
        return {"gpu_available": False, "error": str(e)}


def setup_gpu_optimization() -> None:
    """Setup GPU optimizations."""
    try:
        device_manager = DeviceManager()
        device_manager.setup_gpu_optimization()
        logger.info("✅ GPU optimizations configured")
    except Exception as e:
        logger.warning(f"⚠️ GPU optimization setup failed: {e}")


def load_evaluation_config(config_path: Optional[Path] = None) -> Dict:
    """Load evaluation configuration."""
    try:
        config_loader = ConfigLoader()
        
        if config_path and config_path.exists():
            logger.info(f"📁 Loading config from: {config_path}")
            config = config_loader.load_config(config_path)
        else:
            logger.info("📁 Loading default configuration")
            config = config_loader.load_config()
        
        return config.to_dict()
        
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        # Return minimal default config
        return {
            "benchmarks": {
                "mirage": {"enabled": True},
                "medreason": {"enabled": True},
                "msmarco": {"enabled": True}
            },
            "systems": {
                "kg_system": {"enabled": True},
                "hierarchical_system": {"enabled": True}
            }
        }


def initialize_benchmarks(config: Dict, benchmark_filter: Optional[str] = None) -> Dict:
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
    logger.info(f"🏁 Starting {benchmark_name.upper()} evaluation")
    start_time = time.time()
    
    benchmark_results = {
        "benchmark_name": benchmark_name,
        "start_time": datetime.now().isoformat(),
        "status": "running",
        "results": {},
        "timing": {
            "start": start_time
        }
    }
    
    try:
        # Get benchmark questions
        questions = benchmark.get_questions()
        if not questions:
            logger.warning(f"⚠️ No questions available for {benchmark_name}")
            benchmark_results["status"] = "skipped"
            benchmark_results["error"] = "No questions available"
            return benchmark_results
        
        logger.info(f"   📊 Processing {len(questions)} questions")
        
        # Evaluate each system
        for system_name, evaluator in evaluators.items():
            if not evaluator.enabled:
                logger.info(f"   🚫 Skipping disabled system: {system_name}")
                continue
            
            logger.info(f"   🔧 Setting up {system_name}...")
            system_start = time.time()
            
            try:
                # Setup system
                evaluator.setup_model()
                
                # Run evaluation
                system_results = []
                for i, question in enumerate(questions):
                    try:
                        # Retrieve documents
                        retrieved_docs = evaluator.retrieve_documents(
                            question.get("question", "")
                        )
                        
                        # Generate response
                        response = evaluator.generate_response(
                            question.get("question", "")
                        )
                        
                        # Evaluate response
                        evaluation = benchmark.evaluate_response(
                            question, response, retrieved_docs
                        )
                        
                        evaluation.update({
                            "question_id": question.get("question_id", f"q_{i}"),
                            "system_name": system_name,
                            "response": response,
                            "retrieved_docs_count": len(retrieved_docs)
                        })
                        
                        system_results.append(evaluation)
                        
                        # Progress logging
                        if (i + 1) % 100 == 0:
                            logger.info(f"     Progress: {i + 1}/{len(questions)} questions")
                    
                    except Exception as e:
                        logger.error(f"Error processing question {i}: {e}")
                        system_results.append({
                            "question_id": question.get("question_id", f"q_{i}"),
                            "system_name": system_name,
                            "error": str(e),
                            "status": "failed"
                        })
                        continue
                
                system_time = time.time() - system_start
                
                # Calculate system metrics
                metrics = benchmark.calculate_metrics(system_results)
                
                benchmark_results["results"][system_name] = {
                    "individual_results": system_results,
                    "metrics": metrics,
                    "processing_time": system_time,
                    "questions_processed": len(system_results)
                }
                
                logger.info(f"   ✅ {system_name} completed in {system_time:.2f}s")
                logger.info(f"     Overall Score: {metrics.get('accuracy', 0.0):.1f}%")
                
            except Exception as e:
                logger.error(f"❌ {system_name} evaluation failed: {e}")
                benchmark_results["results"][system_name] = {
                    "error": str(e),
                    "status": "failed"
                }
                continue
        
        benchmark_results["status"] = "completed"
        total_time = time.time() - start_time
        benchmark_results["timing"]["total"] = total_time
        
        logger.info(f"🏁 {benchmark_name.upper()} evaluation completed in {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ {benchmark_name} evaluation failed: {e}")
        benchmark_results["status"] = "failed"
        benchmark_results["error"] = str(e)
    
    return benchmark_results


def convert_gpu_info_to_dict(gpu_info) -> Dict[str, Any]:
    """Convert GPUInfo object to JSON-serializable dictionary."""
    if hasattr(gpu_info, 'name'):  # Check if it's a GPUInfo object
        return {
            "name": gpu_info.name,
            "memory_total": gpu_info.memory_total,
            "memory_free": gpu_info.memory_free,
            "memory_used": gpu_info.memory_used,
            "cuda_version": gpu_info.cuda_version,
            "device_id": gpu_info.device_id,
            "compute_capability": gpu_info.compute_capability
        }
    return gpu_info  # Already a dict or other serializable type


def save_evaluation_results(all_results: Dict, results_dir: Path, models_filter: Optional[Set[str]]) -> Path:
    """Save evaluation results with proper JSON serialization."""
    
    # Convert GPUInfo to serializable dict
    if "environment" in all_results and "gpu_info" in all_results["environment"]:
        all_results["environment"]["gpu_info"] = convert_gpu_info_to_dict(
            all_results["environment"]["gpu_info"]
        )
    
    # Ensure all numpy types are converted to Python types
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        else:
            return obj
    
    # Convert numpy types in results
    all_results = convert_numpy_types(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if models_filter:
        model_suffix = "_".join(sorted(models_filter))
        results_file = results_dir / f"evaluation_results_{model_suffix}_{timestamp}.json"
    else:
        results_file = results_dir / f"evaluation_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"💾 Results saved to: {results_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        # Try to save with basic serialization
        backup_file = results_dir / f"evaluation_results_backup_{timestamp}.json"
        with open(backup_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"💾 Backup results saved to: {backup_file}")
        return backup_file
    
    return results_file


def run_evaluation(
    config_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    quick_mode: bool = False,
    benchmark_filter: Optional[str] = None,
    models_filter: Optional[Set[str]] = None
) -> Dict:
    """
    Run complete evaluation across benchmarks and models.
    
    Args:
        config_path: Path to evaluation configuration file
        results_dir: Directory to save results
        quick_mode: Whether to run quick evaluation with limited samples
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
    
    # Override with quick mode settings
    if quick_mode:
        logger.info("⚡ Quick mode enabled - limiting sample sizes")
        for benchmark_name in config.get("benchmarks", {}):
            if config["benchmarks"][benchmark_name].get("enabled", True):
                config["benchmarks"][benchmark_name]["sample_size"] = 200
    
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
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation with limited samples")
    parser.add_argument("--benchmark", type=str, choices=["mirage", "medreason", "msmarco", "pubmedqa"], 
                       help="Run specific benchmark only")
    
    # --models parameter
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