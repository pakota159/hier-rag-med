# src/evaluation/run_evaluation.py
"""
Updated HierRAGMed Evaluation Runner
PubMedQA disabled as it's already included in MIRAGE benchmark
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

from src.evaluation.data.data_loader import BenchmarkDataLoader
from src.evaluation.benchmarks.mirage_benchmark import MIRAGEBenchmark
from src.evaluation.benchmarks.medreason_benchmark import MedReasonBenchmark
from src.evaluation.benchmarks.pubmedqa_benchmark import PubMedQABenchmark  # Disabled benchmark
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
    log_dir = Path(results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluation_{timestamp}.log"
    
    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add(log_file, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}")
    
    logger.info(f"📝 Logging initialized - log file: {log_file}")


def load_evaluation_config(config_path: Optional[Path] = None) -> Dict:
    """Load evaluation configuration."""
    if config_path and config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"📋 Loaded config from {config_path}")
    else:
        # Default configuration with PubMedQA disabled
        config = {
            "benchmarks": {
                "mirage": {"enabled": True, "sample_size": None},
                "medreason": {"enabled": True, "sample_size": None},
                "pubmedqa": {"enabled": False, "disabled_reason": "Already included in MIRAGE"},
                "msmarco": {"enabled": True, "sample_size": None}
            },
            "systems": {
                "kg_system": {"enabled": True},
                "hierarchical_system": {"enabled": True}
            },
            "evaluation": {
                "results_dir": "evaluation/results",
                "cache_dir": "evaluation/cache",
                "save_detailed_results": True
            }
        }
        logger.info("📋 Using default configuration")
    
    return config


def initialize_benchmarks(config: Dict) -> Dict[str, object]:
    """Initialize benchmark objects."""
    benchmarks = {}
    benchmark_config = config.get("benchmarks", {})
    
    # MIRAGE - Enhanced with QADataset integration
    if benchmark_config.get("mirage", {}).get("enabled", True):
        logger.info("🧠 Initializing MIRAGE benchmark...")
        benchmarks["mirage"] = MIRAGEBenchmark(benchmark_config.get("mirage", {}))
        logger.info("   ✅ MIRAGE ready (with QADataset integration)")
    
    # MedReason - Knowledge graph reasoning
    if benchmark_config.get("medreason", {}).get("enabled", True):
        logger.info("🔗 Initializing MedReason benchmark...")
        benchmarks["medreason"] = MedReasonBenchmark(benchmark_config.get("medreason", {}))
        logger.info("   ✅ MedReason ready")
    
    # PubMedQA - DISABLED
    if benchmark_config.get("pubmedqa", {}).get("enabled", False):
        logger.warning("⚠️ PubMedQA is configured as enabled, but it's disabled by design")
        logger.info("   PubMedQA questions are already included in MIRAGE benchmark")
    else:
        logger.info("🚫 PubMedQA benchmark DISABLED (already in MIRAGE)")
    
    # MS MARCO - Retrieval evaluation
    if benchmark_config.get("msmarco", {}).get("enabled", True):
        logger.info("🔍 Initializing MS MARCO benchmark...")
        benchmarks["msmarco"] = MSMARCOBenchmark(benchmark_config.get("msmarco", {}))
        logger.info("   ✅ MS MARCO ready")
    
    logger.info(f"🎯 Initialized {len(benchmarks)} active benchmarks")
    return benchmarks


def initialize_evaluators(config: Dict) -> Dict[str, object]:
    """Initialize evaluator objects."""
    evaluators = {}
    system_config = config.get("systems", {})
    
    # KG System Evaluator
    if system_config.get("kg_system", {}).get("enabled", True):
        logger.info("🔗 Initializing KG System Evaluator...")
        evaluators["kg_system"] = KGEvaluator(system_config.get("kg_system", {}))
        logger.info("   ✅ KG Evaluator ready")
    
    # Hierarchical System Evaluator
    if system_config.get("hierarchical_system", {}).get("enabled", True):
        logger.info("🏗️ Initializing Hierarchical System Evaluator...")
        evaluators["hierarchical_system"] = HierarchicalEvaluator(system_config.get("hierarchical_system", {}))
        logger.info("   ✅ Hierarchical Evaluator ready")
    
    logger.info(f"🎯 Initialized {len(evaluators)} system evaluators")
    return evaluators


def run_single_benchmark(benchmark_name: str, benchmark: object, evaluators: Dict, config: Dict) -> Dict:
    """Run evaluation on a single benchmark."""
    logger.info(f"🚀 Starting evaluation on {benchmark_name.upper()} benchmark")
    
    # Skip disabled benchmarks
    if benchmark_name == "pubmedqa":
        logger.warning(f"⚠️ Skipping {benchmark_name} - disabled (already in MIRAGE)")
        return {
            "benchmark": benchmark_name,
            "status": "disabled",
            "reason": "Already included in MIRAGE benchmark",
            "results": {}
        }
    
    start_time = time.time()
    benchmark_results = {
        "benchmark": benchmark_name,
        "status": "running",
        "results": {},
        "timing": {}
    }
    
    try:
        # Load dataset
        logger.info(f"   📥 Loading {benchmark_name} dataset...")
        dataset_start = time.time()
        questions = benchmark.load_dataset()
        dataset_time = time.time() - dataset_start
        
        if not questions:
            logger.warning(f"   ⚠️ No questions loaded for {benchmark_name}")
            benchmark_results["status"] = "no_data"
            return benchmark_results
        
        logger.info(f"   📊 Loaded {len(questions)} questions in {dataset_time:.2f}s")
        benchmark_results["timing"]["dataset_load"] = dataset_time
        benchmark_results["question_count"] = len(questions)
        
        # Run evaluation for each system
        for system_name, evaluator in evaluators.items():
            logger.info(f"   🔄 Evaluating {system_name} on {benchmark_name}...")
            system_start = time.time()
            
            system_results = []
            for i, question in enumerate(questions):
                try:
                    # Generate response using the evaluator
                    response, retrieved_docs = evaluator.generate_response(question)
                    
                    # Evaluate the response
                    evaluation = benchmark.evaluate_response(question, response, retrieved_docs)
                    
                    # Store result
                    result = {
                        "question_id": question.get("question_id", f"{benchmark_name}_{i}"),
                        "question": question.get("question", ""),
                        "response": response,
                        "evaluation": evaluation,
                        "retrieved_docs_count": len(retrieved_docs)
                    }
                    system_results.append(result)
                    
                    # Progress logging
                    if (i + 1) % 10 == 0 or (i + 1) == len(questions):
                        logger.info(f"     Progress: {i + 1}/{len(questions)} questions processed")
                
                except Exception as e:
                    logger.error(f"     ❌ Failed to process question {i}: {e}")
                    continue
            
            system_time = time.time() - system_start
            
            # Generate summary
            summary = benchmark.get_evaluation_summary(system_results)
            
            benchmark_results["results"][system_name] = {
                "individual_results": system_results,
                "summary": summary,
                "processing_time": system_time,
                "questions_processed": len(system_results)
            }
            
            logger.info(f"   ✅ {system_name} completed in {system_time:.2f}s")
            logger.info(f"     Overall Score: {summary.get('average_scores', {}).get('overall_score', 0.0):.3f}")
        
        benchmark_results["status"] = "completed"
        total_time = time.time() - start_time
        benchmark_results["timing"]["total"] = total_time
        
        logger.info(f"🏁 {benchmark_name.upper()} evaluation completed in {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ {benchmark_name} evaluation failed: {e}")
        benchmark_results["status"] = "failed"
        benchmark_results["error"] = str(e)
    
    return benchmark_results


def run_evaluation(config_path: Optional[Path] = None, results_dir: Optional[Path] = None, 
                  quick_mode: bool = False) -> Dict:
    """
    Run complete evaluation across all enabled benchmarks.
    
    Args:
        config_path: Path to evaluation configuration file
        results_dir: Directory to save results
        quick_mode: Whether to run quick evaluation with limited samples
        
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
                config["benchmarks"][benchmark_name]["sample_size"] = 10
    
    # Initialize components
    benchmarks = initialize_benchmarks(config)
    evaluators = initialize_evaluators(config)
    
    if not benchmarks:
        logger.error("❌ No benchmarks initialized")
        return {"error": "No benchmarks available"}
    
    if not evaluators:
        logger.error("❌ No evaluators initialized")
        return {"error": "No evaluators available"}
    
    # Run evaluations
    evaluation_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "environment": env_info,
            "config": config,
            "quick_mode": quick_mode
        },
        "benchmarks": {},
        "summary": {}
    }
    
    total_start_time = time.time()
    
    # Process each benchmark
    for benchmark_name, benchmark in benchmarks.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"BENCHMARK: {benchmark_name.upper()}")
        logger.info(f"{'='*60}")
        
        benchmark_result = run_single_benchmark(benchmark_name, benchmark, evaluators, config)
        evaluation_results["benchmarks"][benchmark_name] = benchmark_result
        
        # Clear GPU memory between benchmarks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Generate overall summary
    total_time = time.time() - total_start_time
    evaluation_results["metadata"]["total_evaluation_time"] = total_time
    
    successful_benchmarks = [
        name for name, result in evaluation_results["benchmarks"].items()
        if result.get("status") == "completed"
    ]
    
    evaluation_results["summary"] = {
        "total_benchmarks": len(benchmarks),
        "successful_benchmarks": len(successful_benchmarks),
        "failed_benchmarks": len(benchmarks) - len(successful_benchmarks),
        "total_time": total_time,
        "benchmarks_completed": successful_benchmarks
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"evaluation_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        logger.info(f"💾 Results saved to {results_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successful benchmarks: {len(successful_benchmarks)}")
    logger.info(f"❌ Failed benchmarks: {len(benchmarks) - len(successful_benchmarks)}")
    logger.info(f"⏱️  Total time: {total_time:.2f}s")
    logger.info(f"💾 Results saved to: {results_file}")
    
    if "pubmedqa" not in benchmarks:
        logger.info("ℹ️  Note: PubMedQA benchmark disabled (questions included in MIRAGE)")
    
    return evaluation_results


def main():
    """Command line interface for evaluation."""
    parser = argparse.ArgumentParser(description="HierRAGMed Evaluation Runner")
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument("--results-dir", type=Path, help="Directory to save results")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation with limited samples")
    parser.add_argument("--benchmark", type=str, choices=["mirage", "medreason", "msmarco"], 
                       help="Run specific benchmark only")
    
    args = parser.parse_args()
    
    try:
        if args.benchmark:
            # Run single benchmark
            logger.info(f"🎯 Running single benchmark: {args.benchmark}")
            # Implementation for single benchmark would go here
            logger.info("Single benchmark mode not implemented yet - running full evaluation")
        
        # Run full evaluation
        results = run_evaluation(
            config_path=args.config,
            results_dir=args.results_dir,
            quick_mode=args.quick
        )
        
        if "error" in results:
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