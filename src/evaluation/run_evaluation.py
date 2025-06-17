#!/usr/bin/env python3
"""
Main evaluation runner for HierRAGMed evaluation system.
Executes comprehensive evaluation across MIRAGE, MedReason, PubMedQA, and MS MARCO benchmarks.
"""

import argparse
import sys
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


def load_evaluation_config(config_path: Path) -> Dict:
    """Load evaluation configuration."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Loaded evaluation config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        raise


def setup_logging(config: Dict) -> None:
    """Setup logging for evaluation."""
    log_config = config.get("logging", {})
    
    # Create results directory
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup loguru
    logger.add(
        results_dir / "evaluation.log",
        rotation="1 day",
        retention="7 days",
        level=log_config.get("level", "INFO")
    )


def initialize_benchmarks(config: Dict) -> Dict:
    """Initialize all enabled benchmarks."""
    benchmarks = {}
    benchmark_config = config["benchmarks"]
    
    if benchmark_config["mirage"]["enabled"]:
        logger.info("🎯 Initializing MIRAGE benchmark...")
        benchmarks["mirage"] = MIRAGEBenchmark(benchmark_config["mirage"])
    
    if benchmark_config["medreason"]["enabled"]:
        logger.info("🧠 Initializing MedReason benchmark...")
        benchmarks["medreason"] = MedReasonBenchmark(benchmark_config["medreason"])
    
    if benchmark_config["pubmedqa"]["enabled"]:
        logger.info("📚 Initializing PubMedQA benchmark...")
        benchmarks["pubmedqa"] = PubMedQABenchmark(benchmark_config["pubmedqa"])
    
    if benchmark_config["msmarco"]["enabled"]:
        logger.info("🔍 Initializing MS MARCO benchmark...")
        benchmarks["msmarco"] = MSMARCOBenchmark(benchmark_config["msmarco"])
    
    logger.info(f"✅ Initialized {len(benchmarks)} benchmarks")
    return benchmarks


def initialize_evaluators(config: Dict) -> Dict:
    """Initialize all enabled model evaluators."""
    evaluators = {}
    model_config = config["models"]
    
    if model_config["kg_system"]["enabled"]:
        logger.info("📊 Initializing KG System evaluator...")
        evaluators["kg_system"] = KGEvaluator(model_config["kg_system"])
    
    if model_config["hierarchical_system"]["enabled"]:
        logger.info("🏗️ Initializing Hierarchical System evaluator...")
        evaluators["hierarchical_system"] = HierarchicalEvaluator(model_config["hierarchical_system"])
    
    logger.info(f"✅ Initialized {len(evaluators)} evaluators")
    return evaluators


def run_single_benchmark(
    benchmark_name: str,
    benchmark,
    evaluators: Dict,
    config: Dict
) -> Dict:
    """Run evaluation on a single benchmark for all models."""
    logger.info(f"🚀 Starting {benchmark_name} evaluation...")
    
    results = {}
    
    for model_name, evaluator in evaluators.items():
        logger.info(f"   📈 Evaluating {model_name} on {benchmark_name}...")
        
        try:
            # Run evaluation
            model_results = evaluator.evaluate_benchmark(benchmark)
            results[model_name] = model_results
            
            # Log key metrics
            if "overall_score" in model_results:
                score = model_results["overall_score"]
                logger.info(f"   ✅ {model_name}: {score:.2f}%")
            
        except Exception as e:
            logger.error(f"   ❌ {model_name} evaluation failed: {e}")
            results[model_name] = {"error": str(e), "status": "failed"}
    
    logger.info(f"✅ Completed {benchmark_name} evaluation")
    return results


def run_evaluation(
    config_path: Optional[Path] = None,
    benchmarks_filter: Optional[List[str]] = None,
    models_filter: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run comprehensive evaluation across all benchmarks and models.
    
    Args:
        config_path: Path to evaluation config file
        benchmarks_filter: List of benchmark names to run (None = all)
        models_filter: List of model names to evaluate (None = all)
        output_dir: Custom output directory
    
    Returns:
        Dict containing all evaluation results
    """
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    config = load_evaluation_config(config_path)
    setup_logging(config)
    
    # Override output directory if specified
    if output_dir:
        config["results_dir"] = str(output_dir)
    
    logger.info("🎯 Starting HierRAGMed Comprehensive Evaluation")
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
            "models": list(evaluators.keys())
        },
        "results": {}
    }
    
    for benchmark_name, benchmark in benchmarks.items():
        try:
            benchmark_results = run_single_benchmark(
                benchmark_name, benchmark, evaluators, config
            )
            all_results["results"][benchmark_name] = benchmark_results
            
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
    
    logger.info("🎉 Evaluation completed successfully!")
    return processed_results


def main():
    """Command line interface for evaluation."""
    parser = argparse.ArgumentParser(description="Run HierRAGMed evaluation")
    
    parser.add_argument(
        "--config", 
        type=Path,
        default=None,
        help="Path to evaluation config file"
    )
    
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["mirage", "medreason", "pubmedqa", "msmarco"],
        help="Specific benchmarks to run (default: all enabled)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+", 
        choices=["kg_system", "hierarchical_system"],
        help="Specific models to evaluate (default: all enabled)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory for results"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation with reduced sample sizes"
    )
    
    args = parser.parse_args()
    
    try:
        # Run evaluation
        results = run_evaluation(
            config_path=args.config,
            benchmarks_filter=args.benchmarks,
            models_filter=args.models,
            output_dir=args.output_dir
        )
        
        print("\n🎉 Evaluation Summary:")
        for benchmark, bench_results in results["results"].items():
            print(f"\n📊 {benchmark.upper()}:")
            for model, model_results in bench_results.items():
                if "overall_score" in model_results:
                    print(f"   {model}: {model_results['overall_score']:.2f}%")
                elif "error" in model_results:
                    print(f"   {model}: FAILED")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️ Evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())