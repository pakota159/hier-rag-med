#!/usr/bin/env python3
"""
Enhanced Evaluation Script for HierRAGMed with Medical Embedding Support
File: src/evaluation/run_evaluation.py

Updated to support Microsoft BiomedNLP-PubMedBERT medical embedding and enhanced evaluation.
"""

import argparse
import json
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from loguru import logger
from src.evaluation.benchmarks.mirage_benchmark import MIRAGEBenchmark
from src.evaluation.benchmarks.med_qa_benchmark import MedQABenchmark
from src.evaluation.benchmarks.pub_med_qa_benchmark import PubMedQABenchmark
from src.evaluation.evaluators.hierarchical_evaluator import HierarchicalEvaluator
from src.evaluation.evaluators.kg_evaluator import KGEvaluator
from src.evaluation.metrics.qa_metrics import QAMetrics


def setup_logging(log_level: str = "INFO"):
    """Setup enhanced logging for evaluation."""
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>Eval</cyan> | {message}"
    )
    
    # Add file logging
    log_dir = Path("evaluation/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_dir / f"evaluation_{int(time.time())}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    )


def parse_arguments():
    """Parse command line arguments with enhanced options."""
    parser = argparse.ArgumentParser(
        description="Enhanced HierRAGMed Evaluation with Medical Embedding Support"
    )
    
    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["hierarchical_system", "kg_system", "all"],
        default=["hierarchical_system"],
        help="Models to evaluate"
    )
    
    # Benchmark selection
    parser.add_argument(
        "--benchmarks",
        nargs="+", 
        choices=["mirage", "med_qa", "pub_med_qa", "all"],
        default=["mirage"],
        help="Benchmarks to run"
    )
    
    # Evaluation modes
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation with subset of questions"
    )
    
    parser.add_argument(
        "--full",
        action="store_true", 
        help="Full evaluation with all questions"
    )
    
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate per benchmark"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to evaluation config file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/results",
        help="Output directory for results"
    )
    
    # Enhanced options
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to use for evaluation"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    return parser.parse_args()


def load_evaluation_config(config_path: Optional[str], device: str) -> Dict:
    """Load and validate evaluation configuration."""
    if config_path:
        # Use specified config
        config_file = Path(config_path)
    else:
        # Auto-detect best config
        config_dir = Path("src/evaluation")
        
        if device == "cuda" or (device == "auto" and is_cuda_available()):
            gpu_config = config_dir / "gpu_runpod_config.yaml"
            if gpu_config.exists():
                config_file = gpu_config
                logger.info("🎮 Using GPU configuration")
            else:
                config_file = config_dir / "config.yaml"
                logger.info("💻 Using default configuration (GPU config not found)")
        else:
            config_file = config_dir / "config.yaml"
            logger.info("💻 Using default configuration")
    
    if not config_file.exists():
        logger.error(f"❌ Configuration file not found: {config_file}")
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✅ Loaded configuration from {config_file}")
        return config
        
    except Exception as e:
        logger.error(f"❌ Failed to load config from {config_file}: {e}")
        raise


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def initialize_benchmarks(
    benchmark_names: List[str],
    config: Dict,
    max_questions: Optional[int] = None
) -> Dict[str, object]:
    """Initialize evaluation benchmarks."""
    benchmarks = {}
    
    if "mirage" in benchmark_names or "all" in benchmark_names:
        try:
            mirage = MIRAGEBenchmark(config)
            benchmarks["mirage"] = mirage
            logger.info("✅ MIRAGE benchmark initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MIRAGE: {e}")
    
    if "med_qa" in benchmark_names or "all" in benchmark_names:
        try:
            med_qa = MedQABenchmark(config)
            benchmarks["med_qa"] = med_qa
            logger.info("✅ MedQA benchmark initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize MedQA: {e}")
    
    if "pub_med_qa" in benchmark_names or "all" in benchmark_names:
        try:
            pub_med_qa = PubMedQABenchmark(config)
            benchmarks["pub_med_qa"] = pub_med_qa
            logger.info("✅ PubMedQA benchmark initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize PubMedQA: {e}")
    
    return benchmarks


def initialize_evaluators(
    model_names: List[str],
    config: Dict
) -> Dict[str, object]:
    """Initialize model evaluators."""
    evaluators = {}
    
    if "hierarchical_system" in model_names or "all" in model_names:
        try:
            hierarchical_config = config.copy()
            hierarchical_config["config_path"] = "src/basic_reasoning/config.yaml"
            
            hierarchical_evaluator = HierarchicalEvaluator(hierarchical_config)
            hierarchical_evaluator.setup_model()
            evaluators["hierarchical_system"] = hierarchical_evaluator
            logger.info("✅ Hierarchical system evaluator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize hierarchical evaluator: {e}")
            raise
    
    if "kg_system" in model_names or "all" in model_names:
        try:
            kg_config = config.copy()
            kg_config["config_path"] = "src/kg/config.yaml"
            
            kg_evaluator = KGEvaluator(kg_config)
            kg_evaluator.setup_model()
            evaluators["kg_system"] = kg_evaluator
            logger.info("✅ KG system evaluator initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize KG evaluator: {e}")
    
    return evaluators


def run_single_evaluation(
    evaluator,
    benchmark,
    model_name: str,
    benchmark_name: str,
    max_questions: Optional[int] = None
) -> Dict:
    """Run evaluation for a single model-benchmark pair."""
    logger.info(f"🔍 Evaluating {model_name} on {benchmark_name}")
    
    start_time = time.time()
    
    try:
        # Get questions from benchmark
        questions = benchmark.get_questions()
        if not questions:
            logger.error(f"❌ No questions loaded from {benchmark_name}")
            return {"error": "No questions loaded"}
        
        # Limit questions if specified
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
            logger.info(f"📋 Limited to {max_questions} questions")
        
        logger.info(f"📋 Loaded {len(questions)} questions from {benchmark_name}")
        
        # Run evaluation
        results = []
        correct_count = 0
        
        for i, question in enumerate(questions):
            try:
                # Evaluate single question
                result = evaluator.evaluate_single(question)
                results.append(result)
                
                # Track accuracy
                if result.get("is_correct", False):
                    correct_count += 1
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    current_accuracy = (correct_count / (i + 1)) * 100
                    logger.info(f"   📊 Progress: {i + 1}/{len(questions)} ({current_accuracy:.1f}% accuracy)")
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate question {i}: {e}")
                results.append({
                    "question_id": question.get("id", f"q_{i}"),
                    "error": str(e),
                    "is_correct": False,
                    "accuracy": 0.0
                })
        
        # Calculate final metrics
        total_time = time.time() - start_time
        accuracy = (correct_count / len(questions)) * 100 if questions else 0
        
        # Calculate additional metrics
        qa_metrics = QAMetrics()
        detailed_metrics = qa_metrics.calculate_metrics(results, questions)
        
        evaluation_result = {
            "model": model_name,
            "benchmark": benchmark_name,
            "total_questions": len(questions),
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "total_time": total_time,
            "avg_time_per_question": total_time / len(questions) if questions else 0,
            "detailed_metrics": detailed_metrics,
            "results": results
        }
        
        logger.info(f"✅ {model_name} on {benchmark_name}: {accuracy:.1f}% accuracy ({correct_count}/{len(questions)})")
        
        return evaluation_result
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed for {model_name} on {benchmark_name}: {e}")
        return {
            "model": model_name,
            "benchmark": benchmark_name,
            "error": str(e),
            "accuracy": 0.0,
            "total_time": time.time() - start_time
        }


def save_results(results: Dict, output_dir: str, timestamp: str):
    """Save evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_path / f"evaluation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "summary": {}
    }
    
    for model_name, model_results in results.items():
        if isinstance(model_results, dict) and "benchmarks" in model_results:
            summary["summary"][model_name] = {}
            for benchmark_name, benchmark_result in model_results["benchmarks"].items():
                if "accuracy" in benchmark_result:
                    summary["summary"][model_name][benchmark_name] = {
                        "accuracy": benchmark_result["accuracy"],
                        "total_questions": benchmark_result.get("total_questions", 0),
                        "correct_answers": benchmark_result.get("correct_answers", 0)
                    }
    
    summary_file = output_path / f"evaluation_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"💾 Results saved to {results_file}")
    logger.info(f"📊 Summary saved to {summary_file}")


def print_final_summary(results: Dict):
    """Print final evaluation summary."""
    logger.info("\n" + "="*80)
    logger.info("🎯 FINAL EVALUATION SUMMARY")
    logger.info("="*80)
    
    for model_name, model_results in results.items():
        if isinstance(model_results, dict) and "benchmarks" in model_results:
            logger.info(f"\n📋 {model_name.upper()}")
            logger.info("-" * 40)
            
            total_questions = 0
            total_correct = 0
            
            for benchmark_name, benchmark_result in model_results["benchmarks"].items():
                if "accuracy" in benchmark_result:
                    accuracy = benchmark_result["accuracy"]
                    questions = benchmark_result.get("total_questions", 0)
                    correct = benchmark_result.get("correct_answers", 0)
                    
                    logger.info(f"   {benchmark_name}: {accuracy:.1f}% ({correct}/{questions})")
                    
                    total_questions += questions
                    total_correct += correct
            
            if total_questions > 0:
                overall_accuracy = (total_correct / total_questions) * 100
                logger.info(f"   📊 Overall: {overall_accuracy:.1f}% ({total_correct}/{total_questions})")
    
    logger.info("\n" + "="*80)


def main():
    """Enhanced main evaluation function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    logger.info("🚀 Starting Enhanced HierRAGMed Evaluation")
    logger.info(f"🎯 Models: {args.models}")
    logger.info(f"📊 Benchmarks: {args.benchmarks}")
    
    # Load configuration
    try:
        config = load_evaluation_config(args.config, args.device)
        
        # Set question limits
        if args.quick:
            max_questions = 20
        elif args.full:
            max_questions = None
        elif args.max_questions:
            max_questions = args.max_questions
        else:
            max_questions = 100  # Default moderate evaluation
        
        logger.info(f"📋 Question limit: {max_questions if max_questions else 'No limit'}")
        
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return 1
    
    # Initialize benchmarks
    try:
        benchmark_names = ["all"] if "all" in args.benchmarks else args.benchmarks
        benchmarks = initialize_benchmarks(benchmark_names, config, max_questions)
        
        if not benchmarks:
            logger.error("❌ No benchmarks initialized successfully")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Benchmark initialization failed: {e}")
        return 1
    
    # Initialize evaluators
    try:
        model_names = ["all"] if "all" in args.models else args.models
        evaluators = initialize_evaluators(model_names, config)
        
        if not evaluators:
            logger.error("❌ No evaluators initialized successfully")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Evaluator initialization failed: {e}")
        return 1
    
    # Run evaluations
    timestamp = str(int(time.time()))
    all_results = {}
    
    for model_name, evaluator in evaluators.items():
        logger.info(f"\n🔬 Starting evaluation for {model_name}")
        
        model_results = {
            "model": model_name,
            "benchmarks": {}
        }
        
        for benchmark_name, benchmark in benchmarks.items():
            result = run_single_evaluation(
                evaluator, benchmark, model_name, benchmark_name, max_questions
            )
            model_results["benchmarks"][benchmark_name] = result
        
        all_results[model_name] = model_results
        
        # Cleanup evaluator
        try:
            evaluator.cleanup()
        except:
            pass
    
    # Save results and print summary
    save_results(all_results, args.output_dir, timestamp)
    print_final_summary(all_results)
    
    logger.info("✅ Evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())