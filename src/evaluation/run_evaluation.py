#!/usr/bin/env python3
"""
Enhanced Evaluation Script for HierRAGMed with GPU/Medical Embedding Support
File: src/evaluation/run_evaluation.py

Updated to support Microsoft BiomedNLP-PubMedBERT medical embedding and GPU environments.
"""

import argparse
import json
import time
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from loguru import logger

# Import available benchmarks only
from src.evaluation.benchmarks.mirage_benchmark import MIRAGEBenchmark
from src.evaluation.benchmarks.medreason_benchmark import MedReasonBenchmark
from src.evaluation.benchmarks.msmarco_benchmark import MSMARCOBenchmark

# Import evaluators
from src.evaluation.evaluators.hierarchical_evaluator import HierarchicalEvaluator
from src.evaluation.evaluators.kg_evaluator import KGEvaluator

# Import config for hierarchical system
from src.basic_reasoning.config import Config


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
        description="HierRAGMed Enhanced Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick evaluation on GPU
  python src/evaluation/run_evaluation.py --quick --models hierarchical_system --benchmark mirage
  
  # Full evaluation on all benchmarks
  python src/evaluation/run_evaluation.py --full --models all --benchmark all
  
  # Custom evaluation
  python src/evaluation/run_evaluation.py --max-questions 50 --models hierarchical_system
        """
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
        "--benchmark", 
        nargs="+", 
        choices=["mirage", "medreason", "msmarco", "all"],
        default=["mirage"],
        help="Benchmarks to run"
    )
    
    # Evaluation modes
    parser.add_argument("--quick", action="store_true", help="Quick evaluation (20 questions)")
    parser.add_argument("--full", action="store_true", help="Full evaluation (all questions)")
    parser.add_argument("--max-questions", type=int, help="Maximum questions per benchmark")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Custom config file path")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu", "mps"], 
                       default="auto", help="Device to use")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="evaluation/results", 
                       help="Output directory for results")
    
    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False


def is_runpod_environment() -> bool:
    """Check if running in RunPod environment."""
    return (os.path.exists("/workspace") or 
            "RUNPOD_POD_ID" in os.environ or
            "runpod" in os.environ.get("HOSTNAME", "").lower())


def load_evaluation_config(config_path: Optional[str] = None, device: str = "auto") -> Dict:
    """Load evaluation configuration with GPU support."""
    logger.info("🔧 Loading evaluation configuration...")
    
    # Determine config file priority
    config_files = []
    
    if config_path:
        config_files.append(Path(config_path))
    
    # GPU environment detection and config selection
    is_gpu = device == "cuda" or (device == "auto" and is_cuda_available())
    is_runpod = is_runpod_environment()
    
    if is_gpu:
        if is_runpod:
            logger.info("🎮 Detected RunPod GPU environment")
            config_files.extend([
                Path("src/evaluation/configs/gpu_runpod_config.yaml"),
                Path("src/evaluation/gpu_runpod_config.yaml"),
            ])
        else:
            logger.info("🖥️ Detected local GPU environment")
            config_files.extend([
                Path("src/evaluation/configs/gpu_config.yaml"),
                Path("src/evaluation/gpu_config.yaml"),
            ])
    
    # Fallback configs
    config_files.extend([
        Path("src/evaluation/config.yaml"),
        Path("config.yaml")
    ])
    
    # Try loading configs in priority order
    for config_file in config_files:
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✅ Loaded configuration from {config_file}")
                
                # Apply GPU optimizations if detected
                if is_gpu:
                    config = apply_gpu_optimizations(config, is_runpod)
                
                return config
            except Exception as e:
                logger.warning(f"⚠️ Failed to load config from {config_file}: {e}")
                continue
    
    # Fallback to default config
    logger.warning("⚠️ Using default configuration")
    return get_default_config(is_gpu, is_runpod)


def apply_gpu_optimizations(config: Dict, is_runpod: bool = False) -> Dict:
    """Apply GPU-specific optimizations to config."""
    logger.info("🚀 Applying GPU optimizations...")
    
    # Ensure models section exists
    if "models" not in config:
        config["models"] = {}
    
    # GPU-specific model settings
    gpu_settings = {
        "embedding": {
            "device": "cuda",
            "batch_size": 32 if is_runpod else 16,
            "mixed_precision": True
        },
        "llm": {
            "device": "cuda", 
            "batch_size": 16 if is_runpod else 8,
            "mixed_precision": True
        }
    }
    
    for model_type, settings in gpu_settings.items():
        if model_type not in config["models"]:
            config["models"][model_type] = {}
        config["models"][model_type].update(settings)
    
    # Set environment flags
    config["gpu_optimized"] = True
    config["environment"] = "runpod_gpu" if is_runpod else "local_gpu"
    
    return config


def get_default_config(is_gpu: bool = False, is_runpod: bool = False) -> Dict:
    """Get default configuration."""
    config = {
        "results_dir": "evaluation/results",
        "models": {
            "embedding": {
                "device": "cuda" if is_gpu else "cpu",
                "batch_size": 32 if is_gpu else 8
            },
            "llm": {
                "device": "cuda" if is_gpu else "cpu",
                "batch_size": 16 if is_gpu else 4
            }
        },
        "benchmarks": {
            "mirage": {"enabled": True},
            "medreason": {"enabled": True},
            "msmarco": {"enabled": True}
        }
    }
    
    if is_gpu:
        config = apply_gpu_optimizations(config, is_runpod)
    
    return config


def initialize_benchmarks(
    benchmark_names: List[str],
    config: Dict,
    max_questions: Optional[int] = None
) -> Dict[str, object]:
    """Initialize evaluation benchmarks."""
    logger.info("📊 Initializing benchmarks...")
    benchmarks = {}
    
    # Handle "all" selection
    if "all" in benchmark_names:
        benchmark_names = ["mirage", "medreason", "msmarco"]
    
    # Initialize MIRAGE benchmark
    if "mirage" in benchmark_names:
        try:
            mirage = MIRAGEBenchmark(config)
            benchmarks["mirage"] = mirage
            logger.info("✅ MIRAGE benchmark initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MIRAGE: {e}")
    
    # Initialize MedReason benchmark
    if "medreason" in benchmark_names:
        try:
            medreason = MedReasonBenchmark(config)
            benchmarks["medreason"] = medreason
            logger.info("✅ MedReason benchmark initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize MedReason: {e}")
    
    # Initialize MS MARCO benchmark
    if "msmarco" in benchmark_names:
        try:
            msmarco = MSMARCOBenchmark(config)
            benchmarks["msmarco"] = msmarco
            logger.info("✅ MS MARCO benchmark initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize MS MARCO: {e}")
    
    if not benchmarks:
        logger.error("❌ No benchmarks successfully initialized")
    
    return benchmarks


def initialize_evaluators(
    model_names: List[str],
    config: Dict
) -> Dict[str, object]:
    """Initialize model evaluators."""
    logger.info("🤖 Initializing evaluators...")
    evaluators = {}
    
    # Handle "all" selection
    if "all" in model_names:
        model_names = ["hierarchical_system", "kg_system"]
    
    # Initialize Hierarchical System Evaluator
    if "hierarchical_system" in model_names:
        try:
            logger.info("🔧 Initializing Hierarchical System...")
            
            # Use the same pattern as debug_gen.py
            hierarchical_config = Config()  # This handles GPU detection automatically
            
            # Create evaluator with proper config
            evaluator_config = {
                "hierarchical_config": hierarchical_config,
                "model_name": "hierarchical_system",
                "gpu_optimized": config.get("gpu_optimized", False)
            }
            
            hierarchical_evaluator = HierarchicalEvaluator(evaluator_config)
            evaluators["hierarchical_system"] = hierarchical_evaluator
            logger.info("✅ Hierarchical system evaluator initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize hierarchical evaluator: {e}")
            raise
    
    # Initialize KG System Evaluator (optional)
    if "kg_system" in model_names:
        try:
            logger.info("🔧 Initializing KG System...")
            
            kg_config = config.copy()
            kg_config["config_path"] = "src/kg/config.yaml"
            
            kg_evaluator = KGEvaluator(kg_config)
            evaluators["kg_system"] = kg_evaluator
            logger.info("✅ KG system evaluator initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize KG evaluator: {e}")
            logger.info("   Continuing without KG system...")
    
    if not evaluators:
        logger.error("❌ No evaluators successfully initialized")
    
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
        # Setup model
        logger.info(f"🔧 Setting up {model_name} model...")
        evaluator.setup_model()
        logger.info(f"✅ {model_name} model setup completed")
        
        # Get questions from benchmark
        questions = benchmark.get_questions()
        if not questions:
            logger.error(f"❌ No questions loaded from {benchmark_name}")
            return {"error": "No questions loaded", "accuracy": 0.0}
        
        # Limit questions if specified
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
            logger.info(f"📋 Limited to {max_questions} questions")
        
        logger.info(f"📋 Processing {len(questions)} questions from {benchmark_name}")
        
        # Run evaluation
        results = []
        correct_count = 0
        
        for i, question in enumerate(questions):
            try:
                # FIXED: Use correct field names from MIRAGE benchmark
                question_id = question.get("id", f"q_{i}")
                query = question.get("question", "")
                
                # FIX: Use 'correct_answer' field (not 'answer')
                correct_answer = question.get("correct_answer", "").strip().upper()
                
                # Fallback to 'answer' field for compatibility
                if not correct_answer:
                    correct_answer = question.get("answer", "").strip().upper()
                
                options = question.get("options", {})
                
                logger.info(f"\n{'='*80}")
                logger.info(f"📝 QUESTION {i+1}/{len(questions)} - ID: {question_id}")
                logger.info(f"❓ Question: {query[:100]}..." if len(query) > 100 else f"❓ Question: {query}")
                
                # Log all options
                if options:
                    logger.info(f"📋 Options:")
                    if isinstance(options, dict):
                        for key in sorted(options.keys()):
                            logger.info(f"   {key}: {options[key]}")
                    elif isinstance(options, list):
                        for j, option in enumerate(options):
                            letter = chr(65 + j)  # A, B, C, D, E
                            logger.info(f"   {letter}: {option}")
                
                logger.info(f"✅ Correct Answer: {correct_answer}")
                
                # Show progress every 10 questions (in addition to detailed logging)
                if (i + 1) % 10 == 0:
                    logger.info(f"📊 Progress: {i + 1}/{len(questions)} questions processed")
                
                # Retrieve documents
                logger.info(f"🔍 Retrieving documents...")
                retrieved_docs = evaluator.retrieve_documents(query, top_k=5)
                logger.info(f"📄 Retrieved {len(retrieved_docs)} documents")
                
                # Generate answer
                logger.info(f"🤖 Generating answer...")
                answer = evaluator.generate_answer(query, retrieved_docs)
                logger.info(f"💬 Generated Response: {answer}")
                
                # Extract predicted answer
                evaluation_result = benchmark.evaluate_response(question, answer, retrieved_docs)
                predicted_answer = evaluation_result.get("predicted_answer", "NONE")
                is_correct = evaluation_result.get("is_correct", False)

                
                # DETAILED ANSWER COMPARISON LOGGING
                logger.info(f"🎯 ANSWER ANALYSIS:")
                logger.info(f"   📍 Predicted Answer: {predicted_answer}")
                logger.info(f"   ✅ Correct Answer: {correct_answer}")
                logger.info(f"   {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")

                if is_correct:
                    correct_count += 1
                    logger.info(f"🎉 Question {i+1} answered correctly!")
                else:
                    logger.info(f"💔 Question {i+1} answered incorrectly.")
                
                # Running accuracy
                current_accuracy = (correct_count / (i + 1)) * 100
                logger.info(f"📊 Running Accuracy: {correct_count}/{i+1} ({current_accuracy:.1f}%)")
                logger.info(f"{'='*80}")
                
                results.append({
                    "question_id": question_id,
                    "question": query,
                    "options": options,
                    "answer": answer,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "evaluation": evaluation_result
                })
                
            except Exception as e:
                logger.error(f"❌ Error processing question {i + 1}: {e}")
                results.append({
                    "question_id": question.get("id", f"q_{i}"),
                    "error": str(e),
                    "is_correct": False
                })
        
        # Calculate final metrics
        total_questions = len(questions)
        accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0.0
        evaluation_time = time.time() - start_time
        
        # FINAL SUMMARY LOGGING
        logger.info(f"\n{'='*80}")
        logger.info(f"🏁 EVALUATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"📊 Model: {model_name}")
        logger.info(f"📊 Benchmark: {benchmark_name}")
        logger.info(f"📊 Total Questions: {total_questions}")
        logger.info(f"📊 Correct Answers: {correct_count}")
        logger.info(f"📊 Final Accuracy: {accuracy:.1f}%")
        logger.info(f"⏱️ Total Time: {evaluation_time:.1f} seconds")
        logger.info(f"⏱️ Average Time per Question: {evaluation_time/total_questions:.1f}s")
        logger.info(f"{'='*80}")
        
        final_result = {
            "model": model_name,
            "benchmark": benchmark_name,
            "total_questions": total_questions,
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "evaluation_time": evaluation_time,
            "results": results
        }
        
        logger.info(f"✅ {benchmark_name} evaluation completed: {accuracy:.1f}% accuracy ({correct_count}/{total_questions})")
        return final_result
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed for {model_name} on {benchmark_name}: {e}")
        return {
            "model": model_name,
            "benchmark": benchmark_name,
            "error": str(e),
            "accuracy": 0.0
        }
    
def save_results(results: Dict, output_dir: str, timestamp: str):
    """Save evaluation results to files."""
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
    logger.info(f"📊 Benchmarks: {args.benchmark}")
    
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
        benchmarks = initialize_benchmarks(args.benchmark, config, max_questions)
        
        if not benchmarks:
            logger.error("❌ No benchmarks initialized successfully")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Benchmark initialization failed: {e}")
        return 1
    
    # Initialize evaluators
    try:
        evaluators = initialize_evaluators(args.models, config)
        
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
            if hasattr(evaluator, 'cleanup'):
                evaluator.cleanup()
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning for {model_name}: {e}")
    
    # Save results and print summary
    save_results(all_results, args.output_dir, timestamp)
    print_final_summary(all_results)
    
    logger.info("✅ Evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())