"""
Enhanced Result Processor for HierRAGMed Evaluation
File: src/evaluation/utils/result_processor.py

Processes and analyzes evaluation results with medical-specific insights.
"""

import json
from typing import Dict, List, Any
from pathlib import Path
import numpy as np
from loguru import logger


class ResultProcessor:
    """Enhanced result processor with medical evaluation insights."""

    def __init__(self):
        """Initialize result processor."""
        pass

    def process_results(self, raw_results: Dict) -> Dict:
        """Process raw evaluation results with enhanced analysis."""
        try:
            processed_results = {}
            
            for model_name, model_data in raw_results.items():
                if isinstance(model_data, dict) and "benchmarks" in model_data:
                    processed_model = self._process_model_results(model_name, model_data)
                    processed_results[model_name] = processed_model
                else:
                    processed_results[model_name] = model_data
            
            # Add cross-model analysis
            processed_results["analysis"] = self._analyze_cross_model_performance(processed_results)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Result processing failed: {e}")
            return raw_results

    def _process_model_results(self, model_name: str, model_data: Dict) -> Dict:
        """Process results for a single model."""
        processed_model = model_data.copy()
        
        # Process each benchmark
        for benchmark_name, benchmark_data in model_data.get("benchmarks", {}).items():
            if "results" in benchmark_data:
                processed_benchmark = self._process_benchmark_results(
                    benchmark_name, benchmark_data
                )
                processed_model["benchmarks"][benchmark_name] = processed_benchmark
        
        # Add model-level analysis
        processed_model["model_analysis"] = self._analyze_model_performance(
            model_name, processed_model
        )
        
        return processed_model

    def _process_benchmark_results(self, benchmark_name: str, benchmark_data: Dict) -> Dict:
        """Process results for a single benchmark."""
        processed_benchmark = benchmark_data.copy()
        
        if "results" in benchmark_data:
            results = benchmark_data["results"]
            
            # Calculate enhanced metrics
            enhanced_metrics = self._calculate_enhanced_metrics(results)
            processed_benchmark["enhanced_metrics"] = enhanced_metrics
            
            # Analyze error patterns
            error_analysis = self._analyze_error_patterns(results)
            processed_benchmark["error_analysis"] = error_analysis
            
            # Medical-specific analysis
            if benchmark_name in ["mirage", "med_qa", "pub_med_qa"]:
                medical_analysis = self._analyze_medical_performance(results)
                processed_benchmark["medical_analysis"] = medical_analysis
        
        return processed_benchmark

    def _calculate_enhanced_metrics(self, results: List[Dict]) -> Dict:
        """Calculate enhanced evaluation metrics."""
        if not results:
            return {}
        
        # Basic metrics
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.get("is_correct", False))
        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        # Response time metrics
        response_times = [
            r.get("total_time", 0) for r in results 
            if "total_time" in r and isinstance(r["total_time"], (int, float))
        ]
        
        time_metrics = {}
        if response_times:
            time_metrics = {
                "avg_response_time": np.mean(response_times),
                "median_response_time": np.median(response_times),
                "min_response_time": np.min(response_times),
                "max_response_time": np.max(response_times)
            }
        
        # Answer extraction metrics
        extraction_stats = self._analyze_answer_extraction(results)
        
        return {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "accuracy": accuracy,
            "time_metrics": time_metrics,
            "extraction_stats": extraction_stats
        }

    def _analyze_error_patterns(self, results: List[Dict]) -> Dict:
        """Analyze patterns in incorrect answers."""
        incorrect_results = [r for r in results if not r.get("is_correct", True)]
        
        if not incorrect_results:
            return {"error_count": 0, "error_types": {}}
        
        error_types = {
            "no_answer_extracted": 0,
            "wrong_answer": 0,
            "generation_error": 0,
            "retrieval_error": 0
        }
        
        for result in incorrect_results:
            if "error" in result:
                error_types["generation_error"] += 1
            elif not result.get("extracted_answer"):
                error_types["no_answer_extracted"] += 1
            else:
                error_types["wrong_answer"] += 1
        
        return {
            "error_count": len(incorrect_results),
            "error_types": error_types,
            "error_rate": len(incorrect_results) / len(results) * 100
        }

    def _analyze_answer_extraction(self, results: List[Dict]) -> Dict:
        """Analyze answer extraction performance."""
        total_results = len(results)
        extracted_count = sum(1 for r in results if r.get("extracted_answer"))
        
        return {
            "total_responses": total_results,
            "answers_extracted": extracted_count,
            "extraction_rate": (extracted_count / total_results * 100) if total_results > 0 else 0,
            "extraction_failures": total_results - extracted_count
        }

    def _analyze_medical_performance(self, results: List[Dict]) -> Dict:
        """Analyze medical-specific performance metrics."""
        # Medical specialty analysis
        specialty_performance = {}
        
        for result in results:
            specialty = result.get("medical_specialty", "unknown")
            if specialty not in specialty_performance:
                specialty_performance[specialty] = {"total": 0, "correct": 0}
            
            specialty_performance[specialty]["total"] += 1
            if result.get("is_correct", False):
                specialty_performance[specialty]["correct"] += 1
        
        # Calculate accuracy per specialty
        for specialty, stats in specialty_performance.items():
            if stats["total"] > 0:
                stats["accuracy"] = (stats["correct"] / stats["total"]) * 100
        
        # Question type analysis
        question_type_performance = {}
        for result in results:
            q_type = result.get("question_type", "unknown")
            if q_type not in question_type_performance:
                question_type_performance[q_type] = {"total": 0, "correct": 0}
            
            question_type_performance[q_type]["total"] += 1
            if result.get("is_correct", False):
                question_type_performance[q_type]["correct"] += 1
        
        # Calculate accuracy per question type
        for q_type, stats in question_type_performance.items():
            if stats["total"] > 0:
                stats["accuracy"] = (stats["correct"] / stats["total"]) * 100
        
        return {
            "specialty_performance": specialty_performance,
            "question_type_performance": question_type_performance,
            "medical_metrics": {
                "total_medical_questions": len(results),
                "avg_medical_accuracy": np.mean([r.get("accuracy", 0) for r in results])
            }
        }

    def _analyze_model_performance(self, model_name: str, model_data: Dict) -> Dict:
        """Analyze overall model performance."""
        benchmarks = model_data.get("benchmarks", {})
        
        # Aggregate metrics across benchmarks
        total_questions = sum(
            b.get("total_questions", 0) for b in benchmarks.values()
        )
        total_correct = sum(
            b.get("correct_answers", 0) for b in benchmarks.values()
        )
        
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        
        # Performance by benchmark
        benchmark_accuracies = {
            name: data.get("accuracy", 0) 
            for name, data in benchmarks.items()
            if "accuracy" in data
        }
        
        # Medical embedding analysis
        config = model_data.get("config", {})
        medical_optimized = config.get("medical_embedding", False)
        embedding_model = config.get("embedding_model", "unknown")
        
        return {
            "overall_accuracy": overall_accuracy,
            "total_questions": total_questions,
            "total_correct": total_correct,
            "benchmark_accuracies": benchmark_accuracies,
            "medical_optimized": medical_optimized,
            "embedding_model": embedding_model,
            "performance_tier": self._classify_performance_tier(overall_accuracy)
        }

    def _classify_performance_tier(self, accuracy: float) -> str:
        """Classify performance into tiers."""
        if accuracy >= 80:
            return "excellent"
        elif accuracy >= 70:
            return "good"
        elif accuracy >= 60:
            return "moderate"
        elif accuracy >= 50:
            return "poor"
        else:
            return "critical"

    def _analyze_cross_model_performance(self, processed_results: Dict) -> Dict:
        """Analyze performance across different models."""
        models = [k for k in processed_results.keys() if k != "analysis"]
        
        if not models:
            return {}
        
        # Compare overall accuracies
        model_accuracies = {}
        for model in models:
            model_data = processed_results.get(model, {})
            if "model_analysis" in model_data:
                accuracy = model_data["model_analysis"].get("overall_accuracy", 0)
                model_accuracies[model] = accuracy
        
        # Find best and worst performing models
        best_model = max(model_accuracies, key=model_accuracies.get) if model_accuracies else None
        worst_model = min(model_accuracies, key=model_accuracies.get) if model_accuracies else None
        
        # Medical embedding comparison
        medical_embedding_comparison = {}
        for model in models:
            model_data = processed_results.get(model, {})
            config = model_data.get("config", {})
            medical_optimized = config.get("medical_embedding", False)
            accuracy = model_accuracies.get(model, 0)
            
            medical_embedding_comparison[model] = {
                "medical_optimized": medical_optimized,
                "accuracy": accuracy
            }
        
        return {
            "model_count": len(models),
            "model_accuracies": model_accuracies,
            "best_model": best_model,
            "worst_model": worst_model,
            "accuracy_range": {
                "min": min(model_accuracies.values()) if model_accuracies else 0,
                "max": max(model_accuracies.values()) if model_accuracies else 0,
                "avg": np.mean(list(model_accuracies.values())) if model_accuracies else 0
            },
            "medical_embedding_comparison": medical_embedding_comparison
        }

    def export_csv_summary(self, processed_results: Dict, output_path: Path):
        """Export results summary to CSV format."""
        try:
            import pandas as pd
            
            # Prepare data for CSV
            csv_data = []
            
            for model_name, model_data in processed_results.items():
                if model_name == "analysis":
                    continue
                
                if "benchmarks" in model_data:
                    for benchmark_name, benchmark_data in model_data["benchmarks"].items():
                        row = {
                            "model": model_name,
                            "benchmark": benchmark_name,
                            "accuracy": benchmark_data.get("accuracy", 0),
                            "total_questions": benchmark_data.get("total_questions", 0),
                            "correct_answers": benchmark_data.get("correct_answers", 0),
                            "medical_embedding": model_data.get("config", {}).get("medical_embedding", False),
                            "embedding_model": model_data.get("config", {}).get("embedding_model", "unknown")
                        }
                        csv_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            csv_path = output_path / "evaluation_summary.csv"
            df.to_csv(csv_path, index=False)
            
            logger.info(f"📊 CSV summary exported to {csv_path}")
            
        except ImportError:
            logger.warning("⚠️ pandas not available, skipping CSV export")
        except Exception as e:
            logger.error(f"❌ CSV export failed: {e}")