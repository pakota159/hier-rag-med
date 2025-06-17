"""
Result processor for evaluation data processing and aggregation.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger


class ResultProcessor:
    """Process and aggregate evaluation results."""
    
    def __init__(self, config: Dict):
        """Initialize result processor."""
        self.config = config
        self.results_dir = Path(config.get("results_dir", "src/evaluation/results"))
        self.output_formats = config.get("output", {}).get("export_formats", ["json", "csv"])
        
    def process_results(self, raw_results: Dict) -> Dict[str, Any]:
        """Process raw evaluation results into structured format."""
        logger.info("📊 Processing evaluation results...")
        
        processed_results = {
            "metadata": raw_results.get("metadata", {}),
            "summary": {},
            "detailed_results": {},
            "benchmark_analysis": {},
            "model_comparison": {},
            "performance_trends": {},
            "export_info": {}
        }
        
        # Process results by benchmark
        raw_benchmark_results = raw_results.get("results", {})
        
        for benchmark_name, benchmark_results in raw_benchmark_results.items():
            processed_benchmark = self._process_benchmark_results(
                benchmark_name, benchmark_results
            )
            processed_results["detailed_results"][benchmark_name] = processed_benchmark
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(processed_results["detailed_results"])
        processed_results["summary"] = summary_stats
        
        # Benchmark analysis
        benchmark_analysis = self._analyze_benchmark_performance(processed_results["detailed_results"])
        processed_results["benchmark_analysis"] = benchmark_analysis
        
        # Model comparison
        if len(raw_benchmark_results) > 0:
            model_comparison = self._compare_models(processed_results["detailed_results"])
            processed_results["model_comparison"] = model_comparison
        
        # Save processed results
        self._save_processed_results(processed_results)
        
        logger.info("✅ Results processing completed")
        return processed_results
    
    def _process_benchmark_results(self, benchmark_name: str, benchmark_results: Dict) -> Dict:
        """Process results for a specific benchmark."""
        processed_benchmark = {
            "benchmark_name": benchmark_name,
            "models": {},
            "statistics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        for model_name, model_results in benchmark_results.items():
            if "error" in model_results:
                processed_benchmark["models"][model_name] = {
                    "status": "failed",
                    "error": model_results["error"]
                }
                continue
            
            # Process individual model results
            processed_model = self._process_model_results(model_name, model_results)
            processed_benchmark["models"][model_name] = processed_model
        
        # Calculate benchmark statistics
        benchmark_stats = self._calculate_benchmark_statistics(processed_benchmark["models"])
        processed_benchmark["statistics"] = benchmark_stats
        
        return processed_benchmark
    
    def _process_model_results(self, model_name: str, model_results: Dict) -> Dict:
        """Process results for a specific model."""
        processed_model = {
            "model_name": model_name,
            "status": model_results.get("status", "completed"),
            "performance_metrics": {},
            "efficiency_metrics": {},
            "error_analysis": {},
            "sample_analysis": {}
        }
        
        # Extract performance metrics
        metrics = model_results.get("metrics", {})
        processed_model["performance_metrics"] = {
            "overall_accuracy": metrics.get("accuracy", 0),
            "average_score": metrics.get("average_score", 0),
            "total_questions": metrics.get("total_questions", 0),
            "correct_answers": metrics.get("correct_answers", 0)
        }
        
        # Extract efficiency metrics
        processed_model["efficiency_metrics"] = {
            "evaluation_time": model_results.get("evaluation_time", 0),
            "questions_per_minute": self._calculate_questions_per_minute(
                model_results.get("total_questions", 0),
                model_results.get("evaluation_time", 1)
            ),
            "avg_time_per_question": self._calculate_avg_time_per_question(
                model_results.get("evaluation_time", 0),
                model_results.get("total_questions", 1)
            )
        }
        
        # Analyze individual results for error patterns
        individual_results = model_results.get("individual_results", [])
        if individual_results:
            error_analysis = self._analyze_errors(individual_results)
            processed_model["error_analysis"] = error_analysis
            
            sample_analysis = self._analyze_sample_performance(individual_results)
            processed_model["sample_analysis"] = sample_analysis
        
        return processed_model
    
    def _generate_summary_statistics(self, detailed_results: Dict) -> Dict:
        """Generate overall summary statistics."""
        summary = {
            "total_benchmarks": len(detailed_results),
            "total_models": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "average_performance": {},
            "best_performers": {},
            "evaluation_coverage": {}
        }
        
        all_models = set()
        benchmark_performances = {}
        
        for benchmark_name, benchmark_data in detailed_results.items():
            models = benchmark_data.get("models", {})
            all_models.update(models.keys())
            
            # Collect performance data
            for model_name, model_data in models.items():
                if model_data.get("status") == "completed":
                    summary["successful_evaluations"] += 1
                    
                    # Store performance for averaging
                    if model_name not in benchmark_performances:
                        benchmark_performances[model_name] = []
                    
                    accuracy = model_data.get("performance_metrics", {}).get("overall_accuracy", 0)
                    benchmark_performances[model_name].append(accuracy)
                else:
                    summary["failed_evaluations"] += 1
        
        summary["total_models"] = len(all_models)
        
        # Calculate average performance per model
        for model_name, accuracies in benchmark_performances.items():
            if accuracies:
                summary["average_performance"][model_name] = {
                    "mean_accuracy": np.mean(accuracies),
                    "std_accuracy": np.std(accuracies),
                    "benchmarks_completed": len(accuracies)
                }
        
        # Identify best performers
        if summary["average_performance"]:
            best_overall = max(
                summary["average_performance"].items(),
                key=lambda x: x[1]["mean_accuracy"]
            )
            summary["best_performers"]["overall"] = {
                "model": best_overall[0],
                "accuracy": best_overall[1]["mean_accuracy"]
            }
        
        return summary
    
    def _analyze_benchmark_performance(self, detailed_results: Dict) -> Dict:
        """Analyze performance patterns across benchmarks."""
        analysis = {}
        
        for benchmark_name, benchmark_data in detailed_results.items():
            models = benchmark_data.get("models", {})
            
            # Calculate benchmark difficulty (inverse of average performance)
            performances = []
            for model_data in models.values():
                if model_data.get("status") == "completed":
                    accuracy = model_data.get("performance_metrics", {}).get("overall_accuracy", 0)
                    performances.append(accuracy)
            
            if performances:
                benchmark_analysis = {
                    "average_performance": np.mean(performances),
                    "performance_std": np.std(performances),
                    "difficulty_score": 100 - np.mean(performances),  # Higher = more difficult
                    "model_count": len(performances),
                    "performance_range": {
                        "min": np.min(performances),
                        "max": np.max(performances)
                    }
                }
                
                # Categorize difficulty
                avg_perf = benchmark_analysis["average_performance"]
                if avg_perf >= 80:
                    benchmark_analysis["difficulty_category"] = "Easy"
                elif avg_perf >= 65:
                    benchmark_analysis["difficulty_category"] = "Medium"
                elif avg_perf >= 50:
                    benchmark_analysis["difficulty_category"] = "Hard"
                else:
                    benchmark_analysis["difficulty_category"] = "Very Hard"
                
                analysis[benchmark_name] = benchmark_analysis
        
        return analysis
    
    def _compare_models(self, detailed_results: Dict) -> Dict:
        """Compare model performance across benchmarks."""
        model_comparison = {
            "head_to_head": {},
            "rankings": {},
            "strengths_weaknesses": {}
        }
        
        # Collect all model performances by benchmark
        model_performances = {}
        
        for benchmark_name, benchmark_data in detailed_results.items():
            models = benchmark_data.get("models", {})
            
            for model_name, model_data in models.items():
                if model_data.get("status") == "completed":
                    if model_name not in model_performances:
                        model_performances[model_name] = {}
                    
                    accuracy = model_data.get("performance_metrics", {}).get("overall_accuracy", 0)
                    model_performances[model_name][benchmark_name] = accuracy
        
        # Generate head-to-head comparisons
        model_names = list(model_performances.keys())
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison = self._compare_two_models(
                    model1, model2, model_performances
                )
                model_comparison["head_to_head"][f"{model1}_vs_{model2}"] = comparison
        
        # Generate overall rankings
        rankings = self._generate_model_rankings(model_performances)
        model_comparison["rankings"] = rankings
        
        # Analyze strengths and weaknesses
        strengths_weaknesses = self._analyze_model_strengths_weaknesses(model_performances)
        model_comparison["strengths_weaknesses"] = strengths_weaknesses
        
        return model_comparison
    
    def _analyze_errors(self, individual_results: List[Dict]) -> Dict:
        """Analyze error patterns in individual results."""
        error_analysis = {
            "total_questions": len(individual_results),
            "failed_questions": 0,
            "error_types": {},
            "performance_distribution": {},
            "low_scoring_questions": []
        }
        
        scores = []
        failed_count = 0
        
        for result in individual_results:
            if "error" in result:
                failed_count += 1
                error_type = result.get("error", "unknown_error")
                error_analysis["error_types"][error_type] = error_analysis["error_types"].get(error_type, 0) + 1
            else:
                score = result.get("score", 0)
                scores.append(score)
                
                # Collect low-scoring questions for analysis
                if score < 50:  # Threshold for low performance
                    error_analysis["low_scoring_questions"].append({
                        "question_id": result.get("question_id"),
                        "score": score,
                        "response": result.get("response", "")[:100] + "..." if len(result.get("response", "")) > 100 else result.get("response", "")
                    })
        
        error_analysis["failed_questions"] = failed_count
        
        if scores:
            error_analysis["performance_distribution"] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "median": np.median(scores),
                "q25": np.percentile(scores, 25),
                "q75": np.percentile(scores, 75),
                "min": np.min(scores),
                "max": np.max(scores)
            }
        
        return error_analysis
    
    def _analyze_sample_performance(self, individual_results: List[Dict]) -> Dict:
        """Analyze performance patterns in sample data."""
        sample_analysis = {
            "score_ranges": {
                "excellent": 0,  # 90-100
                "good": 0,       # 70-89
                "fair": 0,       # 50-69
                "poor": 0        # 0-49
            },
            "question_types": {},
            "response_lengths": []
        }
        
        for result in individual_results:
            if "error" not in result:
                score = result.get("score", 0)
                
                # Categorize by score range
                if score >= 90:
                    sample_analysis["score_ranges"]["excellent"] += 1
                elif score >= 70:
                    sample_analysis["score_ranges"]["good"] += 1
                elif score >= 50:
                    sample_analysis["score_ranges"]["fair"] += 1
                else:
                    sample_analysis["score_ranges"]["poor"] += 1
                
                # Analyze question types if available
                question_type = result.get("question_type", "unknown")
                if question_type not in sample_analysis["question_types"]:
                    sample_analysis["question_types"][question_type] = {"count": 0, "avg_score": 0, "scores": []}
                
                sample_analysis["question_types"][question_type]["count"] += 1
                sample_analysis["question_types"][question_type]["scores"].append(score)
                
                # Collect response lengths
                response = result.get("response", "")
                sample_analysis["response_lengths"].append(len(response.split()))
        
        # Calculate average scores by question type
        for question_type, data in sample_analysis["question_types"].items():
            if data["scores"]:
                data["avg_score"] = np.mean(data["scores"])
        
        return sample_analysis
    
    def _compare_two_models(self, model1: str, model2: str, performances: Dict) -> Dict:
        """Compare two models head-to-head."""
        comparison = {
            "model1": model1,
            "model2": model2,
            "benchmarks_compared": 0,
            "model1_wins": 0,
            "model2_wins": 0,
            "ties": 0,
            "average_difference": 0,
            "benchmark_details": {}
        }
        
        differences = []
        
        # Compare on common benchmarks
        common_benchmarks = set(performances[model1].keys()).intersection(
            set(performances[model2].keys())
        )
        
        for benchmark in common_benchmarks:
            score1 = performances[model1][benchmark]
            score2 = performances[model2][benchmark]
            difference = score1 - score2
            differences.append(difference)
            
            comparison["benchmarks_compared"] += 1
            comparison["benchmark_details"][benchmark] = {
                "model1_score": score1,
                "model2_score": score2,
                "difference": difference
            }
            
            if difference > 1:  # Model 1 wins with >1% margin
                comparison["model1_wins"] += 1
            elif difference < -1:  # Model 2 wins with >1% margin
                comparison["model2_wins"] += 1
            else:  # Tie (within 1%)
                comparison["ties"] += 1
        
        if differences:
            comparison["average_difference"] = np.mean(differences)
            comparison["winner"] = model1 if comparison["average_difference"] > 0 else model2
        
        return comparison
    
    def _generate_model_rankings(self, performances: Dict) -> Dict:
        """Generate overall model rankings."""
        rankings = {
            "overall": [],
            "by_benchmark": {}
        }
        
        # Overall ranking by average performance
        overall_scores = {}
        for model_name, benchmark_scores in performances.items():
            if benchmark_scores:
                overall_scores[model_name] = np.mean(list(benchmark_scores.values()))
        
        # Sort by overall score
        sorted_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (model, score) in enumerate(sorted_models, 1):
            rankings["overall"].append({
                "rank": rank,
                "model": model,
                "average_score": score
            })
        
        # Ranking by individual benchmark
        all_benchmarks = set()
        for benchmark_scores in performances.values():
            all_benchmarks.update(benchmark_scores.keys())
        
        for benchmark in all_benchmarks:
            benchmark_scores = {}
            for model_name, model_benchmarks in performances.items():
                if benchmark in model_benchmarks:
                    benchmark_scores[model_name] = model_benchmarks[benchmark]
            
            sorted_benchmark = sorted(benchmark_scores.items(), key=lambda x: x[1], reverse=True)
            rankings["by_benchmark"][benchmark] = [
                {"rank": rank, "model": model, "score": score}
                for rank, (model, score) in enumerate(sorted_benchmark, 1)
            ]
        
        return rankings
    
    def _analyze_model_strengths_weaknesses(self, performances: Dict) -> Dict:
        """Analyze strengths and weaknesses of each model."""
        analysis = {}
        
        for model_name, benchmark_scores in performances.items():
            if not benchmark_scores:
                continue
            
            scores = list(benchmark_scores.values())
            avg_score = np.mean(scores)
            
            model_analysis = {
                "average_performance": avg_score,
                "strongest_benchmarks": [],
                "weakest_benchmarks": [],
                "consistency": np.std(scores),
                "performance_category": self._categorize_performance(avg_score)
            }
            
            # Find strongest and weakest benchmarks
            sorted_benchmarks = sorted(benchmark_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Top performing benchmarks
            model_analysis["strongest_benchmarks"] = [
                {"benchmark": bench, "score": score}
                for bench, score in sorted_benchmarks[:2]
            ]
            
            # Bottom performing benchmarks
            model_analysis["weakest_benchmarks"] = [
                {"benchmark": bench, "score": score}
                for bench, score in sorted_benchmarks[-2:]
            ]
            
            analysis[model_name] = model_analysis
        
        return analysis
    
    def _categorize_performance(self, score: float) -> str:
        """Categorize performance level."""
        if score >= 80:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Satisfactory"
        elif score >= 50:
            return "Below Average"
        else:
            return "Poor"
    
    def _calculate_questions_per_minute(self, total_questions: int, total_time: float) -> float:
        """Calculate questions processed per minute."""
        if total_time <= 0:
            return 0.0
        return (total_questions / total_time) * 60
    
    def _calculate_avg_time_per_question(self, total_time: float, total_questions: int) -> float:
        """Calculate average time per question."""
        if total_questions <= 0:
            return 0.0
        return total_time / total_questions
    
    def _save_processed_results(self, processed_results: Dict) -> None:
        """Save processed results in multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON format
        if "json" in self.output_formats:
            json_file = self.results_dir / f"processed_results_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(processed_results, f, indent=2, default=str)
            logger.info(f"📄 Saved JSON results: {json_file}")
        
        # Save CSV summaries
        if "csv" in self.output_formats:
            self._save_csv_summaries(processed_results, timestamp)
        
        # Save latest results (overwrite)
        latest_file = self.results_dir / "latest_processed_results.json"
        with open(latest_file, 'w') as f:
            json.dump(processed_results, f, indent=2, default=str)
        
        processed_results["export_info"] = {
            "timestamp": timestamp,
            "formats": self.output_formats,
            "location": str(self.results_dir)
        }
    
    def _save_csv_summaries(self, processed_results: Dict, timestamp: str) -> None:
        """Save CSV summaries of key results."""
        
        # Summary statistics CSV
        summary_data = []
        for benchmark_name, benchmark_data in processed_results["detailed_results"].items():
            for model_name, model_data in benchmark_data.get("models", {}).items():
                if model_data.get("status") == "completed":
                    summary_data.append({
                        "benchmark": benchmark_name,
                        "model": model_name,
                        "accuracy": model_data.get("performance_metrics", {}).get("overall_accuracy", 0),
                        "avg_score": model_data.get("performance_metrics", {}).get("average_score", 0),
                        "total_questions": model_data.get("performance_metrics", {}).get("total_questions", 0),
                        "evaluation_time": model_data.get("efficiency_metrics", {}).get("evaluation_time", 0),
                        "questions_per_minute": model_data.get("efficiency_metrics", {}).get("questions_per_minute", 0)
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_file = self.results_dir / f"summary_results_{timestamp}.csv"
            summary_df.to_csv(csv_file, index=False)
            logger.info(f"📊 Saved CSV summary: {csv_file}")
    
    def load_results(self, results_file: Optional[Path] = None) -> Dict:
        """Load processed results from file."""
        if results_file is None:
            results_file = self.results_dir / "latest_processed_results.json"
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(f"📂 Loaded results from {results_file}")
            return results
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return {}