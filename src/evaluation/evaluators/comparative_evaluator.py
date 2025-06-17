"""
Comparative evaluator for medical RAG systems.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from loguru import logger


class ComparativeEvaluator:
    """Comparative evaluation between different medical RAG systems."""
    
    def __init__(self, config: Dict):
        """Initialize comparative evaluator."""
        self.config = config
        self.significance_level = 0.05
        
    def compare_systems(self, evaluation_results: Dict, significance_level: float = 0.05) -> Dict:
        """
        Compare performance between systems across all benchmarks.
        
        Args:
            evaluation_results: Dict containing evaluation results for all systems
            significance_level: Statistical significance threshold
            
        Returns:
            Dict containing comprehensive comparison analysis
        """
        self.significance_level = significance_level
        
        logger.info("🔄 Starting comprehensive system comparison...")
        
        # Extract system performance data
        system_performances = self._extract_system_performances(evaluation_results)
        
        # Perform benchmark-wise comparison
        benchmark_comparisons = self._compare_benchmarks(system_performances)
        
        # Overall performance comparison
        overall_comparison = self._compare_overall_performance(system_performances)
        
        # Statistical significance testing
        significance_tests = self._perform_significance_tests(system_performances)
        
        # Task-specific analysis
        task_analysis = self._analyze_task_performance(system_performances)
        
        # Efficiency comparison
        efficiency_comparison = self._compare_efficiency(evaluation_results)
        
        return {
            "benchmark_comparison": benchmark_comparisons,
            "overall_comparison": overall_comparison,
            "significance_tests": significance_tests,
            "task_analysis": task_analysis,
            "efficiency_comparison": efficiency_comparison,
            "summary": self._generate_comparison_summary(
                benchmark_comparisons, overall_comparison, significance_tests
            )
        }
    
    def _extract_system_performances(self, evaluation_results: Dict) -> Dict:
        """Extract performance data for each system across benchmarks."""
        system_performances = {}
        
        results = evaluation_results.get("results", {})
        
        for benchmark_name, benchmark_results in results.items():
            for system_name, system_result in benchmark_results.items():
                if "error" in system_result:
                    continue
                
                if system_name not in system_performances:
                    system_performances[system_name] = {}
                
                # Extract key metrics
                metrics = system_result.get("metrics", {})
                individual_results = system_result.get("individual_results", [])
                
                system_performances[system_name][benchmark_name] = {
                    "overall_score": metrics.get("accuracy", 0),
                    "metrics": metrics,
                    "individual_scores": [r.get("score", 0) for r in individual_results if "score" in r],
                    "total_questions": len(individual_results),
                    "successful_questions": len([r for r in individual_results if "error" not in r])
                }
        
        return system_performances
    
    def _compare_benchmarks(self, system_performances: Dict) -> Dict:
        """Compare systems on each benchmark individually."""
        benchmark_comparisons = {}
        
        # Get all benchmarks
        all_benchmarks = set()
        for system_data in system_performances.values():
            all_benchmarks.update(system_data.keys())
        
        for benchmark in all_benchmarks:
            benchmark_scores = {}
            
            for system_name, system_data in system_performances.items():
                if benchmark in system_data:
                    benchmark_scores[system_name] = system_data[benchmark]["overall_score"]
            
            if len(benchmark_scores) >= 2:
                winner = max(benchmark_scores.items(), key=lambda x: x[1])
                score_diff = self._calculate_score_differences(benchmark_scores)
                
                benchmark_comparisons[benchmark] = {
                    "scores": benchmark_scores,
                    "winner": winner[0],
                    "winning_score": winner[1],
                    "score_differences": score_diff,
                    "performance_gap": max(benchmark_scores.values()) - min(benchmark_scores.values())
                }
        
        return benchmark_comparisons
    
    def _compare_overall_performance(self, system_performances: Dict) -> Dict:
        """Compare overall performance across all benchmarks."""
        overall_scores = {}
        
        for system_name, system_data in system_performances.items():
            if system_data:
                scores = [benchmark_data["overall_score"] for benchmark_data in system_data.values()]
                overall_scores[system_name] = {
                    "average_score": np.mean(scores),
                    "median_score": np.median(scores),
                    "std_score": np.std(scores),
                    "min_score": np.min(scores),
                    "max_score": np.max(scores),
                    "benchmarks_won": 0  # Will be calculated later
                }
        
        # Calculate benchmarks won
        for benchmark_comp in self._compare_benchmarks(system_performances).values():
            winner = benchmark_comp["winner"]
            if winner in overall_scores:
                overall_scores[winner]["benchmarks_won"] += 1
        
        # Determine overall winner
        if overall_scores:
            overall_winner = max(overall_scores.items(), key=lambda x: x[1]["average_score"])
            
            return {
                "system_scores": overall_scores,
                "overall_winner": overall_winner[0],
                "winning_average": overall_winner[1]["average_score"],
                "ranking": sorted(overall_scores.items(), key=lambda x: x[1]["average_score"], reverse=True)
            }
        
        return {"error": "No valid performance data found"}
    
    def _perform_significance_tests(self, system_performances: Dict) -> Dict:
        """Perform statistical significance tests between systems."""
        significance_results = {}
        
        system_names = list(system_performances.keys())
        if len(system_names) < 2:
            return {"error": "Need at least 2 systems for comparison"}
        
        # Pairwise comparisons for each benchmark
        for benchmark in set().union(*[system_data.keys() for system_data in system_performances.values()]):
            benchmark_data = {}
            
            for system_name in system_names:
                if benchmark in system_performances[system_name]:
                    scores = system_performances[system_name][benchmark]["individual_scores"]
                    if scores:
                        benchmark_data[system_name] = scores
            
            if len(benchmark_data) >= 2:
                # Perform t-test between systems
                systems = list(benchmark_data.keys())
                if len(systems) == 2:
                    system1_scores = benchmark_data[systems[0]]
                    system2_scores = benchmark_data[systems[1]]
                    
                    if len(system1_scores) > 1 and len(system2_scores) > 1:
                        t_stat, p_value = stats.ttest_ind(system1_scores, system2_scores)
                        effect_size = self._calculate_cohens_d(system1_scores, system2_scores)
                        
                        significance_results[benchmark] = {
                            "systems_compared": systems,
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant": p_value < self.significance_level,
                            "effect_size": effect_size,
                            "effect_magnitude": self._interpret_effect_size(effect_size)
                        }
        
        return significance_results
    
    def _analyze_task_performance(self, system_performances: Dict) -> Dict:
        """Analyze performance by task type (clinical vs research)."""
        
        # Define task categories
        clinical_benchmarks = ["mirage_clinical", "medreason"]  # Clinical reasoning tasks
        research_benchmarks = ["mirage_research", "pubmedqa"]  # Research synthesis tasks
        retrieval_benchmarks = ["msmarco"]  # Retrieval quality tasks
        
        task_analysis = {}
        
        for system_name, system_data in system_performances.items():
            clinical_scores = []
            research_scores = []
            retrieval_scores = []
            
            for benchmark, benchmark_data in system_data.items():
                score = benchmark_data["overall_score"]
                
                # Categorize by task type
                if any(clinical in benchmark.lower() for clinical in ["mirage", "medreason"]):
                    if "clinical" in benchmark.lower() or benchmark.lower() == "medreason":
                        clinical_scores.append(score)
                    elif "research" in benchmark.lower() or benchmark.lower() == "pubmedqa":
                        research_scores.append(score)
                elif "msmarco" in benchmark.lower():
                    retrieval_scores.append(score)
                elif benchmark.lower() == "mirage":
                    # MIRAGE contains both clinical and research - split the score
                    clinical_scores.append(score * 0.5)  # Assume 50/50 split
                    research_scores.append(score * 0.5)
            
            task_analysis[system_name] = {
                "clinical_performance": {
                    "average_score": np.mean(clinical_scores) if clinical_scores else 0,
                    "scores": clinical_scores
                },
                "research_performance": {
                    "average_score": np.mean(research_scores) if research_scores else 0,
                    "scores": research_scores
                },
                "retrieval_performance": {
                    "average_score": np.mean(retrieval_scores) if retrieval_scores else 0,
                    "scores": retrieval_scores
                }
            }
        
        # Determine task-specific winners
        clinical_winner = max(task_analysis.items(), 
                            key=lambda x: x[1]["clinical_performance"]["average_score"])
        research_winner = max(task_analysis.items(), 
                            key=lambda x: x[1]["research_performance"]["average_score"])
        retrieval_winner = max(task_analysis.items(), 
                             key=lambda x: x[1]["retrieval_performance"]["average_score"])
        
        return {
            "system_task_performance": task_analysis,
            "clinical_winner": clinical_winner[0],
            "research_winner": research_winner[0],
            "retrieval_winner": retrieval_winner[0],
            "task_specialization": self._assess_task_specialization(task_analysis)
        }
    
    def _compare_efficiency(self, evaluation_results: Dict) -> Dict:
        """Compare computational efficiency between systems."""
        efficiency_data = {}
        
        results = evaluation_results.get("results", {})
        
        for benchmark_name, benchmark_results in results.items():
            for system_name, system_result in benchmark_results.items():
                if "error" in system_result:
                    continue
                
                if system_name not in efficiency_data:
                    efficiency_data[system_name] = {
                        "total_time": 0,
                        "total_questions": 0,
                        "benchmarks": []
                    }
                
                eval_time = system_result.get("evaluation_time", 0)
                question_count = system_result.get("total_questions", 0)
                
                efficiency_data[system_name]["total_time"] += eval_time
                efficiency_data[system_name]["total_questions"] += question_count
                efficiency_data[system_name]["benchmarks"].append(benchmark_name)
        
        # Calculate efficiency metrics
        for system_name, data in efficiency_data.items():
            if data["total_questions"] > 0:
                data["avg_time_per_question"] = data["total_time"] / data["total_questions"]
                data["questions_per_minute"] = data["total_questions"] / (data["total_time"] / 60)
            else:
                data["avg_time_per_question"] = 0
                data["questions_per_minute"] = 0
        
        # Determine efficiency winner
        if efficiency_data:
            efficiency_winner = min(efficiency_data.items(), 
                                  key=lambda x: x[1]["avg_time_per_question"])
            
            return {
                "system_efficiency": efficiency_data,
                "efficiency_winner": efficiency_winner[0],
                "fastest_time_per_question": efficiency_winner[1]["avg_time_per_question"]
            }
        
        return {"error": "No efficiency data available"}
    
    def _calculate_score_differences(self, scores: Dict[str, float]) -> Dict:
        """Calculate pairwise score differences."""
        differences = {}
        system_names = list(scores.keys())
        
        for i, sys1 in enumerate(system_names):
            for sys2 in system_names[i+1:]:
                diff = scores[sys1] - scores[sys2]
                differences[f"{sys1}_vs_{sys2}"] = diff
        
        return differences
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        if n1 <= 1 or n2 <= 1:
            return 0.0
        
        # Calculate pooled standard deviation
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def _assess_task_specialization(self, task_analysis: Dict) -> Dict:
        """Assess which systems specialize in which tasks."""
        specialization = {}
        
        for system_name, performance in task_analysis.items():
            clinical_score = performance["clinical_performance"]["average_score"]
            research_score = performance["research_performance"]["average_score"]
            retrieval_score = performance["retrieval_performance"]["average_score"]
            
            # Determine specialization based on relative performance
            max_score = max(clinical_score, research_score, retrieval_score)
            
            if max_score == clinical_score:
                specialization[system_name] = "clinical_specialist"
            elif max_score == research_score:
                specialization[system_name] = "research_specialist"
            else:
                specialization[system_name] = "retrieval_specialist"
            
            # Check if system is well-balanced (no clear specialization)
            scores = [clinical_score, research_score, retrieval_score]
            if max(scores) - min(scores) < 5:  # Within 5% difference
                specialization[system_name] = "generalist"
        
        return specialization
    
    def _generate_comparison_summary(self, benchmark_comparisons: Dict, 
                                   overall_comparison: Dict, significance_tests: Dict) -> Dict:
        """Generate executive summary of comparison results."""
        
        # Count benchmark wins
        benchmark_wins = {}
        for benchmark, comp in benchmark_comparisons.items():
            winner = comp["winner"]
            benchmark_wins[winner] = benchmark_wins.get(winner, 0) + 1
        
        # Overall winner
        overall_winner = overall_comparison.get("overall_winner", "unknown")
        
        # Significant differences count
        significant_differences = sum(1 for test in significance_tests.values() 
                                    if test.get("significant", False))
        
        # Key findings
        key_findings = []
        if overall_winner != "unknown":
            wins = benchmark_wins.get(overall_winner, 0)
            total_benchmarks = len(benchmark_comparisons)
            key_findings.append(f"{overall_winner} wins {wins}/{total_benchmarks} benchmarks")
        
        if significant_differences > 0:
            key_findings.append(f"{significant_differences} statistically significant differences found")
        
        return {
            "overall_winner": overall_winner,
            "benchmark_wins": benchmark_wins,
            "significant_differences_count": significant_differences,
            "total_benchmarks": len(benchmark_comparisons),
            "key_findings": key_findings,
            "comparison_completed": True
        }