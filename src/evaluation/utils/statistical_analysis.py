"""
Statistical analysis utilities for evaluation results.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from loguru import logger

try:
    from scipy import stats
    import scipy.stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available - statistical tests will be limited")


class StatisticalAnalysis:
    """Statistical analysis for evaluation results."""
    
    def __init__(self, config: Dict):
        """Initialize statistical analyzer."""
        self.config = config
        self.significance_level = 0.05
        self.confidence_level = 0.95
        
    def analyze_results(self, evaluation_results: Dict) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        logger.info("📊 Performing statistical analysis...")
        
        analysis_results = {
            "descriptive_statistics": {},
            "significance_tests": {},
            "effect_sizes": {},
            "confidence_intervals": {},
            "distribution_analysis": {},
            "correlation_analysis": {}
        }
        
        # Extract performance data
        performance_data = self._extract_performance_data(evaluation_results)
        
        # Descriptive statistics
        analysis_results["descriptive_statistics"] = self._calculate_descriptive_statistics(performance_data)
        
        # Significance testing (if SciPy available)
        if SCIPY_AVAILABLE:
            analysis_results["significance_tests"] = self._perform_significance_tests(performance_data)
            analysis_results["effect_sizes"] = self._calculate_effect_sizes(performance_data)
            analysis_results["confidence_intervals"] = self._calculate_confidence_intervals(performance_data)
            analysis_results["distribution_analysis"] = self._analyze_distributions(performance_data)
            analysis_results["correlation_analysis"] = self._analyze_correlations(performance_data)
        
        logger.info("✅ Statistical analysis completed")
        return analysis_results
    
    def _extract_performance_data(self, evaluation_results: Dict) -> Dict[str, Dict[str, List[float]]]:
        """Extract performance data organized by model and benchmark."""
        performance_data = {}
        
        results = evaluation_results.get("results", {})
        
        for benchmark_name, benchmark_results in results.items():
            for model_name, model_results in benchmark_results.items():
                if "error" in model_results:
                    continue
                
                # Initialize model data structure
                if model_name not in performance_data:
                    performance_data[model_name] = {}
                
                # Extract individual question scores
                individual_results = model_results.get("individual_results", [])
                scores = []
                
                for result in individual_results:
                    if "score" in result and "error" not in result:
                        scores.append(result["score"])
                
                if scores:
                    performance_data[model_name][benchmark_name] = scores
        
        return performance_data
    
    def _calculate_descriptive_statistics(self, performance_data: Dict) -> Dict[str, Any]:
        """Calculate descriptive statistics for each model-benchmark combination."""
        descriptive_stats = {}
        
        for model_name, model_benchmarks in performance_data.items():
            descriptive_stats[model_name] = {}
            
            for benchmark_name, scores in model_benchmarks.items():
                if not scores:
                    continue
                
                stats_dict = {
                    "count": len(scores),
                    "mean": np.mean(scores),
                    "median": np.median(scores),
                    "std": np.std(scores, ddof=1),
                    "variance": np.var(scores, ddof=1),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "range": np.max(scores) - np.min(scores),
                    "q25": np.percentile(scores, 25),
                    "q75": np.percentile(scores, 75),
                    "iqr": np.percentile(scores, 75) - np.percentile(scores, 25),
                    "skewness": self._calculate_skewness(scores),
                    "kurtosis": self._calculate_kurtosis(scores)
                }
                
                descriptive_stats[model_name][benchmark_name] = stats_dict
        
        return descriptive_stats
    
    def _perform_significance_tests(self, performance_data: Dict) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        significance_tests = {}
        
        model_names = list(performance_data.keys())
        
        # Pairwise comparisons between models
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                significance_tests[comparison_key] = {}
                
                # Compare on each benchmark
                common_benchmarks = set(performance_data[model1].keys()).intersection(
                    set(performance_data[model2].keys())
                )
                
                for benchmark in common_benchmarks:
                    scores1 = performance_data[model1][benchmark]
                    scores2 = performance_data[model2][benchmark]
                    
                    if len(scores1) > 1 and len(scores2) > 1:
                        test_results = self._perform_two_sample_tests(scores1, scores2)
                        significance_tests[comparison_key][benchmark] = test_results
        
        return significance_tests
    
    def _perform_two_sample_tests(self, sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
        """Perform two-sample statistical tests."""
        test_results = {}
        
        # T-test (assuming equal variances initially)
        try:
            t_stat, t_p_value = stats.ttest_ind(sample1, sample2)
            test_results["t_test"] = {
                "statistic": float(t_stat),
                "p_value": float(t_p_value),
                "significant": t_p_value < self.significance_level,
                "interpretation": "significant difference" if t_p_value < self.significance_level else "no significant difference"
            }
        except Exception as e:
            test_results["t_test"] = {"error": str(e)}
        
        # Welch's t-test (unequal variances)
        try:
            welch_t_stat, welch_p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
            test_results["welch_t_test"] = {
                "statistic": float(welch_t_stat),
                "p_value": float(welch_p_value),
                "significant": welch_p_value < self.significance_level
            }
        except Exception as e:
            test_results["welch_t_test"] = {"error": str(e)}
        
        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, u_p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
            test_results["mann_whitney_u"] = {
                "statistic": float(u_stat),
                "p_value": float(u_p_value),
                "significant": u_p_value < self.significance_level
            }
        except Exception as e:
            test_results["mann_whitney_u"] = {"error": str(e)}
        
        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_p_value = stats.ks_2samp(sample1, sample2)
            test_results["kolmogorov_smirnov"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_p_value),
                "significant": ks_p_value < self.significance_level
            }
        except Exception as e:
            test_results["kolmogorov_smirnov"] = {"error": str(e)}
        
        return test_results
    
    def _calculate_effect_sizes(self, performance_data: Dict) -> Dict[str, Any]:
        """Calculate effect sizes for model comparisons."""
        effect_sizes = {}
        
        model_names = list(performance_data.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                effect_sizes[comparison_key] = {}
                
                common_benchmarks = set(performance_data[model1].keys()).intersection(
                    set(performance_data[model2].keys())
                )
                
                for benchmark in common_benchmarks:
                    scores1 = performance_data[model1][benchmark]
                    scores2 = performance_data[model2][benchmark]
                    
                    if len(scores1) > 1 and len(scores2) > 1:
                        effects = self._calculate_pairwise_effect_sizes(scores1, scores2)
                        effect_sizes[comparison_key][benchmark] = effects
        
        return effect_sizes
    
    def _calculate_pairwise_effect_sizes(self, sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
        """Calculate various effect size measures."""
        effects = {}
        
        # Cohen's d
        effects["cohens_d"] = self._cohens_d(sample1, sample2)
        
        # Glass's delta
        effects["glass_delta"] = self._glass_delta(sample1, sample2)
        
        return effects
    
    def _calculate_confidence_intervals(self, performance_data: Dict) -> Dict[str, Any]:
        """Calculate confidence intervals for means."""
        confidence_intervals = {}
        
        for model_name, model_benchmarks in performance_data.items():
            confidence_intervals[model_name] = {}
            
            for benchmark_name, scores in model_benchmarks.items():
                if len(scores) > 1:
                    mean = np.mean(scores)
                    std_err = stats.sem(scores)
                    dof = len(scores) - 1
                    
                    # t-distribution confidence interval
                    t_critical = stats.t.ppf((1 + self.confidence_level) / 2, dof)
                    margin_of_error = t_critical * std_err
                    
                    confidence_intervals[model_name][benchmark_name] = {
                        "mean": float(mean),
                        "lower_bound": float(mean - margin_of_error),
                        "upper_bound": float(mean + margin_of_error),
                        "margin_of_error": float(margin_of_error),
                        "confidence_level": self.confidence_level
                    }
        
        return confidence_intervals
    
    def _analyze_distributions(self, performance_data: Dict) -> Dict[str, Any]:
        """Analyze the distributions of performance scores."""
        distribution_analysis = {}
        
        for model_name, model_benchmarks in performance_data.items():
            distribution_analysis[model_name] = {}
            
            for benchmark_name, scores in model_benchmarks.items():
                if len(scores) >= 3:  # Minimum for distribution analysis
                    try:
                        # Normality test
                        shapiro_stat, shapiro_p = stats.shapiro(scores)
                        
                        distribution_analysis[model_name][benchmark_name] = {
                            "normality_test": {
                                "shapiro_wilk_statistic": float(shapiro_stat),
                                "shapiro_wilk_p_value": float(shapiro_p),
                                "is_normal": shapiro_p > self.significance_level
                            },
                            "distribution_properties": {
                                "skewness": float(stats.skew(scores)),
                                "kurtosis": float(stats.kurtosis(scores)),
                                "jarque_bera_stat": None,
                                "jarque_bera_p": None
                            }
                        }
                        
                        # Jarque-Bera test if enough samples
                        if len(scores) >= 20:
                            jb_stat, jb_p = stats.jarque_bera(scores)
                            distribution_analysis[model_name][benchmark_name]["distribution_properties"]["jarque_bera_stat"] = float(jb_stat)
                            distribution_analysis[model_name][benchmark_name]["distribution_properties"]["jarque_bera_p"] = float(jb_p)
                    
                    except Exception as e:
                        distribution_analysis[model_name][benchmark_name] = {"error": str(e)}
        
        return distribution_analysis
    
    def _analyze_correlations(self, performance_data: Dict) -> Dict[str, Any]:
        """Analyze correlations between benchmarks within each model."""
        correlation_analysis = {}
        
        for model_name, model_benchmarks in performance_data.items():
            benchmarks = list(model_benchmarks.keys())
            
            if len(benchmarks) >= 2:
                correlation_matrix = {}
                
                for i, benchmark1 in enumerate(benchmarks):
                    correlation_matrix[benchmark1] = {}
                    
                    for benchmark2 in benchmarks:
                        if benchmark1 == benchmark2:
                            correlation_matrix[benchmark1][benchmark2] = 1.0
                        elif benchmark2 in correlation_matrix and benchmark1 in correlation_matrix[benchmark2]:
                            # Already calculated
                            correlation_matrix[benchmark1][benchmark2] = correlation_matrix[benchmark2][benchmark1]
                        else:
                            scores1 = model_benchmarks[benchmark1]
                            scores2 = model_benchmarks[benchmark2]
                            
                            # Need same number of scores for correlation
                            min_len = min(len(scores1), len(scores2))
                            if min_len >= 3:
                                try:
                                    corr_coef, corr_p = stats.pearsonr(scores1[:min_len], scores2[:min_len])
                                    correlation_matrix[benchmark1][benchmark2] = {
                                        "correlation": float(corr_coef),
                                        "p_value": float(corr_p),
                                        "significant": corr_p < self.significance_level
                                    }
                                except Exception:
                                    correlation_matrix[benchmark1][benchmark2] = {"error": "correlation_failed"}
                            else:
                                correlation_matrix[benchmark1][benchmark2] = {"error": "insufficient_data"}
                
                correlation_analysis[model_name] = correlation_matrix
        
        return correlation_analysis
    
    def _calculate_skewness(self, scores: List[float]) -> float:
        """Calculate skewness of score distribution."""
        if len(scores) < 3:
            return 0.0
        return float(stats.skew(scores))
    
    def _calculate_kurtosis(self, scores: List[float]) -> float:
        """Calculate kurtosis of score distribution."""
        if len(scores) < 4:
            return 0.0
        return float(stats.kurtosis(scores))
    
    def _cohens_d(self, sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
        """Calculate Cohen's d effect size."""
        if len(sample1) < 2 or len(sample2) < 2:
            return {"value": 0.0, "interpretation": "insufficient_data"}
        
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
        n1, n2 = len(sample1), len(sample2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return {"value": 0.0, "interpretation": "zero_variance"}
        
        cohens_d = (mean1 - mean2) / pooled_std
        interpretation = self._interpret_effect_size(abs(cohens_d))
        
        return {
            "value": float(cohens_d),
            "interpretation": interpretation,
            "magnitude": abs(float(cohens_d))
        }
    
    def _glass_delta(self, sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
        """Calculate Glass's delta effect size."""
        if len(sample1) < 2 or len(sample2) < 2:
            return {"value": 0.0, "interpretation": "insufficient_data"}
        
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std2 = np.std(sample2, ddof=1)  # Control group standard deviation
        
        if std2 == 0:
            return {"value": 0.0, "interpretation": "zero_control_variance"}
        
        glass_delta = (mean1 - mean2) / std2
        interpretation = self._interpret_effect_size(abs(glass_delta))
        
        return {
            "value": float(glass_delta),
            "interpretation": interpretation,
            "magnitude": abs(float(glass_delta))
        }
    
    def _interpret_effect_size(self, magnitude: float) -> str:
        """Interpret effect size magnitude according to Cohen's conventions."""
        if magnitude < 0.2:
            return "negligible"
        elif magnitude < 0.5:
            return "small"
        elif magnitude < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_summary_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a human-readable summary of statistical analysis."""
        report_lines = [
            "Statistical Analysis Summary",
            "=" * 40,
            ""
        ]
        
        # Descriptive statistics summary
        desc_stats = analysis_results.get("descriptive_statistics", {})
        if desc_stats:
            report_lines.append("Descriptive Statistics:")
            for model_name, model_stats in desc_stats.items():
                report_lines.append(f"\n{model_name}:")
                for benchmark, stats in model_stats.items():
                    mean = stats.get("mean", 0)
                    std = stats.get("std", 0)
                    count = stats.get("count", 0)
                    report_lines.append(f"  {benchmark}: μ={mean:.2f}, σ={std:.2f}, n={count}")
        
        # Significance tests summary
        sig_tests = analysis_results.get("significance_tests", {})
        if sig_tests:
            report_lines.append("\n\nSignificance Tests:")
            significant_count = 0
            total_count = 0
            
            for comparison, benchmarks in sig_tests.items():
                for benchmark, tests in benchmarks.items():
                    total_count += 1
                    t_test = tests.get("t_test", {})
                    if t_test.get("significant", False):
                        significant_count += 1
                        p_val = t_test.get("p_value", 1.0)
                        report_lines.append(f"  {comparison} on {benchmark}: p={p_val:.4f} (significant)")
            
            report_lines.append(f"\nTotal significant differences: {significant_count}/{total_count}")
        
        # Effect sizes summary
        effect_sizes = analysis_results.get("effect_sizes", {})
        if effect_sizes:
            report_lines.append("\n\nEffect Sizes:")
            large_effects = 0
            total_effects = 0
            
            for comparison, benchmarks in effect_sizes.items():
                for benchmark, effects in benchmarks.items():
                    total_effects += 1
                    cohens_d = effects.get("cohens_d", {})
                    interpretation = cohens_d.get("interpretation", "unknown")
                    value = cohens_d.get("value", 0)
                    
                    if interpretation == "large":
                        large_effects += 1
                        report_lines.append(f"  {comparison} on {benchmark}: d={value:.3f} (large effect)")
            
            report_lines.append(f"\nLarge effect sizes: {large_effects}/{total_effects}")
        
        return "\n".join(report_lines)