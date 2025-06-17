"""
Combined metrics for comprehensive medical RAG evaluation.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from loguru import logger

from .base_metrics import BaseMetrics
from .qa_metrics import QAMetrics
from .retrieval_metrics import RetrievalMetrics
from .clinical_metrics import ClinicalMetrics


class CombinedMetrics(BaseMetrics):
    """Combined evaluation metrics for comprehensive medical RAG assessment."""
    
    def __init__(self, config: Dict = None):
        """Initialize combined metrics."""
        super().__init__(config)
        
        # Initialize component metrics
        self.qa_metrics = QAMetrics(config.get("qa", {}) if config else {})
        self.retrieval_metrics = RetrievalMetrics(config.get("retrieval", {}) if config else {})
        self.clinical_metrics = ClinicalMetrics(config.get("clinical", {}) if config else {})
        
        # Weights for combining metrics
        self.weights = config.get("weights", {
            "qa": 0.4,
            "retrieval": 0.2,
            "clinical": 0.4
        }) if config else {"qa": 0.4, "retrieval": 0.2, "clinical": 0.4}
        
        # Benchmark-specific configurations
        self.benchmark_configs = config.get("benchmark_configs", {}) if config else {}
    
    def calculate_comprehensive_score(self, 
                                    predictions: List[str],
                                    references: List[str],
                                    retrieved_lists: List[List[str]] = None,
                                    relevant_lists: List[List[str]] = None,
                                    contexts: Optional[List[str]] = None,
                                    benchmark_name: str = "default",
                                    **kwargs) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation score combining all metrics.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            retrieved_lists: Retrieved document lists for each query
            relevant_lists: Relevant document lists for each query  
            contexts: Context used for generation
            benchmark_name: Name of benchmark for specific weighting
        """
        
        if not self.validate_inputs(predictions, references):
            return {"error": "Invalid inputs"}
        
        comprehensive_results = {
            "benchmark": benchmark_name,
            "total_samples": len(predictions),
            "component_metrics": {},
            "combined_scores": {},
            "metadata": {}
        }
        
        # Calculate QA metrics
        logger.info("Calculating QA metrics...")
        try:
            qa_results = self.qa_metrics.calculate(predictions, references, **kwargs)
            comprehensive_results["component_metrics"]["qa"] = qa_results
        except Exception as e:
            logger.error(f"QA metrics calculation failed: {e}")
            comprehensive_results["component_metrics"]["qa"] = {"error": str(e)}
        
        # Calculate retrieval metrics (if data available)
        if retrieved_lists and relevant_lists:
            logger.info("Calculating retrieval metrics...")
            try:
                retrieval_results = self.retrieval_metrics.calculate(retrieved_lists, relevant_lists, **kwargs)
                comprehensive_results["component_metrics"]["retrieval"] = retrieval_results
            except Exception as e:
                logger.error(f"Retrieval metrics calculation failed: {e}")
                comprehensive_results["component_metrics"]["retrieval"] = {"error": str(e)}
        
        # Calculate clinical metrics
        logger.info("Calculating clinical metrics...")
        try:
            clinical_results = self.clinical_metrics.calculate(predictions, references, contexts, **kwargs)
            comprehensive_results["component_metrics"]["clinical"] = clinical_results
        except Exception as e:
            logger.error(f"Clinical metrics calculation failed: {e}")
            comprehensive_results["component_metrics"]["clinical"] = {"error": str(e)}
        
        # Calculate combined scores
        combined_scores = self._calculate_combined_scores(
            comprehensive_results["component_metrics"], 
            benchmark_name
        )
        comprehensive_results["combined_scores"] = combined_scores
        
        # Add benchmark-specific analysis
        benchmark_analysis = self._analyze_benchmark_performance(
            comprehensive_results["component_metrics"], 
            benchmark_name
        )
        comprehensive_results["benchmark_analysis"] = benchmark_analysis
        
        # Calculate confidence intervals
        if len(predictions) > 1:
            confidence_intervals = self._calculate_confidence_intervals(
                comprehensive_results["component_metrics"]
            )
            comprehensive_results["confidence_intervals"] = confidence_intervals
        
        return comprehensive_results
    
    def _calculate_combined_scores(self, component_metrics: Dict, benchmark_name: str) -> Dict[str, float]:
        """Calculate weighted combined scores."""
        
        # Get benchmark-specific weights or use defaults
        weights = self.benchmark_configs.get(benchmark_name, {}).get("weights", self.weights)
        
        combined_scores = {}
        
        # Extract key scores from each component
        qa_score = self._extract_primary_score(component_metrics.get("qa", {}), "qa")
        retrieval_score = self._extract_primary_score(component_metrics.get("retrieval", {}), "retrieval")
        clinical_score = self._extract_primary_score(component_metrics.get("clinical", {}), "clinical")
        
        # Calculate weighted averages
        if qa_score is not None and clinical_score is not None:
            if retrieval_score is not None:
                # All three components available
                overall_score = (
                    qa_score * weights["qa"] +
                    retrieval_score * weights["retrieval"] +
                    clinical_score * weights["clinical"]
                )
            else:
                # Only QA and clinical (re-normalize weights)
                total_weight = weights["qa"] + weights["clinical"]
                overall_score = (
                    qa_score * (weights["qa"] / total_weight) +
                    clinical_score * (weights["clinical"] / total_weight)
                )
        else:
            overall_score = 0.0
        
        combined_scores["overall_score"] = overall_score
        combined_scores["qa_score"] = qa_score or 0.0
        combined_scores["retrieval_score"] = retrieval_score or 0.0
        combined_scores["clinical_score"] = clinical_score or 0.0
        
        # Benchmark-specific scores
        if benchmark_name == "mirage":
            combined_scores["mirage_score"] = self._calculate_mirage_score(component_metrics)
        elif benchmark_name == "medreason":
            combined_scores["reasoning_score"] = self._calculate_reasoning_score(component_metrics)
        elif benchmark_name == "pubmedqa":
            combined_scores["research_score"] = self._calculate_research_score(component_metrics)
        elif benchmark_name == "msmarco":
            combined_scores["retrieval_effectiveness"] = retrieval_score or 0.0
        
        return combined_scores
    
    def _extract_primary_score(self, metrics: Dict, component: str) -> Optional[float]:
        """Extract primary score from component metrics."""
        if not metrics or "error" in metrics:
            return None
        
        if component == "qa":
            # Primary QA score: weighted average of key metrics
            rouge_l = metrics.get("rouge_rougeL", 0)
            semantic_sim = metrics.get("semantic_similarity", 0)
            bleu = metrics.get("bleu", 0)
            return (rouge_l * 0.4 + semantic_sim * 0.4 + bleu * 0.2) * 100
        
        elif component == "retrieval":
            # Primary retrieval score: NDCG@10
            return metrics.get("ndcg_at_10", 0) * 100
        
        elif component == "clinical":
            # Primary clinical score: weighted average of clinical metrics
            medical_acc = metrics.get("medical_accuracy", 0)
            clinical_rel = metrics.get("clinical_relevance", 0)
            safety = metrics.get("safety_score", 0)
            return (medical_acc * 0.4 + clinical_rel * 0.3 + safety * 0.3) * 100
        
        return None
    
    def _calculate_mirage_score(self, component_metrics: Dict) -> float:
        """Calculate MIRAGE-specific composite score."""
        qa_metrics = component_metrics.get("qa", {})
        clinical_metrics = component_metrics.get("clinical", {})
        
        # MIRAGE emphasizes clinical accuracy and reasoning
        clinical_reasoning = clinical_metrics.get("clinical_reasoning_quality", 0) * 100
        medical_accuracy = clinical_metrics.get("medical_accuracy", 0) * 100
        semantic_similarity = qa_metrics.get("semantic_similarity", 0) * 100
        safety_score = clinical_metrics.get("safety_score", 0) * 100
        
        mirage_score = (
            clinical_reasoning * 0.3 +
            medical_accuracy * 0.3 +
            semantic_similarity * 0.2 +
            safety_score * 0.2
        )
        
        return mirage_score
    
    def _calculate_reasoning_score(self, component_metrics: Dict) -> float:
        """Calculate reasoning-specific score for MedReason benchmark."""
        clinical_metrics = component_metrics.get("clinical", {})
        qa_metrics = component_metrics.get("qa", {})
        
        # Reasoning emphasizes logical structure and clinical thinking
        reasoning_quality = clinical_metrics.get("clinical_reasoning_quality", 0) * 100
        diagnostic_accuracy = clinical_metrics.get("diagnostic_accuracy", 0) * 100
        rouge_l = qa_metrics.get("rouge_rougeL", 0) * 100
        
        reasoning_score = (
            reasoning_quality * 0.5 +
            diagnostic_accuracy * 0.3 +
            rouge_l * 0.2
        )
        
        return reasoning_score
    
    def _calculate_research_score(self, component_metrics: Dict) -> float:
        """Calculate research-specific score for PubMedQA benchmark."""
        qa_metrics = component_metrics.get("qa", {})
        clinical_metrics = component_metrics.get("clinical", {})
        
        # Research emphasizes evidence synthesis and accuracy
        evidence_quality = clinical_metrics.get("evidence_quality", 0) * 100
        semantic_similarity = qa_metrics.get("semantic_similarity", 0) * 100
        rouge_l = qa_metrics.get("rouge_rougeL", 0) * 100
        bertscore = qa_metrics.get("bertscore_f1", 0) * 100
        
        research_score = (
            evidence_quality * 0.4 +
            semantic_similarity * 0.25 +
            rouge_l * 0.2 +
            bertscore * 0.15
        )
        
        return research_score
    
    def _analyze_benchmark_performance(self, component_metrics: Dict, benchmark_name: str) -> Dict[str, Any]:
        """Analyze performance specific to benchmark type."""
        
        analysis = {
            "benchmark_type": benchmark_name,
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        # Extract key scores
        qa_metrics = component_metrics.get("qa", {})
        clinical_metrics = component_metrics.get("clinical", {})
        retrieval_metrics = component_metrics.get("retrieval", {})
        
        # Benchmark-specific analysis
        if benchmark_name == "mirage":
            analysis.update(self._analyze_mirage_performance(qa_metrics, clinical_metrics))
        elif benchmark_name == "medreason":
            analysis.update(self._analyze_reasoning_performance(qa_metrics, clinical_metrics))
        elif benchmark_name == "pubmedqa":
            analysis.update(self._analyze_research_performance(qa_metrics, clinical_metrics))
        elif benchmark_name == "msmarco":
            analysis.update(self._analyze_retrieval_performance(retrieval_metrics))
        
        return analysis
    
    def _analyze_mirage_performance(self, qa_metrics: Dict, clinical_metrics: Dict) -> Dict:
        """Analyze MIRAGE-specific performance patterns."""
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Check clinical reasoning
        reasoning_quality = clinical_metrics.get("clinical_reasoning_quality", 0)
        if reasoning_quality > 0.7:
            strengths.append("Strong clinical reasoning capabilities")
        elif reasoning_quality < 0.5:
            weaknesses.append("Weak clinical reasoning structure")
            recommendations.append("Improve step-by-step diagnostic thinking")
        
        # Check safety awareness
        safety_score = clinical_metrics.get("safety_score", 0)
        if safety_score > 0.8:
            strengths.append("Good safety awareness")
        elif safety_score < 0.6:
            weaknesses.append("Insufficient safety considerations")
            recommendations.append("Include more safety warnings and contraindications")
        
        # Check semantic quality
        semantic_sim = qa_metrics.get("semantic_similarity", 0)
        if semantic_sim > 0.7:
            strengths.append("High semantic similarity with references")
        elif semantic_sim < 0.5:
            weaknesses.append("Poor semantic alignment")
            recommendations.append("Improve response relevance and accuracy")
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses, 
            "recommendations": recommendations
        }
    
    def _analyze_reasoning_performance(self, qa_metrics: Dict, clinical_metrics: Dict) -> Dict:
        """Analyze MedReason-specific performance patterns."""
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Check reasoning structure
        reasoning_quality = clinical_metrics.get("clinical_reasoning_quality", 0)
        if reasoning_quality > 0.8:
            strengths.append("Excellent structured clinical reasoning")
        elif reasoning_quality < 0.6:
            weaknesses.append("Poor reasoning chain structure")
            recommendations.append("Implement hierarchical diagnostic reasoning patterns")
        
        # Check diagnostic accuracy
        diagnostic_acc = clinical_metrics.get("diagnostic_accuracy", 0)
        if diagnostic_acc > 0.7:
            strengths.append("Strong diagnostic capabilities")
        elif diagnostic_acc < 0.5:
            weaknesses.append("Weak diagnostic accuracy")
            recommendations.append("Improve pattern recognition and differential diagnosis")
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations
        }
    
    def _analyze_research_performance(self, qa_metrics: Dict, clinical_metrics: Dict) -> Dict:
        """Analyze PubMedQA research performance patterns."""
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Check evidence synthesis
        evidence_quality = clinical_metrics.get("evidence_quality", 0)
        if evidence_quality > 0.7:
            strengths.append("Good evidence synthesis capabilities")
        elif evidence_quality < 0.5:
            weaknesses.append("Poor evidence integration")
            recommendations.append("Improve research literature synthesis")
        
        # Check BERTScore for semantic understanding
        bertscore = qa_metrics.get("bertscore_f1", 0)
        if bertscore > 0.8:
            strengths.append("Excellent semantic understanding")
        elif bertscore < 0.6:
            weaknesses.append("Weak semantic comprehension")
            recommendations.append("Enhance research question understanding")
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations
        }
    
    def _analyze_retrieval_performance(self, retrieval_metrics: Dict) -> Dict:
        """Analyze MS MARCO retrieval performance patterns."""
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Check NDCG performance
        ndcg_10 = retrieval_metrics.get("ndcg_at_10", 0)
        if ndcg_10 > 0.4:
            strengths.append("Strong retrieval ranking quality")
        elif ndcg_10 < 0.25:
            weaknesses.append("Poor retrieval ranking")
            recommendations.append("Improve document ranking algorithms")
        
        # Check precision performance
        precision_5 = retrieval_metrics.get("precision_at_5", 0)
        if precision_5 > 0.6:
            strengths.append("High precision in top results")
        elif precision_5 < 0.3:
            weaknesses.append("Low precision in retrieved documents")
            recommendations.append("Enhance query-document matching")
        
        # Check MAP performance
        map_score = retrieval_metrics.get("map", 0)
        if map_score > 0.3:
            strengths.append("Good overall retrieval effectiveness")
        elif map_score < 0.15:
            weaknesses.append("Poor overall retrieval performance")
            recommendations.append("Redesign retrieval architecture")
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations
        }
    
    def _calculate_confidence_intervals(self, component_metrics: Dict) -> Dict[str, Dict]:
        """Calculate confidence intervals for key metrics."""
        confidence_intervals = {}
        
        # This would require individual sample scores
        # For now, provide placeholder structure
        for component in ["qa", "clinical", "retrieval"]:
            if component in component_metrics and "error" not in component_metrics[component]:
                confidence_intervals[component] = {
                    "method": "bootstrap",
                    "confidence_level": 0.95,
                    "note": "Confidence intervals require individual sample scores"
                }
        
        return confidence_intervals
    
    def calculate_benchmark_comparison(self, 
                                     results_a: Dict, 
                                     results_b: Dict, 
                                     benchmark_name: str) -> Dict[str, Any]:
        """Compare two systems on a specific benchmark."""
        
        comparison = {
            "benchmark": benchmark_name,
            "system_a": results_a.get("model_name", "System A"),
            "system_b": results_b.get("model_name", "System B"),
            "score_comparison": {},
            "winner": None,
            "margin": 0.0,
            "significant_differences": []
        }
        
        # Extract combined scores
        scores_a = results_a.get("combined_scores", {})
        scores_b = results_b.get("combined_scores", {})
        
        # Compare overall scores
        overall_a = scores_a.get("overall_score", 0)
        overall_b = scores_b.get("overall_score", 0)
        
        comparison["score_comparison"]["overall"] = {
            "system_a": overall_a,
            "system_b": overall_b,
            "difference": overall_a - overall_b
        }
        
        # Determine winner
        if overall_a > overall_b:
            comparison["winner"] = comparison["system_a"]
            comparison["margin"] = overall_a - overall_b
        elif overall_b > overall_a:
            comparison["winner"] = comparison["system_b"]
            comparison["margin"] = overall_b - overall_a
        else:
            comparison["winner"] = "Tie"
            comparison["margin"] = 0.0
        
        # Compare component scores
        for component in ["qa_score", "clinical_score", "retrieval_score"]:
            if component in scores_a and component in scores_b:
                score_a = scores_a[component]
                score_b = scores_b[component]
                comparison["score_comparison"][component] = {
                    "system_a": score_a,
                    "system_b": score_b,
                    "difference": score_a - score_b
                }
                
                # Check for significant differences (>5% threshold)
                if abs(score_a - score_b) > 5.0:
                    comparison["significant_differences"].append({
                        "component": component,
                        "difference": abs(score_a - score_b),
                        "better_system": comparison["system_a"] if score_a > score_b else comparison["system_b"]
                    })
        
        return comparison
    
    def generate_performance_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate executive summary of performance results."""
        
        combined_scores = results.get("combined_scores", {})
        component_metrics = results.get("component_metrics", {})
        
        summary = {
            "overall_performance": "Unknown",
            "key_strengths": [],
            "improvement_areas": [],
            "grade": "Unknown",
            "benchmark_specific_notes": []
        }
        
        # Overall performance assessment
        overall_score = combined_scores.get("overall_score", 0)
        
        if overall_score >= 80:
            summary["overall_performance"] = "Excellent"
            summary["grade"] = "A"
        elif overall_score >= 70:
            summary["overall_performance"] = "Good"
            summary["grade"] = "B"
        elif overall_score >= 60:
            summary["overall_performance"] = "Satisfactory"
            summary["grade"] = "C"
        elif overall_score >= 50:
            summary["overall_performance"] = "Below Average"
            summary["grade"] = "D"
        else:
            summary["overall_performance"] = "Poor"
            summary["grade"] = "F"
        
        # Identify strengths and weaknesses
        qa_score = combined_scores.get("qa_score", 0)
        clinical_score = combined_scores.get("clinical_score", 0)
        retrieval_score = combined_scores.get("retrieval_score", 0)
        
        # Strengths (scores > 75)
        if qa_score > 75:
            summary["key_strengths"].append("Strong question answering capabilities")
        if clinical_score > 75:
            summary["key_strengths"].append("Excellent clinical knowledge and reasoning")
        if retrieval_score > 75:
            summary["key_strengths"].append("High-quality document retrieval")
        
        # Improvement areas (scores < 60)
        if qa_score < 60:
            summary["improvement_areas"].append("Question answering accuracy and fluency")
        if clinical_score < 60:
            summary["improvement_areas"].append("Clinical reasoning and medical accuracy")
        if retrieval_score < 60:
            summary["improvement_areas"].append("Document retrieval effectiveness")
        
        # Benchmark-specific notes
        benchmark = results.get("benchmark", "unknown")
        if benchmark == "mirage":
            mirage_score = combined_scores.get("mirage_score", 0)
            summary["benchmark_specific_notes"].append(f"MIRAGE composite score: {mirage_score:.1f}%")
        elif benchmark == "medreason":
            reasoning_score = combined_scores.get("reasoning_score", 0)
            summary["benchmark_specific_notes"].append(f"Clinical reasoning score: {reasoning_score:.1f}%")
        
        return summary
    
    def get_metrics_info(self) -> Dict[str, Any]:
        """Get comprehensive information about all metrics."""
        return {
            "name": "CombinedMetrics",
            "components": {
                "qa_metrics": self.qa_metrics.get_metric_info(),
                "retrieval_metrics": self.retrieval_metrics.get_metric_info(),
                "clinical_metrics": self.clinical_metrics.get_metric_info()
            },
            "weights": self.weights,
            "benchmark_configs": self.benchmark_configs,
            "description": "Comprehensive medical RAG evaluation combining QA, retrieval, and clinical metrics"
        }