"""
Data validator for HierRAGMed evaluation system.
Validates benchmark data quality and consistency.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from loguru import logger
import pandas as pd
from collections import defaultdict, Counter


class DataValidator:
    """Validate benchmark data quality and consistency."""
    
    def __init__(self, config: Dict):
        """Initialize data validator."""
        self.config = config
        self.validation_config = config.get("data_validation", {})
        
        # Validation thresholds
        self.min_question_length = self.validation_config.get("min_question_length", 10)
        self.max_question_length = self.validation_config.get("max_question_length", 1000)
        self.min_context_length = self.validation_config.get("min_context_length", 20)
        self.max_context_length = self.validation_config.get("max_context_length", 5000)
        self.min_answer_length = self.validation_config.get("min_answer_length", 1)
        self.max_answer_length = self.validation_config.get("max_answer_length", 2000)
        
        # Required fields for different benchmark types
        self.required_fields = {
            "base": ["question_id", "question", "answer"],
            "mirage": ["question_id", "question", "answer", "context", "options"],
            "medreason": ["question_id", "question", "answer", "reasoning_chain"],
            "pubmedqa": ["question_id", "question", "answer", "context"],
            "msmarco": ["question_id", "question", "answer", "context"]
        }
        
        # Medical validation patterns
        self.medical_term_patterns = [
            r'\b(?:diagnos|symptom|treatment|therapy|disease|disorder|syndrome)\w*\b',
            r'\b(?:patient|clinical|medical|health|hospital|doctor)\w*\b',
            r'\b(?:mg|ml|dose|medication|drug|prescription)\b',
            r'\b(?:blood|heart|lung|liver|kidney|brain)\w*\b'
        ]
        
        # Quality indicators
        self.quality_indicators = {
            "high": ["peer-reviewed", "randomized", "controlled", "meta-analysis", "systematic"],
            "medium": ["study", "research", "clinical", "evidence", "published"],
            "low": ["opinion", "anecdotal", "case report", "preliminary"]
        }
    
    def validate_dataset(self, data: List[Dict[str, Any]], benchmark_name: str) -> Dict[str, Any]:
        """Validate entire dataset and return validation report."""
        logger.info(f"🔍 Validating {benchmark_name} dataset...")
        logger.info(f"   Total samples: {len(data)}")
        
        validation_report = {
            "benchmark_name": benchmark_name,
            "total_samples": len(data),
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "validation_summary": {
                "valid_samples": 0,
                "invalid_samples": 0,
                "warning_samples": 0,
                "critical_errors": 0
            },
            "field_validation": {},
            "content_validation": {},
            "quality_assessment": {},
            "detailed_errors": [],
            "recommendations": []
        }
        
        if not data:
            validation_report["validation_summary"]["critical_errors"] = 1
            validation_report["detailed_errors"].append({
                "type": "critical",
                "message": "Dataset is empty",
                "sample_id": None
            })
            return validation_report
        
        # Validate each sample
        valid_samples = 0
        invalid_samples = 0
        warning_samples = 0
        all_errors = []
        
        for i, sample in enumerate(data):
            sample_validation = self._validate_sample(sample, benchmark_name, i)
            
            if sample_validation["is_valid"]:
                valid_samples += 1
            else:
                invalid_samples += 1
            
            if sample_validation["has_warnings"]:
                warning_samples += 1
            
            all_errors.extend(sample_validation["errors"])
        
        # Update summary
        validation_report["validation_summary"]["valid_samples"] = valid_samples
        validation_report["validation_summary"]["invalid_samples"] = invalid_samples
        validation_report["validation_summary"]["warning_samples"] = warning_samples
        validation_report["validation_summary"]["critical_errors"] = sum(
            1 for error in all_errors if error["severity"] == "critical"
        )
        
        # Field validation analysis
        validation_report["field_validation"] = self._analyze_field_completeness(data, benchmark_name)
        
        # Content validation analysis
        validation_report["content_validation"] = self._analyze_content_quality(data)
        
        # Quality assessment
        validation_report["quality_assessment"] = self._assess_overall_quality(data)
        
        # Store detailed errors (limit to prevent large reports)
        validation_report["detailed_errors"] = all_errors[:100]  # Limit to first 100 errors
        
        # Generate recommendations
        validation_report["recommendations"] = self._generate_recommendations(validation_report)
        
        # Log summary
        self._log_validation_summary(validation_report)
        
        return validation_report
    
    def _validate_sample(self, sample: Dict[str, Any], benchmark_name: str, sample_index: int) -> Dict[str, Any]:
        """Validate a single sample."""
        validation_result = {
            "is_valid": True,
            "has_warnings": False,
            "errors": []
        }
        
        sample_id = sample.get("question_id", f"sample_{sample_index}")
        
        # Check required fields
        required_fields = self.required_fields.get(benchmark_name, self.required_fields["base"])
        for field in required_fields:
            if field not in sample or not sample[field]:
                validation_result["is_valid"] = False
                validation_result["errors"].append({
                    "type": "missing_field",
                    "severity": "critical",
                    "message": f"Missing required field: {field}",
                    "sample_id": sample_id,
                    "sample_index": sample_index
                })
        
        # Validate text lengths
        text_validation = self._validate_text_lengths(sample, sample_id, sample_index)
        validation_result["errors"].extend(text_validation["errors"])
        if text_validation["has_warnings"]:
            validation_result["has_warnings"] = True
        if not text_validation["is_valid"]:
            validation_result["is_valid"] = False
        
        # Validate medical content
        medical_validation = self._validate_medical_content(sample, sample_id, sample_index)
        validation_result["errors"].extend(medical_validation["errors"])
        if medical_validation["has_warnings"]:
            validation_result["has_warnings"] = True
        
        # Benchmark-specific validation
        if benchmark_name == "mirage":
            mirage_validation = self._validate_mirage_specific(sample, sample_id, sample_index)
            validation_result["errors"].extend(mirage_validation["errors"])
            if not mirage_validation["is_valid"]:
                validation_result["is_valid"] = False
        
        elif benchmark_name == "medreason":
            medreason_validation = self._validate_medreason_specific(sample, sample_id, sample_index)
            validation_result["errors"].extend(medreason_validation["errors"])
            if not medreason_validation["is_valid"]:
                validation_result["is_valid"] = False
        
        elif benchmark_name == "pubmedqa":
            pubmedqa_validation = self._validate_pubmedqa_specific(sample, sample_id, sample_index)
            validation_result["errors"].extend(pubmedqa_validation["errors"])
            if not pubmedqa_validation["is_valid"]:
                validation_result["is_valid"] = False
        
        return validation_result
    
    def _validate_text_lengths(self, sample: Dict[str, Any], sample_id: str, sample_index: int) -> Dict[str, Any]:
        """Validate text field lengths."""
        validation_result = {
            "is_valid": True,
            "has_warnings": False,
            "errors": []
        }
        
        # Check question length
        question = sample.get("question", "")
        if len(question) < self.min_question_length:
            validation_result["is_valid"] = False
            validation_result["errors"].append({
                "type": "text_length",
                "severity": "critical",
                "message": f"Question too short: {len(question)} chars (min: {self.min_question_length})",
                "sample_id": sample_id,
                "sample_index": sample_index
            })
        elif len(question) > self.max_question_length:
            validation_result["has_warnings"] = True
            validation_result["errors"].append({
                "type": "text_length",
                "severity": "warning",
                "message": f"Question very long: {len(question)} chars (max: {self.max_question_length})",
                "sample_id": sample_id,
                "sample_index": sample_index
            })
        
        # Check context length (if present)
        context = sample.get("context", "")
        if context:
            if len(context) < self.min_context_length:
                validation_result["has_warnings"] = True
                validation_result["errors"].append({
                    "type": "text_length",
                    "severity": "warning",
                    "message": f"Context short: {len(context)} chars (min: {self.min_context_length})",
                    "sample_id": sample_id,
                    "sample_index": sample_index
                })
            elif len(context) > self.max_context_length:
                validation_result["has_warnings"] = True
                validation_result["errors"].append({
                    "type": "text_length",
                    "severity": "warning",
                    "message": f"Context very long: {len(context)} chars (max: {self.max_context_length})",
                    "sample_id": sample_id,
                    "sample_index": sample_index
                })
        
        # Check answer length
        answer = sample.get("answer", "")
        if len(answer) < self.min_answer_length:
            validation_result["is_valid"] = False
            validation_result["errors"].append({
                "type": "text_length",
                "severity": "critical",
                "message": f"Answer too short: {len(answer)} chars (min: {self.min_answer_length})",
                "sample_id": sample_id,
                "sample_index": sample_index
            })
        elif len(answer) > self.max_answer_length:
            validation_result["has_warnings"] = True
            validation_result["errors"].append({
                "type": "text_length",
                "severity": "warning",
                "message": f"Answer very long: {len(answer)} chars (max: {self.max_answer_length})",
                "sample_id": sample_id,
                "sample_index": sample_index
            })
        
        return validation_result
    
    def _validate_medical_content(self, sample: Dict[str, Any], sample_id: str, sample_index: int) -> Dict[str, Any]:
        """Validate medical content quality."""
        validation_result = {
            "has_warnings": False,
            "errors": []
        }
        
        question = sample.get("question", "").lower()
        context = sample.get("context", "").lower()
        combined_text = question + " " + context
        
        # Check for medical terminology
        medical_term_found = False
        for pattern in self.medical_term_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                medical_term_found = True
                break
        
        if not medical_term_found:
            validation_result["has_warnings"] = True
            validation_result["errors"].append({
                "type": "content_quality",
                "severity": "warning",
                "message": "No clear medical terminology found in question/context",
                "sample_id": sample_id,
                "sample_index": sample_index
            })
        
        # Check for placeholder text
        placeholder_patterns = [
            r'\[.*?\]', r'<.*?>', r'placeholder', r'todo', r'xxx', r'test'
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                validation_result["has_warnings"] = True
                validation_result["errors"].append({
                    "type": "content_quality",
                    "severity": "warning",
                    "message": f"Possible placeholder text found: {pattern}",
                    "sample_id": sample_id,
                    "sample_index": sample_index
                })
                break
        
        return validation_result
    
    def _validate_mirage_specific(self, sample: Dict[str, Any], sample_id: str, sample_index: int) -> Dict[str, Any]:
        """Validate MIRAGE-specific requirements."""
        validation_result = {
            "is_valid": True,
            "errors": []
        }
        
        # Check options format
        options = sample.get("options", [])
        if not isinstance(options, list) or len(options) < 2:
            validation_result["is_valid"] = False
            validation_result["errors"].append({
                "type": "format_error",
                "severity": "critical",
                "message": f"Invalid options format or too few options: {len(options) if isinstance(options, list) else 'not a list'}",
                "sample_id": sample_id,
                "sample_index": sample_index
            })
        
        # Check if answer is in options
        answer = sample.get("answer", "")
        if isinstance(options, list) and answer not in options:
            validation_result["is_valid"] = False
            validation_result["errors"].append({
                "type": "consistency_error",
                "severity": "critical",
                "message": "Answer not found in options list",
                "sample_id": sample_id,
                "sample_index": sample_index
            })
        
        return validation_result
    
    def _validate_medreason_specific(self, sample: Dict[str, Any], sample_id: str, sample_index: int) -> Dict[str, Any]:
        """Validate MedReason-specific requirements."""
        validation_result = {
            "is_valid": True,
            "errors": []
        }
        
        # Check reasoning chain format
        reasoning_chain = sample.get("reasoning_chain", [])
        if not isinstance(reasoning_chain, list):
            validation_result["is_valid"] = False
            validation_result["errors"].append({
                "type": "format_error",
                "severity": "critical",
                "message": "Reasoning chain must be a list",
                "sample_id": sample_id,
                "sample_index": sample_index
            })
        elif len(reasoning_chain) == 0:
            validation_result["is_valid"] = False
            validation_result["errors"].append({
                "type": "content_error",
                "severity": "critical",
                "message": "Reasoning chain cannot be empty",
                "sample_id": sample_id,
                "sample_index": sample_index
            })
        
        return validation_result
    
    def _validate_pubmedqa_specific(self, sample: Dict[str, Any], sample_id: str, sample_index: int) -> Dict[str, Any]:
        """Validate PubMedQA-specific requirements."""
        validation_result = {
            "is_valid": True,
            "errors": []
        }
        
        # Check yes/no answer format
        answer = sample.get("answer", "").lower()
        if answer not in ["yes", "no", "maybe"]:
            validation_result["is_valid"] = False
            validation_result["errors"].append({
                "type": "format_error",
                "severity": "critical",
                "message": f"Answer must be 'yes', 'no', or 'maybe', got: '{answer}'",
                "sample_id": sample_id,
                "sample_index": sample_index
            })
        
        return validation_result
    
    def _analyze_field_completeness(self, data: List[Dict[str, Any]], benchmark_name: str) -> Dict[str, Any]:
        """Analyze field completeness across the dataset."""
        field_stats = defaultdict(lambda: {"present": 0, "missing": 0, "empty": 0})
        
        for sample in data:
            for field in self.required_fields.get(benchmark_name, self.required_fields["base"]):
                if field in sample:
                    if sample[field]:
                        field_stats[field]["present"] += 1
                    else:
                        field_stats[field]["empty"] += 1
                else:
                    field_stats[field]["missing"] += 1
        
        # Calculate percentages
        total_samples = len(data)
        field_analysis = {}
        
        for field, stats in field_stats.items():
            field_analysis[field] = {
                "present_count": stats["present"],
                "missing_count": stats["missing"],
                "empty_count": stats["empty"],
                "completeness_rate": stats["present"] / total_samples if total_samples > 0 else 0,
                "missing_rate": stats["missing"] / total_samples if total_samples > 0 else 0,
                "empty_rate": stats["empty"] / total_samples if total_samples > 0 else 0
            }
        
        return field_analysis
    
    def _analyze_content_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content quality across the dataset."""
        content_analysis = {
            "question_lengths": [],
            "context_lengths": [],
            "answer_lengths": [],
            "medical_terminology_coverage": 0,
            "avg_question_length": 0,
            "avg_context_length": 0,
            "avg_answer_length": 0,
            "text_quality_score": 0
        }
        
        medical_samples = 0
        
        for sample in data:
            # Collect text lengths
            question = sample.get("question", "")
            context = sample.get("context", "")
            answer = sample.get("answer", "")
            
            content_analysis["question_lengths"].append(len(question))
            content_analysis["context_lengths"].append(len(context))
            content_analysis["answer_lengths"].append(len(answer))
            
            # Check medical terminology
            combined_text = (question + " " + context).lower()
            for pattern in self.medical_term_patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    medical_samples += 1
                    break
        
        # Calculate averages
        if data:
            content_analysis["avg_question_length"] = sum(content_analysis["question_lengths"]) / len(data)
            content_analysis["avg_context_length"] = sum(content_analysis["context_lengths"]) / len(data)
            content_analysis["avg_answer_length"] = sum(content_analysis["answer_lengths"]) / len(data)
            content_analysis["medical_terminology_coverage"] = medical_samples / len(data)
        
        # Calculate quality score (0-1)
        quality_factors = [
            min(content_analysis["medical_terminology_coverage"], 1.0),  # Medical relevance
            min(content_analysis["avg_question_length"] / 50, 1.0),     # Question completeness
            min(content_analysis["avg_context_length"] / 200, 1.0),     # Context richness
            min(content_analysis["avg_answer_length"] / 20, 1.0)        # Answer completeness
        ]
        
        content_analysis["text_quality_score"] = sum(quality_factors) / len(quality_factors)
        
        return content_analysis
    
    def _assess_overall_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall dataset quality."""
        quality_assessment = {
            "overall_score": 0.0,
            "quality_grade": "F",
            "strengths": [],
            "weaknesses": [],
            "quality_metrics": {
                "completeness": 0.0,
                "consistency": 0.0,
                "medical_relevance": 0.0,
                "text_quality": 0.0
            }
        }
        
        if not data:
            return quality_assessment
        
        # Calculate completeness (all required fields present)
        complete_samples = 0
        for sample in data:
            if all(sample.get(field) for field in ["question", "answer"]):
                complete_samples += 1
        
        quality_assessment["quality_metrics"]["completeness"] = complete_samples / len(data)
        
        # Calculate consistency (uniform format)
        consistent_samples = 0
        expected_fields = set(data[0].keys()) if data else set()
        for sample in data:
            if set(sample.keys()) == expected_fields:
                consistent_samples += 1
        
        quality_assessment["quality_metrics"]["consistency"] = consistent_samples / len(data)
        
        # Medical relevance (from content analysis)
        medical_samples = 0
        for sample in data:
            combined_text = (sample.get("question", "") + " " + sample.get("context", "")).lower()
            for pattern in self.medical_term_patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    medical_samples += 1
                    break
        
        quality_assessment["quality_metrics"]["medical_relevance"] = medical_samples / len(data)
        
        # Text quality (average length adequacy)
        avg_question_len = sum(len(sample.get("question", "")) for sample in data) / len(data)
        avg_answer_len = sum(len(sample.get("answer", "")) for sample in data) / len(data)
        
        text_quality = min((avg_question_len / 30) + (avg_answer_len / 20), 2.0) / 2.0
        quality_assessment["quality_metrics"]["text_quality"] = text_quality
        
        # Calculate overall score
        metrics = quality_assessment["quality_metrics"]
        overall_score = (
            metrics["completeness"] * 0.3 +
            metrics["consistency"] * 0.2 +
            metrics["medical_relevance"] * 0.3 +
            metrics["text_quality"] * 0.2
        )
        
        quality_assessment["overall_score"] = overall_score
        
        # Assign quality grade
        if overall_score >= 0.9:
            quality_assessment["quality_grade"] = "A"
        elif overall_score >= 0.8:
            quality_assessment["quality_grade"] = "B"
        elif overall_score >= 0.7:
            quality_assessment["quality_grade"] = "C"
        elif overall_score >= 0.6:
            quality_assessment["quality_grade"] = "D"
        else:
            quality_assessment["quality_grade"] = "F"
        
        # Identify strengths and weaknesses
        if metrics["completeness"] >= 0.9:
            quality_assessment["strengths"].append("High data completeness")
        elif metrics["completeness"] < 0.7:
            quality_assessment["weaknesses"].append("Low data completeness")
        
        if metrics["medical_relevance"] >= 0.8:
            quality_assessment["strengths"].append("Strong medical relevance")
        elif metrics["medical_relevance"] < 0.6:
            quality_assessment["weaknesses"].append("Weak medical relevance")
        
        if metrics["consistency"] >= 0.9:
            quality_assessment["strengths"].append("Consistent data format")
        elif metrics["consistency"] < 0.8:
            quality_assessment["weaknesses"].append("Inconsistent data format")
        
        if metrics["text_quality"] >= 0.8:
            quality_assessment["strengths"].append("Good text quality")
        elif metrics["text_quality"] < 0.6:
            quality_assessment["weaknesses"].append("Poor text quality")
        
        return quality_assessment
    
    def _generate_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        summary = validation_report["validation_summary"]
        quality = validation_report.get("quality_assessment", {})
        
        # Critical issues
        if summary["critical_errors"] > 0:
            recommendations.append("🚨 Fix critical errors before using this dataset for evaluation")
        
        # Invalid samples
        invalid_rate = summary["invalid_samples"] / summary["total_samples"] if summary["total_samples"] > 0 else 0
        if invalid_rate > 0.1:
            recommendations.append(f"⚠️ High invalid sample rate ({invalid_rate:.1%}). Consider data cleaning.")
        
        # Quality-based recommendations
        overall_score = quality.get("overall_score", 0)
        if overall_score < 0.7:
            recommendations.append("📈 Overall quality is low. Consider data enhancement or replacement.")
        
        metrics = quality.get("quality_metrics", {})
        
        if metrics.get("completeness", 0) < 0.8:
            recommendations.append("📝 Improve data completeness by filling missing fields")
        
        if metrics.get("medical_relevance", 0) < 0.7:
            recommendations.append("🏥 Enhance medical terminology and clinical relevance")
        
        if metrics.get("consistency", 0) < 0.8:
            recommendations.append("🔧 Standardize data format and field structure")
        
        if metrics.get("text_quality", 0) < 0.6:
            recommendations.append("✍️ Improve text quality - expand short answers and questions")
        
        # Warning-based recommendations
        warning_rate = summary["warning_samples"] / summary["total_samples"] if summary["total_samples"] > 0 else 0
        if warning_rate > 0.2:
            recommendations.append(f"⚡ Review {warning_rate:.1%} of samples with warnings")
        
        # Positive reinforcement
        if overall_score >= 0.8:
            recommendations.append("✅ Dataset quality is good. Ready for evaluation.")
        
        if not recommendations:
            recommendations.append("✨ No major issues found. Dataset appears to be in good condition.")
        
        return recommendations
    
    def _log_validation_summary(self, validation_report: Dict[str, Any]) -> None:
        """Log validation summary."""
        summary = validation_report["validation_summary"]
        quality = validation_report.get("quality_assessment", {})
        
        logger.info(f"📊 Validation Summary for {validation_report['benchmark_name']}:")
        logger.info(f"   Valid samples: {summary['valid_samples']}/{summary['total_samples']} "
                   f"({summary['valid_samples']/summary['total_samples']:.1%})")
        logger.info(f"   Invalid samples: {summary['invalid_samples']}")
        logger.info(f"   Warning samples: {summary['warning_samples']}")
        logger.info(f"   Critical errors: {summary['critical_errors']}")
        
        if quality:
            logger.info(f"   Overall quality: {quality['overall_score']:.2f} (Grade: {quality['quality_grade']})")
        
        # Log top recommendations
        recommendations = validation_report.get("recommendations", [])
        if recommendations:
            logger.info("💡 Top recommendations:")
            for rec in recommendations[:3]:
                logger.info(f"   {rec}")
    
    def save_validation_report(self, validation_report: Dict[str, Any], output_path: Path) -> None:
        """Save validation report to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        logger.info(f"💾 Saved validation report to {output_path}")
    
    def load_validation_report(self, input_path: Path) -> Dict[str, Any]:
        """Load validation report from file."""
        with open(input_path, 'r') as f:
            validation_report = json.load(f)
        
        logger.info(f"📂 Loaded validation report from {input_path}")
        return validation_report
    
    def validate_cross_benchmark_consistency(self, benchmark_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Validate consistency across multiple benchmarks."""
        logger.info("🔄 Validating cross-benchmark consistency...")
        
        consistency_report = {
            "benchmarks_analyzed": list(benchmark_data.keys()),
            "total_samples": sum(len(data) for data in benchmark_data.values()),
            "field_consistency": {},
            "format_consistency": {},
            "quality_comparison": {},
            "recommendations": []
        }
        
        # Analyze field consistency across benchmarks
        all_fields = set()
        common_fields = None
        
        for benchmark_name, data in benchmark_data.items():
            if data:
                benchmark_fields = set(data[0].keys())
                all_fields.update(benchmark_fields)
                
                if common_fields is None:
                    common_fields = benchmark_fields.copy()
                else:
                    common_fields &= benchmark_fields
        
        consistency_report["field_consistency"] = {
            "all_fields": sorted(list(all_fields)),
            "common_fields": sorted(list(common_fields)) if common_fields else [],
            "field_coverage": {}
        }
        
        # Analyze each benchmark's field coverage
        for benchmark_name, data in benchmark_data.items():
            if data:
                benchmark_fields = set(data[0].keys())
                coverage = len(benchmark_fields & common_fields) / len(all_fields) if all_fields else 0
                consistency_report["field_consistency"]["field_coverage"][benchmark_name] = {
                    "fields": sorted(list(benchmark_fields)),
                    "common_field_coverage": coverage,
                    "unique_fields": sorted(list(benchmark_fields - common_fields))
                }
        
        # Quality comparison across benchmarks
        for benchmark_name, data in benchmark_data.items():
            quality_metrics = self._assess_overall_quality(data)
            consistency_report["quality_comparison"][benchmark_name] = quality_metrics
        
        # Generate cross-benchmark recommendations
        consistency_report["recommendations"] = self._generate_cross_benchmark_recommendations(consistency_report)
        
        logger.info(f"✅ Cross-benchmark consistency analysis completed")
        return consistency_report
    
    def _generate_cross_benchmark_recommendations(self, consistency_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations for cross-benchmark consistency."""
        recommendations = []
        
        # Field consistency recommendations
        field_consistency = consistency_report["field_consistency"]
        common_fields = field_consistency["common_fields"]
        all_fields = field_consistency["all_fields"]
        
        if len(common_fields) < len(all_fields) * 0.5:
            recommendations.append("⚠️ Low field consistency across benchmarks. Consider standardizing field names.")
        
        # Quality consistency recommendations
        quality_comparison = consistency_report["quality_comparison"]
        quality_scores = [metrics["overall_score"] for metrics in quality_comparison.values()]
        
        if quality_scores:
            min_quality = min(quality_scores)
            max_quality = max(quality_scores)
            quality_variance = max_quality - min_quality
            
            if quality_variance > 0.3:
                recommendations.append(f"📊 High quality variance across benchmarks ({quality_variance:.2f}). "
                                     "Consider improving lower-quality datasets.")
            
            # Identify problematic benchmarks
            avg_quality = sum(quality_scores) / len(quality_scores)
            for benchmark_name, metrics in quality_comparison.items():
                if metrics["overall_score"] < avg_quality - 0.2:
                    recommendations.append(f"🔍 {benchmark_name} has below-average quality. "
                                         f"Focus improvement efforts here.")
        
        if not recommendations:
            recommendations.append("✅ Good consistency across benchmarks.")
        
        return recommendations
    
    def generate_data_quality_summary(self, benchmark_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate comprehensive data quality summary."""
        logger.info("📋 Generating comprehensive data quality summary...")
        
        summary = {
            "generation_timestamp": pd.Timestamp.now().isoformat(),
            "benchmarks": {},
            "overall_statistics": {
                "total_benchmarks": len(benchmark_data),
                "total_samples": 0,
                "average_quality": 0.0,
                "quality_distribution": {}
            },
            "cross_benchmark_analysis": {},
            "recommendations": {
                "immediate_actions": [],
                "quality_improvements": [],
                "long_term_goals": []
            }
        }
        
        # Validate each benchmark
        all_quality_scores = []
        total_samples = 0
        
        for benchmark_name, data in benchmark_data.items():
            validation_report = self.validate_dataset(data, benchmark_name)
            summary["benchmarks"][benchmark_name] = validation_report
            
            quality_score = validation_report.get("quality_assessment", {}).get("overall_score", 0)
            all_quality_scores.append(quality_score)
            total_samples += len(data)
        
        # Overall statistics
        summary["overall_statistics"]["total_samples"] = total_samples
        if all_quality_scores:
            summary["overall_statistics"]["average_quality"] = sum(all_quality_scores) / len(all_quality_scores)
        
        # Quality distribution
        grade_counts = Counter()
        for benchmark_name, benchmark_summary in summary["benchmarks"].items():
            grade = benchmark_summary.get("quality_assessment", {}).get("quality_grade", "F")
            grade_counts[grade] += 1
        
        summary["overall_statistics"]["quality_distribution"] = dict(grade_counts)
        
        # Cross-benchmark analysis
        summary["cross_benchmark_analysis"] = self.validate_cross_benchmark_consistency(benchmark_data)
        
        # Compile recommendations
        all_recommendations = []
        for benchmark_summary in summary["benchmarks"].values():
            all_recommendations.extend(benchmark_summary.get("recommendations", []))
        
        # Categorize recommendations
        immediate_keywords = ["critical", "fix", "🚨"]
        improvement_keywords = ["improve", "enhance", "📈", "📝", "🏥"]
        
        for rec in all_recommendations:
            rec_lower = rec.lower()
            if any(keyword in rec_lower for keyword in immediate_keywords):
                summary["recommendations"]["immediate_actions"].append(rec)
            elif any(keyword in rec_lower for keyword in improvement_keywords):
                summary["recommendations"]["quality_improvements"].append(rec)
            else:
                summary["recommendations"]["long_term_goals"].append(rec)
        
        # Remove duplicates
        for category in summary["recommendations"]:
            summary["recommendations"][category] = list(set(summary["recommendations"][category]))
        
        logger.info(f"✅ Data quality summary generated for {len(benchmark_data)} benchmarks")
        return summary