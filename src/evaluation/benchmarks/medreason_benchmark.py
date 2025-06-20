# src/evaluation/benchmarks/medreason_benchmark.py
"""
Updated MedReason Benchmark for Knowledge Graph Reasoning
Focuses on multi-step medical reasoning and diagnostic chains
"""

import re
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base_benchmark import BaseBenchmark
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.data.data_loader import BenchmarkDataLoader


class MedReasonBenchmark(BaseBenchmark):
    """
    MedReason benchmark for evaluating knowledge graph-guided medical reasoning.
    Focuses on multi-step diagnostic reasoning and clinical decision-making.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MedReason"
        self.data_loader = BenchmarkDataLoader(config)
        
        # Initialize evaluation models
        self.similarity_model = None
        self._init_evaluation_models()
        
        # Reasoning-specific patterns
        self.reasoning_patterns = {
            'logical_steps': ['first', 'then', 'next', 'therefore', 'consequently', 'thus'],
            'causal_reasoning': ['because', 'due to', 'caused by', 'leads to', 'results in'],
            'differential': ['differential', 'consider', 'rule out', 'exclude', 'possible'],
            'evidence_integration': ['based on', 'evidence shows', 'studies indicate', 'research suggests'],
            'clinical_correlation': ['correlates with', 'consistent with', 'typical of', 'characteristic of']
        }
        
        self.knowledge_domains = {
            'pathophysiology': ['mechanism', 'pathway', 'process', 'physiology', 'pathogenesis'],
            'pharmacology': ['drug', 'medication', 'treatment', 'therapy', 'pharmacokinetics'],
            'anatomy': ['anatomical', 'structure', 'organ', 'system', 'location'],
            'biochemistry': ['molecular', 'biochemical', 'enzyme', 'protein', 'metabolic'],
            'clinical_signs': ['symptom', 'sign', 'presentation', 'manifestation', 'finding']
        }
    
    def load_dataset(self) -> List[Dict]:
        """Load MedReason dataset with enhanced reasoning evaluation."""
        logger.info(f"🔄 Loading MedReason benchmark...")
        
        try:
            max_samples = self.sample_size if not self.is_unlimited else None
            data = self.data_loader.load_benchmark_data("medreason", max_samples=max_samples)
            
            if data and len(data) > 0:
                # Enhance data with reasoning analysis
                enhanced_data = []
                for item in data:
                    enhanced_item = self._enhance_reasoning_item(item)
                    enhanced_data.append(enhanced_item)
                
                logger.info(f"✅ Loaded {len(enhanced_data)} MedReason questions with reasoning analysis")
                return enhanced_data
            else:
                raise ConnectionError("Failed to load MedReason benchmark data.")
                
        except Exception as e:
            logger.error(f"❌ Failed to load MedReason data: {e}")
            raise e
    
    def _enhance_reasoning_item(self, item: Dict) -> Dict:
        """Enhance item with reasoning analysis."""
        enhanced = item.copy()
        
        # Analyze reasoning chain complexity
        reasoning_chain = item.get("reasoning_chain", [])
        if isinstance(reasoning_chain, str):
            reasoning_chain = self._parse_reasoning_chain(reasoning_chain)
        
        enhanced.update({
            "reasoning_steps": len(reasoning_chain) if reasoning_chain else 1,
            "reasoning_complexity": self._assess_reasoning_complexity(reasoning_chain),
            "knowledge_domains": self._identify_knowledge_domains(item.get("question", "")),
            "reasoning_type": self._classify_reasoning_type(item.get("question", "")),
            "benchmark": "medreason"
        })
        
        return enhanced
    
    def _parse_reasoning_chain(self, reasoning_text: str) -> List[str]:
        """Parse reasoning chain from text."""
        if not reasoning_text:
            return []
        
        # Split by common delimiters
        steps = re.split(r'[.\n]|Step \d+:|Then:|Next:|Therefore:', reasoning_text)
        steps = [step.strip() for step in steps if step.strip()]
        
        return steps
    
    def _assess_reasoning_complexity(self, reasoning_chain: List[str]) -> str:
        """Assess complexity of reasoning chain."""
        if not reasoning_chain:
            return "simple"
        
        step_count = len(reasoning_chain)
        if step_count <= 2:
            return "simple"
        elif step_count <= 4:
            return "moderate"
        else:
            return "complex"
    
    def _identify_knowledge_domains(self, question: str) -> List[str]:
        """Identify medical knowledge domains in question."""
        question_lower = question.lower()
        domains = []
        
        for domain, keywords in self.knowledge_domains.items():
            if any(keyword in question_lower for keyword in keywords):
                domains.append(domain)
        
        return domains if domains else ["general_medicine"]
    
    def _classify_reasoning_type(self, question: str) -> str:
        """Classify the type of reasoning required."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['mechanism', 'why', 'how', 'explain']):
            return "causal_reasoning"
        elif any(word in question_lower for word in ['diagnose', 'diagnosis', 'differential']):
            return "diagnostic_reasoning"
        elif any(word in question_lower for word in ['treatment', 'management', 'therapy']):
            return "therapeutic_reasoning"
        elif any(word in question_lower for word in ['prognosis', 'outcome', 'course']):
            return "prognostic_reasoning"
        else:
            return "general_reasoning"
    
    def _init_evaluation_models(self):
        """Initialize models for evaluation metrics."""
        try:
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Initialized similarity model for MedReason evaluation")
        except ImportError:
            logger.warning("⚠️ SentenceTransformers not available - some metrics will be limited")
            self.similarity_model = None
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[str]) -> Dict[str, Any]:
        """
        Evaluate response using MedReason-specific metrics focusing on reasoning quality.
        
        Args:
            question: Question data including reasoning chain
            response: Generated response to evaluate
            retrieved_docs: List of retrieved document snippets
            
        Returns:
            Dictionary containing evaluation scores and metrics
        """
        logger.debug(f"🔍 Evaluating MedReason response for question {question.get('question_id', 'unknown')}")
        
        # Initialize metrics
        metrics = {
            "reasoning_accuracy": 0.0,
            "reasoning_completeness": 0.0,
            "logical_consistency": 0.0,
            "knowledge_integration": 0.0,
            "step_by_step_quality": 0.0,
            "medical_reasoning": 0.0,
            "overall_score": 0.0
        }
        
        try:
            # 1. Reasoning accuracy
            metrics["reasoning_accuracy"] = self._evaluate_reasoning_accuracy(response, question)
            
            # 2. Reasoning completeness
            metrics["reasoning_completeness"] = self._evaluate_reasoning_completeness(response, question)
            
            # 3. Logical consistency
            metrics["logical_consistency"] = self._evaluate_logical_consistency(response)
            
            # 4. Knowledge integration
            metrics["knowledge_integration"] = self._evaluate_knowledge_integration(response, question)
            
            # 5. Step-by-step quality
            metrics["step_by_step_quality"] = self._evaluate_step_quality(response, question)
            
            # 6. Medical reasoning
            metrics["medical_reasoning"] = self._evaluate_medical_reasoning(response, question)
            
            # 7. Calculate overall score
            metrics["overall_score"] = self._calculate_reasoning_score(metrics)
            
            logger.debug(f"   📊 MedReason evaluation complete. Overall score: {metrics['overall_score']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ MedReason evaluation failed: {e}")
            
        return metrics
    
    def _evaluate_reasoning_accuracy(self, response: str, question: Dict) -> float:
        """Evaluate accuracy of reasoning steps."""
        correct_answer = question.get("answer", "").strip()
        if not correct_answer:
            return 0.0
        
        # Basic answer matching
        response_clean = self._normalize_text(response)
        answer_clean = self._normalize_text(correct_answer)
        
        # Exact match
        if response_clean == answer_clean:
            return 1.0
        
        # Semantic similarity if available
        if self.similarity_model:
            try:
                embeddings = self.similarity_model.encode([response, correct_answer])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return max(0.0, similarity)
            except:
                pass
        
        # Keyword overlap as fallback
        response_words = set(response_clean.split())
        answer_words = set(answer_clean.split())
        if answer_words:
            overlap = len(response_words.intersection(answer_words)) / len(answer_words)
            return overlap
        
        return 0.0
    
    def _evaluate_reasoning_completeness(self, response: str, question: Dict) -> float:
        """Evaluate completeness of reasoning chain."""
        expected_steps = question.get("reasoning_steps", 1)
        reasoning_chain = question.get("reasoning_chain", [])
        
        response_lower = response.lower()
        
        # Count reasoning indicators in response
        reasoning_indicators = 0
        for category, patterns in self.reasoning_patterns.items():
            for pattern in patterns:
                if pattern in response_lower:
                    reasoning_indicators += 1
        
        # Score based on expected complexity
        if expected_steps <= 2:
            return 1.0 if reasoning_indicators >= 1 else 0.5
        elif expected_steps <= 4:
            return min(1.0, reasoning_indicators / 3.0)
        else:
            return min(1.0, reasoning_indicators / 5.0)
    
    def _evaluate_logical_consistency(self, response: str) -> float:
        """Evaluate logical consistency of reasoning."""
        response_lower = response.lower()
        
        # Check for logical flow indicators
        logical_score = 0.0
        
        # Positive indicators
        logical_indicators = ['therefore', 'thus', 'consequently', 'because', 'since']
        logical_count = sum(1 for indicator in logical_indicators if indicator in response_lower)
        logical_score += min(0.5, logical_count * 0.1)
        
        # Sequential indicators
        sequential_indicators = ['first', 'second', 'then', 'next', 'finally']
        sequential_count = sum(1 for indicator in sequential_indicators if indicator in response_lower)
        logical_score += min(0.3, sequential_count * 0.1)
        
        # Contradiction indicators (negative)
        contradiction_indicators = ['however', 'but', 'although', 'despite']
        contradiction_count = sum(1 for indicator in contradiction_indicators if indicator in response_lower)
        logical_score -= contradiction_count * 0.1
        
        # Response length consideration
        word_count = len(response.split())
        if 30 <= word_count <= 300:  # Appropriate length for reasoning
            logical_score += 0.2
        
        return max(0.0, min(1.0, logical_score))
    
    def _evaluate_knowledge_integration(self, response: str, question: Dict) -> float:
        """Evaluate integration of medical knowledge."""
        response_lower = response.lower()
        question_domains = question.get("knowledge_domains", [])
        
        if not question_domains:
            return 0.5  # Neutral score if no domains identified
        
        domain_coverage = 0.0
        for domain in question_domains:
            domain_keywords = self.knowledge_domains.get(domain, [])
            domain_matches = sum(1 for keyword in domain_keywords if keyword in response_lower)
            if domain_matches > 0:
                domain_coverage += 1.0
        
        coverage_score = domain_coverage / len(question_domains) if question_domains else 0.0
        
        # Bonus for cross-domain integration
        total_domains_mentioned = sum(
            1 for domain_keywords in self.knowledge_domains.values()
            if any(keyword in response_lower for keyword in domain_keywords)
        )
        
        integration_bonus = min(0.3, (total_domains_mentioned - 1) * 0.1) if total_domains_mentioned > 1 else 0.0
        
        return min(1.0, coverage_score + integration_bonus)
    
    def _evaluate_step_quality(self, response: str, question: Dict) -> float:
        """Evaluate quality of step-by-step reasoning."""
        response_lower = response.lower()
        
        # Check for step indicators
        step_patterns = [
            r'step \d+', r'first\b', r'second\b', r'third\b', r'then\b', 
            r'next\b', r'finally\b', r'therefore\b'
        ]
        
        step_count = sum(1 for pattern in step_patterns if re.search(pattern, response_lower))
        
        # Quality indicators
        quality_indicators = [
            'analysis', 'evaluation', 'assessment', 'consideration', 
            'examination', 'investigation', 'reasoning'
        ]
        quality_count = sum(1 for indicator in quality_indicators if indicator in response_lower)
        
        # Calculate step quality score
        step_score = min(0.6, step_count * 0.15)
        quality_score = min(0.4, quality_count * 0.1)
        
        return step_score + quality_score
    
    def _evaluate_medical_reasoning(self, response: str, question: Dict) -> float:
        """Evaluate medical-specific reasoning quality."""
        response_lower = response.lower()
        
        # Medical reasoning indicators
        medical_reasoning_terms = [
            'diagnosis', 'differential', 'etiology', 'pathophysiology',
            'treatment', 'prognosis', 'complications', 'risk factors',
            'clinical presentation', 'management', 'therapy'
        ]
        
        medical_count = sum(1 for term in medical_reasoning_terms if term in response_lower)
        medical_score = min(0.7, medical_count * 0.1)
        
        # Clinical thinking patterns
        clinical_patterns = [
            'patient presents with', 'likely diagnosis', 'rule out',
            'consider', 'differential includes', 'based on symptoms'
        ]
        
        clinical_count = sum(1 for pattern in clinical_patterns if pattern in response_lower)
        clinical_score = min(0.3, clinical_count * 0.1)
        
        return medical_score + clinical_score
    
    def _calculate_reasoning_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall reasoning score."""
        weights = {
            "reasoning_accuracy": 0.25,
            "reasoning_completeness": 0.20,
            "logical_consistency": 0.15,
            "knowledge_integration": 0.15,
            "step_by_step_quality": 0.15,
            "medical_reasoning": 0.10
        }
        
        overall = sum(metrics.get(metric, 0.0) * weight for metric, weight in weights.items())
        return max(0.0, min(1.0, overall))
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return re.sub(r'[^\w\s]', '', text.lower().strip())
    
    def get_evaluation_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for MedReason evaluation."""
        if not results:
            return {"error": "No results to summarize"}
        
        metrics = ["reasoning_accuracy", "reasoning_completeness", "logical_consistency",
                  "knowledge_integration", "step_by_step_quality", "medical_reasoning", "overall_score"]
        
        summary = {
            "total_questions": len(results),
            "average_scores": {},
            "reasoning_analysis": {},
            "performance_by_complexity": {},
            "benchmark": "MedReason"
        }
        
        # Calculate averages
        for metric in metrics:
            scores = [r.get(metric, 0.0) for r in results]
            summary["average_scores"][metric] = np.mean(scores) if scores else 0.0
        
        # Reasoning complexity analysis
        complexity_types = ["simple", "moderate", "complex"]
        for complexity in complexity_types:
            complexity_results = [r for r in results if r.get("reasoning_complexity") == complexity]
            if complexity_results:
                complexity_scores = [r.get("overall_score", 0.0) for r in complexity_results]
                summary["performance_by_complexity"][complexity] = {
                    "count": len(complexity_results),
                    "average_score": np.mean(complexity_scores),
                    "accuracy": np.mean([r.get("reasoning_accuracy", 0.0) for r in complexity_results])
                }
        
        # Reasoning type analysis
        reasoning_types = set(r.get("reasoning_type", "unknown") for r in results)
        summary["reasoning_analysis"] = {}
        for rtype in reasoning_types:
            type_results = [r for r in results if r.get("reasoning_type") == rtype]
            if type_results:
                type_scores = [r.get("overall_score", 0.0) for r in type_results]
                summary["reasoning_analysis"][rtype] = {
                    "count": len(type_results),
                    "average_score": np.mean(type_scores),
                    "reasoning_quality": np.mean([r.get("logical_consistency", 0.0) for r in type_results])
                }
        
        return summary