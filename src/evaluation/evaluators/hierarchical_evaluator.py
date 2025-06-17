"""
Hierarchical System evaluator for medical RAG evaluation.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from .base_evaluator import BaseEvaluator

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class HierarchicalEvaluator(BaseEvaluator):
    """Evaluator for Hierarchical Diagnostic RAG system."""
    
    def __init__(self, config: Dict):
        """Initialize Hierarchical system evaluator."""
        super().__init__(config)
        self.config_path = config.get("config_path", "src/basic_reasoning/config.yaml")
        self.collection_names = config.get("collection_names", {})
        
        # Hierarchical system components
        self.hierarchical_config = None
        self.retriever = None
        self.generator = None
        
    def setup_model(self) -> None:
        """Initialize and setup the Hierarchical system."""
        try:
            # Import Hierarchical system components
            from src.basic_reasoning.config import Config
            from src.basic_reasoning.retrieval import HierarchicalRetriever
            from src.basic_reasoning.generation import HierarchicalGenerator
            
            # Load Hierarchical configuration
            config_path = Path(self.config_path)
            self.hierarchical_config = Config(config_path)
            
            # Initialize components
            self.retriever = HierarchicalRetriever(self.hierarchical_config)
            self.generator = HierarchicalGenerator(self.hierarchical_config)
            
            # Load hierarchical collections
            self.retriever.load_hierarchical_collections()
            
            logger.info("✅ Hierarchical system setup completed")
            
        except Exception as e:
            logger.error(f"❌ Hierarchical system setup failed: {e}")
            raise
    
    def retrieve_documents(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve documents using Hierarchical system."""
        if not self.retriever:
            raise RuntimeError("Hierarchical retriever not initialized")
        
        try:
            # Use hierarchical retrieval
            hierarchical_results = self.retriever.hierarchical_search(query)
            
            # Combine results from all tiers
            all_results = []
            
            # Add tier 1 results
            for result in hierarchical_results.get("tier1_patterns", []):
                result["tier"] = "tier1_pattern_recognition"
                all_results.append(result)
            
            # Add tier 2 results
            for result in hierarchical_results.get("tier2_hypotheses", []):
                result["tier"] = "tier2_hypothesis_testing"
                all_results.append(result)
            
            # Add tier 3 results
            for result in hierarchical_results.get("tier3_confirmation", []):
                result["tier"] = "tier3_confirmation"
                all_results.append(result)
            
            # Sort by score and return top_k
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return all_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hierarchical retrieval failed: {e}")
            return []
    
    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """Generate response using Hierarchical system."""
        if not self.generator or not self.retriever:
            raise RuntimeError("Hierarchical system not initialized")
        
        try:
            # Retrieve hierarchical context if not provided
            if context is None:
                hierarchical_results = self.retriever.hierarchical_search(question)
            else:
                # Use provided context (convert to hierarchical format)
                hierarchical_results = {
                    "tier1_patterns": [{"text": context, "metadata": {}, "score": 1.0}],
                    "tier2_hypotheses": [],
                    "tier3_confirmation": []
                }
            
            # Generate response using hierarchical generator
            response = self.generator.generate_hierarchical_response(question, hierarchical_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Hierarchical generation failed: {e}")
            return f"Error: {str(e)}"
    
    def evaluate_hierarchical_specific_metrics(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate Hierarchical-specific performance metrics."""
        
        # Three-tier reasoning assessment
        tier_performance = self._assess_tier_performance(retrieved_docs)
        
        # Clinical workflow alignment
        workflow_alignment = self._assess_clinical_workflow(response)
        
        # Diagnostic reasoning quality
        reasoning_quality = self._assess_diagnostic_reasoning(response, question)
        
        # Evidence stratification effectiveness
        evidence_stratification = self._assess_evidence_stratification(retrieved_docs)
        
        # Hierarchical integration score
        integration_score = self._assess_hierarchical_integration(response, retrieved_docs)
        
        return {
            "tier_performance": tier_performance,
            "clinical_workflow_alignment": workflow_alignment,
            "diagnostic_reasoning_quality": reasoning_quality,
            "evidence_stratification": evidence_stratification,
            "hierarchical_integration": integration_score,
            "hierarchical_specific_score": (
                sum(tier_performance.values()) / 3 * 0.3 +
                workflow_alignment * 0.25 +
                reasoning_quality * 0.25 +
                evidence_stratification * 0.1 +
                integration_score * 0.1
            )
        }
    
    def _assess_tier_performance(self, retrieved_docs: List[Dict]) -> Dict[str, float]:
        """Assess performance of each tier in the hierarchy."""
        tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0}
        tier_scores = {"tier1": 0.0, "tier2": 0.0, "tier3": 0.0}
        
        for doc in retrieved_docs:
            tier = doc.get("tier", "unknown")
            score = doc.get("score", 0.0)
            
            if "tier1" in tier:
                tier_counts["tier1"] += 1
                tier_scores["tier1"] += score
            elif "tier2" in tier:
                tier_counts["tier2"] += 1
                tier_scores["tier2"] += score
            elif "tier3" in tier:
                tier_counts["tier3"] += 1
                tier_scores["tier3"] += score
        
        # Calculate average scores per tier
        tier_performance = {}
        for tier in ["tier1", "tier2", "tier3"]:
            if tier_counts[tier] > 0:
                tier_performance[tier] = tier_scores[tier] / tier_counts[tier]
            else:
                tier_performance[tier] = 0.0
        
        return tier_performance
    
    def _assess_clinical_workflow(self, response: str) -> float:
        """Assess alignment with clinical workflow patterns."""
        response_lower = response.lower()
        
        # Check for clinical workflow components
        workflow_patterns = [
            # Pattern Recognition indicators
            ["pattern", "recognition", "initial", "presentation"],
            # Hypothesis Testing indicators
            ["hypothesis", "differential", "testing", "evidence"],
            # Confirmation indicators
            ["confirmation", "final", "diagnosis", "conclusion"]
        ]
        
        pattern_scores = []
        for pattern_group in workflow_patterns:
            pattern_count = sum(1 for pattern in pattern_group if pattern in response_lower)
            pattern_score = min(pattern_count / len(pattern_group), 1.0)
            pattern_scores.append(pattern_score)
        
        return sum(pattern_scores) / len(pattern_scores)
    
    def _assess_diagnostic_reasoning(self, response: str, question: Dict) -> float:
        """Assess quality of diagnostic reasoning in response."""
        response_lower = response.lower()
        
        # Check for reasoning indicators
        reasoning_indicators = [
            "because", "therefore", "suggests", "indicates", "likely", "probable",
            "differential", "diagnosis", "consider", "rule out", "evidence"
        ]
        
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
        reasoning_score = min(reasoning_count / len(reasoning_indicators), 1.0)
        
        # Check for structured thinking
        structure_indicators = ["first", "second", "next", "then", "finally", "step"]
        structure_count = sum(1 for indicator in structure_indicators if indicator in response_lower)
        structure_score = min(structure_count / 3, 1.0)  # Expect at least 3 steps
        
        # Check for medical specificity
        question_type = question.get("type", "")
        if question_type == "clinical":
            specificity_score = self._check_clinical_specificity(response_lower)
        else:
            specificity_score = 0.8  # Default for non-clinical questions
        
        return (reasoning_score * 0.4 + structure_score * 0.3 + specificity_score * 0.3)
    
    def _assess_evidence_stratification(self, retrieved_docs: List[Dict]) -> float:
        """Assess effectiveness of evidence stratification."""
        if not retrieved_docs:
            return 0.0
        
        # Check distribution across tiers
        tier_distribution = {"tier1": 0, "tier2": 0, "tier3": 0}
        
        for doc in retrieved_docs:
            tier = doc.get("tier", "unknown")
            if "tier1" in tier:
                tier_distribution["tier1"] += 1
            elif "tier2" in tier:
                tier_distribution["tier2"] += 1
            elif "tier3" in tier:
                tier_distribution["tier3"] += 1
        
        # Ideal distribution: some from each tier
        total_docs = len(retrieved_docs)
        if total_docs == 0:
            return 0.0
        
        # Check if all tiers are represented
        active_tiers = sum(1 for count in tier_distribution.values() if count > 0)
        tier_diversity = active_tiers / 3.0
        
        # Check balance (no single tier dominating)
        max_proportion = max(tier_distribution.values()) / total_docs
        balance_score = 1.0 - (max_proportion - 0.33) if max_proportion > 0.33 else 1.0
        
        return (tier_diversity * 0.6 + balance_score * 0.4)
    
    def _assess_hierarchical_integration(self, response: str, retrieved_docs: List[Dict]) -> float:
        """Assess how well response integrates information across tiers."""
        response_lower = response.lower()
        
        # Check for integration indicators
        integration_indicators = [
            "combining", "integrating", "together", "overall", "comprehensive",
            "considering", "taking into account", "based on multiple"
        ]
        
        integration_count = sum(1 for indicator in integration_indicators if indicator in response_lower)
        integration_score = min(integration_count / len(integration_indicators), 1.0)
        
        # Check tier-specific content usage
        tier_usage = {"tier1": False, "tier2": False, "tier3": False}
        
        for doc in retrieved_docs:
            tier = doc.get("tier", "unknown")
            doc_words = set(doc.get("text", "").lower().split())
            response_words = set(response_lower.split())
            
            if doc_words.intersection(response_words):
                if "tier1" in tier:
                    tier_usage["tier1"] = True
                elif "tier2" in tier:
                    tier_usage["tier2"] = True
                elif "tier3" in tier:
                    tier_usage["tier3"] = True
        
        tier_integration = sum(tier_usage.values()) / 3.0
        
        return (integration_score * 0.4 + tier_integration * 0.6)
    
    def _check_clinical_specificity(self, response_lower: str) -> float:
        """Check for clinical specificity in response."""
        clinical_terms = [
            "diagnosis", "treatment", "medication", "therapy", "clinical",
            "patient", "symptoms", "condition", "disease", "medical"
        ]
        
        clinical_count = sum(1 for term in clinical_terms if term in response_lower)
        return min(clinical_count / len(clinical_terms), 1.0)
    
    def get_hierarchical_status(self) -> Dict:
        """Get current status of Hierarchical system."""
        return {
            "model_name": self.model_name,
            "collection_names": self.collection_names,
            "retriever_ready": self.retriever is not None,
            "generator_ready": self.generator is not None,
            "config_loaded": self.hierarchical_config is not None,
            "tiers_active": 3 if self.retriever else 0
        }