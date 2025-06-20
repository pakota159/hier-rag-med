"""
Complete Hierarchical Evaluator implementation
src/evaluation/evaluators/hierarchical_evaluator.py
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
            self.hierarchical_config = Config()
            
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
    
    def validate_setup(self) -> bool:
        """Validate that the hierarchical system is properly setup."""
        try:
            if not self.retriever or not self.generator:
                return False
            
            # Test basic functionality
            test_query = "What is diabetes?"
            docs = self.retrieve_documents(test_query, top_k=1)
            response = self.generate_response(test_query)
            
            return len(response) > 0
            
        except Exception as e:
            logger.error(f"Hierarchical system validation failed: {e}")
            return False