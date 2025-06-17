"""
KG System evaluator for medical RAG evaluation.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from .base_evaluator import BaseEvaluator

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class KGEvaluator(BaseEvaluator):
    """Evaluator for KG Enhanced RAG system."""
    
    def __init__(self, config: Dict):
        """Initialize KG system evaluator."""
        super().__init__(config)
        self.config_path = config.get("config_path", "src/kg/config.yaml")
        self.collection_name = config.get("collection_name", "kg_medical_docs")
        
        # KG system components
        self.kg_config = None
        self.retriever = None
        self.generator = None
        
    def setup_model(self) -> None:
        """Initialize and setup the KG system."""
        try:
            # Import KG system components
            from src.kg.config import Config
            from src.kg.retrieval import Retriever
            from src.kg.generation import Generator
            
            # Load KG configuration
            config_path = Path(self.config_path)
            self.kg_config = Config(config_path)
            
            # Initialize components
            self.retriever = Retriever(self.kg_config)
            self.generator = Generator(self.kg_config)
            
            # Load collection
            self.retriever.load_collection(self.collection_name)
            
            logger.info("✅ KG system setup completed")
            
        except Exception as e:
            logger.error(f"❌ KG system setup failed: {e}")
            raise
    
    def retrieve_documents(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve documents using KG system."""
        if not self.retriever:
            raise RuntimeError("KG retriever not initialized")
        
        try:
            # Use KG retrieval
            results = self.retriever.search(query, n_results=top_k)
            
            # Format results for evaluation
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "score": result["score"]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"KG retrieval failed: {e}")
            return []
    
    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """Generate response using KG system."""
        if not self.generator or not self.retriever:
            raise RuntimeError("KG system not initialized")
        
        try:
            # Retrieve relevant context if not provided
            if context is None:
                retrieved_docs = self.retrieve_documents(question, top_k=5)
                context_docs = retrieved_docs
            else:
                # Use provided context
                context_docs = [{"text": context, "metadata": {}}]
            
            # Generate response using KG generator
            response = self.generator.generate(question, context_docs)
            
            return response
            
        except Exception as e:
            logger.error(f"KG generation failed: {e}")
            return f"Error: {str(e)}"
    
    def evaluate_kg_specific_metrics(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate KG-specific performance metrics."""
        
        # Multi-source evidence integration
        source_diversity = self._calculate_source_diversity(retrieved_docs)
        
        # Knowledge coverage assessment
        knowledge_coverage = self._assess_knowledge_coverage(question, retrieved_docs)
        
        # Response comprehensiveness
        comprehensiveness = self._assess_response_comprehensiveness(response, retrieved_docs)
        
        # Information synthesis quality
        synthesis_quality = self._assess_information_synthesis(response, retrieved_docs)
        
        return {
            "source_diversity": source_diversity,
            "knowledge_coverage": knowledge_coverage,
            "response_comprehensiveness": comprehensiveness,
            "information_synthesis": synthesis_quality,
            "kg_specific_score": (source_diversity + knowledge_coverage + comprehensiveness + synthesis_quality) / 4
        }
    
    def _calculate_source_diversity(self, retrieved_docs: List[Dict]) -> float:
        """Calculate diversity of retrieved document sources."""
        if not retrieved_docs:
            return 0.0
        
        # Extract source types
        sources = set()
        for doc in retrieved_docs:
            source = doc.get("metadata", {}).get("source", "unknown")
            sources.add(source)
        
        # Normalize by expected number of sources in KG system
        expected_sources = 4  # pubmed, mtsamples, mesh, etc.
        return min(len(sources) / expected_sources, 1.0)
    
    def _assess_knowledge_coverage(self, question: Dict, retrieved_docs: List[Dict]) -> float:
        """Assess how well retrieved documents cover the question topic."""
        question_text = question.get("question", "").lower()
        
        # Extract key medical terms from question
        medical_keywords = self._extract_medical_keywords(question_text)
        
        if not medical_keywords:
            return 0.5
        
        # Check coverage in retrieved documents
        covered_keywords = set()
        for doc in retrieved_docs:
            doc_text = doc.get("text", "").lower()
            for keyword in medical_keywords:
                if keyword in doc_text:
                    covered_keywords.add(keyword)
        
        return len(covered_keywords) / len(medical_keywords)
    
    def _assess_response_comprehensiveness(self, response: str, retrieved_docs: List[Dict]) -> float:
        """Assess comprehensiveness of response using retrieved knowledge."""
        if not retrieved_docs:
            return 0.0
        
        response_lower = response.lower()
        
        # Check if response incorporates information from multiple sources
        source_incorporation = 0
        for doc in retrieved_docs[:5]:  # Check top 5 documents
            doc_text = doc.get("text", "").lower()
            doc_words = set(doc_text.split())
            response_words = set(response_lower.split())
            
            # Calculate overlap
            if doc_words:
                overlap = len(response_words.intersection(doc_words)) / len(doc_words)
                if overlap > 0.1:  # Threshold for meaningful incorporation
                    source_incorporation += 1
        
        return min(source_incorporation / len(retrieved_docs[:5]), 1.0)
    
    def _assess_information_synthesis(self, response: str, retrieved_docs: List[Dict]) -> float:
        """Assess quality of information synthesis across sources."""
        response_lower = response.lower()
        
        # Check for synthesis indicators
        synthesis_indicators = [
            "according to", "research shows", "studies indicate", "evidence suggests",
            "multiple sources", "various studies", "combined evidence", "overall"
        ]
        
        synthesis_count = sum(1 for indicator in synthesis_indicators if indicator in response_lower)
        return min(synthesis_count / len(synthesis_indicators), 1.0)
    
    def _extract_medical_keywords(self, text: str) -> List[str]:
        """Extract medical keywords from text."""
        # Simple medical keyword extraction
        medical_terms = [
            "diabetes", "hypertension", "heart", "blood", "pressure", "disease",
            "treatment", "medication", "symptoms", "diagnosis", "patient", "medical",
            "therapy", "clinical", "condition", "syndrome", "infection", "cancer"
        ]
        
        text_words = text.lower().split()
        found_keywords = [term for term in medical_terms if term in text_words]
        
        return found_keywords
    
    def get_system_status(self) -> Dict:
        """Get current status of KG system."""
        return {
            "model_name": self.model_name,
            "collection_name": self.collection_name,
            "retriever_ready": self.retriever is not None,
            "generator_ready": self.generator is not None,
            "config_loaded": self.kg_config is not None
        }