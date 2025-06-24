#!/usr/bin/env python3
"""
Enhanced Hierarchical Evaluator implementation with Medical Embedding Support
Updated for Microsoft BiomedNLP-PubMedBERT integration

File: src/evaluation/evaluators/hierarchical_evaluator.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
import time

from .base_evaluator import BaseEvaluator

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class HierarchicalEvaluator(BaseEvaluator):
    """Enhanced evaluator for Hierarchical Diagnostic RAG system with medical embedding support."""
    
    def __init__(self, config: Dict):
        """Initialize Hierarchical system evaluator with medical optimizations."""
        super().__init__(config)
        self.model_name = "hierarchical_system"
        self.config_path = config.get("config_path", "src/basic_reasoning/config.yaml")
        
        # Hierarchical system components
        self.hierarchical_config = None
        self.retriever = None
        self.generator = None
        
        # Medical evaluation settings
        self.medical_validation_enabled = config.get("evaluation", {}).get("enable_medical_validation", True)
        self.check_medical_terminology = config.get("evaluation", {}).get("check_medical_terminology", True)
        
        # Performance tracking
        self.setup_time = None
        self.inference_times = []
        
    def setup_model(self) -> None:
        """Initialize and setup the Hierarchical system with medical embedding."""
        setup_start = time.time()
        
        try:
            logger.info("🏥 Setting up Enhanced Hierarchical Medical RAG System")
            
            # Import Hierarchical system components
            from src.basic_reasoning.config import Config
            from src.basic_reasoning.retrieval import HierarchicalRetriever
            from src.basic_reasoning.generation import HierarchicalGenerator
            
            # Load Hierarchical configuration
            logger.info("📋 Loading hierarchical configuration...")
            self.hierarchical_config = Config()
            
            # Validate medical setup
            if not self.hierarchical_config.validate_medical_setup():
                logger.warning("⚠️ Medical embedding setup validation failed")
            
            # Display configuration info
            device_info = self.hierarchical_config.get_device_info()
            logger.info("🔧 Hierarchical System Configuration:")
            logger.info(f"   🖥️  Environment: {device_info['environment']}")
            logger.info(f"   🎯 Device: {device_info['device']}")
            logger.info(f"   🧠 Embedding: {device_info['embedding_model']}")
            logger.info(f"   🏥 Medical optimized: {device_info.get('medical_optimized', False)}")
            logger.info(f"   📏 Batch size: {device_info['batch_size']}")
            
            # Initialize components
            logger.info("🔧 Initializing retrieval system...")
            self.retriever = HierarchicalRetriever(self.hierarchical_config)
            
            # Perform health check
            health = self.retriever.health_check()
            logger.info("🔍 System Health Check:")
            for check, status in health.items():
                status_icon = "✅" if status else "❌"
                logger.info(f"   {status_icon} {check}: {status}")
            
            if not all(health.values()):
                raise RuntimeError("Health check failed - system not ready")
            
            # Optimize for inference
            self.retriever.optimize_for_inference()
            
            # Load hierarchical collections
            logger.info("📚 Loading hierarchical collections...")
            if not self.retriever.load_hierarchical_collections():
                raise RuntimeError("Failed to load hierarchical collections")
            
            # Get collection statistics
            stats = self.retriever.get_collection_stats()
            logger.info("📊 Collection Statistics:")
            logger.info(f"   📚 Total documents: {stats['total']:,}")
            logger.info(f"   🧠 Embedding model: {stats['embedding_model']}")
            logger.info(f"   🏥 Medical optimized: {stats['medical_optimized']}")
            logger.info(f"   🎯 Device: {stats['device']}")
            
            for tier in ["tier1", "tier2", "tier3"]:
                tier_stats = stats[tier]
                if tier_stats["exists"]:
                    logger.info(f"   📁 {tier_stats['name']}: {tier_stats['count']:,} docs")
            
            # Initialize generator
            logger.info("🔧 Initializing generation system...")
            self.generator = HierarchicalGenerator(self.hierarchical_config)
            
            self.setup_time = time.time() - setup_start
            logger.info(f"✅ Hierarchical system setup completed in {self.setup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Hierarchical system setup failed: {e}")
            raise
    
    def retrieve_documents(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve documents using enhanced Hierarchical system."""
        if not self.retriever:
            raise RuntimeError("Retriever not initialized")
        
        try:
            # Perform hierarchical search with medical optimizations
            search_results = self.retriever.search_hierarchical(
                query=query,
                use_all_tiers=True,
                adaptive_selection=True
            )
            
            # Extract combined results
            documents = search_results.get("combined", [])
            
            # Limit to requested number
            documents = documents[:top_k]
            
            # Add evaluation metadata
            for i, doc in enumerate(documents):
                doc.update({
                    "retrieval_rank": i + 1,
                    "query_classification": search_results.get("query_classification"),
                    "search_strategy": search_results.get("search_strategy")
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Document retrieval failed: {e}")
            return []
    
    def generate_answer(self, query: str, documents: List[Dict]) -> str:
        """Generate answer using Hierarchical system with medical context."""
        if not self.generator:
            raise RuntimeError("Generator not initialized")
        
        inference_start = time.time()
        
        try:
            # Prepare context with hierarchical information
            context = self._prepare_hierarchical_context(documents)
            
            # Generate answer with medical prompts
            answer = self.generator.generate_hierarchical_answer(
                query=query,
                context=context,
                documents=documents
            )
            
            inference_time = time.time() - inference_start
            self.inference_times.append(inference_time)
            
            # Validate medical content if enabled
            if self.medical_validation_enabled:
                answer = self._validate_medical_answer(answer, query, documents)
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _prepare_hierarchical_context(self, documents: List[Dict]) -> str:
        """Prepare context with hierarchical tier information."""
        if not documents:
            return ""
        
        # Group documents by tier
        tier_docs = {"tier1": [], "tier2": [], "tier3": []}
        for doc in documents:
            tier = f"tier{doc.get('tier', 2)}"
            if tier in tier_docs:
                tier_docs[tier].append(doc)
        
        # Build hierarchical context
        context_parts = []
        
        # Tier 1: Foundational Knowledge
        if tier_docs["tier1"]:
            context_parts.append("=== FOUNDATIONAL MEDICAL KNOWLEDGE ===")
            for i, doc in enumerate(tier_docs["tier1"][:3], 1):
                context_parts.append(f"[Foundation {i}] {doc['text']}")
        
        # Tier 2: Clinical Reasoning
        if tier_docs["tier2"]:
            context_parts.append("\n=== CLINICAL REASONING ===")
            for i, doc in enumerate(tier_docs["tier2"][:4], 1):
                context_parts.append(f"[Clinical {i}] {doc['text']}")
        
        # Tier 3: Evidence-Based Medicine
        if tier_docs["tier3"]:
            context_parts.append("\n=== EVIDENCE-BASED MEDICINE ===")
            for i, doc in enumerate(tier_docs["tier3"][:3], 1):
                context_parts.append(f"[Evidence {i}] {doc['text']}")
        
        return "\n".join(context_parts)
    
    def _validate_medical_answer(self, answer: str, query: str, documents: List[Dict]) -> str:
        """Validate medical accuracy and terminology in generated answer."""
        if not self.check_medical_terminology:
            return answer
        
        try:
            # Basic medical terminology validation
            medical_terms = [
                "diagnosis", "treatment", "therapy", "medication", "patient",
                "clinical", "medical", "disease", "condition", "symptom"
            ]
            
            answer_lower = answer.lower()
            query_lower = query.lower()
            
            # Check if medical context is maintained
            query_has_medical = any(term in query_lower for term in medical_terms)
            answer_has_medical = any(term in answer_lower for term in medical_terms)
            
            if query_has_medical and not answer_has_medical:
                logger.warning("⚠️ Medical context may be lost in answer")
            
            # Check for medical disclaimer requirements
            if query_has_medical and len(answer) > 100:
                disclaimer_terms = ["consult", "professional", "physician", "doctor"]
                has_disclaimer = any(term in answer_lower for term in disclaimer_terms)
                
                if not has_disclaimer:
                    # Add basic medical disclaimer
                    answer += "\n\nNote: This information is for educational purposes. Please consult with a healthcare professional for medical advice."
            
            return answer
            
        except Exception as e:
            logger.warning(f"⚠️ Medical validation failed: {e}")
            return answer
    
    def evaluate_single(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question with enhanced medical metrics."""
        question_id = question.get("id", "unknown")
        query = question.get("question", "")
        
        logger.debug(f"🔍 Evaluating question {question_id}: {query[:100]}...")
        
        try:
            # Retrieve documents
            retrieval_start = time.time()
            documents = self.retrieve_documents(query, top_k=10)
            retrieval_time = time.time() - retrieval_start
            
            # Generate answer
            generation_start = time.time()
            answer = self.generate_answer(query, documents)
            generation_time = time.time() - generation_start
            
            # Calculate medical relevance score
            medical_relevance = self._calculate_medical_relevance(query, documents, answer)
            
            # Prepare evaluation result
            result = {
                "question_id": question_id,
                "question": query,
                "generated_answer": answer,
                "retrieved_documents": len(documents),
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": retrieval_time + generation_time,
                "medical_relevance": medical_relevance,
                "tier_distribution": self._analyze_tier_distribution(documents),
                "embedding_model": self.retriever.model_name if self.retriever else "unknown",
                "medical_optimized": self.retriever.is_medical_model if self.retriever else False
            }
            
            # Add ground truth comparison if available
            if "answer" in question:
                result["ground_truth"] = question["answer"]
            
            if "options" in question:
                result["options"] = question["options"]
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed for question {question_id}: {e}")
            return {
                "question_id": question_id,
                "question": query,
                "generated_answer": f"Error: {str(e)}",
                "error": str(e),
                "retrieval_time": 0,
                "generation_time": 0,
                "total_time": 0
            }
    
    def _calculate_medical_relevance(self, query: str, documents: List[Dict], answer: str) -> float:
        """Calculate medical relevance score for the response."""
        try:
            # Medical terminology overlap
            medical_terms = [
                "medical", "clinical", "patient", "treatment", "diagnosis",
                "therapy", "medication", "disease", "condition", "symptom",
                "healthcare", "hospital", "physician", "doctor", "nurse"
            ]
            
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            doc_words = set()
            
            for doc in documents:
                doc_words.update(doc.get("text", "").lower().split())
            
            # Calculate medical term presence
            query_medical = len([term for term in medical_terms if term in query_words])
            answer_medical = len([term for term in medical_terms if term in answer_words])
            doc_medical = len([term for term in medical_terms if term in doc_words])
            
            # Normalize scores
            max_medical = len(medical_terms)
            query_score = min(1.0, query_medical / max_medical)
            answer_score = min(1.0, answer_medical / max_medical)
            doc_score = min(1.0, doc_medical / max_medical)
            
            # Combined relevance score
            relevance = (query_score + answer_score + doc_score) / 3
            
            return float(relevance)
            
        except Exception as e:
            logger.warning(f"⚠️ Medical relevance calculation failed: {e}")
            return 0.5  # Default neutral score
    
    def _analyze_tier_distribution(self, documents: List[Dict]) -> Dict[str, int]:
        """Analyze the distribution of documents across tiers."""
        distribution = {"tier1": 0, "tier2": 0, "tier3": 0, "unknown": 0}
        
        for doc in documents:
            tier = doc.get("tier")
            if tier in [1, 2, 3]:
                distribution[f"tier{tier}"] += 1
            else:
                distribution["unknown"] += 1
        
        return distribution
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics."""
        metrics = super().get_performance_metrics()
        
        # Add hierarchical-specific metrics
        if self.setup_time:
            metrics["setup_time"] = self.setup_time
        
        if self.inference_times:
            metrics.update({
                "avg_inference_time": sum(self.inference_times) / len(self.inference_times),
                "min_inference_time": min(self.inference_times),
                "max_inference_time": max(self.inference_times),
                "total_inferences": len(self.inference_times)
            })
        
        # Add memory usage if available
        if self.retriever:
            try:
                memory_stats = self.retriever.get_memory_usage()
                metrics.update({
                    f"memory_{k}": v for k, v in memory_stats.items() 
                    if isinstance(v, (int, float))
                })
            except:
                pass
        
        return metrics
    
    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        try:
            if self.retriever:
                self.retriever.cleanup()
            
            # Clear references
            self.retriever = None
            self.generator = None
            self.hierarchical_config = None
            
            logger.info("✅ Hierarchical evaluator cleaned up")
            
        except Exception as e:
            logger.warning(f"⚠️ Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass