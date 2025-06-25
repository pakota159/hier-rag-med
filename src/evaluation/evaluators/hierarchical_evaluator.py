#!/usr/bin/env python3
"""
Enhanced Hierarchical Evaluator implementation with Medical Embedding Support
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

            # ADD LOGGING HERE - Debug config paths
            logger.info(f"🔍 EVALUATOR Config data_dir: {self.hierarchical_config.config.get('data_dir', 'NOT_SET')}")
            vector_db_path = self.hierarchical_config.get_data_dir("vector_db")
            logger.info(f"🔍 EVALUATOR Vector DB path: {vector_db_path}")
            logger.info(f"🔍 EVALUATOR Vector DB exists: {vector_db_path.exists()}")
            logger.info(f"🔍 EVALUATOR Current working directory: {Path.cwd()}")

            # Check if collections exist at this path
            if vector_db_path.exists():
                contents = list(vector_db_path.iterdir())
                logger.info(f"🔍 EVALUATOR Vector DB contents: {[str(p.name) for p in contents]}")
            
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
            # Format complete question with options for evaluation
            formatted_query = self._format_question_for_generation(query)
            
            # Generate answer with medical optimizations
            answer = self.generator.generate_answer(formatted_query, documents)
            
            inference_time = time.time() - inference_start
            self.inference_times.append(inference_time)
            
            # Validate medical content if enabled
            if self.medical_validation_enabled:
                answer = self._validate_medical_answer(answer, query, documents)
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}"
    
    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """
        Generate response for a given question using the hierarchical system.
        
        This method implements the abstract method from BaseEvaluator by internally
        calling the HierarchicalGenerator.generate_answer() method, just like debug_gen.py does.
        
        Args:
            question (str): The medical question to answer
            context (Optional[str]): Optional context string (not used in hierarchical approach)
            
        Returns:
            str: Generated answer response
        """
        if not self.generator:
            logger.error("❌ Generator not initialized")
            return "Error: Generator not initialized"
        
        try:
            # Retrieve relevant documents first
            documents = self.retrieve_documents(question, top_k=10)
            
            # Use the existing generate_answer method which calls HierarchicalGenerator.generate_answer()
            response = self.generate_answer(question, documents)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ generate_response failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def _format_question_for_generation(self, query: str) -> str:
        """Format question for generation, preserving options if present."""
        # The query might already be formatted with options from the benchmark
        # Just return as-is since it should include options
        return query
    
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
                
                if not has_disclaimer and "anaphylaxis" not in query_lower:
                    # Add basic medical disclaimer for certain types of questions
                    answer += "\n\nNote: This information is for educational purposes. Consult healthcare professionals for medical advice."
            
            return answer
            
        except Exception as e:
            logger.warning(f"⚠️ Medical validation failed: {e}")
            return answer
    
    def evaluate_single(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question with enhanced medical metrics."""
        question_id = question.get("id", "unknown")
        query = question.get("question", "")
        
        # Format question with options for proper evaluation
        if "options" in question and question["options"]:
            formatted_query = self._format_question_with_options(question)
        else:
            formatted_query = query
        
        logger.debug(f"🔍 Evaluating question {question_id}: {query[:100]}...")
        
        try:
            # Retrieve documents
            retrieval_start = time.time()
            documents = self.retrieve_documents(formatted_query, top_k=10)
            retrieval_time = time.time() - retrieval_start
            
            # Generate answer
            generation_start = time.time()
            answer = self.generate_answer(formatted_query, documents)
            generation_time = time.time() - generation_start
            
            # Extract answer choice
            extracted_answer = self._extract_answer_choice(answer)
            
            # Calculate medical relevance score
            medical_relevance = self._calculate_medical_relevance(query, documents, answer)
            
            # Prepare evaluation result
            result = {
                "question_id": question_id,
                "question": query,
                "formatted_question": formatted_query,
                "generated_answer": answer,
                "extracted_answer": extracted_answer,
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
                correct_answer = question["answer"]
                is_correct = self._compare_answers(extracted_answer, correct_answer)
                result.update({
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                    "accuracy": 1.0 if is_correct else 0.0
                })
            
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
                "total_time": 0,
                "is_correct": False,
                "accuracy": 0.0
            }
    
    def _format_question_with_options(self, question: Dict) -> str:
        """Format question text with options for the LLM."""
        question_text = question['question']
        
        if 'options' not in question or not question['options']:
            return question_text
        
        options = question['options']
        
        # Handle different option formats
        if isinstance(options, dict):
            # Options are in dict format like {'A': 'text', 'B': 'text'}
            formatted_options = []
            for key in sorted(options.keys()):
                formatted_options.append(f"{key}: {options[key]}")
            options_text = "\n".join(formatted_options)
        elif isinstance(options, list):
            # Options are in list format
            formatted_options = []
            for i, option in enumerate(options):
                letter = chr(65 + i)  # A, B, C, D, E
                formatted_options.append(f"{letter}: {option}")
            options_text = "\n".join(formatted_options)
        else:
            return question_text
        
        # Combine question and options
        full_question = f"{question_text}\n\nOptions:\n{options_text}"
        return full_question
    
    def _extract_answer_choice(self, response: str) -> Optional[str]:
        """Extract answer choice from response."""
        import re
        
        # Look for patterns like "Answer: A" or "The answer is B"
        patterns = [
            r'(?:answer|Answer):\s*([A-E])',
            r'(?:answer|Answer)\s+is\s*([A-E])',
            r'\b([A-E])\s*(?:is\s+correct|is\s+the\s+answer)',
            r'(?:^|\s)([A-E])(?:\s*$|\s*\.)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).upper()
        
        return None
    
    def _compare_answers(self, predicted: Optional[str], correct: str) -> bool:
        """Compare predicted and correct answers."""
        if not predicted or not correct:
            return False
        
        return predicted.strip().upper() == correct.strip().upper()
    
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