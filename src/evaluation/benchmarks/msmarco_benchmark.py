# src/evaluation/benchmarks/msmarco_benchmark.py
"""
Updated MS MARCO Benchmark for Medical Passage Retrieval
Focuses on retrieval quality and passage relevance for medical queries
"""

import re
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .base_benchmark import BaseBenchmark
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.data.data_loader import BenchmarkDataLoader


class MSMARCOBenchmark(BaseBenchmark):
    """
    MS MARCO benchmark for evaluating medical passage retrieval quality.
    Focuses on retrieval accuracy, relevance, and ranking performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "MS MARCO"
        self.data_loader = BenchmarkDataLoader(config)
        
        # Initialize evaluation models
        self.similarity_model = None
        self._init_evaluation_models()
        
        # Retrieval-specific configuration
        self.k_values = config.get("k_values", [1, 3, 5, 10])  # For recall@k and precision@k
        self.relevance_threshold = config.get("relevance_threshold", 0.5)
        
        # Medical query patterns
        self.medical_query_patterns = {
            'symptom_based': ['symptoms of', 'signs of', 'what causes', 'why do'],
            'diagnostic': ['how to diagnose', 'diagnostic criteria', 'test for', 'screening'],
            'treatment': ['treatment for', 'how to treat', 'therapy for', 'management of'],
            'prevention': ['prevent', 'prevention', 'avoid', 'reduce risk'],
            'mechanism': ['how does', 'mechanism of', 'pathophysiology', 'why does']
        }
        
        # Passage quality indicators
        self.quality_indicators = {
            'authoritative': ['study', 'research', 'clinical trial', 'guidelines', 'WHO', 'CDC', 'FDA'],
            'specific': ['specifically', 'precisely', 'exactly', 'particular', 'detailed'],
            'current': ['recent', 'latest', 'current', '2020', '2021', '2022', '2023', '2024'],
            'comprehensive': ['comprehensive', 'complete', 'thorough', 'extensive', 'detailed']
        }
    
    def load_dataset(self) -> List[Dict]:
        """Load MS MARCO dataset with medical filtering and enhancement."""
        logger.info(f"🔄 Loading MS MARCO benchmark...")
        
        try:
            max_samples = self.sample_size if not self.is_unlimited else None
            data = self.data_loader.load_benchmark_data("msmarco", max_samples=max_samples)
            
            if data and len(data) > 0:
                # Filter and enhance for medical relevance
                medical_data = []
                for item in data:
                    if self._is_medical_query(item.get("query", "")):
                        enhanced_item = self._enhance_retrieval_item(item)
                        medical_data.append(enhanced_item)
                
                if not medical_data and len(data) > 0:
                    # If no medical queries found, enhance all items anyway
                    logger.warning("No specifically medical queries found, processing all queries")
                    medical_data = [self._enhance_retrieval_item(item) for item in data[:max_samples or len(data)]]
                
                logger.info(f"✅ Loaded {len(medical_data)} MS MARCO queries with medical relevance")
                return medical_data
            else:
                raise ConnectionError("Failed to load MS MARCO benchmark data.")
                
        except Exception as e:
            logger.error(f"❌ Failed to load MS MARCO data: {e}")
            raise e
    
    def _is_medical_query(self, query: str) -> bool:
        """Check if query is medical-related."""
        query_lower = query.lower()
        
        # Medical keywords
        medical_keywords = [
            'health', 'medical', 'disease', 'condition', 'symptom', 'treatment',
            'therapy', 'medicine', 'drug', 'hospital', 'doctor', 'patient',
            'diagnosis', 'syndrome', 'infection', 'cancer', 'diabetes', 'heart',
            'blood', 'brain', 'surgery', 'clinic', 'physician', 'nurse'
        ]
        
        return any(keyword in query_lower for keyword in medical_keywords)
    
    def _enhance_retrieval_item(self, item: Dict) -> Dict:
        """Enhance item with retrieval analysis."""
        enhanced = item.copy()
        
        query = item.get("query", "")
        passages = item.get("passages", [])
        
        enhanced.update({
            "query_type": self._classify_query_type(query),
            "query_complexity": self._assess_query_complexity(query),
            "expected_passages": len(passages),
            "benchmark": "msmarco"
        })
        
        return enhanced
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of medical query."""
        query_lower = query.lower()
        
        for qtype, patterns in self.medical_query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return qtype
        
        return "general_medical"
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess complexity of the query."""
        word_count = len(query.split())
        
        if word_count <= 3:
            return "simple"
        elif word_count <= 7:
            return "moderate"
        else:
            return "complex"
    
    def _init_evaluation_models(self):
        """Initialize models for evaluation metrics."""
        try:
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Initialized similarity model for MS MARCO evaluation")
        except ImportError:
            logger.warning("⚠️ SentenceTransformers not available - some metrics will be limited")
            self.similarity_model = None
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[str]) -> Dict[str, Any]:
        """
        Evaluate retrieval performance using MS MARCO-specific metrics.
        
        Args:
            question: Query data including expected passages
            response: Generated response (not used for retrieval metrics)
            retrieved_docs: List of retrieved passages to evaluate
            
        Returns:
            Dictionary containing retrieval evaluation scores
        """
        logger.debug(f"🔍 Evaluating MS MARCO retrieval for query {question.get('question_id', 'unknown')}")
        
        # Initialize metrics
        metrics = {
            "mrr": 0.0,  # Mean Reciprocal Rank
            "ndcg": 0.0,  # Normalized Discounted Cumulative Gain
            "map": 0.0,   # Mean Average Precision
            "retrieval_accuracy": 0.0,
            "passage_quality": 0.0,
            "relevance_score": 0.0,
            "overall_score": 0.0
        }
        
        # Add recall@k and precision@k for different k values
        for k in self.k_values:
            metrics[f"recall_at_{k}"] = 0.0
            metrics[f"precision_at_{k}"] = 0.0
        
        try:
            query = question.get("query", "")
            expected_passages = question.get("passages", [])
            
            if not retrieved_docs:
                logger.warning("No retrieved documents to evaluate")
                return metrics
            
            # 1. Calculate MRR (Mean Reciprocal Rank)
            metrics["mrr"] = self._calculate_mrr(query, retrieved_docs, expected_passages)
            
            # 2. Calculate NDCG (Normalized Discounted Cumulative Gain)
            metrics["ndcg"] = self._calculate_ndcg(query, retrieved_docs, expected_passages)
            
            # 3. Calculate MAP (Mean Average Precision)
            metrics["map"] = self._calculate_map(query, retrieved_docs, expected_passages)
            
            # 4. Calculate Recall@k and Precision@k
            for k in self.k_values:
                recall_k, precision_k = self._calculate_recall_precision_at_k(
                    query, retrieved_docs, expected_passages, k
                )
                metrics[f"recall_at_{k}"] = recall_k
                metrics[f"precision_at_{k}"] = precision_k
            
            # 5. Retrieval accuracy
            metrics["retrieval_accuracy"] = self._calculate_retrieval_accuracy(
                query, retrieved_docs, expected_passages
            )
            
            # 6. Passage quality assessment
            metrics["passage_quality"] = self._assess_passage_quality(retrieved_docs, query)
            
            # 7. Relevance score
            metrics["relevance_score"] = self._calculate_relevance_score(
                query, retrieved_docs, expected_passages
            )
            
            # 8. Calculate overall score
            metrics["overall_score"] = self._calculate_retrieval_score(metrics)
            
            logger.debug(f"   📊 MS MARCO evaluation complete. Overall score: {metrics['overall_score']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ MS MARCO evaluation failed: {e}")
            
        return metrics
    
    def _calculate_mrr(self, query: str, retrieved_docs: List[str], expected_passages: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not expected_passages:
            return 0.0
        
        for rank, doc in enumerate(retrieved_docs, 1):
            if self._is_relevant_passage(doc, expected_passages, query):
                return 1.0 / rank
        
        return 0.0
    
    def _calculate_ndcg(self, query: str, retrieved_docs: List[str], expected_passages: List[str]) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not expected_passages:
            return 0.0
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for rank, doc in enumerate(retrieved_docs, 1):
            relevance = self._calculate_passage_relevance(doc, expected_passages, query)
            dcg += relevance / np.log2(rank + 1)
        
        # Calculate IDCG (Ideal DCG)
        ideal_relevances = sorted([
            self._calculate_passage_relevance(passage, expected_passages, query) 
            for passage in expected_passages
        ], reverse=True)
        
        idcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(ideal_relevances))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_map(self, query: str, retrieved_docs: List[str], expected_passages: List[str]) -> float:
        """Calculate Mean Average Precision."""
        if not expected_passages:
            return 0.0
        
        relevant_found = 0
        precision_sum = 0.0
        
        for rank, doc in enumerate(retrieved_docs, 1):
            if self._is_relevant_passage(doc, expected_passages, query):
                relevant_found += 1
                precision_at_rank = relevant_found / rank
                precision_sum += precision_at_rank
        
        return precision_sum / len(expected_passages) if expected_passages else 0.0
    
    def _calculate_recall_precision_at_k(self, query: str, retrieved_docs: List[str], 
                                       expected_passages: List[str], k: int) -> Tuple[float, float]:
        """Calculate Recall@k and Precision@k."""
        if not expected_passages:
            return 0.0, 0.0
        
        top_k_docs = retrieved_docs[:k]
        relevant_retrieved = sum(
            1 for doc in top_k_docs 
            if self._is_relevant_passage(doc, expected_passages, query)
        )
        
        recall_k = relevant_retrieved / len(expected_passages)
        precision_k = relevant_retrieved / len(top_k_docs) if top_k_docs else 0.0
        
        return recall_k, precision_k
    
    def _calculate_retrieval_accuracy(self, query: str, retrieved_docs: List[str], 
                                    expected_passages: List[str]) -> float:
        """Calculate overall retrieval accuracy."""
        if not expected_passages or not retrieved_docs:
            return 0.0
        
        # Check if at least one relevant passage is in top-5
        top_5 = retrieved_docs[:5]
        has_relevant = any(
            self._is_relevant_passage(doc, expected_passages, query) 
            for doc in top_5
        )
        
        return 1.0 if has_relevant else 0.0
    
    def _assess_passage_quality(self, retrieved_docs: List[str], query: str) -> float:
        """Assess quality of retrieved passages."""
        if not retrieved_docs:
            return 0.0
        
        quality_scores = []
        
        for doc in retrieved_docs[:5]:  # Evaluate top 5
            score = 0.0
            doc_lower = doc.lower()
            
            # Check for quality indicators
            for category, indicators in self.quality_indicators.items():
                category_score = sum(0.1 for indicator in indicators if indicator in doc_lower)
                score += min(0.25, category_score)  # Cap each category
            
            # Length consideration
            word_count = len(doc.split())
            if 50 <= word_count <= 500:  # Appropriate length
                score += 0.2
            elif word_count < 20:  # Too short
                score -= 0.1
            
            # Query relevance
            query_words = set(query.lower().split())
            doc_words = set(doc_lower.split())
            overlap = len(query_words.intersection(doc_words)) / len(query_words) if query_words else 0
            score += overlap * 0.3
            
            quality_scores.append(min(1.0, score))
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _calculate_relevance_score(self, query: str, retrieved_docs: List[str], 
                                 expected_passages: List[str]) -> float:
        """Calculate overall relevance score."""
        if not retrieved_docs:
            return 0.0
        
        # Use semantic similarity if available
        if self.similarity_model and expected_passages:
            similarity_scores = []
            for doc in retrieved_docs[:5]:
                max_sim = 0.0
                for expected in expected_passages:
                    try:
                        embeddings = self.similarity_model.encode([doc, expected])
                        sim = np.dot(embeddings[0], embeddings[1]) / (
                            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                        )
                        max_sim = max(max_sim, sim)
                    except:
                        continue
                similarity_scores.append(max_sim)
            
            return np.mean(similarity_scores) if similarity_scores else 0.0
        
        # Fallback to keyword overlap
        relevance_scores = []
        for doc in retrieved_docs[:5]:
            max_relevance = 0.0
            for expected in expected_passages:
                relevance = self._calculate_passage_relevance(doc, [expected], query)
                max_relevance = max(max_relevance, relevance)
            relevance_scores.append(max_relevance)
        
        return np.mean(relevance_scores) if relevance_scores else 0.0
    
    def _is_relevant_passage(self, passage: str, expected_passages: List[str], query: str) -> bool:
        """Check if passage is relevant to expected passages."""
        relevance_score = self._calculate_passage_relevance(passage, expected_passages, query)
        return relevance_score >= self.relevance_threshold
    
    def _calculate_passage_relevance(self, passage: str, expected_passages: List[str], query: str) -> float:
        """Calculate relevance score for a passage."""
        if not expected_passages:
            # Fallback: check relevance to query
            return self._calculate_query_relevance(passage, query)
        
        max_relevance = 0.0
        for expected in expected_passages:
            # Calculate word overlap
            passage_words = set(passage.lower().split())
            expected_words = set(expected.lower().split())
            
            if expected_words:
                overlap = len(passage_words.intersection(expected_words)) / len(expected_words)
                max_relevance = max(max_relevance, overlap)
        
        return max_relevance
    
    def _calculate_query_relevance(self, passage: str, query: str) -> float:
        """Calculate relevance of passage to query."""
        passage_words = set(passage.lower().split())
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(passage_words.intersection(query_words)) / len(query_words)
        return overlap
    
    def _calculate_retrieval_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall retrieval score."""
        weights = {
            "mrr": 0.25,
            "ndcg": 0.20,
            "map": 0.15,
            "retrieval_accuracy": 0.15,
            "passage_quality": 0.10,
            "relevance_score": 0.10,
            "recall_at_5": 0.05
        }
        
        overall = sum(metrics.get(metric, 0.0) * weight for metric, weight in weights.items())
        return max(0.0, min(1.0, overall))
    
    def get_evaluation_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for MS MARCO evaluation."""
        if not results:
            return {"error": "No results to summarize"}
        
        # Core retrieval metrics
        retrieval_metrics = ["mrr", "ndcg", "map", "retrieval_accuracy", 
                           "passage_quality", "relevance_score", "overall_score"]
        
        # Recall@k and Precision@k metrics
        recall_precision_metrics = []
        for k in self.k_values:
            recall_precision_metrics.extend([f"recall_at_{k}", f"precision_at_{k}"])
        
        all_metrics = retrieval_metrics + recall_precision_metrics
        
        summary = {
            "total_queries": len(results),
            "average_scores": {},
            "retrieval_performance": {},
            "performance_by_query_type": {},
            "benchmark": "MS MARCO"
        }
        
        # Calculate averages
        for metric in all_metrics:
            scores = [r.get(metric, 0.0) for r in results]
            summary["average_scores"][metric] = np.mean(scores) if scores else 0.0
        
        # Retrieval performance analysis
        summary["retrieval_performance"] = {
            "queries_with_relevant_results": sum(1 for r in results if r.get("retrieval_accuracy", 0.0) > 0),
            "average_mrr": summary["average_scores"]["mrr"],
            "average_ndcg": summary["average_scores"]["ndcg"],
            "top_k_performance": {
                f"recall_at_{k}": summary["average_scores"][f"recall_at_{k}"] 
                for k in self.k_values
            }
        }
        
        # Performance by query type
        query_types = set(r.get("query_type", "unknown") for r in results)
        for qtype in query_types:
            type_results = [r for r in results if r.get("query_type") == qtype]
            if type_results:
                type_scores = [r.get("overall_score", 0.0) for r in type_results]
                summary["performance_by_query_type"][qtype] = {
                    "count": len(type_results),
                    "average_score": np.mean(type_scores),
                    "mrr": np.mean([r.get("mrr", 0.0) for r in type_results]),
                    "retrieval_success_rate": np.mean([r.get("retrieval_accuracy", 0.0) for r in type_results])
                }
        
        # Query complexity analysis
        complexity_types = ["simple", "moderate", "complex"]
        summary["performance_by_complexity"] = {}
        for complexity in complexity_types:
            complexity_results = [r for r in results if r.get("query_complexity") == complexity]
            if complexity_results:
                complexity_scores = [r.get("overall_score", 0.0) for r in complexity_results]
                summary["performance_by_complexity"][complexity] = {
                    "count": len(complexity_results),
                    "average_score": np.mean(complexity_scores),
                    "retrieval_accuracy": np.mean([r.get("retrieval_accuracy", 0.0) for r in complexity_results])
                }
        
        return summary