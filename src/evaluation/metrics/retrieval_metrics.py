"""
Retrieval metrics for medical RAG evaluation.
"""

import math
from typing import Dict, List, Set, Tuple
import numpy as np
from loguru import logger

from .base_metrics import BaseMetrics


class RetrievalMetrics(BaseMetrics):
    """Retrieval evaluation metrics for RAG systems."""
    
    def __init__(self, config: Dict = None):
        """Initialize retrieval metrics."""
        super().__init__(config)
        self.k_values = config.get("precision_at_k", [1, 3, 5, 10]) if config else [1, 3, 5, 10]
        self.calculate_map = config.get("map", True) if config else True
        self.calculate_mrr = config.get("mrr", True) if config else True
        
    def calculate(self, retrieved_lists: List[List[str]], relevant_lists: List[List[str]], **kwargs) -> Dict[str, float]:
        """
        Calculate retrieval metrics.
        
        Args:
            retrieved_lists: List of retrieved document IDs for each query
            relevant_lists: List of relevant document IDs for each query
        """
        if len(retrieved_lists) != len(relevant_lists):
            logger.error("Retrieved and relevant lists must have same length")
            return {"error": "Invalid inputs"}
        
        metrics = {}
        
        # Precision@K
        precision_scores = self._calculate_precision_at_k(retrieved_lists, relevant_lists)
        metrics.update(precision_scores)
        
        # Recall@K
        recall_scores = self._calculate_recall_at_k(retrieved_lists, relevant_lists)
        metrics.update(recall_scores)
        
        # F1@K
        f1_scores = self._calculate_f1_at_k(precision_scores, recall_scores)
        metrics.update(f1_scores)
        
        # NDCG@K
        ndcg_scores = self._calculate_ndcg_at_k(retrieved_lists, relevant_lists)
        metrics.update(ndcg_scores)
        
        # MAP (Mean Average Precision)
        if self.calculate_map:
            map_score = self._calculate_map(retrieved_lists, relevant_lists)
            metrics["map"] = map_score
        
        # MRR (Mean Reciprocal Rank)
        if self.calculate_mrr:
            mrr_score = self._calculate_mrr(retrieved_lists, relevant_lists)
            metrics["mrr"] = mrr_score
        
        # Coverage and diversity metrics
        coverage_metrics = self._calculate_coverage_metrics(retrieved_lists, relevant_lists)
        metrics.update(coverage_metrics)
        
        return metrics
    
    def _calculate_precision_at_k(self, retrieved_lists: List[List[str]], relevant_lists: List[List[str]]) -> Dict[str, float]:
        """Calculate Precision@K for different K values."""
        precision_scores = {}
        
        for k in self.k_values:
            precision_at_k = []
            
            for retrieved, relevant in zip(retrieved_lists, relevant_lists):
                retrieved_at_k = retrieved[:k]
                relevant_set = set(relevant)
                
                if len(retrieved_at_k) == 0:
                    precision_at_k.append(0.0)
                else:
                    relevant_retrieved = len([doc for doc in retrieved_at_k if doc in relevant_set])
                    precision = relevant_retrieved / len(retrieved_at_k)
                    precision_at_k.append(precision)
            
            precision_scores[f"precision_at_{k}"] = np.mean(precision_at_k)
        
        return precision_scores
    
    def _calculate_recall_at_k(self, retrieved_lists: List[List[str]], relevant_lists: List[List[str]]) -> Dict[str, float]:
        """Calculate Recall@K for different K values."""
        recall_scores = {}
        
        for k in self.k_values:
            recall_at_k = []
            
            for retrieved, relevant in zip(retrieved_lists, relevant_lists):
                retrieved_at_k = retrieved[:k]
                relevant_set = set(relevant)
                
                if len(relevant) == 0:
                    recall_at_k.append(1.0)  # Perfect recall when no relevant docs
                else:
                    relevant_retrieved = len([doc for doc in retrieved_at_k if doc in relevant_set])
                    recall = relevant_retrieved / len(relevant)
                    recall_at_k.append(recall)
            
            recall_scores[f"recall_at_{k}"] = np.mean(recall_at_k)
        
        return recall_scores
    
    def _calculate_f1_at_k(self, precision_scores: Dict[str, float], recall_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate F1@K scores from precision and recall."""
        f1_scores = {}
        
        for k in self.k_values:
            precision_key = f"precision_at_{k}"
            recall_key = f"recall_at_{k}"
            
            if precision_key in precision_scores and recall_key in recall_scores:
                precision = precision_scores[precision_key]
                recall = recall_scores[recall_key]
                
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
                
                f1_scores[f"f1_at_{k}"] = f1
        
        return f1_scores
    
    def _calculate_ndcg_at_k(self, retrieved_lists: List[List[str]], relevant_lists: List[List[str]]) -> Dict[str, float]:
        """Calculate NDCG@K (Normalized Discounted Cumulative Gain)."""
        ndcg_scores = {}
        
        for k in self.k_values:
            ndcg_at_k = []
            
            for retrieved, relevant in zip(retrieved_lists, relevant_lists):
                retrieved_at_k = retrieved[:k]
                relevant_set = set(relevant)
                
                # Calculate DCG@K
                dcg = 0.0
                for i, doc in enumerate(retrieved_at_k):
                    relevance = 1.0 if doc in relevant_set else 0.0
                    dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0
                
                # Calculate IDCG@K (Ideal DCG)
                ideal_relevances = [1.0] * min(k, len(relevant)) + [0.0] * max(0, k - len(relevant))
                idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))
                
                # Calculate NDCG@K
                if idcg > 0:
                    ndcg = dcg / idcg
                else:
                    ndcg = 0.0
                
                ndcg_at_k.append(ndcg)
            
            ndcg_scores[f"ndcg_at_{k}"] = np.mean(ndcg_at_k)
        
        return ndcg_scores
    
    def _calculate_map(self, retrieved_lists: List[List[str]], relevant_lists: List[List[str]]) -> float:
        """Calculate Mean Average Precision."""
        average_precisions = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            if not relevant:
                continue
            
            relevant_set = set(relevant)
            precision_sum = 0.0
            relevant_found = 0
            
            for i, doc in enumerate(retrieved):
                if doc in relevant_set:
                    relevant_found += 1
                    precision_at_i = relevant_found / (i + 1)
                    precision_sum += precision_at_i
            
            if relevant_found > 0:
                average_precision = precision_sum / len(relevant)
                average_precisions.append(average_precision)
        
        return np.mean(average_precisions) if average_precisions else 0.0
    
    def _calculate_mrr(self, retrieved_lists: List[List[str]], relevant_lists: List[List[str]]) -> float:
        """Calculate Mean Reciprocal Rank."""
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            relevant_set = set(relevant)
            
            for i, doc in enumerate(retrieved):
                if doc in relevant_set:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    def _calculate_coverage_metrics(self, retrieved_lists: List[List[str]], relevant_lists: List[List[str]]) -> Dict[str, float]:
        """Calculate coverage and diversity metrics."""
        
        # Query coverage: percentage of queries with at least one relevant document retrieved
        queries_with_relevant = 0
        total_relevant_retrieved = 0
        total_relevant_available = 0
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            relevant_set = set(relevant)
            retrieved_relevant = [doc for doc in retrieved if doc in relevant_set]
            
            if retrieved_relevant:
                queries_with_relevant += 1
            
            total_relevant_retrieved += len(retrieved_relevant)
            total_relevant_available += len(relevant)
        
        query_coverage = queries_with_relevant / len(retrieved_lists) if retrieved_lists else 0.0
        
        # Overall recall: total relevant retrieved / total relevant available
        overall_recall = total_relevant_retrieved / total_relevant_available if total_relevant_available > 0 else 0.0
        
        # Average retrieved length
        avg_retrieved_length = np.mean([len(retrieved) for retrieved in retrieved_lists])
        
        # Average relevant length
        avg_relevant_length = np.mean([len(relevant) for relevant in relevant_lists])
        
        return {
            "query_coverage": query_coverage,
            "overall_recall": overall_recall,
            "avg_retrieved_length": avg_retrieved_length,
            "avg_relevant_length": avg_relevant_length
        }
    
    def calculate_retrieval_distribution(self, retrieved_lists: List[List[str]]) -> Dict[str, float]:
        """Calculate distribution statistics for retrieved documents."""
        all_lengths = [len(retrieved) for retrieved in retrieved_lists]
        
        return {
            "mean_retrieved": np.mean(all_lengths),
            "median_retrieved": np.median(all_lengths),
            "std_retrieved": np.std(all_lengths),
            "min_retrieved": np.min(all_lengths),
            "max_retrieved": np.max(all_lengths)
        }
    
    def calculate_rank_correlation(self, retrieved_lists: List[List[str]], 
                                 relevant_lists: List[List[str]], 
                                 relevance_scores: List[List[float]] = None) -> float:
        """Calculate rank correlation between retrieved and ideal ranking."""
        if relevance_scores is None:
            # Binary relevance: relevant=1, not relevant=0
            relevance_scores = []
            for retrieved, relevant in zip(retrieved_lists, relevant_lists):
                relevant_set = set(relevant)
                scores = [1.0 if doc in relevant_set else 0.0 for doc in retrieved]
                relevance_scores.append(scores)
        
        try:
            from scipy.stats import spearmanr
            
            correlations = []
            for retrieved, scores in zip(retrieved_lists, relevance_scores):
                if len(scores) > 1:
                    # Ideal ranking: sort by relevance score (descending)
                    ideal_ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                    actual_ranking = list(range(len(retrieved)))
                    
                    if len(set(scores)) > 1:  # Only calculate if there's variation in scores
                        corr, _ = spearmanr(actual_ranking, ideal_ranking)
                        if not math.isnan(corr):
                            correlations.append(corr)
            
            return np.mean(correlations) if correlations else 0.0
            
        except ImportError:
            logger.warning("scipy not available for rank correlation")
            return 0.0