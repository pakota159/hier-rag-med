"""
MS MARCO Medical benchmark implementation for passage retrieval evaluation.
"""

import json
import math
from typing import Dict, List, Tuple
from pathlib import Path
from loguru import logger

from .base_benchmark import BaseBenchmark


class MSMARCOBenchmark(BaseBenchmark):
    """MS MARCO Medical benchmark for passage retrieval evaluation."""
    
    def __init__(self, config: Dict):
        """Initialize MS MARCO benchmark."""
        super().__init__(config)
        self.dataset_url = config.get("dataset_url", "")
        self.collection = config.get("collection", "medical_passages")
    
    def load_dataset(self) -> List[Dict]:
        """Load MS MARCO Medical dataset."""
        try:
            # Try to load from cache
            cache_path = Path("data/evaluation/cache/msmarco_dataset.json")
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded MS MARCO dataset from cache: {len(data)} queries")
                return data
            
            # Generate sample data for now
            logger.warning("Using sample MS MARCO data - replace with actual dataset")
            return self._generate_sample_data()
            
        except Exception as e:
            logger.error(f"Failed to load MS MARCO dataset: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> List[Dict]:
        """Generate sample MS MARCO-style queries."""
        medical_queries = [
            {
                "query": "symptoms of diabetes mellitus",
                "relevant_passages": [
                    "Type 2 diabetes symptoms include increased thirst, frequent urination, increased hunger, weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections.",
                    "Diabetes mellitus is characterized by hyperglycemia resulting from defects in insulin secretion, insulin action, or both."
                ],
                "passage_ids": ["med_001", "med_002"],
                "query_type": "symptom_lookup"
            },
            {
                "query": "treatment for hypertension first line",
                "relevant_passages": [
                    "First-line antihypertensive medications include ACE inhibitors, ARBs, thiazide or thiazide-like diuretics, and calcium channel blockers.",
                    "Lifestyle modifications including diet, exercise, and weight loss are first-line treatments for hypertension."
                ],
                "passage_ids": ["med_003", "med_004"],
                "query_type": "treatment_lookup"
            },
            {
                "query": "myocardial infarction diagnosis criteria",
                "relevant_passages": [
                    "Myocardial infarction diagnosis requires elevation of cardiac biomarkers (preferably troponin) with at least one of: symptoms of ischemia, new ECG changes, or imaging evidence.",
                    "The universal definition of MI includes troponin elevation above the 99th percentile upper reference limit with clinical evidence of myocardial ischemia."
                ],
                "passage_ids": ["med_005", "med_006"],
                "query_type": "diagnostic_criteria"
            },
            {
                "query": "side effects of statins",
                "relevant_passages": [
                    "Common statin side effects include muscle pain, liver enzyme elevation, digestive problems, and rarely rhabdomyolysis.",
                    "Statin-associated muscle symptoms (SAMS) occur in 5-10% of patients and may require dose adjustment or alternative therapy."
                ],
                "passage_ids": ["med_007", "med_008"],
                "query_type": "adverse_effects"
            },
            {
                "query": "heart failure classification NYHA",
                "relevant_passages": [
                    "NYHA Class I: No limitation of physical activity. Ordinary physical activity does not cause symptoms.",
                    "NYHA Class II: Slight limitation of physical activity. Comfortable at rest but ordinary activity results in symptoms.",
                    "NYHA Class III: Marked limitation of physical activity. Less than ordinary activity causes symptoms.",
                    "NYHA Class IV: Unable to carry on any physical activity without symptoms. Symptoms may be present at rest."
                ],
                "passage_ids": ["med_009", "med_010", "med_011", "med_012"],
                "query_type": "classification"
            }
        ]
        
        # Replicate for sample size
        queries = medical_queries * (self.sample_size // len(medical_queries) + 1)
        
        # Add unique IDs
        for i, q in enumerate(queries[:self.sample_size]):
            q["id"] = f"msmarco_{i:04d}"
        
        return queries[:self.sample_size]
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate retrieval performance for MS MARCO."""
        
        # Extract query information
        query = question.get("query", "")
        relevant_passage_ids = question.get("passage_ids", [])
        query_type = question.get("query_type", "unknown")
        
        # Extract retrieved document IDs
        retrieved_ids = [doc.get("metadata", {}).get("doc_id", f"doc_{i}") 
                        for i, doc in enumerate(retrieved_docs)]
        
        # Calculate retrieval metrics
        precision_scores = self._calculate_precision_at_k(retrieved_ids, relevant_passage_ids)
        recall_scores = self._calculate_recall_at_k(retrieved_ids, relevant_passage_ids)
        ndcg_scores = self._calculate_ndcg_at_k(retrieved_ids, relevant_passage_ids)
        
        # Calculate MAP and MRR
        map_score = self._calculate_map(retrieved_ids, relevant_passage_ids)
        mrr_score = self._calculate_mrr(retrieved_ids, relevant_passage_ids)
        
        # Overall retrieval score
        overall_score = (ndcg_scores.get(10, 0) * 0.4 + 
                        precision_scores.get(5, 0) * 0.3 + 
                        map_score * 0.2 + 
                        mrr_score * 0.1)
        
        return {
            "question_id": question.get("id"),
            "query": query,
            "query_type": query_type,
            "score": overall_score * 100,
            "correct": overall_score > 0.3,  # Lower threshold for retrieval
            "metrics": {
                "precision_at_k": precision_scores,
                "recall_at_k": recall_scores,
                "ndcg_at_k": ndcg_scores,
                "map": map_score,
                "mrr": mrr_score
            },
            "retrieved_count": len(retrieved_docs),
            "relevant_count": len(relevant_passage_ids)
        }
    
    def _calculate_precision_at_k(self, retrieved: List[str], relevant: List[str]) -> Dict[int, float]:
        """Calculate Precision@K for different K values."""
        k_values = [1, 3, 5, 10]
        precision_scores = {}
        
        for k in k_values:
            if k > len(retrieved):
                precision_scores[k] = 0.0
                continue
                
            retrieved_at_k = retrieved[:k]
            relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
            precision_scores[k] = relevant_retrieved / k
        
        return precision_scores
    
    def _calculate_recall_at_k(self, retrieved: List[str], relevant: List[str]) -> Dict[int, float]:
        """Calculate Recall@K for different K values."""
        k_values = [1, 3, 5, 10]
        recall_scores = {}
        
        if not relevant:
            return {k: 0.0 for k in k_values}
        
        for k in k_values:
            retrieved_at_k = retrieved[:k]
            relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
            recall_scores[k] = relevant_retrieved / len(relevant)
        
        return recall_scores
    
    def _calculate_ndcg_at_k(self, retrieved: List[str], relevant: List[str]) -> Dict[int, float]:
        """Calculate NDCG@K for different K values."""
        k_values = [1, 3, 5, 10]
        ndcg_scores = {}
        
        for k in k_values:
            if k > len(retrieved):
                ndcg_scores[k] = 0.0
                continue
                
            # Calculate DCG@K
            dcg = 0.0
            for i in range(min(k, len(retrieved))):
                relevance = 1.0 if retrieved[i] in relevant else 0.0
                dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) is 0
            
            # Calculate IDCG@K (ideal DCG)
            ideal_relevances = [1.0] * min(k, len(relevant)) + [0.0] * max(0, k - len(relevant))
            idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))
            
            # Calculate NDCG@K
            ndcg_scores[k] = dcg / idcg if idcg > 0 else 0.0
        
        return ndcg_scores
    
    def _calculate_map(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Mean Average Precision."""
        if not relevant:
            return 0.0
        
        precision_sum = 0.0
        relevant_found = 0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant) if relevant else 0.0
    
    def _calculate_mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0