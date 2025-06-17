"""
Question answering metrics for medical RAG evaluation.
"""

from typing import Dict, List, Optional
import numpy as np
from loguru import logger

from .base_metrics import BaseMetrics


class QAMetrics(BaseMetrics):
    """Question answering evaluation metrics."""
    
    def __init__(self, config: Dict = None):
        """Initialize QA metrics."""
        super().__init__(config)
        self.rouge_variants = config.get("rouge_variants", ["rouge1", "rouge2", "rougeL"]) if config else ["rouge1", "rouge2", "rougeL"]
        self.use_bleu = config.get("bleu", True) if config else True
        self.use_bertscore = config.get("bertscore", True) if config else True
        
    def calculate(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """Calculate QA metrics."""
        if not self.validate_inputs(predictions, references):
            return {"error": "Invalid inputs"}
        
        metrics = {}
        
        # ROUGE scores
        rouge_scores = self._calculate_rouge(predictions, references)
        metrics.update(rouge_scores)
        
        # BLEU score
        if self.use_bleu:
            bleu_score = self._calculate_bleu(predictions, references)
            metrics["bleu"] = bleu_score
        
        # BERTScore
        if self.use_bertscore:
            bertscore = self._calculate_bertscore(predictions, references)
            metrics.update(bertscore)
        
        # Exact match
        exact_match = self._calculate_exact_match(predictions, references)
        metrics["exact_match"] = exact_match
        
        # Semantic similarity
        semantic_sim = self._calculate_semantic_similarity(predictions, references)
        metrics["semantic_similarity"] = semantic_sim
        
        # Length-based metrics
        length_metrics = self._calculate_length_metrics(predictions, references)
        metrics.update(length_metrics)
        
        return self.normalize_scores(metrics)
    
    def _calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(self.rouge_variants, use_stemmer=True)
            rouge_scores = {variant: [] for variant in self.rouge_variants}
            
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                for variant in self.rouge_variants:
                    rouge_scores[variant].append(scores[variant].fmeasure)
            
            # Calculate average scores
            avg_rouge = {}
            for variant in self.rouge_variants:
                avg_rouge[f"rouge_{variant}"] = np.mean(rouge_scores[variant])
            
            return avg_rouge
            
        except ImportError:
            logger.warning("rouge_score not available, using simple ROUGE approximation")
            return self._simple_rouge_approximation(predictions, references)
    
    def _simple_rouge_approximation(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Simple ROUGE approximation using word overlap."""
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # ROUGE-1 (unigram overlap)
            if ref_words:
                overlap = len(set(pred_words).intersection(set(ref_words)))
                rouge1 = overlap / len(set(ref_words))
                rouge1_scores.append(rouge1)
            
            # ROUGE-2 (bigram overlap)
            pred_bigrams = set(zip(pred_words[:-1], pred_words[1:]))
            ref_bigrams = set(zip(ref_words[:-1], ref_words[1:]))
            if ref_bigrams:
                overlap = len(pred_bigrams.intersection(ref_bigrams))
                rouge2 = overlap / len(ref_bigrams)
                rouge2_scores.append(rouge2)
            
            # ROUGE-L (longest common subsequence)
            lcs_length = self._lcs_length(pred_words, ref_words)
            if ref_words:
                rougeL = lcs_length / len(ref_words)
                rougeL_scores.append(rougeL)
        
        return {
            "rouge_rouge1": np.mean(rouge1_scores) if rouge1_scores else 0,
            "rouge_rouge2": np.mean(rouge2_scores) if rouge2_scores else 0,
            "rouge_rougeL": np.mean(rougeL_scores) if rougeL_scores else 0
        }
    
    def _calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import nltk
            nltk.download('punkt', quiet=True)
            
            bleu_scores = []
            smoothing = SmoothingFunction().method1
            
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.lower().split()
                ref_tokens = [ref.lower().split()]  # BLEU expects list of reference lists
                
                if pred_tokens and ref_tokens[0]:
                    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                    bleu_scores.append(score)
            
            return np.mean(bleu_scores) if bleu_scores else 0.0
            
        except ImportError:
            logger.warning("NLTK not available, using simple BLEU approximation")
            return self._simple_bleu_approximation(predictions, references)
    
    def _simple_bleu_approximation(self, predictions: List[str], references: List[str]) -> float:
        """Simple BLEU approximation using precision."""
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            if pred_words and ref_words:
                # Simple precision-based approximation
                overlap = len(set(pred_words).intersection(set(ref_words)))
                precision = overlap / len(pred_words)
                bleu_scores.append(precision)
        
        return np.mean(bleu_scores) if bleu_scores else 0.0
    
    def _calculate_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore."""
        try:
            from bert_score import score
            
            P, R, F1 = score(predictions, references, lang="en", verbose=False)
            
            return {
                "bertscore_precision": P.mean().item(),
                "bertscore_recall": R.mean().item(),
                "bertscore_f1": F1.mean().item()
            }
            
        except ImportError:
            logger.warning("bert_score not available, using semantic similarity approximation")
            return {"bertscore_f1": self._calculate_semantic_similarity(predictions, references)}
    
    def _calculate_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """Calculate exact match accuracy."""
        exact_matches = []
        
        for pred, ref in zip(predictions, references):
            # Normalize for comparison
            pred_norm = pred.lower().strip()
            ref_norm = ref.lower().strip()
            exact_matches.append(1.0 if pred_norm == ref_norm else 0.0)
        
        return np.mean(exact_matches)
    
    def _calculate_semantic_similarity(self, predictions: List[str], references: List[str]) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Generate embeddings
            pred_embeddings = model.encode(predictions)
            ref_embeddings = model.encode(references)
            
            # Calculate pairwise cosine similarities
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
                similarities.append(sim)
            
            return np.mean(similarities)
            
        except ImportError:
            logger.warning("sentence_transformers not available, using simple word overlap")
            return self._simple_similarity(predictions, references)
    
    def _simple_similarity(self, predictions: List[str], references: List[str]) -> float:
        """Simple similarity based on word overlap."""
        similarities = []
        
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if pred_words or ref_words:
                intersection = len(pred_words.intersection(ref_words))
                union = len(pred_words.union(ref_words))
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_length_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate length-based metrics."""
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        # Length ratio
        length_ratios = []
        for pred_len, ref_len in zip(pred_lengths, ref_lengths):
            if ref_len > 0:
                ratio = pred_len / ref_len
                length_ratios.append(ratio)
        
        return {
            "avg_prediction_length": np.mean(pred_lengths),
            "avg_reference_length": np.mean(ref_lengths),
            "length_ratio": np.mean(length_ratios) if length_ratios else 0.0,
            "length_difference": np.mean([abs(p - r) for p, r in zip(pred_lengths, ref_lengths)])
        }
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]