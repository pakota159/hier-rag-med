"""
Evaluation module for HierRAGMed.
"""

from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config import Config


class Evaluator:
    """Evaluator for RAG system performance."""

    def __init__(self, config: Config):
        """Initialize evaluator."""
        self.config = config
        self.embedding_model = SentenceTransformer(
            config.models["embedding"]["name"],
            device=config.models["embedding"]["device"]
        )
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate_retrieval(
        self,
        retrieved_docs: List[Dict[str, str]],
        relevant_docs: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """Evaluate retrieval performance."""
        # Calculate precision and recall
        retrieved_ids = {doc["metadata"]["doc_id"] for doc in retrieved_docs}
        relevant_ids = {doc["metadata"]["doc_id"] for doc in relevant_docs}
        
        true_positives = len(retrieved_ids.intersection(relevant_ids))
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
        recall = true_positives / len(relevant_ids) if relevant_ids else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate semantic similarity
        retrieved_texts = [doc["text"] for doc in retrieved_docs]
        relevant_texts = [doc["text"] for doc in relevant_docs]

        retrieved_embeddings = self.embedding_model.encode(
            retrieved_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        relevant_embeddings = self.embedding_model.encode(
            relevant_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        similarity_matrix = cosine_similarity(retrieved_embeddings, relevant_embeddings)
        semantic_similarity = np.mean(np.max(similarity_matrix, axis=1))

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "semantic_similarity": semantic_similarity
        }

    def evaluate_generation(
        self,
        generated_text: str,
        reference_text: str
    ) -> Dict[str, float]:
        """Evaluate generation performance."""
        # Calculate ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference_text, generated_text)
        
        # Calculate semantic similarity
        generated_embedding = self.embedding_model.encode(
            generated_text,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        reference_embedding = self.embedding_model.encode(
            reference_text,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        semantic_similarity = cosine_similarity(
            generated_embedding.reshape(1, -1),
            reference_embedding.reshape(1, -1)
        )[0][0]

        return {
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
            "semantic_similarity": semantic_similarity
        }

    def evaluate_rag(
        self,
        query: str,
        retrieved_docs: List[Dict[str, str]],
        generated_text: str,
        reference_docs: List[Dict[str, str]],
        reference_text: str
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate complete RAG pipeline."""
        retrieval_metrics = self.evaluate_retrieval(retrieved_docs, reference_docs)
        generation_metrics = self.evaluate_generation(generated_text, reference_text)

        return {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics
        } 