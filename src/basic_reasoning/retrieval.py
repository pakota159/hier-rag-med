"""
Hierarchical retrieval system for basic reasoning.
Implements three-tier architecture using foundation datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import Config


class HierarchicalRetriever:
    """Three-tier hierarchical retriever for diagnostic reasoning."""

    def __init__(self, config: Config):
        """Initialize hierarchical retriever."""
        self.config = config
        
        embedding_config = config.config["models"]["embedding"]
        
        self.embedding_model = SentenceTransformer(
            embedding_config["name"],
            device=embedding_config["device"]
        )
        
        self.client = chromadb.PersistentClient(
            path=str(config.get_data_dir("vector_db")),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Three separate collections for three tiers
        self.tier1_collection = None  # Pattern Recognition
        self.tier2_collection = None  # Hypothesis Testing
        self.tier3_collection = None  # Confirmation

    def create_hierarchical_collections(self) -> None:
        """Create three-tier collections."""
        collection_names = [
            "tier1_pattern_recognition",
            "tier2_hypothesis_testing", 
            "tier3_confirmation"
        ]
        
        # Delete existing collections
        for name in collection_names:
            try:
                self.client.delete_collection(name)
                logger.info(f"Deleted existing collection: {name}")
            except Exception:
                pass
        
        # Create new collections
        self.tier1_collection = self.client.create_collection(
            name="tier1_pattern_recognition",
            metadata={"hnsw:space": "cosine"}
        )
        self.tier2_collection = self.client.create_collection(
            name="tier2_hypothesis_testing",
            metadata={"hnsw:space": "cosine"}
        )
        self.tier3_collection = self.client.create_collection(
            name="tier3_confirmation",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("✅ Created hierarchical collections")

    def load_hierarchical_collections(self) -> None:
        """Load existing hierarchical collections."""
        try:
            self.tier1_collection = self.client.get_collection("tier1_pattern_recognition")
            self.tier2_collection = self.client.get_collection("tier2_hypothesis_testing")
            self.tier3_collection = self.client.get_collection("tier3_confirmation")
            logger.info("✅ Loaded hierarchical collections")
        except Exception as e:
            logger.error(f"Collections not found: {e}")
            raise ValueError("Hierarchical collections do not exist. Create them first.")

    def add_documents_to_tiers(self, organized_docs: Dict[str, List[Dict]]) -> None:
        """Add documents to appropriate tier collections."""
        if not all([self.tier1_collection, self.tier2_collection, self.tier3_collection]):
            raise ValueError("Collections not initialized. Call create_hierarchical_collections first.")

        tier_collections = {
            "pattern_recognition": self.tier1_collection,
            "hypothesis_testing": self.tier2_collection,
            "confirmation": self.tier3_collection
        }

        for tier_name, documents in organized_docs.items():
            if not documents:
                continue
                
            collection = tier_collections[tier_name]
            
            # Prepare documents for batch processing
            texts = []
            metadatas = []
            ids = []

            for doc in tqdm(documents, desc=f"Preparing {tier_name} documents"):
                texts.append(doc["text"])
                metadatas.append(doc["metadata"])
                ids.append(f"{tier_name}_{doc['metadata']['doc_id']}_{doc['metadata']['chunk_id']}")

            # Generate embeddings in batches
            batch_size = self.config.config["models"]["embedding"]["batch_size"]
            embeddings = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Generating {tier_name} embeddings"):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings.extend(batch_embeddings)

            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"✅ Added {len(documents)} documents to {tier_name}")

    def hierarchical_search(self, query: str) -> Dict[str, List[Dict]]:
        """Perform three-tier hierarchical search."""
        if not all([self.tier1_collection, self.tier2_collection, self.tier3_collection]):
            raise ValueError("Collections not loaded.")

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        retrieval_config = self.config.config["hierarchical_retrieval"]
        
        results = {}
        
        # Tier 1: Pattern Recognition - Fast initial screening
        tier1_results = self.tier1_collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieval_config["tier1_top_k"]
        )
        results["tier1_patterns"] = self._format_results(tier1_results)
        
        # Tier 2: Hypothesis Testing - Reasoning chains
        tier2_results = self.tier2_collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieval_config["tier2_top_k"]
        )
        results["tier2_hypotheses"] = self._format_results(tier2_results)
        
        # Tier 3: Confirmation - Clinical evidence
        tier3_results = self.tier3_collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieval_config["tier3_top_k"]
        )
        results["tier3_confirmation"] = self._format_results(tier3_results)
        
        return results

    def _format_results(self, results) -> List[Dict[str, Union[str, float]]]:
        """Format ChromaDB results."""
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - float(results["distances"][0][i])
            })
        return formatted_results

    def combined_search(self, query: str, n_results: int = 10) -> List[Dict]:
        """Traditional combined search across all tiers (fallback)."""
        hierarchical_results = self.hierarchical_search(query)
        
        # Combine all tier results
        all_results = []
        all_results.extend(hierarchical_results["tier1_patterns"])
        all_results.extend(hierarchical_results["tier2_hypotheses"])
        all_results.extend(hierarchical_results["tier3_confirmation"])
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:n_results]