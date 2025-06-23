"""
Hierarchical retrieval system for basic reasoning.
Updated to handle unique ID generation and prevent ChromaDB collisions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import hashlib

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

    def generate_unique_id(self, tier_name: str, doc_metadata: Dict, index: int) -> str:
        """
        Generate a truly unique ID for ChromaDB to prevent collisions.
        Uses multiple fallback strategies to ensure uniqueness.
        """
        # Strategy 1: Use unique_id if available
        if "unique_id" in doc_metadata:
            return f"{tier_name}_{doc_metadata['unique_id']}"
        
        # Strategy 2: Use tier_chunk_id if available
        if "tier_chunk_id" in doc_metadata:
            doc_id = doc_metadata.get("doc_id", "doc")
            return f"{tier_name}_{doc_id}_{doc_metadata['tier_chunk_id']}"
        
        # Strategy 3: Create hash-based ID from content and metadata
        doc_id = doc_metadata.get("doc_id", "doc")
        chunk_id = doc_metadata.get("chunk_id", 0)
        source = doc_metadata.get("source", "unknown")
        
        # Create a unique string to hash
        unique_string = f"{tier_name}_{doc_id}_{chunk_id}_{source}_{index}"
        hash_suffix = hashlib.md5(unique_string.encode()).hexdigest()[:8]
        
        return f"{tier_name}_{doc_id}_{chunk_id}_{hash_suffix}"

    def add_documents_to_tiers(self, organized_docs: Dict[str, List[Dict]]) -> None:
        """Add documents to appropriate tier collections with unique ID generation."""
        if not all([self.tier1_collection, self.tier2_collection, self.tier3_collection]):
            raise ValueError("Collections not initialized. Call create_hierarchical_collections first.")

        tier_collections = {
            "pattern_recognition": self.tier1_collection,
            "hypothesis_testing": self.tier2_collection,
            "confirmation": self.tier3_collection
        }

        for tier_name, documents in organized_docs.items():
            if not documents:
                logger.warning(f"⚠️ No documents found for tier: {tier_name}")
                continue
                
            collection = tier_collections[tier_name]
            
            # Prepare documents for batch processing
            texts = []
            metadatas = []
            ids = []
            used_ids = set()  # Track used IDs to prevent duplicates

            for i, doc in enumerate(tqdm(documents, desc=f"Preparing {tier_name} documents")):
                texts.append(doc["text"])
                
                # Clean metadata for ChromaDB compatibility
                cleaned_metadata = self.clean_metadata(doc["metadata"])
                metadatas.append(cleaned_metadata)
                
                # Generate unique ID
                unique_id = self.generate_unique_id(tier_name, doc["metadata"], i)
                
                # Ensure ID is truly unique within this batch
                counter = 0
                original_id = unique_id
                while unique_id in used_ids:
                    counter += 1
                    unique_id = f"{original_id}_{counter}"
                
                used_ids.add(unique_id)
                ids.append(unique_id)

            logger.info(f"📝 Generated {len(ids)} unique IDs for {tier_name}")
            
            # Verify no duplicates
            if len(set(ids)) != len(ids):
                duplicates = len(ids) - len(set(ids))
                raise ValueError(f"❌ Found {duplicates} duplicate IDs in {tier_name} - this should not happen!")

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

            # Add to collection in batches to avoid ChromaDB limits
            chroma_batch_size = 5000  # ChromaDB safe batch size
            for i in tqdm(range(0, len(texts), chroma_batch_size), desc=f"Adding {tier_name} to ChromaDB"):
                batch_end = min(i + chroma_batch_size, len(texts))
                
                try:
                    collection.add(
                        embeddings=embeddings[i:batch_end],
                        documents=texts[i:batch_end],
                        metadatas=metadatas[i:batch_end],
                        ids=ids[i:batch_end]
                    )
                except Exception as e:
                    logger.error(f"❌ Error adding batch {i//chroma_batch_size + 1} to {tier_name}: {e}")
                    # Log the problematic IDs for debugging
                    batch_ids = ids[i:batch_end]
                    duplicate_ids = [id for id in batch_ids if batch_ids.count(id) > 1]
                    if duplicate_ids:
                        logger.error(f"Duplicate IDs in batch: {duplicate_ids[:10]}")
                    raise
            
            logger.info(f"✅ Added {len(documents)} documents to {tier_name}")

    def clean_metadata(self, metadata: Dict) -> Dict:
        """Clean metadata to ensure ChromaDB compatibility."""
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                cleaned[key] = value
            elif isinstance(value, list):
                # Convert lists to semicolon-separated strings
                cleaned[key] = "; ".join(str(item) for item in value) if value else ""
            else:
                # Convert other types to strings
                cleaned[key] = str(value)
        return cleaned

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

    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics for all collections."""
        stats = {}
        if self.tier1_collection:
            stats["tier1_pattern_recognition"] = self.tier1_collection.count()
        if self.tier2_collection:
            stats["tier2_hypothesis_testing"] = self.tier2_collection.count()
        if self.tier3_collection:
            stats["tier3_confirmation"] = self.tier3_collection.count()
        return stats