"""
Retrieval module for Basic Reasoning system.
Implements hierarchical medical knowledge retrieval.
"""

from typing import Dict, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from loguru import logger

from .config import Config


class HierarchicalRetriever:
    """Hierarchical retriever for medical knowledge."""

    def __init__(self, config: Config):
        """Initialize hierarchical retriever."""
        self.config = config
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(config.get_data_dir("vector_db"))
        )
        
        # Initialize embedding model
        embedding_config = config.config["models"]["embedding"]
        self.embedding_model = SentenceTransformer(
            embedding_config["name"],
            device=embedding_config["device"]
        )
        
        # Retrieval configurations
        retrieval_config = config.config["hierarchical_retrieval"]
        self.tier1_top_k = retrieval_config["tier1_top_k"]
        self.tier2_top_k = retrieval_config["tier2_top_k"]
        self.tier3_top_k = retrieval_config["tier3_top_k"]
        
        # Collections
        self.tier1_collection = None
        self.tier2_collection = None
        self.tier3_collection = None

    def create_hierarchical_collections(self):
        """Create hierarchical collections for medical knowledge."""
        logger.info("🔧 Creating hierarchical medical knowledge collections")
        
        collection_names = [
            "tier1_pattern_recognition",
            "tier2_hypothesis_testing", 
            "tier3_confirmation"
        ]
        
        # Delete existing collections
        for name in collection_names:
            try:
                self.client.delete_collection(name)
                logger.info(f"🗑️ Deleted existing collection: {name}")
            except:
                pass
        
        # Create new collections
        self.tier1_collection = self.client.create_collection(
            name="tier1_pattern_recognition",
            metadata={"description": "Basic medical concepts, anatomy, definitions"}
        )
        
        self.tier2_collection = self.client.create_collection(
            name="tier2_hypothesis_testing",
            metadata={"description": "Clinical reasoning, pathophysiology, diagnostics"}
        )
        
        self.tier3_collection = self.client.create_collection(
            name="tier3_confirmation", 
            metadata={"description": "Evidence-based medicine, guidelines, clinical trials"}
        )
        
        logger.info("✅ Created hierarchical medical collections")

    def load_hierarchical_collections(self):
        """Load existing hierarchical collections."""
        try:
            self.tier1_collection = self.client.get_collection("tier1_pattern_recognition")
            self.tier2_collection = self.client.get_collection("tier2_hypothesis_testing")
            self.tier3_collection = self.client.get_collection("tier3_confirmation")
            
            # Verify collections have content
            tier1_count = self.tier1_collection.count()
            tier2_count = self.tier2_collection.count()
            tier3_count = self.tier3_collection.count()
            
            logger.info(f"📚 Loaded hierarchical collections:")
            logger.info(f"   Tier 1 (Basic Medical): {tier1_count} documents")
            logger.info(f"   Tier 2 (Clinical): {tier2_count} documents")
            logger.info(f"   Tier 3 (Evidence): {tier3_count} documents")
            
            if tier1_count == 0 or tier2_count == 0 or tier3_count == 0:
                raise ValueError("One or more collections are empty")
                
        except Exception as e:
            logger.error(f"❌ Failed to load hierarchical collections: {e}")
            raise

    def add_documents_to_tiers(self, organized_docs: Dict[str, List[Dict]]):
        """Add documents to appropriate tier collections."""
        logger.info("📝 Adding documents to hierarchical medical tiers")
        
        # ChromaDB batch size limit
        max_batch_size = 5000
        
        # Add to Tier 1 - Basic Medical Knowledge
        tier1_docs = organized_docs["pattern_recognition"]
        if tier1_docs:
            self._add_documents_in_batches(tier1_docs, self.tier1_collection, "Tier 1 (Basic Medical)", max_batch_size)
        
        # Add to Tier 2 - Clinical Reasoning  
        tier2_docs = organized_docs["hypothesis_testing"]
        if tier2_docs:
            self._add_documents_in_batches(tier2_docs, self.tier2_collection, "Tier 2 (Clinical)", max_batch_size)
        
        # Add to Tier 3 - Evidence-Based Medicine
        tier3_docs = organized_docs["confirmation"]
        if tier3_docs:
            self._add_documents_in_batches(tier3_docs, self.tier3_collection, "Tier 3 (Evidence)", max_batch_size)

    def _add_documents_in_batches(self, docs: List[Dict], collection, tier_name: str, batch_size: int):
        """Add documents to collection in batches."""
        total_docs = len(docs)
        
        for i in range(0, total_docs, batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            logger.info(f"📝 Adding batch {batch_num}/{total_batches} to {tier_name} ({len(batch_docs)} docs)")
            
            texts = [doc["text"] for doc in batch_docs]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            ids = [doc["metadata"]["tier_chunk_id"] for doc in batch_docs]
            metadatas = [doc["metadata"] for doc in batch_docs]
            
            collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        logger.info(f"✅ Added {total_docs} docs to {tier_name}")

    def hierarchical_search(self, query: str) -> Dict[str, List[Dict]]:
        """Perform hierarchical search across all medical tiers."""
        logger.info(f"🔍 Hierarchical medical search: {query[:100]}...")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        results = {}
        
        # Search Tier 1 - Basic Medical Knowledge
        try:
            tier1_results = self.tier1_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=self.tier1_top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            results["tier1_patterns"] = [
                {
                    "text": doc,
                    "metadata": meta,
                    "score": 1 - dist  # Convert distance to similarity
                }
                for doc, meta, dist in zip(
                    tier1_results["documents"][0],
                    tier1_results["metadatas"][0], 
                    tier1_results["distances"][0]
                )
            ]
        except Exception as e:
            logger.warning(f"Tier 1 search failed: {e}")
            results["tier1_patterns"] = []
        
        # Search Tier 2 - Clinical Reasoning
        try:
            tier2_results = self.tier2_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=self.tier2_top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            results["tier2_hypotheses"] = [
                {
                    "text": doc,
                    "metadata": meta,
                    "score": 1 - dist
                }
                for doc, meta, dist in zip(
                    tier2_results["documents"][0],
                    tier2_results["metadatas"][0],
                    tier2_results["distances"][0]
                )
            ]
        except Exception as e:
            logger.warning(f"Tier 2 search failed: {e}")
            results["tier2_hypotheses"] = []
        
        # Search Tier 3 - Evidence-Based Medicine
        try:
            tier3_results = self.tier3_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=self.tier3_top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            results["tier3_confirmation"] = [
                {
                    "text": doc,
                    "metadata": meta,
                    "score": 1 - dist
                }
                for doc, meta, dist in zip(
                    tier3_results["documents"][0],
                    tier3_results["metadatas"][0],
                    tier3_results["distances"][0]
                )
            ]
        except Exception as e:
            logger.warning(f"Tier 3 search failed: {e}")
            results["tier3_confirmation"] = []
        
        # Log search results
        tier1_count = len(results["tier1_patterns"])
        tier2_count = len(results["tier2_hypotheses"]) 
        tier3_count = len(results["tier3_confirmation"])
        
        logger.info(f"🔍 Retrieved: T1={tier1_count}, T2={tier2_count}, T3={tier3_count}")
        
        return results

    def search_single_tier(self, query: str, tier: int, top_k: int = 5) -> List[Dict]:
        """Search a single tier for debugging."""
        query_embedding = self.embedding_model.encode([query])
        
        if tier == 1 and self.tier1_collection:
            collection = self.tier1_collection
        elif tier == 2 and self.tier2_collection:
            collection = self.tier2_collection
        elif tier == 3 and self.tier3_collection:
            collection = self.tier3_collection
        else:
            return []
        
        try:
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            return [
                {
                    "text": doc,
                    "metadata": meta,
                    "score": 1 - dist
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ]
        except Exception as e:
            logger.error(f"Single tier {tier} search failed: {e}")
            return []