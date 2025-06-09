"""
Retrieval module for HierRAGMed.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import Config


class Retriever:
    """Document retriever using ChromaDB and sentence transformers."""

    def __init__(self, config: Config):
        """Initialize retriever."""
        self.config = config
        self.embedding_model = SentenceTransformer(
            config.models["embedding"]["name"],
            device=config.models["embedding"]["device"]
        )
        self.client = chromadb.PersistentClient(
            path=str(config.get_data_dir("vector_db")),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = None

    def create_collection(self, collection_name: str) -> None:
        """Create a new collection in ChromaDB."""
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Created collection: {collection_name}")

    def load_collection(self, collection_name: str) -> None:
        """Load an existing collection from ChromaDB."""
        self.collection = self.client.get_collection(collection_name)
        logger.info(f"Loaded collection: {collection_name}")

    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Add documents to the collection."""
        if not self.collection:
            raise ValueError("No collection loaded. Call create_collection or load_collection first.")

        # Prepare documents for batch processing
        texts = []
        metadatas = []
        ids = []

        for doc in tqdm(documents, desc="Preparing documents"):
            texts.append(doc["text"])
            metadatas.append(doc["metadata"])
            ids.append(f"{doc['metadata']['doc_id']}_{doc['metadata']['chunk_id']}")

        # Generate embeddings in batches
        batch_size = self.config.models["embedding"]["batch_size"]
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings)

        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(documents)} documents to collection")

    def search(
        self,
        query: str,
        n_results: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """Search for relevant documents."""
        if not self.collection:
            raise ValueError("No collection loaded. Call create_collection or load_collection first.")

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Search in collection
        n_results = n_results or self.config.retrieval["top_k"]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )

        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": float(results["distances"][0][i])
            })

        return formatted_results

    def hybrid_search(
        self,
        query: str,
        n_results: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """Perform hybrid search combining semantic and keyword search."""
        if not self.config.retrieval["hybrid_search"]:
            return self.search(query, n_results, filter_metadata)

        # Get semantic search results
        semantic_results = self.search(
            query,
            n_results=n_results,
            filter_metadata=filter_metadata
        )

        # Get keyword search results
        keyword_results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_metadata
        )

        # Combine results using weighted scoring
        alpha = self.config.retrieval["alpha"]
        combined_results = {}

        for result in semantic_results:
            doc_id = result["metadata"]["doc_id"]
            combined_results[doc_id] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "score": (1 - alpha) * result["score"]
            }

        for i, doc_id in enumerate(keyword_results["ids"][0]):
            if doc_id in combined_results:
                combined_results[doc_id]["score"] += alpha * (1 - float(keyword_results["distances"][0][i]))
            else:
                combined_results[doc_id] = {
                    "text": keyword_results["documents"][0][i],
                    "metadata": keyword_results["metadatas"][0][i],
                    "score": alpha * (1 - float(keyword_results["distances"][0][i]))
                }

        # Sort by combined score
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:n_results]

        return sorted_results 