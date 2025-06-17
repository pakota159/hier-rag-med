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

from .config import Config

class Retriever:
    """Document retriever using ChromaDB and sentence transformers."""

    def __init__(self, config: Config):
        """Initialize retriever."""
        self.config = config
        
        # Access config through config.config dictionary
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
        self.collection = None

    def create_collection(self, collection_name: str) -> None:
        """Create a new collection in ChromaDB."""
        try:
            # Delete existing collection if it exists
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, that's fine
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Created new collection: {collection_name}")

    def load_collection(self, collection_name: str) -> None:
        """Load an existing collection from ChromaDB."""
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded collection: {collection_name}")
        except Exception as e:
            logger.error(f"Collection {collection_name} not found: {e}")
            raise ValueError(f"Collection {collection_name} does not exist. Create it first with create_collection().")

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
        batch_size = self.config.config["models"]["embedding"]["batch_size"]
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

    def list_collections(self) -> List[str]:
        """List all available collections."""
        collections = self.client.list_collections()
        return [col.name for col in collections]

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
        n_results = n_results or self.config.config["retrieval"]["top_k"]
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
                "score": 1.0 - float(results["distances"][0][i])  # Convert distance to similarity
            })

        return formatted_results

    def hybrid_search(
        self,
        query: str,
        n_results: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """Perform hybrid search combining semantic and keyword search."""
        if not self.config.config["retrieval"]["hybrid_search"]:
            return self.search(query, n_results, filter_metadata)

        # For now, just return semantic search results
        # In a full implementation, this would combine semantic + keyword search
        return self.search(query, n_results, filter_metadata)