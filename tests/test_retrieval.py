"""
Tests for the retrieval module.
"""

import pytest
from pathlib import Path

from src.config import Config
from src.retrieval import Retriever


@pytest.fixture
def config():
    """Create test configuration."""
    return Config.from_yaml("config/config.yaml")


@pytest.fixture
def retriever(config):
    """Create test retriever."""
    return Retriever(config)


def test_create_collection(retriever):
    """Test collection creation."""
    collection_name = "test_collection"
    retriever.create_collection(collection_name)
    assert retriever.collection is not None
    assert retriever.collection.name == collection_name


def test_add_documents(retriever):
    """Test document addition."""
    # Create test collection
    collection_name = "test_collection"
    retriever.create_collection(collection_name)
    
    # Create test documents
    documents = [
        {
            "text": "This is a test document about diabetes.",
            "metadata": {
                "source": "test",
                "chunk_id": 0,
                "doc_id": "1"
            }
        },
        {
            "text": "Another test document about heart disease.",
            "metadata": {
                "source": "test",
                "chunk_id": 0,
                "doc_id": "2"
            }
        }
    ]
    
    # Add documents
    retriever.add_documents(documents)
    
    # Verify documents were added
    results = retriever.search("diabetes")
    assert len(results) > 0
    assert "diabetes" in results[0]["text"].lower()


def test_search(retriever):
    """Test document search."""
    # Create test collection
    collection_name = "test_collection"
    retriever.create_collection(collection_name)
    
    # Add test documents
    documents = [
        {
            "text": "This is a test document about diabetes.",
            "metadata": {
                "source": "test",
                "chunk_id": 0,
                "doc_id": "1"
            }
        }
    ]
    retriever.add_documents(documents)
    
    # Test search
    results = retriever.search("diabetes")
    assert len(results) > 0
    assert results[0]["score"] > 0
    assert "diabetes" in results[0]["text"].lower()


def test_hybrid_search(retriever):
    """Test hybrid search."""
    # Create test collection
    collection_name = "test_collection"
    retriever.create_collection(collection_name)
    
    # Add test documents
    documents = [
        {
            "text": "This is a test document about diabetes.",
            "metadata": {
                "source": "test",
                "chunk_id": 0,
                "doc_id": "1"
            }
        }
    ]
    retriever.add_documents(documents)
    
    # Test hybrid search
    results = retriever.hybrid_search("diabetes")
    assert len(results) > 0
    assert results[0]["score"] > 0
    assert "diabetes" in results[0]["text"].lower() 