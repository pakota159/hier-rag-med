"""Base class for dataset fetchers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
import time
import requests
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    source: str
    url: str
    format: str
    expected_size: int
    description: str
    api_url: Optional[str] = None
    requires_api_key: bool = False

@dataclass
class Document:
    """Unified document format."""
    text: str
    metadata: Dict[str, Any]
    source_dataset: str
    doc_id: str
    reasoning_type: str

class BaseFetcher(ABC):
    """Base class for dataset fetchers."""
    
    def __init__(self, config: Dict[str, Any], checkpoint_dir: Optional[str] = None):
        """
        Initialize base fetcher.
        
        Args:
            config: Dataset configuration
            checkpoint_dir: Optional directory for checkpoints
        """
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Medical Research Dataset Collector/1.0'
        })
        
        self.rate_limit_delay = 1.0  # Default delay between requests
        self.last_request_time = 0.0
    
    @abstractmethod
    def get_dataset_info(self) -> DatasetInfo:
        """
        Get information about the dataset.
        
        Returns:
            DatasetInfo object
        """
        pass
    
    @abstractmethod
    def fetch_documents(self, max_documents: Optional[int] = None) -> List[Document]:
        """
        Fetch documents from the dataset.
        
        Args:
            max_documents: Optional maximum number of documents to fetch
            
        Returns:
            List of Document objects
        """
        pass
    
    def _rate_limit(self) -> None:
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _save_checkpoint(self, documents: List[Document], checkpoint_name: str) -> None:
        """
        Save checkpoint of fetched documents.
        
        Args:
            documents: List of documents to save
            checkpoint_name: Name for the checkpoint
        """
        if not self.checkpoint_dir:
            return
            
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump([doc.__dict__ for doc in documents], f, indent=2)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_name: str) -> Optional[List[Document]]:
        """
        Load latest checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint to load
            
        Returns:
            List of Document objects if checkpoint exists, None otherwise
        """
        if not self.checkpoint_dir:
            return None
            
        checkpoints = list(self.checkpoint_dir.glob(f"{checkpoint_name}_*.json"))
        if not checkpoints:
            return None
            
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        with open(latest_checkpoint, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded checkpoint from {latest_checkpoint}")
        return [Document(**doc) for doc in data]
    
    def _validate_document(self, doc: Document) -> bool:
        """
        Validate a document.
        
        Args:
            doc: Document to validate
            
        Returns:
            True if document is valid, False otherwise
        """
        required_fields = ['text', 'metadata', 'source_dataset', 'doc_id', 'reasoning_type']
        return all(hasattr(doc, field) for field in required_fields)
    
    def _make_request(self, url: str, method: str = 'GET', **kwargs) -> requests.Response:
        """
        Make HTTP request with rate limiting.
        
        Args:
            url: URL to request
            method: HTTP method
            **kwargs: Additional request parameters
            
        Returns:
            Response object
        """
        self._rate_limit()
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response 