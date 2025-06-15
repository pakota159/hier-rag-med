"""Fetcher for MedReason dataset."""

import json
import logging
from typing import Any, Dict, List, Optional
from .base_fetcher import BaseFetcher, DatasetInfo, Document

logger = logging.getLogger(__name__)

class MedReasonFetcher(BaseFetcher):
    """Fetcher for MedReason dataset from GitHub."""
    
    def __init__(self, config: Dict[str, Any], checkpoint_dir: Optional[str] = None):
        """
        Initialize MedReason fetcher.
        
        Args:
            config: Dataset configuration
            checkpoint_dir: Optional directory for checkpoints
        """
        super().__init__(config, checkpoint_dir)
        self.rate_limit_delay = 1.0  # GitHub API rate limit
    
    def get_dataset_info(self) -> DatasetInfo:
        """
        Get information about the MedReason dataset.
        
        Returns:
            DatasetInfo object
        """
        return DatasetInfo(
            name="medreason",
            source=self.config["source"],
            url=self.config["url"],
            format=self.config["format"],
            expected_size=self.config["expected_size"],
            description=self.config["description"],
            api_url=self.config["api_url"]
        )
    
    def fetch_documents(self, max_documents: Optional[int] = None) -> List[Document]:
        """
        Fetch reasoning chains from MedReason dataset.
        
        Args:
            max_documents: Optional maximum number of documents to fetch
            
        Returns:
            List of Document objects
        """
        # Try to load from checkpoint first
        checkpoint_docs = self._load_checkpoint("medreason")
        if checkpoint_docs:
            logger.info(f"Loaded {len(checkpoint_docs)} documents from checkpoint")
            if max_documents:
                return checkpoint_docs[:max_documents]
            return checkpoint_docs
        
        # Fetch from GitHub API
        documents = []
        try:
            response = self._make_request(self.config["api_url"])
            data = response.json()
            
            for item in data:
                if item["type"] == "file" and item["name"].endswith(".json"):
                    file_response = self._make_request(item["download_url"])
                    reasoning_chain = file_response.json()
                    
                    # Create document from reasoning chain
                    doc = Document(
                        text=reasoning_chain.get("text", ""),
                        metadata={
                            "title": reasoning_chain.get("title", ""),
                            "authors": reasoning_chain.get("authors", []),
                            "publication_date": reasoning_chain.get("date", ""),
                            "medical_specialty": reasoning_chain.get("specialty", ""),
                            "evidence_level": reasoning_chain.get("evidence_level", ""),
                            "reasoning_chain": reasoning_chain.get("reasoning_steps", [])
                        },
                        source_dataset="medreason",
                        doc_id=reasoning_chain.get("id", ""),
                        reasoning_type=reasoning_chain.get("reasoning_type", "diagnostic")
                    )
                    
                    if self._validate_document(doc):
                        documents.append(doc)
                    
                    # Save checkpoint every 100 documents
                    if len(documents) % 100 == 0:
                        self._save_checkpoint(documents, "medreason")
                    
                    if max_documents and len(documents) >= max_documents:
                        break
            
            # Save final checkpoint
            self._save_checkpoint(documents, "medreason")
            
        except Exception as e:
            logger.error(f"Error fetching MedReason data: {str(e)}")
            raise
        
        return documents 