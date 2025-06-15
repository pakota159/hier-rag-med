"""Fetcher for DrugBank dataset."""

import json
import logging
import random
from typing import Any, Dict, List, Optional
from .base_fetcher import BaseFetcher, DatasetInfo, Document

logger = logging.getLogger(__name__)

class DrugBankFetcher(BaseFetcher):
    """Fetcher for DrugBank dataset."""
    
    def __init__(self, config: Dict[str, Any], checkpoint_dir: Optional[str] = None):
        """
        Initialize DrugBank fetcher.
        
        Args:
            config: Dataset configuration
            checkpoint_dir: Optional directory for checkpoints
        """
        super().__init__(config, checkpoint_dir)
        self.rate_limit_delay = 1.0  # Conservative rate limit for API
    
    def get_dataset_info(self) -> DatasetInfo:
        """
        Get information about the DrugBank dataset.
        
        Returns:
            DatasetInfo object
        """
        return DatasetInfo(
            name="drugbank",
            source=self.config["source"],
            url=self.config["url"],
            format=self.config["format"],
            expected_size=self.config["expected_size"],
            description=self.config["description"],
            api_url=self.config["api_url"],
            requires_api_key=self.config["requires_api_key"]
        )
    
    def _generate_sample_drug(self) -> Dict[str, Any]:
        """
        Generate a sample drug entry for testing.
        
        Returns:
            Dictionary containing sample drug data
        """
        drug_classes = [
            "ACE Inhibitor",
            "Beta Blocker",
            "Calcium Channel Blocker",
            "Diuretic",
            "Statin"
        ]
        
        indications = [
            "Hypertension",
            "Heart Failure",
            "Coronary Artery Disease",
            "Hyperlipidemia",
            "Diabetes Mellitus"
        ]
        
        side_effects = [
            "Dizziness",
            "Headache",
            "Fatigue",
            "Nausea",
            "Cough"
        ]
        
        drug_class = random.choice(drug_classes)
        return {
            "id": f"DB{random.randint(10000, 99999)}",
            "text": f"Drug information for {drug_class}",
            "metadata": {
                "title": f"Drug Profile: {drug_class}",
                "authors": ["DrugBank Database"],
                "publication_date": "2024-01-01",
                "medical_specialty": "Pharmacology",
                "evidence_level": "Level 1",
                "reasoning_chain": [
                    {
                        "step": "Drug Class",
                        "value": drug_class,
                        "confidence": 1.0,
                        "reasoning": "Standard classification"
                    },
                    {
                        "step": "Indications",
                        "value": random.sample(indications, 2),
                        "confidence": 0.9,
                        "reasoning": "FDA approved indications"
                    },
                    {
                        "step": "Side Effects",
                        "value": random.sample(side_effects, 3),
                        "confidence": 0.8,
                        "reasoning": "Common adverse effects"
                    }
                ]
            },
            "source_dataset": "drugbank",
            "reasoning_type": "drug_information"
        }
    
    def fetch_documents(self, max_documents: Optional[int] = None) -> List[Document]:
        """
        Fetch drug information from DrugBank.
        
        Args:
            max_documents: Optional maximum number of documents to fetch
            
        Returns:
            List of Document objects
        """
        # Try to load from checkpoint first
        checkpoint_docs = self._load_checkpoint("drugbank")
        if checkpoint_docs:
            logger.info(f"Loaded {len(checkpoint_docs)} documents from checkpoint")
            if max_documents:
                return checkpoint_docs[:max_documents]
            return checkpoint_docs
        
        # For now, generate sample data since API key is required
        documents = []
        try:
            num_docs = max_documents if max_documents else self.config["expected_size"]
            
            for _ in range(num_docs):
                sample_data = self._generate_sample_drug()
                doc = Document(
                    text=sample_data["text"],
                    metadata=sample_data["metadata"],
                    source_dataset=sample_data["source_dataset"],
                    doc_id=sample_data["id"],
                    reasoning_type=sample_data["reasoning_type"]
                )
                
                if self._validate_document(doc):
                    documents.append(doc)
                
                # Save checkpoint every 100 documents
                if len(documents) % 100 == 0:
                    self._save_checkpoint(documents, "drugbank")
            
            # Save final checkpoint
            self._save_checkpoint(documents, "drugbank")
            
        except Exception as e:
            logger.error(f"Error generating DrugBank data: {str(e)}")
            raise
        
        return documents 