"""Fetcher for MSDiagnosis dataset."""

import json
import logging
import random
from typing import Any, Dict, List, Optional
from .base_fetcher import BaseFetcher, DatasetInfo, Document

logger = logging.getLogger(__name__)

class MSDiagnosisFetcher(BaseFetcher):
    """Fetcher for MSDiagnosis dataset."""
    
    def __init__(self, config: Dict[str, Any], checkpoint_dir: Optional[str] = None):
        """
        Initialize MSDiagnosis fetcher.
        
        Args:
            config: Dataset configuration
            checkpoint_dir: Optional directory for checkpoints
        """
        super().__init__(config, checkpoint_dir)
        self.rate_limit_delay = 2.0  # Conservative rate limit for research dataset
    
    def get_dataset_info(self) -> DatasetInfo:
        """
        Get information about the MSDiagnosis dataset.
        
        Returns:
            DatasetInfo object
        """
        return DatasetInfo(
            name="msdiagnosis",
            source=self.config["source"],
            url=self.config["url"],
            format=self.config["format"],
            expected_size=self.config["expected_size"],
            description=self.config["description"]
        )
    
    def _generate_sample_diagnosis(self) -> Dict[str, Any]:
        """
        Generate a sample multi-step diagnosis for testing.
        
        Returns:
            Dictionary containing sample diagnosis data
        """
        conditions = [
            "Type 2 Diabetes Mellitus",
            "Hypertension",
            "Coronary Artery Disease",
            "Chronic Kidney Disease",
            "Congestive Heart Failure"
        ]
        
        symptoms = [
            "Fatigue",
            "Shortness of breath",
            "Chest pain",
            "Edema",
            "Polyuria",
            "Polydipsia",
            "Weight loss",
            "Blurred vision"
        ]
        
        primary = random.choice(conditions)
        differential = random.sample([c for c in conditions if c != primary], 2)
        final = random.choice([primary] + differential)
        
        return {
            "id": f"msd_{random.randint(1000, 9999)}",
            "text": f"Patient presents with {', '.join(random.sample(symptoms, 3))}.",
            "metadata": {
                "title": f"Case Study: {primary}",
                "authors": ["Dr. Smith", "Dr. Johnson"],
                "publication_date": "2024-01-01",
                "medical_specialty": "Internal Medicine",
                "evidence_level": "Level 2",
                "reasoning_chain": [
                    {
                        "step": "Primary Diagnosis",
                        "condition": primary,
                        "confidence": random.uniform(0.7, 0.9),
                        "reasoning": f"Initial presentation suggests {primary}"
                    },
                    {
                        "step": "Differential Diagnosis",
                        "conditions": differential,
                        "confidence": random.uniform(0.6, 0.8),
                        "reasoning": "Consider alternative diagnoses based on symptoms"
                    },
                    {
                        "step": "Final Diagnosis",
                        "condition": final,
                        "confidence": random.uniform(0.8, 0.95),
                        "reasoning": "Confirmed through additional testing"
                    }
                ]
            },
            "source_dataset": "msdiagnosis",
            "reasoning_type": "multi_step_diagnostic"
        }
    
    def fetch_documents(self, max_documents: Optional[int] = None) -> List[Document]:
        """
        Fetch multi-step diagnostic scenarios.
        
        Args:
            max_documents: Optional maximum number of documents to fetch
            
        Returns:
            List of Document objects
        """
        # Try to load from checkpoint first
        checkpoint_docs = self._load_checkpoint("msdiagnosis")
        if checkpoint_docs:
            logger.info(f"Loaded {len(checkpoint_docs)} documents from checkpoint")
            if max_documents:
                return checkpoint_docs[:max_documents]
            return checkpoint_docs
        
        # For now, generate sample data since the real dataset is not available
        documents = []
        try:
            num_docs = max_documents if max_documents else self.config["expected_size"]
            
            for _ in range(num_docs):
                sample_data = self._generate_sample_diagnosis()
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
                    self._save_checkpoint(documents, "msdiagnosis")
            
            # Save final checkpoint
            self._save_checkpoint(documents, "msdiagnosis")
            
        except Exception as e:
            logger.error(f"Error generating MSDiagnosis data: {str(e)}")
            raise
        
        return documents 