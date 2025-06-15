"""Data validation for collected documents."""

import logging
from typing import Any, Dict, List, Optional
from .base_fetcher import Document

logger = logging.getLogger(__name__)

class DataValidator:
    """Validate collected documents."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.required_fields = config["processing"]["unified_format"]["required_fields"]
        self.metadata_fields = config["processing"]["unified_format"]["metadata_fields"]
    
    def validate_document(self, doc: Document) -> bool:
        """
        Validate a single document.
        
        Args:
            doc: Document to validate
            
        Returns:
            True if document is valid, False otherwise
        """
        # Check required fields
        if not all(hasattr(doc, field) for field in self.required_fields):
            logger.warning(f"Document {doc.doc_id} missing required fields")
            return False
        
        # Check text length
        if not doc.text or len(doc.text.strip()) < 50:
            logger.warning(f"Document {doc.doc_id} text too short")
            return False
        
        # Check metadata fields
        if not all(field in doc.metadata for field in self.metadata_fields):
            logger.warning(f"Document {doc.doc_id} missing metadata fields")
            return False
        
        # Check reasoning chain
        if "reasoning_chain" in doc.metadata:
            chain = doc.metadata["reasoning_chain"]
            if not isinstance(chain, list) or not chain:
                logger.warning(f"Document {doc.doc_id} invalid reasoning chain")
                return False
        
        return True
    
    def validate_dataset(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Validate a dataset of documents.
        
        Args:
            documents: List of documents to validate
            
        Returns:
            Dictionary containing validation statistics
        """
        stats = {
            "total_documents": len(documents),
            "valid_documents": 0,
            "invalid_documents": 0,
            "validation_errors": {
                "missing_fields": 0,
                "short_text": 0,
                "missing_metadata": 0,
                "invalid_reasoning": 0
            },
            "reasoning_types": {},
            "medical_specialties": {}
        }
        
        for doc in documents:
            # Track reasoning types
            stats["reasoning_types"][doc.reasoning_type] = stats["reasoning_types"].get(doc.reasoning_type, 0) + 1
            
            # Track medical specialties
            specialty = doc.metadata.get("medical_specialty", "Unknown")
            stats["medical_specialties"][specialty] = stats["medical_specialties"].get(specialty, 0) + 1
            
            # Validate document
            if self.validate_document(doc):
                stats["valid_documents"] += 1
            else:
                stats["invalid_documents"] += 1
                
                # Count specific validation errors
                if not all(hasattr(doc, field) for field in self.required_fields):
                    stats["validation_errors"]["missing_fields"] += 1
                if not doc.text or len(doc.text.strip()) < 50:
                    stats["validation_errors"]["short_text"] += 1
                if not all(field in doc.metadata for field in self.metadata_fields):
                    stats["validation_errors"]["missing_metadata"] += 1
                if "reasoning_chain" in doc.metadata:
                    chain = doc.metadata["reasoning_chain"]
                    if not isinstance(chain, list) or not chain:
                        stats["validation_errors"]["invalid_reasoning"] += 1
        
        return stats
    
    def get_validation_report(self, stats: Dict[str, Any]) -> str:
        """
        Generate a formatted validation report.
        
        Args:
            stats: Validation statistics
            
        Returns:
            Formatted validation report string
        """
        report = ["\nDataset Validation Report:"]
        report.append("-" * 50)
        
        # Overall statistics
        report.append(f"Total Documents: {stats['total_documents']}")
        report.append(f"Valid Documents: {stats['valid_documents']} ({stats['valid_documents']/stats['total_documents']*100:.1f}%)")
        report.append(f"Invalid Documents: {stats['invalid_documents']} ({stats['invalid_documents']/stats['total_documents']*100:.1f}%)")
        
        # Validation errors
        report.append("\nValidation Errors:")
        for error, count in stats["validation_errors"].items():
            report.append(f"- {error}: {count}")
        
        # Reasoning types
        report.append("\nReasoning Types:")
        for rtype, count in stats["reasoning_types"].items():
            report.append(f"- {rtype}: {count}")
        
        # Medical specialties
        report.append("\nMedical Specialties:")
        for specialty, count in stats["medical_specialties"].items():
            report.append(f"- {specialty}: {count}")
        
        return "\n".join(report) 