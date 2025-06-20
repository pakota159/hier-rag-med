"""
Document processing for Basic Reasoning system.
COMPLETELY UPDATED VERSION - Handles explicit tier assignments from foundation fetchers
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
import tqdm


class HierarchicalDocumentProcessor:
    """Document processor for hierarchical medical reasoning."""

    def __init__(self, config: Dict):
        """Initialize hierarchical document processor."""
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def process_text(self, text: str, metadata: Dict) -> List[Dict[str, str]]:
        """Process text document into chunks."""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)

            # Prepare documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata.update({
                    "chunk_id": i
                })
                documents.append({
                    "text": chunk,
                    "metadata": doc_metadata
                })

            return documents

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return []

    def load_foundation_dataset(self, foundation_dir: Path) -> List[Dict[str, str]]:
        """Load foundation dataset only."""
        all_documents = []
        
        foundation_file = foundation_dir / "foundation_medical_data.json"
        if foundation_file.exists():
            logger.info(f"📚 Loading foundation dataset from {foundation_file}")
            with open(foundation_file, "r") as f:
                foundation_data = json.load(f)
            
            for doc in tqdm.tqdm(foundation_data, desc="Processing foundation docs"):
                # Process each document through chunking
                chunks = self.process_text(doc["text"], doc["metadata"]) 
                all_documents.extend(chunks)
        else:
            logger.error(f"Foundation dataset not found at {foundation_file}")
            logger.error("Run: python fetch_foundation_data.py first")
            raise FileNotFoundError(f"Foundation dataset not found: {foundation_file}")
        
        logger.info(f"✅ Loaded {len(all_documents)} foundation documents")
        return all_documents

    def organize_by_reasoning_type(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize documents by reasoning type - USES EXPLICIT TIER ASSIGNMENTS."""
        
        organized = {
            "pattern_recognition": [],  # Tier 1
            "hypothesis_testing": [],   # Tier 2
            "confirmation": []          # Tier 3
        }
        
        tier_counts = {1: 0, 2: 0, 3: 0}
        
        for doc in documents:
            # Use explicit tier assignment from foundation fetchers
            tier = doc["metadata"].get("tier", None)
            
            if tier == 1:
                organized["pattern_recognition"].append(doc)
                tier_counts[1] += 1
            elif tier == 2:
                organized["hypothesis_testing"].append(doc)
                tier_counts[2] += 1
            elif tier == 3:
                organized["confirmation"].append(doc)
                tier_counts[3] += 1
            else:
                # Fallback for documents without explicit tier
                source = doc["metadata"].get("source", "")
                reasoning_type = doc["metadata"].get("reasoning_type", "")
                
                # Map based on updated sources
                if source in ["therapeutic_pharmacology", "evidence_based_pharmacology"]:
                    organized["pattern_recognition"].append(doc)
                    tier_counts[1] += 1
                elif source in ["therapeutic_guidelines"]:
                    organized["hypothesis_testing"].append(doc)
                    tier_counts[2] += 1
                elif source in ["clinical_outcomes"]:
                    organized["confirmation"].append(doc)
                    tier_counts[3] += 1
                else:
                    # Default to confirmation
                    organized["confirmation"].append(doc)
                    tier_counts[3] += 1
        
        # Log final distribution
        logger.info(f"📊 Organized documents by reasoning tiers:")
        logger.info(f"   Tier 1 (Pattern Recognition): {tier_counts[1]}")
        logger.info(f"   Tier 2 (Hypothesis Testing): {tier_counts[2]}")
        logger.info(f"   Tier 3 (Confirmation): {tier_counts[3]}")
        
        # Validate no empty tiers
        if tier_counts[1] == 0 or tier_counts[2] == 0 or tier_counts[3] == 0:
            logger.warning("⚠️ One or more tiers are empty - this may affect retrieval performance")
        
        return organized

    def analyze_dataset_quality(self, documents: List[Dict]) -> Dict:
        """Analyze the quality and composition of the foundation dataset."""
        
        analysis = {
            "total_documents": len(documents),
            "sources": {},
            "reasoning_types": {},
            "tiers": {},
            "avg_document_length": 0,
            "quality_indicators": {
                "evidence_based": 0,
                "synthetic": 0,
                "clinical": 0
            }
        }
        
        total_length = 0
        
        for doc in documents:
            # Source analysis
            source = doc["metadata"].get("source", "unknown")
            analysis["sources"][source] = analysis["sources"].get(source, 0) + 1
            
            # Reasoning type analysis
            reasoning_type = doc["metadata"].get("reasoning_type", "unknown")
            analysis["reasoning_types"][reasoning_type] = analysis["reasoning_types"].get(reasoning_type, 0) + 1
            
            # Tier analysis
            tier = doc["metadata"].get("tier", "unknown")
            analysis["tiers"][f"tier_{tier}"] = analysis["tiers"].get(f"tier_{tier}", 0) + 1
            
            # Length analysis
            text_length = len(doc.get("text", ""))
            total_length += text_length
            
            # Quality indicators
            text_lower = doc.get("text", "").lower()
            if any(term in text_lower for term in ["evidence", "trial", "study", "research"]):
                analysis["quality_indicators"]["evidence_based"] += 1
            if any(term in text_lower for term in ["sample", "example", "mock", "test"]):
                analysis["quality_indicators"]["synthetic"] += 1
            if any(term in text_lower for term in ["patient", "clinical", "diagnosis", "treatment"]):
                analysis["quality_indicators"]["clinical"] += 1
        
        if len(documents) > 0:
            analysis["avg_document_length"] = total_length / len(documents)
        
        return analysis

    def save_documents(self, documents: List[Dict[str, str]], output_path: Path) -> None:
        """Save processed documents to JSON file."""
        try:
            with open(output_path, "w") as f:
                json.dump(documents, f, indent=2)
            logger.info(f"Saved {len(documents)} documents to {output_path}")
        except Exception as e:
            logger.error(f"Error saving documents to {output_path}: {str(e)}")

    def load_documents(self, input_path: Path) -> List[Dict[str, str]]:
        """Load processed documents from JSON file."""
        try:
            with open(input_path, "r") as f:
                documents = json.load(f)
            logger.info(f"Loaded {len(documents)} documents from {input_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents from {input_path}: {str(e)}")
            return []