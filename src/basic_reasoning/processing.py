"""
Document processing for Basic Reasoning system.
Only processes foundation datasets from data/foundation/
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

            logger.info(f"Processed text: {len(documents)} chunks")
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
        """Organize documents by reasoning type for tiered retrieval."""
        organized = {
            "pattern_recognition": [],  # Tier 1: Quick patterns (drugs, symptoms)
            "hypothesis_testing": [],   # Tier 2: Reasoning chains (MedReason, MSDiagnosis)
            "confirmation": []          # Tier 3: Clinical evidence (PMC cases)
        }
        
        for doc in documents:
            reasoning_type = doc["metadata"].get("reasoning_type", "confirmation")
            source = doc["metadata"].get("source", "unknown")
            
            # Map reasoning types to tiers based on source and type
            if reasoning_type == "knowledge_graph_guided" or source == "medreason":
                organized["hypothesis_testing"].append(doc)
            elif reasoning_type == "multi_step_diagnostic" or source == "msdiagnosis":
                organized["hypothesis_testing"].append(doc)
            elif reasoning_type == "case_study" or source == "pmc_patients":
                organized["confirmation"].append(doc)
            elif reasoning_type == "drug_information" or source == "drugbank":
                organized["pattern_recognition"].append(doc)
            else:
                # Default classification based on source
                if source in ["medreason", "msdiagnosis"]:
                    organized["hypothesis_testing"].append(doc)
                elif source in ["pmc_patients"]:
                    organized["confirmation"].append(doc)
                elif source in ["drugbank"]:
                    organized["pattern_recognition"].append(doc)
                else:
                    # Default to confirmation tier
                    organized["confirmation"].append(doc)
        
        logger.info(f"📊 Organized documents by reasoning tiers:")
        logger.info(f"   Tier 1 (Pattern Recognition): {len(organized['pattern_recognition'])}")
        logger.info(f"   Tier 2 (Hypothesis Testing): {len(organized['hypothesis_testing'])}")
        logger.info(f"   Tier 3 (Confirmation): {len(organized['confirmation'])}")
        
        return organized

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