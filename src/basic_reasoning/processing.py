"""
Document processing for Basic Reasoning system.
COMPLETELY UPDATED VERSION - Handles PubMed/MTSamples/MeSH data with intelligent tier assignment
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import uuid

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

    def assign_document_tier(self, doc: Dict) -> int:
        """
        Assign hierarchical tier to documents based on content and metadata.
        Tier 1 (Pattern Recognition): Basic symptoms, definitions, classifications
        Tier 2 (Hypothesis Testing): Clinical studies, treatments, procedures  
        Tier 3 (Confirmation): Meta-analyses, guidelines, high-evidence content
        """
        metadata = doc.get("metadata", {})
        text = doc.get("text", "").lower()
        title = metadata.get("title", "").lower()
        source = metadata.get("source", "")
        
        # Combined text for analysis
        combined_text = f"{title} {text}"
        
        # Tier 3 (Confirmation) - High-evidence content
        tier3_indicators = [
            "meta-analysis", "systematic review", "clinical practice guideline",
            "consensus statement", "cochrane review", "evidence-based",
            "practice guideline", "treatment guideline", "clinical guideline",
            "practice parameter", "best practice", "gold standard",
            "randomized controlled trial", "clinical trial", "rct"
        ]
        
        # Tier 2 (Hypothesis Testing) - Clinical studies and procedures
        tier2_indicators = [
            "prospective study", "cohort study", "case-control",
            "diagnostic", "treatment", "therapy", "intervention",
            "clinical effectiveness", "therapeutic", "procedure",
            "management", "protocol", "approach", "method",
            "patient", "clinical", "diagnosis", "assessment"
        ]
        
        # Tier 1 (Pattern Recognition) - Basic concepts and symptoms
        tier1_indicators = [
            "symptom", "symptoms", "sign", "signs", "definition",
            "characterized by", "presents with", "manifestation",
            "classification", "category", "type", "pathophysiology",
            "etiology", "cause", "epidemiology", "prevalence",
            "risk factor", "anatomy", "physiology", "term", "concept"
        ]
        
        # Check for high-impact journals (Tier 3)
        high_impact_journals = [
            "n engl j med", "lancet", "jama", "bmj", "nature medicine",
            "cell", "science", "nature", "circulation"
        ]
        journal = metadata.get("journal", "").lower()
        
        # Publication type analysis
        pub_types = metadata.get("publication_types", "").lower()
        
        # Assign tiers based on indicators
        if any(indicator in combined_text for indicator in tier3_indicators):
            return 3
        elif any(indicator in pub_types for indicator in ["meta-analysis", "systematic review", "guideline"]):
            return 3
        elif any(journal_name in journal for journal_name in high_impact_journals):
            return 3
        elif any(indicator in combined_text for indicator in tier2_indicators):
            return 2
        elif any(indicator in pub_types for indicator in ["clinical trial"]):
            return 2
        elif any(indicator in combined_text for indicator in tier1_indicators):
            return 1
        else:
            # Source-based assignment for edge cases
            if source == "mesh":
                return 1  # MeSH terms are basic concepts
            elif source == "mtsamples":
                return 2  # Clinical documentation
            elif source == "pubmed":
                # For PubMed without clear indicators, distribute based on content length
                if len(text) < 500:
                    return 1  # Short abstracts likely basic concepts
                elif len(text) < 1500:
                    return 2  # Medium abstracts likely clinical studies
                else:
                    return 3  # Long abstracts likely comprehensive reviews
            else:
                return 2  # Default to middle tier

    def preprocess_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Preprocess documents to fix metadata for ChromaDB compatibility.
        ChromaDB only accepts str, int, float, bool, or None as metadata values.
        """
        processed_docs = []
        
        for i, doc in enumerate(documents):
            # Create a copy of the document
            processed_doc = doc.copy()
            metadata = doc.get("metadata", {}).copy()
            
            # Convert list fields to strings
            for key, value in metadata.items():
                if isinstance(value, list):
                    if value:  # Non-empty list
                        # Join list elements with semicolon separator
                        metadata[key] = "; ".join(str(item) for item in value)
                    else:  # Empty list
                        metadata[key] = ""
                elif value is None:
                    metadata[key] = ""
                else:
                    # Keep primitive types as-is
                    metadata[key] = value
            
            # Assign tier information
            tier = self.assign_document_tier(doc)
            metadata["tier"] = tier
            
            # Ensure required metadata fields exist with unique IDs
            if "doc_id" not in metadata:
                # Create a unique doc_id based on content hash
                content_hash = hashlib.md5(doc.get("text", "").encode()).hexdigest()[:8]
                metadata["doc_id"] = f"doc_{content_hash}_{i}"
            
            if "chunk_id" not in metadata:
                metadata["chunk_id"] = 0
                
            # Create a globally unique identifier
            unique_id = str(uuid.uuid4())[:8]
            metadata["unique_id"] = unique_id
            
            if "medical_specialty" not in metadata:
                source = metadata.get("source", "")
                if source == "mesh":
                    metadata["medical_specialty"] = "general"
                elif source == "mtsamples":
                    metadata["medical_specialty"] = "clinical"
                else:
                    metadata["medical_specialty"] = "unknown"
            
            processed_doc["metadata"] = metadata
            processed_docs.append(processed_doc)
        
        return processed_docs

    def organize_by_reasoning_type(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Organize documents by reasoning type using tier assignments.
        Now properly handles documents with tier metadata.
        """
        organized = {
            "pattern_recognition": [],  # Tier 1
            "hypothesis_testing": [],   # Tier 2
            "confirmation": []          # Tier 3
        }
        
        tier_counts = {1: 0, 2: 0, 3: 0}
        tier_id_counters = {1: 0, 2: 0, 3: 0}  # To ensure unique IDs
        
        for doc in documents:
            # Get tier assignment from metadata
            tier = doc["metadata"].get("tier", 2)  # Default to tier 2
            
            # Ensure unique chunk_id within tier to prevent ChromaDB collisions
            tier_id_counters[tier] += 1
            doc["metadata"]["tier_chunk_id"] = tier_id_counters[tier]
            
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
                # Fallback for invalid tier values
                logger.warning(f"Invalid tier value {tier}, defaulting to tier 2")
                organized["hypothesis_testing"].append(doc)
                tier_counts[2] += 1
        
        # Log final distribution
        logger.info(f"📊 Organized documents by reasoning tiers:")
        logger.info(f"   Tier 1 (Pattern Recognition): {tier_counts[1]}")
        logger.info(f"   Tier 2 (Hypothesis Testing): {tier_counts[2]}")
        logger.info(f"   Tier 3 (Confirmation): {tier_counts[3]}")
        
        # Validate no empty tiers
        empty_tiers = []
        if tier_counts[1] == 0:
            empty_tiers.append("Tier 1 (Pattern Recognition)")
        if tier_counts[2] == 0:
            empty_tiers.append("Tier 2 (Hypothesis Testing)")
        if tier_counts[3] == 0:
            empty_tiers.append("Tier 3 (Confirmation)")
            
        if empty_tiers:
            logger.warning(f"⚠️ Empty tiers detected: {', '.join(empty_tiers)}")
            logger.warning("This may affect hierarchical retrieval performance")
        
        return organized

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