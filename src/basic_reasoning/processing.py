#!/usr/bin/env python3
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
        IMPROVED LOGIC - Ensures balanced distribution across all tiers
        
        Tier 1 (Pattern Recognition): Basic concepts, definitions, symptoms, classifications
        Tier 2 (Hypothesis Testing): Clinical studies, treatments, procedures, diagnostics
        Tier 3 (Confirmation): High-evidence content, guidelines, meta-analyses
        """
        metadata = doc.get("metadata", {})
        
        # Safely handle text field - convert to string if it's a list
        text_raw = doc.get("text", "")
        if isinstance(text_raw, list):
            text = " ".join(str(item) for item in text_raw).lower()
        elif text_raw is None:
            text = ""
        else:
            text = str(text_raw).lower()
        
        # Safely handle title field - convert to string if it's a list
        title_raw = metadata.get("title", "")
        if isinstance(title_raw, list):
            title = " ".join(str(item) for item in title_raw).lower()
        elif title_raw is None:
            title = ""
        else:
            title = str(title_raw).lower()
        
        source = metadata.get("source", "")
        combined_text = f"{title} {text}"
        
        # Safely handle journal and publication types
        journal_raw = metadata.get("journal", "")
        journal = " ".join(str(item) for item in journal_raw).lower() if isinstance(journal_raw, list) else str(journal_raw).lower()
        
        pub_types_raw = metadata.get("publication_types", "")
        pub_types = " ".join(str(item) for item in pub_types_raw).lower() if isinstance(pub_types_raw, list) else str(pub_types_raw).lower()
        
        # TIER 1 (Pattern Recognition) - Basic concepts and definitions
        # Enhanced indicators to capture more foundational content
        tier1_indicators = [
            "definition", "defined as", "is a", "are a", "refers to", "term",
            "symptom", "symptoms", "sign", "signs", "manifestation", "presents with",
            "characterized by", "classification", "category", "type", "types",
            "pathophysiology", "etiology", "cause", "causes", "epidemiology", 
            "prevalence", "incidence", "risk factor", "anatomy", "physiology",
            "concept", "basic", "fundamental", "introduction", "overview",
            "what is", "description", "features", "characteristics"
        ]
        
        # TIER 3 (Confirmation) - High-evidence content (check first for priority)
        tier3_indicators = [
            "meta-analysis", "systematic review", "clinical practice guideline",
            "consensus statement", "cochrane review", "evidence-based",
            "practice guideline", "treatment guideline", "clinical guideline",
            "practice parameter", "best practice", "gold standard",
            "randomized controlled trial", "clinical trial", "rct",
            "level i evidence", "level a recommendation"
        ]
        
        # High-impact journals automatically go to Tier 3
        high_impact_journals = [
            "n engl j med", "lancet", "jama", "bmj", "nature medicine",
            "cell", "science", "nature", "circulation", "cochrane"
        ]
        
        # TIER 2 (Hypothesis Testing) - Clinical applications
        tier2_indicators = [
            "treatment", "therapy", "therapeutic", "intervention", "procedure",
            "management", "protocol", "approach", "method", "technique",
            "diagnostic", "diagnosis", "assessment", "evaluation", "examination",
            "clinical", "patient", "case", "study", "cohort", "case-control",
            "prospective", "retrospective", "clinical effectiveness"
        ]
        
        # PRIORITY ASSIGNMENT (check in order):
        
        # 1. High-evidence content → Tier 3
        if (any(indicator in combined_text for indicator in tier3_indicators) or
            any(indicator in pub_types for indicator in ["meta-analysis", "systematic review", "guideline"]) or
            any(journal_name in journal for journal_name in high_impact_journals)):
            return 3
        
        # 2. Basic concepts and definitions → Tier 1
        if any(indicator in combined_text for indicator in tier1_indicators):
            return 1
        
        # 3. Clinical applications → Tier 2
        if (any(indicator in combined_text for indicator in tier2_indicators) or
            any(indicator in pub_types for indicator in ["clinical trial", "case report"])):
            return 2
        
        # 4. Source-based fallback assignment
        if source in ["mesh", "mtsamples"]:
            return 1  # Basic medical concepts
        elif "guideline" in source.lower():
            return 3  # Guidelines are high-evidence
        elif source == "pubmed":
            # Distribute PubMed content more evenly
            text_length = len(text)
            title_length = len(title)
            
            # Short content with basic terms → Tier 1
            if text_length < 300 or any(basic in title for basic in ["definition", "what is", "overview"]):
                return 1
            # Long comprehensive content → Tier 3
            elif text_length > 2000 or any(evidence in title for evidence in ["review", "guideline", "meta"]):
                return 3
            else:
                return 2
        
        # 5. Final fallback - distribute evenly
        # Use document hash to ensure consistent but distributed assignment
        doc_hash = hash(combined_text) % 3
        if doc_hash == 0:
            return 1
        elif doc_hash == 1:
            return 2
        else:
            return 2  # Favor Tier 2 for unknown content

    def rebalance_tier_distribution(self, documents: List[Dict]) -> List[Dict]:
        """
        Rebalance tier distribution if too imbalanced.
        Ensures each tier has at least 10% of total documents.
        """
        total_docs = len(documents)
        tier_counts = {1: 0, 2: 0, 3: 0}
        
        # Count current distribution
        for doc in documents:
            tier = doc["metadata"].get("tier", 2)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        logger.info(f"📊 Initial tier distribution:")
        logger.info(f"   Tier 1: {tier_counts.get(1, 0)} ({tier_counts.get(1, 0)/total_docs*100:.1f}%)")
        logger.info(f"   Tier 2: {tier_counts.get(2, 0)} ({tier_counts.get(2, 0)/total_docs*100:.1f}%)")
        logger.info(f"   Tier 3: {tier_counts.get(3, 0)} ({tier_counts.get(3, 0)/total_docs*100:.1f}%)")
        
        # Check if rebalancing is needed
        min_threshold = int(total_docs * 0.10)  # Each tier should have at least 10%
        needs_rebalancing = False
        
        for tier, count in tier_counts.items():
            if count < min_threshold:
                needs_rebalancing = True
                logger.warning(f"⚠️ Tier {tier} has only {count} documents ({count/total_docs*100:.1f}%), minimum is {min_threshold}")
        
        if not needs_rebalancing:
            logger.info("✅ Tier distribution is balanced, no rebalancing needed")
            return documents
        
        logger.info("🔄 Rebalancing tier distribution...")
        
        # Rebalance by reassigning some documents
        rebalanced_docs = []
        tier_targets = {1: min_threshold, 2: min_threshold, 3: min_threshold}
        current_counts = {1: 0, 2: 0, 3: 0}
        
        # First pass: keep documents that are in under-represented tiers
        for doc in documents:
            original_tier = doc["metadata"].get("tier", 2)
            if current_counts[original_tier] < tier_targets[original_tier]:
                current_counts[original_tier] += 1
                rebalanced_docs.append(doc)
            else:
                # Reassign to tier that needs more documents
                for target_tier in [1, 2, 3]:
                    if current_counts[target_tier] < tier_targets[target_tier]:
                        doc["metadata"]["tier"] = target_tier
                        current_counts[target_tier] += 1
                        rebalanced_docs.append(doc)
                        break
                else:
                    # All tiers have minimum, keep original assignment
                    rebalanced_docs.append(doc)
        
        # Log final distribution
        final_counts = {1: 0, 2: 0, 3: 0}
        for doc in rebalanced_docs:
            tier = doc["metadata"].get("tier", 2)
            final_counts[tier] = final_counts.get(tier, 0) + 1
        
        logger.info(f"📊 Rebalanced tier distribution:")
        logger.info(f"   Tier 1: {final_counts.get(1, 0)} ({final_counts.get(1, 0)/total_docs*100:.1f}%)")
        logger.info(f"   Tier 2: {final_counts.get(2, 0)} ({final_counts.get(2, 0)/total_docs*100:.1f}%)")
        logger.info(f"   Tier 3: {final_counts.get(3, 0)} ({final_counts.get(3, 0)/total_docs*100:.1f}%)")
        
        return rebalanced_docs

    def preprocess_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Preprocess documents to fix metadata for ChromaDB compatibility.
        ChromaDB only accepts str, int, float, bool, or None as metadata values.
        """
        processed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                # Create a copy of the document
                processed_doc = doc.copy()
                metadata = doc.get("metadata", {}).copy()
                
                # Safely handle text field
                text_raw = doc.get("text", "")
                if isinstance(text_raw, list):
                    # Join list elements with space separator
                    processed_doc["text"] = " ".join(str(item) for item in text_raw)
                elif text_raw is None:
                    processed_doc["text"] = ""
                else:
                    processed_doc["text"] = str(text_raw)
                
                # Convert list fields to strings in metadata
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
                        # Keep primitive types as-is, convert complex types to strings
                        metadata[key] = str(value) if not isinstance(value, (int, float, bool)) else value
                
                # Assign tier information
                tier = self.assign_document_tier(doc)
                metadata["tier"] = tier
                
                # Ensure required metadata fields exist with unique IDs
                if "doc_id" not in metadata:
                    # Create a unique doc_id based on content hash
                    content_hash = hashlib.md5(processed_doc["text"].encode()).hexdigest()[:8]
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
                
            except Exception as e:
                logger.error(f"Error preprocessing document {i}: {e}")
                # Skip problematic documents rather than failing entirely
                continue
        
        logger.info(f"✅ Preprocessed {len(processed_docs)} documents successfully")
        
        # Rebalance tier distribution if needed
        processed_docs = self.rebalance_tier_distribution(processed_docs)
        
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
        foundation_files = [
            "foundation_medical_data.json",
            "unified_dataset.json"
        ]
        
        all_docs = []
        
        for filename in foundation_files:
            file_path = foundation_dir / filename
            if file_path.exists():
                logger.info(f"Loading {filename}...")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        all_docs.extend(data)
                    elif isinstance(data, dict) and "documents" in data:
                        all_docs.extend(data["documents"])
                    
                    logger.info(f"Loaded {len(data)} documents from {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        logger.info(f"Total foundation documents loaded: {len(all_docs)}")
        return all_docs