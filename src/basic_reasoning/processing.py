"""
Document processing for Basic Reasoning system.
Only processes foundation datasets from data/foundation/
UPDATED VERSION - Handles new foundation sources and proper tier assignment
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import re

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
        """Organize documents by reasoning type for tiered retrieval - UPDATED VERSION."""
        
        organized = {
            "pattern_recognition": [],  # Tier 1: Drug information, quick patterns
            "hypothesis_testing": [],   # Tier 2: Clinical guidelines, therapeutic reasoning
            "confirmation": []          # Tier 3: Clinical outcomes, case studies, evidence
        }
        
        tier1_count = 0
        tier2_count = 0
        tier3_count = 0
        
        for doc in documents:
            reasoning_type = doc["metadata"].get("reasoning_type", "")
            source = doc["metadata"].get("source", "")
            doc_type = doc["metadata"].get("type", "")
            text_lower = doc.get("text", "").lower()
            
            # Explicit tier assignment (if present)
            explicit_tier = doc["metadata"].get("tier", None)
            if explicit_tier:
                if explicit_tier == 1:
                    organized["pattern_recognition"].append(doc)
                    tier1_count += 1
                elif explicit_tier == 2:
                    organized["hypothesis_testing"].append(doc)
                    tier2_count += 1
                else:
                    organized["confirmation"].append(doc)
                    tier3_count += 1
                continue
            
            # TIER 1: Pattern Recognition (Drug info, quick patterns)
            if (reasoning_type in [
                "drug_information", "evidence_based_pharmacology", "drug_benefits"
            ] or source in [
                "drugbank", "therapeutic_pharmacology", "evidence_based_pharmacology"
            ] or doc_type in [
                "drug_profile", "drug_benefits", "drug_class_profile"
            ] or any(keyword in text_lower for keyword in [
                "medication", "dosage", "pharmacology", "drug class", "mechanism of action"
            ])):
                organized["pattern_recognition"].append(doc)
                tier1_count += 1
            
            # TIER 2: Hypothesis Testing (Clinical guidelines, therapeutic reasoning)
            elif (reasoning_type in [
                "evidence_based_medicine", "therapeutic_guideline", "knowledge_graph_guided", 
                "multi_step_diagnostic"
            ] or source in [
                "therapeutic_guidelines", "medreason", "msdiagnosis"
            ] or doc_type in [
                "therapeutic_guideline", "reasoning_chain", "treatment_hierarchy"
            ] or any(keyword in text_lower for keyword in [
                "first-line", "guideline", "recommendation", "evidence-based", "clinical trial"
            ])):
                organized["hypothesis_testing"].append(doc)
                tier2_count += 1
            
            # TIER 3: Confirmation (Clinical outcomes, case studies, evidence)
            elif (reasoning_type in [
                "clinical_outcomes", "real_world_evidence", "case_study"
            ] or source in [
                "clinical_outcomes", "pmc_patients"
            ] or doc_type in [
                "success_story", "clinical_symptoms", "case_study"
            ] or any(keyword in text_lower for keyword in [
                "case study", "patient outcome", "clinical success", "real-world"
            ])):
                organized["confirmation"].append(doc)
                tier3_count += 1
            
            # Legacy source handling
            elif source in ["medreason", "msdiagnosis"]:
                organized["hypothesis_testing"].append(doc)
                tier2_count += 1
            elif source in ["pmc_patients"]:
                organized["confirmation"].append(doc)
                tier3_count += 1
            elif source in ["drugbank"]:
                organized["pattern_recognition"].append(doc)
                tier1_count += 1
            
            # Default assignment based on content analysis
            else:
                # Analyze text content for tier assignment
                if self._is_drug_related(text_lower):
                    organized["pattern_recognition"].append(doc)
                    tier1_count += 1
                elif self._is_guideline_related(text_lower):
                    organized["hypothesis_testing"].append(doc)
                    tier2_count += 1
                else:
                    organized["confirmation"].append(doc)
                    tier3_count += 1
        
        # Log initial distribution
        logger.info(f"📊 Initial document distribution:")
        logger.info(f"   Tier 1 (Pattern Recognition): {tier1_count}")
        logger.info(f"   Tier 2 (Hypothesis Testing): {tier2_count}")
        logger.info(f"   Tier 3 (Confirmation): {tier3_count}")
        
        # Ensure balanced distribution (prevent empty tiers)
        total_docs = len(documents)
        if total_docs > 0:
            organized = self._rebalance_tiers(organized, total_docs)
        
        # Final logging
        logger.info(f"📊 Final organized documents by reasoning tiers:")
        logger.info(f"   Tier 1 (Pattern Recognition): {len(organized['pattern_recognition'])}")
        logger.info(f"   Tier 2 (Hypothesis Testing): {len(organized['hypothesis_testing'])}")
        logger.info(f"   Tier 3 (Confirmation): {len(organized['confirmation'])}")
        
        return organized

    def _is_drug_related(self, text: str) -> bool:
        """Check if text is primarily about drugs/medications."""
        drug_keywords = [
            "drug", "medication", "therapy", "treatment", "dosage", "pharmacology",
            "mechanism", "therapeutic", "prescription", "pharmaceutical", "clinical pharmacology"
        ]
        keyword_count = sum(1 for keyword in drug_keywords if keyword in text)
        return keyword_count >= 2

    def _is_guideline_related(self, text: str) -> bool:
        """Check if text is primarily about clinical guidelines."""
        guideline_keywords = [
            "guideline", "recommendation", "first-line", "evidence-based", "clinical trial",
            "systematic review", "meta-analysis", "consensus", "standard of care"
        ]
        keyword_count = sum(1 for keyword in guideline_keywords if keyword in text)
        return keyword_count >= 2

    def _rebalance_tiers(self, organized: Dict[str, List[Dict]], total_docs: int) -> Dict[str, List[Dict]]:
        """Rebalance tiers to ensure no tier is empty and maintain reasonable distribution."""
        
        # Target distribution: 30% Tier1, 35% Tier2, 35% Tier3
        target_tier1 = max(int(total_docs * 0.30), 1)
        target_tier2 = max(int(total_docs * 0.35), 1)
        target_tier3 = max(int(total_docs * 0.35), 1)
        
        current_tier1 = len(organized["pattern_recognition"])
        current_tier2 = len(organized["hypothesis_testing"])
        current_tier3 = len(organized["confirmation"])
        
        # If any tier is empty, redistribute
        if current_tier1 == 0 or current_tier2 == 0 or current_tier3 == 0:
            logger.info("🔄 Rebalancing tiers to prevent empty tiers...")
            
            # Collect all documents
            all_docs = (organized["pattern_recognition"] + 
                       organized["hypothesis_testing"] + 
                       organized["confirmation"])
            
            # Redistribute evenly
            organized["pattern_recognition"] = all_docs[:target_tier1]
            organized["hypothesis_testing"] = all_docs[target_tier1:target_tier1 + target_tier2]
            organized["confirmation"] = all_docs[target_tier1 + target_tier2:]
            
            logger.info(f"🔄 Redistributed documents:")
            logger.info(f"   Target Tier 1: {target_tier1}, Actual: {len(organized['pattern_recognition'])}")
            logger.info(f"   Target Tier 2: {target_tier2}, Actual: {len(organized['hypothesis_testing'])}")
            logger.info(f"   Target Tier 3: {target_tier3}, Actual: {len(organized['confirmation'])}")
        
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

    def analyze_dataset_quality(self, documents: List[Dict]) -> Dict:
        """Analyze the quality and composition of the foundation dataset."""
        
        analysis = {
            "total_documents": len(documents),
            "sources": {},
            "reasoning_types": {},
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