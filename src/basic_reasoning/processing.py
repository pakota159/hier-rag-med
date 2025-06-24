"""
Processing module for Basic Reasoning system.
Implements hierarchical document processing for medical knowledge.
"""

from typing import Dict, List
import json
from pathlib import Path
from loguru import logger


class HierarchicalDocumentProcessor:
    """Document processor for hierarchical medical reasoning."""

    def __init__(self, config: Dict):
        """Initialize processor."""
        self.config = config

    def preprocess_documents(self, documents: List[Dict]) -> List[Dict]:
        """Preprocess documents for ChromaDB compatibility and assign tiers."""
        logger.info(f"🔧 Preprocessing {len(documents)} documents for hierarchical processing")
        
        processed_docs = []
        for i, doc in enumerate(documents):
            try:
                # Create unique document ID
                doc_id = f"doc_{i:06d}"
                
                # Ensure required fields exist
                if "text" not in doc or not doc["text"]:
                    continue
                
                if "metadata" not in doc:
                    doc["metadata"] = {}
                
                # Convert text to string if needed
                text = str(doc["text"]) if doc["text"] else ""
                if len(text.strip()) < 10:  # Skip very short documents
                    continue
                
                # Clean metadata for ChromaDB compatibility
                clean_metadata = self._clean_metadata(doc["metadata"])
                
                # Assign tier based on medical content
                tier = self._assign_medical_tier(doc)
                
                processed_doc = {
                    "text": text,
                    "metadata": {
                        **clean_metadata,
                        "doc_id": doc_id,
                        "tier": tier,
                        "tier_chunk_id": f"tier{tier}_{i}"
                    }
                }
                
                processed_docs.append(processed_doc)
                
            except Exception as e:
                logger.warning(f"Failed to process document {i}: {e}")
                continue
        
        logger.info(f"✅ Preprocessed {len(processed_docs)} documents")
        return processed_docs

    def _clean_metadata(self, metadata: Dict) -> Dict:
        """Clean metadata for ChromaDB compatibility."""
        clean_meta = {}
        
        for key, value in metadata.items():
            if value is None:
                # Skip None values completely
                continue
            elif isinstance(value, (str, int, float, bool)):
                # Only add if not empty string
                if isinstance(value, str) and value.strip() == "":
                    continue
                clean_meta[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                if value:
                    clean_meta[key] = ", ".join(str(item) for item in value if item is not None)
                # Skip empty lists
            elif isinstance(value, dict):
                # Convert dicts to string representation, skip if empty
                if value:
                    clean_meta[key] = str(value)
            else:
                # Convert other types to string, skip if results in empty
                str_value = str(value).strip()
                if str_value and str_value.lower() not in ["none", "null", ""]:
                    clean_meta[key] = str_value
        
        return clean_meta

    def _assign_medical_tier(self, doc: Dict) -> int:
        """Assign medical knowledge tier based on content analysis."""
        metadata = doc.get("metadata", {})
        
        # Handle text field safely
        text_raw = doc.get("text", "")
        if isinstance(text_raw, list):
            text = " ".join(str(item) for item in text_raw).lower()
        else:
            text = str(text_raw).lower()
        
        # Handle title field safely
        title_raw = metadata.get("title", "")
        if isinstance(title_raw, list):
            title = " ".join(str(item) for item in title_raw).lower()
        else:
            title = str(title_raw).lower()
        
        combined_text = f"{title} {text}"
        
        # TIER 3 (Evidence/Confirmation) - High-evidence medical content
        tier3_indicators = [
            "meta-analysis", "systematic review", "clinical trial", "randomized controlled",
            "evidence-based", "cochrane", "guidelines", "consensus", "recommendation",
            "grade evidence", "level of evidence", "clinical practice guideline",
            "acog", "aha", "acc", "esc", "who guidelines", "fda approved",
            "grade a", "grade b", "class i", "level 1 evidence",
            "peer-reviewed", "pubmed", "medline", "systematic", "pooled analysis"
        ]
        
        # TIER 1 (Pattern Recognition) - Basic medical concepts
        tier1_indicators = [
            "definition", "anatomy", "physiology", "basic", "introduction",
            "what is", "overview", "classification", "types of", "categories",
            "symptoms", "signs", "presentation", "manifestation", "features",
            "etiology", "causes", "risk factors", "epidemiology", "prevalence",
            "pathophysiology", "mechanism", "normal", "abnormal", "structure",
            "function", "cell", "tissue", "organ", "system", "terminology"
        ]
        
        # Count indicators
        tier3_count = sum(1 for indicator in tier3_indicators if indicator in combined_text)
        tier1_count = sum(1 for indicator in tier1_indicators if indicator in combined_text)
        
        # Decision logic for medical knowledge
        if tier3_count >= 2:
            return 3
        elif tier1_count >= 3:
            return 1
        elif any(word in combined_text for word in ["clinical", "diagnosis", "treatment", "therapy", "management", "patient"]):
            return 2
        else:
            # Default distribution for medical content
            doc_hash = hash(combined_text) % 100
            if doc_hash < 20:
                return 1
            elif doc_hash < 75:
                return 2
            else:
                return 3

    def organize_by_reasoning_type(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize documents by medical reasoning tiers."""
        logger.info(f"📊 Organizing {len(documents)} documents by medical reasoning tiers")
        
        organized = {
            "pattern_recognition": [],  # Tier 1 - Basic medical knowledge
            "hypothesis_testing": [],   # Tier 2 - Clinical reasoning
            "confirmation": []          # Tier 3 - Evidence-based medicine
        }
        
        tier_counts = {1: 0, 2: 0, 3: 0}
        tier_id_counters = {1: 0, 2: 0, 3: 0}
        
        for doc in documents:
            tier = doc["metadata"].get("tier", 2)
            
            # Ensure unique IDs within tiers
            tier_id_counters[tier] += 1
            doc["metadata"]["tier_chunk_id"] = f"tier{tier}_{tier_id_counters[tier]:06d}"
            
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
                # Fallback for invalid tiers
                organized["hypothesis_testing"].append(doc)
                tier_counts[2] += 1
        
        # Log medical knowledge distribution
        logger.info(f"📊 Medical knowledge tier distribution:")
        logger.info(f"   Tier 1 (Basic Medical Knowledge): {tier_counts[1]} docs")
        logger.info(f"   Tier 2 (Clinical Reasoning): {tier_counts[2]} docs")
        logger.info(f"   Tier 3 (Evidence-Based Medicine): {tier_counts[3]} docs")
        
        # Validate distribution for medical Q&A
        total_docs = sum(tier_counts.values())
        if total_docs > 0:
            tier1_pct = (tier_counts[1] / total_docs) * 100
            tier2_pct = (tier_counts[2] / total_docs) * 100
            tier3_pct = (tier_counts[3] / total_docs) * 100
            
            logger.info(f"📊 Distribution percentages:")
            logger.info(f"   Tier 1: {tier1_pct:.1f}% | Tier 2: {tier2_pct:.1f}% | Tier 3: {tier3_pct:.1f}%")
            
            # Warn about imbalanced distribution
            if tier1_pct < 10 or tier3_pct < 10:
                logger.warning("⚠️ Imbalanced tier distribution may affect medical Q&A performance")
        
        return organized

    def load_foundation_dataset(self, foundation_dir: Path) -> List[Dict[str, str]]:
        """Load foundation dataset for medical knowledge."""
        logger.info(f"📚 Loading medical foundation dataset from {foundation_dir}")
        
        all_docs = []
        
        # Look for unified dataset first
        unified_file = foundation_dir / "unified_dataset.json"
        if unified_file.exists():
            logger.info("📖 Loading unified foundation dataset")
            with open(unified_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_docs.extend(data)
                else:
                    logger.warning("Unified dataset is not a list format")
        
        # Load individual datasets if no unified dataset
        if not all_docs:
            dataset_files = [
                "medreason_documents.json",
                "msdiagnosis_documents.json", 
                "pmc_patients_documents.json",
                "drugbank_documents.json"
            ]
            
            for filename in dataset_files:
                file_path = foundation_dir / filename
                if file_path.exists():
                    logger.info(f"📖 Loading {filename}")
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                all_docs.extend(data)
                                logger.info(f"   Added {len(data)} documents from {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to load {filename}: {e}")
        
        logger.info(f"✅ Loaded {len(all_docs)} medical documents total")
        return all_docs