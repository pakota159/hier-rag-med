#!/usr/bin/env python3
"""
Setup script for Hierarchical Reasoning System.
Completely updated version that handles PubMed/MTSamples/MeSH data properly.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger
from src.basic_reasoning.config import Config
from src.basic_reasoning.processing import HierarchicalDocumentProcessor
from src.basic_reasoning.retrieval import HierarchicalRetriever


def load_foundation_dataset() -> tuple[List[Dict], Dict[str, Any]]:
    """Load foundation dataset with enhanced validation."""
    logger.info("📖 Loading foundation dataset...")
    
    # Try to load from multiple possible locations
    possible_paths = [
        Path("data/foundation_dataset.json"),
        Path("data/foundation_dataset/unified_dataset.json"),
        Path("data/kg_raw/combined/all_medical_data.json"),
        Path("foundation_dataset.json")
    ]
    
    dataset_path = None
    for path in possible_paths:
        if path.exists():
            dataset_path = path
            break
    
    if not dataset_path:
        raise FileNotFoundError(
            "❌ Foundation dataset not found!\n"
            "Available paths checked:\n" + 
            "\n".join(f"   - {p}" for p in possible_paths) + 
            "\n\n🔧 Solutions:\n"
            "   1. Run: python fetch_foundation_data.py\n"
            "   2. Or: python fetch_data.py --source all\n"
            "   3. Check if file exists in any of the above locations"
        )
    
    logger.info(f"📂 Loading from: {dataset_path}")
    
    try:
        with open(dataset_path) as f:
            data = json.load(f)
        
        # Handle different data formats
        if isinstance(data, dict):
            if "documents" in data:
                all_docs = data["documents"]
            elif "data" in data:
                all_docs = data["data"]
            else:
                # Assume the dict values are the documents
                all_docs = []
                for value in data.values():
                    if isinstance(value, list):
                        all_docs.extend(value)
        elif isinstance(data, list):
            all_docs = data
        else:
            raise ValueError(f"Unexpected data format: {type(data)}")
        
        # Analyze dataset
        analysis = analyze_foundation_dataset(all_docs)
        
        logger.info(f"✅ Loaded {len(all_docs)} documents from {dataset_path.name}")
        
        return all_docs, analysis
        
    except Exception as e:
        logger.error(f"❌ Failed to load foundation dataset: {e}")
        raise


def analyze_foundation_dataset(data: List[Dict]) -> Dict[str, Any]:
    """Analyze foundation dataset characteristics."""
    if not data:
        return {"type": "empty", "therapeutic_focus": False, "sources": {}}
    
    sources = {}
    quality_indicators = {
        "evidence_based": 0,
        "clinical": 0,
        "synthetic": 0
    }
    
    for doc in data:
        metadata = doc.get("metadata", {})
        source = metadata.get("source", "unknown")
        
        # Count source occurrences
        sources[source] = sources.get(source, 0) + 1
        
        # Analyze quality indicators
        text = doc.get("text", "").lower()
        title = metadata.get("title", "").lower()
        
        # Evidence-based indicators
        if any(term in text + title for term in 
               ["evidence", "meta-analysis", "systematic review", "clinical trial", "rct"]):
            quality_indicators["evidence_based"] += 1
        
        # Clinical indicators
        if any(term in text + title for term in 
               ["clinical", "patient", "treatment", "diagnosis", "therapy"]):
            quality_indicators["clinical"] += 1
        
        # Synthetic indicators (exam-focused)
        if any(term in text + title for term in 
               ["case study", "reasoning", "differential", "multiple choice"]):
            quality_indicators["synthetic"] += 1
    
    # Determine dataset characteristics
    total_docs = len(data)
    therapeutic_sources = {"who_clinical_guidelines", "esc_cardiovascular_guidelines", 
                          "aha_acc_guidelines", "uspstf_preventive_guidelines", 
                          "uptodate_clinical_recommendations"}
    
    therapeutic_count = sum(sources.get(src, 0) for src in therapeutic_sources)
    therapeutic_ratio = therapeutic_count / total_docs if total_docs > 0 else 0
    
    if therapeutic_ratio > 0.7:
        dataset_type = "therapeutic"
    elif therapeutic_ratio > 0.3:
        dataset_type = "hybrid"
    else:
        dataset_type = "exam_focused"
    
    return {
        "type": dataset_type,
        "therapeutic_focus": therapeutic_ratio > 0.5,
        "sources": sources,
        "quality_indicators": quality_indicators,
        "therapeutic_ratio": therapeutic_ratio,
        "total_documents": total_docs
    }


def enhance_tier_mapping(organized_docs: Dict[str, List[Dict]], 
                        foundation_type: str) -> Dict[str, List[Dict]]:
    """Enhance tier mapping based on foundation dataset type."""
    
    if foundation_type == "therapeutic":
        logger.info("🎯 Optimizing tier mapping for therapeutic foundation")
        
        # For therapeutic datasets, ensure proper distribution
        # Move some Tier 2 docs to Tier 1 if Tier 1 is too small
        tier1_docs = organized_docs.get("pattern_recognition", [])
        tier2_docs = organized_docs.get("hypothesis_testing", [])
        
        if len(tier1_docs) < len(tier2_docs) * 0.3:  # Tier 1 should be at least 30% of Tier 2
            # Move USPSTF and some WHO docs to Tier 1
            docs_to_move = []
            remaining_tier2 = []
            
            for doc in tier2_docs:
                source = doc["metadata"].get("source", "")
                if source in ["uspstf_preventive_guidelines", "who_clinical_guidelines"]:
                    if len(docs_to_move) < len(tier2_docs) // 3:  # Move up to 1/3
                        docs_to_move.append(doc)
                        continue
                remaining_tier2.append(doc)
            
            if docs_to_move:
                organized_docs["pattern_recognition"].extend(docs_to_move)
                organized_docs["hypothesis_testing"] = remaining_tier2
                logger.info(f"🔄 Moved {len(docs_to_move)} preventive guidelines to Tier 1")
    
    elif foundation_type == "hybrid":
        logger.info("🔄 Optimizing tier mapping for hybrid foundation")
        # Hybrid datasets usually have good distribution already
        pass
    
    return organized_docs


def validate_tier_distribution(organized_docs: Dict[str, List[Dict]]) -> bool:
    """Validate that tier distribution is reasonable for hierarchical retrieval."""
    
    total_docs = sum(len(docs) for docs in organized_docs.values())
    
    if total_docs == 0:
        logger.error("❌ No documents found in any tier")
        return False
    
    tier1_count = len(organized_docs.get("pattern_recognition", []))
    tier2_count = len(organized_docs.get("hypothesis_testing", []))
    tier3_count = len(organized_docs.get("confirmation", []))
    
    # Check for empty tiers
    empty_tiers = []
    if tier1_count == 0:
        empty_tiers.append("Tier 1 (Pattern Recognition)")
    if tier2_count == 0:
        empty_tiers.append("Tier 2 (Hypothesis Testing)")
    if tier3_count == 0:
        empty_tiers.append("Tier 3 (Confirmation)")
    
    if empty_tiers:
        logger.error(f"❌ Empty tiers detected: {', '.join(empty_tiers)}")
        logger.error("This will cause hierarchical retrieval to fail!")
        logger.error("🔧 Solutions:")
        logger.error("   1. Fetch more diverse foundation data")
        logger.error("   2. Use hybrid dataset: python fetch_foundation_data.py --hybrid")
        logger.error("   3. Check your data sources and tier assignment logic")
        return False
    
    # Check for severely imbalanced tiers (one tier having >80% of documents)
    max_tier_ratio = max(tier1_count, tier2_count, tier3_count) / total_docs
    if max_tier_ratio > 0.8:
        logger.warning(f"⚠️ Tier distribution is severely imbalanced (max ratio: {max_tier_ratio:.1%})")
        logger.warning("This may reduce hierarchical retrieval effectiveness")
    
    # Log distribution
    logger.info("📊 Tier distribution validation:")
    logger.info(f"   Tier 1 (Pattern Recognition): {tier1_count} ({tier1_count/total_docs:.1%})")
    logger.info(f"   Tier 2 (Hypothesis Testing): {tier2_count} ({tier2_count/total_docs:.1%})")
    logger.info(f"   Tier 3 (Confirmation): {tier3_count} ({tier3_count/total_docs:.1%})")
    logger.info(f"   Total: {total_docs} documents")
    
    return True


def setup_hierarchical_system():
    """Main setup function for Hierarchical system."""
    start_time = time.time()
    
    logger.info("🚀 Starting Hierarchical System Setup")
    logger.info("=" * 70)
    
    # Initialize configuration
    try:
        config = Config()
        device_info = config.get_device_info()
        logger.info(f"✅ Loaded config: {device_info['environment']} on {device_info.get('cuda_device_name', 'CPU')}")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return False
    
    # Initialize components
    try:
        processor = HierarchicalDocumentProcessor(config.config["processing"])
        retriever = HierarchicalRetriever(config)
        logger.info("✅ Initialized Hierarchical components")
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        return False
    
    # Check existing collections
    try:
        retriever.load_hierarchical_collections()
        logger.info("✅ Hierarchical collections already exist")
        
        # Check if we should recreate
        response = input("\n🔄 Hierarchical collections already exist. Recreate? (y/N): ")
        if response.lower() != 'y':
            logger.info("✅ Using existing hierarchical collections")
            elapsed_time = time.time() - start_time
            logger.info(f"⏱️ Setup completed in {elapsed_time:.1f} seconds")
            return True
            
    except Exception:
        logger.info("📝 No existing collections found, creating new ones...")
    
    # Load and process foundation dataset
    try:
        all_docs, analysis = load_foundation_dataset()
        foundation_info = {
            "type": analysis["type"],
            "therapeutic_focus": analysis["therapeutic_focus"]
        }
        
        logger.info("📊 Foundation dataset analysis:")
        logger.info(f"   Total documents: {analysis['total_documents']}")
        logger.info(f"   Sources: {list(analysis['sources'].keys())}")
        logger.info(f"   Quality indicators:")
        logger.info(f"     - Evidence-based: {analysis['quality_indicators']['evidence_based']}")
        logger.info(f"     - Clinical: {analysis['quality_indicators']['clinical']}")
        logger.info(f"     - Synthetic: {analysis['quality_indicators']['synthetic']}")
        
        # Preprocess documents (fix metadata, assign tiers)
        logger.info("🔧 Preprocessing documents for ChromaDB compatibility...")
        all_docs = processor.preprocess_documents(all_docs)
        
        # Log tier assignment results
        tier_counts = {1: 0, 2: 0, 3: 0}
        for doc in all_docs:
            tier = doc["metadata"].get("tier", 2)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        logger.info("📊 Tier assignment results:")
        logger.info(f"   Tier 1 (Pattern Recognition): {tier_counts.get(1, 0)} documents")
        logger.info(f"   Tier 2 (Hypothesis Testing): {tier_counts.get(2, 0)} documents") 
        logger.info(f"   Tier 3 (Confirmation): {tier_counts.get(3, 0)} documents")
        
        # Organize by reasoning type
        organized_docs = processor.organize_by_reasoning_type(all_docs)
        
        # Enhance tier mapping based on dataset type
        organized_docs = enhance_tier_mapping(organized_docs, foundation_info["type"])
        
        # Validate tier distribution
        if not validate_tier_distribution(organized_docs):
            logger.error("❌ Tier distribution validation failed")
            logger.error("💡 Try using hybrid dataset for better distribution:")
            logger.error("   python fetch_foundation_data.py --hybrid --max-results 3000")
            return False
        
        logger.info("✅ Loaded and organized foundation dataset")
        
    except Exception as e:
        logger.error(f"❌ Failed to load foundation data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Log organization stats
    logger.info("📊 Final document organization:")
    total_docs = 0
    for tier_name, docs in organized_docs.items():
        logger.info(f"   {tier_name}: {len(docs)} documents")
        total_docs += len(docs)
    logger.info(f"   Total: {total_docs} documents")
    
    # Create collections and add documents
    try:
        logger.info("🔧 Creating hierarchical collections...")
        retriever.create_hierarchical_collections()
        
        logger.info("📝 Adding documents to hierarchical tiers...")
        retriever.add_documents_to_tiers(organized_docs)
        
        # Verify collections were created successfully
        stats = retriever.get_collection_stats()
        logger.info("📊 Collection statistics:")
        for collection_name, count in stats.items():
            logger.info(f"   {collection_name}: {count} documents")
        
        logger.info("✅ Hierarchical system setup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to create collections: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    elapsed_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info("🎉 HIERARCHICAL SYSTEM SETUP COMPLETE")
    logger.info(f"⏱️ Total time: {elapsed_time:.1f} seconds")
    logger.info(f"📊 Total documents processed: {total_docs}")
    logger.info(f"🎯 Foundation type: {foundation_info['type']}")
    logger.info("=" * 70)
    
    # Next steps
    print("\n🎯 Next Steps:")
    print("1. Test hierarchical retrieval:")
    print("   python -c \"from src.basic_reasoning.retrieval import HierarchicalRetriever; from src.basic_reasoning.config import Config; r = HierarchicalRetriever(Config()); r.load_hierarchical_collections(); print(r.hierarchical_search('diabetes symptoms'))\"")
    print("2. Run the Streamlit app:")
    print("   streamlit run src/basic_reasoning/streamlit_app.py --server.port 8503")
    print("3. Run MIRAGE evaluation:")
    print("   python src/evaluation/run_evaluation.py --benchmark mirage")
    
    return True


if __name__ == "__main__":
    try:
        success = setup_hierarchical_system()
        if not success:
            logger.error("❌ Hierarchical system setup failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Setup interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)