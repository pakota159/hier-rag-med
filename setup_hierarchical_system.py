#!/usr/bin/env python3
"""
Setup script for Hierarchical Reasoning System.
ONLY processes foundation datasets - NO KG data.
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


def find_foundation_dataset() -> Path:
    """Find the foundation dataset file (single combined JSON)."""
    logger.info("🔍 Looking for foundation dataset...")
    
    # Check foundation dataset locations in priority order
    candidate_paths = [
        # Primary location
        Path("data/foundation_dataset/foundation_medical_data.json"),
        
        # Timestamped files (most recent first)
        *sorted(Path("data/foundation_dataset").glob("foundation_specialty_rebalanced_*.json"), reverse=True),
        
        # Alternative locations
        Path("data/foundation/foundation_medical_data.json"),
        Path("data/foundation_medical_data.json"),
        Path("data/foundation_dataset.json"),
        Path("data/foundation_dataset/unified_dataset.json"),
        Path("foundation_dataset.json"),
        
        # Other timestamped locations
        *sorted(Path("data").glob("foundation_specialty_rebalanced_*.json"), reverse=True),
        *sorted(Path("data/foundation").glob("foundation_specialty_rebalanced_*.json"), reverse=True),
    ]
    
    for path in candidate_paths:
        if path.exists() and path.is_file():
            # Verify it's not KG data
            if "kg_raw" in str(path):
                logger.error(f"❌ CRITICAL ERROR: Found KG data instead of foundation data!")
                logger.error(f"   File: {path}")
                logger.error("🚫 KG DATA IS NOT ACCEPTABLE")
                logger.error("   Contains only basic pubmed/mtsamples/mesh data")
                logger.error("")
                logger.error("✅ REQUIRED: Foundation datasets from fetch_foundation_data.py")
                logger.error("🔧 SOLUTION: Run foundation data fetcher first:")
                logger.error("   python fetch_foundation_data.py --max-results 5000 --email your@email.com")
                raise ValueError("KG data rejected - foundation datasets required")
            
            logger.info(f"   ✅ Found foundation dataset: {path}")
            return path
    
    # No foundation dataset found
    logger.error("❌ CRITICAL ERROR: No foundation datasets found!")
    logger.error("")
    logger.error("🚫 FOUNDATION DATASETS ARE REQUIRED")
    logger.error("   This system requires comprehensive medical foundation datasets")
    logger.error("   Single combined JSON file with all medical sources")
    logger.error("")
    logger.error("📋 SOLUTION: Create foundation datasets first!")
    logger.error("")
    logger.error("🔧 Run one of these commands:")
    logger.error("   # Quick test (1K documents, ~5 minutes)")
    logger.error("   python fetch_foundation_data.py --quick --max-results 1000 --email your@email.com")
    logger.error("")
    logger.error("   # Medium dataset (5K documents, ~15 minutes)")
    logger.error("   python fetch_foundation_data.py --max-results 5000 --email your@email.com")
    logger.error("")
    logger.error("   # Full dataset (50K documents, ~60 minutes)")
    logger.error("   python fetch_foundation_data.py --max-results 50000 --email your@email.com")
    logger.error("")
    logger.error("💡 This creates a combined file with all medical sources:")
    logger.error("   - WHO, ESC, AHA/ACC, USPSTF, UpToDate guidelines")
    logger.error("   - MedReason, MSDiagnosis, PMC, DrugBank datasets")
    logger.error("   - ACOG, IDSA specialty guidelines")
    logger.error("   - Specialty-focused PubMed abstracts (18+ specialties)")
    logger.error("")
    logger.error("❌ SETUP CANNOT CONTINUE")
    raise FileNotFoundError("Foundation datasets required - run fetch_foundation_data.py first")


def load_foundation_dataset(dataset_path: Path) -> tuple[List[Dict], Dict[str, Any]]:
    """Load foundation dataset from single combined JSON file."""
    logger.info(f"📖 Loading foundation dataset: {dataset_path}")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Cannot read foundation dataset file")
        logger.error(f"   File: {dataset_path}")
        logger.error(f"   Error: {e}")
        logger.error("")
        logger.error("🔧 POSSIBLE CAUSES:")
        logger.error("   1. File is corrupted")
        logger.error("   2. File is incomplete")
        logger.error("   3. Insufficient permissions")
        logger.error("   4. Disk space issues")
        logger.error("")
        logger.error("🔧 SOLUTIONS:")
        logger.error("   1. Re-create foundation dataset:")
        logger.error("      python fetch_foundation_data.py --max-results 5000")
        logger.error("   2. Check file permissions and disk space")
        logger.error("")
        logger.error("❌ SETUP TERMINATED")
        raise RuntimeError(f"Cannot read foundation dataset: {e}")
    
    # Parse document structure
    documents = []
    
    if isinstance(data, list):
        documents = data
    elif isinstance(data, dict):
        if "documents" in data:
            documents = data["documents"]
        elif "data" in data:
            documents = data["data"]
        else:
            # Try to extract documents from dict values
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict) and "text" in value[0]:
                        documents.extend(value)
                elif isinstance(value, dict) and "text" in value:
                    documents.append(value)
    else:
        logger.error(f"❌ CRITICAL ERROR: Unexpected data format in foundation dataset")
        logger.error(f"   Expected: list or dict with documents")
        logger.error(f"   Found: {type(data)}")
        logger.error("")
        logger.error("🔧 SOLUTION: Re-create foundation dataset with correct format")
        logger.error("   python fetch_foundation_data.py --max-results 5000")
        raise ValueError(f"Invalid foundation dataset format: {type(data)}")
    
    if not documents:
        logger.error("❌ CRITICAL ERROR: No documents found in foundation dataset")
        logger.error(f"   File: {dataset_path}")
        logger.error("   The dataset file exists but contains no usable documents")
        logger.error("")
        logger.error("🔧 SOLUTION: Re-create foundation dataset")
        logger.error("   python fetch_foundation_data.py --max-results 5000")
        raise ValueError("Foundation dataset contains no documents")
    
    # Process and validate documents
    valid_docs = []
    source_counts = {}
    
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            logger.warning(f"   Skipping invalid document {i}: not a dict")
            continue
            
        if not doc.get("text"):
            logger.warning(f"   Skipping document {i}: no text content")
            continue
        
        # Ensure metadata exists
        if "metadata" not in doc:
            doc["metadata"] = {}
        
        # Count sources
        source = doc["metadata"].get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        
        valid_docs.append(doc)
    
    logger.info(f"✅ Loaded {len(valid_docs)} valid documents from foundation dataset")
    
    if len(valid_docs) < 100:
        logger.warning(f"⚠️ Low document count: {len(valid_docs)}")
        logger.warning("   Consider fetching more comprehensive data")
    
    # Analyze dataset
    analysis = analyze_foundation_dataset(valid_docs, source_counts)
    
    return valid_docs, analysis


def analyze_foundation_dataset(documents: List[Dict], source_counts: Dict[str, int]) -> Dict[str, Any]:
    """Analyze foundation dataset characteristics."""
    total_docs = len(documents)
    
    if total_docs == 0:
        return {
            "type": "empty",
            "therapeutic_focus": False,
            "sources": source_counts,
            "total_documents": 0,
            "data_quality": "none"
        }
    
    # Quality indicators
    quality_indicators = {
        "evidence_based": 0,
        "clinical": 0,
        "synthetic": 0,
        "guideline_based": 0,
        "research_based": 0,
        "specialty_focused": 0
    }
    
    # Analyze content
    for doc in documents:
        # Handle None values safely
        text = doc.get("text") or ""
        title = doc.get("metadata", {}).get("title") or ""
        
        # Ensure strings before calling lower()
        text = str(text).lower() if text else ""
        title = str(title).lower() if title else ""
        content = text + " " + title
        
        # Evidence-based indicators
        if any(term in content for term in 
               ["evidence", "meta-analysis", "systematic review", "clinical trial", "rct"]):
            quality_indicators["evidence_based"] += 1
        
        # Clinical indicators
        if any(term in content for term in 
               ["clinical", "patient", "treatment", "diagnosis", "therapy"]):
            quality_indicators["clinical"] += 1
        
        # Synthetic/educational indicators
        if any(term in content for term in 
               ["case study", "reasoning", "differential", "multiple choice"]):
            quality_indicators["synthetic"] += 1
        
        # Guideline-based indicators
        if any(term in content for term in 
               ["guideline", "recommendation", "consensus", "standard", "protocol"]):
            quality_indicators["guideline_based"] += 1
        
        # Research-based indicators
        if any(term in content for term in 
               ["pubmed", "pmid", "research", "study", "journal", "abstract"]):
            quality_indicators["research_based"] += 1
        
        # Specialty-focused indicators
        if any(specialty in content for specialty in 
               ["cardiology", "surgery", "obstetrics", "gynecology", "infectious", 
                "emergency", "psychiatry", "dermatology", "gastroenterology", 
                "pediatrics", "oncology", "neurology", "endocrinology"]):
            quality_indicators["specialty_focused"] += 1
    
    # Determine dataset characteristics
    therapeutic_sources = {
        "who_guidelines", "esc_guidelines", "aha_acc_guidelines", 
        "uspstf_guidelines", "uptodate_guidelines", "acog_guidelines", "idsa_guidelines"
    }
    
    research_sources = {
        "pubmed", "pubmed_specialty", "pmc_patients"
    }
    
    exam_sources = {
        "medreason", "msdiagnosis", "mtsamples"
    }
    
    therapeutic_count = sum(source_counts.get(src, 0) for src in therapeutic_sources)
    research_count = sum(source_counts.get(src, 0) for src in research_sources)
    exam_count = sum(source_counts.get(src, 0) for src in exam_sources)
    
    therapeutic_ratio = therapeutic_count / total_docs if total_docs > 0 else 0
    research_ratio = research_count / total_docs if total_docs > 0 else 0
    exam_ratio = exam_count / total_docs if total_docs > 0 else 0
    
    # Determine dataset type
    if therapeutic_ratio > 0.4:
        dataset_type = "therapeutic_focused"
    elif research_ratio > 0.4:
        dataset_type = "research_focused"
    elif exam_ratio > 0.4:
        dataset_type = "exam_focused"
    elif (therapeutic_ratio + research_ratio) > 0.6:
        dataset_type = "hybrid_clinical"
    else:
        dataset_type = "mixed_sources"
    
    # Data quality assessment
    high_quality_count = (quality_indicators["evidence_based"] + 
                         quality_indicators["guideline_based"])
    
    if high_quality_count > total_docs * 0.5:
        data_quality = "high"
    elif high_quality_count > total_docs * 0.2:
        data_quality = "medium"
    else:
        data_quality = "mixed"
    
    return {
        "type": dataset_type,
        "therapeutic_focus": therapeutic_ratio > 0.3,
        "sources": source_counts,
        "source_distribution": {
            "therapeutic": therapeutic_ratio,
            "research": research_ratio,
            "exam": exam_ratio
        },
        "quality_indicators": quality_indicators,
        "total_documents": total_docs,
        "data_quality": data_quality
    }


def enhance_tier_mapping(organized_docs: Dict[str, List[Dict]], 
                        foundation_type: str) -> Dict[str, List[Dict]]:
    """Enhance tier mapping based on foundation dataset type."""
    if foundation_type in ["therapeutic_focused", "hybrid_clinical"]:
        logger.info("🎯 Applying therapeutic-focused tier mapping")
        
        # Move high-quality guidelines to tier 3 (confirmation)
        tier2_to_tier3 = []
        for doc in organized_docs.get("hypothesis_testing", []):
            text = doc.get("text", "").lower()
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "")
            
            # Check for high-quality guideline indicators
            if (any(term in text for term in ["guideline", "recommendation", "consensus"]) or
                any(source_type in source for source_type in ["who", "esc", "aha", "uspstf", "acog", "idsa"])):
                
                # Move to tier 3
                doc["metadata"]["tier"] = 3
                tier2_to_tier3.append(doc)
        
        # Remove from tier 2 and add to tier 3
        organized_docs["hypothesis_testing"] = [
            doc for doc in organized_docs.get("hypothesis_testing", [])
            if doc not in tier2_to_tier3
        ]
        
        if "confirmation" not in organized_docs:
            organized_docs["confirmation"] = []
        organized_docs["confirmation"].extend(tier2_to_tier3)
        
        if tier2_to_tier3:
            logger.info(f"🔄 Enhanced tier mapping: moved {len(tier2_to_tier3)} guideline docs to confirmation tier")
    
    return organized_docs


def validate_tier_distribution(organized_docs: Dict[str, List[Dict]]) -> bool:
    """Validate tier distribution for hierarchical effectiveness."""
    tier1_count = len(organized_docs.get("pattern_recognition", []))
    tier2_count = len(organized_docs.get("hypothesis_testing", []))
    tier3_count = len(organized_docs.get("confirmation", []))
    total_docs = tier1_count + tier2_count + tier3_count
    
    if total_docs == 0:
        logger.error("❌ CRITICAL ERROR: No documents found after tier organization")
        logger.error("🔧 This indicates a processing failure")
        return False
    
    # Check for empty critical tiers
    if tier2_count == 0:
        logger.error("❌ CRITICAL ERROR: Tier 2 (Hypothesis Testing) is empty")
        logger.error("   This tier is essential for hierarchical reasoning")
        logger.error("")
        logger.error("🔧 SOLUTIONS:")
        logger.error("   1. Fetch more diverse foundation data:")
        logger.error("      python fetch_foundation_data.py --max-results 10000")
        logger.error("   2. Check foundation dataset quality")
        logger.error("   3. Verify document processing logic")
        return False
    
    # Warnings for empty tiers
    if tier1_count == 0:
        logger.warning("⚠️ Tier 1 (Pattern Recognition) is empty")
        logger.warning("   Consider adding more case studies and diagnostic patterns")
    
    if tier3_count == 0:
        logger.warning("⚠️ Tier 3 (Confirmation) is empty")
        logger.warning("   Consider adding more guidelines and evidence-based sources")
    
    # Check for severely imbalanced tiers
    max_tier_ratio = max(tier1_count, tier2_count, tier3_count) / total_docs
    if max_tier_ratio > 0.8:
        logger.warning(f"⚠️ Tier distribution is imbalanced (max ratio: {max_tier_ratio:.1%})")
        logger.warning("   This may reduce hierarchical retrieval effectiveness")
    
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
        logger.error(f"❌ CRITICAL ERROR: Failed to load config: {e}")
        logger.error("🔧 Check your configuration files and environment")
        raise RuntimeError(f"Configuration loading failed: {e}")
    
    # Initialize components
    try:
        processor = HierarchicalDocumentProcessor(config.config["processing"])
        retriever = HierarchicalRetriever(config)
        logger.info("✅ Initialized Hierarchical components")
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Failed to initialize components: {e}")
        logger.error("🔧 Check your system dependencies and configuration")
        raise RuntimeError(f"Component initialization failed: {e}")
    
    # Check existing collections
    try:
        retriever.load_hierarchical_collections()
        logger.info("✅ Hierarchical collections already exist")
        
        # Ask if we should recreate
        response = input("\n🔄 Hierarchical collections already exist. Recreate? (y/N): ")
        if response.lower() != 'y':
            logger.info("✅ Using existing hierarchical collections")
            elapsed_time = time.time() - start_time
            logger.info(f"⏱️ Setup completed in {elapsed_time:.1f} seconds")
            return True
            
    except Exception:
        logger.info("📝 No existing collections found, creating new ones...")
    
    # Find and load foundation dataset
    try:
        dataset_path = find_foundation_dataset()
        all_docs, analysis = load_foundation_dataset(dataset_path)
        
        foundation_info = {
            "type": analysis["type"],
            "therapeutic_focus": analysis["therapeutic_focus"]
        }
        
        logger.info("📊 Foundation dataset analysis:")
        logger.info(f"   Total documents: {analysis['total_documents']}")
        logger.info(f"   Dataset type: {analysis['type']}")
        logger.info(f"   Data quality: {analysis['data_quality']}")
        logger.info(f"   Sources: {list(analysis['sources'].keys())}")
        logger.info(f"   Quality indicators:")
        logger.info(f"     - Evidence-based: {analysis['quality_indicators']['evidence_based']}")
        logger.info(f"     - Clinical: {analysis['quality_indicators']['clinical']}")
        logger.info(f"     - Guideline-based: {analysis['quality_indicators']['guideline_based']}")
        logger.info(f"     - Research-based: {analysis['quality_indicators']['research_based']}")
        logger.info(f"     - Specialty-focused: {analysis['quality_indicators']['specialty_focused']}")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Foundation dataset loading failed: {e}")
        raise
    
    # Preprocess documents
    try:
        logger.info("🔧 Preprocessing documents for ChromaDB compatibility...")
        all_docs = processor.preprocess_documents(all_docs)
        
        # Log tier assignment results
        tier_counts = {1: 0, 2: 0, 3: 0}
        for doc in all_docs:
            tier = doc["metadata"].get("tier", 2)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        logger.info("📊 Initial tier assignment:")
        logger.info(f"   Tier 1 (Pattern Recognition): {tier_counts.get(1, 0)} documents")
        logger.info(f"   Tier 2 (Hypothesis Testing): {tier_counts.get(2, 0)} documents") 
        logger.info(f"   Tier 3 (Confirmation): {tier_counts.get(3, 0)} documents")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Document preprocessing failed: {e}")
        logger.error("🔧 Check document processor configuration")
        raise RuntimeError(f"Document preprocessing failed: {e}")
    
    # Organize by reasoning type
    try:
        organized_docs = processor.organize_by_reasoning_type(all_docs)
        
        # Enhance tier mapping based on dataset type
        organized_docs = enhance_tier_mapping(organized_docs, foundation_info["type"])
        
        # Validate tier distribution
        if not validate_tier_distribution(organized_docs):
            logger.error("❌ CRITICAL ERROR: Tier distribution validation failed")
            logger.error("🔧 Dataset inadequate for hierarchical reasoning")
            raise RuntimeError("Tier distribution validation failed")
        
        logger.info("✅ Loaded and organized foundation dataset")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Document organization failed: {e}")
        raise
    
    # Display final organization
    logger.info("📊 Final document organization:")
    total_docs = 0
    for tier_name, docs in organized_docs.items():
        count = len(docs)
        total_docs += count
        logger.info(f"   {tier_name}: {count} documents")
    logger.info(f"   Total: {total_docs} documents")
    
    # Create hierarchical collections
    try:
        logger.info("🔧 Creating hierarchical collections...")
        retriever.create_hierarchical_collections()
        
        logger.info("📝 Adding documents to hierarchical tiers...")
        retriever.add_documents_to_tiers(organized_docs)
        
        # Log collection statistics
        logger.info("📊 Collection statistics:")
        collection_names = ["tier1_pattern_recognition", "tier2_hypothesis_testing", "tier3_confirmation"]
        for collection_name in collection_names:
            try:
                collection = retriever.client.get_collection(collection_name)
                count = collection.count()
                logger.info(f"   {collection_name}: {count} documents")
            except Exception as e:
                logger.warning(f"   Could not get count for {collection_name}: {e}")
        
        logger.info("✅ Hierarchical system setup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Hierarchical collection creation failed: {e}")
        logger.error("🔧 Check ChromaDB installation and system resources")
        raise RuntimeError(f"Collection creation failed: {e}")
    
    # Final summary
    elapsed_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info("🎉 HIERARCHICAL SYSTEM SETUP COMPLETE")
    logger.info(f"⏱️ Total time: {elapsed_time:.1f} seconds")
    logger.info(f"📊 Total documents processed: {total_docs}")
    logger.info(f"🎯 Foundation type: {foundation_info['type']}")
    logger.info(f"📋 Sources processed: {len(analysis['sources'])} different medical sources")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        success = setup_hierarchical_system()
        
        if success:
            print("\n🎯 Next Steps:")
            print("1. Test hierarchical retrieval:")
            print('   python -c "from src.basic_reasoning.retrieval import HierarchicalRetriever; from src.basic_reasoning.config import Config; r = HierarchicalRetriever(Config()); r.load_hierarchical_collections(); print(r.hierarchical_search(\'diabetes symptoms\'))"')
            print("2. Run the Streamlit app:")
            print("   streamlit run src/basic_reasoning/streamlit_app.py --server.port 8503")
            print("3. Run MIRAGE evaluation:")
            print("   python src/evaluation/run_evaluation.py --benchmark mirage")
        
    except FileNotFoundError as e:
        logger.error("❌ SETUP FAILED: Missing foundation data")
        print("\n🔧 SOLUTION: Create foundation datasets first!")
        print("python fetch_foundation_data.py --max-results 5000 --email your@email.com")
        sys.exit(1)
        
    except ValueError as e:
        logger.error("❌ SETUP FAILED: Invalid data")
        print("\n🔧 SOLUTION: Check data quality and re-create foundation datasets")
        print("python fetch_foundation_data.py --max-results 5000 --email your@email.com")
        sys.exit(1)
        
    except RuntimeError as e:
        logger.error("❌ SETUP FAILED: Runtime error")
        print("\n🔧 SOLUTION: Check system resources and configuration")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ SETUP FAILED: Unexpected error: {e}")
        import traceback
        logger.error("📋 Full traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)