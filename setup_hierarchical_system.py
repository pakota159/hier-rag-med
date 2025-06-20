#!/usr/bin/env python3
"""
Setup script to create Hierarchical system collections.
COMPLETELY UPDATED VERSION - Works with new foundation fetchers
"""

import sys
import os
from pathlib import Path
from typing import List, Dict
import json
import time
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    from basic_reasoning.config import Config
    from basic_reasoning.processing import HierarchicalDocumentProcessor
    from basic_reasoning.retrieval import HierarchicalRetriever
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory.")
    sys.exit(1)


def check_foundation_data():
    """Check if foundation dataset exists."""
    foundation_dir = Path("data/foundation")
    foundation_file = foundation_dir / "foundation_medical_data.json"
    
    if foundation_dir.exists() and foundation_file.exists():
        try:
            with open(foundation_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                return {"exists": False, "path": foundation_file, "count": 0, "error": "Foundation file is empty"}
            
            return {
                "exists": True,
                "path": foundation_file,
                "count": len(data) if isinstance(data, list) else 1,
                "error": None
            }
        except Exception as e:
            return {"exists": False, "path": foundation_file, "count": 0, "error": f"Could not read foundation file: {e}"}
    
    return {"exists": False, "path": foundation_file, "count": 0, "error": "Foundation directory or file doesn't exist"}


def validate_tier_distribution(organized_docs: Dict[str, List[Dict]]) -> bool:
    """Validate that tier distribution is reasonable."""
    
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
        return False
    
    # Log distribution percentages
    tier1_pct = (tier1_count / total_docs) * 100
    tier2_pct = (tier2_count / total_docs) * 100
    tier3_pct = (tier3_count / total_docs) * 100
    
    logger.info(f"📊 Tier distribution:")
    logger.info(f"   Tier 1: {tier1_count} docs ({tier1_pct:.1f}%)")
    logger.info(f"   Tier 2: {tier2_count} docs ({tier2_pct:.1f}%)")
    logger.info(f"   Tier 3: {tier3_count} docs ({tier3_pct:.1f}%)")
    
    logger.info(f"✅ Tier distribution validation passed")
    return True


def setup_hierarchical_system():
    """Main setup function for Hierarchical system."""
    start_time = time.time()
    
    logger.info("🚀 Starting Hierarchical System Setup")
    logger.info("=" * 60)
    
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
        response = input("\n🔄 Hierarchical collections already exist. Recreate? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            logger.info("✅ Using existing collections")
            return True
            
    except ValueError:
        logger.info("📝 Hierarchical collections don't exist, will create new ones")
    
    # Check foundation data
    foundation_info = check_foundation_data()
    
    if not foundation_info["exists"]:
        logger.error("❌ NO FOUNDATION DATASET FOUND!")
        logger.error(f"📋 Error: {foundation_info['error']}")
        logger.error(f"📁 Expected location: {foundation_info['path']}")
        logger.error("")
        logger.error("🔧 To create foundation dataset:")
        logger.error("   python fetch_foundation_data.py --max-results 3000")
        return False
    
    # Load and process foundation data
    try:
        logger.info(f"📂 Found foundation data: {foundation_info['count']} items")
        
        # Load foundation dataset
        all_docs = processor.load_foundation_dataset(foundation_info["path"].parent)
        
        if not all_docs:
            logger.error("❌ Foundation dataset loaded but contains no documents")
            return False
        
        # Analyze dataset quality
        analysis = processor.analyze_dataset_quality(all_docs)
        logger.info(f"📊 Dataset analysis:")
        logger.info(f"   Total documents: {analysis['total_documents']}")
        logger.info(f"   Sources: {list(analysis['sources'].keys())}")
        logger.info(f"   Tiers: {analysis['tiers']}")
        logger.info(f"   Evidence-based: {analysis['quality_indicators']['evidence_based']}")
        logger.info(f"   Synthetic: {analysis['quality_indicators']['synthetic']}")
        logger.info(f"   Clinical: {analysis['quality_indicators']['clinical']}")
        
        # Organize by reasoning type
        organized_docs = processor.organize_by_reasoning_type(all_docs)
        
        # Validate tier distribution
        if not validate_tier_distribution(organized_docs):
            logger.error("❌ Tier distribution validation failed")
            return False
        
        logger.info("✅ Loaded and organized foundation dataset")
        
    except Exception as e:
        logger.error(f"❌ Failed to load foundation data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Log organization stats
    logger.info("📊 Document organization summary:")
    total_docs = 0
    for tier_name, docs in organized_docs.items():
        logger.info(f"   {tier_name}: {len(docs)} documents")
        total_docs += len(docs)
    logger.info(f"   Total: {total_docs} documents")
    
    # Create collections and add documents
    try:
        logger.info("🔧 Creating hierarchical collections...")
        
        # Create collections
        retriever.create_hierarchical_collections()
        
        # Add documents to tiers
        retriever.add_documents_to_tiers(organized_docs)
        
        # Verify collections
        retriever.load_hierarchical_collections()
        
        # Test hierarchical search
        test_results = retriever.hierarchical_search("diabetes treatment metformin")
        total_test_results = sum(len(test_results.get(tier, [])) for tier in ["tier1_patterns", "tier2_hypotheses", "tier3_confirmation"])
        
        if total_test_results > 0:
            logger.info("✅ Collections created successfully")
            logger.info(f"🔍 Test search returned {total_test_results} results from 3 tiers")
            
            # Show test results breakdown
            for tier_name, results in test_results.items():
                logger.info(f"   {tier_name}: {len(results)} results")
        else:
            logger.warning("⚠️ Collections created but test search returned no results")
        
        setup_time = time.time() - start_time
        logger.info(f"⏱️ Hierarchical setup completed in {setup_time:.1f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create collections: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | Hierarchical | {message}"
    )
    
    success = setup_hierarchical_system()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 HIERARCHICAL SYSTEM SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Collections created with proper tier distribution:")
        print("   - tier1_pattern_recognition (Drug information)")
        print("   - tier2_hypothesis_testing (Clinical guidelines)")
        print("   - tier3_confirmation (Clinical outcomes)")
        print("")
        print("🔬 Ready for evaluation!")
        print("   python src/evaluation/run_evaluation.py --quick --models hierarchical_system")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ HIERARCHICAL SYSTEM SETUP FAILED!")
        print("=" * 60)
        print("🔧 Common fixes:")
        print("   1. Run: python fetch_foundation_data.py --max-results 3000")
        print("   2. Check that data/foundation/foundation_medical_data.json exists")
        print("   3. Ensure Ollama is running: ollama serve")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()