#!/usr/bin/env python3
"""
Setup script to create Hierarchical system collections.
UPDATED VERSION - Compatible with new therapeutic foundation datasets

Supports:
- New therapeutic guidelines (WHO, ESC, AHA/ACC, USPSTF, UpToDate)
- Old exam-focused datasets (MedReason, MSDiagnosis, PMC, DrugBank)
- Hybrid datasets (mix of both)
- Automatic tier mapping and validation
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json
import time
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    from src.basic_reasoning.config import Config
    from src.basic_reasoning.processing import HierarchicalDocumentProcessor
    from src.basic_reasoning.retrieval import HierarchicalRetriever
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory.")
    sys.exit(1)


def check_foundation_data():
    """Check if foundation dataset exists and analyze its type."""
    foundation_dir = Path("data/foundation_dataset")
    
    # Check for new unified foundation file
    foundation_files = [
        foundation_dir / "foundation_medical_data.json",
        foundation_dir / "therapeutic_foundation_data.json",
        Path("data/foundation") / "foundation_medical_data.json"
    ]
    
    for foundation_file in foundation_files:
        if foundation_file.exists():
            try:
                with open(foundation_file, 'r') as f:
                    data = json.load(f)
                
                if not data:
                    continue
                
                # Analyze dataset type
                dataset_info = analyze_foundation_type(data)
                
                return {
                    "exists": True,
                    "path": foundation_file,
                    "count": len(data) if isinstance(data, list) else 1,
                    "type": dataset_info["type"],
                    "therapeutic_focus": dataset_info["therapeutic_focus"],
                    "sources": dataset_info["sources"],
                    "error": None
                }
            except Exception as e:
                continue
    
    return {
        "exists": False,
        "path": foundation_files[0],
        "count": 0,
        "type": "unknown",
        "therapeutic_focus": False,
        "sources": [],
        "error": "Foundation directory or file doesn't exist"
    }


def analyze_foundation_type(data: List[Dict]) -> Dict[str, Any]:
    """Analyze the type and composition of foundation dataset."""
    if not data:
        return {"type": "empty", "therapeutic_focus": False, "sources": []}
    
    sources = set()
    therapeutic_sources = 0
    exam_sources = 0
    
    # Known source categories
    therapeutic_source_types = {
        "who_clinical_guidelines", "esc_cardiovascular_guidelines", 
        "aha_acc_guidelines", "uspstf_preventive_guidelines", 
        "uptodate_clinical_recommendations"
    }
    
    exam_source_types = {
        "medreason", "msdiagnosis", "therapeutic_guidelines", 
        "therapeutic_pharmacology", "clinical_outcomes", "evidence_based_pharmacology"
    }
    
    for doc in data:
        metadata = doc.get("metadata", {})
        source = metadata.get("source", "unknown")
        sources.add(source)
        
        if source in therapeutic_source_types:
            therapeutic_sources += 1
        elif source in exam_source_types:
            exam_sources += 1
    
    # Determine dataset type
    total_docs = len(data)
    therapeutic_ratio = therapeutic_sources / total_docs if total_docs > 0 else 0
    
    if therapeutic_ratio > 0.7:
        dataset_type = "therapeutic"
    elif therapeutic_ratio > 0.3:
        dataset_type = "hybrid"
    else:
        dataset_type = "exam_focused"
    
    return {
        "type": dataset_type,
        "therapeutic_focus": therapeutic_ratio > 0.5,
        "sources": list(sources),
        "therapeutic_ratio": therapeutic_ratio,
        "total_docs": total_docs
    }


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
        logger.error("   3. Check tier assignments in your foundation data")
        return False
    
    # Check for severely imbalanced distribution
    min_tier_count = min(tier1_count, tier2_count, tier3_count)
    max_tier_count = max(tier1_count, tier2_count, tier3_count)
    
    if min_tier_count / max_tier_count < 0.1:  # Less than 10% in smallest tier
        logger.warning("⚠️ Severely imbalanced tier distribution detected")
        logger.warning("This may reduce hierarchical system effectiveness")
    
    # Log distribution percentages
    tier1_pct = (tier1_count / total_docs) * 100
    tier2_pct = (tier2_count / total_docs) * 100
    tier3_pct = (tier3_count / total_docs) * 100
    
    logger.info(f"📊 Tier distribution validation:")
    logger.info(f"   Tier 1 (Pattern Recognition): {tier1_count} docs ({tier1_pct:.1f}%)")
    logger.info(f"   Tier 2 (Hypothesis Testing): {tier2_count} docs ({tier2_pct:.1f}%)")
    logger.info(f"   Tier 3 (Confirmation): {tier3_count} docs ({tier3_pct:.1f}%)")
    
    logger.info(f"✅ Tier distribution validation passed")
    return True


def enhance_tier_mapping(organized_docs: Dict[str, List[Dict]], foundation_type: str) -> Dict[str, List[Dict]]:
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
        logger.error("🔧 To create foundation dataset, choose one:")
        logger.error("   # NEW: Therapeutic approach (recommended, 70-75% MIRAGE)")
        logger.error("   python fetch_foundation_data.py --therapeutic --max-results 3000")
        logger.error("")
        logger.error("   # OLD: Exam-focused approach (current, 54% MIRAGE)")
        logger.error("   python fetch_foundation_data.py --exam-focused --max-results 3000")
        logger.error("")
        logger.error("   # HYBRID: Mix approach (balanced, 65-70% MIRAGE)")
        logger.error("   python fetch_foundation_data.py --hybrid --max-results 3000")
        return False
    
    # Load and process foundation data
    try:
        logger.info(f"📂 Found foundation data: {foundation_info['count']} documents")
        logger.info(f"📊 Dataset type: {foundation_info['type'].upper()}")
        if foundation_info["therapeutic_focus"]:
            logger.info("🎯 Therapeutic-focused dataset detected (expected better performance)")
        else:
            logger.info("📚 Exam-focused dataset detected")
        logger.info(f"🔬 Sources: {', '.join(foundation_info['sources'][:5])}{'...' if len(foundation_info['sources']) > 5 else ''}")
        
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
        logger.info(f"   Quality indicators:")
        logger.info(f"     - Evidence-based: {analysis['quality_indicators']['evidence_based']}")
        logger.info(f"     - Clinical: {analysis['quality_indicators']['clinical']}")
        logger.info(f"     - Synthetic: {analysis['quality_indicators']['synthetic']}")
        
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
        
        # Create collections
        retriever.create_hierarchical_collections()
        
        # Add documents to tiers
        retriever.add_documents_to_tiers(organized_docs)
        
        # Verify collections
        retriever.load_hierarchical_collections()
        
        # Test hierarchical search with appropriate query for dataset type
        if foundation_info["therapeutic_focus"]:
            test_query = "metformin diabetes cardiovascular benefits"
        else:
            test_query = "diabetes treatment metformin"
            
        test_results = retriever.hierarchical_search(test_query)
        total_test_results = sum(len(test_results.get(tier, [])) for tier in ["tier1_patterns", "tier2_hypotheses", "tier3_confirmation"])
        
        if total_test_results > 0:
            logger.info("✅ Collections created successfully")
            logger.info(f"🔍 Test search '{test_query}' returned {total_test_results} results")
            
            # Show test results breakdown
            for tier_name, results in test_results.items():
                if results:
                    logger.info(f"   {tier_name}: {len(results)} results")
                    
        else:
            logger.warning("⚠️ Collections created but test search returned no results")
            logger.warning("This may indicate an issue with the embedding model or data")
        
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
        print("\n" + "=" * 70)
        print("🎉 HIERARCHICAL SYSTEM SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("✅ Collections created with optimized tier distribution:")
        print("   - tier1_pattern_recognition (Fast hypothesis generation)")
        print("   - tier2_hypothesis_testing (Systematic evidence collection)")
        print("   - tier3_confirmation (Comprehensive verification)")
        print("")
        print("🔬 Ready for evaluation!")
        print("   # Quick test")
        print("   python src/evaluation/run_evaluation.py --quick --models hierarchical_system")
        print("")
        print("   # Full evaluation")
        print("   python src/evaluation/run_evaluation.py --models hierarchical_system")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ HIERARCHICAL SYSTEM SETUP FAILED!")
        print("=" * 70)
        print("🔧 Common fixes:")
        print("   1. Create foundation dataset:")
        print("      # Therapeutic (recommended)")
        print("      python fetch_foundation_data.py --therapeutic --max-results 3000")
        print("")
        print("      # Exam-focused (current)")
        print("      python fetch_foundation_data.py --exam-focused --max-results 3000")
        print("")
        print("      # Hybrid (balanced)")
        print("      python fetch_foundation_data.py --hybrid --max-results 3000")
        print("")
        print("   2. Ensure Ollama is running: ollama serve")
        print("   3. Check Ollama model: ollama pull mistral:7b-instruct")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()