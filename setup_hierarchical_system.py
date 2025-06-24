#!/usr/bin/env python3
"""
Enhanced Setup script for Hierarchical Medical Q&A System
File: setup_hierarchical_system.py

Optimized for medical Q&A with improved tier validation and integration
with all validated fetchers. Enhanced for MIRAGE benchmark performance.
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
    """Find the foundation dataset file with enhanced search."""
    logger.info("🔍 Looking for foundation dataset...")
    
    # Enhanced search paths for all validated fetchers
    candidate_paths = [
        # Primary location (from fetch_foundation_data.py)
        Path("data/foundation_dataset/foundation_medical_data.json"),
        
        # Alternative timestamped files (most recent first)
        *sorted(Path("data/foundation_dataset").glob("foundation_*.json"), reverse=True),
        
        # Legacy locations
        Path("data/foundation/foundation_medical_data.json"),
        Path("data/foundation_medical_data.json"),
        Path("data/foundation_dataset.json"),
        Path("foundation_dataset.json"),
        
        # Other possible locations
        *sorted(Path("data").glob("foundation_*.json"), reverse=True),
    ]
    
    for path in candidate_paths:
        if path.exists() and path.is_file():
            # Enhanced validation to ensure it's medical data
            try:
                file_size = path.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"   📁 Found dataset: {path} ({file_size:.1f} MB)")
                
                # Quick validation by reading first few docs
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, list) and len(data) > 0:
                    sample_doc = data[0]
                    doc_count = len(data)
                elif isinstance(data, dict) and 'documents' in data:
                    sample_doc = data['documents'][0] if data['documents'] else {}
                    doc_count = len(data.get('documents', []))
                else:
                    continue
                
                # Validate it's medical content
                if _validate_medical_content(sample_doc):
                    logger.info(f"   ✅ Validated medical dataset: {doc_count:,} documents")
                    return path
                else:
                    logger.warning(f"   ⚠️  Dataset doesn't appear to be medical content: {path}")
                    
            except Exception as e:
                logger.warning(f"   ❌ Error validating {path}: {e}")
                continue
    
    # No foundation dataset found
    logger.error("❌ CRITICAL ERROR: No medical foundation dataset found!")
    logger.error("")
    logger.error("🚫 MEDICAL FOUNDATION DATASET IS REQUIRED")
    logger.error("   This system requires comprehensive medical foundation datasets")
    logger.error("   from ALL validated fetchers (StatPearls, UMLS, DrugBank, etc.)")
    logger.error("")
    logger.error("📋 SOLUTION: Create foundation dataset using ALL validated fetchers!")
    logger.error("   python fetch_foundation_data.py --max-results 50000 --email your@email.com \\")
    logger.error("     --umls-key YOUR_UMLS_KEY --drugbank-key YOUR_DRUGBANK_KEY")
    logger.error("")
    logger.error("🔧 Alternative: Use critical sources only:")
    logger.error("   python fetch_foundation_data.py --critical-only --max-results 10000 \\")
    logger.error("     --email your@email.com --umls-key YOUR_KEY --drugbank-key YOUR_KEY")
    
    raise FileNotFoundError("No medical foundation dataset found")


def _validate_medical_content(sample_doc: Dict) -> bool:
    """Validate that content is medical/healthcare related."""
    if not isinstance(sample_doc, dict):
        return False
    
    text = str(sample_doc.get('text', '')).lower()
    metadata = sample_doc.get('metadata', {})
    
    # Check for medical indicators
    medical_indicators = [
        'medical', 'clinical', 'patient', 'diagnosis', 'treatment', 'disease',
        'medication', 'drug', 'therapy', 'symptom', 'syndrome', 'pathology',
        'anatomy', 'physiology', 'surgery', 'hospital', 'healthcare'
    ]
    
    # Check for validated medical sources
    source = str(metadata.get('source', '')).lower()
    medical_sources = [
        'statpearls', 'umls', 'drugbank', 'medlineplus', 'who', 'esc',
        'aha', 'acc', 'uspstf', 'uptodate', 'acog', 'idsa', 'pubmed'
    ]
    
    has_medical_text = any(indicator in text for indicator in medical_indicators)
    has_medical_source = any(src in source for src in medical_sources)
    has_medical_specialty = 'medical_specialty' in metadata
    
    return has_medical_text or has_medical_source or has_medical_specialty


def load_foundation_dataset(dataset_path: Path) -> tuple[List[Dict], Dict]:
    """Enhanced foundation dataset loading with medical content analysis."""
    logger.info(f"📂 Loading and analyzing medical foundation dataset: {dataset_path}")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to read dataset file: {e}")
        raise
    
    # Enhanced document extraction
    documents = []
    if isinstance(data, list):
        documents = data
    elif isinstance(data, dict):
        if "documents" in data:
            documents = data["documents"]
        elif "data" in data:
            documents = data["data"]
        else:
            # Try to find documents in nested structure
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict) and "text" in value[0]:
                        documents.extend(value)
                        logger.info(f"   📄 Found documents in key: {key}")
    
    if not documents:
        logger.error("❌ No documents found in dataset")
        raise ValueError("Dataset contains no documents")
    
    # Enhanced medical content analysis
    analysis = analyze_medical_dataset(documents)
    
    logger.info("📊 Enhanced Medical Dataset Analysis:")
    logger.info(f"   📚 Total documents: {analysis['total_documents']:,}")
    logger.info(f"   🔬 Dataset type: {analysis['type']}")
    logger.info(f"   ⭐ Data quality: {analysis['data_quality']}")
    logger.info(f"   🏥 Medical specialties: {len(analysis['specialties'])}")
    logger.info(f"   📋 Validated sources: {len(analysis['sources'])}")
    
    return documents, analysis


def analyze_medical_dataset(documents: List[Dict]) -> Dict:
    """Enhanced medical dataset analysis for all validated fetchers."""
    total_docs = len(documents)
    source_counts = {}
    specialty_counts = {}
    evidence_levels = {}
    tier_distribution = {1: 0, 2: 0, 3: 0}
    
    # Enhanced medical content indicators
    quality_indicators = {
        "statpearls_content": 0,
        "umls_terminology": 0,
        "drugbank_data": 0,
        "medlineplus_education": 0,
        "clinical_guidelines": 0,
        "evidence_based": 0,
        "specialty_specific": 0,
        "research_based": 0
    }
    
    # Analyze each document
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        
        metadata = doc.get("metadata", {})
        text = str(doc.get("text", "")).lower()
        
        # Count sources
        source = metadata.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        
        # Count specialties
        specialty = metadata.get("medical_specialty", "Unknown")
        specialty_counts[specialty] = specialty_counts.get(specialty, 0) + 1
        
        # Count evidence levels
        evidence = metadata.get("evidence_level", "unknown")
        evidence_levels[evidence] = evidence_levels.get(evidence, 0) + 1
        
        # Count tier distribution
        tier = metadata.get("tier", 2)
        if tier in [1, 2, 3]:
            tier_distribution[tier] += 1
        
        # Enhanced quality indicators
        if "statpearls" in source.lower():
            quality_indicators["statpearls_content"] += 1
        if "umls" in source.lower():
            quality_indicators["umls_terminology"] += 1
        if "drugbank" in source.lower():
            quality_indicators["drugbank_data"] += 1
        if "medlineplus" in source.lower():
            quality_indicators["medlineplus_education"] += 1
        if any(term in text for term in ["guideline", "recommendation", "consensus"]):
            quality_indicators["clinical_guidelines"] += 1
        if any(term in text for term in ["evidence", "study", "trial", "research"]):
            quality_indicators["evidence_based"] += 1
        if specialty != "Unknown" and specialty != "General Medicine":
            quality_indicators["specialty_specific"] += 1
        if any(term in text for term in ["pubmed", "clinical trial", "meta-analysis"]):
            quality_indicators["research_based"] += 1
    
    # Enhanced dataset type determination
    validated_fetcher_ratio = (
        quality_indicators["statpearls_content"] + 
        quality_indicators["umls_terminology"] + 
        quality_indicators["drugbank_data"] + 
        quality_indicators["medlineplus_education"]
    ) / total_docs
    
    clinical_ratio = quality_indicators["clinical_guidelines"] / total_docs
    specialty_ratio = quality_indicators["specialty_specific"] / total_docs
    
    if validated_fetcher_ratio > 0.3:
        dataset_type = "enhanced_validated_fetchers"
    elif clinical_ratio > 0.4:
        dataset_type = "clinical_guidelines_focused"
    elif specialty_ratio > 0.5:
        dataset_type = "specialty_balanced"
    else:
        dataset_type = "mixed_medical_sources"
    
    # Enhanced data quality assessment
    high_quality_count = (
        quality_indicators["statpearls_content"] + 
        quality_indicators["clinical_guidelines"] + 
        quality_indicators["evidence_based"]
    )
    
    if high_quality_count > total_docs * 0.6:
        data_quality = "high_medical_quality"
    elif high_quality_count > total_docs * 0.3:
        data_quality = "moderate_medical_quality"
    else:
        data_quality = "mixed_quality"
    
    return {
        "type": dataset_type,
        "data_quality": data_quality,
        "total_documents": total_docs,
        "sources": source_counts,
        "specialties": specialty_counts,
        "evidence_levels": evidence_levels,
        "tier_distribution": tier_distribution,
        "quality_indicators": quality_indicators,
        "validated_fetcher_coverage": validated_fetcher_ratio,
        "specialty_coverage": specialty_ratio,
        "clinical_coverage": clinical_ratio
    }


def enhance_tier_mapping(organized_docs: Dict[str, List[Dict]], 
                        foundation_analysis: Dict) -> Dict[str, List[Dict]]:
    """Enhanced tier mapping based on medical content analysis."""
    logger.info("🎯 Applying enhanced medical tier mapping")
    
    dataset_type = foundation_analysis["type"]
    
    # Enhanced tier optimization for medical Q&A
    if dataset_type == "enhanced_validated_fetchers":
        logger.info("   🔬 Optimizing for validated medical fetchers")
        
        # Move high-authority medical content to appropriate tiers
        tier_moves = {"to_tier1": [], "to_tier3": []}
        
        for tier_name, docs in organized_docs.items():
            for doc in docs[:]:  # Create copy to avoid modification during iteration
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "").lower()
                specialty = metadata.get("medical_specialty", "")
                text = str(doc.get("text", "")).lower()
                
                # Enhanced StatPearls content -> Tier 1 (basic medical knowledge)
                if "statpearls" in source and any(term in text for term in [
                    "definition", "anatomy", "basic", "introduction", "overview"
                ]):
                    if doc["metadata"]["tier"] != 1:
                        doc["metadata"]["tier"] = 1
                        tier_moves["to_tier1"].append(doc)
                
                # Enhanced guidelines/evidence -> Tier 3 (confirmation)
                elif any(source_type in source for source_type in [
                    "who", "esc", "aha", "acc", "uspstf", "acog", "idsa"
                ]) or any(term in text for term in [
                    "guideline", "evidence-based", "recommendation", "consensus"
                ]):
                    if doc["metadata"]["tier"] != 3:
                        doc["metadata"]["tier"] = 3
                        tier_moves["to_tier3"].append(doc)
        
        # Reorganize based on moves
        organized_docs = _apply_tier_moves(organized_docs, tier_moves)
        
        if tier_moves["to_tier1"] or tier_moves["to_tier3"]:
            logger.info(f"   ✅ Enhanced tier mapping: {len(tier_moves['to_tier1'])} to Tier 1, {len(tier_moves['to_tier3'])} to Tier 3")
    
    return organized_docs


def _apply_tier_moves(organized_docs: Dict[str, List[Dict]], 
                     tier_moves: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Apply tier moves and reorganize documents."""
    # Remove moved documents from their original tiers
    moved_docs = set(id(doc) for doc in tier_moves["to_tier1"] + tier_moves["to_tier3"])
    
    for tier_name in ["pattern_recognition", "hypothesis_testing", "confirmation"]:
        organized_docs[tier_name] = [
            doc for doc in organized_docs.get(tier_name, [])
            if id(doc) not in moved_docs
        ]
    
    # Add documents to their new tiers
    organized_docs["pattern_recognition"].extend(tier_moves["to_tier1"])
    organized_docs["confirmation"].extend(tier_moves["to_tier3"])
    
    return organized_docs


def validate_medical_qa_distribution(organized_docs: Dict[str, List[Dict]]) -> bool:
    """Enhanced validation for medical Q&A effectiveness."""
    tier1_count = len(organized_docs.get("pattern_recognition", []))
    tier2_count = len(organized_docs.get("hypothesis_testing", []))
    tier3_count = len(organized_docs.get("confirmation", []))
    total_docs = tier1_count + tier2_count + tier3_count
    
    if total_docs == 0:
        logger.error("❌ CRITICAL ERROR: No documents found after enhanced organization")
        logger.error("🔧 This indicates a processing failure in the enhanced system")
        return False
    
    # Enhanced validation for medical Q&A
    tier1_pct = (tier1_count / total_docs) * 100
    tier2_pct = (tier2_count / total_docs) * 100
    tier3_pct = (tier3_count / total_docs) * 100
    
    logger.info("📊 Enhanced Medical Q&A Distribution:")
    logger.info(f"   Tier 1 (Pattern Recognition): {tier1_count} ({tier1_pct:.1f}%)")
    logger.info(f"   Tier 2 (Clinical Reasoning): {tier2_count} ({tier2_pct:.1f}%)")
    logger.info(f"   Tier 3 (Evidence Confirmation): {tier3_count} ({tier3_pct:.1f}%)")
    logger.info(f"   📊 Total: {total_docs} documents")
    
    # Enhanced warnings for medical Q&A optimization
    validation_passed = True
    
    if tier1_count == 0:
        logger.error("❌ CRITICAL: Tier 1 (Pattern Recognition) is empty")
        logger.error("   Essential for basic medical concept questions")
        validation_passed = False
    elif tier1_pct < 10:
        logger.warning(f"⚠️  Low Tier 1 content ({tier1_pct:.1f}%) - may affect basic medical questions")
    
    if tier2_count == 0:
        logger.error("❌ CRITICAL: Tier 2 (Clinical Reasoning) is empty")
        logger.error("   Essential for clinical decision-making questions")
        validation_passed = False
    elif tier2_pct < 20:
        logger.warning(f"⚠️  Low Tier 2 content ({tier2_pct:.1f}%) - may affect clinical reasoning questions")
    
    if tier3_count == 0:
        logger.warning("⚠️  Tier 3 (Evidence Confirmation) is empty")
        logger.warning("   May affect evidence-based medical questions")
    elif tier3_pct < 10:
        logger.warning(f"⚠️  Low Tier 3 content ({tier3_pct:.1f}%) - may affect evidence-based questions")
    
    # Check for severe imbalance
    max_tier_pct = max(tier1_pct, tier2_pct, tier3_pct)
    if max_tier_pct > 80:
        logger.warning(f"⚠️  Severely imbalanced distribution (max: {max_tier_pct:.1f}%)")
        logger.warning("   This may reduce hierarchical medical reasoning effectiveness")
        logger.warning("🔧 Consider fetching more diverse medical sources")
    else:
        logger.info("✅ Balanced distribution for enhanced medical Q&A")
    
    # Enhanced recommendations
    if not validation_passed:
        logger.error("")
        logger.error("🔧 ENHANCED SOLUTIONS:")
        logger.error("   1. Fetch comprehensive medical data with ALL validated fetchers:")
        logger.error("      python fetch_foundation_data.py --max-results 50000 --email your@email.com \\")
        logger.error("        --umls-key YOUR_UMLS_KEY --drugbank-key YOUR_DRUGBANK_KEY")
        logger.error("   2. Use targeted fetching for missing tiers:")
        logger.error("      python fetch_foundation_data.py --critical-only --max-results 10000")
        logger.error("   3. Verify medical content quality and relevance")
    
    return validation_passed


def setup_hierarchical_system():
    """Enhanced main setup function for Hierarchical Medical Q&A system."""
    start_time = time.time()
    
    logger.info("🚀 Starting Enhanced Hierarchical Medical Q&A System Setup")
    logger.info("🎯 Optimized for MIRAGE benchmark and medical multiple choice questions")
    logger.info("=" * 80)
    
    # Initialize enhanced configuration
    try:
        config = Config()
        device_info = config.get_device_info()
        qa_settings = config.get_medical_qa_settings()
        
        logger.info(f"✅ Enhanced Config: {device_info['environment']} on {device_info.get('cuda_device_name', 'CPU')}")
        logger.info(f"🎯 Medical Q&A Settings: temp={qa_settings['temperature']}, strict_validation={qa_settings['strict_validation']}")
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Failed to load enhanced config: {e}")
        logger.error("🔧 Check your configuration files and environment")
        raise RuntimeError(f"Enhanced configuration loading failed: {e}")
    
    # Initialize enhanced components
    try:
        processor = HierarchicalDocumentProcessor(config.config["processing"])
        retriever = HierarchicalRetriever(config)
        logger.info("✅ Initialized enhanced Hierarchical components for medical Q&A")
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Failed to initialize enhanced components: {e}")
        logger.error("🔧 Check your system dependencies and configuration")
        raise RuntimeError(f"Enhanced component initialization failed: {e}")
    
    # Check existing collections
    try:
        retriever.load_hierarchical_collections()
        logger.info("✅ Enhanced hierarchical collections already exist")
        
        # Ask if we should recreate
        response = input("\n🔄 Enhanced hierarchical collections already exist. Recreate for better medical Q&A? (y/N): ")
        if response.lower() != 'y':
            logger.info("✅ Using existing enhanced hierarchical collections")
            elapsed_time = time.time() - start_time
            logger.info(f"⏱️  Enhanced setup completed in {elapsed_time:.1f} seconds")
            
            # Display enhanced usage instructions
            _display_enhanced_usage_instructions()
            return True
            
    except Exception:
        logger.info("📝 No existing collections found, creating enhanced ones...")
    
    # Find and load foundation dataset
    try:
        dataset_path = find_foundation_dataset()
        all_docs, foundation_analysis = load_foundation_dataset(dataset_path)
        
        logger.info("📊 Enhanced Foundation Analysis:")
        logger.info(f"   📚 Total documents: {foundation_analysis['total_documents']:,}")
        logger.info(f"   🔬 Dataset type: {foundation_analysis['type']}")
        logger.info(f"   ⭐ Data quality: {foundation_analysis['data_quality']}")
        logger.info(f"   🏥 Medical specialties: {len(foundation_analysis['specialties'])}")
        logger.info(f"   📋 Validated fetcher coverage: {foundation_analysis['validated_fetcher_coverage']:.1%}")
        logger.info(f"   🎯 Specialty coverage: {foundation_analysis['specialty_coverage']:.1%}")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Enhanced foundation dataset loading failed: {e}")
        raise
    
    # Enhanced preprocessing
    try:
        logger.info("🔧 Enhanced preprocessing for medical Q&A...")
        all_docs = processor.preprocess_documents(all_docs)
        
        # Enhanced tier logging
        tier_counts = {1: 0, 2: 0, 3: 0}
        for doc in all_docs:
            tier = doc["metadata"].get("tier", 2)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        logger.info("📊 Enhanced tier assignment results:")
        logger.info(f"   Tier 1 (Pattern Recognition): {tier_counts.get(1, 0)} documents")
        logger.info(f"   Tier 2 (Clinical Reasoning): {tier_counts.get(2, 0)} documents") 
        logger.info(f"   Tier 3 (Evidence Confirmation): {tier_counts.get(3, 0)} documents")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Enhanced document preprocessing failed: {e}")
        logger.error("🔧 Check enhanced document processor configuration")
        raise RuntimeError(f"Enhanced document preprocessing failed: {e}")
    
    # Enhanced organization by reasoning type
    try:
        organized_docs = processor.organize_by_reasoning_type(all_docs)
        
        # Enhanced tier mapping
        organized_docs = enhance_tier_mapping(organized_docs, foundation_analysis)
        
        # Enhanced validation
        if not validate_medical_qa_distribution(organized_docs):
            logger.error("❌ CRITICAL ERROR: Enhanced tier distribution validation failed")
            logger.error("🔧 Medical Q&A effectiveness may be compromised")
            # Continue anyway but warn user
            
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Enhanced document organization failed: {e}")
        raise RuntimeError(f"Enhanced document organization failed: {e}")
    
    # Create enhanced hierarchical collections
    try:
        logger.info("🏗️  Creating enhanced hierarchical collections for medical Q&A...")
        
        # Create empty collections first
        retriever.create_hierarchical_collections()
        
        # Add documents to the collections
        retriever.add_documents_to_tiers(organized_docs)
        
        # Enhanced performance logging
        total_docs = sum(len(docs) for docs in organized_docs.values())
        logger.info(f"✅ Enhanced hierarchical collections created successfully!")
        logger.info(f"📊 Total indexed: {total_docs:,} medical documents")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Enhanced collection creation failed: {e}")
        raise RuntimeError(f"Enhanced collection creation failed: {e}")
    
    # Enhanced setup completion
    elapsed_time = time.time() - start_time
    logger.info("🎉 ENHANCED HIERARCHICAL MEDICAL Q&A SYSTEM SETUP COMPLETE!")
    logger.info(f"⏱️  Total setup time: {elapsed_time:.1f} seconds")
    logger.info(f"🎯 System optimized for medical multiple choice questions")
    logger.info(f"🏆 Ready for MIRAGE benchmark evaluation")
    
    # Display enhanced usage instructions
    _display_enhanced_usage_instructions()
    
    return True


def _display_enhanced_usage_instructions():
    """Display enhanced usage instructions for medical Q&A system."""
    logger.info("")
    logger.info("🎯 ENHANCED MEDICAL Q&A SYSTEM READY!")
    logger.info("=" * 50)
    logger.info("📋 Next Steps:")
    logger.info("1. Test enhanced hierarchical retrieval:")
    logger.info('   python -c "from src.basic_reasoning.retrieval import HierarchicalRetriever; from src.basic_reasoning.config import Config; r = HierarchicalRetriever(Config()); r.load_hierarchical_collections(); print(r.hierarchical_search(\'What is the most common cause of pneumonia?\'))"')
    logger.info("2. Run enhanced Streamlit app:")
    logger.info("   streamlit run src/basic_reasoning/streamlit_app.py --server.port 8503")
    logger.info("3. Run enhanced MIRAGE evaluation:")
    logger.info("   python src/evaluation/run_evaluation.py --benchmark mirage --models hierarchical_system")
    logger.info("4. Run comprehensive medical benchmark:")
    logger.info("   python src/evaluation/run_evaluation.py --full --models hierarchical_system")
    logger.info("")
    logger.info("🔥 Enhanced Features:")
    logger.info("• Intelligent tier assignment based on medical content")
    logger.info("• Enhanced answer extraction for multiple choice questions")
    logger.info("• Optimized prompts for medical knowledge assessment")
    logger.info("• Support for ALL validated medical fetchers")
    logger.info("• MIRAGE benchmark optimization")


if __name__ == "__main__":
    try:
        setup_hierarchical_system()
        
    except FileNotFoundError as e:
        logger.error("❌ SETUP FAILED: Missing enhanced medical data")
        print("\n🔧 SOLUTION: Create comprehensive medical dataset using ALL validated fetchers!")
        print("python fetch_foundation_data.py --max-results 50000 --email your@email.com \\")
        print("  --umls-key YOUR_UMLS_KEY --drugbank-key YOUR_DRUGBANK_KEY")
        sys.exit(1)
        
    except ValueError as e:
        logger.error("❌ SETUP FAILED: Invalid enhanced data")
        print("\n🔧 SOLUTION: Re-create enhanced medical dataset with proper validation")
        print("python fetch_foundation_data.py --max-results 20000 --email your@email.com")
        sys.exit(1)
        
    except RuntimeError as e:
        logger.error("❌ SETUP FAILED: Enhanced system runtime error")
        print("\n🔧 SOLUTION: Check enhanced system resources and configuration")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ SETUP FAILED: Unexpected enhanced error: {e}")
        import traceback
        logger.error("📋 Enhanced system traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)