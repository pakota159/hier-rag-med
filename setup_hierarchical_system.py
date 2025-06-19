#!/usr/bin/env python3
"""
Setup script to create Hierarchical system collections with real data ONLY.
Replicates exactly what the Hierarchical Streamlit app does but in headless mode.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict
import json
import time
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.basic_reasoning.config import Config
    from src.basic_reasoning.processing import HierarchicalDocumentProcessor
    from src.basic_reasoning.retrieval import HierarchicalRetriever
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory.")
    sys.exit(1)


def check_foundation_data():
    """Check if foundation dataset exists - REAL DATA ONLY."""
    foundation_dir = Path("data/foundation")
    foundation_file = foundation_dir / "foundation_medical_data.json"
    
    if not foundation_dir.exists():
        return {"exists": False, "path": foundation_file, "count": 0, "error": "Foundation directory doesn't exist"}
    
    if not foundation_file.exists():
        return {"exists": False, "path": foundation_file, "count": 0, "error": "Foundation data file doesn't exist"}
    
    try:
        with open(foundation_file, 'r') as f:
            data = json.load(f)
        
        if not data:
            return {"exists": False, "path": foundation_file, "count": 0, "error": "Foundation file is empty"}
        
        count = len(data) if isinstance(data, list) else 1
        return {"exists": True, "path": foundation_file, "count": count, "error": None}
        
    except Exception as e:
        return {"exists": False, "path": foundation_file, "count": 0, "error": f"Could not read foundation file: {e}"}


def setup_hierarchical_system():
    """Main setup function for Hierarchical system - REAL DATA ONLY."""
    start_time = time.time()
    
    logger.info("🚀 Starting Hierarchical System Setup (Real Data Only)")
    logger.info("=" * 60)
    
    # Initialize configuration
    try:
        config = Config()
        logger.info(f"✅ Loaded config: {config.get_device_info()}")
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
    collection_names = ["tier1_pattern_recognition", "tier2_hypothesis_testing", "tier3_confirmation"]
    
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
    
    # Check foundation data (REQUIRED - NO FALLBACK)
    foundation_info = check_foundation_data()
    
    if not foundation_info["exists"]:
        logger.error("❌ NO FOUNDATION DATASET FOUND!")
        logger.error(f"📋 Error: {foundation_info['error']}")
        logger.error(f"📁 Expected location: {foundation_info['path']}")
        logger.error("")
        logger.error("🔧 To create foundation dataset:")
        logger.error("   # Quick test dataset:")
        logger.error("   python fetch_foundation_data.py --quick")
        logger.error("")
        logger.error("   # Full dataset:")
        logger.error("   python fetch_foundation_data.py --max-results 1000")
        logger.error("")
        logger.error("   # MedReason only:")
        logger.error("   python fetch_foundation_data.py --medreason-only --max-results 500")
        raise ValueError(f"Foundation dataset required but not found: {foundation_info['error']}")
    
    # Load and process foundation data exactly like Streamlit app
    try:
        logger.info(f"📂 Found foundation data: {foundation_info['count']} items")
        
        # Load foundation dataset exactly like Streamlit app does
        all_docs = processor.load_foundation_dataset(foundation_info["path"].parent)
        
        if not all_docs:
            raise ValueError("Foundation dataset loaded but contains no documents")
        
        # Organize by reasoning type exactly like Streamlit app
        organized_docs = processor.organize_by_reasoning_type(all_docs)
        
        # Verify organization produced results
        total_organized = sum(len(docs) for docs in organized_docs.values())
        if total_organized == 0:
            raise ValueError("Document organization failed - no documents in any tier")
        
        logger.info("✅ Loaded and organized foundation dataset")
        
    except Exception as e:
        logger.error(f"❌ Failed to load foundation data: {e}")
        raise
    #!/usr/bin/env python3
"""
Setup script to create Hierarchical system collections with real data.
Replicates what the Hierarchical Streamlit app does but in headless mode.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict
import json
import time
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.basic_reasoning.config import Config
    from src.basic_reasoning.processing import HierarchicalDocumentProcessor
    from src.basic_reasoning.retrieval import HierarchicalRetriever
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
            return {
                "exists": True,
                "path": foundation_file,
                "count": len(data) if isinstance(data, list) else 1
            }
        except Exception as e:
            logger.warning(f"Foundation file exists but couldn't read: {e}")
    
    return {"exists": False, "path": foundation_file, "count": 0}


def create_sample_hierarchical_data():
    """Create sample hierarchical medical data organized by reasoning types."""
    
    # Tier 1: Pattern Recognition (Fast screening, symptom patterns)
    pattern_recognition_docs = [
        {
            "text": "Chest pain with radiation to left arm, diaphoresis, and nausea suggests acute coronary syndrome. Immediate ECG and cardiac enzymes required.",
            "metadata": {
                "source": "emergency_medicine",
                "doc_id": "pattern_001",
                "chunk_id": 0,
                "reasoning_type": "pattern_recognition",
                "specialty": "emergency_medicine",
                "tier": 1
            }
        },
        {
            "text": "Polyuria, polydipsia, and weight loss in young patient indicates possible type 1 diabetes. Check blood glucose and ketones immediately.",
            "metadata": {
                "source": "endocrinology",
                "doc_id": "pattern_002", 
                "chunk_id": 0,
                "reasoning_type": "pattern_recognition",
                "specialty": "endocrinology",
                "tier": 1
            }
        },
        {
            "text": "Fever, productive cough, and dyspnea suggest pneumonia. Chest X-ray and sputum culture indicated for diagnosis.",
            "metadata": {
                "source": "pulmonology",
                "doc_id": "pattern_003",
                "chunk_id": 0,
                "reasoning_type": "pattern_recognition", 
                "specialty": "pulmonology",
                "tier": 1
            }
        }
    ]
    
    # Tier 2: Hypothesis Testing (Diagnostic reasoning chains)
    hypothesis_testing_docs = [
        {
            "text": "For suspected myocardial infarction: 1) Assess chest pain characteristics 2) Obtain ECG within 10 minutes 3) Check troponin levels 4) Consider differential: PE, aortic dissection, GERD 5) Initiate dual antiplatelet therapy if STEMI confirmed.",
            "metadata": {
                "source": "medreason",
                "doc_id": "reasoning_001",
                "chunk_id": 0,
                "reasoning_type": "knowledge_graph_guided",
                "specialty": "cardiology",
                "tier": 2
            }
        },
        {
            "text": "Diabetes diagnosis pathway: 1) Classic symptoms + random glucose ≥200 mg/dL OR 2) Fasting glucose ≥126 mg/dL OR 3) 2-hour OGTT ≥200 mg/dL OR 4) HbA1c ≥6.5%. Consider MODY if family history and young onset.",
            "metadata": {
                "source": "msdiagnosis",
                "doc_id": "reasoning_002",
                "chunk_id": 0,
                "reasoning_type": "multi_step_diagnostic",
                "specialty": "endocrinology",
                "tier": 2
            }
        },
        {
            "text": "Pneumonia evaluation algorithm: 1) Clinical assessment (CURB-65) 2) Chest imaging 3) Laboratory tests if severe 4) Pathogen identification if indicated 5) Antibiotic selection based on severity and risk factors.",
            "metadata": {
                "source": "medreason",
                "doc_id": "reasoning_003",
                "chunk_id": 0,
                "reasoning_type": "knowledge_graph_guided",
                "specialty": "pulmonology",
                "tier": 2
            }
        }
    ]
    
    # Tier 3: Confirmation (Clinical evidence, case studies)
    confirmation_docs = [
        {
            "text": "Case study: 55-year-old male with chest pain, elevated troponin I (15.2 ng/mL), and anterior STEMI on ECG. Underwent primary PCI with stent placement. Recovery was uncomplicated with dual antiplatelet therapy and ACE inhibitor.",
            "metadata": {
                "source": "pmc_patients",
                "doc_id": "case_001",
                "chunk_id": 0,
                "reasoning_type": "case_study",
                "specialty": "cardiology",
                "tier": 3
            }
        },
        {
            "text": "Clinical evidence: A 22-year-old female presented with polyuria, polydipsia, and 15-pound weight loss. HbA1c was 12.5%, GAD antibodies positive. Diagnosed with type 1 diabetes, started on insulin therapy with excellent glycemic control.",
            "metadata": {
                "source": "pmc_patients",
                "doc_id": "case_002",
                "chunk_id": 0,
                "reasoning_type": "case_study",
                "specialty": "endocrinology",
                "tier": 3
            }
        },
        {
            "text": "Patient outcome: 68-year-old with community-acquired pneumonia, CURB-65 score 2. Treated with ceftriaxone and azithromycin. Chest X-ray cleared after 10 days, full recovery achieved.",
            "metadata": {
                "source": "pmc_patients",
                "doc_id": "case_003",
                "chunk_id": 0,
                "reasoning_type": "case_study",
                "specialty": "pulmonology",
                "tier": 3
            }
        }
    ]
    
    # Additional mixed data for better coverage
    mixed_docs = [
        {
            "text": "Hypertensive emergency requires immediate reduction of BP by 10-20% in first hour, then gradual reduction over 24 hours. Avoid sublingual nifedipine due to unpredictable hypotension.",
            "metadata": {
                "source": "clinical_guidelines",
                "doc_id": "mixed_001",
                "chunk_id": 0,
                "reasoning_type": "clinical_guideline",
                "specialty": "emergency_medicine",
                "tier": 1
            }
        },
        {
            "text": "Stroke evaluation: 1) NIHSS assessment 2) CT head to exclude hemorrhage 3) Check glucose 4) If ischemic and <4.5 hours, consider tPA 5) Determine large vessel occlusion for thrombectomy.",
            "metadata": {
                "source": "neurology_protocols",
                "doc_id": "mixed_002",
                "chunk_id": 0,
                "reasoning_type": "diagnostic_protocol",
                "specialty": "neurology",
                "tier": 2
            }
        },
        {
            "text": "Successful case: 45-year-old male with acute ischemic stroke, NIHSS 8, received tPA within 3 hours. Complete neurological recovery at 90 days with modified Rankin Scale 0.",
            "metadata": {
                "source": "stroke_registry",
                "doc_id": "mixed_003",
                "chunk_id": 0,
                "reasoning_type": "outcome_study",
                "specialty": "neurology",
                "tier": 3
            }
        }
    ]
    
    return {
        "pattern_recognition": pattern_recognition_docs,
        "hypothesis_testing": hypothesis_testing_docs,
        "confirmation": confirmation_docs,
        "mixed": mixed_docs
    }


def organize_sample_data_by_tiers(sample_data: Dict) -> Dict[str, List[Dict]]:
    """Organize sample data into hierarchical tiers."""
    
    organized = {
        "pattern_recognition": [],
        "hypothesis_testing": [],
        "confirmation": []
    }
    
    # Add tier-specific data
    organized["pattern_recognition"].extend(sample_data["pattern_recognition"])
    organized["hypothesis_testing"].extend(sample_data["hypothesis_testing"]) 
    organized["confirmation"].extend(sample_data["confirmation"])
    
    # Distribute mixed data based on reasoning type
    for doc in sample_data["mixed"]:
        reasoning_type = doc["metadata"].get("reasoning_type", "")
        tier = doc["metadata"].get("tier", 0)
        
        if tier == 1 or reasoning_type in ["pattern_recognition", "clinical_guideline"]:
            organized["pattern_recognition"].append(doc)
        elif tier == 2 or reasoning_type in ["diagnostic_protocol", "knowledge_graph_guided"]:
            organized["hypothesis_testing"].append(doc)
        else:
            organized["confirmation"].append(doc)
    
    return organized


def setup_hierarchical_system():
    """Main setup function for Hierarchical system."""
    start_time = time.time()
    
    logger.info("🚀 Starting Hierarchical System Setup")
    logger.info("=" * 60)
    
    # Initialize configuration
    try:
        config = Config()
        logger.info(f"✅ Loaded config: {config.get_device_info()}")
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
    collection_names = ["tier1_pattern_recognition", "tier2_hypothesis_testing", "tier3_confirmation"]
    
    try:
        retriever.load_hierarchical_collections()
        logger.info("✅ Hierarchical collections already exist")
        
        # Check if we should recreate
        response = input("\n🔄 Collections already exist. Recreate? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            logger.info("✅ Using existing collections")
            return True
            
    except ValueError:
        logger.info("📝 Hierarchical collections don't exist, will create new ones")
    
    # Check foundation data
    foundation_info = check_foundation_data()
    
    if foundation_info["exists"]:
        logger.info(f"📂 Found foundation data: {foundation_info['count']} items")
        try:
            # Load real foundation data
            all_docs = processor.load_foundation_dataset(foundation_info["path"].parent)
            organized_docs = processor.organize_by_reasoning_type(all_docs)
            logger.info("✅ Loaded and organized foundation dataset")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load foundation data: {e}")
            logger.info("🔄 Falling back to sample data")
            sample_data = create_sample_hierarchical_data()
            organized_docs = organize_sample_data_by_tiers(sample_data)
    else:
        logger.warning("⚠️ No foundation dataset found, using sample data")
        logger.info("💡 To use real data, run: python fetch_foundation_data.py --quick")
        sample_data = create_sample_hierarchical_data()
        organized_docs = organize_sample_data_by_tiers(sample_data)
    
    # Log organization stats
    logger.info("📊 Document organization:")
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
        test_results = retriever.hierarchical_search("chest pain diagnosis")
        logger.info("✅ Collections created successfully")
        logger.info(f"🔍 Test search returned results from {len(test_results)} tiers")
        
        setup_time = time.time() - start_time
        logger.info(f"⏱️ Hierarchical setup completed in {setup_time:.1f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create collections: {e}")
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
        print("✅ Collections created:")
        print("   - tier1_pattern_recognition")
        print("   - tier2_hypothesis_testing")
        print("   - tier3_confirmation")
        print("🔬 You can now run: python src/evaluation/run_evaluation.py --quick")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ HIERARCHICAL SYSTEM SETUP FAILED!")
        print("=" * 60)
        print("🔧 Check the logs above for specific errors")
        print("💡 Try running with sample data or check foundation data availability")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()